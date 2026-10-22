/*
 * libtntsniff.so — LD_PRELOAD socket interposer for the TNT client.
 *
 * Loaded into the game process via the `wrap.<package>` mechanism:
 * the wrap script exports LD_PRELOAD pointing here. We shadow libc's
 * socket functions with symbol interposition — no ptrace, no inline
 * patching. Qt SSL pointers are intercepted at QLibrary::resolve (tnt_tls.h).
 * Captured byte streams are forwarded over a TCP
 * tunnel to 127.0.0.1:19001, which `adb reverse tcp:19001` maps to the
 * Mac host where the Rust decoder listens.
 *
 * Record framing (all little-endian, 21-byte header):
 *   u8 kind | u32 pid | u32 fd | u64 mono_ms | u32 len | payload
 * kinds: 1 connect("ip:port") 2 tx 3 rx 4 close 5 hello 6 rand 7 log 8 fmt
 *        9 qnum 10 qtext 11 SSL plaintext tx 12 SSL plaintext rx
 */
#define _GNU_SOURCE
#include <android/log.h>
#include <math.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define TAG "TntSniff"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)

#define TUNNEL_PORT 19001
static int g_tunnel_port = TUNNEL_PORT;
#define FD_MAX 4096
#define RING_SIZE (512 * 1024)
#define REC_MAX_PAYLOAD (48 * 1024)

/* fd states */
#define FD_NONE 0
#define FD_TCP 1 /* connected non-loopback TCP socket */
#define FD_SELF 2 /* our own tunnel fd */

static _Atomic uint8_t g_fd_state[FD_MAX];

/* ---------------- tunnel writer ---------------- */

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static uint8_t g_ring[RING_SIZE];
static size_t g_rpos = 0; /* read pos  */
static size_t g_wpos = 0; /* write pos */
static _Atomic uint64_t g_dropped = 0;

static uint64_t mono_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ull + ts.tv_nsec / 1000000ull;
}

static void ring_push(uint8_t kind, uint32_t fd, const void *payload, uint32_t len) {
    if (len > REC_MAX_PAYLOAD) {
        len = REC_MAX_PAYLOAD;
    }
    uint8_t hdr[21];
    uint32_t pid = (uint32_t)getpid();
    uint64_t ms = mono_ms();
    hdr[0] = kind;
    memcpy(hdr + 1, &pid, 4);
    memcpy(hdr + 5, &fd, 4);
    memcpy(hdr + 9, &ms, 8);
    memcpy(hdr + 17, &len, 4);

    pthread_mutex_lock(&g_mu);
    size_t used = g_wpos - g_rpos;
    size_t free_space = RING_SIZE - used;
    if (free_space < sizeof(hdr) + len) {
        g_dropped++;
        pthread_mutex_unlock(&g_mu);
        return;
    }
    /* compact when the tail can't fit the record but the front can */
    if (RING_SIZE - g_wpos < sizeof(hdr) + len) {
        memmove(g_ring, g_ring + g_rpos, used);
        g_rpos = 0;
        g_wpos = used;
    }
    memcpy(g_ring + g_wpos, hdr, sizeof(hdr));
    memcpy(g_ring + g_wpos + sizeof(hdr), payload, len);
    g_wpos += sizeof(hdr) + len;
    pthread_cond_signal(&g_cv);
    pthread_mutex_unlock(&g_mu);
}

/* raw syscalls so the tunnel itself never re-enters the interposers */
static int tun_open_connect(void) {
    int fd = (int)syscall(SYS_socket, AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    if (fd < FD_MAX) {
        g_fd_state[fd] = FD_SELF;
    }
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)g_tunnel_port);
    sa.sin_addr.s_addr = htonl(0x7f000001); /* 127.0.0.1 -> adb reverse */
    if (syscall(SYS_connect, fd, &sa, sizeof(sa)) != 0) {
        syscall(SYS_close, fd);
        if (fd < FD_MAX) {
            g_fd_state[fd] = FD_NONE;
        }
        return -1;
    }
    return fd;
}

static void *writer_main(void *unused) {
    (void)unused;
    int tun = -1;
    for (;;) {
        if (tun < 0) {
            tun = tun_open_connect();
            if (tun < 0) {
                sleep(1);
                continue;
            }
            char hello[64];
            int n = snprintf(hello, sizeof(hello), "tntsniff v1 tls=3 pid=%d", getpid());
            ring_push(5, 0, hello, (uint32_t)n);
            LOGI("tunnel up");
        }

        pthread_mutex_lock(&g_mu);
        while (g_rpos == g_wpos) {
            pthread_cond_wait(&g_cv, &g_mu);
        }
        /* snapshot into a stack buffer so the lock isn't held during send */
        size_t avail = g_wpos - g_rpos;
        if (avail > RING_SIZE / 2) {
            avail = RING_SIZE / 2;
        }
        uint8_t *chunk = malloc(avail);
        if (!chunk) {
            pthread_mutex_unlock(&g_mu);
            continue;
        }
        memcpy(chunk, g_ring + g_rpos, avail);
        pthread_mutex_unlock(&g_mu);

        size_t off = 0;
        bool ok = true;
        while (off < avail) {
            ssize_t w = syscall(SYS_sendto, tun, chunk + off, avail - off,
                                MSG_NOSIGNAL, NULL, 0);
            if (w <= 0) {
                ok = false;
                break;
            }
            off += (size_t)w;
        }
        free(chunk);

        pthread_mutex_lock(&g_mu);
        g_rpos += avail;
        if (g_rpos == g_wpos) {
            g_rpos = g_wpos = 0;
        }
        pthread_mutex_unlock(&g_mu);

        if (!ok) {
            syscall(SYS_close, tun);
            tun = -1;
            LOGW("tunnel lost, reconnecting");
        }
    }
    return NULL;
}

/* ---------------- fd table ---------------- */

static bool fd_tracked(int fd) {
    return fd >= 0 && fd < FD_MAX && g_fd_state[fd] == FD_TCP;
}

static void note_connect(int fd, const struct sockaddr *addr, socklen_t len) {
    if (fd < 0 || fd >= FD_MAX || !addr || len < sizeof(struct sockaddr_in)) {
        return;
    }
    if (addr->sa_family != AF_INET) {
        return;
    }
    const struct sockaddr_in *in = (const struct sockaddr_in *)addr;
    uint32_t ip = ntohl(in->sin_addr.s_addr);
    if ((ip >> 24) == 127 || ip == 0) {
        return;
    }
    int st = -1;
    getsockopt(fd, SOL_SOCKET, SO_TYPE, &st, &(socklen_t){sizeof(st)});
    if (st != SOCK_STREAM) {
        return;
    }
    g_fd_state[fd] = FD_TCP;
    char ipport[64];
    inet_ntop(AF_INET, &in->sin_addr, ipport, sizeof(ipport));
    size_t l = strlen(ipport);
    snprintf(ipport + l, sizeof(ipport) - l, ":%u", ntohs(in->sin_port));
    ring_push(1, (uint32_t)fd, ipport, (uint32_t)strlen(ipport));
}

static void note_close(int fd) {
    if (fd >= 0 && fd < FD_MAX && g_fd_state[fd] == FD_TCP) {
        g_fd_state[fd] = FD_NONE;
        ring_push(4, (uint32_t)fd, "", 0);
    } else if (fd >= 0 && fd < FD_MAX && g_fd_state[fd] == FD_SELF) {
        g_fd_state[fd] = FD_NONE;
    }
}

static void note_data(uint8_t kind, int fd, const void *buf, ssize_t n) {
    if (n <= 0 || !fd_tracked(fd)) {
        return;
    }
    if (n > REC_MAX_PAYLOAD) {
        /* oversized single write: split */
        const uint8_t *p = buf;
        ssize_t left = n;
        while (left > 0) {
            uint32_t chunk = left > REC_MAX_PAYLOAD ? REC_MAX_PAYLOAD : (uint32_t)left;
            ring_push(kind, (uint32_t)fd, p, chunk);
            p += chunk;
            left -= chunk;
        }
        return;
    }
    ring_push(kind, (uint32_t)fd, buf, (uint32_t)n);
}

static void note_iov(uint8_t kind, int fd, const struct iovec *iov, int iovcnt, ssize_t n) {
    if (n <= 0 || !fd_tracked(fd) || !iov || iovcnt <= 0) {
        return;
    }
    ssize_t need = n;
    uint8_t stackbuf[16384];
    uint8_t *tmp = need <= (ssize_t)sizeof(stackbuf) ? stackbuf : malloc((size_t)need);
    if (!tmp) {
        return;
    }
    size_t off = 0;
    for (int i = 0; i < iovcnt && need > 0; i++) {
        size_t take = iov[i].iov_len < (size_t)need ? iov[i].iov_len : (size_t)need;
        memcpy(tmp + off, iov[i].iov_base, take);
        off += take;
        need -= (ssize_t)take;
    }
    note_data(kind, fd, tmp, (ssize_t)off);
    if (tmp != stackbuf) {
        free(tmp);
    }
}

/* ---------------- real symbol resolution ---------------- */

#define RESOLVE(name)                                            \
    static name##_t real_##name;                                 \
    if (!real_##name) {                                          \
        real_##name = (name##_t)dlsym(RTLD_NEXT, #name);         \
        if (!real_##name) {                                      \
            errno = ENOSYS;                                      \
            return -1;                                           \
        }                                                        \
    }

#define RESOLVE_VOID(name)                                       \
    static name##_t real_##name;                                 \
    if (!real_##name) {                                          \
        real_##name = (name##_t)dlsym(RTLD_NEXT, #name);         \
        if (!real_##name) {                                      \
            return;                                              \
        }                                                        \
    }

typedef int (*connect_t)(int, const struct sockaddr *, socklen_t);
typedef ssize_t (*send_t)(int, const void *, size_t, int);
typedef ssize_t (*sendto_t)(int, const void *, size_t, int, const struct sockaddr *, socklen_t);
typedef ssize_t (*sendmsg_t)(int, const struct msghdr *, int);
typedef ssize_t (*recv_t)(int, void *, size_t, int);
typedef ssize_t (*recvfrom_t)(int, void *, size_t, int, struct sockaddr *, socklen_t *);
typedef ssize_t (*recvmsg_t)(int, struct msghdr *, int);
typedef ssize_t (*read_t)(int, void *, size_t);
typedef ssize_t (*write_t)(int, const void *, size_t);
typedef ssize_t (*writev_t)(int, const struct iovec *, int);
typedef ssize_t (*readv_t)(int, const struct iovec *, int);
typedef int (*close_t)(int);
typedef int (*shutdown_t)(int, int);

int connect(int fd, const struct sockaddr *addr, socklen_t len) {
    RESOLVE(connect);
    int r = real_connect(fd, addr, len);
    if (r == 0 || errno == EINPROGRESS) {
        note_connect(fd, addr, len);
    }
    return r;
}

ssize_t send(int fd, const void *buf, size_t n, int flags) {
    RESOLVE(send);
    ssize_t r = real_send(fd, buf, n, flags);
    note_data(2, fd, buf, r);
    return r;
}

ssize_t sendto(int fd, const void *buf, size_t n, int flags,
               const struct sockaddr *a, socklen_t al) {
    RESOLVE(sendto);
    ssize_t r = real_sendto(fd, buf, n, flags, a, al);
    note_data(2, fd, buf, r);
    return r;
}

ssize_t sendmsg(int fd, const struct msghdr *msg, int flags) {
    RESOLVE(sendmsg);
    ssize_t r = real_sendmsg(fd, msg, flags);
    note_iov(2, fd, msg->msg_iov, (int)msg->msg_iovlen, r);
    return r;
}

ssize_t recv(int fd, void *buf, size_t n, int flags) {
    RESOLVE(recv);
    ssize_t r = real_recv(fd, buf, n, flags);
    note_data(3, fd, buf, r);
    return r;
}

ssize_t recvfrom(int fd, void *buf, size_t n, int flags,
                 struct sockaddr *a, socklen_t *al) {
    RESOLVE(recvfrom);
    ssize_t r = real_recvfrom(fd, buf, n, flags, a, al);
    note_data(3, fd, buf, r);
    return r;
}

ssize_t recvmsg(int fd, struct msghdr *msg, int flags) {
    RESOLVE(recvmsg);
    ssize_t r = real_recvmsg(fd, msg, flags);
    note_iov(3, fd, msg->msg_iov, (int)msg->msg_iovlen, r);
    return r;
}

ssize_t read(int fd, void *buf, size_t n) {
    RESOLVE(read);
    ssize_t r = real_read(fd, buf, n);
    if (r > 0 && fd_tracked(fd)) {
        note_data(3, fd, buf, r);
    }
    return r;
}

ssize_t write(int fd, const void *buf, size_t n) {
    RESOLVE(write);
    ssize_t r = real_write(fd, buf, n);
    if (r > 0 && fd_tracked(fd)) {
        note_data(2, fd, buf, r);
    }
    return r;
}

ssize_t writev(int fd, const struct iovec *iov, int iovcnt) {
    RESOLVE(writev);
    ssize_t r = real_writev(fd, iov, iovcnt);
    if (r > 0 && fd_tracked(fd)) {
        note_iov(2, fd, iov, iovcnt, r);
    }
    return r;
}

ssize_t readv(int fd, const struct iovec *iov, int iovcnt) {
    RESOLVE(readv);
    ssize_t r = real_readv(fd, iov, iovcnt);
    if (r > 0 && fd_tracked(fd)) {
        note_iov(3, fd, iov, iovcnt, r);
    }
    return r;
}

int close(int fd) {
    RESOLVE(close);
    note_close(fd);
    return real_close(fd);
}

int shutdown(int fd, int how) {
    RESOLVE(shutdown);
    note_close(fd);
    return real_shutdown(fd, how);
}

/* ---------------- RNG observation (wind seed -> value mapping) ----------
 * Enabled only when the wrap script exports TNT_RAND=1. We record srand*
 * calls and the next RNG outputs so the desktop side can correlate the
 * network seed with the displayed wind. Rate-capped to stay harmless.
 */

static _Atomic int g_rand_enabled = 0;
static _Atomic uint32_t g_rand_budget = 4000;
static _Atomic uint64_t g_rand_window = 0;

static void rand_emit(const char *name, long long a0, long long a1) {
    if (!g_rand_enabled) {
        return;
    }
    uint64_t sec = mono_ms() / 1000;
    if (sec != g_rand_window) {
        g_rand_window = sec;
        g_rand_budget = 4000;
    }
    if (g_rand_budget == 0) {
        return;
    }
    g_rand_budget--;
    char buf[96];
    int n = snprintf(buf, sizeof(buf), "%s(%lld)=%lld", name, a0, a1);
    ring_push(6, 0, buf, (uint32_t)n);
}

typedef int (*rand_t)(void);
typedef void (*srand_t)(unsigned int);
typedef long (*lrand48_t)(void);
typedef void (*srand48_t)(long);
typedef long (*random_t)(void);
typedef void (*srandom_t)(unsigned int);
typedef double (*drand48_t)(void);
typedef uint32_t (*arc4random_uniform_t)(uint32_t);

int rand(void) {
    RESOLVE(rand);
    int r = real_rand();
    rand_emit("rand", -1, r);
    return r;
}

void srand(unsigned int s) {
    RESOLVE_VOID(srand);
    rand_emit("srand", (long long)s, 0);
    real_srand(s);
}

long lrand48(void) {
    RESOLVE(lrand48);
    long r = real_lrand48();
    rand_emit("lrand48", -1, r);
    return r;
}

void srand48(long s) {
    RESOLVE_VOID(srand48);
    rand_emit("srand48", s, 0);
    real_srand48(s);
}

long random(void) {
    RESOLVE(random);
    long r = real_random();
    rand_emit("random", -1, r);
    return r;
}

void srandom(unsigned int s) {
    RESOLVE_VOID(srandom);
    rand_emit("srandom", (long long)s, 0);
    real_srandom(s);
}

double drand48(void) {
    RESOLVE(drand48);
    double r = real_drand48();
    rand_emit("drand48", -1, (long long)(r * 1e9));
    return r;
}

uint32_t arc4random_uniform(uint32_t bound) {
    RESOLVE(arc4random_uniform);
    uint32_t r = real_arc4random_uniform(bound);
    rand_emit("arc4u", (long long)bound, (long long)r);
    return r;
}

typedef int (*rand_r_t)(unsigned int *);

int rand_r(unsigned int *ctx) {
    RESOLVE(rand_r);
    int r = real_rand_r(ctx);
    rand_emit("rand_r", ctx ? (long long)*ctx : -1, r);
    return r;
}

/* ---------------- fmt observe ----------------
 * TNT_FMT=1 时拦截 vsnprintf:游戏 UI 格式化出的短小十进制小数
 * 字符串(风速/力度等)以 kind=8 记录转发,含 callsite 库内偏移。
 * 预算 500/s;观察阶段 TLS guard 防 dladdr/ring_push 递归。
 */

static _Atomic int g_fmt_enabled = 0;
static _Atomic uint32_t g_fmt_budget = 500;
static _Atomic uint64_t g_fmt_window = 0;
static _Thread_local int g_fmt_guard = 0;
/* 记录前缀;used 保证 rodata 里有 "FMT+" 可 grep */
static const char fmt_tag[] __attribute__((used)) = "FMT+";

/* 字符串里是否含一个十进制小数 token: [+-]?<digits>.<digits>。
 * 9.5 / -0.3 / wind=10.0 命中;纯整数或无小数不命中。不用 regex/locale。 */
static int fmt_has_decimal(const char *s, size_t len) {
    for (size_t i = 0; i < len; i++) {
        size_t j = i;
        if (s[j] == '+' || s[j] == '-')
            j++;
        size_t d0 = j;
        while (j < len && s[j] >= '0' && s[j] <= '9')
            j++;
        if (j == d0)
            continue;
        if (j >= len || s[j] != '.')
            continue;
        j++;
        size_t d1 = j;
        while (j < len && s[j] >= '0' && s[j] <= '9')
            j++;
        if (j > d1)
            return 1;
    }
    return 0;
}

/* 手写小写 hex(不借 printf 家族,防递归),不带 0x,至多 16 位,至少一位 */
static size_t fmt_hex(char *dst, size_t cap, uint64_t v) {
    char tmp[16];
    int n = 0;
    do {
        unsigned d = (unsigned)(v & 0xf);
        tmp[n++] = (char)(d < 10 ? '0' + d : 'a' + (d - 10));
        v >>= 4;
    } while (v && n < 16);
    size_t out = 0;
    while (n > 0 && out < cap) {
        dst[out++] = tmp[--n];
    }
    return out;
}

/* 共享发送预算:所有 fmt/qnum/qtext 观察共用同一 500/s 窗口 */
static int fmt_budget_take(void) {
    uint64_t sec = mono_ms() / 1000;
    if (sec != g_fmt_window) {
        g_fmt_window = sec;
        g_fmt_budget = 500;
    }
    if (g_fmt_budget == 0)
        return 0;
    g_fmt_budget--;
    return 1;
}

/* 返回地址 → 所在库基址偏移(dladdr 失败时退回绝对地址) */
static uint64_t fmt_callsite_offset(void *ra) {
    uint64_t off = (uint64_t)(uintptr_t)ra;
    Dl_info di;
    if (dladdr(ra, &di) && di.dli_fbase)
        off = (uint64_t)((uintptr_t)ra - (uintptr_t)di.dli_fbase);
    return off;
}

static void fmt_emit(const char *s, size_t vis, void *ra) {
    if (!fmt_budget_take())
        return;
    g_fmt_guard = 1;
    uint64_t off = fmt_callsite_offset(ra);
    char buf[160];
    size_t p = 0;
    memcpy(buf, fmt_tag, 4);
    p += 4;
    p += fmt_hex(buf + p, 16, off);
    buf[p++] = ' ';
    for (size_t i = 0; i < vis && p < sizeof(buf); i++) {
        uint8_t c = (uint8_t)s[i];
        buf[p++] = (c >= 0x20 && c <= 0x7e) ? (char)c : '.';
    }
    ring_push(8, 0, buf, (uint32_t)p);
    g_fmt_guard = 0;
}

typedef int (*vsnprintf_t)(char *, size_t, const char *, va_list);

int vsnprintf(char *dst, size_t cap, const char *fmt, va_list ap) {
    RESOLVE(vsnprintf);
    int r = real_vsnprintf(dst, cap, fmt, ap);
    if (g_fmt_enabled && g_fmt_guard == 0 && dst && cap > 0 && r > 0) {
        size_t vis = (size_t)r;
        if (vis > cap - 1)
            vis = cap - 1;
        if (vis >= 1 && vis <= 80 && fmt_has_decimal(dst, vis)) {
            void *ra = __builtin_return_address(0);
            fmt_emit(dst, vis, ra);
        }
    }
    return r;
}

/* ---- Qt5 数字/文本 hook:TNT_FMT 观察器的扩展 ----
 * 直接拦 QString::setNum(double) 和 QLabel::setNum/setText。
 * arm64 ABI: this=x0, double=v0;real 先执行,观察后置,原样返回。
 */

/* kind=9: QNUM+<tag>+<offsethex> <%.12g> */
static void qnum_emit(const char *tag, double value, void *ra) {
    if (!g_fmt_enabled)
        return;
    if (!fmt_budget_take())
        return;
    g_fmt_guard = 1;
    uint64_t off = fmt_callsite_offset(ra);
    char buf[96];
    /* guard 已置位,这里 snprintf 内部走 vsnprintf 不会递归记录 */
    int n = snprintf(buf, sizeof(buf), "QNUM+%s+%llx %.12g", tag,
                     (unsigned long long)off, value);
    if (n > 0) {
        size_t len = (size_t)n;
        if (len > sizeof(buf) - 1)
            len = sizeof(buf) - 1; /* 截断时按实际写入长度发 */
        ring_push(9, 0, buf, (uint32_t)len);
    }
    g_fmt_guard = 0;
}

/* kind=10: QTEXT+<offsethex> <text>。保守手工解码 Qt5 QString:
 * 首字段是 Data*;Data: size int32@+4, qptrdiff offset@+16, UTF-16 at Data+offset。
 * 不调任何 QString API;调用期传入的是有效对象,bounds 检查防异常布局。 */
static void qtext_emit(const void *qstr, void *ra) {
    if (!g_fmt_enabled || !qstr)
        return;
    g_fmt_guard = 1;
    void *data;
    memcpy(&data, qstr, sizeof(data));
    if (data) {
        int32_t size = 0;
        intptr_t doff = 0;
        memcpy(&size, (const uint8_t *)data + 4, sizeof(size));
        memcpy(&doff, (const uint8_t *)data + 16, sizeof(doff));
        if (size >= 1 && size <= 80 && doff >= 0 && doff <= (intptr_t)(1 << 20)) {
            char text[80];
            const uint8_t *base = (const uint8_t *)data + doff;
            for (int32_t i = 0; i < size; i++) {
                uint16_t ch;
                memcpy(&ch, base + (size_t)i * 2, 2);
                text[i] = (ch >= 0x20 && ch <= 0x7e) ? (char)ch : '.';
            }
            if (fmt_has_decimal(text, (size_t)size) && fmt_budget_take()) {
                uint64_t off = fmt_callsite_offset(ra);
                char buf[160];
                size_t p = 0;
                memcpy(buf, "QTEXT+", 6);
                p += 6;
                p += fmt_hex(buf + p, 16, off);
                buf[p++] = ' ';
                for (int32_t i = 0; i < size && p < sizeof(buf); i++)
                    buf[p++] = text[i];
                ring_push(10, 0, buf, (uint32_t)p);
            }
        }
    }
    g_fmt_guard = 0;
}

typedef void *(*qstring_setnum_d_t)(void *, double, char, int);
typedef void (*qlabel_setnum_d_t)(void *, double);
typedef void (*qlabel_settext_t)(void *, const void *);

void *tnt_qstring_setnum_d(void *self, double value, char format, int precision)
    __asm__("_ZN7QString6setNumEdci");
void tnt_qlabel_setnum_d(void *self, double value)
    __asm__("_ZN6QLabel6setNumEd");
void tnt_qlabel_settext(void *self, const void *qstr)
    __asm__("_ZN6QLabel7setTextERK7QString");

/* QString::setNum(double) → QString&(返回还是 this),先 real 后观察数值 */
void *tnt_qstring_setnum_d(void *self, double value, char format, int precision) {
    static qstring_setnum_d_t real;
    if (!real) {
        real = (qstring_setnum_d_t)dlsym(RTLD_NEXT, "_ZN7QString6setNumEdci");
        if (!real)
            return self;
    }
    void *ret = real(self, value, format, precision);
    if (g_fmt_enabled && g_fmt_guard == 0) {
        void *ra = __builtin_return_address(0);
        qnum_emit("qstr", value, ra);
    }
    return ret;
}

void tnt_qlabel_setnum_d(void *self, double value) {
    static qlabel_setnum_d_t real;
    if (!real) {
        real = (qlabel_setnum_d_t)dlsym(RTLD_NEXT, "_ZN6QLabel6setNumEd");
        if (!real)
            return;
    }
    real(self, value);
    if (g_fmt_enabled && g_fmt_guard == 0) {
        void *ra = __builtin_return_address(0);
        qnum_emit("qlabel", value, ra);
    }
}

void tnt_qlabel_settext(void *self, const void *qstr) {
    static qlabel_settext_t real;
    if (!real) {
        real = (qlabel_settext_t)dlsym(RTLD_NEXT, "_ZN6QLabel7setTextERK7QString");
        if (!real)
            return;
    }
    real(self, qstr);
    /* setText 返回后 qstr 仍有效,这时再保守解码文本 */
    if (g_fmt_enabled && g_fmt_guard == 0) {
        void *ra = __builtin_return_address(0);
        qtext_emit(qstr, ra);
    }
}

/* ---------------- memory scan ----------------
 * 控制文件 /data/local/tmp/tnt_ctl,一行一条命令:
 *   SEED <i64>   → 全部 rw 区 memmem 搜 LE i32/i64,dump 命中点 ±128B hex
 *   F1 <f64>     → 全扫 float32==v(±0.05),命中地址存 tnt_hits,明细写 tnt_scan.out
 *   FN <f64>     → 只复查 tnt_hits 里的地址,筛留仍等于 v 的
 *   SNAP         → 全部 rw 区原始字节快照进 tnt_snap.dat + tnt_snap.idx
 *   DIFF [same]  → 差分扫:首跑 快照vs当前 逐4B字比对;再跑 只复查 H 命中
 *   RATIO <a> <b>→ H 命中按 当前/基线 ≈ b/a(±15%) 筛,f32/f64/i32 任一即可
 *   EQ <f64>     → H 命中当前值==v 筛(f32/f64±0.05,i32==llround(v)或v*10)
 *   HITS         → 打印前 100 个 H 命中的当前值
 *   RD <hex>     → 读该地址 64B hex
 * 命中地址文件 /data/local/tmp/tnt_hits,输出 /data/local/tmp/tnt_scan.out
 */

#define CTL_PATH "/data/local/tmp/tnt_ctl"
#define OUT_PATH "/data/local/tmp/tnt_scan.out"
#define HITS_PATH "/data/local/tmp/tnt_hits"
#define SNAP_DAT "/data/local/tmp/tnt_snap.dat"
#define SNAP_IDX "/data/local/tmp/tnt_snap.idx"
#define HIT_CAP 2000000

static int g_memfd = -1;

static int mem_open(void) {
    if (g_memfd < 0)
        g_memfd = open("/proc/self/mem", O_RDONLY | O_CLOEXEC);
    return g_memfd;
}

struct region {
    uint64_t lo, hi;
};

static int read_regions(struct region *out, int cap) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f)
        return 0;
    /* 本函数与 ctl_main/扫描线程同一调用栈:栈上有搜索目标的副本
     * (dv/target/ft/dt/i10),跳过该映射防自扫描假命中 */
    uint8_t stack_probe;
    uint64_t scanner_stack_addr = (uint64_t)(uintptr_t)&stack_probe;
    char line[256];
    int n = 0;
    while (fgets(line, sizeof(line), f) && n < cap) {
        unsigned long lo, hi;
        char perm[8];
        if (sscanf(line, "%lx-%lx %4s", &lo, &hi, perm) == 3) {
            /* 变量必然在可写区;跳过超大映射保持扫描 < 几秒钟 */
            if (perm[0] == 'r' && perm[1] == 'w' && (hi - lo) < (64UL << 20)) {
                if (scanner_stack_addr >= lo && scanner_stack_addr < hi)
                    continue;
                out[n].lo = lo;
                out[n].hi = hi;
                n++;
            }
        }
    }
    fclose(f);
    return n;
}

static void hexdump_at(FILE *out, uint64_t addr, const uint8_t *buf, size_t len) {
    fprintf(out, "@%lx:", (unsigned long)addr);
    for (size_t i = 0; i < len; i++)
        fprintf(out, " %02x", buf[i]);
    fputc('\n', out);
}

/* 在单个缓冲区里找 pat 的所有出现位置,回调打印 */
static void memmem_dump(uint64_t base, const uint8_t *buf, size_t len,
                        const uint8_t *pat, size_t plen, FILE *out, int fd) {
    if (plen > len)
        return;
    for (size_t k = 0; k + plen <= len; k++) {
        if (memcmp(buf + k, pat, plen) == 0) {
            uint64_t a = base + k;
            uint8_t ctx[256];
            uint64_t caddr = a >= 128 ? a - 128 : a;
            ssize_t got = pread(fd, ctx, sizeof(ctx), (off_t)caddr);
            fprintf(out, "SEED@0x%lx ctx@%lx:\n", (unsigned long)a,
                    (unsigned long)caddr);
            if (got > 0)
                hexdump_at(out, caddr, ctx, (size_t)got);
        }
    }
}

/* F1: 全扫三种存法 float32/double(≈target ±eps)/int32==round(target*10),
 * 命中按类型存 tnt_hits ("F64 <addr>" / "I10 <addr>") */
static void scan_float_first(double target, float eps) {
    int fd = mem_open();
    FILE *out = fopen(OUT_PATH, "w");
    FILE *hf = fopen(HITS_PATH, "w");
    if (fd < 0 || !out) {
        if (out)
            fclose(out);
        if (hf)
            fclose(hf);
        return;
    }
    struct region *rs = malloc(sizeof(struct region) * 8192);
    int n = read_regions(rs, 8192);
    uint8_t *buf = malloc(1 << 22);
    float ft = (float)target;
    double dt = target;
    int32_t i10 = (int32_t)lrint(target * 10.0);
    int hits = 0, scanned = 0;
    for (int i = 0; i < n; i++) {
        uint64_t lo = rs[i].lo, hi = rs[i].hi;
        for (uint64_t off = lo; off < hi; off += (1 << 22)) {
            size_t want = (hi - off) < (1UL << 22) ? (size_t)(hi - off) : (1UL << 22);
            ssize_t got = pread(fd, buf, want, (off_t)off);
            if (got < 4)
                continue;
            scanned += (int)got;
            for (ssize_t k = 0; k + 8 <= got; k += 4) {
                float v32;
                double v64;
                int32_t iv;
                memcpy(&v32, buf + k, 4);
                memcpy(&iv, buf + k, 4);
                memcpy(&v64, buf + k, 8);
                uint64_t a = off + (uint64_t)k;
                if (fabsf(v32 - ft) < eps) {
                    if (hf) fprintf(hf, "F32 %lx\n", (unsigned long)a);
                    fprintf(out, "F32 %lx = %g\n", (unsigned long)a, v32);
                    hits++;
                }
                if (fabs(v64 - dt) < (double)eps) {
                    if (hf) fprintf(hf, "F64 %lx\n", (unsigned long)a);
                    fprintf(out, "F64 %lx = %g\n", (unsigned long)a, v64);
                    hits++;
                }
                if (iv == i10) {
                    if (hf) fprintf(hf, "I10 %lx\n", (unsigned long)a);
                    fprintf(out, "I10 %lx = %d\n", (unsigned long)a, iv);
                    hits++;
                }
            }
        }
    }
    fprintf(out, "# F1 %.4f eps %.3f: %d regions, %d bytes, %d hits\n", target,
            eps, n, scanned, hits);
    if (hf)
        fclose(hf);
    fclose(out);
    free(buf);
    free(rs);
}

/* FN: 按 hits 文件里的类型分别复筛 */
static void scan_float_next(double target, float eps) {
    FILE *hf = fopen(HITS_PATH, "r");
    FILE *out = fopen(OUT_PATH, "w");
    int fd = mem_open();
    if (!hf || !out || fd < 0) {
        if (hf) fclose(hf);
        if (out) fclose(out);
        return;
    }
    /* app 不能在 /data/local/tmp 建 .new/rename:
     * 先把旧命中全读进内存再原地 truncate 写回 */
    struct fnent {
        char tag[4];
        uint64_t addr;
    };
    struct fnent *ents = malloc(sizeof(struct fnent) * 8192);
    int n = 0;
    if (ents) {
        char line[64];
        while (fgets(line, sizeof(line), hf) && n < 8192) {
            char tag[8];
            uint64_t a;
            if (sscanf(line, "%3s %lx", tag, &a) != 2)
                continue;
            memcpy(ents[n].tag, tag, 4);
            ents[n].addr = a;
            n++;
        }
    }
    fclose(hf);
    if (!ents) {
        fprintf(out, "# ERROR: cannot write hits\n");
        fclose(out);
        return;
    }
    FILE *of = fopen(HITS_PATH, "w");
    if (!of) {
        fprintf(out, "# ERROR: cannot write hits\n");
        free(ents);
        fclose(out);
        return;
    }
    int32_t i10 = (int32_t)lrint(target * 10.0);
    int keep = 0;
    for (int i = 0; i < n; i++) {
        const char *tag = ents[i].tag;
        uint64_t a = ents[i].addr;
        int ok = 0;
        double rv = 0;
        if (tag[0] == 'F' && tag[1] == '3') {
            float v;
            ok = pread(fd, &v, 4, (off_t)a) == 4 && fabsf(v - (float)target) < eps;
            rv = v;
        } else if (tag[0] == 'F' && tag[1] == '6') {
            double v;
            ok = pread(fd, &v, 8, (off_t)a) == 8 && fabs(v - target) < (double)eps;
            rv = v;
        } else if (tag[0] == 'I') {
            int32_t v;
            ok = pread(fd, &v, 4, (off_t)a) == 4 && v == i10;
            rv = v;
        }
        if (ok) {
            fprintf(of, "%s %lx\n", tag, (unsigned long)a);
            fprintf(out, "%s %lx = %g\n", tag, (unsigned long)a, rv);
            keep++;
        }
    }
    fclose(of);
    fprintf(out, "# FN %.4f: %d/%d 存活\n", target, keep, n);
    fclose(out);
    free(ents);
}

/* SEED: 全 rw 区 memmem 搜 i32/i64 LE 并 dump 上下文 */
static void scan_seed(int64_t seed) {
    int fd = mem_open();
    FILE *out = fopen(OUT_PATH, "w");
    if (fd < 0 || !out) {
        if (out) fclose(out);
        return;
    }
    struct region *rs = malloc(sizeof(struct region) * 8192);
    int n = read_regions(rs, 8192);
    uint8_t *buf = malloc(1 << 22);
    uint8_t p4[4], p8[8];
    memcpy(p4, &seed, 4);
    memcpy(p8, &seed, 8);
    int hits = 0;
    for (int i = 0; i < n; i++) {
        uint64_t lo = rs[i].lo, hi = rs[i].hi;
        for (uint64_t off = lo; off < hi; off += (1 << 22)) {
            size_t want = (hi - off) < (1UL << 22) ? (size_t)(hi - off) : (1UL << 22);
            ssize_t got = pread(fd, buf, want, (off_t)off);
            if (got < 8)
                continue;
            memmem_dump(off, buf, (size_t)got, p8, 8, out, fd);
            memmem_dump(off, buf, (size_t)got, p4, 4, out, fd);
        }
    }
    fprintf(out, "# SEED %lld done\n", (long long)seed);
    fclose(out);
    free(buf);
    free(rs);
    (void)hits;
}

/* RD: 读某地址 64B */
static void scan_read(uint64_t addr) {
    int fd = mem_open();
    FILE *out = fopen(OUT_PATH, "w");
    if (fd < 0 || !out) {
        if (out) fclose(out);
        return;
    }
    uint8_t buf[64];
    ssize_t got = pread(fd, buf, sizeof(buf), (off_t)addr);
    if (got > 0)
        hexdump_at(out, addr, buf, (size_t)got);
    else
        fprintf(out, "# RD %lx failed errno\n", (unsigned long)addr);
    fclose(out);
}

/* ---------------- 差分扫描(GameGuardian 风格) ----------------
 * SNAP        → 全部 rw 区字节 → snap.dat,索引 snap.idx("lo hi off len")
 * DIFF [same] → 首跑:快照 vs 当前逐 4B 字比对;再跑:只复查 tnt_hits 的 H 行
 *               命中行 "H <addr> <当前8B hex>" 存回 tnt_hits(基线随写随更新)
 * RATIO a b   → H 命中按 当前/基线 ≈ b/a (±15%) 筛,f32/f64/i32 任一满足
 * EQ <f64>    → H 命中当前值 == v 筛(f32/f64±0.05,i32==v 或 v*10)
 * HITS        → 打印前 100 个命中地址的当前值
 */

struct hitent {
    uint64_t addr, base;
};

/* SNAP: dump 所有 rw 区原始字节到 snap.dat;idx 每行 "lo hi file_off len" */
static void do_snap(void) {
    int fd = mem_open();
    FILE *out = fopen(OUT_PATH, "w");
    /* mem 打不开就别动旧快照;dat/idx 用 "w" 打开会截断 */
    if (fd < 0 || !out) {
        if (out) fclose(out);
        return;
    }
    FILE *dat = fopen(SNAP_DAT, "wb");
    FILE *idx = fopen(SNAP_IDX, "w");
    if (!dat || !idx) {
        if (dat) fclose(dat);
        if (idx) fclose(idx);
        fclose(out);
        return;
    }
    struct region *rs = malloc(sizeof(struct region) * 8192);
    int n = read_regions(rs, 8192);
    uint8_t *buf = malloc(1 << 22);
    long long foff = 0;
    for (int i = 0; i < n; i++) {
        uint64_t lo = rs[i].lo, hi = rs[i].hi;
        long long len = (long long)(hi - lo);
        fprintf(idx, "%lx %lx %lld %lld\n", (unsigned long)lo,
                (unsigned long)hi, foff, len);
        for (uint64_t off = lo; off < hi; off += (1 << 22)) {
            size_t want = (hi - off) < (1UL << 22) ? (size_t)(hi - off) : (1UL << 22);
            /* pread 失败的块写零,保持文件偏移与 idx 一致 */
            ssize_t got = pread(fd, buf, want, (off_t)off);
            if (got < 0)
                got = 0;
            if ((size_t)got < want)
                memset(buf + got, 0, want - (size_t)got);
            fwrite(buf, 1, want, dat);
        }
        foff += len;
    }
    fflush(dat);
    fprintf(out, "# SNAP: %d regions %lld bytes\n", n, foff);
    fclose(idx);
    fclose(dat);
    fclose(out);
    /* SNAP 开启新差分会话:清掉旧候选(只能 truncate,不能建文件) */
    FILE *hf = fopen(HITS_PATH, "w");
    if (hf) {
        fclose(hf);
    } else {
        FILE *o2 = fopen(OUT_PATH, "a");
        if (o2) {
            fprintf(o2, "# ERROR: cannot clear hits\n");
            fclose(o2);
        }
    }
    free(buf);
    free(rs);
}

/* 读 tnt_hits 里所有 "H <addr> <base8hex>" 行进内存数组(上限 HIT_CAP)。
 * 返回条目数;文件不存在返回 -1(区别于"空/全是旧格式"的 0)。 */
static int load_hits(struct hitent **out) {
    *out = NULL;
    FILE *hf = fopen(HITS_PATH, "r");
    if (!hf)
        return -1;
    struct hitent *arr = malloc(sizeof(struct hitent) * HIT_CAP);
    if (!arr) {
        fclose(hf);
        return 0;
    }
    char line[96];
    int n = 0;
    while (fgets(line, sizeof(line), hf) && n < HIT_CAP) {
        unsigned long a;
        unsigned long long b;
        if (line[0] != 'H' || line[1] != ' ')
            continue;
        if (sscanf(line, "H %lx %llx", &a, &b) != 2 || a == 0)
            continue;
        arr[n].addr = a;
        arr[n].base = b;
        n++;
    }
    fclose(hf);
    *out = arr;
    return n;
}

/* DIFF: same=0 找变了的,same=1 找没变的。
 * hits 里有 H 行 → 逐条复查并更新基线;无 H 行(空/旧格式) → 对快照全量比对。 */
static void do_diff(int same) {
    int fd = mem_open();
    FILE *out = fopen(OUT_PATH, "w");
    if (!out)
        return;
    /* mem 读不了就不动 hits,防止一次失败清空全部命中 */
    if (fd < 0) {
        fprintf(out, "# DIFF: no hits\n");
        fclose(out);
        return;
    }
    struct hitent *hits = NULL;
    int n = load_hits(&hits);
    const char *mode = same ? "same" : "changed";
    if (n > 0) {
        /* 二次以后:只复查 H 命中,比对前 4 字节,基线更新为当前 8B。
         * hits 已全在内存,直接原地 truncate 写回(tmp 不能建 .new/rename) */
        FILE *of = fopen(HITS_PATH, "w");
        if (!of) {
            fprintf(out, "# ERROR: cannot write hits\n");
            free(hits);
            fclose(out);
            return;
        }
        int keep = 0;
        for (int i = 0; i < n; i++) {
            uint8_t cur[8];
            memset(cur, 0, 8);
            ssize_t got = pread(fd, cur, 8, (off_t)hits[i].addr);
            if (got < 4)
                continue;
            uint8_t old[8];
            memcpy(old, &hits[i].base, 8);
            int changed = memcmp(old, cur, 4) != 0;
            if (changed != same) {
                uint64_t v;
                memcpy(&v, cur, 8);
                fprintf(of, "H %lx %016llx\n", (unsigned long)hits[i].addr,
                        (unsigned long long)v);
                keep++;
            }
        }
        fclose(of);
        fprintf(out, "# DIFF %s: %d hits\n", mode, keep);
    } else if (n == 0) {
        /* 首跑:快照 vs 当前内存,4B 对齐逐字比 */
        FILE *idx = fopen(SNAP_IDX, "r");
        FILE *dat = fopen(SNAP_DAT, "rb");
        if (!idx || !dat) {
            fprintf(out, "# DIFF: no hits\n");
        } else {
            FILE *of = fopen(HITS_PATH, "w");
            if (!of) {
                fprintf(out, "# ERROR: cannot write hits\n");
            } else {
            uint8_t *sb = malloc(1 << 22); /* 快照块 */
            uint8_t *cb = malloc(1 << 22); /* 当前块 */
            char line[128];
            int keep = 0;
            while (fgets(line, sizeof(line), idx)) {
                unsigned long lo, hi;
                long long foff, len;
                if (sscanf(line, "%lx %lx %lld %lld", &lo, &hi, &foff, &len) != 4)
                    continue;
                for (long long d = 0; d < len; d += (1 << 22)) {
                    size_t want = (len - d) < (1LL << 22) ? (size_t)(len - d)
                                                          : (size_t)(1 << 22);
                    ssize_t gs = pread(fileno(dat), sb, want, (off_t)(foff + d));
                    if (gs < 0)
                        gs = 0;
                    if ((size_t)gs < want)
                        memset(sb + gs, 0, want - (size_t)gs);
                    /* 读当前内存失败的块按全零比 */
                    ssize_t gc = pread(fd, cb, want, (off_t)(lo + (uint64_t)d));
                    if (gc < 0)
                        gc = 0;
                    if ((size_t)gc < want)
                        memset(cb + gc, 0, want - (size_t)gc);
                    for (size_t k = 0; k + 4 <= want; k += 4) {
                        uint32_t a, b;
                        memcpy(&a, sb + k, 4);
                        memcpy(&b, cb + k, 4);
                        if ((a != b) != same) {
                            uint64_t addr = lo + (uint64_t)d + (uint64_t)k;
                            uint8_t v8[8];
                            memset(v8, 0, 8);
                            if (k + 8 <= want)
                                memcpy(v8, cb + k, 8);
                            else
                                pread(fd, v8, 8, (off_t)addr);
                            uint64_t v;
                            memcpy(&v, v8, 8);
                            fprintf(of, "H %lx %016llx\n",
                                    (unsigned long)addr,
                                    (unsigned long long)v);
                            keep++;
                        }
                    }
                }
            }
            free(sb);
            free(cb);
            fclose(of);
            fprintf(out, "# DIFF %s: %d hits\n", mode, keep);
            }
        }
        if (idx)
            fclose(idx);
        if (dat)
            fclose(dat);
    } else {
        /* hits 文件丢失/不可解析,无从差分 */
        fprintf(out, "# DIFF: no hits\n");
    }
    free(hits);
    fclose(out);
}

/* RATIO a b: 命中里 当前/基线 ≈ b/a (±15%) 的留下,f32/f64/i32 任一满足即可 */
static void do_ratio(double a, double b) {
    int fd = mem_open();
    FILE *out = fopen(OUT_PATH, "w");
    if (!out)
        return;
    struct hitent *hits = NULL;
    int n = load_hits(&hits);
    if (n < 0)
        n = 0;
    if (fd < 0) {
        fprintf(out, "# RATIO %g %g: 0/%d kept\n", a, b, n);
        free(hits);
        fclose(out);
        return;
    }
    /* 只比绝对比例:用户输入屏幕风速大小,不带方向符号 */
    if (fabs(a) < 1e-9) {
        fprintf(out, "# RATIO: previous value is zero\n");
        free(hits);
        fclose(out);
        return;
    }
    double target = fabs(b / a);
    double tol = 0.15 * fabs(target);
    FILE *of = fopen(HITS_PATH, "w");
    if (!of) {
        fprintf(out, "# ERROR: cannot write hits\n");
        free(hits);
        fclose(out);
        return;
    }
    int keep = 0;
    for (int i = 0; i < n; i++) {
        uint8_t cur[8];
        memset(cur, 0, 8);
        if (pread(fd, cur, 8, (off_t)hits[i].addr) < 8)
            continue;
        uint8_t old[8];
        memcpy(old, &hits[i].base, 8);
        float f0, f1;
        double d0, d1;
        int32_t i0, i1;
        memcpy(&f0, old, 4);
        memcpy(&f1, cur, 4);
        memcpy(&d0, old, 8);
        memcpy(&d1, cur, 8);
        memcpy(&i0, old, 4);
        memcpy(&i1, cur, 4);
        int ok = (f0 != 0 && fabs(fabs((double)f1 / f0) - target) <= tol) ||
                 (d0 != 0 && fabs(fabs(d1 / d0) - target) <= tol) ||
                 (i0 != 0 && fabs(fabs((double)i1 / i0) - target) <= tol);
        if (ok) {
            uint64_t v;
            memcpy(&v, cur, 8);
            fprintf(of, "H %lx %016llx\n", (unsigned long)hits[i].addr,
                    (unsigned long long)v);
            keep++;
        }
    }
    fclose(of);
    fprintf(out, "# RATIO %g %g: %d/%d kept\n", a, b, keep, n);
    free(hits);
    fclose(out);
}

/* EQ v: 命中当前值==v 的留下(f32/f64 ±0.05,i32==llround(v) 或 llround(v*10)) */
static void do_eq(double v) {
    int fd = mem_open();
    FILE *out = fopen(OUT_PATH, "w");
    if (!out)
        return;
    struct hitent *hits = NULL;
    int n = load_hits(&hits);
    if (n < 0)
        n = 0;
    if (fd < 0) {
        fprintf(out, "# EQ %g: 0/%d kept\n", v, n);
        free(hits);
        fclose(out);
        return;
    }
    float fv = (float)v;
    int32_t i10 = (int32_t)llround(v * 10.0);
    int32_t i1x = (int32_t)llround(v);
    FILE *of = fopen(HITS_PATH, "w");
    if (!of) {
        fprintf(out, "# ERROR: cannot write hits\n");
        free(hits);
        fclose(out);
        return;
    }
    int keep = 0;
    for (int i = 0; i < n; i++) {
        uint8_t cur[8];
        memset(cur, 0, 8);
        if (pread(fd, cur, 8, (off_t)hits[i].addr) < 8)
            continue;
        float f1;
        double d1;
        int32_t i1;
        memcpy(&f1, cur, 4);
        memcpy(&d1, cur, 8);
        memcpy(&i1, cur, 4);
        int ok = fabsf(f1 - fv) <= 0.05f || fabs(d1 - v) <= 0.05 ||
                 i1 == i10 || i1 == i1x;
        if (ok) {
            uint64_t hv;
            memcpy(&hv, cur, 8);
            fprintf(of, "H %lx %016llx\n", (unsigned long)hits[i].addr,
                    (unsigned long long)hv);
            keep++;
        }
    }
    fclose(of);
    fprintf(out, "# EQ %g: %d/%d kept\n", v, keep, n);
    free(hits);
    fclose(out);
}

/* HITS: 打印前 100 个命中的当前值(实时读,不是基线) */
static void show_hits(void) {
    int fd = mem_open();
    FILE *out = fopen(OUT_PATH, "w");
    if (!out)
        return;
    struct hitent *hits = NULL;
    int n = load_hits(&hits);
    if (n < 0)
        n = 0;
    int shown = n < 100 ? n : 100;
    for (int i = 0; i < shown; i++) {
        uint8_t cur[8];
        memset(cur, 0, 8);
        if (fd >= 0)
            pread(fd, cur, 8, (off_t)hits[i].addr);
        float f1;
        double d1;
        int32_t i1;
        uint64_t hv;
        memcpy(&f1, cur, 4);
        memcpy(&d1, cur, 8);
        memcpy(&i1, cur, 4);
        memcpy(&hv, cur, 8);
        fprintf(out, "H %lx: f32=%g f64=%g i32=%d hex=%016llx\n",
                (unsigned long)hits[i].addr, f1, d1, i1, (unsigned long long)hv);
    }
    fprintf(out, "# HITS: %d shown / %d total\n", shown, n);
    free(hits);
    fclose(out);
}

/* ---------------- wind watch ----------------
 * WATCH f32|f64|i32 <hexaddr> → 每200ms读该地址,值变就发 ring type-7 "WIND <v>"
 * WATCH off → 停止
 */
static _Atomic uint64_t g_watch_addr = 0;
static _Atomic int g_watch_type = 0; /* 0=off 1=f32 2=f64 3=i32 */

static void *watch_main(void *arg) {
    (void)arg;
    double last = 1e30;
    for (;;) {
        uint64_t a = g_watch_addr;
        int t = g_watch_type;
        if (a && t) {
            int fd = mem_open();
            if (fd >= 0) {
                double v = 0;
                if (t == 1) {
                    float x;
                    if (pread(fd, &x, 4, (off_t)a) == 4) v = x;
                } else if (t == 2) {
                    double x;
                    if (pread(fd, &x, 8, (off_t)a) == 8) v = x;
                } else {
                    int32_t x;
                    if (pread(fd, &x, 4, (off_t)a) == 4) v = x / 10.0;
                }
                if (v != last && v > -1e6 && v < 1e6) {
                    last = v;
                    char buf[48];
                    int n = snprintf(buf, sizeof(buf), "WIND %.3f", v);
                    ring_push(7, 0, buf, (uint32_t)n);
                }
            }
        }
        usleep(200 * 1000);
    }
    return NULL;
}

static void *ctl_main(void *arg) {
    (void)arg;
    char line[256];
    for (;;) {
        FILE *f = fopen(CTL_PATH, "r");
        if (f) {
            if (fgets(line, sizeof(line), f)) {
                /* 立即清文件,避免重复执行 */
                FILE *t = fopen(CTL_PATH, "w");
                if (t) fclose(t);
                char cmd[16];
                double dv, dv2;
                long long sv;
                if (sscanf(line, "F1 %lf", &dv) == 1) {
                    scan_float_first(dv, 0.05f);
                } else if (sscanf(line, "FN %lf", &dv) == 1) {
                    scan_float_next(dv, 0.05f);
                } else if (sscanf(line, "SEED %lld", &sv) == 1) {
                    scan_seed(sv);
                } else if (strncmp(line, "SNAP", 4) == 0) {
                    do_snap();
                } else if (strncmp(line, "DIFF", 4) == 0) {
                    char mode[16] = "changed";
                    sscanf(line, "DIFF %15s", mode);
                    do_diff(strcmp(mode, "same") == 0);
                } else if (sscanf(line, "RATIO %lf %lf", &dv, &dv2) == 2) {
                    do_ratio(dv, dv2);
                } else if (sscanf(line, "EQ %lf", &dv) == 1) {
                    do_eq(dv);
                } else if (strncmp(line, "HITS", 4) == 0) {
                    show_hits();
                } else if (sscanf(line, "RD %15s", cmd) == 1) {
                    uint64_t a = strtoull(cmd, NULL, 16);
                    if (a) scan_read(a);
                } else if (strncmp(line, "WATCH", 5) == 0) {
                    char wtype[8];
                    uint64_t a;
                    if (sscanf(line, "WATCH %3s %lx", wtype, &a) == 2) {
                        g_watch_type = (wtype[1] == '6') ? 2
                                     : (wtype[0] == 'I') ? 3 : 1;
                        g_watch_addr = a;
                    } else {
                        g_watch_type = 0;
                        g_watch_addr = 0;
                    }
                }
            }
            fclose(f);
        }
        usleep(200 * 1000);
    }
    return NULL;
}

#include "tnt_tls.h"

/* ---------------- init ---------------- */

__attribute__((constructor)) static void tnt_sniff_init(void) {
    pthread_t th;
    const char *port = getenv("TNT_TUNNEL_PORT");
    if (port) {
        char *end;
        long value = strtol(port, &end, 10);
        if (*port && !*end && value > 0 && value <= 65535) g_tunnel_port = (int)value;
    }
    g_rand_enabled = getenv("TNT_RAND") != NULL ? 1 : 0;
    g_fmt_enabled = getenv("TNT_FMT") != NULL ? 1 : 0;
    if (pthread_create(&th, NULL, writer_main, NULL) == 0) {
        pthread_detach(th);
    }
    pthread_t ct;
    if (pthread_create(&ct, NULL, ctl_main, NULL) == 0) {
        pthread_detach(ct);
    }
    pthread_t wt;
    if (pthread_create(&wt, NULL, watch_main, NULL) == 0) {
        pthread_detach(wt);
    }
    LOGI("loaded pid=%d rand=%d fmt=%d memscan=7 tls=3", getpid(), g_rand_enabled, g_fmt_enabled);
}
