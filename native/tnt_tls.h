/* Qt 5 resolves OpenSSL through QLibrary::resolve, bypassing ordinary
 * SSL_read/SSL_write symbol interposition. Replace only the resolved pointers;
 * leave all other QLibrary lookups alone. No TLS keys/certificates are changed.
 *
 * Qt uses memory BIOs, so SSL_get_fd can return -1. Each SSL object instead gets
 * a synthetic tunnel stream ID with bit 31 set. IDs are never socket fds.
 * kind 11/12 = accepted TX / decrypted RX. Socket ciphertext stays kind 2/3.
 * Include after ring_push and the libc interposers in tnt_sniff.c.
 */
#ifndef TNT_TLS_H
#define TNT_TLS_H

typedef int (*tls_read_fn)(void *, void *, int);
typedef int (*tls_write_fn)(void *, const void *, int);
typedef int (*tls_read_ex_fn)(void *, void *, size_t, size_t *);
typedef int (*tls_write_ex_fn)(void *, const void *, size_t, size_t *);
typedef void (*tls_free_fn)(void *);
typedef int (*tls_clear_fn)(void *);
typedef void *(*qt_resolve_fn)(void *, const char *);
static _Atomic(tls_read_fn) tls_real_read;
static _Atomic(tls_write_fn) tls_real_write;
static _Atomic(tls_read_ex_fn) tls_real_read_ex;
static _Atomic(tls_write_ex_fn) tls_real_write_ex;
static _Atomic(tls_free_fn) tls_real_free;
static _Atomic(tls_clear_fn) tls_real_clear;
static _Atomic(qt_resolve_fn) tls_real_resolve;
static _Thread_local unsigned tls_depth;

static pthread_mutex_t tls_mu = PTHREAD_MUTEX_INITIALIZER;
static struct { void *ssl; uint32_t id; } tls_streams[128];
static uint32_t tls_next_id = 0x80000000u;

static void tls_capture(void *ssl, uint8_t kind, const void *buf, size_t n) {
    if (!ssl || !buf || !n) return;
    pthread_mutex_lock(&tls_mu);
    size_t slot = 128;
    for (size_t i = 0; i < 128; i++) {
        if (tls_streams[i].ssl == ssl) { slot = i; break; }
        if (!tls_streams[i].ssl && slot == 128) slot = i;
    }
    /* Capture exhaustion must never prevent the game's real SSL operation. */
    if (slot == 128 || (!tls_streams[slot].ssl && tls_next_id == UINT32_MAX)) {
        pthread_mutex_unlock(&tls_mu);
        return;
    }
    if (!tls_streams[slot].ssl) {
        tls_streams[slot].ssl = ssl;
        tls_streams[slot].id = ++tls_next_id;
        const char peer[] = "tls:openssl (synthetic SSL stream, not socket fd)";
        ring_push(1, tls_streams[slot].id, peer, sizeof(peer) - 1);
    }
    uint32_t id = tls_streams[slot].id;
    const uint8_t *p = buf;
    while (n) {
        uint32_t take = n > REC_MAX_PAYLOAD ? REC_MAX_PAYLOAD : (uint32_t)n;
        ring_push(kind, id, p, take);
        p += take;
        n -= take;
    }
    pthread_mutex_unlock(&tls_mu);
}

static void tls_forget(void *ssl) {
    if (!ssl) return;
    pthread_mutex_lock(&tls_mu);
    for (size_t i = 0; i < 128; i++) {
        if (tls_streams[i].ssl == ssl) {
            ring_push(4, tls_streams[i].id, "", 0);
            tls_streams[i].ssl = NULL;
            break;
        }
    }
    pthread_mutex_unlock(&tls_mu);
}

static int tnt_ssl_read(void *ssl, void *buf, int n) {
    tls_read_fn fn = atomic_load(&tls_real_read);
    ++tls_depth;
    int r = fn(ssl, buf, n);
    int saved = errno;
    if (--tls_depth == 0 && r > 0 && r <= n) tls_capture(ssl, 12, buf, (size_t)r);
    errno = saved;
    return r;
}
static int tnt_ssl_write(void *ssl, const void *buf, int n) {
    tls_write_fn fn = atomic_load(&tls_real_write);
    ++tls_depth;
    int r = fn(ssl, buf, n);
    int saved = errno;
    if (--tls_depth == 0 && r > 0 && r <= n) tls_capture(ssl, 11, buf, (size_t)r);
    errno = saved;
    return r;
}
static int tnt_ssl_read_ex(void *ssl, void *buf, size_t n, size_t *done) {
    tls_read_ex_fn fn = atomic_load(&tls_real_read_ex);
    ++tls_depth;
    int r = fn(ssl, buf, n, done);
    int saved = errno;
    if (--tls_depth == 0 && r == 1 && done && *done <= n) tls_capture(ssl, 12, buf, *done);
    errno = saved;
    return r;
}
static int tnt_ssl_write_ex(void *ssl, const void *buf, size_t n, size_t *done) {
    tls_write_ex_fn fn = atomic_load(&tls_real_write_ex);
    ++tls_depth;
    int r = fn(ssl, buf, n, done);
    int saved = errno;
    if (--tls_depth == 0 && r == 1 && done && *done <= n) tls_capture(ssl, 11, buf, *done);
    errno = saved;
    return r;
}
static void tnt_ssl_free(void *ssl) {
    int saved = errno;
    tls_forget(ssl);
    errno = saved;
    atomic_load(&tls_real_free)(ssl);
}
static int tnt_ssl_clear(void *ssl) {
    int r = atomic_load(&tls_real_clear)(ssl);
    int saved = errno;
    if (r == 1) tls_forget(ssl);
    errno = saved;
    return r;
}

/* Avoid silently forwarding one OpenSSL build's objects to another build. */
static void *tls_wrap_symbol(const char *name, void *original) {
    if (!original || !name) return original;
#define TLS_WRAP(symbol, type, storage, replacement)                      \
    if (strcmp(name, symbol) == 0) {                                     \
        type expected = NULL;                                           \
        if (atomic_compare_exchange_strong(&storage, &expected, (type)original)) { \
            LOGI("tls=3 resolved %s", symbol);                           \
            return (void *)replacement;                                 \
        }                                                               \
        return expected == (type)original ? (void *)replacement : original; \
    }
    TLS_WRAP("SSL_read", tls_read_fn, tls_real_read, tnt_ssl_read)
    TLS_WRAP("SSL_write", tls_write_fn, tls_real_write, tnt_ssl_write)
    TLS_WRAP("SSL_read_ex", tls_read_ex_fn, tls_real_read_ex, tnt_ssl_read_ex)
    TLS_WRAP("SSL_write_ex", tls_write_ex_fn, tls_real_write_ex, tnt_ssl_write_ex)
    TLS_WRAP("SSL_free", tls_free_fn, tls_real_free, tnt_ssl_free)
    TLS_WRAP("SSL_clear", tls_clear_fn, tls_real_clear, tnt_ssl_clear)
#undef TLS_WRAP
    return original;
}

#ifdef __ANDROID__
void *tnt_qlibrary_resolve(void *, const char *) __asm__("_ZN8QLibrary7resolveEPKc");
#endif
void *tnt_qlibrary_resolve(void *library, const char *name) {
    const void *caller = __builtin_return_address(0);
    qt_resolve_fn fn = atomic_load(&tls_real_resolve);
    if (!fn) {
        fn = (qt_resolve_fn)dlsym(RTLD_NEXT, "_ZN8QLibrary7resolveEPKc");
        /* Android's Qt loader uses RTLD_LOCAL. RTLD_NEXT cannot see QtCore
         * in that local dependency group, even though QtNetwork calls us.
         * Search the already-loaded QtCore explicitly, without loading a
         * different Qt build. Keep the handle to pin the resolved function.
         */
        if (!fn) {
            void *core = dlopen("libQt5Core_arm64-v8a.so", RTLD_NOW | RTLD_NOLOAD);
            if (core) fn = (qt_resolve_fn)dlsym(core, "_ZN8QLibrary7resolveEPKc");
        }
        /* A Java classloader namespace is separate from the preload's default
         * namespace. Bionic's loader entry accepts the original Qt caller
         * address, selecting its namespace. NOLOAD forbids duplicate Qt loads.
         * This optional Android ABI is looked up dynamically, never linked.
         */
#ifdef __ANDROID__
        if (!fn) {
            typedef void *(*loader_open_fn)(const char *, int, const void *);
            loader_open_fn open_in_caller = (loader_open_fn)dlsym(RTLD_DEFAULT, "__loader_dlopen");
            if (open_in_caller) {
                void *core = open_in_caller("libQt5Core_arm64-v8a.so", RTLD_NOW | RTLD_NOLOAD, caller);
                if (core) fn = (qt_resolve_fn)dlsym(core, "_ZN8QLibrary7resolveEPKc");
            }
        }
#else
        (void)caller;
#endif
        if (!fn || (void *)fn == (void *)tnt_qlibrary_resolve) {
            LOGI("tls=3 ERROR: cannot find original QLibrary::resolve");
            return NULL;
        }
        LOGI("tls=3 original QLibrary::resolve found");
        atomic_store(&tls_real_resolve, fn);
    }
    void *original = fn(library, name);
    int saved = errno;
    void *result = tls_wrap_symbol(name, original);
    errno = saved;
    return result;
}
#endif
