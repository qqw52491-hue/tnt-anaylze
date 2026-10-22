/* Standalone regression checks: cc -std=gnu11 -pthread native/test_tls.c -o /tmp/tnt-test-tls */
#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define REC_MAX_PAYLOAD 4
#define LOGI(...) ((void)0)
static struct { uint8_t kind; uint32_t id, len; char data[128]; } records[1024];
static size_t count;
static void ring_push(uint8_t kind, uint32_t id, const void *p, uint32_t n) {
    assert(count < 1024 && n < sizeof(records[0].data));
    records[count].kind = kind; records[count].id = id; records[count].len = n;
    memcpy(records[count++].data, p, n);
    errno = EIO; /* recording must not change the real SSL errno */
}
#include "tnt_tls.h"
static int result;
static int calls;
static int fake_read(void *s, void *b, int n) {
    (void)s; ++calls;
    if (result > 0) { assert(result <= n); memcpy(b, "abcdef", result); }
    errno = EAGAIN; return result;
}
static int fake_write(void *s, const void *b, int n) {
    (void)s; (void)b; (void)n; ++calls; errno = EAGAIN; return result;
}
static int fake_read_ex(void *s, void *b, size_t n, size_t *done) {
    if (result <= 0) { errno = EAGAIN; return 0; }
    *done = (size_t)tnt_ssl_read(s, b, (int)n); return 1;
}
static int fake_write_ex(void *s, const void *b, size_t n, size_t *done) {
    if (result <= 0) { errno = EAGAIN; return 0; }
    *done = (size_t)tnt_ssl_write(s, b, (int)n); return 1;
}
static void fake_free(void *s) { (void)s; ++calls; errno = ERANGE; }
static int fake_clear(void *s) { (void)s; ++calls; errno = ERANGE; return result; }
static void *fake_resolve(void *lib, const char *name) {
    (void)lib;
    if (!strcmp(name, "SSL_read")) return (void *)fake_read;
    if (!strcmp(name, "SSL_write")) return (void *)fake_write;
    if (!strcmp(name, "SSL_read_ex")) return (void *)fake_read_ex;
    if (!strcmp(name, "SSL_write_ex")) return (void *)fake_write_ex;
    if (!strcmp(name, "SSL_free")) return (void *)fake_free;
    if (!strcmp(name, "SSL_clear")) return (void *)fake_clear;
    return !strcmp(name, "other") ? (void *)fake_resolve : NULL;
}
int main(void) {
    atomic_store(&tls_real_resolve, fake_resolve);
    tls_read_fn rd = (tls_read_fn)tnt_qlibrary_resolve(NULL, "SSL_read");
    tls_write_fn wr = (tls_write_fn)tnt_qlibrary_resolve(NULL, "SSL_write");
    tls_read_ex_fn rx = (tls_read_ex_fn)tnt_qlibrary_resolve(NULL, "SSL_read_ex");
    tls_write_ex_fn wx = (tls_write_ex_fn)tnt_qlibrary_resolve(NULL, "SSL_write_ex");
    tls_free_fn fr = (tls_free_fn)tnt_qlibrary_resolve(NULL, "SSL_free");
    tls_clear_fn cl = (tls_clear_fn)tnt_qlibrary_resolve(NULL, "SSL_clear");
    assert(tnt_qlibrary_resolve(NULL, "other") == (void *)fake_resolve);
    assert(tnt_qlibrary_resolve(NULL, "missing") == NULL);
    assert(tls_wrap_symbol("SSL_read", (void *)fake_write) == (void *)fake_write);
    char buf[32]; void *ssl = (void *)(uintptr_t)123;
    result = -1; assert(rd(ssl, buf, 6) == -1 && errno == EAGAIN && count == 0);
    assert(wr(ssl, "abcdef", 6) == -1 && errno == EAGAIN && count == 0);
    result = 0; assert(rd(ssl, buf, 6) == 0 && count == 0);
    size_t done = 999;
    assert(rx(ssl, buf, 6, &done) == 0 && done == 999 && count == 0);
    assert(wx(ssl, buf, 6, &done) == 0 && count == 0);
    result = 3; int before = calls;
    assert(wr(ssl, "abcdef", 6) == 3 && calls == before + 1 && errno == EAGAIN);
    assert(count == 2 && records[0].kind == 1 && records[0].id & 0x80000000u);
    uint32_t first = records[0].id;
    assert(records[1].kind == 11 && records[1].len == 3 && !memcmp(records[1].data, "abc", 3));
    result = 6; before = calls;
    assert(rx(ssl, buf, 6, &done) == 1 && done == 6 && errno == EAGAIN && calls == before + 1);
    assert(count == 4 && records[2].kind == 12 && records[2].len == 4 && records[3].len == 2);
    assert(!memcmp(records[2].data, "abcd", 4) && !memcmp(records[3].data, "ef", 2));
    result = 2;
    assert(wx(ssl, "abcdef", 6, &done) == 1 && done == 2 && count == 5 && records[4].kind == 11);
    result = 0; assert(cl(ssl) == 0 && count == 5 && errno == ERANGE);
    result = 1; assert(cl(ssl) == 1 && count == 6 && records[5].kind == 4 && errno == ERANGE);
    assert(wr(ssl, "x", 1) == 1 && count == 8 && records[6].id != first);
    fr(ssl); assert(count == 9 && records[8].kind == 4 && errno == ERANGE);
    assert(wr(ssl, "x", 1) == 1 && count == 11 && records[9].id != records[6].id);
    fr(ssl);
    /* Capacity exhaustion drops capture only; the SSL call still runs once. */
    for (uintptr_t i = 1; i <= 128; i++) assert(wr((void *)(i + 1000), "x", 1) == 1);
    size_t full = count; before = calls;
    assert(wr((void *)(uintptr_t)9999, "x", 1) == 1 && count == full && calls == before + 1);
    puts("TLS capture tests passed: Qt resolution, retries, partial I/O, nested ex APIs, errno, lifetime reuse, capacity");
}
