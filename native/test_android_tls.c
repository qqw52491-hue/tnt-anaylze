/* Independent process, actual game-bundled OpenSSL, TLS 1.3 over memory BIOs.
 * No game server, game process, account or battle data is used.
 * The QLibrary fixture is a dependency in a locally-loaded DSO group, matching
 * Android Qt's loader scope, rather than the executable's global group.
 */
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <android/dlext.h>
typedef void SSL;
typedef void SSL_CTX;
typedef void BIO;
#define API(ret, name, args) ret (*p_##name) args = (ret (*) args)dlsym(lib, #name); assert(p_##name)
int main(int argc, char **argv) {
    assert(argc == 7);
    typedef struct android_namespace_t *(*create_ns_fn)(const char *, const char *, const char *,
                                                       unsigned long long, const char *,
                                                       struct android_namespace_t *, const void *);
    create_ns_fn create_ns = (void *)dlsym(RTLD_DEFAULT, "__loader_android_create_namespace");
    assert(create_ns);
    /* SHARED copies the existing default libraries (including the preload),
     * then loads the Qt fixtures only into this new classloader-like scope. */
    struct android_namespace_t *ns = create_ns("tnt-qt-test", argv[6], argv[6], 2, NULL, NULL, (void *)main);
    assert(ns);
    android_dlextinfo info = { .flags = ANDROID_DLEXT_USE_NAMESPACE, .library_namespace = ns };
    void *qt = android_dlopen_ext(argv[5], RTLD_NOW | RTLD_LOCAL, &info);
    if (!qt) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    void *(*qt_resolve)(void *, const char *) = (void *)dlsym(qt, "test_qt_lookup");
    assert(qt_resolve);
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    API(void *, TLS_method, (void));
    API(SSL_CTX *, SSL_CTX_new, (const void *));
    API(int, SSL_CTX_use_certificate_file, (SSL_CTX *, const char *, int));
    API(int, SSL_CTX_use_PrivateKey_file, (SSL_CTX *, const char *, int));
    API(SSL *, SSL_new, (SSL_CTX *));
    API(BIO *, BIO_new, (const void *));
    API(void *, BIO_s_mem, (void));
    API(void, SSL_set_bio, (SSL *, BIO *, BIO *));
    API(int, SSL_connect, (SSL *));
    API(int, SSL_accept, (SSL *));
    API(int, SSL_get_error, (SSL *, int));
    API(int, SSL_version, (const SSL *));
    API(int, BIO_read, (BIO *, void *, int));
    API(int, BIO_write, (BIO *, const void *, int));
    API(void, SSL_CTX_free, (SSL_CTX *));
    int (*wr)(SSL *, const void *, int) = (void *)qt_resolve(lib, "SSL_write");
    int (*rd)(SSL *, void *, int) = (void *)qt_resolve(lib, "SSL_read");
    void (*fr)(SSL *) = (void *)qt_resolve(lib, "SSL_free");
    assert(wr && rd && fr);
    /* The shim normally returns dlsym's pointer. Verify preload replaced it. */
    assert((void *)wr != dlsym(lib, "SSL_write"));
    assert((void *)rd != dlsym(lib, "SSL_read"));
    SSL_CTX *cc = p_SSL_CTX_new(p_TLS_method());
    SSL_CTX *sc = p_SSL_CTX_new(p_TLS_method());
    assert(cc && sc);
    assert(p_SSL_CTX_use_certificate_file(sc, argv[2], 1) == 1);
    assert(p_SSL_CTX_use_PrivateKey_file(sc, argv[3], 1) == 1);
    SSL *client = p_SSL_new(cc), *server = p_SSL_new(sc);
    assert(client && server);
    BIO *cr = p_BIO_new(p_BIO_s_mem()), *cw = p_BIO_new(p_BIO_s_mem());
    BIO *sr = p_BIO_new(p_BIO_s_mem()), *sw = p_BIO_new(p_BIO_s_mem());
    assert(cr && cw && sr && sw);
    p_SSL_set_bio(client, cr, cw); p_SSL_set_bio(server, sr, sw);
    char tmp[32768];
#define PUMP(from, to) do { int n; while ((n = p_BIO_read(from, tmp, sizeof(tmp))) > 0) assert(p_BIO_write(to, tmp, n) == n); } while (0)
    int cdone = 0, sdone = 0;
    for (int i = 0; i < 100 && (!cdone || !sdone); i++) {
        if (!cdone) { int r = p_SSL_connect(client); if (r == 1) cdone = 1; else assert(p_SSL_get_error(client, r) == 2); }
        PUMP(cw, sr);
        if (!sdone) { int r = p_SSL_accept(server); if (r == 1) sdone = 1; else assert(p_SSL_get_error(server, r) == 2); }
        PUMP(sw, cr);
    }
    assert(cdone && sdone && p_SSL_version(client) == 0x0304);
    FILE *f = fopen(argv[4], "rb"); assert(f);
    unsigned char frame[4096]; size_t len = fread(frame, 1, sizeof(frame), f); fclose(f);
    assert(len > 0 && len < sizeof(frame));
    assert(wr(server, frame, (int)len) == (int)len); PUMP(sw, cr);
    int n = rd(client, tmp, sizeof(tmp));
    assert(n == (int)len && !memcmp(tmp, frame, len));
    fr(client); fr(server); p_SSL_CTX_free(cc); p_SSL_CTX_free(sc);
    /* Allow the existing asynchronous tunnel writer to flush this test. */
    sleep(2);
    puts("Android integration passed: dynamic QLibrary interception + real TLS 1.3 memory BIO plaintext roundtrip");
    return 0;
}
