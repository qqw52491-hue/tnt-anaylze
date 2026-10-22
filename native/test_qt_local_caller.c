/* Mimic QtNetwork's imported QLibrary call within an RTLD_LOCAL group. */
void *qt_resolve(void *, const char *) __asm__("_ZN8QLibrary7resolveEPKc");
void *test_qt_lookup(void *library, const char *name) { return qt_resolve(library, name); }
