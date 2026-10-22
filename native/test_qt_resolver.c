/* Android integration fixture: the same imported ABI used by QtNetwork. */
#include <dlfcn.h>
void *qt_resolve(void *, const char *) __asm__("_ZN8QLibrary7resolveEPKc");
void *qt_resolve(void *library, const char *name) { return dlsym(library, name); }
