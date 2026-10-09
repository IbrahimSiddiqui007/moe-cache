/* Only used by `build.sh --portable`. glibc 2.38+ headers redirect sscanf/strtol/strtoull to __isoc23_* symbols, which older
 * distros do not have. These local versions call the classic functions (this file is built without C2x redirects). */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

int __isoc23_sscanf(const char * s, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vsscanf(s, fmt, ap);
    va_end(ap);
    return r;
}
long __isoc23_strtol(const char * s, char ** e, int base) { return strtol(s, e, base); }
unsigned long __isoc23_strtoul(const char * s, char ** e, int base) { return strtoul(s, e, base); }
unsigned long long __isoc23_strtoull(const char * s, char ** e, int base) { return strtoull(s, e, base); }
