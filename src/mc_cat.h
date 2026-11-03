// Appending formatted text to a fixed buffer without ever writing past it (QUALITY-027). Header-only, no ggml: tests/cat_test.cpp includes it.
// snprintf returns the length it WOULD have written; adding that to the fill position and passing cap - n to the next call let a long line wrap the size
// around. The Windows crash report (a signal-context handler: no allocation, no stdio) builds its text this way.
#pragma once
#include <cstdarg>
#include <cstddef>
#include <cstdio>

namespace mccat {

// Appends at buf + n (n = bytes used so far, 0 <= n < cap), returns the new fill, never more than cap - 1; the text is cut when it does not fit.
inline int cat(char * buf, size_t cap, int n, const char * fmt, ...) {
    if (n < 0) {
        n = 0;
    }
    if ((size_t) n >= cap) {
        return (int) cap - 1;
    }
    va_list ap;
    va_start(ap, fmt);
    const int r = vsnprintf(buf + n, cap - (size_t) n, fmt, ap);
    va_end(ap);
    if (r < 0) {
        return n;
    }
    return (size_t) (n + r) >= cap ? (int) cap - 1 : n + r;
}

}  // namespace mccat
