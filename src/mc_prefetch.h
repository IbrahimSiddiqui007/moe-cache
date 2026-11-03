// R-02 (MOE_CACHE_EXEC_PREFETCH=N, default off): software prefetch of the weight rows the CPU executor reads a few rows ahead.
// The rows of one expert are contiguous, but the hardware stream prefetcher restarts at every 4 KiB page and a row is 1.5-4 KB, so the
// dot product stalls on the first lines of each new page. This asks for the lines of row r+N while row r is computed. It reads nothing and
// changes no arithmetic: output cannot change. Header-only, no ggml: tests/prefetch_test.cpp includes it.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace mcpf {

constexpr int kMaxRows = 16;      // distance cap (rows)
constexpr size_t kLine = 64;
constexpr size_t kMaxLines = 64;  // lines asked per row (4 KB); a longer row is covered at its start and by the hardware prefetcher after that

// "N" -> rows ahead, 0 = off. Unset, empty, "off", "false", "no", "0" and anything that is not a positive number = 0; "on" / "true" / "yes" = 4.
inline int parse_rows(const char * v) {
    if (!v || !*v) {
        return 0;
    }
    if (!strcmp(v, "on") || !strcmp(v, "true") || !strcmp(v, "yes")) {
        return 4;
    }
    char * end = nullptr;
    const long n = strtol(v, &end, 10);
    if (end == v || *end != '\0' || n <= 0) {
        return 0;
    }
    return n > kMaxRows ? kMaxRows : (int) n;
}

// number of cache lines to ask for a row of `bytes` bytes
inline size_t lines_for(size_t bytes) {
    const size_t n = (bytes + kLine - 1) / kLine;
    return n > kMaxLines ? kMaxLines : n;
}

inline void touch(const char * p) {
#if defined(__GNUC__)
    __builtin_prefetch(p, 0, 3);
#elif defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    _mm_prefetch(p, _MM_HINT_T0);
#else
    (void) p;
#endif
}

// ask for row r + ahead of a weight matrix (base, row stride nb01) that has `rows` rows; nothing past the last row
inline void rows_ahead(const char * base, size_t nb01, int64_t r, int ahead, int64_t rows) {
    const int64_t t = r + ahead;
    if (ahead <= 0 || t >= rows) {
        return;
    }
    const char * p = base + (size_t) t * nb01;
    const size_t n = lines_for(nb01);
    for (size_t i = 0; i < n; i++) {
        touch(p + i * kLine);
    }
}

// how many lines rows_ahead asks (for the test)
inline size_t lines_asked(size_t nb01, int64_t r, int ahead, int64_t rows) {
    return (ahead <= 0 || r + ahead >= rows) ? 0 : lines_for(nb01);
}

}  // namespace mcpf
