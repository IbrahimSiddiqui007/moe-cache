// WINDOWS-672: the file-open hint of the model file view (src/platform.h): off by default and then exactly FILE_ATTRIBUTE_NORMAL as before; the switch maps to the
// sequential / random flags. Pure functions, so this runs on Linux.
//   g++ -std=c++17 -Isrc tests/open_hint_test.cpp -o /tmp/open_hint_test && /tmp/open_hint_test
#pragma GCC diagnostic ignored "-Wunused-parameter"      // platform.h has unused parameters on Linux (random, keep); this test only needs the pure functions
#include "platform.h"
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)
int main() {
    CHECK(plat::file_open_hint(nullptr) == 0);
    CHECK(plat::file_open_hint("") == 0);
    CHECK(plat::file_open_hint("0") == 0 && plat::file_open_hint("off") == 0 && plat::file_open_hint("1") == 0);   // only the two names switch it on
    CHECK(plat::file_open_hint("Sequential") == 0);                                                            // exact names, like the other switches' values
    CHECK(plat::file_open_hint("sequential") == 1);
    CHECK(plat::file_open_hint("random") == 2);
    CHECK(plat::file_view_open_flags(0) == 0x80u);                       // the default: the flags the code always used
    CHECK(plat::file_view_open_flags(1) == (0x80u | 0x08000000u));
    CHECK(plat::file_view_open_flags(2) == (0x80u | 0x10000000u));
    CHECK(plat::file_view_open_flags(plat::file_open_hint("bogus")) == 0x80u);
#ifdef _WIN32
    CHECK(plat::file_view_open_flags(1) == (FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN));
    CHECK(plat::file_view_open_flags(2) == (FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS));
#endif
    if (fails) { printf("open hint checks: %d FAILED\n", fails); return 1; }
    printf("open hint checks passed\n");
    return 0;
}
