// CACHE-685, model-free: file_view_load on a mapped file returns true for readable pages and false (no SIGBUS, process survives) when the file was cut short
// under the mapping. The sigbus_test next to it checks that an unguarded touch still prints the message and dies.
// Run: g++ -std=c++17 -Wall -Werror -Wno-unused-function -Isrc tests/file_view_load_test.cpp -o /tmp/fvl_test && /tmp/fvl_test
#include "platform.h"

static int fails = 0;
static void check(bool ok, const char * what) {
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    fails += ok ? 0 : 1;
}

int main() {
#ifndef __linux__
    printf("file_view_load_test: skipped (Linux only)\n");
    return 0;
#else
    char path[] = "/var/tmp/fvl_test_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        strcpy(path, "/tmp/fvl_test_XXXXXX");
        fd = mkstemp(path);
    }
    if (fd < 0 || ftruncate(fd, 8 * 4096) != 0) {
        printf("file_view_load_test: FAIL (temp file)\n");
        return 1;
    }
    close(fd);
    size_t n = 0;
    const char * v = plat::file_view_map(path, &n);
    check(v && n == 8 * 4096, "the file maps");
    check(plat::file_view_load(v + 100, 3 * 4096), "a readable range loads: true");
    check(plat::file_view_load(v, 0), "an empty range: true");
    check(truncate(path, 2 * 4096) == 0, "the file is cut to 2 pages under the mapping");
    check(plat::file_view_load(v, 4096), "pages still inside the file load: true");
    check(!plat::file_view_load(v + 4 * 4096 + 7, 4096), "a range past the new end: false, no crash");
    check(!plat::file_view_load(v + 1 * 4096, 4 * 4096), "a range that crosses the new end: false, no crash");
    check(plat::file_view_load(v, 4096), "the process still works afterwards: true");
    unlink(path);
    printf("file_view_load_test: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
#endif
}
