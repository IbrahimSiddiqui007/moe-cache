// Tests the operating-system layer (src/platform.h): reserve/commit/decommit, unbuffered reads, limits, file helpers.
//   g++ -std=c++17 -Isrc tests/platform_test.cpp -o /tmp/platform_test && /tmp/platform_test        (Linux)
//   x86_64-w64-mingw32-g++ -std=c++17 -Isrc tests/platform_test.cpp -o platform_test.exe             (Windows, or run under wine)
#include "platform.h"
#include <vector>
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } else { printf("ok   %s\n", #c); } } while (0)

int main() {
    const size_t MB = 1 << 20;
    // reserve, commit, write, decommit, commit again (content must be gone or zero, never an old value after decommit on Windows)
    char * p = (char *) plat::mem_reserve(64 * MB);
    CHECK(p != nullptr);
    plat::mem_commit(p + 4096, 8192);
    memset(p + 4096, 0xAB, 8192);
    CHECK((unsigned char) p[4096] == 0xAB && (unsigned char) p[4096 + 8191] == 0xAB);
    plat::mem_decommit(p + 4096, 8192);
    plat::mem_commit(p + 4096, 8192);
    CHECK(p[4096] == 0 && p[4096 + 8191] == 0);
    plat::mem_commit(p + 100, 10);                         // unaligned range is rounded outwards
    p[100] = 7; plat::mem_commit(p + 100, 10); CHECK(p[100] == 7);   // committing again keeps the content
    plat::mem_release(p, 64 * MB);

    // aligned buffer + unbuffered read at an offset
    std::string path = "moe_cache_platform_test.bin";
    { FILE * f = fopen(path.c_str(), "wb"); std::vector<unsigned char> d(MB); for (size_t i = 0; i < d.size(); i++) d[i] = (unsigned char) (i * 31 + 7); fwrite(d.data(), 1, d.size(), f); fclose(f); }
    plat::file_t fd = plat::file_open(path.c_str(), true);
    if (!fd.ok()) { printf("note: direct open not supported on this file system, using buffered\n"); fd = plat::file_open(path.c_str(), false); }
    CHECK(fd.ok());
    char * buf = (char *) plat::aligned_alloc_page(16384);
    CHECK(buf != nullptr);
    ptrdiff_t got = plat::file_pread(fd, buf, 8192, 4096 * 3);
    CHECK(got == 8192);
    bool same = true; for (size_t i = 0; i < 8192; i++) if ((unsigned char) buf[i] != (unsigned char) ((4096 * 3 + i) * 31 + 7)) same = false;
    CHECK(same);
    got = plat::file_pread(fd, buf, 8192, MB - 4096);      // crosses the end of the file: must return the short count, not an error
    CHECK(got == 4096);
    got = plat::file_pread(fd, buf, 4096, MB);             // at the end: 0
    CHECK(got == 0);
    plat::aligned_free_page(buf);

    // limits and accounting
    CHECK(plat::mem_limit() > 100 * MB);
    CHECK(plat::rss_private() > 0);
    // directories and replacing a file
    plat::make_dir("moe_cache_pt_dir");
    { FILE * f = fopen("moe_cache_pt_dir/a.tmp", "wb"); fputs("new", f); fclose(f); }
    { FILE * f = fopen("moe_cache_pt_dir/a", "wb"); fputs("old", f); fclose(f); }
    plat::replace_file("moe_cache_pt_dir/a.tmp", "moe_cache_pt_dir/a");
    { char b[8] = { 0 }; FILE * f = fopen("moe_cache_pt_dir/a", "rb"); size_t n = fread(b, 1, 7, f); (void) n; fclose(f); CHECK(strcmp(b, "new") == 0); }
    remove("moe_cache_pt_dir/a"); remove(path.c_str());
    printf(fails ? "FAILED: %d\n" : "ALL OK\n", fails);
    return fails ? 1 : 0;
}
