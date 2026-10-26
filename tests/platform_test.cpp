// Tests the operating-system layer (src/platform.h): reserve/commit/decommit, unbuffered reads, limits, file helpers.
//   g++ -std=c++17 -Isrc tests/platform_test.cpp -o /tmp/platform_test && /tmp/platform_test        (Linux)
//   x86_64-w64-mingw32-g++ -std=c++17 -Isrc tests/platform_test.cpp -o platform_test.exe             (Windows, or run under wine)
#include "platform.h"
#include <vector>
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } else { printf("ok   %s\n", #c); } } while (0)

int main(int argc, char ** argv) {
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

    // a commit that runs past the end of the reservation must still commit the pages inside it (Windows: one VirtualAlloc call fails as a whole;
    // the last expert of a buffer plus the extra page did exactly that and the next write crashed, RESULTS 50)
    char * q = (char *) plat::mem_reserve(64 * 1024);
    CHECK(q != nullptr);
    plat::mem_commit(q + 60 * 1024, 8192);
    q[60 * 1024] = 5; q[64 * 1024 - 1] = 6;
    CHECK(q[60 * 1024] == 5 && q[64 * 1024 - 1] == 6);
    plat::mem_release(q, 64 * 1024);

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

    // non-ASCII paths (C:\Users\José\...): the environment gives UTF-8, and every file helper takes it (Windows converts to UTF-16 at the call)
    const char * jose = "moe_cache_pt_dir/Jos\xc3\xa9 \xe6\xa8\xa1\xe5\x9e\x8b";        // "José 模型": not representable in one ANSI code page
#ifdef _WIN32
    SetEnvironmentVariableW(L"MOE_CACHE_PT_PATH", plat::widen(jose).c_str());
#else
    setenv("MOE_CACHE_PT_PATH", jose, 1);
#endif
    std::string ev;
    CHECK(plat::env_utf8("MOE_CACHE_PT_PATH", ev) && ev == jose);
    CHECK(!plat::env_utf8("MOE_CACHE_PT_UNSET_VARIABLE", ev));
    plat::make_dir(ev.c_str());
    const std::string jf = ev + "/p", jt = ev + "/p.tmp";
    { FILE * f = plat::fopen_utf8(jt.c_str(), "wb"); CHECK(f != nullptr); if (f) { fputs("utf8", f); fclose(f); } }
    plat::replace_file(jt.c_str(), jf.c_str());
    { char b[8] = { 0 }; FILE * f = plat::fopen_utf8(jf.c_str(), "rb"); CHECK(f != nullptr); if (f) { size_t n = fread(b, 1, 7, f); (void) n; fclose(f); } CHECK(strcmp(b, "utf8") == 0); }
    { plat::file_t g = plat::file_open(jf.c_str(), false); CHECK(g.ok()); char b[8] = { 0 }; CHECK(g.ok() && plat::file_pread(g, b, 4, 0) == 4 && memcmp(b, "utf8", 4) == 0);
#ifdef _WIN32
      if (g.ok()) { CloseHandle(g.h); } }
    _wremove(plat::widen(jf.c_str()).c_str()); _wrmdir(plat::widen(ev.c_str()).c_str());
#else
      if (g.ok()) { close(g.fd); } }
    remove(jf.c_str()); rmdir(ev.c_str());
#endif

    // the process's own command line (the plugin looks for --no-repack and --cache-ram in llama-server's)
    const std::vector<std::string> cl = plat::command_line();
    CHECK(cl.size() == (size_t) argc && cl[0] == argv[0] && (argc < 2 || cl[1] == argv[1]));
    printf(fails ? "FAILED: %d\n" : "ALL OK\n", fails);
    return fails ? 1 : 0;
}
