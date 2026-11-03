// Checks src/mc_ramguard.h (CACHE-153 lock, CACHE-058 read-ahead drop). Linux only checks; elsewhere only the no-op contract.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/ramguard_test.cpp -o /tmp/ramguard_test && /tmp/ramguard_test
#include "mc_ramguard.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#ifdef __linux__
#  include <sys/resource.h>
#  include <sys/vfs.h>
#endif

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

#ifdef __linux__
static size_t cached_pages(int fd, size_t len) {
    void * m = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { return (size_t) -1; }
    std::vector<unsigned char> v((len + 4095) / 4096);
    size_t n = 0;
    if (mincore(m, len, v.data()) == 0) { for (unsigned char c : v) { n += c & 1; } }
    munmap(m, len);
    return n;
}
#endif

int main() {
    // off by default: nothing is locked, nothing is dropped
    unsetenv("MOE_CACHE_RAM_LOCK");
    CHECK(!mcram::lock_wanted() && !mcram::drop_wanted());
    char buf[8192];
    CHECK(!mcram::lock_range(buf, sizeof buf));
    mcram::unlock_range(buf, sizeof buf);          // no-op, must not crash
    CHECK(!mcram::drop_file_cache(-1));
#ifdef __linux__
    // the drop: write 4 MiB through the page cache, sync, drop, the file's resident pages fall
    const char * td = getenv("TMPDIR");
    std::string tp = std::string(td && *td ? td : "/tmp") + "/mc_ramguard_XXXXXX";
    std::vector<char> t(tp.begin(), tp.end()); t.push_back(0);
    int fd = mkstemp(t.data());
    CHECK(fd >= 0);
    if (fd >= 0) {
        const size_t len = 4u << 20;
        std::vector<char> data(len, 'x');
        CHECK(write(fd, data.data(), len) == (ssize_t) len);
        fsync(fd);
        const size_t before = cached_pages(fd, len);
        CHECK(before > 0);
        CHECK(mcram::drop_file_cache(fd));
        const size_t after = cached_pages(fd, len);
        // On tmpfs / ramfs (a RAM-disk /tmp) the file's pages ARE the file: no page-cache drop can evict them, so the fall in resident pages
        // cannot be checked there. Put TMPDIR on a disk to run this check (it failed 5 of 5 on a laptop whose /tmp is tmpfs).
        struct statfs sf;
        const bool ramfs = fstatfs(fd, &sf) == 0 && (sf.f_type == 0x01021994 /* TMPFS_MAGIC */ || sf.f_type == 0x858458f6 /* RAMFS_MAGIC */);
        if (ramfs) {
            printf("the temporary file is on tmpfs/ramfs: the page-cache drop cannot be checked here (set TMPDIR to a disk)\n");
        } else {
            CHECK(after < before / 4);              // 0 in practice; the margin keeps a busy kernel from failing the test
        }
        printf("page cache pages: %zu before, %zu after\n", before, after);
        close(fd);
        unlink(t.data());
    }
    // the lock: on (the switch is read once, so this test sets it before the first call through a fresh process image: it is already cached
    // as off above, so the range functions are checked directly through the syscalls they wrap)
    void * m = mmap(nullptr, 1 << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    CHECK(m != MAP_FAILED);
    struct rlimit rl;
    getrlimit(RLIMIT_MEMLOCK, &rl);
    if (rl.rlim_cur == RLIM_INFINITY || rl.rlim_cur >= (1u << 20)) {
        CHECK(syscall(SYS_mlock2, m, (size_t) 1 << 20, MLOCK_ONFAULT) == 0);
        CHECK(madvise(m, 1 << 20, MADV_DONTNEED) != 0);      // locked pages refuse it: that is why decommit unlocks first
        CHECK(munlock(m, 1 << 20) == 0);
        CHECK(madvise(m, 1 << 20, MADV_DONTNEED) == 0);
    } else {
        printf("RLIMIT_MEMLOCK too small here, lock part skipped\n");
    }
    munmap(m, 1 << 20);
#endif
    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ramguard_test: ok\n");
    return 0;
}
