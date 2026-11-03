// CACHE-153 / CACHE-058: keep the RAM cache out of swap, drop the file read-ahead after the load. Header-only, no ggml (tests/ramguard_test.cpp).
//
//   MOE_CACHE_RAM_LOCK=1        the cache's private memory is locked as it is touched (mlock2 MLOCK_ONFAULT), so the kernel cannot swap it out.
//                               Needs RLIMIT_MEMLOCK room (ulimit -l / LimitMEMLOCK=infinity); when the limit is hit the guard switches itself
//                               off with one line and the cache goes on unlocked. Eviction unlocks before it gives the pages back.
//   MOE_CACHE_DROP_READAHEAD=1  after the startup fill, the model files' page cache copy is dropped (posix_fadvise DONTNEED) when the cache holds
//                               its own copy in private memory. Not done in file-view mode: there the page cache IS the cache.
//
// Both default off and change no byte of any result. Windows: no-op (the file-view cache keeps its pages in the working set already).
#pragma once
#include "mc_env.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#ifdef __linux__
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <unistd.h>
#  include <cerrno>
#  ifndef MLOCK_ONFAULT
#    define MLOCK_ONFAULT 1
#  endif
#  include <sys/syscall.h>
#endif

namespace mcram {

inline bool lock_wanted() {
    static const bool on = mcenv::env_on("MOE_CACHE_RAM_LOCK");
    return on;
}
inline bool drop_wanted() {
    static const bool on = mcenv::env_on("MOE_CACHE_DROP_READAHEAD");
    return on;
}
inline std::atomic<bool> & lock_failed() {
    static std::atomic<bool> f{false};
    return f;
}
inline std::atomic<unsigned long long> & locked_bytes() {
    static std::atomic<unsigned long long> n{0};
    return n;
}

// lock [p, p + n) (rounded outwards to pages) without touching it; false when not locked (switched off, limit reached, no Linux)
inline bool lock_range(void * p, size_t n) {
#ifdef __linux__
    if (!lock_wanted() || lock_failed().load(std::memory_order_relaxed) || n == 0) {
        return false;
    }
    const uintptr_t a = (uintptr_t) p / 4096 * 4096, b = ((uintptr_t) p + n + 4095) / 4096 * 4096;
    long r = -1;
#  ifdef SYS_mlock2
    r = syscall(SYS_mlock2, (void *) a, (size_t) (b - a), MLOCK_ONFAULT);
#  endif
    if (r != 0) {
        if (!lock_failed().exchange(true)) {
            fprintf(stderr, "moe-cache: MOE_CACHE_RAM_LOCK: the memory lock was refused (%s); the cache goes on unlocked. "
                    "Raise it with `ulimit -l unlimited` or LimitMEMLOCK=infinity\n", strerror(errno));
        }
        return false;
    }
    locked_bytes() += (unsigned long long) (b - a);
    return true;
#else
    (void) p; (void) n;
    return false;
#endif
}
// undo lock_range before the pages are given back (MADV_DONTNEED fails with EINVAL on locked pages). Harmless on pages that are not locked.
inline void unlock_range(void * p, size_t n) {
#ifdef __linux__
    if (!lock_wanted() || locked_bytes().load(std::memory_order_relaxed) == 0 || n == 0) {
        return;
    }
    munlock(p, n);
#else
    (void) p; (void) n;
#endif
}
// drop the clean page cache of a whole file. Pages mapped by some process stay (the kernel skips them), so this never harms a running mapping.
inline bool drop_file_cache(int fd) {
#ifdef __linux__
    if (fd < 0) {
        return false;
    }
    return posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0;
#else
    (void) fd;
    return false;
#endif
}

}  // namespace mcram
