// Operating-system layer of moe-cache: everything that differs between Linux and Windows lives here.
//  Linux  : mmap / madvise / O_DIRECT pread / cgroups
//  Windows: VirtualAlloc (reserve + commit) / FILE_FLAG_NO_BUFFERING overlapped reads / job object + GlobalMemoryStatusEx
// A reserved range has no memory behind it. "commit" gives pages to the range, "decommit" gives the physical pages back.
// On Linux commit is a no-op (pages appear when touched) and decommit is madvise(DONTNEED).
#pragma once
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <psapi.h>
#  include <direct.h>
#  define MOE_CACHE_EXPORT __declspec(dllexport)
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define MOE_CACHE_EXPORT __attribute__((visibility("default")))
#endif

namespace plat {

constexpr bool is_windows =
#ifdef _WIN32
    true;
#else
    false;
#endif

struct file_t {
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
    bool ok() const { return h != INVALID_HANDLE_VALUE; }
#else
    int fd = -1;
    bool ok() const { return fd >= 0; }
#endif
};

inline std::string last_error() {
#ifdef _WIN32
    return "Windows error " + std::to_string((unsigned long) GetLastError());
#else
    return strerror(errno);
#endif
}

// ---- files ----
// direct = bypass the OS page cache (O_DIRECT / FILE_FLAG_NO_BUFFERING). Offsets, lengths and buffers must then be 4096-aligned.
inline file_t file_open(const char * path, bool direct) {
    file_t f;
#ifdef _WIN32
    int n = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
    std::wstring w((size_t) std::max(n, 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path, -1, &w[0], n);
    f.h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                      FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : 0), nullptr);
#else
    f.fd = open(path, O_RDONLY | (direct ? O_DIRECT : 0));
#endif
    return f;
}

// read len bytes at offset off; returns the number of bytes read, 0 at end of file, -1 on error (retry is up to the caller)
inline ptrdiff_t file_pread(const file_t & f, void * buf, size_t len, uint64_t off) {
#ifdef _WIN32
    static thread_local HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset = (DWORD) (off & 0xffffffffu);
    ov.OffsetHigh = (DWORD) (off >> 32);
    ov.hEvent = ev;
    ResetEvent(ev);
    DWORD got = 0;
    const DWORD want = (DWORD) std::min<size_t>(len, 0x40000000);
    if (!ReadFile(f.h, buf, want, nullptr, &ov)) {
        const DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF) {
            return 0;
        }
        if (e != ERROR_IO_PENDING) {
            return -1;
        }
        if (!GetOverlappedResult(f.h, &ov, &got, TRUE)) {
            return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
        }
        return (ptrdiff_t) got;
    }
    GetOverlappedResult(f.h, &ov, &got, TRUE);
    return (ptrdiff_t) got;
#else
    for (;;) {
        const ssize_t r = pread(f.fd, buf, len, (off_t) off);
        if (r < 0 && errno == EINTR) {
            continue;
        }
        return (ptrdiff_t) r;
    }
#endif
}

// ---- memory ----
inline void * mem_reserve(size_t size) {
#ifdef _WIN32
    return VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE);
#else
    void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}
inline void mem_release(void * p, size_t size) {
#ifdef _WIN32
    (void) size;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, size);
#endif
}
// p and size are rounded outwards to whole pages; committing an already committed page keeps its content
inline void mem_commit(void * p, size_t size) {
#ifdef _WIN32
    const uintptr_t a = (uintptr_t) p / 4096 * 4096, b = ((uintptr_t) p + size + 4095) / 4096 * 4096;
    VirtualAlloc((void *) a, b - a, MEM_COMMIT, PAGE_READWRITE);
#else
    (void) p; (void) size;
#endif
}
// p and size must be page aligned
inline void mem_decommit(void * p, size_t size) {
#ifdef _WIN32
    VirtualFree(p, size, MEM_DECOMMIT);
#else
    madvise(p, size, MADV_DONTNEED);
#endif
}
inline void mem_hugepage(void * p, size_t size, bool on) {
#if defined(__linux__)
    madvise(p, size, on ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
#else
    (void) p; (void) size; (void) on;
#endif
}
inline void * aligned_alloc_page(size_t size) {
#ifdef _WIN32
    return VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void * p = nullptr;
    return posix_memalign(&p, 4096, size) == 0 ? p : nullptr;
#endif
}
inline void aligned_free_page(void * p) {
#ifdef _WIN32
    if (p) { VirtualFree(p, 0, MEM_RELEASE); }
#else
    free(p);
#endif
}
inline void sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD) ms);
#else
    usleep((useconds_t) ms * 1000);
#endif
}

// ---- memory accounting ----
// memory this process may use: Linux = smallest cgroup memory.max on the path, else MemAvailable; Windows = job limit, else available physical memory
inline size_t mem_limit() {
    size_t lim = (size_t) -1;
#ifdef _WIN32
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION ji;
    memset(&ji, 0, sizeof(ji));
    if (QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &ji, sizeof(ji), nullptr)) {
        if (ji.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_JOB_MEMORY) { lim = std::min<size_t>(lim, ji.JobMemoryLimit); }
        if (ji.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) { lim = std::min<size_t>(lim, ji.ProcessMemoryLimit); }
    }
    if (lim == (size_t) -1) {
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) { lim = (size_t) ms.ullAvailPhys; }
    }
#else
    std::string path;
    if (FILE * f = fopen("/proc/self/cgroup", "r")) {
        char line[1024];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "0::", 3) == 0) {
                path = line + 3;
                while (!path.empty() && (path.back() == '\n' || path.back() == '\r')) { path.pop_back(); }
            }
        }
        fclose(f);
    }
    std::string p = path;
    for (;;) {
        const std::string fn = "/sys/fs/cgroup" + p + "/memory.max";
        if (FILE * g = fopen(fn.c_str(), "r")) {
            char b[64] = { 0 };
            if (fgets(b, sizeof(b), g) && strncmp(b, "max", 3) != 0) {
                lim = std::min<size_t>(lim, (size_t) strtoull(b, nullptr, 10));
            }
            fclose(g);
        }
        if (p.empty() || p == "/") {
            break;
        }
        const size_t sl = p.rfind('/');
        p = sl == std::string::npos ? std::string() : p.substr(0, sl);
    }
    if (lim == (size_t) -1) {
        if (FILE * f = fopen("/proc/meminfo", "r")) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                unsigned long long kb = 0;
                if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                    lim = (size_t) kb * 1024;
                }
            }
            fclose(f);
        }
    }
#endif
    return lim;
}

// private memory of this process (Linux RssAnon; Windows PrivateUsage, which counts committed pages: exactly the loaded experts)
inline size_t rss_private() {
    size_t r = 0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    memset(&pmc, 0, sizeof(pmc));
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc))) {
        r = (size_t) pmc.PrivateUsage;
        if (r == 0) {
            r = (size_t) std::max<SIZE_T>(pmc.PagefileUsage, pmc.WorkingSetSize);   // some environments (Wine) leave PrivateUsage empty
        }
    }
#else
    if (FILE * f = fopen("/proc/self/status", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            unsigned long long kb = 0;
            if (sscanf(line, "RssAnon: %llu kB", &kb) == 1) {
                r = (size_t) kb * 1024;
                break;
            }
        }
        fclose(f);
    }
#endif
    return r;
}

// ---- small file helpers ----
inline void make_dir(const char * path) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0755);
#endif
}
// rename tmp over dst (replaces an existing file on both systems)
inline void replace_file(const char * tmp, const char * dst) {
#ifdef _WIN32
    MoveFileExA(tmp, dst, MOVEFILE_REPLACE_EXISTING);
#else
    rename(tmp, dst);
#endif
}

}  // namespace plat
