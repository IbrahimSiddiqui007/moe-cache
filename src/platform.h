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
#include <vector>
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

// ---- text: paths are UTF-8 everywhere in the plugin (ggml's gguf reader takes UTF-8 too); Windows converts at the system call ----
#ifdef _WIN32
inline std::wstring widen(const char * s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    std::wstring w((size_t) std::max(n, 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
    w.resize(wcslen(w.c_str()));
    return w;
}
inline std::string narrow(const wchar_t * w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s((size_t) std::max(n, 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    s.resize(strlen(s.c_str()));
    return s;
}
#endif

// environment variable as UTF-8, false if unset. Use it for paths: Windows getenv() gives the ANSI code page, so MOE_CACHE_GGUF=C:\Users\José\...
// reached the UTF-8 file functions mangled and the model did not open. Linux: getenv.
inline bool env_utf8(const char * name, std::string & out) {
#ifdef _WIN32
    const std::wstring wn = widen(name);
    const DWORD n = GetEnvironmentVariableW(wn.c_str(), nullptr, 0);
    if (n == 0) {
        return false;
    }
    std::vector<wchar_t> b(n + 1, L'\0');
    GetEnvironmentVariableW(wn.c_str(), b.data(), n + 1);
    out = narrow(b.data());
    return true;
#else
    const char * v = getenv(name);
    if (!v) {
        return false;
    }
    out = v;
    return true;
#endif
}

// this process's command line (UTF-8 words): Linux /proc/self/cmdline; Windows GetCommandLineW split at blanks, double quotes group (as the C runtime
// does for argv; backslash escapes are not needed for the flags the plugin looks for); empty elsewhere
inline std::vector<std::string> command_line() {
    std::vector<std::string> out;
#if defined(__linux__)
    if (FILE * cf = fopen("/proc/self/cmdline", "rb")) {
        std::string cmd;
        char buf[4096];
        size_t nr;
        while ((nr = fread(buf, 1, sizeof(buf), cf)) > 0) {
            cmd.append(buf, nr);
        }
        fclose(cf);
        for (size_t b = 0; b < cmd.size();) {
            const size_t e = cmd.find('\0', b);
            out.push_back(cmd.substr(b, e == std::string::npos ? std::string::npos : e - b));
            if (e == std::string::npos) {
                break;
            }
            b = e + 1;
        }
    }
#elif defined(_WIN32)
    const wchar_t * c = GetCommandLineW();
    while (c && *c) {
        while (*c == L' ' || *c == L'\t') { c++; }
        if (!*c) {
            break;
        }
        std::wstring a;
        bool q = false;
        for (; *c && (q || (*c != L' ' && *c != L'\t')); c++) {
            if (*c == L'"') { q = !q; } else { a += *c; }
        }
        out.push_back(narrow(a.c_str()));
    }
#endif
    return out;
}

// fopen with a UTF-8 path (Windows: _wfopen)
inline FILE * fopen_utf8(const char * path, const char * mode) {
#ifdef _WIN32
    return _wfopen(widen(path).c_str(), widen(mode).c_str());
#else
    return fopen(path, mode);
#endif
}

// ---- files ----
// direct = bypass the OS page cache (O_DIRECT / FILE_FLAG_NO_BUFFERING). Offsets, lengths and buffers must then be 4096-aligned.
// random = Windows, buffered: FILE_FLAG_RANDOM_ACCESS (no read-ahead around each read); ignored elsewhere
inline file_t file_open(const char * path, bool direct, bool random = false) {
    file_t f;
#ifdef _WIN32
    f.h = CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                      FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : 0) | (!direct && random ? FILE_FLAG_RANDOM_ACCESS : 0), nullptr);
#else
    f.fd = open(path, O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
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
// MOE_CACHE_GUARD=1 (tests only): evicted pages become PROT_NONE, so a read the plugin did not intercept crashes instead of seeing zeros.
// Each protected range is its own mapping, so keep it to small models (vm.max_map_count, default 65530). Windows decommit already faults.
inline bool mem_guard() {
    static const bool on = [] { const char * e = getenv("MOE_CACHE_GUARD"); return e && strcmp(e, "0") != 0; }();
    return on;
}
// p and size are rounded outwards to whole pages; committing an already committed page keeps its content
inline void mem_commit(void * p, size_t size) {
#ifdef _WIN32
    const uintptr_t a = (uintptr_t) p / 4096 * 4096, b = ((uintptr_t) p + size + 4095) / 4096 * 4096;
    if (!VirtualAlloc((void *) a, b - a, MEM_COMMIT, PAGE_READWRITE)) {
        // one call fails as a whole when the range runs past the end of its reservation (e.g. the extra page after the last expert of a buffer):
        // then nothing was committed and the next write faulted (Windows CI, RESULTS 50). Commit page by page, as far as the reservation goes.
        static bool told = false;
        if (!told) {
            told = true;
            fprintf(stderr, "moe-cache: note: commit of %zu bytes failed (Windows error %lu), committing page by page\n", (size_t) (b - a), (unsigned long) GetLastError());
        }
        for (uintptr_t q = a; q < b; q += 4096) {
            if (!VirtualAlloc((void *) q, 4096, MEM_COMMIT, PAGE_READWRITE)) {
                break;      // past the reservation: the rest is not ours
            }
        }
    }
#else
    if (mem_guard()) {
        const uintptr_t a = (uintptr_t) p / 4096 * 4096, b = ((uintptr_t) p + size + 4095) / 4096 * 4096;
        mprotect((void *) a, b - a, PROT_READ | PROT_WRITE);
    }
#endif
}
// p and size must be page aligned
inline void mem_decommit(void * p, size_t size) {
#ifdef _WIN32
    VirtualFree(p, size, MEM_DECOMMIT);
#else
    madvise(p, size, MADV_DONTNEED);
    if (mem_guard()) {
        mprotect(p, size, PROT_NONE);
    }
#endif
}
// Maps the file range [off, off + size) over p (copy-on-write, the way llama.cpp's own mmap loading sees the weights): the pages come from the
// system's file cache, which the system can reclaim. p and off must be page aligned and equal modulo the page size. Linux only; false elsewhere.
inline bool mem_map_file(void * p, size_t size, const file_t & f, uint64_t off) {
#if defined(__linux__)
    void * r = mmap(p, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, f.fd, (off_t) off);
    if (r == MAP_FAILED) {
        return false;
    }
    madvise(p, size, MADV_WILLNEED);
    return true;
#else
    (void) p; (void) size; (void) f; (void) off;
    return false;
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
inline unsigned long process_id() {
#ifdef _WIN32
    return (unsigned long) GetCurrentProcessId();
#else
    return (unsigned long) getpid();
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
// Windows: bytes in the system file cache working set: file data read through the cache that Windows keeps mapped. It trims it to the standby
// list as soon as memory is needed, so it is as good as available, but GlobalMemoryStatusEx counts it as in use. 0 elsewhere or if unknown.
inline size_t file_cache_ws() {
#ifdef _WIN32
    struct {    // SYSTEM_FILECACHE_INFORMATION (NtQuerySystemInformation class 21)
        SIZE_T CurrentSize, PeakSize;
        ULONG  PageFaultCount;
        SIZE_T MinimumWorkingSet, MaximumWorkingSet, CurrentSizeIncludingTransitionInPages, PeakSizeIncludingTransitionInPages;
        ULONG  TransitionRePurposeCount, Flags;
    } info;
    typedef LONG (WINAPI * nqsi_t)(int, void *, ULONG, ULONG *);
    static const nqsi_t fn = (nqsi_t) (void *) GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation");   // no import: looked up
    memset(&info, 0, sizeof(info));
    if (fn && fn(21, &info, (ULONG) sizeof(info), nullptr) == 0) {
        return (size_t) info.CurrentSize;
    }
#endif
    return 0;
}
// set by the plugin when it reads the model through the Windows file cache (MOE_CACHE_WIN_BUFFERED): that cache is then counted as available,
// else a warm start found the model's cached pages "in use" and started with a smaller budget (PC, Qwen3.5-122B: 9.1 instead of 12 GiB)
inline bool & count_file_cache() {
    static bool on = false;
    return on;
}

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
        if (GlobalMemoryStatusEx(&ms)) { lim = (size_t) ms.ullAvailPhys + (count_file_cache() ? file_cache_ws() : 0); }
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

// memory the system could still give out right now (Linux MemAvailable, Windows available physical memory); 0 if unknown
inline size_t mem_available() {
    size_t r = 0;
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) { r = (size_t) ms.ullAvailPhys + (count_file_cache() ? file_cache_ws() : 0); }
#else
    if (FILE * f = fopen("/proc/meminfo", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            unsigned long long kb = 0;
            if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { r = (size_t) kb * 1024; }
        }
        fclose(f);
    }
#endif
    return r;
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

// Windows: take the pages of this process's views of the given files out of its working set (VirtualUnlock on a range that is not locked does
// exactly that; the pages stay in the standby list, so a later touch is a soft fault). Only views mapped from these files are touched, never private
// memory. Returns the working-set bytes released; *views = how many mapped views matched. Elsewhere: 0.
// Why (I37, PC 2026-10-27): llama.cpp's Windows loader reads the GPU tensors through its mapping of the model file and never unmaps those pages,
// so ~13 GB of model file stayed in llama-server's working set (counted as in use) after loading; the plugin's cache then got that much less.
inline size_t trim_file_pages(const std::vector<std::string> & paths, size_t * views) {
    *views = 0;
#ifdef _WIN32
    std::vector<std::wstring> names;
    for (const std::string & p : paths) {
        std::wstring w = widen(p.c_str());
        const size_t sl = w.find_last_of(L"\\/");
        w = sl == std::wstring::npos ? w : w.substr(sl + 1);
        for (auto & ch : w) { ch = (wchar_t) towlower(ch); }
        names.push_back(w);
    }
    PROCESS_MEMORY_COUNTERS pmc;
    memset(&pmc, 0, sizeof(pmc));
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    const size_t ws0 = (size_t) pmc.WorkingSetSize;
    MEMORY_BASIC_INFORMATION mbi;
    const char * q = nullptr;
    while (VirtualQuery(q, &mbi, sizeof(mbi)) == sizeof(mbi)) {
        if (mbi.State == MEM_COMMIT && mbi.Type == MEM_MAPPED) {
            wchar_t fn[2048];
            const DWORD n = GetMappedFileNameW(GetCurrentProcess(), mbi.BaseAddress, fn, 2048);
            if (n > 0 && n < 2048) {
                std::wstring f(fn, n);
                const size_t sl = f.find_last_of(L"\\/");
                f = sl == std::wstring::npos ? f : f.substr(sl + 1);
                for (auto & ch : f) { ch = (wchar_t) towlower(ch); }
                if (std::find(names.begin(), names.end(), f) != names.end()) {
                    VirtualUnlock(mbi.BaseAddress, mbi.RegionSize);   // fails with ERROR_NOT_LOCKED, and releases the pages: intended
                    (*views)++;
                }
            }
        }
        const char * next = (const char *) mbi.BaseAddress + mbi.RegionSize;
        if (next <= q) {
            break;
        }
        q = next;
    }
    memset(&pmc, 0, sizeof(pmc));
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return ws0 > (size_t) pmc.WorkingSetSize ? ws0 - (size_t) pmc.WorkingSetSize : 0;
#else
    (void) paths;
    return 0;
#endif
}

// ---- small file helpers (UTF-8 paths) ----
inline void make_dir(const char * path) {
#ifdef _WIN32
    _wmkdir(widen(path).c_str());
#else
    mkdir(path, 0700);
#endif
}
// rename tmp over dst (replaces an existing file on both systems); false (with a message) if it could not
inline bool replace_file(const char * tmp, const char * dst) {
#ifdef _WIN32
    if (!MoveFileExW(widen(tmp).c_str(), widen(dst).c_str(), MOVEFILE_REPLACE_EXISTING)) {     // e.g. a virus scanner holding dst open
        fprintf(stderr, "moe-cache: cannot replace %s (Windows error %lu); the new data is in %s\n", dst, (unsigned long) GetLastError(), tmp);
        return false;
    }
#else
    if (rename(tmp, dst) != 0) {
        fprintf(stderr, "moe-cache: cannot replace %s (%s); the new data is in %s\n", dst, strerror(errno), tmp);
        return false;
    }
#endif
    return true;
}
inline void remove_file(const char * path) {
#ifdef _WIN32
    _wremove(widen(path).c_str());
#else
    std::remove(path);
#endif
}

}  // namespace plat
