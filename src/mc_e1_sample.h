// moe-cache plugin part: MOE_CACHE_E1_SAMPLE=1 (Windows, diagnostic, with MOE_CACHE_E1_TIMES=1): what the graph thread runs between our graphs. About every 15 ms during a gap the
// thread is suspended, its instruction pointer read, and the module + 4 KB page counted; stats JSON "e1_sample". Off (and nothing started) without the variable.
#pragma once
#include <string>
#ifdef _WIN32
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
namespace e1s {
inline std::atomic<bool> in_gap{false};
inline std::atomic<HANDLE> th{nullptr};
inline std::atomic<DWORD> tid{0};
inline std::mutex mu;
inline std::map<std::string, uint64_t> mods;
inline std::map<std::string, uint64_t> pages;
inline std::map<std::string, uint64_t> rips;      // exact leaf address, 16-byte rounded
inline std::map<std::string, uint64_t> chains;    // leaf module < first return addresses found on the stack in other modules
inline bool mod_of(uintptr_t a, std::string & name, uintptr_t & off) {
    HMODULE m = nullptr;
    char path[MAX_PATH];
    if (a < 0x10000 || !GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) a, &m) || !m || !GetModuleFileNameA(m, path, MAX_PATH)) { return false; }
    name = path;
    const size_t sl = name.find_last_of("\\/");
    if (sl != std::string::npos) { name = name.substr(sl + 1); }
    off = a - (uintptr_t) m;
    return true;
}
inline void loop() {
    for (;;) {
        Sleep(1);
        HANDLE h = th.load();
        if (!h || !in_gap.load()) { continue; }
        if (SuspendThread(h) == (DWORD) -1) { continue; }
        CONTEXT c;
        c.ContextFlags = CONTEXT_CONTROL;
        const BOOL ok = GetThreadContext(h, &c);       // nothing here may allocate or take a lock: the suspended thread may hold one
        const bool still = in_gap.load();
        static uintptr_t stk[1024];                    // top of its stack, copied while it is stopped (no unwinding: that takes locks)
        size_t n_stk = 0;
        MEMORY_BASIC_INFORMATION mi;
        if (ok && VirtualQuery((LPCVOID) c.Rsp, &mi, sizeof mi)) {
            const size_t room = (uintptr_t) mi.BaseAddress + mi.RegionSize - (uintptr_t) c.Rsp;
            n_stk = std::min(room / sizeof(uintptr_t), (size_t) 1024);
            memcpy(stk, (const void *) c.Rsp, n_stk * sizeof(uintptr_t));
        }
        ResumeThread(h);
        if (!ok || !still) { continue; }
        std::string name = "unknown";
        uintptr_t off = (uintptr_t) c.Rip;
        mod_of((uintptr_t) c.Rip, name, off);
        char pg[32], ex[32];
        snprintf(pg, sizeof pg, "+%llx", (unsigned long long) (off & ~(uintptr_t) 0xfff));
        snprintf(ex, sizeof ex, "+%llx", (unsigned long long) (off & ~(uintptr_t) 0xf));
        std::string chain = name, last = name;         // the first 4 stack values that point into a different module than the one before (may include stale values)
        int links = 0;
        for (size_t i = 0; i < n_stk && links < 4; i++) {
            std::string nm;
            uintptr_t o = 0;
            if (mod_of(stk[i], nm, o) && nm != last) {
                char cb[32];
                snprintf(cb, sizeof cb, "+%llx", (unsigned long long) o);
                chain += "<" + nm + cb;
                last = nm;
                links++;
            }
        }
        std::lock_guard<std::mutex> l(mu);
        mods[name]++;
        pages[name + pg]++;
        rips[name + ex]++;
        chains[chain]++;
    }
}
inline bool on() { static const bool v = [] { if (!getenv("MOE_CACHE_E1_SAMPLE")) { return false; } std::thread(loop).detach(); return true; }(); return v; }
inline void enter() { if (on()) { in_gap = false; } }
inline void exit_graph() {
    if (!on()) { return; }
    const DWORD me = GetCurrentThreadId();
    if (tid.load() != me) {
        HANDLE d = nullptr;
        if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &d, THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0)) { th = d; tid = me; }
    }
    in_gap = true;
}
inline std::string json() {
    if (!on()) { return std::string(); }
    std::lock_guard<std::mutex> l(mu);
    auto top = [](const std::map<std::string, uint64_t> & m, size_t k) {
        std::vector<std::pair<uint64_t, std::string>> v;
        for (const auto & e : m) { v.push_back({e.second, e.first}); }
        std::sort(v.rbegin(), v.rend());
        std::string j = "{";
        for (size_t i = 0; i < v.size() && i < k; i++) { j += (i ? ",\"" : "\"") + v[i].second + "\":" + std::to_string(v[i].first); }
        return j + "}";
    };
    return ",\"e1_sample\":{\"mods\":" + top(mods, 12) + ",\"pages\":" + top(pages, 15) + ",\"rips\":" + top(rips, 25) + ",\"chains\":" + top(chains, 25) + "}";
}
}
#else
namespace e1s { inline void enter() {} inline void exit_graph() {} inline std::string json() { return std::string(); } }
#endif
