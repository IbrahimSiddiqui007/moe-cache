// RAM-LIVE: the cache follows the RAM the system has available while the server runs. Header-only, no ggml (tests/livecap_test.cpp).
//
// The launcher sizes the cache once from the RAM available at start. Other programs then take or free RAM: this keeps `keep` bytes of the system's
// available RAM free by lowering the cache's ceiling (tune() evicts down to it) and gives the room back slowly once the RAM is free again.
// Off unless MOE_CACHE_AVAIL_KEEP_GIB is set (the launcher sets it: max(4 GiB, 12.5 % of the RAM)) AND follows() is true for the cache path (the plugin then sets live_keep); with live_keep 0 nothing here is called.
// Only the cache's own memory can be given back: what llama-server or other programs map or allocate is not touched.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace mclive {

constexpr size_t GIB = (size_t) 1 << 30;
constexpr size_t GROW_MARGIN = GIB;                      // grow only when available is this far over `keep` (no flapping around the line)
constexpr size_t GROW_STEP = (size_t) 256 << 20;         // at most this much per grow ...
constexpr uint64_t GROW_EVERY_MS = 1000;                 // ... and at most one grow per second

// Whether RAM-LIVE runs on this cache path (LAUNCHER-1277). It does not on Linux's file-view path: evicting there only drops mapped pages to the page cache, which
// MemAvailable counts already, so the figure would not rise and the cache would shrink to its floor (critic 2026-11-17). tune(), the plugin's start-up line and the
// launcher's plan line all use or mirror this one rule (tests/livecap_test.cpp, tests/ram_cap_decision_test.py).
constexpr bool follows(bool file_view, bool is_windows) {
    return !file_view || is_windows;
}

// The most private memory the process may hold: `target` (the ordinary limit), and never so much that available RAM falls under `keep`.
// avail < keep: the process gives back the shortfall. avail >= keep: it may hold what it holds now plus the room over `keep`.
inline size_t ceiling(size_t target, size_t rss, size_t avail, size_t keep) {
    if (!keep || !avail) {                               // off, or the system gave no figure: nothing changes
        return target;
    }
    if (avail < keep) {
        const size_t shortfall = keep - avail;
        return std::min(target, rss > shortfall ? rss - shortfall : (size_t) 0);
    }
    return std::min(target, rss + (avail - keep));
}

// May the cache's budget grow now? True when available RAM is `keep` + GROW_MARGIN or more and the last grow was a second ago or more.
inline bool grow_allowed(size_t avail, size_t keep, uint64_t now_ms, uint64_t last_grow_ms) {
    return keep && avail >= keep + GROW_MARGIN && now_ms >= last_grow_ms + GROW_EVERY_MS;
}

// How far one grow step goes: GROW_STEP, less when the room over keep + GROW_MARGIN is smaller.
inline size_t grow_step(size_t avail, size_t keep) {
    return avail > keep + GROW_MARGIN ? std::min(GROW_STEP, avail - keep - GROW_MARGIN) : (size_t) 0;
}

}  // namespace mclive
