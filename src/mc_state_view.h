// CLEANUP-002: file-view cache members
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_view_t {
    bool   win_view = false;                    // file-view cache (default on Windows and Linux; MOE_CACHE_WIN_VIEW=0 / MOE_CACHE_LINUX_VIEW=0 turn it off; see the constructor)
    std::vector<const char *> views;            // win_view: one read-only view per model file (split GGUF)
    std::vector<size_t> view_sizes;
    size_t view_bytes = 0;                      // win_view: resident bytes of experts held in the views (not private memory: not in rss_private())
    size_t ws_min_cur = 0;                      // win_view: the hard working-set minimum set last (0 = none)
    bool   ws_min_warned = false;
};
