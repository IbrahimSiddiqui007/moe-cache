// CLEANUP-002: replacement policy members (class LRU lists, decay table, clock)
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_policy_t {
    // Replacement policy. MOE_CACHE_POLICY=lru (default): one list, evict the least recently used. MOE_CACHE_POLICY=classlru (llama.cpp #31233):
    // nclass LRU lists; a node's decayed use count crf (half-life classlru_hl decode tokens, kept after eviction) picks its list
    // (class = min(nclass-1, floor(log2 crf))); the victim is the oldest evictable node of each list with the largest age / 2^class.
    // Only the choice of victim changes, never what is computed. The plain policy is the same code with one list.
    static constexpr int MAXCLS = 8;
    int    mruc[MAXCLS] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    int    lruc[MAXCLS] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    bool   classlru = false;
    int    nclass = 1;
    double classlru_hl = 128.0;
    std::vector<float> decay_tab;                       // 2^(-dt / half-life) for dt = 0 .. size-1 (beyond: 0)
    uint32_t tok = 0;                                   // decode token counter (classlru clock)
    int    last_dec_layer = -1;
    bool   track_crf = false;                           // keep the decayed use counts (classlru, or the idle rebalance wants them)
};
