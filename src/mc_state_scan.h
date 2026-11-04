// CLEANUP-002: sibling batch load, long-prompt scan settings and small-cache counters
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_scan_t {
    bool   siblings = true;                             // MOE_CACHE_SIBLINGS=0 turns the sibling batch load off
    std::atomic<uint64_t> sibling_loads{0};
    bool   cur_scan = false;                            // the group being prepared is a long prompt batch (MOE_CACHE_SCAN_TOKENS)
    int    scan_tokens = 512;                           // MOE_CACHE_SCAN_TOKENS=N (default 512, 0 = off; I3, RESULTS 48): batches of N+ tokens load experts at the evict-first end, so the experts
                                                        // cached before the prompt survive from chunk to chunk (with LRU a cyclic scan bigger than the cache keeps nothing)
    uint64_t small_mapped = 0, small_over = 0;          // cache smaller than one op: experts mapped from the file (stock way) / loaded over the budget
    bool   small_warned = false;
};
