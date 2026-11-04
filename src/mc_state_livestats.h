// CLEANUP-002: live statistics, decode flag, id trace file and per-layer hit counters
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_livestats_t {
    // live statistics: with MOE_CACHE_STATS_SIGNAL=1, SIGUSR1 makes the plugin print one "moe-cache-stats {json}" line (cumulative counters)
    std::atomic<uint64_t> hits_dec{0}, misses_dec{0}, evict_age_sum{0};
    bool cur_decode = false;                            // the group being prepared is a single-token (decode) step
    FILE * trace_f = nullptr;                           // MOE_CACHE_TRACE_IDS=file: record the experts picked in every decode step (int16 layer, int16 n, n x int16 expert)
    std::vector<uint64_t> layer_hits, layer_misses;     // per layer, updated under mu
};
