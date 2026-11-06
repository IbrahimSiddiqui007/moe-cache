// CLEANUP-002: last-graph flags, rebalance and class-list counters
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_rbstats_t {
    std::atomic<uint64_t> last_end_ns{0};               // steady clock, end of the last graph
    bool   rb_acted = false;                            // the last round read something: measure the next graphs
    bool   rb_armed_dec = false, rb_armed_pre = false;  // count the hits of the first decode / prefill graph after a rebalance
    bool   g_saw_dec = false, g_saw_pre = false;
    uint64_t rb_rounds = 0, rb_reads = 0, rb_bytes_read = 0, rb_swaps = 0, rb_cancelled = 0, rb_evicted_bytes = 0;
    uint64_t rb_dec_hits = 0, rb_dec_miss = 0, rb_pre_hits = 0, rb_pre_miss = 0;
    uint64_t cls_promotions = 0, cls_demotions = 0;     // touches that moved a node to a higher / lower class list
    int    unmatched = 0;              // expert tensors that could not be cached (see register_tensor)
};
