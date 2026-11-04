// CLEANUP-002: idle rebalance members (BO-2 settings, thread, round counters)
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_rebalance_t {
    // idle rebalance (BO-2): while no graph runs for rb_idle_ms, read the experts with the highest decayed use count that are missing into free room or
    // the room of the lowest-count resident ones (rb_choose), at most rb_cap_bytes per idle period, one expert at a time, stopping as soon as a graph starts
    bool   rb_on = true;                                // BO-2 idle refill: on since 0.3.3 (won every PC pair, RESULTS 73); MOE_CACHE_IDLE_REBALANCE=0 turns it off
    int    rb_idle_ms = 1500;
    size_t rb_cap_bytes = (size_t) 512 << 20;
    double rb_gain = 2.0, rb_min_score = 0.75;
    std::thread rb_th;
    std::condition_variable rb_cv;
    bool   rb_stop = false;
    int    graph_active = 0;                            // be_compute calls in flight (mu)
    uint64_t graph_seq = 0;                             // graphs started (mu)
    uint64_t rb_round_seq = ~0ull;                      // graph_seq of the last rebalance round: one round per idle period
};
