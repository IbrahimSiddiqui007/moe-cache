// CLEANUP-002 / CLEANUP-1198: coalesce settings, the snapshot / save / idle threads, the load-finished wait and the layer index (the lookahead prefetch that lived here was removed)
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_io_t {
    size_t coalesce_bytes = (size_t) 16 << 20;          // MOE_CACHE_COALESCE_MIB=N (default 16, 0 = off): neighbouring expert reads merged up to N MiB
                                                        // (synchronous reads; Flash-Next prompts: 2.5 -> 3.4 GB/s, prefill +14-24 %)
    std::atomic<uint64_t> coalesced{0};                 // expert reads merged into the previous one
    size_t max_nb2_all = 0;                             // largest expert slice, for the bounce buffers
    std::thread stats_th;                                // snapshot thread (MOE_CACHE_STATS_SIGNAL: SIGUSR1 / event): joined in the destructor
#ifdef _WIN32
    std::thread save_th;                                 // profile save on request: joined in the destructor
#endif
    std::thread idle_th;                                 // idle release on request (SIGUSR2 / Local\moe-cache-idle-<pid>): joined in the destructor
    bool   idle_released = false;                       // idle_release() ran and no graph came since: the next graph sets the working-set minimum again
    uint64_t idle_releases = 0;
    std::condition_variable load_cv;                    // used with mu: a background read finished
    std::unordered_map<int, std::vector<int>> layer_tensors;   // layer -> indices into tensors
};
