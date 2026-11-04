// CLEANUP-002: lookahead prefetch, helper threads, router tables and sibling/coalesce settings
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_prefetch_t {
    bool   lookahead = false;
    bool   la_exec = false;                             // MOE_CACHE_LOOKAHEAD=2: lookahead together with the executor (it waits on the same loading flags)
    size_t coalesce_bytes = (size_t) 16 << 20;          // MOE_CACHE_COALESCE_MIB=N (default 16, 0 = off): neighbouring expert reads merged up to N MiB
                                                        // (synchronous reads; Flash-Next prompts: 2.5 -> 3.4 GB/s, prefill +14-24 %)
    std::atomic<uint64_t> coalesced{0};                 // expert reads merged into the previous one
    int    la_guess = 0;                                // experts looked at per layer (0 = the number used per token)
    int    la_cap = 2;                                  // at most this many experts not yet in RAM are prefetched per layer
    bool   la_ecores = true;                            // run the helper threads on the efficiency cores of a hybrid CPU
    size_t max_nb2_all = 0;                             // largest expert slice, for the helper threads' bounce buffers
    std::vector<std::thread> pf_threads;
    std::thread stats_th;                                // snapshot thread (MOE_CACHE_STATS_SIGNAL: SIGUSR1 / event): joined in the destructor
#ifdef _WIN32
    std::thread save_th;                                 // profile save on request: joined in the destructor
#endif
    std::thread idle_th;                                 // idle release on request (SIGUSR2 / Local\moe-cache-idle-<pid>): joined in the destructor
    bool   idle_released = false;                       // idle_release() ran and no graph came since: the next graph sets the working-set minimum again
    uint64_t idle_releases = 0;
    std::mutex pf_mu;
    std::condition_variable pf_cv;
    std::deque<std::function<void()>> pf_q;
    bool   pf_stop = false;
    std::condition_variable load_cv;                    // used with mu: a lookahead read finished
    std::unordered_map<int, std::vector<int>> layer_tensors;   // layer -> indices into tensors
    struct tmeta_t { int type; size_t off; size_t size; int shard; };
    std::unordered_map<std::string, tmeta_t> tmeta;     // router and norm tensors of the GGUF
    struct router_t { std::vector<float> w, bias, normw; bool ok = false; };
    std::unordered_map<int, router_t> routers;
    std::mutex router_mu;
    std::unordered_map<int, std::vector<char>> pred_flags;     // layer -> experts predicted for it (under mu)
    std::atomic<uint64_t> pf_issued{0}, pf_used{0}, pf_wasted{0}, pf_late{0}, pf_pred_total{0}, pf_pred_hit{0}, pf_skipped{0};
};
