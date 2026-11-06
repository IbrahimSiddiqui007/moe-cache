// CLEANUP-002: executor members (settings, pool, reader threads and queue, per-type counters)
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_exec_t {
    // executor (MOE_CACHE_EXEC=1): asynchronous reads + the plugin's own expert matrix products for decode steps
    bool   exec_on = true;                              // on by default; MOE_CACHE_EXEC=0 switches the executor off
    bool   tiled_cache = false;                         // MOE_CACHE_TILED_CACHE=1 (D19, default off): the tiled/duplicate-id scan runs once per ids tensor and graph run instead of for gate, up and down
    bool   exec_force = false;                          // MOE_CACHE_EXEC=2: the executor runs every eligible step (tests)
    int    exec_chunk = 128;                            // MOE_CACHE_EXEC_CHUNK=N: weight rows per pool chunk; 128 decodes +3.9 % over 32 on the PC (N25, every ABBA pair), the arithmetic per row is unchanged
    bool   exec_fuse = true;                            // gate and up of one layer share one pool job (same arithmetic, one barrier fewer): decode +0.3-2.6 % in every PC ABBA pair, text identical (E3, RESULTS 73); MOE_CACHE_EXEC_FUSE=0 turns it off
    bool   exec_check = false;                          // MOE_CACHE_EXEC_CHECK=1: recompute every op the stock way and compare (slow, for tests)
    int    exec_threads = 0;                            // 0 = the thread count llama.cpp asks for
    bool   exec_pool_decode = false;                    // MOE_CACHE_EXEC_POOL_DECODE=1 (DECODE-137, default off): size the pool from the first single-token graph (src/mc_poolsize.h)
    int    exec_pool_n = 0;                             // compute threads of the current pool
    bool   exec_pool_from_decode = false;               // the pool was built (or rebuilt) from a single-token graph
    int    exec_maxcol = EXEC_MAXCOL;                   // MOE_CACHE_EXEC_MAXCOL=N (1-8): largest batch (tokens) the executor takes
    int    exec_ecores = 0;                             // extra workers on efficiency cores
    int    exec_readers = 4;
    std::unique_ptr<exec_pool_t> pool;
    std::vector<std::thread> readers;
    std::mutex rq_mu;
    std::condition_variable rq_cv;
    std::deque<io_job_t> rq;
    std::atomic<int> demand_pending{0};          // idea I02: demand reads queued or running in the reader threads (advisory prefetch reads wait while this is above 0)
    bool   rq_stop = false;
    std::atomic<uint64_t> exec_fused{0}, exec_ops{0}, exec_mismatch{0}, exec_wait_ns{0}, exec_checked{0};
    uint64_t stepaside_graphs = 0, stepaside_waits = 0;   // executor graphs handed back to llama.cpp with everything resident / of them, graphs that had to wait for a load
    // per weight type: wall time of the expert products (includes waiting for experts still loading), ops, expert bytes touched
    std::atomic<uint64_t> exec_type_ns[64] = {}, exec_type_ops[64] = {}, exec_type_bytes[64] = {};
};
