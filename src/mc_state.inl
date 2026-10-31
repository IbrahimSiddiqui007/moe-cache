// moe-cache plugin part: the cache: settings and state (state_t; its methods continue in mc_state_io.inl, mc_state_profile.inl, mc_state_lru.inl).
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

ggml_backend_buffer_t buffer_init_padded(ggml_backend_buffer_type_t buft, const ggml_backend_buffer_i & i, void * context, size_t size);   // mc_buffer.inl

// llama-server's own settings that decide where its CPU weights go (read once: environment, then the command line, the last occurrence winning)
static const mcargs::server_args & server_args() {
    static const mcargs::server_args a = mcargs::parse_server_args(plat::command_line(), getenv("LLAMA_ARG_REPACK"), getenv("LLAMA_ARG_NO_HOST"),
                                                                   getenv("LLAMA_ARG_DEVICE"));
    return a;
}

// Windows: the plugin repacked although llama-server had --no-repack, so CPU-only output differed from stock (Windows CI executor step: 1/4 identical)
static bool server_repack_disabled() {
    return !server_args().repack;
}

struct state_t;
#ifdef _WIN32
void install_crash_report(state_t * s);
#endif

struct state_t {
    bool   cache_on = false;
    bool   stats = false;
    int    lo = 0, hi = 1 << 30;
    // "all" for the default range (hi = 2^30 means no upper bound; the log printed "layers 0-1073741824", T7)
    std::string layer_range() const { return lo == 0 && hi >= (1 << 30) ? std::string("all") : std::to_string(lo) + "-" + std::to_string(hi); }
    std::vector<plat::file_t> fds;              // one per GGUF file (split models have several)
    std::vector<std::string> shard_paths;
    std::vector<plat::file_t> fds_buf;          // buffered (no O_DIRECT) handles, opened only if a direct read keeps failing
    std::mutex fds_buf_mu;
    int    test_read_delay_us = 0;              // MOE_CACHE_TEST_READ_DELAY_US=N: test switch, background expert reads start N us late
    int    fault_every = 0;                     // MOE_CACHE_FAULT_EVERY=N: test switch, every Nth expert read fails all its direct attempts
    bool   test_poison = false;                 // MOE_CACHE_TEST_POISON_READS=1: test switch, a read first fills what it will write with 0xFF and waits
                                                //   MOE_CACHE_TEST_READ_DELAY_US (what a page under an unbuffered Windows read looks like to other threads)
    std::atomic<uint64_t> fault_ctr{0}, read_fallbacks{0};
    std::unordered_map<std::string, int> file_shard;
    ggml_backend_buffer_type_t repack_buft = nullptr;   // llama.cpp CPU_REPACK buffer type (mode on)
    ggml_backend_buffer_t repack_buf = nullptr;         // a real repack buffer, used for init_tensor / set_tensor only
    ggml_backend_buffer_t repack_tag = nullptr;         // empty buffer of the repack type, swapped in while an operation runs
    std::atomic<uint64_t> repack_jobs{0};
    ggml_backend_dev_t gpu_dev = nullptr;               // first GPU backend device, used for long prompts
    int    gpu_min_tokens = 32;                         // batches of at least this many tokens run their expert block on the GPU (0 = never)
    std::atomic<uint64_t> hyb_groups{0}, hyb_picks{0}, hyb_hits{0};   // MOE_CACHE_HYBRID: layers split between GPU bank and CPU, picks, picks served by the bank
    std::atomic<uint64_t> hyb_drops{0}, hyb_reloads{0}, hyb_reload_ns{0};
    std::atomic<uint64_t> hyb_swaps{0}, hyb_swap_ns{0}, hyb_reverts{0};   // MOE_CACHE_HYBRID_ADAPT: experts swapped into the bank, time spent uploading them   // bank freed for a long prompt's GPU block, loaded again at the next decode step
    std::atomic<uint64_t> gpu_segments{0}, gpu_fallbacks{0}, gpu_nomem{0}, gpu_split_blocks{0};   // gpu_nomem: blocks kept on the CPU because the GPU had too little free memory; gpu_split_blocks: blocks run in parts (N23)
    std::atomic<uint64_t> gpu_up_bytes{0};
    // live statistics: with MOE_CACHE_STATS_SIGNAL=1, SIGUSR1 makes the plugin print one "moe-cache-stats {json}" line (cumulative counters)
    std::atomic<uint64_t> hits_dec{0}, misses_dec{0}, evict_age_sum{0};
    bool cur_decode = false;                            // the group being prepared is a single-token (decode) step
    // approximate routing (MOE_CACHE_ROUTE=skip|subst, decode and small batches, executor only; off by default; quality-gated, see docs/EXPERIMENTS.md):
    // the lowest-ranked picks of a token that are NOT in RAM are dropped (skip) or replaced by the most recently used resident expert of that layer (subst).
    // Ranked picks are in order of the router's weights, so the experts given up are the ones that contribute least.
    int  route_mode = 0;                                // 0 off, 1 skip, 2 subst
    int  route_keep = -1;                               // MOE_CACHE_ROUTE_KEEP: picks always honoured (default: all but the last route_drop)
    int  route_drop = 2;                                // MOE_CACHE_ROUTE_DROP: lowest-ranked picks that may be skipped (--fast=2: 4)
    uint64_t route_skipped = 0, route_substituted = 0, route_considered = 0;
    // fewer experts (MOE_CACHE_FEWER=N, approximate, B3): on layers the executor computes (CPU-held), decode and small batches use only the top n_used-N picks
    // (never fewer than half), whether they are in RAM or not. Prompt batches keep all picks. The router's weights are not renormalised (the dropped
    // experts are the lowest-ranked ones); MOE_CACHE_FEWER_RENORM=1 is not implemented, quality is judged by the gate (docs/QUALITY_GATE.md, level fewer).
    int  fewer = 0;
    uint64_t fewer_dropped = 0, fewer_considered = 0;
    std::unordered_map<const ggml_tensor *, std::vector<int32_t>> route_ids;   // ids tensor -> effective expert per (token, pick); -1 = skipped
    FILE * trace_f = nullptr;                           // MOE_CACHE_TRACE_IDS=file: record the experts picked in every decode step (int16 layer, int16 n, n x int16 expert)
    std::vector<uint64_t> layer_hits, layer_misses;     // per layer, updated under mu
    // ---- lookahead prefetch (MOE_CACHE_LOOKAHEAD=1): while layer N computes, predict the experts of layer N+1 from the router weights and read them
    bool   siblings = true;                             // MOE_CACHE_SIBLINGS=0 turns the sibling batch load off
    std::atomic<uint64_t> sibling_loads{0};
    bool   cur_scan = false;                            // the group being prepared is a long prompt batch (MOE_CACHE_SCAN_TOKENS)
    int    scan_tokens = 512;                           // MOE_CACHE_SCAN_TOKENS=N (default 512, 0 = off; I3, RESULTS 48): batches of N+ tokens load experts at the evict-first end, so the experts
                                                        // cached before the prompt survive from chunk to chunk (with LRU a cyclic scan bigger than the cache keeps nothing)
    uint64_t small_mapped = 0, small_over = 0;          // cache smaller than one op: experts mapped from the file (stock way) / loaded over the budget
    bool   small_warned = false;
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
    bool   disabled = false;                    // setup failed or MOE_CACHE_DISABLE: the plugin claims nothing, the model loads normally
    bool   direct_ok = false;                   // read experts straight into place when the alignment allows
#ifdef _WIN32
    bool   win_buffered = false;                // model files opened without FILE_FLAG_NO_BUFFERING: MOE_CACHE_WIN_BUFFERED=1, or the private-memory path by default (see the constructor)
    bool   win_random = false;                  // MOE_CACHE_WIN_BUFFERED=2: and with FILE_FLAG_RANDOM_ACCESS
#else
    static constexpr bool win_buffered = false, win_random = false;
#endif
    // BO-1 (MOE_CACHE_PACK, see bin/moe_cache_pack.py): MODEL.moepack holds each expert's gate/up/down slices in one 4 KiB-aligned block; a missing expert is
    // then one large unbuffered read instead of three. Replaces the file-view cache while it is in use (reads fill private memory).
    struct pack_layer_t { uint64_t base = 0, stride = 0; int n_expert = 0, n = 0; std::string name[4]; size_t nb2[4] = {}, in_off[4] = {}; };
    bool   pack_on = false;
    plat::file_t pack_fd;
    std::string pack_path;
    std::vector<pack_layer_t> pack_layers;
    std::unordered_map<std::string, std::pair<int, int>> pack_by_name;   // tensor name -> (layer record, slot)
    std::atomic<uint64_t> pack_reads{0}, pack_bytes{0}, pack_slices{0}, pack_fallbacks{0};
    bool   win_view = false;                    // file-view cache (default on Windows and Linux; MOE_CACHE_WIN_VIEW=0 / MOE_CACHE_LINUX_VIEW=0 turn it off; see the constructor)
    std::vector<const char *> views;            // win_view: one read-only view per model file (split GGUF)
    std::vector<size_t> view_sizes;
    size_t view_bytes = 0;                      // win_view: resident bytes of experts held in the views (not private memory: not in rss_private())
    size_t ws_min_cur = 0;                      // win_view: the hard working-set minimum set last (0 = none)
    bool   ws_min_warned = false;
    size_t budget = 0;
    bool   commit_warned = false;      // Windows: the commit-limit warning was printed (tune())
    bool   commit_fail_warned = false; // Windows: a refused expert commit was reported (commit_for())
    uint64_t commit_evictions = 0;     // experts evicted because Windows refused a commit or system commit ran low
    size_t budget_peak = 0;          // largest budget so far; a fall below half of it is reported once (shrink_warned)
    bool shrink_warned = false;
    bool   auto_mode = false;          // MOE_CACHE_SIZE_GIB=auto: size the cache from the real memory limit
    size_t limit_bytes = 0;
    size_t margin = (size_t) 1 << 30;  // keep this much below the limit (MOE_CACHE_MARGIN_GIB, default 1 GiB)
    size_t pending_bytes = 0;          // expert bytes marked resident but not read yet
    size_t max_budget = 0;             // all registered expert bytes
    size_t min_budget = 0;
    uint64_t tune_shrinks = 0, tune_grows = 0;
    size_t resident_bytes = 0;
    uint64_t epoch = 0;
    // experts with epoch >= protect_epoch are used by the graph being prepared/computed: no eviction may take them. All groups of a graph are
    // prepared before any is computed, so protecting only the current group (epoch == epoch) let a later group (or tune() shrinking the budget)
    // evict an earlier group's experts before llama.cpp read them: decommitted memory on Windows (access violation, PC 2026-10-27), zeros on Linux
    uint64_t protect_epoch = UINT64_MAX;
    uint64_t cpu_check_reloads = 0;                     // experts cpu_path_check() had to load again (should stay 0)
    bool in_use(const node_t & n) const { return n.loading || n.epoch == epoch || n.epoch >= protect_epoch; }
#ifdef _WIN32
    // once, at the first graph (llama.cpp has uploaded every tensor) and before the warm start fills the cache: release the model-file pages
    // llama.cpp's loader left in the working set (I37), then size the auto budget again from what is available now
    void release_model_pages() {
        size_t nviews = 0;
        const size_t freed = plat::trim_file_pages(shard_paths, &nviews, views);     // never the file-view cache's own views
        if (nviews) {
            fprintf(stderr, "moe-cache: released %.2f GiB of model-file pages from the working set (Windows, %zu mapped views)\n", freed / 1073741824.0, nviews);
        }
        // the start budget again, now that llama.cpp has allocated everything (weights, KV, compute buffers, an MTP draft context): from what
        // the rest of the process really uses instead of the 3.5 GiB guess, and never more than the system's free commit minus commit_keep()
        // (GLM + MTP on the PC: 18.3 GiB start budget, and the whole system ran out of commit)
        std::lock_guard<std::mutex> lk(mu);
        if (!auto_mode) {
            return;
        }
        const size_t old = budget;
        if (!getenv("MOE_CACHE_RAM_GIB")) {
            limit_bytes = read_mem_limit(view_bytes);
        }
        const size_t rss = read_rss_anon() + view_bytes;
        const size_t other = rss > resident_bytes ? rss - resident_bytes : 0;
        const size_t floor_b = (size_t) 512 << 20;
        size_t nb = limit_bytes > margin + other + floor_b ? limit_bytes - margin - other : floor_b;
        const size_t ca = plat::commit_available(), keep = plat::commit_keep();
        const size_t by_commit = (ca > keep ? ca - keep : 0) + resident_bytes;
        nb = std::max(floor_b, win_view ? nb : std::min(nb, by_commit));     // file-view experts use no commit
        if (max_budget) {
            nb = std::min(nb, max_budget);
        }
        budget = nb;
        fprintf(stderr, "moe-cache: memory limit now %.2f GiB, rest of llama-server %.2f GiB, free commit %.2f GiB (keeping %.2f GiB): start budget %.2f GiB "
                "(was %.2f)\n", limit_bytes / 1073741824.0, other / 1073741824.0, ca / 1073741824.0, keep / 1073741824.0, budget / 1073741824.0,
                old / 1073741824.0);
        update_ws_min();
    }
#endif
    void begin_graph() {
        std::lock_guard<std::mutex> lk(mu);
        protect_epoch = epoch + 1;
        if (idle_released) {
            idle_released = false;
            update_ws_min();             // the first graph after an idle release holds the experts in RAM again (no-op on Linux)
        }
    }
    std::unordered_map<std::string, std::pair<size_t, size_t>> file_tensors;
    std::vector<std::unique_ptr<tinfo_t>> tensors;
    std::unordered_map<const ggml_tensor *, int> by_tensor;
    std::vector<node_t> nodes;
    std::vector<uint32_t> node_owner;           // index in `tensors` of the tensor that owns each node (same size as nodes)
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
    std::atomic<uint64_t> last_end_ns{0};               // steady clock, end of the last graph
    bool   rb_acted = false;                            // the last round read something: measure the next graphs
    bool   rb_armed_dec = false, rb_armed_pre = false;  // count the hits of the first decode / prefill graph after a rebalance
    bool   g_saw_dec = false, g_saw_pre = false;
    uint64_t rb_rounds = 0, rb_reads = 0, rb_bytes_read = 0, rb_swaps = 0, rb_cancelled = 0, rb_evicted_bytes = 0;
    uint64_t rb_dec_hits = 0, rb_dec_miss = 0, rb_pre_hits = 0, rb_pre_miss = 0;
    uint64_t cls_promotions = 0, cls_demotions = 0;     // touches that moved a node to a higher / lower class list
    int    unmatched = 0;              // expert tensors that could not be cached (see register_tensor)
    // executor (MOE_CACHE_EXEC=1): asynchronous reads + the plugin's own expert matrix products for decode steps
    bool   exec_on = true;                              // on by default; MOE_CACHE_EXEC=0 switches the executor off
    bool   exec_force = false;                          // MOE_CACHE_EXEC=2: the executor runs every eligible step (tests)
    int    exec_chunk = 128;                            // MOE_CACHE_EXEC_CHUNK=N: weight rows per pool chunk; 128 decodes +3.9 % over 32 on the PC (N25, every ABBA pair), the arithmetic per row is unchanged
    bool   exec_fuse = true;                            // gate and up of one layer share one pool job (same arithmetic, one barrier fewer): decode +0.3-2.6 % in every PC ABBA pair, text identical (E3, RESULTS 73); MOE_CACHE_EXEC_FUSE=0 turns it off
    bool   exec_check = false;                          // MOE_CACHE_EXEC_CHECK=1: recompute every op the stock way and compare (slow, for tests)
    int    exec_threads = 0;                            // 0 = the thread count llama.cpp asks for
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

    // usage profile and warm start
    std::string profile_path;          // learn here and warm start from it
    std::string warm_path;             // warm start only
    bool   preload_on = false;
    bool   preload_done = false;
    double preload_frac = -1.0;        // < 0: everything if it fits, else 0.7
    double profile_decay = 0.5;
    uint64_t fingerprint = 0;
    usage_t learned;
    std::vector<std::vector<double>> session;   // [layer][expert] picks of this run
    uint64_t groups_since_save = 0;
    std::mutex mu;

    std::atomic<uint64_t> hits{0}, misses{0}, bytes{0}, groups{0}, ops{0}, graphs{0};
    std::atomic<uint64_t> ns_prep{0}, ns_cpu{0}, ns_total{0}, ns_read{0}, evictions{0}, direct_jobs{0}, bounce_jobs{0}, async_in{0};
    // MOE_CACHE_SEG_TIMES=1 (N23-F6): where the time outside our own work goes. gap = from the end of one of our graph splits to the start of the next
    // (llama.cpp's other splits, the scheduler, copies, GPU waits), sync = the wait for inputs read asynchronously from the GPU (it contains the GPU
    // time of the split before us), gpu_seg = long-prompt GPU blocks incl. uploads. gap buckets: < 0.2 ms, < 1 ms, < 5 ms, >= 5 ms.
    std::atomic<uint64_t> ns_gap{0}, ns_sync{0}, ns_gpu_seg{0}, n_gap{0}, last_exit_ns{0};
    std::atomic<uint64_t> gap_b_ns[4]{}, gap_b_n[4]{};
    std::atomic<uint64_t> gap_faults{0}, in_faults{0}, last_exit_faults{0}, entry_faults{0}, top_gap_ns[4]{}, top_gap_graph[4]{}, top_gap_faults[4]{};   // faults of the process inside the gaps / inside our backend; the 4 longest gaps
    std::atomic<uint64_t> ex_quant_ns{0}, ex_build_ns{0}, ex_run_ns{0}, ex_part_ns[3]{}, ex_part_bytes[3]{}, ex_part_n[3]{};   // MOE_CACHE_EXEC_TIMES: executor phases; part 0 gate, 1 up, 2 down

    // io workers
    std::vector<std::thread> workers;
    std::mutex wmu;
    std::condition_variable wcv, dcv;
    std::function<void(int)> task;
    int    gen = 0;
    int    busy = 0;
    bool   stop = false;

    state_t() {
        stats = getenv("MOE_CACHE_STATS") != nullptr;
        fprintf(stderr, "moe-cache: plugin v0.3.3 (ggml backend API %d), running with ggml %s (%s)\n", GGML_BACKEND_API_VERSION, ggml_version(), ggml_commit());
        if (!moe_cache_tested_ggml(ggml_version())) {
            fprintf(stderr, "moe-cache: warning: tested with ggml %s only (this is %s); if the server crashes or the plugin misbehaves, rebuild it against your llama.cpp headers (or set MOE_CACHE_DISABLE=1)\n", MOE_CACHE_TESTED_GGML, ggml_version());
        }
        plat::env_utf8("MOE_CACHE_PROFILE", profile_path);     // paths: UTF-8 (Windows getenv gives the ANSI code page)
        plat::env_utf8("MOE_CACHE_WARM", warm_path);
        if (const char * v = getenv("MOE_CACHE_PRELOAD_FRAC")) {
            const double f = atof(v);
            if (f >= 0.0 && f <= 1.0) {          // also false for nan: a bad value keeps the default instead of loading past the budget (audit N6)
                preload_frac = f;
            } else {
                fprintf(stderr, "moe-cache: MOE_CACHE_PRELOAD_FRAC=%s ignored (a fraction 0..1)\n", v);
            }
        }
        if (const char * v = getenv("MOE_CACHE_PROFILE_DECAY")) {
            profile_decay = atof(v);
        }
        if (const char * v = getenv("MOE_CACHE_LAYERS")) {
            int a = 0, b = 0;
            if (sscanf(v, "%d-%d", &a, &b) == 2) {
                lo = a;
                hi = b;
            }
        }
        const char * gib = getenv("MOE_CACHE_SIZE_GIB");
        std::string file_s;
        const char * file = plat::env_utf8("MOE_CACHE_GGUF", file_s) ? file_s.c_str() : nullptr;
        if (!gib) {
            fprintf(stderr, "moe-cache: pass-through mode (no cache), layers %s\n", layer_range().c_str());
            return;
        }
        auto fail = [&](const std::string & m) {
            fprintf(stderr, "moe-cache: %s - plugin disabled, the model loads the normal way\n", m.c_str());
            disabled = true;
        };
        if (getenv("MOE_CACHE_DISABLE")) {
            fail("MOE_CACHE_DISABLE is set");
            return;
        }
        {
            const int api = moe_cache_api_of_ggml(ggml_version());
            if (api > 0 && api != GGML_BACKEND_API_VERSION) {
                fail("this plugin was built for ggml backend API " + std::to_string(GGML_BACKEND_API_VERSION) + " but the server uses ggml " + ggml_version() +
                     " (API " + std::to_string(api) + "): use the plugin build that matches your llama.cpp (see docs/QUICKSTART.md)");
                return;
            }
            const std::string abi = ggml_abi_mismatch();
            if (!abi.empty()) {
                fail(std::string("the running ggml ") + ggml_version() + " does not match the ggml headers this plugin was built with (" + abi +
                     "): use the plugin build that matches your llama.cpp (see docs/QUICKSTART.md)");
                return;
            }
        }
        if (!file) {
            fail("MOE_CACHE_GGUF is not set");
            return;
        }
#ifdef _WIN32
        // MOE_CACHE_WIN_BUFFERED=1 (Windows only; idea I38; the default without the file views, below): read the experts through the Windows file cache instead of FILE_FLAG_NO_BUFFERING.
        // Unbuffered reads never warm the file cache, so every start reads the SSD, while stock llama-server (mmap) starts warm from it (Flash-Next on
        // a 32 GB PC: 3-5 tok/s first session vs 13-16 warm stock). Same aligned offsets and lengths, so the same bytes. An expert then sits twice in
        // RAM (cache + file cache), but the file-cache copy moves to the standby list, which Windows counts as available and gives back on demand:
        // the budget (available + this process's private memory) does not shrink for it.
        // =2: also FILE_FLAG_RANDOM_ACCESS (no read-ahead around each expert; A/B switch)
        if (const char * v = getenv("MOE_CACHE_WIN_BUFFERED")) {
            win_buffered = strcmp(v, "0") != 0;
            win_random = strcmp(v, "2") == 0;
        }
        if (win_buffered) {
            plat::count_file_cache() = true;
            fprintf(stderr, "moe-cache: MOE_CACHE_WIN_BUFFERED: experts are read through the Windows file cache%s; its %.2f GiB in memory count as available\n",
                    win_random ? " (random access, no read-ahead)" : "", plat::file_cache_ws() / 1073741824.0);
        }
#endif
        // split models: name-00001-of-0000N.gguf -> open all parts
        std::vector<std::string> shards = { file };
        {
            std::string f = file;
            int idx = 0, cnt = 0;
            const size_t of = f.rfind("-of-");
            if (of != std::string::npos && of >= 6) {
                const size_t start = of - 6;   // "-00001-of-00003.gguf"
                if (sscanf(f.c_str() + start, "-%d-of-%d.gguf", &idx, &cnt) == 2 && cnt > 1 && cnt < 1000) {
                    shards.clear();
                    for (int i = 1; i <= cnt; i++) {
                        char suffix[64];
                        snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", i, cnt);
                        shards.push_back(f.substr(0, start) + suffix);
                    }
                }
            }
        }
        for (size_t si = 0; si < shards.size(); si++) {
            gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
            gguf_context * g = gguf_init_from_file(shards[si].c_str(), ip);
            if (!g) {
                fail("cannot read " + shards[si] + " (MOE_CACHE_GGUF must be the model file llama-server loads)");
                return;
            }
            const size_t data_off = gguf_get_data_offset(g);
            const int64_t n = gguf_get_n_tensors(g);
            for (int64_t i = 0; i < n; i++) {
                const char * nm = gguf_get_tensor_name(g, i);
                file_tensors[nm] = { data_off + gguf_get_tensor_offset(g, i), gguf_get_tensor_size(g, i) };
                if (strstr(nm, "ffn_gate_inp") || strstr(nm, "ffn_norm") || strstr(nm, "post_attention_norm")) {
                    tmeta[nm] = { (int) gguf_get_tensor_type(g, i), data_off + gguf_get_tensor_offset(g, i), gguf_get_tensor_size(g, i), (int) si };
                }
                file_shard[nm] = (int) si;
            }
            gguf_free(g);
            plat::file_t f = plat::file_open(shards[si].c_str(), !win_buffered, win_random);
#ifndef _WIN32
            if (!f.ok() && !win_buffered && errno == EINVAL) {
                // a file system without O_DIRECT (ntfs-3g and other FUSE mounts, e.g. a dual-boot model folder): the same aligned reads work
                // through the page cache; before, the whole plugin was turned off (audit 2026-10-29 N8)
                f = plat::file_open(shards[si].c_str(), false, win_random);
                if (f.ok()) {
                    fprintf(stderr, "moe-cache: %s: this file system refuses O_DIRECT; reading through the page cache instead\n", shards[si].c_str());
                }
            }
#endif
            if (!f.ok()) {
                fail("cannot open " + shards[si] + (win_buffered ? std::string() : std::string(" with O_DIRECT")));
                return;
            }
            fds.push_back(f);
            shard_paths.push_back(shards[si]);
            fds_buf.push_back(plat::file_t());
        }
        pack_try_open();
#ifdef _WIN32
        // Windows default (idea I47; MOE_CACHE_WIN_VIEW=0 goes back to private memory): keep the cached experts in read-only views of the model files.
        // PC cold ABBA vs stock at launcher defaults (RESULTS 59): Flash-Next IQ3_XXS 19.86 vs 15.14 tok/s, IQ2_XS 21.94 vs 14.89.
        // Private memory is charged to Windows commit (RAM + page file), and so are Vulkan's GPU allocations: on the PC that held the cache at
        // 8-11 GiB, resized hundreds of times per request, while stock llama.cpp's mmap used ~27 GB of file pages that cost no commit (RESULTS 59).
        // File-view pages cost no commit; a hard working-set minimum (update_ws_min) keeps the resident ones in RAM. Which experts are resident,
        // prefetch and the executor work exactly as on Linux; only loading (touch the view instead of reading into private pages) and eviction
        // (drop the pages from the working set; they stay in the file cache) differ.
        win_view = true;
        if (const char * v = getenv("MOE_CACHE_WIN_VIEW")) {
            win_view = strcmp(v, "0") != 0;
        }
        if (pack_on) {
            win_view = false;       // the pack's reads fill private memory
        }
        if (win_view) {
            for (const std::string & sp : shard_paths) {
                size_t n = 0;
                const char * vb = plat::file_view_map(sp.c_str(), &n);
                if (!vb) {
                    fprintf(stderr, "moe-cache: file-view cache: cannot map %s read-only (Windows error %lu): the cache uses private memory\n", sp.c_str(), GetLastError());
                    break;
                }
                views.push_back(vb);
                view_sizes.push_back(n);
            }
            win_view = views.size() == shard_paths.size();
            if (win_view) {
                plat::count_file_cache() = false;      // the views' pages are counted as the cache's own (view_bytes), not as available file cache
                fprintf(stderr, "moe-cache: experts are kept in read-only views of the model file (%zu), held in RAM by a hard "
                                "working-set minimum; they use no commit (RAM + page file; MOE_CACHE_WIN_VIEW=0: private memory)\n", views.size());
            }
        }
        // Private memory (MOE_CACHE_WIN_VIEW=0, the expert pack, a failed view map): read through the Windows file cache by default too, as the views do.
        // Flash-Next Q2_0 on the PC, ABBA: reader speed 0.26 -> 0.84 GB/s, decode +39 %, prompts 2x, text identical (RESULTS 75). MOE_CACHE_WIN_BUFFERED=0: unbuffered.
        if (!win_view && !win_buffered && !getenv("MOE_CACHE_WIN_BUFFERED")) {
            win_buffered = true;
            plat::count_file_cache() = true;
            for (size_t si = 0; si < fds.size(); si++) {
                plat::file_t f = plat::file_open(shard_paths[si].c_str(), false, win_random);
                if (f.ok()) {
                    CloseHandle(fds[si].h);
                    fds[si] = f;
                }
            }
            fprintf(stderr, "moe-cache: private-memory cache: experts are read through the Windows file cache (MOE_CACHE_WIN_BUFFERED=0: unbuffered)\n");
        }
#elif defined(__linux__)
        // Linux default too (same as Windows; MOE_CACHE_LINUX_VIEW=0 goes back to private memory + O_DIRECT reads): the same file-view cache.
        // Load = MADV_WILLNEED + MADV_POPULATE_READ, evict = MADV_COLD + MADV_DONTNEED (the pages stay in the page cache). A cgroup charges page
        // cache like private memory, so the budget (tune(): RssAnon + view_bytes) is unchanged; no mlock (RLIMIT_MEMLOCK), the kernel may reclaim
        // resident experts. Laptop ABBA (Qwen3.6, 10 GB cgroup): steady decode +4.7 % (every pair +0.8 to +8.4 %), first request +49 %, tokens
        // identical (RESULTS 66); kept under the owner's rule of 2026-10-28 (every repeatable exact gain).
        win_view = true;
        if (const char * v = getenv("MOE_CACHE_LINUX_VIEW")) {
            win_view = strcmp(v, "0") != 0;
        }
        if (pack_on) {
            win_view = false;       // the pack's reads fill private memory
        }
        if (win_view) {
            for (const std::string & sp : shard_paths) {
                size_t n = 0;
                const char * vb = plat::file_view_map(sp.c_str(), &n);
                if (!vb) {
                    fprintf(stderr, "moe-cache: MOE_CACHE_LINUX_VIEW: cannot map %s read-only (%s): the cache uses private memory\n", sp.c_str(), strerror(errno));
                    break;
                }
                views.push_back(vb);
                view_sizes.push_back(n);
            }
            win_view = views.size() == shard_paths.size();
            if (win_view) {
                fprintf(stderr, "moe-cache: experts are kept in read-only mappings of the model file (%zu, page cache; MOE_CACHE_LINUX_VIEW=0: private memory)\n", views.size());
            }
        }
#endif
        if (const char * v = getenv("MOE_CACHE_FAULT_EVERY")) {
            fault_every = atoi(v);
        }
        if (const char * v = getenv("MOE_CACHE_TEST_READ_DELAY_US")) {
            test_read_delay_us = std::max(0, atoi(v));
        }
        if (const char * v = getenv("MOE_CACHE_TEST_POISON_READS")) {
            test_poison = strcmp(v, "0") != 0;
        }
        const char * direct_env = getenv("MOE_CACHE_DIRECT");   // 1 / 0 forces it; unset = on when the model does not fit (decided below)
        direct_ok = direct_env && strcmp(direct_env, "0") != 0;
        if (const char * v = getenv("MOE_CACHE_GPU_MIN_TOKENS")) {
            gpu_min_tokens = atoi(v);
            if (gpu_min_tokens == 1) {
                // 1-token batches on the GPU produce one repeated token (Granite 3.1, T25; 2+ tokens are fine) and would upload whole layers for every
                // decoded token anyway: the executor handles single tokens
                fprintf(stderr, "moe-cache: MOE_CACHE_GPU_MIN_TOKENS=1 is not supported (wrong output for single-token batches): using 2\n");
                gpu_min_tokens = 2;
            }
        }
        // the GPU llama.cpp offloads to: --device none means none; --device A,B the first GPU of that list; else its default choice below
        const mcargs::server_args & sa = server_args();
        for (size_t k = 0; k < sa.devices.size() && gpu_min_tokens > 0 && !gpu_dev; k++) {
            ggml_backend_dev_t d = ggml_backend_dev_by_name(sa.devices[k].c_str());
            const auto t = d ? ggml_backend_dev_type(d) : GGML_BACKEND_DEVICE_TYPE_CPU;
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                gpu_dev = d;
            }
        }
        for (size_t i = 0; i < ggml_backend_dev_count() && gpu_min_tokens > 0 && !sa.device_none && sa.devices.empty(); i++) {
            // a discrete GPU first, else an integrated one (ggml-vulkan reports APUs as IGPU; stock llama.cpp offloads to them too).
            // Taking only GPU left the Vega 8 unused: prefill 363 vs stock 693 tok/s (mini PC, T7); with the iGPU 569-597
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const auto t = ggml_backend_dev_type(d);
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU) {
                gpu_dev = d;
                break;
            }
            if (t == GGML_BACKEND_DEVICE_TYPE_IGPU && !gpu_dev) {
                gpu_dev = d;
            }
        }
        {
            const char * rp = getenv("MOE_CACHE_REPACK");
            const std::string mode = rp ? rp : "auto";
            bool want = mode == "on";
            if (mode == "auto") {
                // do what stock llama.cpp does, so results stay identical. Its CPU buffer types are tried in the order ACCEL (this plugin), the
                // host buffer of its first GPU, CPU_REPACK, CPU: with a GPU in use its CPU weights land in the GPU's host buffer and are not
                // repacked; with no GPU in use (none loaded, or --device none), or with --no-host (no host buffer), they are repacked
                bool gpu = false;
                for (size_t i = 0; i < ggml_backend_dev_count() && !sa.device_none; i++) {
                    const auto t = ggml_backend_dev_type(ggml_backend_dev_get(i));
                    gpu = gpu || t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU;
                }
                want = (!gpu || sa.no_host) && !server_repack_disabled();
            }
            if (want) {
                ggml_backend_dev_t cd = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                auto fn = cd ? (ggml_backend_buffer_type_t * (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cd), "ggml_backend_dev_get_extra_bufts") : nullptr;
                for (ggml_backend_buffer_type_t * b = fn ? fn(cd) : nullptr; b && *b; ++b) {
                    if (strcmp(ggml_backend_buft_name(*b), "CPU_REPACK") == 0) {
                        repack_buft = *b;
                    }
                }
                if (repack_buft) {
                    repack_buf = ggml_backend_buft_alloc_buffer(repack_buft, 64);
                    ggml_backend_buffer_i empty;
                    memset(&empty, 0, sizeof(empty));
                    repack_tag = buffer_init_padded(repack_buft, empty, nullptr, 0);
                }
                fprintf(stderr, "moe-cache: CPU weight repacking %s\n", repack_buf && repack_tag ? "on (faster CPU kernels)" : "requested but not available in this llama.cpp");
                if (!repack_buf || !repack_tag) {
                    repack_buf = nullptr;
                    repack_tag = nullptr;
                }
            }
        }   // off by default: reading straight into place measured no faster than the bounce buffer
        if (strcmp(gib, "auto") == 0) {
            auto_mode = true;
            limit_bytes = read_mem_limit();
            if (ram_cap() != (size_t) -1) {
                fprintf(stderr, "moe-cache: memory cap %.1f GiB (%.1f GB) for the whole server (%s; MOE_CACHE_MAX_RAM_GB or the launcher's --max-ram changes it)\n",
                        ram_cap() / 1073741824.0, ram_cap() / 1e9, getenv("MOE_CACHE_MAX_RAM_GB") ? "set" : "physical RAM minus max(4 GiB, 12.5 %)");
            }
            if (const char * v = getenv("MOE_CACHE_RAM_GIB")) {
                size_t x = 0;
                if (parse_gib("MOE_CACHE_RAM_GIB", v, x) && x > 0) {
                    limit_bytes = std::min(x, ram_cap());      // --ram cannot go above the hard cap either
                }
            }
            if (const char * v = getenv("MOE_CACHE_MARGIN_GIB")) {
                parse_gib("MOE_CACHE_MARGIN_GIB", v, margin);
            }
            // llama-server keeps saved prompt states in host RAM (--cache-ram, MiB): that memory is outside this cache's control and arrives in bursts
            // when a new request starts, so it is added to the margin (a 24 GB limit was OOM-killed at 128k context with the 8 GiB default)
            if (!getenv("MOE_CACHE_MARGIN_GIB")) {
                long cram = -1;
                const char * from = "the command line";
                const char * reserve = getenv("MOE_CACHE_HOST_RESERVE_MIB");   // set by moe-cache-server; the command line below is only a fallback
                if (reserve) {
                    cram = atol(reserve);
                    from = "moe-cache-server (MOE_CACHE_HOST_RESERVE_MIB)";
                } else if (const char * v = getenv("LLAMA_ARG_CACHE_RAM")) {
                    cram = atol(v);
                    from = "LLAMA_ARG_CACHE_RAM";
                }
                if (!reserve) {
                    const std::vector<std::string> argv_ = plat::command_line();
                    for (size_t k = 0; k + 1 < argv_.size(); k++) {
                        if (argv_[k] == "--cache-ram" || argv_[k] == "-cram") {
                            cram = atol(argv_[k + 1].c_str());
                        }
                    }
                }
                if (cram > 0) {
                    const size_t extra = std::min<size_t>((size_t) cram << 20, limit_bytes / 3);
                    margin += extra;
                    fprintf(stderr, "moe-cache: llama-server's prompt cache (--cache-ram %ld MiB, from %s) is kept free: margin now %.2f GiB\n", cram, from, margin / 1073741824.0);
                }
            }
            if (!direct_env) {
                // the model does not fit: SSD reads dominate, and reading straight into place (no bounce buffer, no page re-zeroing) was +22 % on gpt-oss-120b
                size_t total = 0;
                for (const auto & sp : shard_paths) {
                    std::error_code ec;
#ifdef _WIN32
                    total += (size_t) std::filesystem::file_size(std::filesystem::path(plat::widen(sp.c_str())), ec);   // a std::string path is ANSI there
#else
                    total += (size_t) std::filesystem::file_size(sp, ec);
#endif
                }
                direct_ok = total > limit_bytes * 0.85;
            }
            const size_t guess = (size_t) 7 << 29;   // 3.5 GiB for everything that is not expert cache, until measured
            budget = limit_bytes > margin + guess + ((size_t) 1 << 30) ? limit_bytes - margin - guess : (size_t) 1 << 30;
            fprintf(stderr, "moe-cache: auto cache size, memory limit %.2f GiB, margin %.2f GiB, start budget %.2f GiB\n", limit_bytes / 1073741824.0, margin / 1073741824.0, budget / 1073741824.0);
        } else {
            if (!parse_gib("MOE_CACHE_SIZE_GIB", gib, budget) || budget == 0) {
                fail(std::string("MOE_CACHE_SIZE_GIB='") + gib + "' is not a valid size");
                return;
            }
        }
        cache_on = true;
        {
            std::vector<std::string> names;
            for (const auto & kv : file_tensors) {
                names.push_back(kv.first);
            }
            std::sort(names.begin(), names.end());
            uint64_t h = 1469598103934665603ull;
            for (const auto & nm : names) {
                const auto & v = file_tensors[nm];
                for (unsigned char ch : nm) { h = (h ^ ch) * 1099511628211ull; }
                h = (h ^ v.second) * 1099511628211ull;
            }
            fingerprint = h;
        }
        const std::string & src = !profile_path.empty() ? profile_path : warm_path;
        if (!src.empty()) {
            uint64_t fp = 0;
            if (load_usage(src.c_str(), learned, &fp)) {
                if (fp != 0 && fp != fingerprint) {
                    fprintf(stderr, "moe-cache: profile %s belongs to another model, ignored\n", src.c_str());
                    learned = usage_t();
                } else {
                    preload_on = true;
                    fprintf(stderr, "moe-cache: profile %s loaded (%d layers x %d experts)%s\n", src.c_str(), learned.n_layers, learned.n_exp, fp ? "" : " (no model fingerprint)");
                }
            } else {
                fprintf(stderr, "moe-cache: no profile at %s yet%s\n", src.c_str(), profile_path.empty() ? "" : " (it will be created at exit)");
            }
        }
        int n_io = 5;   // helper threads; the calling thread reads too, so MOE_CACHE_IO_THREADS=N means N parallel readers in total
        if (const char * v = getenv("MOE_CACHE_IO_THREADS")) {
            n_io = std::max(0, std::min(63, atoi(v) - 1));
        }
        for (int i = 0; i < n_io; i++) {
            workers.emplace_back([this, i] { worker_main(i + 1); });
        }
        fprintf(stderr, "moe-cache: cache mode, %.2f GiB, %zu tensors in %s, layers %s\n", budget / 1073741824.0, file_tensors.size(), file, layer_range().c_str());
        if (const char * ex = getenv("MOE_CACHE_EXEC")) {
            exec_on = strcmp(ex, "0") != 0;
            exec_force = strcmp(ex, "2") == 0;        // tests: never step aside, even when everything is resident and bandwidth-bound
        }
        if (const char * v = getenv("MOE_CACHE_EXEC_FUSE")) {
            exec_fuse = atoi(v) != 0;
        }
        if (const char * v = getenv("MOE_CACHE_EXEC_CHUNK")) {
            exec_chunk = std::max(1, std::min(4096, atoi(v)));
        }
        {
            bool cl = false;
            if (const char * v = getenv("MOE_CACHE_POLICY")) {
                if (strcmp(v, "classlru") == 0) {
                    cl = true;
                } else if (*v && strcmp(v, "lru") != 0) {
                    fprintf(stderr, "moe-cache: warning: MOE_CACHE_POLICY=%s is not known (lru, classlru): using lru\n", v);
                }
            }
            if (const char * v = getenv("MOE_CACHE_IDLE_REBALANCE")) {
                rb_on = strcmp(v, "0") != 0;
            }
            if (const char * v = getenv("MOE_CACHE_IDLE_REBALANCE_MS")) {
                rb_idle_ms = std::max(0, atoi(v));
                rb_on = rb_on && rb_idle_ms > 0;      // 0 = off
            }
            if (const char * v = getenv("MOE_CACHE_IDLE_REBALANCE_MB")) {
                rb_cap_bytes = (size_t) std::max(1, atoi(v)) << 20;
            }
            if (const char * v = getenv("MOE_CACHE_IDLE_REBALANCE_GAIN")) {
                rb_gain = std::max(1.0, atof(v));
            }
            const char * kc = getenv("MOE_CACHE_CLASSES");
            const char * hl = getenv("MOE_CACHE_HALFLIFE");
            set_policy(cl, kc ? atoi(kc) : 4, hl ? atof(hl) : 128.0);
            if (rb_on) {
                rb_th = std::thread([this] { rb_main(); });
                fprintf(stderr, "moe-cache: idle rebalance on (after %d ms without a graph, at most %zu MB per idle period)\n", rb_idle_ms, rb_cap_bytes >> 20);
            }
        }
        if (const char * v = getenv("MOE_CACHE_ROUTE")) {
            route_mode = strcmp(v, "skip") == 0 ? 1 : (strcmp(v, "subst") == 0 ? 2 : 0);
        }
        if (const char * v = getenv("MOE_CACHE_FEWER")) {
            fewer = std::max(0, std::min(64, atoi(v)));
        }
        if (const char * v = getenv("MOE_CACHE_ROUTE_KEEP")) {
            route_keep = atoi(v);
        }
        if (const char * v = getenv("MOE_CACHE_ROUTE_DROP")) {
            route_drop = std::max(0, atoi(v));
        }
        if (const char * v = getenv("MOE_CACHE_SCAN_TOKENS")) {
            scan_tokens = std::max(0, atoi(v));
        }
        if (const char * v = getenv("MOE_CACHE_EXEC_CHECK")) {
            exec_check = strcmp(v, "0") != 0;
        }
        if (const char * v = getenv("MOE_CACHE_EXEC_MAXCOL")) {
            exec_maxcol = std::max(1, std::min(EXEC_MAXCOL, atoi(v)));
        }
        if (const char * v = getenv("MOE_CACHE_EXEC_THREADS")) {
            exec_threads = std::max(0, std::min(64, atoi(v)));
        }
        if (const char * v = getenv("MOE_CACHE_EXEC_ECORES")) {
            exec_ecores = std::max(0, std::min(32, atoi(v)));
        }
        if (const char * v = getenv("MOE_CACHE_EXEC_READERS")) {
            exec_readers = std::max(1, std::min(16, atoi(v)));
        }
        if (exec_on) {
            for (int i = 0; i < exec_readers; i++) {
                readers.emplace_back([this] { reader_main(); });
            }
            fprintf(stderr, "moe-cache: expert executor on (%d reader threads, %d extra efficiency-core workers)\n", exec_readers, exec_ecores);
        }
        if (const char * sb = getenv("MOE_CACHE_SIBLINGS")) {
            siblings = strcmp(sb, "0") != 0;
        }
        std::string trace_path;
        if (plat::env_utf8("MOE_CACHE_TRACE_IDS", trace_path)) {
            trace_f = fopen_private(trace_path.c_str());
        }
        if (const char * v = getenv("MOE_CACHE_COALESCE_MIB")) {
            coalesce_bytes = (size_t) std::max(0, atoi(v)) << 20;
        }
        if (const char * v = getenv("MOE_CACHE_LOOKAHEAD")) {
            lookahead = strcmp(v, "0") != 0;
            la_exec = strcmp(v, "2") == 0;
            if (const char * g = getenv("MOE_CACHE_LOOKAHEAD_GUESS")) {
                la_guess = atoi(g);
            }
            if (const char * g = getenv("MOE_CACHE_LOOKAHEAD_CAP")) {
                la_cap = std::max(1, atoi(g));
            }
            if (const char * g = getenv("MOE_CACHE_LOOKAHEAD_ECORES")) {
                la_ecores = strcmp(g, "0") != 0;
            }
            if (lookahead) {
                int nt = 4;
                if (const char * t = getenv("MOE_CACHE_LOOKAHEAD_THREADS")) {
                    nt = std::max(1, std::min(32, atoi(t)));
                }
                for (int i = 0; i < nt; i++) {
                    pf_threads.emplace_back([this] { pf_main(); });
                }
                fprintf(stderr, "moe-cache: lookahead prefetch on (%d helper threads)\n", nt);
            }
        }
#ifdef __linux__
        if (getenv("MOE_CACHE_STATS_SIGNAL")) {
            if (pipe(g_stats_pipe) == 0) {
                struct sigaction sa;
                memset(&sa, 0, sizeof(sa));
                sa.sa_handler = [](int) { const char c = 1; const ssize_t r = write(g_stats_pipe[1], &c, 1); (void) r; };   // async-signal-safe: only write()
                sa.sa_flags = SA_RESTART;
                sigaction(SIGUSR1, &sa, nullptr);
                stats_th = std::thread([this] { char c; while (read(g_stats_pipe[0], &c, 1) == 1 && !g_stats_stop) { print_snapshot(); } });   // joined in stop_threads
            }
        }
        // idle release (N3): the launcher's on-demand listener sends SIGUSR2 when the server has been idle for a while
        if (cache_on && pipe(g_idle_pipe) == 0) {
            struct sigaction sa;
            memset(&sa, 0, sizeof(sa));
            sa.sa_handler = [](int) { const char c = 1; const ssize_t r = write(g_idle_pipe[1], &c, 1); (void) r; };    // async-signal-safe: only write()
            sa.sa_flags = SA_RESTART;
            sigaction(SIGUSR2, &sa, nullptr);
            idle_th = std::thread([this] { char c; while (read(g_idle_pipe[0], &c, 1) == 1 && !g_stats_stop) { idle_release(); } });
        }
#elif defined(_WIN32)
        // no signals: setting the named event Local\moe-cache-stats-<pid> prints one snapshot (tools/windows_test.ps1 does it before stopping the
        // server, which on Windows is killed, so the summary at exit never appears)
        if (getenv("MOE_CACHE_STATS_SIGNAL")) {
            char name[64];
            snprintf(name, sizeof(name), "Local\\moe-cache-stats-%lu", plat::process_id());
            g_stats_event = CreateEventA(nullptr, FALSE, FALSE, name);
            if (g_stats_event) {
                stats_th = std::thread([this] { while (WaitForSingleObject(g_stats_event, INFINITE) == WAIT_OBJECT_0 && !g_stats_stop) { print_snapshot(); } });
            }
        }
        // Windows stops a server by killing it, so the profile save in the destructor never runs: the launcher sets this event first
        // (learn, server exit); without it only the save every 100,000 groups is kept
        {
            char name[64];
            snprintf(name, sizeof(name), "Local\\moe-cache-save-%lu", plat::process_id());
            g_save_event = CreateEventA(nullptr, FALSE, FALSE, name);
            if (g_save_event) {
                save_th = std::thread([this] {
                    while (WaitForSingleObject(g_save_event, INFINITE) == WAIT_OBJECT_0 && !g_stats_stop) {
                        if (save_profile()) {
                            fprintf(stderr, "moe-cache: usage profile saved (on request)\n");
                        }
                    }
                });
            }
        }
        // idle release (N3): the launcher's on-demand listener sets this event when the server has been idle for a while
        if (cache_on) {
            char name[64];
            snprintf(name, sizeof(name), "Local\\moe-cache-idle-%lu", plat::process_id());
            g_idle_event = CreateEventA(nullptr, FALSE, FALSE, name);
            if (g_idle_event) {
                idle_th = std::thread([this] { while (WaitForSingleObject(g_idle_event, INFINITE) == WAIT_OBJECT_0 && !g_stats_stop) { idle_release(); } });
            }
        }
        if (cache_on) {
            install_crash_report(this);
        }
#endif
    }

    // the rest of state_t lives in three files, included here inside the class (idea I11; the compiled code is byte-identical to the single file)
#include "mc_state_io.inl"
#include "mc_state_profile.inl"
#include "mc_state_lru.inl"
#include "mc_state_pack.inl"
#include "mc_state_rebalance.inl"
};

// The state lives on the heap; a static guard ends it at exit. Normally (threads alive) it is deleted: threads joined, profile saved. When the
// process is exiting on Windows, the other threads are already gone, or (both systems) exit came while a graph held the cache lock, the state is
// only asked to report (see exit_without_threads).
struct state_exit_t {
    state_t * s;
    ~state_exit_t() {
        if (plat::process_exiting()) {
            s->exit_without_threads();
        } else if (!s->mu.try_lock()) {
            // exit() called while a graph holds the cache lock (Linux: llama-server's second Ctrl+C calls exit(1) from its signal handler, in the
            // middle of a prompt). Joining the readers would wait for ever (they need the lock) and the save would lock it again: leave the state
            // as the Windows exit does; the system takes the memory back (audit 2026-10-29 N2)
            fprintf(stderr, "moe-cache: exit during a computation: threads left to the system, usage profile not saved this time\n");
            s->exit_without_threads();
        } else {
            s->mu.unlock();
            delete s;
        }
    }
};

state_t & S() {
    static state_t * s = new state_t;
    static state_exit_t guard{ s };
    return *s;
}

#ifdef _WIN32
// Windows crash report: on an access violation, one stderr line with the faulting code (module+offset) and data address and, when the address is
// inside an expert tensor, which expert and its cache state. The Windows event log gives only the module offset (PC 2026-10-27: 0xC0000005 in
// ggml-cpu-zen4.dll at a prompt of 11-16 tokens), which cannot tell an evicted expert from anything else. A first-chance handler reports faults in
// expert memory (before anyone else sees them), the unhandled-exception filter any crash; both pass the exception on unchanged.
state_t * g_crash_state = nullptr;
std::atomic<int> g_crash_said{0};
LPTOP_LEVEL_EXCEPTION_FILTER g_prev_filter = nullptr;

bool crash_report(EXCEPTION_POINTERS * ep, bool experts_only) {
    const EXCEPTION_RECORD * er = ep ? ep->ExceptionRecord : nullptr;
    state_t * s = g_crash_state;
    if (!er || !s || er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || er->NumberParameters < 2) {
        return false;
    }
    const uintptr_t addr = (uintptr_t) er->ExceptionInformation[1];
    const tinfo_t * hit = nullptr;
    for (const auto & t : s->tensors) {
        // a forgotten tensor (its buffer was freed) has data == nullptr and no experts: it would claim every address below one page
        if (t && t->data && t->n_expert > 0 && addr >= (uintptr_t) t->data && addr < (uintptr_t) t->data + t->nb2 * (size_t) t->n_expert + PAGE) {
            hit = t.get();
            break;
        }
    }
    if ((experts_only && !hit) || g_crash_said.exchange(1) != 0) {
        return false;
    }
    const uintptr_t pc = (uintptr_t) er->ExceptionAddress;
    HMODULE mod = nullptr;
    char mname[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) pc, &mod)) {
        GetModuleFileNameA(mod, mname, MAX_PATH);
    }
    const char * base = std::max(strrchr(mname, '\\'), strrchr(mname, '/'));
    base = base ? base + 1 : mname;
    const ULONG_PTR kind = er->ExceptionInformation[0];
    char buf[4096];
    int n = snprintf(buf, sizeof(buf), "moe-cache: crash: access violation at %s+0x%llx, %s address 0x%llx", base,
                     (unsigned long long) (pc - (uintptr_t) mod), kind == 0 ? "reading" : kind == 1 ? "writing" : "executing", (unsigned long long) addr);
    if (hit) {
        const size_t rel = addr - (uintptr_t) hit->data;
        const int64_t e = std::max<int64_t>(0, std::min<int64_t>((int64_t) (rel / hit->nb2), hit->n_expert - 1));
        const node_t & nd = s->nodes[(size_t) (hit->node0 + e)];
        n += snprintf(buf + n, sizeof(buf) - (size_t) n, " = %s expert %lld (byte %zu of %zu): resident=%d loading=%d mapped=%d epoch=%llu (now %llu, protected from %llu); "
                      "cache %.2f of %.2f GiB, %llu evictions", hit->name.c_str(), (long long) e, rel - (size_t) e * hit->nb2, hit->nb2, (int) nd.resident,
                      (int) nd.loading, (int) nd.mapped, (unsigned long long) nd.epoch, (unsigned long long) s->epoch,
                      (unsigned long long) (s->protect_epoch == UINT64_MAX ? 0 : s->protect_epoch), s->resident_bytes / 1073741824.0,
                      s->budget / 1073741824.0, (unsigned long long) s->evictions);
    } else {
        n += snprintf(buf + n, sizeof(buf) - (size_t) n, " (not in the expert cache)");
    }
    // was the plugin handing a graph to the CPU backend? If so, its nodes without data (a NULL base + offset write points there)
    if (ggml_cgraph * g = g_cpu_graph.load()) {
        const int nn = ggml_graph_n_nodes(g);
        int nulls = 0;
        n += snprintf(buf + n, sizeof(buf) - (size_t) n, "\nmoe-cache: crash: inside a plugin graph handed to the CPU backend (%d nodes)", nn);
        for (int i = 0; i < nn && n < (int) sizeof(buf) - 200; i++) {
            const ggml_tensor * t = ggml_graph_node(g, i);
            const bool null_src = (t->src[0] && !t->src[0]->data) || (t->src[1] && !t->src[1]->data);
            if (t->op != GGML_OP_NONE && (!t->data || null_src) && nulls++ < 8) {
                n += snprintf(buf + n, sizeof(buf) - (size_t) n, "; node %d %s (%s) data=%p src0=%p src1=%p ne=%lld,%lld,%lld", i, t->name, ggml_op_desc(t),
                              t->data, t->src[0] ? t->src[0]->data : nullptr, t->src[1] ? t->src[1]->data : nullptr,
                              (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
            }
        }
        if (!nulls) {
            n += snprintf(buf + n, sizeof(buf) - (size_t) n, "; no node or source without data");
        }
    } else {
        n += snprintf(buf + n, sizeof(buf) - (size_t) n, "\nmoe-cache: crash: not inside a graph the plugin handed to the CPU backend");
    }
    // inside executor work (a pool worker or the compute thread's share): which op, task, row, column and output it was computing
    if (const exec_where_t & wh = g_exec_where; wh.op) {
        n += snprintf(buf + n, sizeof(buf) - (size_t) n, "\nmoe-cache: crash: inside the executor: op %s, task %p", wh.op, (const void *) wh.task);
        if (wh.task) {
            n = std::min<int>(n, (int) sizeof(buf) - 2);
            n += snprintf(buf + n, sizeof(buf) - (size_t) n, " (weights %p, %d columns, %lld rows), row %lld, column %d, dst %p",
                          (const void *) wh.task->w, wh.task->ncol, (long long) wh.task->rows, (long long) wh.r, wh.k, (const void *) wh.dst);
        }
    }
    // what is known so far goes out now: a damaged stack can make the walk below fault inside this handler, and then nothing would be printed
    {
        n = std::min<int>(n, (int) sizeof(buf) - 2);
        buf[n++] = '\n';
        DWORD w0 = 0;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), buf, (DWORD) n, &w0, nullptr);
        n = 0;
    }
    // stack of the faulting thread, module+offset per frame (x64 unwind tables)
    {
        CONTEXT ctx = *ep->ContextRecord;
        n += snprintf(buf + n, sizeof(buf) - (size_t) n, "moe-cache: crash: stack");
        for (int f = 0; f < 20 && ctx.Rip && n < (int) sizeof(buf) - 120; f++) {
            HMODULE fm = nullptr;
            char fname[MAX_PATH] = "?";
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) ctx.Rip, &fm)) {
                GetModuleFileNameA(fm, fname, MAX_PATH);
            }
            const char * fb = std::max(strrchr(fname, '\\'), strrchr(fname, '/'));
            n += snprintf(buf + n, sizeof(buf) - (size_t) n, " %s+0x%llx", fb ? fb + 1 : fname, (unsigned long long) (ctx.Rip - (DWORD64) (uintptr_t) fm));
            DWORD64 image = 0;
            PRUNTIME_FUNCTION rf = RtlLookupFunctionEntry(ctx.Rip, &image, nullptr);
            if (rf) {
                void * hd = nullptr;
                DWORD64 est = 0;
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, ctx.Rip, rf, &ctx, &hd, &est, nullptr);
            } else {
                if (IsBadReadPtr((const void *) (uintptr_t) ctx.Rsp, 8)) {
                    break;
                }
                ctx.Rip = *(const DWORD64 *) (uintptr_t) ctx.Rsp;     // leaf function: the return address is on top of the stack
                ctx.Rsp += 8;
            }
        }
    }
    n = std::min<int>(n, (int) sizeof(buf) - 2);
    buf[n++] = '\n';
    DWORD w = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), buf, (DWORD) n, &w, nullptr);   // no stdio: the crashing thread may hold its lock
    return true;
}
LONG WINAPI crash_veh(EXCEPTION_POINTERS * ep) {
    crash_report(ep, true);
    return EXCEPTION_CONTINUE_SEARCH;
}
LONG WINAPI crash_filter(EXCEPTION_POINTERS * ep) {
    crash_report(ep, false);
    return g_prev_filter ? g_prev_filter(ep) : EXCEPTION_CONTINUE_SEARCH;
}
void install_crash_report(state_t * s) {
    g_crash_state = s;
    AddVectoredExceptionHandler(0, crash_veh);
    g_prev_filter = SetUnhandledExceptionFilter(crash_filter);
}
#endif

}  // namespace
