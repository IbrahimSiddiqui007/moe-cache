// moe-cache plugin part: the cache: settings and state (state_t; its methods continue in mc_state_io.inl, mc_state_profile.inl, mc_state_lru.inl).
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

static bool server_repack_disabled() {
    if (const char * v = getenv("LLAMA_ARG_REPACK")) {
        if (!strcmp(v, "0") || !strcasecmp(v, "false") || !strcasecmp(v, "off") || !strcasecmp(v, "no") || !strcasecmp(v, "disabled")) {
            return true;
        }
    }
    // Windows: the plugin repacked although llama-server had --no-repack, so CPU-only output differed from stock (Windows CI executor step: 1/4 identical)
    for (const std::string & a : plat::command_line()) {
        if (a == "--no-repack" || a == "-nr") {
            return true;
        }
    }
    return false;
}

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
    std::atomic<uint64_t> gpu_segments{0}, gpu_fallbacks{0}, gpu_nomem{0};   // gpu_nomem: blocks kept on the CPU because the GPU had too little free memory
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
    double route_weight = 0;                            // MOE_CACHE_ROUTE_WEIGHT=share (idea I14): skip a non-resident pick whose share of the token's picked router weight is below this (implies ROUTE=skip); 0 = off
    uint64_t route_w_groups = 0, route_w_unavail = 0;   // groups where the weights were read / where they were not on the host yet (no skipping there)
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
    size_t budget = 0;
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
    std::unordered_map<std::string, std::pair<size_t, size_t>> file_tensors;
    std::vector<std::unique_ptr<tinfo_t>> tensors;
    std::unordered_map<const ggml_tensor *, int> by_tensor;
    std::vector<node_t> nodes;
    int    mru = -1;
    int    lru = -1;
    int    unmatched = 0;              // expert tensors that could not be cached (see register_tensor)
    // executor (MOE_CACHE_EXEC=1): asynchronous reads + the plugin's own expert matrix products for decode steps
    bool   exec_on = true;                              // on by default; MOE_CACHE_EXEC=0 switches the executor off
    bool   exec_force = false;                          // MOE_CACHE_EXEC=2: the executor runs every eligible step (tests)
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
    bool   rq_stop = false;
    std::atomic<uint64_t> exec_ops{0}, exec_mismatch{0}, exec_wait_ns{0}, exec_checked{0};
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
    std::atomic<uint64_t> ns_prep{0}, ns_cpu{0}, ns_total{0}, ns_read{0}, evictions{0}, direct_jobs{0}, bounce_jobs{0};

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
        fprintf(stderr, "moe-cache: plugin v0.2.3.5 (ggml backend API %d), running with ggml %s (%s)\n", GGML_BACKEND_API_VERSION, ggml_version(), ggml_commit());
        if (strcmp(ggml_version(), MOE_CACHE_TESTED_GGML) != 0) {
            fprintf(stderr, "moe-cache: warning: tested with ggml %s only; if the server crashes or the plugin misbehaves, rebuild it against your llama.cpp headers (or set MOE_CACHE_DISABLE=1)\n", MOE_CACHE_TESTED_GGML);
        }
        plat::env_utf8("MOE_CACHE_PROFILE", profile_path);     // paths: UTF-8 (Windows getenv gives the ANSI code page)
        plat::env_utf8("MOE_CACHE_WARM", warm_path);
        if (const char * v = getenv("MOE_CACHE_PRELOAD_FRAC")) {
            preload_frac = atof(v);
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
        }
        if (!file) {
            fail("MOE_CACHE_GGUF is not set");
            return;
        }
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
            const plat::file_t f = plat::file_open(shards[si].c_str(), true);
            if (!f.ok()) {
                fail("cannot open " + shards[si] + " with O_DIRECT");
                return;
            }
            fds.push_back(f);
            shard_paths.push_back(shards[si]);
            fds_buf.push_back(plat::file_t());
        }
        if (const char * v = getenv("MOE_CACHE_FAULT_EVERY")) {
            fault_every = atoi(v);
        }
        if (const char * v = getenv("MOE_CACHE_TEST_READ_DELAY_US")) {
            test_read_delay_us = std::max(0, atoi(v));
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
        for (size_t i = 0; i < ggml_backend_dev_count() && gpu_min_tokens > 0; i++) {
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
                // stock llama.cpp repacks CPU weights only when no GPU backend is present: do the same, so results stay identical
                bool gpu = false;
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    const auto t = ggml_backend_dev_type(ggml_backend_dev_get(i));
                    gpu = gpu || t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU;
                }
                want = !gpu && !server_repack_disabled();
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
                    repack_tag = ggml_backend_buffer_init(repack_buft, empty, nullptr, 0);
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
            if (const char * v = getenv("MOE_CACHE_RAM_GIB")) {
                size_t x = 0;
                if (parse_gib("MOE_CACHE_RAM_GIB", v, x) && x > 0) {
                    limit_bytes = x;
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
                    total += (size_t) std::filesystem::file_size(sp, ec);
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
        if (const char * v = getenv("MOE_CACHE_ROUTE")) {
            route_mode = strcmp(v, "skip") == 0 ? 1 : (strcmp(v, "subst") == 0 ? 2 : 0);
        }
        if (const char * v = getenv("MOE_CACHE_ROUTE_KEEP")) {
            route_keep = atoi(v);
        }
        if (const char * v = getenv("MOE_CACHE_ROUTE_WEIGHT")) {
            route_weight = std::max(0.0, atof(v));
            if (route_weight > 0 && route_mode == 0) {
                route_mode = 1;                           // weight-based skipping is a form of ROUTE=skip
            }
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
                std::thread([this] { char c; while (read(g_stats_pipe[0], &c, 1) == 1 && !g_stats_stop) { print_snapshot(); } }).detach();
            }
        }
#elif defined(_WIN32)
        // no signals: setting the named event Local\moe-cache-stats-<pid> prints one snapshot (tools/windows_test.ps1 does it before stopping the
        // server, which on Windows is killed, so the summary at exit never appears)
        if (getenv("MOE_CACHE_STATS_SIGNAL")) {
            char name[64];
            snprintf(name, sizeof(name), "Local\\moe-cache-stats-%lu", plat::process_id());
            g_stats_event = CreateEventA(nullptr, FALSE, FALSE, name);
            if (g_stats_event) {
                std::thread([this] { while (WaitForSingleObject(g_stats_event, INFINITE) == WAIT_OBJECT_0 && !g_stats_stop) { print_snapshot(); } }).detach();
            }
        }
        // Windows stops a server by killing it, so the profile save in the destructor never runs: the launcher sets this event first
        // (learn, server exit); without it only the save every 100,000 groups is kept
        {
            char name[64];
            snprintf(name, sizeof(name), "Local\\moe-cache-save-%lu", plat::process_id());
            g_save_event = CreateEventA(nullptr, FALSE, FALSE, name);
            if (g_save_event) {
                std::thread([this] {
                    while (WaitForSingleObject(g_save_event, INFINITE) == WAIT_OBJECT_0 && !g_stats_stop) {
                        save_profile();
                        fprintf(stderr, "moe-cache: usage profile saved (on request)\n");
                    }
                }).detach();
            }
        }
#endif
    }

    // the rest of state_t lives in three files, included here inside the class (idea I11; the compiled code is byte-identical to the single file)
#include "mc_state_io.inl"
#include "mc_state_profile.inl"
#include "mc_state_lru.inl"
};

state_t & S() {
    static state_t s;
    return s;
}

}  // namespace
