// moe-cache plugin part: the cache: settings, reads, lookahead, LRU, prepare_group, tuner, profile.
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

static bool server_repack_disabled() {
    if (const char * v = getenv("LLAMA_ARG_REPACK")) {
        if (!strcmp(v, "0") || !strcasecmp(v, "false") || !strcasecmp(v, "off") || !strcasecmp(v, "no") || !strcasecmp(v, "disabled")) {
            return true;
        }
    }
#ifdef __linux__
    if (FILE * cf = fopen("/proc/self/cmdline", "rb")) {
        std::string cmd;
        char buf[4096];
        size_t nr;
        while ((nr = fread(buf, 1, sizeof(buf), cf)) > 0) {
            cmd.append(buf, nr);
        }
        fclose(cf);
        for (size_t b = 0; b < cmd.size();) {
            const size_t e = cmd.find('\0', b);
            const std::string a = cmd.substr(b, e == std::string::npos ? std::string::npos : e - b);
            if (a == "--no-repack" || a == "-nr") {
                return true;
            }
            if (e == std::string::npos) {
                break;
            }
            b = e + 1;
        }
    }
#elif defined(_WIN32)
    // no /proc: split the process command line (double quotes group, as the C runtime does for argv); the plugin repacked although llama-server
    // had --no-repack, so CPU-only output differed from stock (Windows CI executor step: 1/4 identical)
    const char * c = GetCommandLineA();
    while (c && *c) {
        while (*c == ' ' || *c == '\t') { c++; }
        std::string a;
        bool q = false;
        for (; *c && (q || (*c != ' ' && *c != '\t')); c++) {
            if (*c == '"') { q = !q; } else { a += *c; }
        }
        if (a == "--no-repack" || a == "-nr") {
            return true;
        }
    }
#endif
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
    int  route_keep = -1;                               // MOE_CACHE_ROUTE_KEEP: picks always honoured (default: all but the last 2)
    uint64_t route_skipped = 0, route_substituted = 0, route_considered = 0;
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
        if (const char * v = getenv("MOE_CACHE_PROFILE")) {
            profile_path = v;
        }
        if (const char * v = getenv("MOE_CACHE_WARM")) {
            warm_path = v;
        }
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
        const char * file = getenv("MOE_CACHE_GGUF");
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
#ifdef __linux__
                if (FILE * cf = reserve ? nullptr : fopen("/proc/self/cmdline", "rb")) {
                    std::string cmd;
                    char buf[4096];
                    size_t nr;
                    while ((nr = fread(buf, 1, sizeof(buf), cf)) > 0) {
                        cmd.append(buf, nr);
                    }
                    fclose(cf);
                    std::vector<std::string> argv_;
                    for (size_t b = 0; b < cmd.size();) {
                        const size_t e = cmd.find('\0', b);
                        argv_.push_back(cmd.substr(b, e == std::string::npos ? std::string::npos : e - b));
                        if (e == std::string::npos) break;
                        b = e + 1;
                    }
                    for (size_t k = 0; k + 1 < argv_.size(); k++) {
                        if (argv_[k] == "--cache-ram" || argv_[k] == "-cram") {
                            cram = atol(argv_[k + 1].c_str());
                        }
                    }
                }
#endif
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
        if (const char * tp = getenv("MOE_CACHE_TRACE_IDS")) {
            trace_f = fopen_private(tp);
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
#endif
    }

    // ---- executor: background expert reads ----
    void reader_main() {
        static thread_local char * bounce = nullptr;
        static thread_local size_t bounce_size = 0;
        for (;;) {
            io_job_t j;
            {
                std::unique_lock<std::mutex> lk(rq_mu);
                rq_cv.wait(lk, [this] { return rq_stop || !rq.empty(); });
                if (rq_stop && rq.empty()) {
                    return;
                }
                j = rq.front();
                rq.pop_front();
            }
            const size_t need = round_up(max_nb2_all) + 2 * PAGE;
            if (bounce_size < need) {
                plat::aligned_free_page(bounce);
                bounce = (char *) plat::aligned_alloc_page(need);
                if (!bounce) {
                    fprintf(stderr, "moe-cache: out of memory for a read buffer\n");
                    abort();
                }
                bounce_size = need;
            }
            if (test_read_delay_us > 0) {
                // test switch: the expert stays "loading" with its bytes not yet there for a while, so any path that does not wait for the
                // loading flag reads stale data every time instead of only when the timing is unlucky (CI regression test for T39/T45)
                std::this_thread::sleep_for(std::chrono::microseconds(test_read_delay_us));
            }
            const auto r0 = std::chrono::steady_clock::now();
            read_job(j, bounce);
            ns_read += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - r0).count();
            {
                std::lock_guard<std::mutex> lk(mu);
                pending_bytes = pending_bytes > j.nb2 ? pending_bytes - j.nb2 : 0;
                __atomic_store_n(&nodes[(size_t) j.node].loading, false, __ATOMIC_RELEASE);
            }
            load_cv.notify_all();
        }
    }

    // ---- lookahead prefetch ----
    void pf_submit(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lk(pf_mu);
            pf_q.push_back(std::move(fn));
        }
        pf_cv.notify_one();
    }

    void pf_main() {
#ifdef __linux__
        if (la_ecores) {
            std::ifstream f("/sys/devices/cpu_atom/cpus");
            std::string txt;
            if (f && std::getline(f, txt)) {
                cpu_set_t set;
                CPU_ZERO(&set);
                size_t i = 0;
                bool any = false;
                while (i < txt.size()) {
                    int a = atoi(txt.c_str() + i), b = a;
                    while (i < txt.size() && isdigit((unsigned char) txt[i])) { i++; }
                    if (i < txt.size() && txt[i] == '-') {
                        i++;
                        b = atoi(txt.c_str() + i);
                        while (i < txt.size() && isdigit((unsigned char) txt[i])) { i++; }
                    }
                    for (int c = a; c <= b && c < CPU_SETSIZE; c++) { CPU_SET(c, &set); any = true; }
                    while (i < txt.size() && !isdigit((unsigned char) txt[i])) { i++; }
                }
                if (any) {
                    sched_setaffinity(0, sizeof(set), &set);
                }
            }
        }
#endif
        for (;;) {
            std::function<void()> fn;
            {
                std::unique_lock<std::mutex> lk(pf_mu);
                pf_cv.wait(lk, [this] { return pf_stop || !pf_q.empty(); });
                if (pf_stop && pf_q.empty()) {
                    return;
                }
                fn = std::move(pf_q.front());
                pf_q.pop_front();
            }
            fn();
        }
    }

    plat::file_t buffered_file(int shard) {
        std::lock_guard<std::mutex> lk(fds_buf_mu);
        if (!fds_buf[shard].ok()) {
            fds_buf[shard] = plat::file_open(shard_paths[shard].c_str(), false);
        }
        return fds_buf[shard];
    }

    bool read_floats(const std::string & name, size_t n, std::vector<float> & out) {
        auto it = tmeta.find(name);
        if (it == tmeta.end()) {
            return false;
        }
        const tmeta_t & m = it->second;
        const ggml_type ty = (ggml_type) m.type;
        if (ty != GGML_TYPE_F32 && ty != GGML_TYPE_F16 && ty != GGML_TYPE_BF16) {
            return false;
        }
        const size_t esz = ggml_type_size(ty);
        if (m.size != n * esz) {
            return false;
        }
        std::vector<char> raw(m.size);
        const plat::file_t f = buffered_file(m.shard);
        size_t done = 0;
        while (done < m.size) {
            const ptrdiff_t got = plat::file_pread(f, raw.data() + done, m.size - done, (uint64_t) (m.off + done));
            if (got <= 0) {
                return false;
            }
            done += (size_t) got;
        }
        out.resize(n);
        for (size_t i = 0; i < n; i++) {
            if (ty == GGML_TYPE_F32) {
                memcpy(&out[i], raw.data() + i * 4, 4);
            } else if (ty == GGML_TYPE_F16) {
                ggml_fp16_t h;
                memcpy(&h, raw.data() + i * 2, 2);
                out[i] = ggml_fp16_to_fp32(h);
            } else {
                ggml_bf16_t h;
                memcpy(&h, raw.data() + i * 2, 2);
                out[i] = ggml_bf16_to_fp32(h);
            }
        }
        return true;
    }

    router_t * get_router(int layer, size_t n_embd, size_t n_exp) {
        std::lock_guard<std::mutex> lk(router_mu);
        auto it = routers.find(layer);
        if (it != routers.end()) {
            return it->second.ok ? &it->second : nullptr;
        }
        router_t & r = routers[layer];
        char nm[160];
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", layer);
        r.ok = read_floats(nm, n_embd * n_exp, r.w);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.bias", layer);
        read_floats(nm, n_exp, r.bias);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", layer);
        if (!read_floats(nm, n_embd, r.normw)) {
            snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", layer);
            read_floats(nm, n_embd, r.normw);
        }
        return r.ok ? &r : nullptr;
    }

    // reserve cache room for the given experts of a layer (without evicting anything the current group needs) and read them in the background
    void reserve_and_read(int layer, const std::vector<int> & guess, int cap = -1) {
        if (cap < 0) {
            cap = la_cap;
        }
        std::vector<std::pair<io_job_t, int>> todo;
        {
            std::lock_guard<std::mutex> lk(mu);
            auto lt = layer_tensors.find(layer);
            if (lt == layer_tensors.end()) {
                return;
            }
            int fresh = 0;
            for (int e : guess) {
                {
                    const tinfo_t & t0 = *tensors[lt->second[0]];
                    if (e < 0 || e >= t0.n_expert || nodes[t0.node0 + e].resident) {
                        continue;                 // out of range, or already in RAM: nothing to read
                    }
                }
                if (++fresh > cap) {
                    break;
                }
                for (int ti_i : lt->second) {
                    tinfo_t & ti = *tensors[ti_i];
                    const int s = ti.node0 + e;
                    if (nodes[s].resident) {
                        continue;
                    }
                    while (resident_bytes + ti.nb2 > budget) {
                        int v = lru;
                        while (v >= 0 && (nodes[v].epoch == epoch || nodes[v].loading)) {
                            v = nodes[v].prev;
                        }
                        if (v < 0) {
                            break;
                        }
                        evict(v);
                    }
                    if (resident_bytes + ti.nb2 > budget) {
                        pf_skipped++;
                        continue;
                    }
                    node_t & n = nodes[s];
                    n.resident = true;
                    __atomic_store_n(&n.loading, true, __ATOMIC_RELEASE);      // the executor reads this flag without the lock
                    n.prefetched = true;
                    n.epoch = epoch;
                    resident_bytes += ti.nb2;
                    push_mru(s);
                    todo.push_back({ io_job_t{ ti.data + (size_t) e * ti.nb2, ti.file_off + (size_t) e * ti.nb2, ti.nb2, ti.fdi, ti.direct, ti.repack ? ti.tensor : nullptr }, s });
                    pf_issued++;
                }
            }
        }
        for (auto & td : todo) {
            pf_submit([this, j = td.first, s = td.second] {
                static thread_local char * bounce = nullptr;
                static thread_local size_t bounce_size = 0;
                const size_t need = round_up(max_nb2_all) + 2 * PAGE;
                if (bounce_size < need) {
                    plat::aligned_free_page(bounce);
                    bounce = (char *) plat::aligned_alloc_page(need);
                    if (!bounce) {
                        fprintf(stderr, "moe-cache: out of memory for a read buffer\n");
                        abort();
                    }
                    bounce_size = need;
                }
                read_job(j, bounce);
                {
                    std::lock_guard<std::mutex> lk(mu);
                    __atomic_store_n(&nodes[s].loading, false, __ATOMIC_RELEASE);   // publishes the expert's bytes to the executor (reads without mu)
                }
                load_cv.notify_all();
            });
        }
    }

    // x = the input of the experts of this layer; predict which experts the next layer will pick and start reading them
    void predict_and_prefetch(int layer, const std::vector<float> & x, int n_used) {
        const int nl = layer + 1;
        size_t n_exp = 0;
        {
            std::lock_guard<std::mutex> lk(mu);
            auto lt = layer_tensors.find(nl);
            if (lt == layer_tensors.end() || lt->second.empty()) {
                return;
            }
            n_exp = (size_t) tensors[lt->second[0]]->n_expert;
        }
        const size_t n_embd = x.size();
        router_t * rn = get_router(nl, n_embd, n_exp);
        if (!rn) {
            return;
        }
        router_t * rc = get_router(layer, n_embd, n_exp);
        std::vector<float> xs(n_embd);
        const bool scale = rc && rn->normw.size() == n_embd && rc->normw.size() == n_embd;
        for (size_t i = 0; i < n_embd; i++) {
            xs[i] = x[i] * (scale && fabsf(rc->normw[i]) > 1e-6f ? rn->normw[i] / rc->normw[i] : 1.0f);
        }
        std::vector<float> logit(n_exp);
        for (size_t e = 0; e < n_exp; e++) {
            const float * w = rn->w.data() + e * n_embd;
            logit[e] = (rn->bias.size() == n_exp ? rn->bias[e] : 0.0f) + dot_f32(w, xs.data(), n_embd);
        }
        const size_t G = std::min<size_t>(n_exp, la_guess > 0 ? (size_t) la_guess : (size_t) n_used);
        std::vector<int> idx(n_exp);
        for (size_t e = 0; e < n_exp; e++) { idx[e] = (int) e; }
        std::partial_sort(idx.begin(), idx.begin() + G, idx.end(), [&](int a, int b) { return logit[a] > logit[b]; });
        idx.resize(G);
        {
            std::lock_guard<std::mutex> lk(mu);
            auto & fl = pred_flags[nl];
            fl.assign(n_exp, 0);
            for (int e : idx) { fl[(size_t) e] = 1; }
        }
        reserve_and_read(nl, idx);
    }

    // one line of cumulative counters on stderr (see MOE_CACHE_STATS_SIGNAL)
    void print_snapshot() {
        std::string j;
        {
            std::lock_guard<std::mutex> lk(mu);
            char b[1600];
            snprintf(b, sizeof(b), "{\"hits\":%llu,\"misses\":%llu,\"hits_dec\":%llu,\"misses_dec\":%llu,\"read_bytes\":%llu,\"evictions\":%llu,\"evict_age_sum\":%llu,"
                     "\"resident_bytes\":%llu,\"budget\":%llu,\"ns_prep\":%llu,\"ns_read\":%llu,\"ns_cpu\":%llu,\"ns_total\":%llu,\"graphs\":%llu,"
                     "\"direct\":%llu,\"bounce\":%llu,\"repack\":%llu,\"gpu_segments\":%llu,\"gpu_up_bytes\":%llu,\"fallbacks\":%llu,\"shrinks\":%llu,\"grows\":%llu,\"pf_issued\":%llu,\"pf_used\":%llu,\"pf_wasted\":%llu,\"pf_late\":%llu,\"pf_pred_total\":%llu,\"pf_pred_hit\":%llu,\"pf_skipped\":%llu,\"sibling_loads\":%llu,\"small_mapped\":%llu,\"small_over\":%llu",
                     (unsigned long long) hits, (unsigned long long) misses, (unsigned long long) hits_dec, (unsigned long long) misses_dec, (unsigned long long) bytes,
                     (unsigned long long) evictions, (unsigned long long) evict_age_sum, (unsigned long long) resident_bytes, (unsigned long long) budget,
                     (unsigned long long) ns_prep, (unsigned long long) ns_read, (unsigned long long) ns_cpu, (unsigned long long) ns_total, (unsigned long long) graphs,
                     (unsigned long long) direct_jobs, (unsigned long long) bounce_jobs, (unsigned long long) repack_jobs, (unsigned long long) gpu_segments,
                     (unsigned long long) gpu_up_bytes, (unsigned long long) read_fallbacks, (unsigned long long) tune_shrinks, (unsigned long long) tune_grows,
                     (unsigned long long) pf_issued, (unsigned long long) pf_used, (unsigned long long) pf_wasted, (unsigned long long) pf_late,
                     (unsigned long long) pf_pred_total, (unsigned long long) pf_pred_hit, (unsigned long long) pf_skipped, (unsigned long long) sibling_loads,
                     (unsigned long long) small_mapped, (unsigned long long) small_over);
            j = b;
            j += ",\"exec_ops\":" + std::to_string(exec_ops.load()) + ",\"exec_checked\":" + std::to_string(exec_checked.load()) +
                 ",\"exec_mismatch\":" + std::to_string(exec_mismatch.load());
            j += ",\"layer_hits\":[";
            for (size_t i = 0; i < layer_hits.size(); i++) { j += (i ? "," : "") + std::to_string(layer_hits[i]); }
            j += "],\"layer_misses\":[";
            for (size_t i = 0; i < layer_misses.size(); i++) { j += (i ? "," : "") + std::to_string(layer_misses[i]); }
            j += "]}";
        }
        fprintf(stderr, "moe-cache-stats %s\n", j.c_str());
        fflush(stderr);
    }

    ~state_t() {
        {
            std::lock_guard<std::mutex> lk(rq_mu);
            rq_stop = true;
        }
        rq_cv.notify_all();
        for (auto & t : readers) {
            t.join();
        }
        pool.reset();
#ifdef __linux__
        g_stats_stop = true;
        if (g_stats_pipe[1] >= 0) {
            close(g_stats_pipe[1]);          // wakes the snapshot thread, which then exits
        }
#elif defined(_WIN32)
        g_stats_stop = true;
        if (g_stats_event) {
            SetEvent(g_stats_event);         // wakes the snapshot thread, which then exits
        }
#endif
        if (trace_f) {
            fclose(trace_f);
        }
        {
            std::lock_guard<std::mutex> lk(pf_mu);
            pf_stop = true;
        }
        pf_cv.notify_all();
        for (auto & t : pf_threads) {
            t.join();
        }
        {
            std::lock_guard<std::mutex> lk(wmu);
            stop = true;
        }
        wcv.notify_all();
        for (auto & t : workers) {
            t.join();
        }
        if (cache_on) {
            save_profile();
        }
        if (!cache_on && stats) {
            fprintf(stderr, "moe-cache: pass-through graphs=%llu MUL_MAT_ID ops=%llu\n", (unsigned long long) graphs, (unsigned long long) ops);
        }
        if (cache_on && stats && hyb_groups) {
            fprintf(stderr, "moe-cache: hybrid: %llu layer steps, %llu of %llu picks computed on the GPU from its bank (%.1f%%), the rest on the CPU\n",
                    (unsigned long long) hyb_groups, (unsigned long long) hyb_hits, (unsigned long long) hyb_picks, hyb_picks ? 100.0 * hyb_hits / hyb_picks : 0.0);
            if (hyb_swaps) {
                fprintf(stderr, "moe-cache: hybrid: %llu experts swapped into the bank while running (%llu of them profile experts brought back; %.2f s uploading)\n",
                        (unsigned long long) hyb_swaps, (unsigned long long) hyb_reverts, hyb_swap_ns / 1e9);
            }
            if (hyb_drops) {
                fprintf(stderr, "moe-cache: hybrid: bank freed %llu times for long prompts, loaded again %llu times (%.2f s in total)\n",
                        (unsigned long long) hyb_drops, (unsigned long long) hyb_reloads, hyb_reload_ns / 1e9);
            }
        }
        if (cache_on && stats && gpu_dev) {
            if (coalesced) {
                fprintf(stderr, "moe-cache: %llu expert reads merged into their neighbours (MOE_CACHE_COALESCE_MIB)\n", (unsigned long long) coalesced.load());
            }
            fprintf(stderr, "moe-cache: long-prompt blocks computed on the GPU %llu (uploaded %.1f GB), fell back to the CPU %llu (of which too little free GPU memory: %llu)\n", (unsigned long long) gpu_segments, gpu_up_bytes / 1e9, (unsigned long long) gpu_fallbacks, (unsigned long long) gpu_nomem);
        }
        if (cache_on && stats) {
            const uint64_t h = hits, m = misses;
            if (read_fallbacks) {
                fprintf(stderr, "moe-cache: reads that needed the buffered fallback: %llu\n", (unsigned long long) read_fallbacks);
            }
            if (lookahead) {
                fprintf(stderr, "moe-cache: lookahead: prefetched %llu experts, used %llu, wasted %llu, late %llu; prediction recall %.1f%% (%llu of %llu), skipped %llu\n",
                        (unsigned long long) pf_issued, (unsigned long long) pf_used, (unsigned long long) pf_wasted, (unsigned long long) pf_late,
                        pf_pred_total ? 100.0 * pf_pred_hit / pf_pred_total : 0.0, (unsigned long long) pf_pred_hit, (unsigned long long) pf_pred_total, (unsigned long long) pf_skipped);
            }
            if (exec_on) {
                fprintf(stderr, "moe-cache: executor ran %llu expert ops itself%s\n", (unsigned long long) exec_ops,
                        exec_check ? (std::string(", checked ") + std::to_string(exec_checked.load()) + ", mismatches " + std::to_string(exec_mismatch.load())).c_str() : "");
                fprintf(stderr, "moe-cache: executor stepped aside (all resident) in %llu graphs; %llu of them waited for experts still loading\n",
                        (unsigned long long) stepaside_graphs, (unsigned long long) stepaside_waits);
            }
            if (exec_on) {
                for (int ty = 0; ty < 64; ty++) {
                    if (exec_type_ops[ty]) {
                        fprintf(stderr, "moe-cache: executor by type %-8s ops=%llu time=%.2fs expert data=%.1f GB (%.2f GB/s incl. waits)\n", ggml_type_name((ggml_type) ty),
                                (unsigned long long) exec_type_ops[ty], exec_type_ns[ty] / 1e9, exec_type_bytes[ty] / 1e9, exec_type_ns[ty] ? (double) exec_type_bytes[ty] / exec_type_ns[ty] : 0.0);
                    }
                }
            }
            if (route_mode) {
                fprintf(stderr, "moe-cache: approximate routing (%s): %llu low-ranked picks looked at, %llu given up, %llu replaced by a resident expert\n",
                        route_mode == 1 ? "skip" : "subst", (unsigned long long) route_considered, (unsigned long long) route_skipped, (unsigned long long) route_substituted);
            }
            if (small_mapped || small_over) {
                fprintf(stderr, "moe-cache: cache smaller than one op: %llu experts mapped from the model file, %llu loaded over the cache size\n",
                        (unsigned long long) small_mapped, (unsigned long long) small_over);
            }
            fprintf(stderr, "moe-cache: reads straight into place %llu, via bounce buffer %llu, repacked %llu\n", (unsigned long long) direct_jobs, (unsigned long long) bounce_jobs, (unsigned long long) repack_jobs);
            fprintf(stderr, "moe-cache: time prep=%.2fs (of which reads %.2fs) cpu_compute=%.2fs total_in_backend=%.2fs\n", ns_prep / 1e9, ns_read / 1e9, ns_cpu / 1e9, ns_total / 1e9);
            fprintf(stderr, "moe-cache: graphs=%llu groups=%llu ops=%llu hits=%llu misses=%llu (hit %.1f%%) read=%.1f MB evictions=%llu resident=%.2f GiB%s\n",
                    (unsigned long long) graphs, (unsigned long long) groups, (unsigned long long) ops, (unsigned long long) h, (unsigned long long) m,
                    h + m ? 100.0 * h / (h + m) : 0.0, bytes / 1e6, (unsigned long long) evictions, resident_bytes / (double) (1u << 30),
                    auto_mode ? (std::string(" | auto: final budget ") + std::to_string(budget / 1073741824.0).substr(0, 5) + " GiB, shrinks " + std::to_string(tune_shrinks) + ", grows " + std::to_string(tune_grows)).c_str() : "");
        }
    }

    // run fn(0..n_io) on the workers and the caller, wait until all are done
    void run_parallel(const std::function<void(int)> & fn) {
        {
            std::lock_guard<std::mutex> lk(wmu);
            task = fn;
            busy = (int) workers.size();
            gen++;
        }
        wcv.notify_all();
        fn(0);
        std::unique_lock<std::mutex> lk(wmu);
        dcv.wait(lk, [this] { return busy == 0; });
    }

    void worker_main(int id) {
        int seen = 0;
        for (;;) {
            std::function<void(int)> fn;
            {
                std::unique_lock<std::mutex> lk(wmu);
                wcv.wait(lk, [&] { return stop || gen != seen; });
                if (stop) {
                    return;
                }
                seen = gen;
                fn = task;
            }
            fn(id);
            {
                std::lock_guard<std::mutex> lk(wmu);
                busy--;
            }
            dcv.notify_all();
        }
    }

    // read one expert slice into place (or through the bounce buffer); retries, then falls back to a buffered read, then stops
    void read_job(const io_job_t & j, char * bounce) {
        const size_t aoff = j.off / PAGE * PAGE;
        const size_t shift = j.off - aoff;
        const size_t len = round_up(shift + j.nb2);
        char * buf = j.direct ? j.dst - shift : bounce;   // direct: the aligned read lands exactly around the expert
        // the pages must exist before they are written (Windows); one extra page covers kernels that read a little past the end
        plat::mem_commit(j.direct ? j.dst - shift : j.dst, (j.direct ? len : j.nb2) + PAGE);
        // direct read with retries; if it keeps failing, one more try through a normal buffered read; then stop (never serve wrong data)
        const bool inject_job = fault_every > 0 && (++fault_ctr % (uint64_t) fault_every) == 0;   // test switch: this read fails every direct attempt
        auto read_all = [&](const plat::file_t & rfd, bool inject) {
            size_t done = 0;
            int retries = 0;
            while (done < shift + j.nb2) {
                ptrdiff_t got;
                if (inject && inject_job) {
                    got = -1;
                    errno = EIO;
                } else {
                    got = plat::file_pread(rfd, buf + done, len - done, (uint64_t) (aoff + done));
                }
                if (got <= 0) {
                    if (++retries <= 3) {
                        plat::sleep_ms(20);
                        continue;
                    }
                    return false;
                }
                done += (size_t) got;
            }
            return true;
        };
        if (!read_all(fds[j.fdi], true)) {
            plat::file_t bfd;
            {
                std::lock_guard<std::mutex> lk(fds_buf_mu);
                if (!fds_buf[j.fdi].ok()) {
                    fds_buf[j.fdi] = plat::file_open(shard_paths[j.fdi].c_str(), false);
                }
                bfd = fds_buf[j.fdi];
            }
            if (!bfd.ok() || !read_all(bfd, false)) {
                fprintf(stderr, "moe-cache: read failed at offset %zu of %s: %s (disk error, or the model file changed after loading)\n", aoff, shard_paths[j.fdi].c_str(), plat::last_error().c_str());
                abort();
            }
            if (read_fallbacks++ == 0) {
                fprintf(stderr, "moe-cache: direct read failed, used a buffered read instead (output stays correct; this may be slower)\n");
            }
        }
        if (j.rtensor) {
            // repack this one expert into its place: use a copy of the tensor that describes a single expert
            ggml_tensor v = *j.rtensor;
            v.ne[2] = 1;
            v.ne[3] = 1;
            v.nb[2] = v.nb[1] * v.ne[1];
            v.nb[3] = v.nb[2];
            v.data = j.dst;
            v.view_src = nullptr;
            repack_buf->iface.set_tensor(repack_buf, &v, bounce + shift, 0, j.nb2);
            repack_jobs++;
        } else if (!j.direct) {
            memcpy(j.dst, bounce + shift, j.nb2);
            bounce_jobs++;
        } else {
            direct_jobs++;
        }
        bytes += len;
    }

    void do_reads(const std::vector<io_job_t> & jobs, size_t max_nb2) {
        std::atomic<size_t> next{0};
        auto fn = [&](int) {
            static thread_local char * bounce = nullptr;
            static thread_local size_t bounce_size = 0;
            const size_t need = round_up(max_nb2) + 2 * PAGE;
            if (bounce_size < need) {
                plat::aligned_free_page(bounce);
                bounce = (char *) plat::aligned_alloc_page(need);
                if (!bounce) {
                    abort();
                }
                bounce_size = need;
            }
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= jobs.size()) {
                    break;
                }
                read_job(jobs[i], bounce);
            }
        };
        if (jobs.size() <= 1) {
            fn(0);
        } else {
            run_parallel(fn);
        }
    }

    // ---- registration (at load time) ----
    // a buffer of ours is being freed (model unloaded): forget every expert inside it, so nothing later reads, evicts (madvise) or counts memory that
    // may by then belong to someone else. Waits for background reads into that range first. A model loaded afterwards registers afresh.
    void forget_range(const char * base, size_t size) {
        std::unique_lock<std::mutex> lk(mu);
        for (size_t i = 0; i < tensors.size(); i++) {
            tinfo_t & ti = *tensors[i];
            if (!ti.data || ti.data < base || ti.data >= base + size) {
                continue;
            }
            for (int64_t e = 0; e < ti.n_expert; e++) {
                node_t & n = nodes[(size_t) (ti.node0 + e)];
                while (__atomic_load_n(&n.loading, __ATOMIC_ACQUIRE)) {
                    load_cv.wait(lk);
                }
                if (n.resident) {
                    unlink(ti.node0 + (int) e);
                    n.resident = false;
                    resident_bytes -= ti.nb2;
                }
                n.mapped = false;
            }
            for (auto it = by_tensor.begin(); it != by_tensor.end();) {
                it = it->second == (int) i ? by_tensor.erase(it) : std::next(it);
            }
            ti.n_expert = 0;          // the entry stays (indices are stable) but owns nothing
            ti.data = nullptr;
        }
    }

    // data: the bytes llama.cpp is loading for this tensor (its own copy of the model). Name and size alone do not prove that MOE_CACHE_GGUF is the
    // same model (another quantisation or fine-tune can have the same layout), so the first bytes are compared with the file as well
    bool same_bytes_in_file(const std::string & name, size_t off, const void * data, size_t size) {
        auto sh = file_shard.find(name);
        const plat::file_t f = buffered_file(sh != file_shard.end() ? sh->second : 0);
        char buf[4096];
        const size_t n = std::min(size, sizeof(buf));
        size_t done = 0;
        while (done < n) {
            const ptrdiff_t got = plat::file_pread(f, buf + done, n - done, (uint64_t) (off + done));
            if (got <= 0) {
                return false;
            }
            done += (size_t) got;
        }
        return memcmp(buf, data, n) == 0;
    }

    int register_tensor(const ggml_tensor * t, const void * data) {
        auto it = by_tensor.find(t);
        if (it != by_tensor.end()) {
            return it->second;
        }
        auto ft = file_tensors.find(t->name);
        if (ft == file_tensors.end() || t->ne[2] <= 1 || ft->second.second != t->nb[2] * (size_t) t->ne[2] ||
            !same_bytes_in_file(t->name, ft->second.first, data, ft->second.second)) {
            // not the model MOE_CACHE_GGUF describes: keep this tensor as plain memory (the stock way) instead of stopping the server
            if (unmatched++ == 0) {
                fprintf(stderr, "moe-cache: tensor %s is not in the GGUF, has another size or other bytes: it is loaded normally and not cached (is MOE_CACHE_GGUF the model you loaded?)\n", t->name);
            }
            return -1;
        }
        auto ti = std::make_unique<tinfo_t>();
        ti->name = t->name;
        ti->file_off = ft->second.first;
        ti->nb2 = t->nb[2];
        ti->n_expert = t->ne[2];
        ti->data = (char *) t->data;
        {
            auto sh = file_shard.find(t->name);
            ti->fdi = sh != file_shard.end() ? sh->second : 0;
            ti->tensor = const_cast<ggml_tensor *>(t);
            if (repack_buf) {
                ti->tensor->extra = nullptr;
                repack_buf->iface.init_tensor(repack_buf, ti->tensor);     // llama.cpp picks the layout, or none for types it cannot repack
                ti->repack = ti->tensor->extra != nullptr;
            }
            ti->direct = direct_ok && !ti->repack && ((uintptr_t) ti->data % PAGE) == (ti->file_off % PAGE);
        }
        sscanf(t->name, "blk.%d.", &ti->layer);
        ti->node0 = (int) nodes.size();
        nodes.resize(nodes.size() + (size_t) t->ne[2]);
        const int idx = (int) tensors.size();
        if (ti->layer >= 0) {
            layer_tensors[ti->layer].push_back(idx);
        }
        max_nb2_all = std::max(max_nb2_all, ti->nb2);
        tensors.push_back(std::move(ti));
        by_tensor[t] = idx;
        return idx;
    }

    void save_profile() {
        if (profile_path.empty() || session.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lk(mu);
        int nl = std::max<int>(learned.n_layers, (int) session.size());
        int ne = learned.n_exp;
        for (const auto & v : session) {
            ne = std::max<int>(ne, (int) v.size());
        }
        std::vector<float> out((size_t) nl * ne, 0.0f);
        for (int l = 0; l < nl; l++) {
            for (int e = 0; e < ne; e++) {
                double v = learned.at(l, e) * profile_decay;
                if (l < (int) session.size() && e < (int) session[l].size()) {
                    v += session[l][e];
                }
                out[(size_t) l * ne + e] = (float) v;
            }
        }
        const size_t slash = profile_path.find_last_of("/\\");
        if (slash != std::string::npos && slash > 0) {
            const std::string dir = profile_path.substr(0, slash);
            size_t pos = 0;
            while ((pos = dir.find_first_of("/\\", pos + 1)) != std::string::npos) {
                plat::make_dir(dir.substr(0, pos).c_str());
            }
            plat::make_dir(dir.c_str());
        }
        // one temporary file per process (two servers may share a profile), and the old profile is only replaced by a complete one
        const std::string tmp = profile_path + "." + std::to_string(plat::process_id()) + ".tmp";
        FILE * f = fopen_private(tmp.c_str());
        if (!f) {
            fprintf(stderr, "moe-cache: cannot write profile %s\n", tmp.c_str());
            return;
        }
        const int32_t h[2] = { nl, ne };
        bool ok = fwrite("XUSE", 1, 4, f) == 4;
        ok = ok && fwrite(h, sizeof(int32_t), 2, f) == 2;
        ok = ok && fwrite(out.data(), sizeof(float), out.size(), f) == out.size();
        ok = ok && fwrite(&fingerprint, 8, 1, f) == 1;
        ok = (fclose(f) == 0) && ok;
        if (!ok) {
            fprintf(stderr, "moe-cache: writing the profile failed (disk full?); the old profile %s is kept\n", profile_path.c_str());
            std::remove(tmp.c_str());
            return;
        }
        plat::replace_file(tmp.c_str(), profile_path.c_str());
    }

    // warm start: read the most used experts of the profile into place, most used last (so it is the most recently used)
    void preload() {
        std::lock_guard<std::mutex> lk(mu);
        if (preload_done) {
            return;
        }
        preload_done = true;
        std::map<int, std::vector<int>> by_layer;
        size_t total = 0;
        for (size_t i = 0; i < tensors.size(); i++) {
            if (tensors[i]->layer >= 0) {
                by_layer[tensors[i]->layer].push_back((int) i);
                total += tensors[i]->nb2 * (size_t) tensors[i]->n_expert;
            }
        }
        if (by_layer.empty()) {
            return;
        }
        // no profile: if every CPU expert fits in the cache, load them all in layer order (a cache filled expert by expert on demand
        // decodes about 15 % slower than one filled in order; MOE_CACHE_PRELOAD_ALL=0 turns this off)
        if (!preload_on) {
            const char * pa = getenv("MOE_CACHE_PRELOAD_ALL");
            if ((pa && strcmp(pa, "0") == 0) || total > budget * 0.97) {
                return;
            }
        }
        double frac = preload_frac;
        if (frac < 0) {
            frac = total <= budget * 0.97 ? 1.0 : 0.7;
        }
        const double limit = frac * (double) budget * 0.97;
        struct cand_t { int l, e; double c; };
        std::vector<cand_t> cand;
        for (auto & kv : by_layer) {
            const int ne = (int) tensors[kv.second[0]]->n_expert;
            for (int e = 0; e < ne; e++) {
                cand.push_back({ kv.first, e, learned.at(kv.first, e) });
            }
        }
        std::stable_sort(cand.begin(), cand.end(), [](const cand_t & a, const cand_t & b) { return a.c > b.c; });
        std::vector<io_job_t> jobs;
        size_t acc = 0, max_nb2 = 0, n_taken = 0;
        std::vector<cand_t> take;
        for (const auto & c : cand) {
            size_t bytes = 0;
            for (int ti : by_layer[c.l]) {
                bytes += tensors[ti]->nb2;
            }
            if ((double) (acc + bytes) > limit) {
                break;
            }
            acc += bytes;
            take.push_back(c);
        }
        for (size_t i = take.size(); i-- > 0;) {
            for (int ti_i : by_layer[take[i].l]) {
                tinfo_t & ti = *tensors[ti_i];
                const int s = ti.node0 + take[i].e;
                if (nodes[s].resident) {
                    continue;
                }
                nodes[s].resident = true;
                nodes[s].epoch = 0;
                resident_bytes += ti.nb2;
                push_mru(s);
                max_nb2 = std::max(max_nb2, ti.nb2);
                jobs.push_back({ ti.data + (size_t) take[i].e * ti.nb2, ti.file_off + (size_t) take[i].e * ti.nb2, ti.nb2, ti.fdi, ti.direct, ti.repack ? ti.tensor : nullptr });
                n_taken++;
            }
        }
        const auto r0 = std::chrono::steady_clock::now();
        do_reads(jobs, max_nb2);
        fprintf(stderr, "moe-cache: warm start loaded %zu expert tensors (%.2f GiB, %.0f%% of the cache) in %.1f s\n", n_taken, acc / (double) (1u << 30),
                100.0 * acc / (double) budget, std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count());
    }

    // auto mode: keep the real anonymous memory of the process under (limit - margin). Called every 32 groups and after every 256 MiB of reads.
    void tune() {
        if (max_budget == 0) {
            size_t tot = 0, big = 0;
            for (const auto & t : tensors) {
                tot += t->nb2 * (size_t) t->n_expert;
                big = std::max(big, t->nb2 * (size_t) t->n_expert);
            }
            max_budget = tot;
            min_budget = std::min(tot, 3 * big);
            budget = std::min(budget, max_budget);
            const size_t tgt = limit_bytes > margin ? limit_bytes - margin : limit_bytes / 2;
            if (min_budget + ((size_t) 512 << 20) > tgt) {
                fprintf(stderr, "moe-cache: warning: the memory limit (%.1f GiB) is very small for this model (the cache needs at least %.1f GiB); expect out-of-memory kills\n",
                        limit_bytes / 1073741824.0, min_budget / 1073741824.0);
            }
        }
        const size_t rss = read_rss_anon();
        if (!rss) {
            return;
        }
        const size_t target = limit_bytes > margin ? limit_bytes - margin : limit_bytes / 2;
        // bytes of expert data that are really in memory (pending ones are marked but not read yet)
        const size_t have = resident_bytes > pending_bytes ? resident_bytes - pending_bytes : 0;
        if (rss > target) {
            // over the target: the new budget is what is loaded now minus the excess, so eviction starts at once
            const size_t over = rss - target + ((size_t) 64 << 20);
            size_t nb = have > over ? have - over : 0;
            nb = std::min(std::max(nb, min_budget), budget);
            if (nb < budget) {
                budget = nb;
                tune_shrinks++;
            }
            while (resident_bytes > budget) {
                int v = lru;
                while (v >= 0 && (nodes[v].epoch == epoch || nodes[v].loading)) {
                    v = nodes[v].prev;
                }
                if (v < 0) {
                    break;
                }
                evict(v);
            }
        } else if (resident_bytes + ((size_t) 512 << 20) >= budget && rss + ((size_t) 512 << 20) < target && budget < max_budget) {
            // the cache is full and there is clearly room: grow, at most 1 GiB per step
            const size_t g = std::min<size_t>(target - rss - ((size_t) 256 << 20), (size_t) 1 << 30);
            budget = std::min(max_budget, budget + g);
            tune_grows++;
        }
    }

    // ---- LRU over resident experts ----
    void unlink(int s) {
        node_t & n = nodes[s];
        if (n.prev >= 0) { nodes[n.prev].next = n.next; } else { mru = n.next; }
        if (n.next >= 0) { nodes[n.next].prev = n.prev; } else { lru = n.prev; }
        n.prev = n.next = -1;
    }

    // evict-first insertion: a long prompt's experts recycle their own slots instead of pushing out what was cached before (MOE_CACHE_SCAN_TOKENS)
    void push_lru(int s) {
        node_t & n = nodes[s];
        n.next = -1;
        n.prev = lru;
        if (lru >= 0) { nodes[lru].next = s; }
        lru = s;
        if (mru < 0) { mru = s; }
    }

    void push_mru(int s) {
        node_t & n = nodes[s];
        n.prev = -1;
        n.next = mru;
        if (mru >= 0) { nodes[mru].prev = s; }
        mru = s;
        if (lru < 0) { lru = s; }
    }

    // give the physical pages of one expert back to the system (pages shared with a neighbour expert stay)
    void evict(int s) {
        // find the tensor that owns node s
        size_t lo_t = 0, hi_t = tensors.size();
        while (lo_t + 1 < hi_t) {
            const size_t mid = (lo_t + hi_t) / 2;
            if (tensors[mid]->node0 <= s) { lo_t = mid; } else { hi_t = mid; }
        }
        tinfo_t & ti = *tensors[lo_t];
        const size_t e = (size_t) (s - ti.node0);
        const uintptr_t start = (uintptr_t) ti.data + e * ti.nb2;
        const uintptr_t a = (start + PAGE - 1) / PAGE * PAGE;
        const uintptr_t b = (start + ti.nb2) / PAGE * PAGE;
        if (b > a) {
            plat::mem_decommit((void *) a, b - a);
        }
        evict_age_sum += epoch - nodes[s].epoch;   // how many groups ago this expert was last used
        if (nodes[s].prefetched) {
            nodes[s].prefetched = false;
            pf_wasted++;
        }
        unlink(s);
        nodes[s].resident = false;
        nodes[s].mapped = false;       // the pages were given back (guard mode: made inaccessible); a later fallback maps them again
        resident_bytes -= ti.nb2;
        evictions++;
    }

    // cache smaller than one op: map expert e of ti straight from the model file over its own address (copy-on-write file pages, the way stock
    // llama.cpp's mmap loading sees it). The expert stays non-resident (outside the budget and the LRU); when it is needed again and fits, it is
    // read into the cache the normal way (the read lands in private copies of these pages). Only for experts stored as in the file (not repacked)
    // whose address matches the file offset modulo the page size (ti.direct). Called with mu held (lk).
    bool map_from_file(const tinfo_t & ti, int64_t e, std::unique_lock<std::mutex> & lk) {
        if (!ti.direct) {
            return false;
        }
        // the pages at both ends are shared with the neighbour experts: no background read may be filling them while the mapping replaces them
        for (;;) {
            bool busy = false;
            for (int64_t k = std::max<int64_t>(0, e - 1); k <= std::min<int64_t>(ti.n_expert - 1, e + 1); k++) {
                busy = busy || nodes[(size_t) (ti.node0 + k)].loading;
            }
            if (!busy) {
                break;
            }
            load_cv.wait(lk);
        }
        const uintptr_t start = (uintptr_t) ti.data + (size_t) e * ti.nb2;
        const uintptr_t a = start / PAGE * PAGE;
        const uintptr_t b = round_up(start + ti.nb2);
        const uint64_t off = ti.file_off + (size_t) e * ti.nb2 - (start - a);
        return plat::mem_map_file((void *) a, b - a, buffered_file(ti.fdi), off);
    }

    // make sure all experts used by the ops of one group (same ids tensor) are resident
    // returns the number of reads handed to the reader threads (async mode), otherwise 0
    size_t prepare_group(const std::vector<int> & tidx, const std::vector<int> & cnt, bool async = false) {
        std::unique_lock<std::mutex> lk(mu);
        epoch++;
        std::vector<int> wait_list;       // experts a lookahead read is still filling
        if (lookahead && cur_decode) {
            auto pf = pred_flags.find(tensors[tidx[0]]->layer);
            if (pf != pred_flags.end()) {
                for (size_t e = 0; e < cnt.size() && e < pf->second.size(); e++) {
                    if (cnt[e]) {
                        pf_pred_total++;
                        pf_pred_hit += pf->second[e] ? 1 : 0;
                    }
                }
                pred_flags.erase(pf);
            }
        }
        groups++;
        if (auto_mode && (groups % (async ? 4 : 32)) == 1) {
            tune();
        }
        if (!profile_path.empty()) {
            const int layer = tensors[tidx[0]]->layer;
            if (layer >= 0) {
                if ((int) session.size() <= layer) {
                    session.resize((size_t) layer + 1);
                }
                auto & row = session[layer];
                if (row.size() < cnt.size()) {
                    row.resize(cnt.size(), 0.0);
                }
                for (size_t e = 0; e < cnt.size(); e++) {
                    row[e] += cnt[e];
                }
            }
        }
        std::vector<io_job_t> jobs;
        size_t max_nb2 = 0;
        // The gate, up and down tensors of a layer use the same experts. Some models (gpt-oss: bias operations in between) hand them to us in
        // separate calls; then load the siblings' experts in the same parallel batch now, instead of three small batches one after the other.
        std::vector<int> tl = tidx;
        const size_t n_counted = tidx.size();
        if (siblings) {
            auto lt = layer_tensors.find(tensors[tidx[0]]->layer);
            if (lt != layer_tensors.end()) {
                for (int t : lt->second) {
                    if (std::find(tl.begin(), tl.end(), t) == tl.end() && tensors[t]->n_expert == tensors[tidx[0]]->n_expert) {
                        tl.push_back(t);
                    }
                }
            }
        }
        for (size_t q = 0; q < tl.size(); q++) {
            const int ti_i = tl[q];
            const bool counted = q < n_counted;     // sibling tensors are loaded now, but their hits and misses are counted when their own call comes
            tinfo_t & ti = *tensors[ti_i];
            max_nb2 = std::max(max_nb2, ti.nb2);
            for (int64_t e = 0; e < ti.n_expert; e++) {
                if (!cnt[(size_t) e]) {
                    continue;
                }
                const int s = ti.node0 + (int) e;
                node_t & n = nodes[s];
                if (!counted) {
                    if (n.resident) {
                        if (n.loading) { wait_list.push_back(s); }
                        unlink(s);
                        push_mru(s);
                        n.epoch = epoch;
                        continue;
                    }
                    sibling_loads++;
                    while (resident_bytes + ti.nb2 > budget) {
                        int v = lru;
                        for (int k = 0; k < 100000 && v >= 0 && (nodes[v].epoch == epoch || nodes[v].loading); k++) {
                            v = nodes[v].prev;
                        }
                        if (v < 0 || nodes[v].epoch == epoch || nodes[v].loading) {
                            break;     // no room without evicting what this group needs: leave it to the sibling's own call
                        }
                        evict(v);
                    }
                    if (resident_bytes + ti.nb2 > budget) {
                        continue;
                    }
                    n.resident = true;
                    n.epoch = epoch;
                    resident_bytes += ti.nb2;
                    if (cur_scan) { push_lru(s); } else { push_mru(s); }
                    { io_job_t jb{ ti.data + (size_t) e * ti.nb2, ti.file_off + (size_t) e * ti.nb2, ti.nb2, ti.fdi, ti.direct, ti.repack ? ti.tensor : nullptr }; jb.node = s; if (async) { __atomic_store_n(&n.loading, true, __ATOMIC_RELEASE); } jobs.push_back(jb); }
                    pending_bytes += ti.nb2;
                    if (!async && auto_mode && pending_bytes >= ((size_t) 256 << 20)) {
                        flush_reads(jobs, max_nb2);
                    }
                    continue;
                }
                if (ti.layer >= 0) {
                    if ((size_t) ti.layer >= layer_hits.size()) {
                        layer_hits.resize((size_t) ti.layer + 1, 0);
                        layer_misses.resize((size_t) ti.layer + 1, 0);
                    }
                    (n.resident ? layer_hits : layer_misses)[(size_t) ti.layer]++;
                }
                if (n.resident) {
                    hits++;
                    if (cur_decode) { hits_dec++; }
                    if (n.prefetched) {
                        n.prefetched = false;
                        pf_used++;
                        if (n.loading) { pf_late++; }
                    }
                    if (n.loading) { wait_list.push_back(s); }
                    if (!cur_scan) {          // a long prompt uses (nearly) everything: it does not make an expert "recently used"
                        unlink(s);
                        push_mru(s);
                    }
                    n.epoch = epoch;
                    continue;
                }
                misses++;
                if (cur_decode) { misses_dec++; }
                bool fits = true;
                while (resident_bytes + ti.nb2 > budget) {
                    int v = lru;
                    for (int k = 0; k < 100000 && v >= 0 && (nodes[v].epoch == epoch || nodes[v].loading); k++) {
                        v = nodes[v].prev;
                    }
                    if (v < 0 || nodes[v].epoch == epoch || nodes[v].loading) {
                        fits = false;      // everything left is needed by this op: the cache is smaller than one op (fixed MOE_CACHE_SIZE_GIB)
                        break;
                    }
                    evict(v);
                }
                if (!fits) {
                    // fall back instead of aborting: map the expert from the model file, as stock llama.cpp does (file cache, reclaimable, the
                    // budget holds); where that is not possible (background reads, repacked or unaligned experts, Windows), load it over the budget
                    if (!small_warned) {
                        small_warned = true;
                        fprintf(stderr, "moe-cache: warning: the cache (%.2f GiB) is smaller than the experts of one op; experts that do not fit are %s. "
                                        "Raise MOE_CACHE_SIZE_GIB (or use auto) for full speed\n", budget / 1073741824.0,
                                !async && ti.direct ? "mapped from the model file (the stock way)" : "loaded over the cache size for that step");
                    }
                    if (!async && (n.mapped || map_from_file(ti, e, lk))) {
                        n.mapped = true;
                        n.epoch = epoch;
                        small_mapped++;
                        continue;
                    }
                    small_over++;
                }
                n.resident = true;
                n.epoch = epoch;
                resident_bytes += ti.nb2;
                if (cur_scan) { push_lru(s); } else { push_mru(s); }
                { io_job_t jb{ ti.data + (size_t) e * ti.nb2, ti.file_off + (size_t) e * ti.nb2, ti.nb2, ti.fdi, ti.direct, ti.repack ? ti.tensor : nullptr }; jb.node = s; if (async) { __atomic_store_n(&n.loading, true, __ATOMIC_RELEASE); } jobs.push_back(jb); }
                pending_bytes += ti.nb2;
                if (!async && auto_mode && pending_bytes >= ((size_t) 256 << 20)) {
                    flush_reads(jobs, max_nb2);
                }
            }
        }
        if (async) {
            // hand the reads to the reader threads and return: the executor waits for each expert's flag when it needs it
            const size_t njobs = jobs.size();
            if (!jobs.empty()) {
                {
                    std::lock_guard<std::mutex> ql(rq_mu);
                    for (const auto & j : jobs) {
                        rq.push_back(j);
                    }
                }
                rq_cv.notify_all();
            }
            return njobs;
        }
        flush_reads(jobs, max_nb2);
        for (int s : wait_list) {
            while (nodes[s].loading) {
                load_cv.wait(lk);
            }
        }
        return 0;
    }

    // read the queued experts, then (auto mode) look at the real memory use again
    void flush_reads(std::vector<io_job_t> & jobs, size_t max_nb2) {
        const bool did_reads = !jobs.empty();
        if (coalesce_bytes && jobs.size() > 1) {
            // long prompts need nearly every expert of a layer, and neighbouring experts of a tensor are neighbours in the file and in memory:
            // one large read instead of many half-megabyte ones (the drive reaches its speed only with large or many parallel reads)
            std::sort(jobs.begin(), jobs.end(), [](const io_job_t & a, const io_job_t & b) { return a.fdi != b.fdi ? a.fdi < b.fdi : a.off < b.off; });
            std::vector<io_job_t> merged;
            merged.reserve(jobs.size());
            for (const auto & j : jobs) {
                if (!merged.empty()) {
                    io_job_t & m = merged.back();
                    if (m.direct && j.direct && !m.rtensor && !j.rtensor && m.fdi == j.fdi && m.off + m.nb2 == j.off && m.dst + m.nb2 == j.dst &&
                        m.nb2 + j.nb2 <= coalesce_bytes) {
                        m.nb2 += j.nb2;
                        coalesced++;
                        continue;
                    }
                }
                merged.push_back(j);
            }
            jobs.swap(merged);
            for (const auto & j : jobs) {
                max_nb2 = std::max(max_nb2, j.nb2);
            }
        }
        if (!jobs.empty()) {
            const auto r0 = std::chrono::steady_clock::now();
            do_reads(jobs, max_nb2);
            ns_read += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - r0).count();
            jobs.clear();
        }
        pending_bytes = 0;
        if (auto_mode && did_reads) {   // nothing was read: memory did not change (prepare_group re-checks every 32 groups anyway); reading /proc each layer cost ~2 ms per token
            tune();
        }
    }
};

state_t & S() {
    static state_t s;
    return s;
}

}  // namespace
