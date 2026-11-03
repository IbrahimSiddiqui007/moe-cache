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

// MULTIGPU-026: false for an OpenVINO device (the NPU is reported as a GPU); see mc_devpick.h
bool block_device_ok(ggml_backend_dev_t d) {
    ggml_backend_reg_t r = d ? ggml_backend_dev_backend_reg(d) : nullptr;
    return mcdev::usable_for_gpu_block(r ? ggml_backend_reg_name(r) : nullptr, d ? ggml_backend_dev_description(d) : nullptr);
}
#ifdef _WIN32
void install_crash_report(state_t * s);
void remove_crash_report();
#endif

#include "mc_state_route.h"
#include "mc_state_testfault.h"
#include "mc_state_gpu.h"
#include "mc_state_pack_members.h"
#include "mc_state_policy.h"
#include "mc_state_rebalance_members.h"
#include "mc_state_exec.h"
#include "mc_state_times.h"
#include "mc_state_prefetch.h"
#include "mc_state_budget.h"
#include "mc_state_profile_members.h"
struct state_t : state_route_t, state_testfault_t, state_gpu_t, state_pack_t, state_policy_t, state_rebalance_t, state_exec_t, state_times_t, state_prefetch_t, state_budget_t, state_profile_t {
    bool   cache_on = false;
    bool   stats = false;
    int    lo = 0, hi = 1 << 30;
    // "all" for the default range (hi = 2^30 means no upper bound; the log printed "layers 0-1073741824", T7)
    std::string layer_range() const { return lo == 0 && hi >= (1 << 30) ? std::string("all") : std::to_string(lo) + "-" + std::to_string(hi); }
    std::vector<plat::file_t> fds;              // one per GGUF file (split models have several)
    std::vector<std::string> shard_paths;
    std::vector<plat::file_t> fds_buf;          // buffered (no O_DIRECT) handles, opened only if a direct read keeps failing
    std::mutex fds_buf_mu;
    std::vector<std::atomic<bool>> direct_broken;   // per file: direct reads fail on this file system (EINVAL/EOPNOTSUPP): buffered reads from then on (AUD-011)
    std::unordered_map<std::string, int> file_shard;
    ggml_backend_buffer_type_t repack_buft = nullptr;   // llama.cpp CPU_REPACK buffer type (mode on)
    ggml_backend_buffer_t repack_buf = nullptr;         // a real repack buffer, used for init_tensor / set_tensor only
    ggml_backend_buffer_t repack_tag = nullptr;         // empty buffer of the repack type, swapped in while an operation runs
    std::atomic<uint64_t> repack_jobs{0};
    // live statistics: with MOE_CACHE_STATS_SIGNAL=1, SIGUSR1 makes the plugin print one "moe-cache-stats {json}" line (cumulative counters)
    std::atomic<uint64_t> hits_dec{0}, misses_dec{0}, evict_age_sum{0};
    bool cur_decode = false;                            // the group being prepared is a single-token (decode) step
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
    bool   disabled = false;                    // setup failed or MOE_CACHE_DISABLE: the plugin claims nothing, the model loads normally
    bool   direct_ok = false;                   // read experts straight into place when the alignment allows
#ifdef _WIN32
    bool   win_buffered = false;                // model files opened without FILE_FLAG_NO_BUFFERING: MOE_CACHE_WIN_BUFFERED=1, or the private-memory path by default (see the constructor)
    bool   win_random = false;                  // MOE_CACHE_WIN_BUFFERED=2: and with FILE_FLAG_RANDOM_ACCESS
#else
    static constexpr bool win_buffered = false, win_random = false;
#endif
    bool   win_view = false;                    // file-view cache (default on Windows and Linux; MOE_CACHE_WIN_VIEW=0 / MOE_CACHE_LINUX_VIEW=0 turn it off; see the constructor)
    std::vector<const char *> views;            // win_view: one read-only view per model file (split GGUF)
    std::vector<size_t> view_sizes;
    size_t view_bytes = 0;                      // win_view: resident bytes of experts held in the views (not private memory: not in rss_private())
    size_t ws_min_cur = 0;                      // win_view: the hard working-set minimum set last (0 = none)
    bool   ws_min_warned = false;
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
    // idea C13 (MOE_CACHE_RESIDENCY_CHECK=N, Linux, default off): every N graphs, ask the kernel (mincore) whether 16 experts marked resident are really in RAM;
    // the exit line "residency check" reports how many were not (the OS reclaimed them under memory pressure). A diagnostic: it changes nothing else.
    uint64_t res_checked = 0, res_missing = 0, res_graphs = 0;
    void residency_sample() {
#ifdef __linux__
        static const int every = getenv("MOE_CACHE_RESIDENCY_CHECK") ? atoi(getenv("MOE_CACHE_RESIDENCY_CHECK")) : 0;
        if (every <= 0 || (++res_graphs % (uint64_t) every) != 0 || tensors.empty()) {
            return;
        }
        static uint64_t rng = 88172645463325252ull;
        for (int k = 0; k < 16; k++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            const tinfo_t & ti = *tensors[rng % tensors.size()];
            if (ti.n_expert <= 0) {
                continue;                      // a tensor forgotten after a model unload: no experts, and a modulo by zero below
            }
            const int64_t e = (int64_t) ((rng >> 20) % (uint64_t) ti.n_expert);
            if (!nodes[(size_t) (ti.node0 + e)].resident || nodes[(size_t) (ti.node0 + e)].loading) {
                continue;
            }
            const uintptr_t a0 = (uintptr_t) ti.data + (size_t) e * ti.nb2;
            const uintptr_t a = (a0 + PAGE - 1) / PAGE * PAGE, b = (a0 + ti.nb2) / PAGE * PAGE;
            if (b <= a) {
                continue;
            }
            std::vector<unsigned char> vec((b - a) / PAGE);
            if (mincore((void *) a, b - a, vec.data()) != 0) {
                continue;
            }
            size_t out = 0;
            for (unsigned char v : vec) { out += (v & 1) ? 0 : 1; }
            res_checked += vec.size();
            res_missing += out;
        }
#endif
    }

    void begin_graph() {
        std::lock_guard<std::mutex> lk(mu);
        residency_sample();
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
    // idea P20 (MOE_CACHE_PREFILL_LEND_MIB=N, default 0 = off): while long prompt batches run, the expert cache gives N MiB back to the system (the cache is cut by N and
    // the surplus evicted), and takes them back at the first small (decode) graph
    size_t lent_bytes = 0;
    size_t lent_budget = 0;                             // the budget right after the lend (a cut by tune() meanwhile is not given back twice, AUD-021)
    void prefill_lend(int64_t max_tok) {
        static const size_t lend = getenv("MOE_CACHE_PREFILL_LEND_MIB") ? (size_t) std::max(0, atoi(getenv("MOE_CACHE_PREFILL_LEND_MIB"))) << 20 : 0;
        if (!lend) {
            return;
        }
        std::lock_guard<std::mutex> lk(mu);
        if (max_tok >= scan_tokens && !lent_bytes && budget > lend) {
            lent_bytes = lend;
            budget -= lend;
            lent_budget = budget;
            while (resident_bytes > budget) {
                int v = pick_victim();
                if (v < 0) { break; }
                evict(v);
            }
            fprintf(stderr, "moe-cache: long prompt: the expert cache lends %.0f MiB to the system until decode (MOE_CACHE_PREFILL_LEND_MIB)\n", lend / 1048576.0);
        } else if (max_tok < scan_tokens && lent_bytes) {
            budget += mclend::restore_amount(lent_bytes, lent_budget, budget);
            if (max_budget && budget > max_budget) { budget = max_budget; }
            lent_bytes = 0;
        }
    }

    bool last_long = false;                             // idea C07: the last graph ended on a long prompt batch (under mu)
    std::atomic<uint64_t> last_end_ns{0};               // steady clock, end of the last graph
    bool   rb_acted = false;                            // the last round read something: measure the next graphs
    bool   rb_armed_dec = false, rb_armed_pre = false;  // count the hits of the first decode / prefill graph after a rebalance
    bool   g_saw_dec = false, g_saw_pre = false;
    uint64_t rb_rounds = 0, rb_reads = 0, rb_bytes_read = 0, rb_swaps = 0, rb_cancelled = 0, rb_evicted_bytes = 0;
    uint64_t rb_dec_hits = 0, rb_dec_miss = 0, rb_pre_hits = 0, rb_pre_miss = 0;
    uint64_t cls_promotions = 0, cls_demotions = 0;     // touches that moved a node to a higher / lower class list
    int    unmatched = 0;              // expert tensors that could not be cached (see register_tensor)

    std::mutex mu;

    std::atomic<uint64_t> hits{0}, misses{0}, bytes{0}, groups{0}, ops{0}, graphs{0};
    std::atomic<uint64_t> ns_prep{0}, ns_cpu{0}, ns_total{0}, ns_read{0}, evictions{0}, direct_jobs{0}, bounce_jobs{0}, async_in{0};

    // io workers
    std::vector<std::thread> workers;
    std::mutex wmu;
    std::condition_variable wcv, dcv;
    std::function<void(int)> task;
    int    gen = 0;
    int    busy = 0;
    bool   stop = false;

    state_t() {
        stats = mcenv::env_on("MOE_CACHE_STATS");
        fprintf(stderr, "moe-cache: plugin v0.3.5 (ggml backend API %d), running with ggml %s (%s)\n", GGML_BACKEND_API_VERSION, ggml_version(), ggml_commit());
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
            double d = 0.0;
            if (mcargs::parse_number(v, 0.0, 1.0, d)) {      // a decay is a factor 0..1; nan or text used to be written into the profile
                profile_decay = d;
            } else {
                fprintf(stderr, "moe-cache: ignoring MOE_CACHE_PROFILE_DECAY=%s (expected a number 0..1)\n", v);
            }
        }
        if (const char * v = getenv("MOE_CACHE_LAYERS")) {
            int a = 0, b = 0;
            if (sscanf(v, "%d-%d", &a, &b) == 2) {
                lo = a;
                hi = b;
            }
        }
        {
            const std::string sched = plat::sched_tuning();
            if (!sched.empty()) {
                fprintf(stderr, "moe-cache: scheduling: %s\n", sched.c_str());
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
        if (const char * dis = getenv("MOE_CACHE_DISABLE"); dis && *dis && strcmp(dis, "0") != 0) {      // =0 or empty means "not disabled"
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
        direct_broken = std::vector<std::atomic<bool>>(fds.size());
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
            if (!win_view) {            // a later file failed to map: nothing keeps the earlier views, so they must not stay mapped (and counted) beside the private-memory cache
                for (size_t vi = 0; vi < views.size(); vi++) {
                    plat::file_view_unmap(views[vi], view_sizes[vi]);
                }
                views.clear();
                view_sizes.clear();
            }
            if (win_view) {
                plat::count_file_cache() = false;      // the views' pages are counted as the cache's own (view_bytes), not as available file cache
                fprintf(stderr, "moe-cache: experts are kept in read-only views of the model file (%zu), held in RAM by a hard "
                                "working-set minimum; they use no commit (RAM + page file; MOE_CACHE_WIN_VIEW=0: private memory)\n", views.size());
            }
        }
        // Private memory (MOE_CACHE_WIN_VIEW=0, the expert pack, a failed view map): read through the Windows file cache by default too, as the views do.
        // Flash-Next Q2_0 on the PC, ABBA: reader speed 0.26 -> 0.84 GB/s, decode +39 %, prompts 2x, text identical (RESULTS 75). MOE_CACHE_WIN_BUFFERED=0: unbuffered.
        if (!win_view && !win_buffered && !getenv("MOE_CACHE_WIN_BUFFERED")) {
            // all shards or none: a mix of buffered and unbuffered handles would make the budget (count_file_cache) wrong for the unbuffered ones
            std::vector<plat::file_t> nf;
            for (size_t si = 0; si < fds.size(); si++) {
                plat::file_t f = plat::file_open(shard_paths[si].c_str(), false, win_random);
                if (!f.ok()) {
                    break;
                }
                nf.push_back(f);
            }
            if (nf.size() == fds.size()) {
                for (size_t si = 0; si < fds.size(); si++) {
                    CloseHandle(fds[si].h);
                    fds[si] = nf[si];
                }
                win_buffered = true;
                plat::count_file_cache() = true;
            } else {
                for (plat::file_t & f : nf) {
                    CloseHandle(f.h);
                }
                fprintf(stderr, "moe-cache: private-memory cache: cannot reopen the model files through the Windows file cache: reads stay unbuffered\n");
            }
            if (win_buffered) {
                fprintf(stderr, "moe-cache: private-memory cache: experts are read through the Windows file cache (MOE_CACHE_WIN_BUFFERED=0: unbuffered)\n");
            }
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
            if (!win_view) {            // a later file failed to map: nothing keeps the earlier views, so they must not stay mapped (and counted) beside the private-memory cache
                for (size_t vi = 0; vi < views.size(); vi++) {
                    plat::file_view_unmap(views[vi], view_sizes[vi]);
                }
                views.clear();
                view_sizes.clear();
            }
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
            if ((t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) && block_device_ok(d)) {
                gpu_dev = d;
            }
        }
        for (size_t i = 0; i < ggml_backend_dev_count() && gpu_min_tokens > 0 && !sa.device_none && sa.devices.empty(); i++) {
            // a discrete GPU first, else an integrated one (ggml-vulkan reports APUs as IGPU; stock llama.cpp offloads to them too).
            // Taking only GPU left the Vega 8 unused: prefill 363 vs stock 693 tok/s (mini PC, T7); with the iGPU 569-597
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const auto t = ggml_backend_dev_type(d);
            if (!block_device_ok(d)) {
                continue;        // MULTIGPU-026: an OpenVINO device (the NPU is reported as a GPU) is never the long-prompt block target
            }
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU) {
                gpu_dev = d;
                break;
            }
            if (t == GGML_BACKEND_DEVICE_TYPE_IGPU && !gpu_dev) {
                gpu_dev = d;
            }
        }
        // N3/MG5 (MOE_CACHE_GPU_DEV2=auto|1|DEVICE-NAME, default off): a second card for the long-prompt block. auto: the next discrete GPU of the same backend as the
        // first (an iGPU only when named); a name: that device. Used only for a block that does not fit on the first card but fits on the second. Never run on two discrete cards.
        if (const char * v = getenv("MOE_CACHE_GPU_DEV2"); v && *v && strcmp(v, "0") != 0 && gpu_dev) {
            const bool any = strcmp(v, "1") == 0 || strcmp(v, "auto") == 0;
            const mcargs::server_args & sa2 = server_args();
            if (!any) {
                gpu_dev2 = ggml_backend_dev_by_name(v);
            } else {
                for (size_t k = 0; k < ggml_backend_dev_count() && !gpu_dev2; k++) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(k);
                    bool listed = sa2.devices.empty();       // with --device A,B only those
                    for (const auto & nm : sa2.devices) {
                        listed = listed || nm == ggml_backend_dev_name(d);
                    }
                    if (d != gpu_dev && listed && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && ggml_backend_dev_backend_reg(d) == ggml_backend_dev_backend_reg(gpu_dev)) {
                        gpu_dev2 = d;
                    }
                }
            }
            if (gpu_dev2 == gpu_dev || (gpu_dev2 && ggml_backend_dev_type(gpu_dev2) == GGML_BACKEND_DEVICE_TYPE_CPU)) {
                gpu_dev2 = nullptr;
            }
            if (gpu_dev2) {
                fprintf(stderr, "moe-cache: MOE_CACHE_GPU_DEV2 (PROTOTYPE, not run on two discrete cards): a long-prompt block that does not fit on %s runs on %s when it fits there\n",
                        ggml_backend_dev_name(gpu_dev), ggml_backend_dev_name(gpu_dev2));
            } else {
                fprintf(stderr, "moe-cache: MOE_CACHE_GPU_DEV2=%s: no second card found: first card only\n", v);
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
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    const auto t = ggml_backend_dev_type(d);
                    gpu = gpu || ((t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) && block_device_ok(d));   // MULTIGPU-026: an NPU is not a GPU that holds llama.cpp's CPU weights
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
        if (mcargs::prompt_cache_unlimited_default(plat::command_line(), getenv("LLAMA_ARG_CACHE_RAM"), getenv("MOE_CACHE_HOST_RESERVE_MIB"))) {
            fprintf(stderr, "moe-cache: note: llama-server's own prompt cache (--cache-ram, 8 GiB by default) is not counted in this cache's RAM budget; "
                            "to limit it pass --cache-ram MiB to llama-server (moe-cache-server does this for you)\n");
        }
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
        peers_start();
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
        std::string src = !profile_path.empty() ? profile_path : warm_path;
        if (!src.empty()) {
            uint64_t fp = 0;
            bool loaded = load_usage(src.c_str(), learned, &fp);
            if (loaded && fp != 0 && fp != fingerprint && !profile_path.empty()) {
                // AUD-016: the file at MOE_CACHE_PROFILE belongs to another model: keep it, and learn in (and warm start from) a file of our own next to it
                const std::string mine = mcprofile::own_path(profile_path, fingerprint);
                fprintf(stderr, "moe-cache: profile %s belongs to another model: kept as it is; this model uses %s\n", profile_path.c_str(), mine.c_str());
                profile_path = mine;
                src = mine;
                learned = usage_t();
                fp = 0;
                loaded = load_usage(src.c_str(), learned, &fp);
            }
            if (loaded) {
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
            exec_fuse = mcenv::flag_value(v, true);
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
        if (const char * v = getenv("MOE_CACHE_TILED_CACHE")) {
            tiled_cache = strcmp(v, "0") != 0;
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
        exec_pool_decode = mcenv::env_on("MOE_CACHE_EXEC_POOL_DECODE");
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
            lookahead = mcenv::flag_value(v, false);
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
            if (plat::wake_pipe_create(g_stats_pipe)) {
                struct sigaction sa;
                memset(&sa, 0, sizeof(sa));
                sa.sa_handler = [](int) { const char c = 1; const ssize_t r = write(g_stats_pipe[1], &c, 1); (void) r; };   // async-signal-safe: only write()
                sa.sa_flags = SA_RESTART;
                sigaction(SIGUSR1, &sa, nullptr);
                stats_th = std::thread([this] { char c; while (read(g_stats_pipe[0], &c, 1) == 1 && !g_stats_stop) { print_snapshot(); } });   // joined in stop_threads
            }
        }
        // idle release (N3): the launcher's on-demand listener sends SIGUSR2 when the server has been idle for a while
        if (cache_on && plat::wake_pipe_create(g_idle_pipe)) {
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
#include "mc_state_peers.inl"
};

// The state lives on the heap; a static guard ends it at exit. Normally (threads alive) it is deleted: threads joined, profile saved. When the
// process is exiting on Windows, the other threads are already gone, or (both systems) exit came while a graph held the cache lock, the state is
// only asked to report (see exit_without_threads).
struct state_exit_t {
    state_t * s;
    ~state_exit_t() {
        if (plat::process_exiting()) {
            s->exit_without_threads();
            return;
        }
        const bool lock_free = s->mu.try_lock();
        int active = 0;
        if (lock_free) {
            active = s->graph_active;
            s->mu.unlock();
        }
        if (mcexit::choose(false, lock_free, active) == mcexit::action::delete_state) {
            delete s;
        } else {
            // exit() called while a graph is running (Linux: llama-server's second Ctrl+C calls exit(1) from its signal handler, in the middle of a
            // prompt; the cache lock is held for short steps only, so graph_active is tested too, AUD-013). Joining the readers would wait for ever
            // (they need the lock) and deleting would pull the state from under the compute threads: leave the state as the Windows exit does;
            // the system takes the memory back (audit 2026-10-29 N2)
            fprintf(stderr, "moe-cache: exit during a computation: threads left to the system, usage profile saved without waiting\n");
            s->exit_without_threads();
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
std::atomic<state_t *> g_crash_state{nullptr};
PVOID g_crash_veh = nullptr;
std::atomic<int> g_crash_said{0};
LPTOP_LEVEL_EXCEPTION_FILTER g_prev_filter = nullptr;

bool crash_report(EXCEPTION_POINTERS * ep, bool experts_only) {
    const EXCEPTION_RECORD * er = ep ? ep->ExceptionRecord : nullptr;
    state_t * s = g_crash_state.load();
    if (!er || !s || er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || er->NumberParameters < 2) {
        return false;
    }
    const uintptr_t addr = (uintptr_t) er->ExceptionInformation[1];
    const tinfo_t * hit = nullptr;
    // WINDOWS-081: the tensor list changes under s->mu (registration); the handler never waits for that lock (the crashing thread may hold it), it skips the lookup
    std::unique_lock<std::mutex> tl(s->mu, std::try_to_lock);
    for (const auto & t : s->tensors) {
        if (!tl.owns_lock()) {
            break;
        }
        // a forgotten tensor (its buffer was freed) has data == nullptr and no experts: it would claim every address below one page
        if (t && t->data && t->n_expert > 0 && addr >= (uintptr_t) t->data && addr < (uintptr_t) t->data + t->nb2 * (size_t) t->n_expert + PAGE) {
            hit = t.get();
            break;
        }
    }
    if (tl.owns_lock()) {
        tl.unlock();
    }
    if ((experts_only && !hit) || g_crash_said.exchange(1) != 0) {
        return false;
    }
    // WINDOWS-081: from here on only the first crashing thread runs (the exchange above), so the large buffers are static: the handler itself used about 5.5 KB of
    // the faulting thread's stack, which may be the very thing that ran out
    static char mname[MAX_PATH];
    static char fname[MAX_PATH];
    static char buf[4096];
    static CONTEXT ctx;
    const uintptr_t pc = (uintptr_t) er->ExceptionAddress;
    HMODULE mod = nullptr;
    strcpy(mname, "?");
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) pc, &mod)) {
        GetModuleFileNameA(mod, mname, MAX_PATH);
    }
    const char * base = std::max(strrchr(mname, '\\'), strrchr(mname, '/'));
    base = base ? base + 1 : mname;
    const ULONG_PTR kind = er->ExceptionInformation[0];
    int n = snprintf(buf, sizeof(buf), "moe-cache: crash: access violation at %s+0x%llx, %s address 0x%llx", base,
                     (unsigned long long) (pc - (uintptr_t) mod), kind == 0 ? "reading" : kind == 1 ? "writing" : "executing", (unsigned long long) addr);
    if (hit) {
        const size_t rel = addr - (uintptr_t) hit->data;
        const int64_t e = std::max<int64_t>(0, std::min<int64_t>((int64_t) (rel / hit->nb2), hit->n_expert - 1));
        const node_t & nd = s->nodes[(size_t) (hit->node0 + e)];
        n = mccat::cat(buf, sizeof(buf), n, " = %s expert %lld (byte %zu of %zu): resident=%d loading=%d mapped=%d epoch=%llu (now %llu, protected from %llu); "
                      "cache %.2f of %.2f GiB, %llu evictions", hit->name.c_str(), (long long) e, rel - (size_t) e * hit->nb2, hit->nb2, (int) nd.resident,
                      (int) nd.loading, (int) nd.mapped, (unsigned long long) nd.epoch, (unsigned long long) s->epoch,
                      (unsigned long long) (s->protect_epoch == UINT64_MAX ? 0 : s->protect_epoch), s->resident_bytes / 1073741824.0,
                      s->budget / 1073741824.0, (unsigned long long) s->evictions);
    } else {
        n = mccat::cat(buf, sizeof(buf), n, " (not in the expert cache)");
    }
    // was the plugin handing a graph to the CPU backend? If so, its nodes without data (a NULL base + offset write points there)
    if (ggml_cgraph * g = g_cpu_graph.load()) {
        const int nn = ggml_graph_n_nodes(g);
        int nulls = 0;
        n = mccat::cat(buf, sizeof(buf), n, "\nmoe-cache: crash: inside a plugin graph handed to the CPU backend (%d nodes)", nn);
        for (int i = 0; i < nn && n < (int) sizeof(buf) - 200; i++) {
            const ggml_tensor * t = ggml_graph_node(g, i);
            const bool null_src = (t->src[0] && !t->src[0]->data) || (t->src[1] && !t->src[1]->data);
            if (t->op != GGML_OP_NONE && (!t->data || null_src) && nulls++ < 8) {
                n = mccat::cat(buf, sizeof(buf), n, "; node %d %s (%s) data=%p src0=%p src1=%p ne=%lld,%lld,%lld", i, t->name, ggml_op_desc(t),
                              t->data, t->src[0] ? t->src[0]->data : nullptr, t->src[1] ? t->src[1]->data : nullptr,
                              (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
            }
        }
        if (!nulls) {
            n = mccat::cat(buf, sizeof(buf), n, "; no node or source without data");
        }
    } else {
        n = mccat::cat(buf, sizeof(buf), n, "\nmoe-cache: crash: not inside a graph the plugin handed to the CPU backend");
    }
    // inside executor work (a pool worker or the compute thread's share): which op, task, row, column and output it was computing
    if (const exec_where_t & wh = g_exec_where; wh.op) {
        n = mccat::cat(buf, sizeof(buf), n, "\nmoe-cache: crash: inside the executor: op %s, task %p", wh.op, (const void *) wh.task);
        if (wh.task) {
            n = std::min<int>(n, (int) sizeof(buf) - 2);
            n = mccat::cat(buf, sizeof(buf), n, " (weights %p, %d columns, %lld rows), row %lld, column %d, dst %p",
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
        ctx = *ep->ContextRecord;
        n = mccat::cat(buf, sizeof(buf), n, "moe-cache: crash: stack");
        for (int f = 0; f < 20 && ctx.Rip && n < (int) sizeof(buf) - 120; f++) {
            HMODULE fm = nullptr;
            strcpy(fname, "?");
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) ctx.Rip, &fm)) {
                GetModuleFileNameA(fm, fname, MAX_PATH);
            }
            const char * fb = std::max(strrchr(fname, '\\'), strrchr(fname, '/'));
            n = mccat::cat(buf, sizeof(buf), n, " %s+0x%llx", fb ? fb + 1 : fname, (unsigned long long) (ctx.Rip - (DWORD64) (uintptr_t) fm));
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
    g_crash_veh = AddVectoredExceptionHandler(0, crash_veh);
    g_prev_filter = SetUnhandledExceptionFilter(crash_filter);
}
// WINDOWS-081: when the state is deleted (an orderly unload) the handlers go with it; a crash after that is not ours to report on freed memory
void remove_crash_report() {
    g_crash_state = nullptr;
    if (g_crash_veh) {
        RemoveVectoredExceptionHandler(g_crash_veh);
        g_crash_veh = nullptr;
    }
    SetUnhandledExceptionFilter(g_prev_filter);        // what was there before us (null: the system default)
    g_prev_filter = nullptr;
}
#endif

}  // namespace
