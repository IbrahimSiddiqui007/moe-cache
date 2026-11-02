// moe-cache plugin part: backend context and the long-prompt GPU block.
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

// ---- backend ----

// The CPU backend is reached through the registry only, so this works with builds that load their backends as separate libraries.
struct cpu_api_t {
    ggml_backend_dev_t dev = nullptr;
    void              (*set_n_threads)(ggml_backend_t, int) = nullptr;
    ggml_threadpool_t (*tp_new)(const ggml_threadpool_params *) = nullptr;
    void              (*tp_free)(ggml_threadpool_t) = nullptr;
    void              (*set_tp)(ggml_backend_t, ggml_threadpool_t) = nullptr;
};

cpu_api_t & CPU() {
    static cpu_api_t a = [] {
        cpu_api_t r;
        r.dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!r.dev) {
            fprintf(stderr, "moe-cache: no CPU backend is loaded\n");
            abort();
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(r.dev);
        r.set_n_threads = (decltype(r.set_n_threads)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        r.tp_new  = (decltype(r.tp_new))  ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
        r.tp_free = (decltype(r.tp_free)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
        r.set_tp  = (decltype(r.set_tp))  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        return r;
    }();
    return a;
}

// CPUs to pin the plugin's compute threads to, best first: one thread per performance core, then their hyper-threading siblings.
// Only on hybrid Intel CPUs (performance + efficiency cores), where a worker on an efficiency core or on a busy sibling slows every layer.
// MOE_CACHE_CPU_PIN=off disables it; MOE_CACHE_CPU_MASK=hex picks the CPUs yourself (bit i = CPU i).
// CPUs for the executor's threads, best first. MOE_CACHE_CPU_MASK (hex) wins; the launcher sets it on Windows from the cores' efficiency class
// (Claw finding 5: Windows had no pinning); on Linux the performance cores of a hybrid Intel CPU come from sysfs.
std::vector<int> pin_cpus() {
    std::vector<int> out;
    const char * off = getenv("MOE_CACHE_CPU_PIN");
    if (off && strcmp(off, "off") == 0) {
        return out;
    }
    if (const char * m = getenv("MOE_CACHE_CPU_MASK")) {
        return mask_cpus(m);
    }
#ifdef __linux__
    auto parse = [](const std::string & txt) {
        std::vector<int> r;
        size_t i = 0;
        while (i < txt.size()) {
            int a = atoi(txt.c_str() + i), b = a;
            while (i < txt.size() && isdigit((unsigned char) txt[i])) { i++; }
            if (i < txt.size() && txt[i] == '-') {
                i++;
                b = atoi(txt.c_str() + i);
                while (i < txt.size() && isdigit((unsigned char) txt[i])) { i++; }
            }
            for (int c = a; c <= b && c < GGML_MAX_N_THREADS; c++) { r.push_back(c); }
            while (i < txt.size() && !isdigit((unsigned char) txt[i])) { i++; }
        }
        return r;
    };
    auto slurp = [](const std::string & fn) {
        std::string t;
        if (FILE * f = fopen(fn.c_str(), "r")) {
            char b[256];
            while (fgets(b, sizeof(b), f)) { t += b; }
            fclose(f);
        }
        return t;
    };
    const std::vector<int> pcpus = parse(slurp("/sys/devices/cpu_core/cpus"));   // exists only on hybrid Intel CPUs
    std::vector<int> primary, secondary;
    for (int c : pcpus) {
        const std::vector<int> sib = parse(slurp("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
        (!sib.empty() && sib[0] != c ? secondary : primary).push_back(c);
    }
    out = primary;
    out.insert(out.end(), secondary.begin(), secondary.end());
#endif
    return out;
}

// idea PL05 (MOE_CACHE_GPU_PREFETCH=1, default off, docs/PREFETCH_DESIGN.md): the whole expert tensors of the block that is expected next are copied into a
// second GPU buffer by one helper thread while llama.cpp runs its own work between two blocks. The helper only copies (no graph work); the block that uses
// the tensors finds them already in place, any tensor that is not there is uploaded as before.
struct gpu_prefetch_t {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<const ggml_tensor *> src;       // what the buffer holds (or is being filled with), same order as dst
    std::vector<ggml_tensor *> dst;
    size_t bytes = 0;
    std::unordered_map<const ggml_tensor *, std::vector<const ggml_tensor *>> next_of;   // first weight tensor of a block -> weight tensors of the block that followed it
    const ggml_tensor * last_first = nullptr;
    uint64_t retry_at = 0;                      // no room (or the card is tightening): not tried again until this many graphs have run
    bool off = false;                           // turned itself off (too many wrong guesses)
    uint64_t launched = 0, used = 0, failed = 0;
    double wait_ns = 0;                         // graph thread waiting for the helper at the start of a block
    std::atomic<uint64_t> busy_ns{0};           // helper busy time
    std::atomic<uint64_t> copied_bytes{0};
    // helper thread
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::function<bool()> job;
    bool has_job = false, running = false, stop = false, result = false;
    bool awaiting = false;                      // a job was launched and its end has not been taken yet (graph thread only)
    bool filled = false;                        // the last taken job copied everything
    // takes the end of the last job (the wait is booked); afterwards `filled` says whether the buffer holds src
    void settle() {
        if (!awaiting) {
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        filled = join();
        wait_ns += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
        awaiting = false;
        if (!filled) {
            failed++;
        }
    }
    bool pending() {
        std::lock_guard<std::mutex> lk(mu);
        return has_job || running;
    }
    void launch(std::function<bool()> f) {
        {
            std::lock_guard<std::mutex> lk(mu);
            job = std::move(f);
            has_job = true;
        }
        if (!th.joinable()) {
            th = std::thread([this] {
                std::unique_lock<std::mutex> lk(mu);
                for (;;) {
                    cv.wait(lk, [this] { return has_job || stop; });
                    if (stop) {
                        return;
                    }
                    std::function<bool()> j = std::move(job);
                    has_job = false;
                    running = true;
                    lk.unlock();
                    bool ok = false;
                    try {
                        ok = j();
                    } catch (const std::exception &) {
                        ok = false;
                    }
                    lk.lock();
                    result = ok;
                    running = false;
                    cv.notify_all();
                }
            });
        }
        cv.notify_all();
    }
    // waits for the last job; true when it copied everything
    bool join() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [this] { return !has_job && !running; });
        return result;
    }
    void release() {        // frees the buffer (the caller has joined and is sure no block still reads it)
        if (buf) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        src.clear();
        dst.clear();
        bytes = 0;
    }
    void shutdown() {
        if (th.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mu);
                stop = true;
            }
            cv.notify_all();
            th.join();
        }
        release();
    }
};

struct moe_cache_backend_ctx {
    ggml_backend_t     cpu = nullptr;
    ggml_threadpool_t  tp = nullptr;
    int                tp_n = 0;
    ggml_threadpool_t  tp2 = nullptr;   // the previous threadpool, kept: llama.cpp alternates n_threads (decode) and n_threads_batch (prompts)
    int                tp2_n = 0;
    int                n_threads = 6;
    ggml_backend_t     gpu = nullptr;
    std::vector<ggml_backend_t> async_src;   // N15: backends with reads into our inputs still queued (cpy_tensor_async); synchronized before use
    ggml_gallocr_t     galloc = nullptr;
    bool               gpu_failed = false;   // the GPU backend could not be started: no GPU blocks for good
    uint64_t           gpu_retry_at = 0;     // a GPU allocation failed: long-prompt blocks stay on the CPU until this many graphs have run
    int                gpu_alloc_fails = 0;
    bool               pin_reported = false;
    // MOE_CACHE_HYBRID=MiB: a bank of the most used experts in GPU memory (from the usage profile); per weight tensor: bank tensor and slot of each expert
    bool               hb_tried = false;
    ggml_context *     hb_ctx = nullptr;
    ggml_backend_buffer_t hb_buf = nullptr;
    bool               hb_dropped = false;   // the bank was freed so a long prompt's GPU block fits; it is loaded again at the next decode step
    bool               seg_nomem = false;    // the last gpu_segment() call failed for lack of free GPU memory
    ggml_backend_buffer_t rb_buf = nullptr;  // N22: the long-prompt block's result is read back into this pinned host buffer (the GPU device's own type)
    // idea P07 (MOE_CACHE_GPU_TMPL=1): finished long-prompt block graphs, kept by shape signature and reused by later blocks of the same shape
    struct gpu_tmpl_t {
        std::string key;
        ggml_context * ctx = nullptr;
        ggml_cgraph * g = nullptr;
        std::vector<ggml_tensor *> node_clones;     // the clone of each segment node, in segment order
        std::vector<ggml_tensor *> ext_clones;      // the input clones, in order of first use
    };
    std::vector<gpu_tmpl_t> tmpls;
    // idea P11 (MOE_CACHE_GPU_KEEP_MIB=N, default 0 = off): whole expert tensors uploaded for a block stay in GPU memory (up to N MiB, least recently used out)
    // and a later block that needs the same tensor (the next micro-batch of the same prompt) skips the upload: model weights never change, so a held copy is valid
    struct gpu_held_t {
        const ggml_tensor * src = nullptr;
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        size_t bytes = 0;
        bool filled = false;
        uint64_t used = 0;
    };
    std::vector<gpu_held_t> held;
    uint64_t held_serial = 0;
    gpu_prefetch_t pf;                       // PL05: next block's weights
    ggml_backend_buffer_t up_buf = nullptr;  // P05: pinned staging area for the weights of one part
    bool               rb_failed = false;    // no pinned host buffer type, or it could not be allocated: plain blocking read-back
    // N3/MG5 (MOE_CACHE_GPU_DEV2, default off): the second card's copy of everything above that is tied to one GPU. gpu_segment() swaps the two sets (card_swap)
    // when a block does not fit on the first card but fits on the second, and swaps back when it ends, so the rest of the code never sees the second card.
    struct card_t {
        ggml_backend_t gpu = nullptr;
        ggml_gallocr_t galloc = nullptr;
        ggml_backend_buffer_t rb_buf = nullptr, up_buf = nullptr;
        bool rb_failed = false;
        std::vector<gpu_tmpl_t> tmpls;
        std::vector<gpu_held_t> held;
        uint64_t held_serial = 0;
    } alt;
    bool               on_alt = false;       // the sets are swapped now (inside gpu_segment only)
    bool               alt_failed = false;   // the second card's backend could not be started: it is not tried again
    // MOE_CACHE_HYBRID_ADAPT=N,K (default 8,16; 0 = off): every N tokens up to K of the experts the bank missed most replace the coldest expert of their layer's bank
    struct hb_layer_t {
        std::vector<int>   tidx;       // weight tensors of this layer that have a bank (gate, up, down)
        std::vector<int>   slot_exp;   // expert held by each slot
        std::vector<float> score;      // decayed hits per slot
        std::vector<float> miss;       // misses per expert since the last swap
        std::vector<char>  base;       // the experts the profile chose for this layer (the shadow bank)
    };
    // MOE_CACHE_HYBRID_DUEL=W (default 32, 0 = off): count what the profile's own bank would have served (the shadow) next to the real bank over the last W
    // tokens; while the shadow is ahead (the session matches the profile) no new swaps, and profile experts come back instead
    int                duel_w = 32;
    int                ha_cur = 0, hb_cur = 0;
    std::deque<std::pair<int, int>> duel;
    std::map<int, hb_layer_t> hbl;
    int                hb_first = -1;          // first layer with a bank: one token has passed when it comes round again
    uint64_t           hb_tok = 0;
    int                ad_every = 8, ad_swaps = 16;
    std::unordered_map<int, std::pair<ggml_tensor *, std::vector<int>>> hb;
};

const char * be_name(ggml_backend_t) {
    return "MOE_CACHE";
}

void be_free(ggml_backend_t backend) {
    auto * c = (moe_cache_backend_ctx *) backend->context;
    if (c->cpu) {
        ggml_backend_free(c->cpu);
    }
    if (c->tp && CPU().tp_free) {
        CPU().tp_free(c->tp);
    }
    if (c->tp2 && CPU().tp_free) {
        CPU().tp_free(c->tp2);
    }
    for (auto & t : c->tmpls) {
        ggml_free(t.ctx);
    }
    c->pf.shutdown();                 // the helper first, then its GPU buffer (before the backend)
    for (auto & h : c->held) {      // GPU buffers are freed before their backend
        ggml_backend_buffer_free(h.buf);
        ggml_free(h.ctx);
    }
    if (c->galloc) {
        ggml_gallocr_free(c->galloc);
    }
    if (c->hb_buf) {                  // hybrid bank (GPU memory): freed before its backend
        ggml_backend_buffer_free(c->hb_buf);
    }
    if (c->hb_ctx) {
        ggml_free(c->hb_ctx);
    }
    if (c->up_buf) {
        ggml_backend_buffer_free(c->up_buf);
    }
    if (c->rb_buf) {                  // pinned read-back buffer: freed before its backend
        ggml_backend_buffer_free(c->rb_buf);
    }
    for (auto & t : c->alt.tmpls) {         // the second card (MOE_CACHE_GPU_DEV2): same order, buffers before their backend
        ggml_free(t.ctx);
    }
    for (auto & h : c->alt.held) {
        ggml_backend_buffer_free(h.buf);
        ggml_free(h.ctx);
    }
    if (c->alt.galloc) {
        ggml_gallocr_free(c->alt.galloc);
    }
    if (c->alt.up_buf) {
        ggml_backend_buffer_free(c->alt.up_buf);
    }
    if (c->alt.rb_buf) {
        ggml_backend_buffer_free(c->alt.rb_buf);
    }
    if (c->alt.gpu) {
        ggml_backend_free(c->alt.gpu);
    }
    if (c->gpu) {
        ggml_backend_free(c->gpu);
    }
    delete c;
    delete backend;
}

// N3/MG5: exchange the first card's GPU state with the second card's (see card_t). Calling it twice puts everything back.
void card_swap(moe_cache_backend_ctx * c) {
    std::swap(c->gpu, c->alt.gpu);
    std::swap(c->galloc, c->alt.galloc);
    std::swap(c->rb_buf, c->alt.rb_buf);
    std::swap(c->up_buf, c->alt.up_buf);
    std::swap(c->rb_failed, c->alt.rb_failed);
    std::swap(c->tmpls, c->alt.tmpls);
    std::swap(c->held, c->alt.held);
    std::swap(c->held_serial, c->alt.held_serial);
    c->on_alt = !c->on_alt;
}

// gives the first card's state back when gpu_segment() ends, whichever way it ends
struct card_guard {
    moe_cache_backend_ctx * c;
    ~card_guard() {
        if (c->on_alt) {
            card_swap(c);
        }
    }
};

// llama.cpp gives a GPU backend only the nodes it supports; the plugin's own GPU graphs must ask too, else the backend aborts on such a node
// (ggml-vulkan refuses for example MUL_MAT_ID at GGML_PREC_F32, or when its shared memory is too small for a type)
bool gpu_graph_supported(moe_cache_backend_ctx * c, ggml_cgraph * g) {
    for (int i = 0; i < ggml_graph_n_nodes(g); i++) {
        const ggml_tensor * n = ggml_graph_node(g, i);
        if (!ggml_backend_supports_op(c->gpu, n)) {
            static std::atomic<int> said{0};
            if (said++ == 0) {
                fprintf(stderr, "moe-cache: the GPU backend does not support %s with %s weights: such expert blocks stay on the CPU\n", ggml_op_desc(n),
                        n->src[0] ? ggml_type_name(n->src[0]->type) : "-");
            }
            return false;
        }
    }
    return true;
}

// N23-F3: experts e0..e0+n of weight t may still be filled by the reader threads (cache fill queued, not waited for): wait for their flags
static void wait_experts(state_t & s, const ggml_tensor * t, int64_t e0, int64_t n) {
    const auto it = s.by_tensor.find(t);
    if (it == s.by_tensor.end()) {
        return;
    }
    const tinfo_t & ti = *s.tensors[it->second];
    for (int64_t e = e0; e < e0 + n && e < ti.n_expert; e++) {
        const bool * f = &s.nodes[(size_t) (ti.node0 + e)].loading;
        while (__atomic_load_n(f, __ATOMIC_ACQUIRE)) {
            std::this_thread::yield();
        }
    }
}

// R2-05 (MOE_CACHE_VRAM_LINE=1, default off, log only): the lowest free GPU memory seen before a long-prompt block, and the block compute times of the prompt.
// One line when the prompt ends (the first decode graph after blocks), and one warning per process when a block takes more than 3x the median of the first
// 20 blocks while less than 600 MiB is free (a nearly full card: on Windows the driver may move memory to system RAM and the GPU waits). Reads the same
// ggml_backend_dev_memory call the block already makes; changes nothing it computes.
struct vram_watch_t {
    bool on = getenv("MOE_CACHE_VRAM_LINE") && atoi(getenv("MOE_CACHE_VRAM_LINE")) > 0;
    std::mutex mu;
    size_t min_free = (size_t) -1, first_free = 0, total = 0;
    std::vector<float> ms;            // compute time (ms) of each block of this prompt
    double first_med = -1;            // median of the first 20 blocks of the process
    bool warned = false;
    std::atomic<int> pending{0};      // blocks since the last line
    void block(size_t fr, size_t tot, double comp_ms) {
        std::lock_guard<std::mutex> lk(mu);
        if (tot == 0) { return; }
        if (min_free == (size_t) -1) { first_free = fr; }
        min_free = std::min(min_free, fr);
        total = tot;
        if (ms.size() < 1000000) { ms.push_back((float) comp_ms); }
        if (first_med < 0 && ms.size() == 20) { first_med = median(ms); }
        if (!warned && first_med > 0 && ms.size() > 20 && comp_ms > 3 * first_med && fr < ((size_t) 600 << 20)) {
            warned = true;
            fprintf(stderr, "moe-cache: WARNING: a long-prompt block took %.0f ms (3x the %.0f ms median of the first 20) with only %zu MiB of GPU memory free: "
                    "the card is nearly full; --n-cpu-moe +4 frees about 2.8 GB (or a smaller -ub)\n", comp_ms, first_med, fr >> 20);
        }
        pending++;
    }
    static double median(std::vector<float> v) {
        if (v.empty()) { return 0; }
        const size_t m = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + (long) m, v.end());
        return v[m];
    }
    void end_prompt() {
        if (!on || pending.load() == 0) { return; }
        std::lock_guard<std::mutex> lk(mu);
        pending = 0;
        const double worst = *std::max_element(ms.begin(), ms.end());
        fprintf(stderr, "moe-cache: VRAM: min free %zu MiB of %zu MiB (%zu MiB free at the first block); %zu long-prompt blocks, compute median %.0f ms, worst %.0f ms\n",
                min_free >> 20, total >> 20, first_free >> 20, ms.size(), median(ms), worst);
        ms.clear();
        min_free = (size_t) -1;
    }
};
static vram_watch_t g_vram_watch;

// PL05: the helper thread calls ggml_backend_tensor_set while the graph thread uses the same device. Checked: Vulkan at the pinned build (a write takes the device mutex,
// docs/PREFETCH_DESIGN.md) and CUDA (a per-thread stream). Anything else, or a ggml build the plugin was not tested with, keeps the switch off.
static bool pf_backend_ok(ggml_backend_dev_t d) {
    static std::atomic<int> said{0};
    const char * rn = d ? ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)) : nullptr;
    const bool dev_ok = rn && (strstr(rn, "Vulkan") || strstr(rn, "CUDA"));
    const bool ok = dev_ok && moe_cache_tested_ggml(ggml_version());
    if (!ok && said++ == 0) {
        fprintf(stderr, "moe-cache: MOE_CACHE_GPU_PREFETCH=1 ignored: %s\n", dev_ok ? "this ggml build is not in the tested list" : "the GPU backend is not Vulkan or CUDA");
    }
    return ok;
}

bool gpu_sum_on();                                  // mc_device.inl
bool is_weighted_experts(const ggml_tensor * t);
bool is_expert_sum(const ggml_tensor * t);

// Run the expert block (MUL_MAT_ID / GLU nodes first..last) on the GPU: clone the nodes into a small graph, upload the weights and inputs
// to a reusable staging area, compute there and copy the last node's result back. Returns false (nothing changed) if anything does not fit.
bool gpu_segment(moe_cache_backend_ctx * c, state_t & s, ggml_cgraph * cg, int first, int last, std::vector<ggml_tensor *> & seg) {
    const auto t_fn0 = std::chrono::steady_clock::now();     // MOE_CACHE_E1_TIMES: start of the block (before the graph is built)
    const int n_all = ggml_graph_n_nodes(cg);
    std::unordered_set<const ggml_tensor *> inside;
    for (int i = first; i <= last; i++) {
        ggml_tensor * n = ggml_graph_node(cg, i);
        // idea P16 (MOE_CACHE_GPU_EXTRA_OPS=1, default off): plain element-wise nodes between the expert matrices (clamp, scale) may stay in the GPU block
        // instead of sending the whole block back to the CPU; they read only nodes of the block
        static const bool extra_ops = getenv("MOE_CACHE_GPU_EXTRA_OPS") && atoi(getenv("MOE_CACHE_GPU_EXTRA_OPS")) > 0;
        if (n->op != GGML_OP_MUL_MAT_ID && n->op != g_op_glu && !(extra_ops && i > first && (n->op == GGML_OP_CLAMP || n->op == GGML_OP_SCALE))) {
            return false;
        }
        inside.insert(n);
        seg.push_back(n);
    }
    ggml_tensor * out = seg.back();
    if (!ggml_is_contiguous(out)) {
        return false;
    }
    // inputs must exist already (not produced by another node of this graph) and be plain F32 / I32 / weights
    std::unordered_set<const ggml_tensor *> produced;
    for (int i = 0; i < n_all; i++) {
        produced.insert(ggml_graph_node(cg, i));
    }
    for (ggml_tensor * n : seg) {
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            const ggml_tensor * src = n->src[k];
            if (!src || inside.count(src)) {
                continue;
            }
            if (produced.count(src) || src->view_src) {
                return false;
            }
            const auto it = s.by_tensor.find(src);
            if (it != s.by_tensor.end()) {
                if (s.tensors[it->second]->repack) {
                    return false;
                }
            } else if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_I32) {
                return false;
            }
        }
    }
    if (!c->gpu) {
        c->gpu = ggml_backend_dev_init(s.gpu_dev, nullptr);
        if (!c->gpu) {
            c->gpu_failed = true;
            return false;
        }
        c->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->gpu));
    }
    ggml_init_params ip = { ggml_tensor_overhead() * 4 * (size_t) (seg.size() + 8) + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    // idea I4 (on by default; MOE_CACHE_GPU_PACK=0 turns it off): upload only the experts this block's tokens use. Short prompts (agent follow-up turns of 33-300 tokens) touch
    // a part of the experts, but the whole layer was uploaded (Qwen3.6: a 79-token turn took 3.4 s, all of it uploads). The used experts are packed
    // into a smaller weight tensor and the ids renumbered; each output still comes from the same expert matrix and the same input row.
    // Identical to stock (identity 4/4 at 74 and 147-token prompts, Qwen3.6); 70-token prompts 1.62 -> 0.90 s, 170-token 1.74 -> 1.12 s (RESULTS 52).
    static const bool pack_env = !getenv("MOE_CACHE_GPU_PACK") || atoi(getenv("MOE_CACHE_GPU_PACK")) > 0;
    const ggml_tensor * pack_ids = nullptr;
    std::vector<int32_t> pack_used, pack_map;
    const ggml_tensor * full_ids = nullptr;
    std::vector<int32_t> full_used;
    if (pack_env) {
        bool ok = true;
        int64_t n_as = -1;
        for (ggml_tensor * n : seg) {
            if (n->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            ok = ok && (!pack_ids || pack_ids == n->src[2]) && ggml_is_contiguous(n->src[0]) && (n_as < 0 || n_as == n->src[0]->ne[2]);
            pack_ids = n->src[2];
            n_as = n->src[0]->ne[2];
        }
        if (ok && pack_ids && pack_ids->type == GGML_TYPE_I32 && pack_ids->data && pack_ids->ne[2] == 1 && pack_ids->ne[3] == 1 && n_as > 0) {
            pack_map.assign((size_t) n_as, -1);
            for (int64_t i1 = 0; ok && i1 < pack_ids->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < pack_ids->ne[0]; i0++) {
                    const int32_t v = *(const int32_t *) ((const char *) pack_ids->data + i0 * pack_ids->nb[0] + i1 * pack_ids->nb[1]);
                    if (v < 0 || v >= n_as) {
                        ok = false;
                        break;
                    }
                    pack_map[(size_t) v] = 0;
                }
            }
            for (int64_t e = 0; ok && e < n_as; e++) {
                if (pack_map[(size_t) e] == 0) {
                    pack_map[(size_t) e] = (int32_t) pack_used.size();
                    pack_used.push_back((int32_t) e);
                }
            }
            if (ok) {
                full_ids = pack_ids;      // N23: a split block packs even when every expert is used
                full_used = pack_used;
            }
            // pack unless every expert is used: the whole-tensor upload also reads the experts this block does not use, which may be evicted
            // (desktop PC 2026-10-26, GLM-5.3-Flash with a 4.6 GiB cache: on Windows an evicted expert is decommitted memory and the upload's copy
            // crashed with an access violation). Linux used a 90 % rule while evicted experts read as zeros; with the file view (its default since
            // 191e6d4) they are read from the disk on this thread and stay mapped outside the cache's budget, so both systems pack now (audit
            // 2026-10-29). At 100 % use it is the same single copy.
            if (!ok || pack_used.size() == (size_t) n_as) {
                pack_used.clear();
            }
        }
        if (pack_used.empty()) {
            pack_ids = nullptr;
        } else {
            static std::atomic<int> said{0};
            if (said++ < 3) {
                fprintf(stderr, "moe-cache: long-prompt block for %lld tokens: uploading only the %zu of %lld experts it uses\n",
                        (long long) pack_ids->ne[1], pack_used.size(), (long long) n_as);
            }
        }
    }
    // idea PL05 (MOE_CACHE_GPU_PREFETCH=1, default off): see gpu_prefetch_t. Weight tensors of this block that the helper has already put (or is putting) into the
    // second buffer: they are not uploaded again and do not count for the room this block needs. Whole-tensor blocks only (no packing, no split, no template, no held copies).
    static const bool pf_env = getenv("MOE_CACHE_GPU_PREFETCH") && atoi(getenv("MOE_CACHE_GPU_PREFETCH")) > 0 &&
                               !(getenv("MOE_CACHE_GPU_TMPL") && atoi(getenv("MOE_CACHE_GPU_TMPL")) > 0) &&
                               !(getenv("MOE_CACHE_GPU_KEEP_MIB") && atoi(getenv("MOE_CACHE_GPU_KEEP_MIB")) > 0) &&
                               !(getenv("MOE_CACHE_GPU_DEV2") && atoi(getenv("MOE_CACHE_GPU_DEV2")) > 0) && pf_backend_ok(s.gpu_dev);
    const bool pf_can = pf_env && !c->pf.off && !c->on_alt;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> pf_hit;      // weight tensor of this block -> its place in the prefetch buffer
    size_t pf_bytes = 0;
    if (pf_can && !pack_ids && c->pf.buf && !c->pf.src.empty()) {
        for (ggml_tensor * n : seg) {
            const ggml_tensor * w = n->op == GGML_OP_MUL_MAT_ID ? n->src[0] : nullptr;
            if (!w || inside.count(w) || pf_hit.count(w)) {
                continue;
            }
            for (size_t i = 0; i < c->pf.src.size(); i++) {
                if (c->pf.src[i] == w && c->pf.dst[i]->type == w->type && memcmp(c->pf.dst[i]->ne, w->ne, sizeof(w->ne)) == 0) {
                    pf_hit[w] = c->pf.dst[i];
                    pf_bytes += ggml_nbytes(c->pf.dst[i]);
                    break;
                }
            }
        }
    }
    // N23: at long contexts llama.cpp's own GPU buffers grow during the prompt (attention scratch), the free GPU memory falls and the blocks went
    // back to the CPU one by one (PC, Flash-Next Q2_0 at 128k: from ~40k tokens on every micro-batch took 22 s instead of ~9 s). When the block
    // does not fit, its used experts are split into parts that fit the buffer already held: each part runs the same graph (ids of the other parts point at
    // the part's first expert and those rows are thrown away) and its rows are copied into the output. Every row comes from the same expert matrix and
    // input row as in one block. MOE_CACHE_GPU_CHUNKS=0: off (old behaviour), =N: always N parts (identity checks).
    static const int chunks_env = getenv("MOE_CACHE_GPU_CHUNKS") ? atoi(getenv("MOE_CACHE_GPU_CHUNKS")) : -1;
    const size_t headroom = (size_t) 256 << 20;                                  // pool buffers, graph, fragmentation
    card_guard card_back{c};
    ggml_backend_dev_t gdev = s.gpu_dev;                                          // the card this block runs on (the second one only with MOE_CACHE_GPU_DEV2)
    size_t held = ggml_gallocr_get_buffer_size(c->galloc, 0);                     // already allocated for an earlier block: reused
    size_t gpu_free = 0, gpu_tot = 0;
    ggml_backend_dev_memory(gdev, &gpu_free, &gpu_tot);
    // a block that needs no more than the held buffer allocates nothing, so it needs no headroom (with chunks off: the old rule, headroom always)
    auto fits = [&](size_t need) {
        const size_t extra = need > held ? need - held : 0;
        return gpu_tot == 0 || (chunks_env != 0 && extra == 0) || gpu_free >= extra + headroom;
    };
    // N3/MG5 (MOE_CACHE_GPU_DEV2): the first card is short of room for the block as a whole; if the second card holds it without splitting, the block runs there
    // (the weights are uploaded to it from the host, the result comes back the same way). The first card keeps its buffers and its place: the next block tries it first.
    auto try_second_card = [&](size_t need) {
        if (!s.gpu_dev2 || c->alt_failed || c->on_alt) {
            return false;
        }
        card_swap(c);
        if (!c->gpu) {
            c->gpu = ggml_backend_dev_init(s.gpu_dev2, nullptr);
            if (c->gpu) {
                c->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->gpu));
            } else {
                c->alt_failed = true;
                fprintf(stderr, "moe-cache: MOE_CACHE_GPU_DEV2: the second card could not be started: long-prompt blocks stay on the first card\n");
            }
        }
        size_t f2 = 0, t2 = 0;
        const size_t h2 = c->gpu ? ggml_gallocr_get_buffer_size(c->galloc, 0) : 0;
        if (c->gpu) {
            ggml_backend_dev_memory(s.gpu_dev2, &f2, &t2);
        }
        const size_t extra = need > h2 ? need - h2 : 0;
        if (!c->gpu || !(t2 == 0 || extra == 0 || f2 >= extra + headroom)) {
            card_swap(c);                     // no room there either: back to the first card, which splits the block
            return false;
        }
        const size_t f1 = gpu_free;
        gdev = s.gpu_dev2;
        held = h2;
        gpu_free = f2;
        gpu_tot = t2;
        s.gpu_alt_blocks++;
        static std::atomic<int> said_alt{0};
        if (said_alt++ < 3) {
            fprintf(stderr, "moe-cache: long-prompt block does not fit on the first card (%.0f MiB free): running it on %s (%.0f MiB free) (MOE_CACHE_GPU_DEV2)\n",
                    f1 / 1048576.0, ggml_backend_dev_name(s.gpu_dev2), f2 / 1048576.0);
        }
        return true;
    };
    struct split_stat_t { double ema = 0; int n = 0; };
    static split_stat_t split_stat[17];     // idea P12: per number of parts, nanoseconds per used expert
    size_t n_chunk = 1;
    if (chunks_env != 0 && full_ids && out->ne[1] == full_ids->ne[0] && out->ne[2] == full_ids->ne[1] && out->ne[3] == 1) {
        auto need_for = [&](size_t n_w) {      // bytes of the block's graph with n_w experts per weight tensor (0: whole tensors)
            size_t need = 0;
            std::unordered_set<const ggml_tensor *> seen;
            for (ggml_tensor * n : seg) {
                need += ggml_row_size(n->type, n->ne[0]) * (size_t) (n->ne[1] * n->ne[2] * n->ne[3]);
                for (int k = 0; k < GGML_MAX_SRC; k++) {
                    const ggml_tensor * src = n->src[k];
                    if (!src || inside.count(src) || !seen.insert(src).second) {
                        continue;
                    }
                    const bool w = n->op == GGML_OP_MUL_MAT_ID && k == 0 && n_w > 0;
                    need += ggml_row_size(src->type, src->ne[0]) * (size_t) (src->ne[1] * (w ? (int64_t) n_w : src->ne[2]) * src->ne[3]);
                }
            }
            return n_w == 0 && need > pf_bytes ? need - pf_bytes : need;
        };
        if (chunks_env > 1) {
            n_chunk = (size_t) chunks_env;
        } else if (!fits(need_for(pack_ids ? pack_used.size() : 0)) && !try_second_card(need_for(pack_ids ? pack_used.size() : 0))) {
            for (size_t k = 2; k <= 16; k++) {
                if (fits(need_for((full_used.size() + k - 1) / k))) {
                    n_chunk = k;
                    break;
                }
            }
            // idea P12 (MOE_CACHE_GPU_SPLIT_ADAPT=1, default off): the first split that fits and the next two are tried in turn, then the one with the
            // lowest measured time per used expert (running average) is used; the block's time is booked at the end of this function
            static const bool adapt = getenv("MOE_CACHE_GPU_SPLIT_ADAPT") && atoi(getenv("MOE_CACHE_GPU_SPLIT_ADAPT")) > 0;
            if (adapt && n_chunk > 1) {
                size_t pick = 0;
                double bestv = 0;
                for (size_t k = n_chunk; k <= std::min<size_t>(n_chunk + 2, 16); k++) {
                    if (!fits(need_for((full_used.size() + k - 1) / k))) {
                        continue;
                    }
                    if (split_stat[k].n < 2) {          // not measured yet: measure it (twice)
                        pick = k;
                        break;
                    }
                    if (!pick || split_stat[k].ema < bestv) {
                        pick = k;
                        bestv = split_stat[k].ema;
                    }
                }
                if (pick) {
                    n_chunk = pick;
                }
            }
        }
        if (n_chunk > 1) {
            pack_ids = full_ids;
            pack_used = full_used;
            n_chunk = std::min(n_chunk, pack_used.size());
            // idea P17 (MOE_CACHE_GPU_BALANCE=1, default off): the parts hold the same number of experts (what fits), but not the same number of picks. Experts are
            // dealt to the parts heaviest first (each to the lightest part with room), so the parts' rows, and with MOE_CACHE_GPU_COMPACT their work, are even.
            static const bool balance = getenv("MOE_CACHE_GPU_BALANCE") && atoi(getenv("MOE_CACHE_GPU_BALANCE")) > 0;
            if (balance && n_chunk > 1 && pack_ids) {
                std::vector<uint32_t> cnt(pack_map.size(), 0);
                for (int64_t i1 = 0; i1 < pack_ids->ne[1]; i1++) {
                    for (int64_t i0 = 0; i0 < pack_ids->ne[0]; i0++) {
                        cnt[(size_t) *(const int32_t *) ((const char *) pack_ids->data + i0 * pack_ids->nb[0] + i1 * pack_ids->nb[1])]++;
                    }
                }
                std::vector<int32_t> order = pack_used;
                std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return cnt[(size_t) a] > cnt[(size_t) b]; });
                const size_t per = (pack_used.size() + n_chunk - 1) / n_chunk;
                std::vector<size_t> cap(n_chunk, per), load(n_chunk, 0);
                cap[n_chunk - 1] = pack_used.size() > (n_chunk - 1) * per ? pack_used.size() - (n_chunk - 1) * per : 0;
                std::vector<std::vector<int32_t>> parts(n_chunk);
                for (int32_t e : order) {
                    size_t best = n_chunk;
                    for (size_t k = 0; k < n_chunk; k++) {
                        if (parts[k].size() < cap[k] && (best == n_chunk || load[k] < load[best])) {
                            best = k;
                        }
                    }
                    parts[best].push_back(e);
                    load[best] += cnt[(size_t) e];
                }
                pack_used.clear();
                for (auto & pt : parts) {
                    std::sort(pt.begin(), pt.end());
                    pack_used.insert(pack_used.end(), pt.begin(), pt.end());
                }
                for (size_t i = 0; i < pack_used.size(); i++) {
                    pack_map[(size_t) pack_used[i]] = (int32_t) i;
                }
            }
            static std::atomic<int> said_ch{0};
            if (said_ch++ < 3) {
                fprintf(stderr, "moe-cache: long-prompt block split into %zu parts of %zu experts to fit in GPU memory (%.0f MiB free, %.0f MiB held)\n",
                        n_chunk, (pack_used.size() + n_chunk - 1) / n_chunk, gpu_free / 1048576.0, held / 1048576.0);
            }
        }
    }
    const size_t per_chunk = n_chunk > 1 ? (pack_used.size() + n_chunk - 1) / n_chunk : pack_used.size();
    if (n_chunk > 1) {
        n_chunk = (pack_used.size() + per_chunk - 1) / per_chunk;      // no empty tail part: 100 experts in 16 parts of 7 leave parts 15 with lo=105 > hi=100, (hi - lo) wraps
    }
    if (n_chunk > 1 || pack_ids || c->on_alt) {
        pf_hit.clear();
        pf_bytes = 0;
    }
    std::unordered_set<const ggml_tensor *> pack_weights;
    if (pack_ids) {
        for (ggml_tensor * n : seg) {
            if (n->op == GGML_OP_MUL_MAT_ID) {
                pack_weights.insert(n->src[0]);
            }
        }
    }
    // idea P01 (MOE_CACHE_GPU_COMPACT=1, default off): a split block computes, for each part, only the (token, slot) rows whose expert is in that part
    // (a graph of 1 x R rows, built per part) instead of the whole n_ids x n_tok shape with the other rows thrown away. Each row is the same
    // expert matrix times the same input row, but the GPU kernel sees another shape, so the rounding may differ: it needs the logit check.
    static const bool compact_env = getenv("MOE_CACHE_GPU_COMPACT") && atoi(getenv("MOE_CACHE_GPU_COMPACT")) > 0;
    if (compact_env && n_chunk > 1 && pack_ids && pack_weights.size() > 0) {
        const int64_t n_ids = pack_ids->ne[0], n_tok = pack_ids->ne[1];
        bool ok = ggml_is_contiguous(out) && out->ne[1] == n_ids && out->ne[2] == n_tok && out->ne[3] == 1;
        for (ggml_tensor * n : seg) {
            ok = ok && n->ne[1] == n_ids && n->ne[2] == n_tok && n->ne[3] == 1;
            for (int k = 0; ok && k < GGML_MAX_SRC; k++) {
                const ggml_tensor * src = n->src[k];
                if (!src || inside.count(src)) {
                    continue;
                }
                if (n->op == GGML_OP_MUL_MAT_ID && k == 0) {
                    ok = pack_weights.count(src) > 0;
                } else if (n->op == GGML_OP_MUL_MAT_ID && k == 2) {
                    ok = src == pack_ids;
                } else if (n->op == GGML_OP_MUL_MAT_ID && k == 1) {
                    ok = src->type == GGML_TYPE_F32 && src->ne[2] == n_tok && src->ne[3] == 1 && (src->ne[1] == 1 || src->ne[1] == n_ids) && src->nb[0] == sizeof(float);
                } else {
                    ok = false;
                }
            }
            if (n->op == GGML_OP_MUL_MAT_ID) {
                const ggml_tensor * b = n->src[1];
                ok = ok && (inside.count(b) ? (b->ne[1] == n_ids && b->ne[2] == n_tok) : true);
            }
        }
        if (ok) {
            static std::atomic<int> said_c{0};
            if (said_c++ < 3) {
                fprintf(stderr, "moe-cache: split long-prompt block: computing only the rows of each part (MOE_CACHE_GPU_COMPACT)\n");
            }
            std::vector<std::pair<int32_t, int32_t>> rows;       // (token, slot)
            std::vector<uint8_t> hostbuf;
            for (size_t ch = 0; ch < n_chunk; ch++) {
                const size_t lo = ch * per_chunk, hi = std::min(pack_used.size(), lo + per_chunk);
                if (lo >= hi) {
                    continue;
                }
                rows.clear();
                std::vector<int32_t> cids;
                for (int64_t i1 = 0; i1 < n_tok; i1++) {
                    for (int64_t i0 = 0; i0 < n_ids; i0++) {
                        const size_t v = (size_t) pack_map[(size_t) *(const int32_t *) ((const char *) pack_ids->data + i0 * pack_ids->nb[0] + i1 * pack_ids->nb[1])];
                        if (v >= lo && v < hi) {
                            rows.emplace_back((int32_t) i1, (int32_t) i0);
                            cids.push_back((int32_t) (v - lo));
                        }
                    }
                }
                if (rows.empty()) {
                    continue;
                }
                const int64_t R = (int64_t) rows.size();
                ggml_context * pctx = ggml_init(ip);
                if (!pctx) {
                    return false;
                }
                std::unordered_map<const ggml_tensor *, ggml_tensor *> pc;
                std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> pup;
                auto pext = [&](const ggml_tensor * t) {
                    auto it = pc.find(t);
                    if (it != pc.end()) {
                        return it->second;
                    }
                    int64_t ne[GGML_MAX_DIMS] = { t->ne[0], 1, R, 1 };
                    if (pack_weights.count(t)) {
                        ne[1] = t->ne[1];
                        ne[2] = (int64_t) (hi - lo);
                    } else if (t == pack_ids) {
                        ne[0] = 1;
                    }
                    ggml_tensor * nt = ggml_new_tensor(pctx, t->type, GGML_MAX_DIMS, ne);
                    ggml_set_input(nt);
                    pc[t] = nt;
                    pup.emplace_back(t, nt);
                    return nt;
                };
                for (ggml_tensor * n : seg) {
                    int64_t ne[GGML_MAX_DIMS] = { n->ne[0], 1, R, 1 };
                    ggml_tensor * nn = ggml_new_tensor(pctx, n->type, GGML_MAX_DIMS, ne);
                    nn->op = n->op;
                    memcpy(nn->op_params, n->op_params, sizeof(nn->op_params));
                    for (int k = 0; k < GGML_MAX_SRC; k++) {
                        if (n->src[k]) {
                            nn->src[k] = inside.count(n->src[k]) ? pc[n->src[k]] : pext(n->src[k]);
                        }
                    }
                    pc[n] = nn;
                }
                ggml_tensor * pout = pc[out];
                ggml_set_output(pout);
                ggml_cgraph * pg = ggml_new_graph(pctx);
                ggml_build_forward_expand(pg, pout);
                if (!gpu_graph_supported(c, pg) || !ggml_gallocr_alloc_graph(c->galloc, pg)) {
                    ggml_free(pctx);
                    return false;
                }
                for (auto & u : pup) {
                    const ggml_tensor * t = u.first;
                    ggml_tensor * d = u.second;
                    if (pack_weights.count(t)) {
                        const size_t nb2 = t->nb[2];
                        for (size_t k = lo; k < hi;) {
                            size_t r = 1;
                            while (k + r < hi && pack_used[k + r] == pack_used[k] + (int32_t) r) {
                                r++;
                            }
                            wait_experts(s, t, pack_used[k], (int64_t) r);
                            ggml_backend_tensor_set(d, (const char *) t->data + (size_t) pack_used[k] * nb2, (k - lo) * nb2, r * nb2);
                            k += r;
                        }
                        s.gpu_up_bytes += (hi - lo) * nb2;
                    } else if (t == pack_ids) {
                        ggml_backend_tensor_set(d, cids.data(), 0, cids.size() * sizeof(int32_t));
                    } else {
                        const size_t rb_bytes = (size_t) t->ne[0] * sizeof(float);       // one input row per picked (token, slot)
                        hostbuf.resize(rb_bytes * (size_t) R);
                        for (int64_t r = 0; r < R; r++) {
                            const int64_t i11 = t->ne[1] == 1 ? 0 : rows[(size_t) r].second;
                            memcpy(hostbuf.data() + (size_t) r * rb_bytes, (const char *) t->data + i11 * t->nb[1] + (int64_t) rows[(size_t) r].first * t->nb[2], rb_bytes);
                        }
                        ggml_backend_tensor_set(d, hostbuf.data(), 0, hostbuf.size());
                        s.gpu_up_bytes += hostbuf.size();
                    }
                }
                const enum ggml_status st = ggml_backend_graph_compute(c->gpu, pg);
                if (st != GGML_STATUS_SUCCESS) {
                    ggml_backend_synchronize(c->gpu);
                    ggml_free(pctx);
                    return false;
                }
                const size_t row = ggml_row_size(out->type, out->ne[0]);
                hostbuf.resize(row * (size_t) R);
                ggml_backend_tensor_get(pout, hostbuf.data(), 0, hostbuf.size());
                for (int64_t r = 0; r < R; r++) {
                    memcpy((char *) out->data + (size_t) rows[(size_t) r].second * out->nb[1] + (size_t) rows[(size_t) r].first * out->nb[2], hostbuf.data() + (size_t) r * row, row);
                }
                ggml_free(pctx);
            }
            ggml_free(ctx);
            s.gpu_segments++;
            s.gpu_split_blocks++;
            return true;
        }
    }
    // MOE_CACHE_GPU_SUM (mc_device.inl): the nodes right after the experts that the plugin claimed join the block, so only their sum is read back. A block split
    // into parts leaves them to the CPU (the row copy of a split block needs the expert output's shape), as does anything that does not match; the result is the same, only slower.
    static const bool tmpl_off = !getenv("MOE_CACHE_GPU_TMPL") || atoi(getenv("MOE_CACHE_GPU_TMPL")) <= 0;
    if (gpu_sum_on() && tmpl_off && n_chunk == 1 && last + 1 < n_all && is_weighted_experts(ggml_graph_node(cg, last + 1)) && ggml_graph_node(cg, last + 1)->src[0] == out) {
        ggml_tensor * mul = ggml_graph_node(cg, last + 1);
        const ggml_tensor * w = mul->src[1];
        std::vector<ggml_tensor *> tail { mul };
        ggml_tensor * sum = nullptr;
        for (int i = last + 2; i < n_all; i++) {
            ggml_tensor * n = ggml_graph_node(cg, i);
            if (n->op == GGML_OP_VIEW && n->view_src == mul) {
                tail.push_back(n);
            } else if (is_expert_sum(n) && n->src[1]->view_src == mul) {
                tail.push_back(n);
                sum = n;
            } else {
                break;
            }
        }
        std::unordered_set<const ggml_tensor *> all_nodes;
        for (int i = 0; i < n_all; i++) {
            all_nodes.insert(ggml_graph_node(cg, i));
        }
        if (sum && ggml_is_contiguous(sum) && w && w->type == GGML_TYPE_F32 && !w->view_src && !all_nodes.count(w)) {
            for (ggml_tensor * n : tail) {
                inside.insert(n);
                seg.push_back(n);
            }
            out = sum;
        }
    }
    // idea P08 (MOE_CACHE_GPU_BUCKET=N, default off): the packed weight tensors are allocated for the part's expert count rounded up to a multiple of N, so
    // blocks that use a few more or fewer experts share one graph shape (with MOE_CACHE_GPU_TMPL); only the used experts are uploaded, the padding is never read
    static const size_t bucket_n = getenv("MOE_CACHE_GPU_BUCKET") ? (size_t) std::max(1, atoi(getenv("MOE_CACHE_GPU_BUCKET"))) : 1;
    const size_t per_alloc = pack_ids ? (per_chunk + bucket_n - 1) / bucket_n * bucket_n : per_chunk;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> clone;
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> uploads;
    static const bool tmpl_env = getenv("MOE_CACHE_GPU_TMPL") && atoi(getenv("MOE_CACHE_GPU_TMPL")) > 0 && !(getenv("MOE_CACHE_GPU_KEEP_MIB") && atoi(getenv("MOE_CACHE_GPU_KEEP_MIB")) > 0);
    std::string tkey;
    if (tmpl_env) {      // the block's shape signature: ops, types, shapes, parameters, how the nodes read each other, and the split
        std::vector<const ggml_tensor *> exts;
        tkey += pack_ids ? 'P' : 'p';
        tkey += std::to_string(per_alloc);
        for (ggml_tensor * n : seg) {
            tkey += '|';
            tkey += std::to_string((int) n->op) + ',' + std::to_string((int) n->type) + ',' + std::to_string(n->ne[0]) + ',' + std::to_string(n->ne[1]) + ',' +
                    std::to_string(n->ne[2]) + ',' + std::to_string(n->ne[3]) + ',';
            tkey.append((const char *) n->op_params, sizeof(n->op_params));
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                const ggml_tensor * src = n->src[k];
                if (!src) {
                    continue;
                }
                tkey += ';' + std::to_string(k) + ':';
                if (inside.count(src)) {
                    tkey += 'i' + std::to_string(std::find(seg.begin(), seg.end(), src) - seg.begin());
                } else {
                    size_t o = std::find(exts.begin(), exts.end(), src) - exts.begin();
                    if (o == exts.size()) {
                        exts.push_back(src);
                    }
                    tkey += 'e' + std::to_string(o) + ',' + std::to_string((int) src->type) + ',' + std::to_string(src->ne[0]) + ',' + std::to_string(src->ne[1]) + ',' +
                            std::to_string(src->ne[2]) + ',' + std::to_string(src->ne[3]) + (pack_weights.count(src) ? 'w' : 'n');
                }
            }
        }
    }
    std::vector<ggml_tensor *> node_clones;
    ggml_tensor * gout = nullptr;
    ggml_cgraph * g = nullptr;
    bool from_tmpl = false;
    if (tmpl_env) {
        for (size_t i = 0; i < c->tmpls.size(); i++) {
            if (c->tmpls[i].key != tkey) {
                continue;
            }
            auto t = std::move(c->tmpls[i]);        // taken out while in use: an early return below frees it
            c->tmpls.erase(c->tmpls.begin() + (long) i);
            ggml_free(ctx);
            ctx = t.ctx;
            g = t.g;
            node_clones = t.node_clones;
            std::unordered_set<const ggml_tensor *> seen;
            size_t ei = 0;
            for (ggml_tensor * n : seg) {
                for (int k = 0; k < GGML_MAX_SRC; k++) {
                    const ggml_tensor * src = n->src[k];
                    if (src && !inside.count(src) && seen.insert(src).second) {
                        uploads.emplace_back(src, t.ext_clones[ei++]);
                        clone[src] = uploads.back().second;
                    }
                }
            }
            for (size_t j = 0; j < seg.size(); j++) {
                clone[seg[j]] = node_clones[j];
            }
            gout = clone[out];
            from_tmpl = true;
            break;
        }
    }
    static const size_t keep_mib = getenv("MOE_CACHE_GPU_KEEP_MIB") ? (size_t) std::max(0, atoi(getenv("MOE_CACHE_GPU_KEEP_MIB"))) : 0;
    std::unordered_map<const ggml_tensor *, const ggml_tensor *> held_for;      // input clone -> source tensor whose GPU copy it sits in
    std::unordered_set<const ggml_tensor *> pf_placed;                          // PL05: input clones placed in the prefetch buffer
    if (keep_mib > 0) {
        c->held_serial++;
    }
    auto ext = [&](const ggml_tensor * t) {
        auto it = clone.find(t);
        if (it != clone.end()) {
            return it->second;
        }
        int64_t ne[GGML_MAX_DIMS] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
        if (pack_weights.count(t)) {
            ne[2] = (int64_t) per_alloc;
        }
        ggml_tensor * nt = ggml_new_tensor(ctx, t->type, GGML_MAX_DIMS, ne);
        ggml_set_input(nt);
        clone[t] = nt;
        uploads.emplace_back(t, nt);
        const auto ph = pf_hit.find(t);
        if (ph != pf_hit.end()) {              // PL05: this weight sits (or is being copied) in the prefetch buffer: the clone is placed there
            ggml_backend_tensor_alloc(c->pf.buf, nt, ph->second->data);
            pf_placed.insert(nt);
            return nt;
        }
        if (keep_mib > 0 && !pack_weights.count(t) && s.by_tensor.count(t) && ggml_is_contiguous(t)) {
            moe_cache_backend_ctx::gpu_held_t * h = nullptr;
            for (auto & x : c->held) {
                if (x.src == t) { h = &x; break; }
            }
            const size_t nbytes = ggml_nbytes(nt);
            if (!h && nbytes <= keep_mib * (size_t) 1048576) {
                size_t tot = 0;
                for (auto & x : c->held) { tot += x.bytes; }
                while (tot + nbytes > keep_mib * (size_t) 1048576) {          // make room: least recently used first, never one this block already uses
                    size_t vi = c->held.size();
                    for (size_t i = 0; i < c->held.size(); i++) {
                        if (c->held[i].used != c->held_serial && (vi == c->held.size() || c->held[i].used < c->held[vi].used)) { vi = i; }
                    }
                    if (vi == c->held.size()) { break; }
                    tot -= c->held[vi].bytes;
                    ggml_backend_buffer_free(c->held[vi].buf);
                    ggml_free(c->held[vi].ctx);
                    c->held.erase(c->held.begin() + (long) vi);
                }
                size_t fr = 0, tt = 0;
                ggml_backend_dev_memory(gdev, &fr, &tt);
                if (tot + nbytes <= keep_mib * (size_t) 1048576 && (tt == 0 || fr >= nbytes + ((size_t) 512 << 20))) {
                    ggml_init_params hp = { ggml_tensor_overhead() * 2 + 1024, nullptr, true };
                    ggml_context * hc = ggml_init(hp);
                    ggml_tensor * ht = hc ? ggml_new_tensor(hc, t->type, GGML_MAX_DIMS, ne) : nullptr;
                    ggml_backend_buffer_t hb = ht ? ggml_backend_alloc_ctx_tensors(hc, c->gpu) : nullptr;
                    if (hb) {
                        moe_cache_backend_ctx::gpu_held_t nh;
                        nh.src = t; nh.ctx = hc; nh.buf = hb; nh.bytes = ggml_backend_buffer_get_size(hb);
                        c->held.push_back(nh);
                        h = &c->held.back();
                    } else if (hc) {
                        ggml_free(hc);
                    }
                }
            }
            if (h) {
                h->used = c->held_serial;
                ggml_backend_tensor_alloc(h->buf, nt, ggml_backend_buffer_get_base(h->buf));
                held_for[nt] = h->src;
            }
        }
        return nt;
    };
    if (!from_tmpl) {
        for (ggml_tensor * n : seg) {
            if (n->op == GGML_OP_VIEW) {         // GPU_SUM: a view of the weighted experts shares their memory
                clone[n] = ggml_view_2d(ctx, clone[n->view_src], n->ne[0], n->ne[1], n->nb[1], n->view_offs);
                continue;
            }
            ggml_tensor * nn = ggml_new_tensor(ctx, n->type, GGML_MAX_DIMS, n->ne);
            nn->op = n->op;
            memcpy(nn->op_params, n->op_params, sizeof(nn->op_params));
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                if (n->src[k]) {
                    nn->src[k] = inside.count(n->src[k]) ? clone[n->src[k]] : ext(n->src[k]);
                }
            }
            clone[n] = nn;
        }
        gout = clone[out];
        ggml_set_output(gout);
        g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, gout);
    }
    if (!from_tmpl && !gpu_graph_supported(c, g)) {
        ggml_free(ctx);
        return false;
    }
    {
        // a whole layer of expert weights is uploaded for this block: with a GPU expert cache next to it the card can be full.
        // Check first and keep the block on the CPU instead of letting the allocation (or the CUDA pool) fail hard.
        size_t need = 0;
        for (auto & u : uploads) {
            if (!pf_placed.count(u.second)) {          // PL05: already in the prefetch buffer, the graph allocator does not give it room
                need += ggml_nbytes(u.second);
            }
        }
        for (ggml_tensor * n : seg) {
            if (n->op != GGML_OP_VIEW && n->op != GGML_OP_MUL && n->op != GGML_OP_ADD) {    // MOE_CACHE_GPU_SUM's nodes reuse the memory of the node before them
                need += ggml_nbytes(clone[n]);
            }
        }
        const size_t extra = need > held ? need - held : 0;
        const size_t fr = gpu_free;
        if (!fits(need)) {
            c->seg_nomem = true;
            if (!c->hb_buf && s.gpu_nomem++ == 0) {
                fprintf(stderr, "moe-cache: a long-prompt block needs %.0f MiB of GPU memory but only %.0f MiB are free: it stays on the CPU "
                                "(a smaller --moe-cache-mib or --gpu-margin leaves room; about %.0f MiB per layer for this model)\n",
                        (extra + headroom) / 1048576.0, fr / 1048576.0, need / 1048576.0);
            }
            ggml_free(ctx);
            return false;
        }
    }
#ifdef _WIN32
    // a whole expert tensor is uploaded (no packing: MOE_CACHE_GPU_PACK=0, or ids the packing cannot handle): all of its experts must be in RAM,
    // else the copy reads decommitted memory (an access violation): keep such a block on the CPU. Linux reads zeros there, unused and harmless,
    // so the check (which would send such blocks to the CPU for nothing) is Windows only.
    for (const auto & u : uploads) {
        if (pack_weights.count(u.first)) {
            continue;
        }
        const auto it = s.by_tensor.find(u.first);
        if (it == s.by_tensor.end()) {
            continue;
        }
        const tinfo_t & ti = *s.tensors[it->second];
        wait_experts(s, u.first, 0, ti.n_expert);      // N23-F3: not under the lock (the readers take it to clear the flags)
        std::lock_guard<std::mutex> lk(s.mu);
        for (int64_t e = 0; e < ti.n_expert; e++) {
            const auto & nd = s.nodes[(size_t) ti.node0 + (size_t) e];
            if (!(nd.resident && !nd.loading) && !nd.mapped) {
                static std::atomic<int> said_nr{0};
                if (said_nr++ == 0) {
                    fprintf(stderr, "moe-cache: a long-prompt block would upload whole expert tensors with experts not in RAM: it stays on the CPU\n");
                }
                ggml_free(ctx);
                return false;
            }
        }
    }
#endif
    if (!ggml_gallocr_alloc_graph(c->galloc, g)) {
        // not for good (it was for the rest of the process): another program may hold the GPU memory for a while (Windows). Try again later.
        ggml_free(ctx);
        c->gpu_retry_at = s.graphs.load() + 64;
        if (c->gpu_alloc_fails++ == 0) {
            fprintf(stderr, "moe-cache: could not allocate GPU memory for a long-prompt block: blocks stay on the CPU for 64 graphs, then the GPU is tried again\n");
        }
        return false;
    }
    // idea I2 kill check (MOE_CACHE_SEG_TIMES=1): upload vs compute time per block; overlap can only pay if compute is a real share of the block
    static const bool seg_times = getenv("MOE_CACHE_SEG_TIMES") != nullptr;
    // N22 (on by default; MOE_CACHE_GPU_READBACK=0 = old path): the result goes into a pinned host buffer of the GPU device, queued right after
    // the compute with one synchronize for both, then one memcpy into the CPU tensor. A blocking read into ordinary memory goes through the
    // backend's staging copy (~1.6 GB/s on the 9070 XT, 18 % of a block). Same bytes either way.
    static const bool rb_env = !getenv("MOE_CACHE_GPU_READBACK") || atoi(getenv("MOE_CACHE_GPU_READBACK")) > 0;
    const size_t out_bytes = ggml_nbytes(out);
    void * rb = nullptr;
    if (rb_env && !c->rb_failed) {
        if (!c->rb_buf || ggml_backend_buffer_get_size(c->rb_buf) < out_bytes) {
            ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(gdev);
            if (c->rb_buf) {
                ggml_backend_buffer_free(c->rb_buf);
                c->rb_buf = nullptr;
            }
            const size_t want = (out_bytes + ((size_t) 16 << 20) - 1) & ~(((size_t) 16 << 20) - 1);     // 16 MiB steps: few re-allocations
            c->rb_buf = hbt ? ggml_backend_buft_alloc_buffer(hbt, want) : nullptr;
            if (!c->rb_buf) {
                c->rb_failed = true;
                fprintf(stderr, "moe-cache: no pinned host buffer for the long-prompt block's result (%s): it is read back the plain way\n",
                        hbt ? "allocation failed" : "the GPU device has none");
            }
        }
        if (c->rb_buf) {
            rb = ggml_backend_buffer_get_base(c->rb_buf);
        }
    }
    // idea P05 (MOE_CACHE_GPU_STAGE=1): the weights of a part are first copied into one pinned host buffer and uploaded from there in one copy per
    // tensor; the next part is staged while the GPU computes this one. idea P04 (MOE_CACHE_GPU_OVERLAP=1, without staging): only the wait for the next
    // part's experts (SSD reads) is moved into that window. Both need the pinned read-back path (the window exists only while the result copy is queued).
    static const bool stage_env = getenv("MOE_CACHE_GPU_STAGE") && atoi(getenv("MOE_CACHE_GPU_STAGE")) > 0;
    static const bool overlap_env = getenv("MOE_CACHE_GPU_OVERLAP") && atoi(getenv("MOE_CACHE_GPU_OVERLAP")) > 0;
    char * stage = nullptr;
    size_t staged_ch = (size_t) -1;
    std::vector<std::pair<const ggml_tensor *, size_t>> stage_tensors;     // weight tensor, offset in the staging buffer
    if ((stage_env || overlap_env) && pack_ids && n_chunk > 1) {
        size_t total = 0;
        for (auto & u : uploads) {
            if (pack_weights.count(u.first)) {
                stage_tensors.emplace_back(u.first, total);
                total += per_chunk * u.first->nb[2];
            }
        }
        if (stage_env && total && !c->up_buf) {
            ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(gdev);
            c->up_buf = hbt ? ggml_backend_buft_alloc_buffer(hbt, total) : nullptr;
        }
        if (c->up_buf && ggml_backend_buffer_get_size(c->up_buf) < total) {
            ggml_backend_buffer_free(c->up_buf);
            ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(gdev);
            c->up_buf = hbt ? ggml_backend_buft_alloc_buffer(hbt, total + ((size_t) 64 << 20)) : nullptr;
        }
        if (c->up_buf && stage_env) {
            stage = (char *) ggml_backend_buffer_get_base(c->up_buf);
        }
    }
    auto stage_part = [&](size_t pch) {         // wait for and copy the part's experts into the staging buffer
        const size_t plo = pch * per_chunk, phi = std::min(pack_used.size(), plo + per_chunk);
        // idea P10 (MOE_CACHE_GPU_INTERLEAVE=1, default off): by run of neighbouring experts, all of its weight tensors (gate, up, down) before the next run,
        // instead of one whole tensor after the other, so the first runs are complete while the reads of the later ones still go on
        static const bool interleave = getenv("MOE_CACHE_GPU_INTERLEAVE") && atoi(getenv("MOE_CACHE_GPU_INTERLEAVE")) > 0;
        auto do_run = [&](const std::pair<const ggml_tensor *, size_t> & st, size_t k, size_t r) {
            wait_experts(s, st.first, pack_used[k], (int64_t) r);
            if (stage) {
                char * dst = stage + st.second + (k - plo) * st.first->nb[2];
                const char * src = (const char *) st.first->data + (size_t) pack_used[k] * st.first->nb[2];
                const size_t nbytes = r * st.first->nb[2];
                // idea P06 (MOE_CACHE_GPU_STAGE_THREADS=N, default 1): a copy of 8 MiB or more is split over N threads (the calling one and N-1 short-lived helpers)
                static const int nth = getenv("MOE_CACHE_GPU_STAGE_THREADS") ? std::max(1, std::min(8, atoi(getenv("MOE_CACHE_GPU_STAGE_THREADS")))) : 1;
                if (nth > 1 && nbytes >= ((size_t) 8 << 20)) {
                    std::vector<std::thread> hs;
                    const size_t part = (nbytes / (size_t) nth + 4095) / 4096 * 4096;
                    for (int ti2 = 1; ti2 < nth; ti2++) {
                        const size_t o = std::min(nbytes, part * (size_t) ti2), n2 = std::min(nbytes - o, part);
                        hs.emplace_back([=] { memcpy(dst + o, src + o, n2); });
                    }
                    memcpy(dst, src, std::min(nbytes, part));
                    for (auto & h : hs) { h.join(); }
                } else {
                    memcpy(dst, src, nbytes);
                }
            }
        };
        if (interleave) {
            for (size_t k = plo; k < phi;) {
                size_t r = 1;
                while (k + r < phi && pack_used[k + r] == pack_used[k] + (int32_t) r) {
                    r++;
                }
                for (auto & st : stage_tensors) {
                    do_run(st, k, r);
                }
                k += r;
            }
        } else {
            for (auto & st : stage_tensors) {
                for (size_t k = plo; k < phi;) {
                    size_t r = 1;
                    while (k + r < phi && pack_used[k + r] == pack_used[k] + (int32_t) r) {
                        r++;
                    }
                    do_run(st, k, r);
                    k += r;
                }
            }
        }
        if (stage) {
            staged_ch = pch;
        }
    };
    // idea P18 (MOE_CACHE_GPU_EVENTS=1, default off): the wait for the queued result copy uses an event recorded right after it, not a synchronize of the whole
    // backend (same wait on one stream; it only differs where the backend has events and other work is queued). Falls back to synchronize without events.
    static const bool ev_env = getenv("MOE_CACHE_GPU_EVENTS") && atoi(getenv("MOE_CACHE_GPU_EVENTS")) > 0;
    auto sync_result = [&] {
        if (ev_env) {
            ggml_backend_event_t ev = ggml_backend_event_new(gdev);
            if (ev) {
                ggml_backend_event_record(ev, c->gpu);
                ggml_backend_event_synchronize(ev);
                ggml_backend_event_free(ev);
                return;
            }
        }
        ggml_backend_synchronize(c->gpu);
    };
    auto next_window = [&](size_t ch) {         // the result copy of part ch is queued: use the GPU's compute time to prepare part ch + 1
        if (rb && ch + 1 < n_chunk && !stage_tensors.empty()) {
            stage_part(ch + 1);
        }
    };
    static const bool res_first = getenv("MOE_CACHE_GPU_RES_FIRST") && atoi(getenv("MOE_CACHE_GPU_RES_FIRST")) > 0;
    std::vector<std::pair<size_t, size_t>> runs;
    static const bool share_up = getenv("MOE_CACHE_GPU_SHARE_UP") && atoi(getenv("MOE_CACHE_GPU_SHARE_UP")) > 0;
    // idea P02 (MOE_CACHE_GPU_RB_ROWS=1): in a split block read back only the rows of this part's experts (runs of neighbouring rows in one copy)
    static const bool rb_rows = getenv("MOE_CACHE_GPU_RB_ROWS") && atoi(getenv("MOE_CACHE_GPU_RB_ROWS")) > 0;
    double up_ns = 0, comp_ns = 0, get_ns = 0;
    static const bool e1 = seg_times && getenv("MOE_CACHE_E1_TIMES") != nullptr;
    double wait_ns = 0;
    uint64_t up_faults = 0;
    auto timed_wait = [&](const ggml_tensor * t, int64_t e0, int64_t n) {
        if (!e1) { wait_experts(s, t, e0, n); return; }
        const auto w0 = std::chrono::steady_clock::now();
        wait_experts(s, t, e0, n);
        wait_ns += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - w0).count();
    };
    std::vector<uint8_t> pack, part;
    const auto t_first_up = std::chrono::steady_clock::now();
    for (size_t ch = 0; ch < n_chunk; ch++) {
        const size_t lo = ch * per_chunk, hi = std::min(pack_used.size(), lo + per_chunk);
        const auto t_up0 = std::chrono::steady_clock::now();
        const uint64_t f_up0 = e1 ? state_t::process_faults() : 0;
        if (stage && staged_ch != ch) {
            stage_part(ch);
        }
        if (ch == 0 && c->pf.awaiting) {          // PL05: the helper must be done before the block reads the buffer or the buffer is filled again
            c->pf.settle();
        }
        for (auto & u : uploads) {
            const ggml_tensor * t = u.first;
            ggml_tensor * d = u.second;
            if (!pf_placed.empty() && pf_placed.count(d) && c->pf.filled) {
                s.gpu_up_saved += ggml_nbytes(d);       // prefetched: the bytes are already there (if the helper failed, the normal upload below fills the same place)
                continue;
            }
            // idea P09 (MOE_CACHE_GPU_SHARE_UP=1): in a split block everything but the part's experts and ids is the same for every part, and the
            // clones are set_input tensors (the allocator never reuses them): upload them with the first part only
            if (share_up && ch > 0 && !pack_weights.count(t) && t != pack_ids) {
                continue;
            }
            if (stage && staged_ch == ch && pack_weights.count(t)) {
                for (auto & st : stage_tensors) {
                    if (st.first == t) {
                        ggml_backend_tensor_set(d, stage + st.second, 0, (hi - lo) * t->nb[2]);
                        s.gpu_up_bytes += (hi - lo) * t->nb[2];
                        break;
                    }
                }
                continue;
            }
            if (pack_weights.count(t)) {
                // the used experts (of this part), in runs of neighbours, into their packed places
                const size_t nb2 = t->nb[2];
                runs.clear();
                for (size_t k = lo; k < hi;) {
                    size_t r = 1;
                    while (k + r < hi && pack_used[k + r] == pack_used[k] + (int32_t) r) {
                        r++;
                    }
                    runs.emplace_back(k, r);
                    k += r;
                }
                if (res_first) {
                    // idea P03 (MOE_CACHE_GPU_RES_FIRST=1): runs whose experts are all in RAM go first, so the reads still running finish while those upload
                    const auto bi = s.by_tensor.find(t);
                    if (bi != s.by_tensor.end()) {
                        const tinfo_t & ti = *s.tensors[bi->second];
                        auto in_ram = [&](const std::pair<size_t, size_t> & q) {
                            for (size_t x = 0; x < q.second; x++) {
                                if (__atomic_load_n(&s.nodes[(size_t) ti.node0 + (size_t) pack_used[q.first + x]].loading, __ATOMIC_ACQUIRE)) {
                                    return false;
                                }
                            }
                            return true;
                        };
                        std::stable_partition(runs.begin(), runs.end(), in_ram);
                    }
                }
                for (const auto & q : runs) {
                    const size_t k = q.first, r = q.second;
                    timed_wait(t, pack_used[k], (int64_t) r);
                    ggml_backend_tensor_set(d, (const char *) t->data + (size_t) pack_used[k] * nb2, (k - lo) * nb2, r * nb2);
                }
                s.gpu_up_bytes += (hi - lo) * nb2;
                continue;
            }
            if (t == pack_ids) {
                std::vector<int32_t> ids((size_t) (t->ne[0] * t->ne[1]));
                for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                    for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                        const int32_t v = pack_map[(size_t) *(const int32_t *) ((const char *) t->data + i0 * t->nb[0] + i1 * t->nb[1])];
                        ids[(size_t) (i1 * t->ne[0] + i0)] = n_chunk == 1 ? v : ((size_t) v >= lo && (size_t) v < hi ? v - (int32_t) lo : 0);
                    }
                }
                ggml_backend_tensor_set(d, ids.data(), 0, ids.size() * sizeof(int32_t));
                continue;
            }
            if (ggml_is_contiguous(t)) {
                bool held_has = false;
                if (!held_for.empty() && held_for.count(d)) {
                    for (auto & x : c->held) {
                        if (x.src == t) {
                            held_has = x.filled;
                            x.filled = true;
                            break;
                        }
                    }
                }
                if (held_has) {
                    s.gpu_up_saved += ggml_nbytes(d);          // the held copy already has these bytes
                    continue;
                }
                timed_wait(t, 0, t->ne[2]);           // whole expert tensors (no-op for activations and ids)
                // idea PL03/VK01 (MOE_CACHE_GPU_UPLOAD_THREADS=N, default 1 = off): a whole expert tensor of 32 MiB or more goes up in N disjoint slices copied by N threads
                // (the calling one and N-1 short-lived helpers). Only worth it if one thread writing through the PCIe window is the limit (the link is Gen5 and the
                // single-thread rate is below it); a helper-thread failure is a warning and that slice is then copied by the caller.
                // SAFE ONLY WHEN THE CARD IS WRITTEN THROUGH ITS PCIe WINDOW (resizable BAR on: the Vulkan buffer is device-local AND host-visible and tensor_set is a plain
                // memcpy, ggml-vulkan-buffers.cpp). Without it tensor_set goes through ONE shared staging buffer and ONE shared fence per device (write_2d: sync_staging,
                // device->fence, waitForFences, resetFences), so two threads corrupt each other's data or hang (audit, round 4). The plugin has no public way to tell: check
                // the heap in `vulkaninfo` (a device-local, host-visible type as large as the card) before using this switch.
                static const int up_threads = []() {
                    const int n = getenv("MOE_CACHE_GPU_UPLOAD_THREADS") ? std::max(1, std::min(8, atoi(getenv("MOE_CACHE_GPU_UPLOAD_THREADS")))) : 1;
                    if (n > 1) {
                        fprintf(stderr, "moe-cache: MOE_CACHE_GPU_UPLOAD_THREADS=%d: uploads are copied by %d threads. Only safe with resizable BAR on (device-local, host-visible "
                                        "memory as large as the card); without it the shared staging buffer and fence of the Vulkan backend are used by all threads at once\n", n, n);
                    }
                    return n;
                }();
                const size_t up_bytes = ggml_nbytes(d);
                if (up_threads > 1 && up_bytes >= ((size_t) 32 << 20)) {
                    const size_t per = ((up_bytes / (size_t) up_threads) + 4095) & ~(size_t) 4095;
                    std::vector<std::thread> hs;
                    std::vector<std::pair<size_t, size_t>> redo;
                    std::mutex redo_mu;
                    for (size_t o = per; o < up_bytes; o += per) {
                        const size_t n2 = std::min(per, up_bytes - o);
                        try {
                            hs.emplace_back([=, &redo, &redo_mu] {
                                ggml_backend_tensor_set(d, (const char *) t->data + o, o, n2);
                            });
                        } catch (const std::exception &) {
                            std::lock_guard<std::mutex> lk(redo_mu);
                            redo.emplace_back(o, n2);
                        }
                    }
                    ggml_backend_tensor_set(d, t->data, 0, std::min(per, up_bytes));
                    for (auto & h : hs) { h.join(); }
                    for (auto & r : redo) {
                        ggml_backend_tensor_set(d, (const char *) t->data + r.first, r.first, r.second);
                    }
                } else {
                    ggml_backend_tensor_set(d, t->data, 0, up_bytes);
                }
                s.gpu_up_bytes += ggml_nbytes(d);
            } else {
                // activations / ids that are views: pack row by row
                const size_t es = ggml_type_size(t->type);
                if (t->nb[0] != es) {
                    ggml_free(ctx);
                    return false;
                }
                pack.assign(ggml_nbytes(d), 0);
                size_t o = 0;
                for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
                    for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
                        for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                            memcpy(pack.data() + o, (const char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], (size_t) t->ne[0] * es);
                            o += (size_t) t->ne[0] * es;
                        }
                    }
                }
                ggml_backend_tensor_set(d, pack.data(), 0, ggml_nbytes(d));
            }
        }
        if (seg_times) {
            ggml_backend_synchronize(c->gpu);
        }
        if (e1) { up_faults += state_t::process_faults() - f_up0; }
        const auto t_c0 = std::chrono::steady_clock::now();
        const enum ggml_status st = ggml_backend_graph_compute(c->gpu, g);
        if (!rb || seg_times) {
            ggml_backend_synchronize(c->gpu);     // seg_times: the compute is timed on its own
        }
        if (st != GGML_STATUS_SUCCESS) {
            ggml_backend_synchronize(c->gpu);
            ggml_free(ctx);
            return false;
        }
        const auto t_g0 = std::chrono::steady_clock::now();
        const char * res = nullptr;           // the block's result on the host (all parts: the part's result)
        if (rb && rb_rows && n_chunk > 1 && pack_ids && ggml_is_contiguous(out) && out->nb[1] == ggml_row_size(out->type, out->ne[0])) {
            const size_t row = out->nb[1];
            size_t run_start = 0, run_len = 0;
            auto flush = [&] {
                if (run_len) {
                    ggml_backend_tensor_get_async(c->gpu, gout, (char *) rb + run_start * row, run_start * row, run_len * row);
                    run_len = 0;
                }
            };
            for (int64_t i1 = 0; i1 < pack_ids->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < pack_ids->ne[0]; i0++) {
                    const size_t v = (size_t) pack_map[(size_t) *(const int32_t *) ((const char *) pack_ids->data + i0 * pack_ids->nb[0] + i1 * pack_ids->nb[1])];
                    const size_t ri = (size_t) i0 + (size_t) i1 * (size_t) pack_ids->ne[0];     // row index in memory order (out->nb[2] == ne[1] * row)
                    if (v >= lo && v < hi) {
                        if (run_len == 0) { run_start = ri; }
                        run_len++;
                    } else {
                        flush();
                    }
                }
            }
            flush();
            next_window(ch);
            sync_result();
            res = (const char *) rb;
        } else if (rb) {
            ggml_backend_tensor_get_async(c->gpu, gout, rb, 0, out_bytes);
            next_window(ch);
            sync_result();
            res = (const char *) rb;
        } else if (n_chunk > 1) {
            part.resize(out_bytes);
            ggml_backend_tensor_get(gout, part.data(), 0, part.size());
            res = (const char *) part.data();
        }
        if (n_chunk == 1) {
            if (res) {
                memcpy(out->data, res, out_bytes);
            } else {
                ggml_backend_tensor_get(gout, out->data, 0, out_bytes);
            }
        } else {
            // keep the rows (one per token and expert slot) whose expert is in this part
            const size_t row = ggml_row_size(out->type, out->ne[0]);
            for (int64_t i1 = 0; i1 < pack_ids->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < pack_ids->ne[0]; i0++) {
                    const size_t v = (size_t) pack_map[(size_t) *(const int32_t *) ((const char *) pack_ids->data + i0 * pack_ids->nb[0] + i1 * pack_ids->nb[1])];
                    if (v >= lo && v < hi) {
                        const size_t off = (size_t) i0 * out->nb[1] + (size_t) i1 * out->nb[2];
                        memcpy((char *) out->data + off, res + off, row);
                    }
                }
            }
        }
        const auto t_end = std::chrono::steady_clock::now();
        up_ns += std::chrono::duration<double, std::nano>(t_c0 - t_up0).count();
        comp_ns += std::chrono::duration<double, std::nano>(t_g0 - t_c0).count();
        get_ns += std::chrono::duration<double, std::nano>(t_end - t_g0).count();
    }
    if (g_vram_watch.on) {
        g_vram_watch.block(gpu_free, gpu_tot, comp_ns / 1e6);
    }
    if (n_chunk > 1 && n_chunk <= 16 && !pack_used.empty()) {
        split_stat_t & ss = split_stat[n_chunk];
        const double v = (up_ns + comp_ns + get_ns) / (double) pack_used.size();
        ss.ema = ss.n == 0 ? v : 0.7 * ss.ema + 0.3 * v;
        ss.n++;
    }
    if (tmpl_env && c->tmpls.size() < 4) {      // keep the finished graph for the next block of this shape (at most 4 shapes)
        moe_cache_backend_ctx::gpu_tmpl_t t;
        t.key = tkey;
        t.ctx = ctx;
        t.g = g;
        for (ggml_tensor * n : seg) {
            t.node_clones.push_back(clone[n]);
        }
        for (auto & u : uploads) {
            t.ext_clones.push_back(u.second);
        }
        c->tmpls.push_back(std::move(t));
    } else {
        ggml_free(ctx);
    }
    if (!pf_placed.empty()) {
        c->pf.used++;
    }
    if (pf_can && n_chunk == 1 && !pack_ids && !from_tmpl) {
        // PL05: learn which block follows this one, and start copying that block's weights into the prefetch buffer. Everything below is a guess that costs
        // time at worst: the next block checks the tensors it really needs against the buffer.
        std::vector<const ggml_tensor *> cur;
        for (ggml_tensor * n : seg) {
            const ggml_tensor * w = n->op == GGML_OP_MUL_MAT_ID ? n->src[0] : nullptr;
            if (w && !inside.count(w) && std::find(cur.begin(), cur.end(), w) == cur.end()) {
                cur.push_back(w);
            }
        }
        bool plain = !cur.empty();
        for (const ggml_tensor * w : cur) {
            plain = plain && s.by_tensor.count(w) && ggml_is_contiguous(w) && !s.tensors[s.by_tensor.find(w)->second]->repack;
        }
        if (plain) {
            if (c->pf.last_first) {
                c->pf.next_of[c->pf.last_first] = cur;
            }
            c->pf.last_first = cur[0];
            c->pf.settle();
            const auto nx = c->pf.next_of.find(cur[0]);
            if (c->pf.launched >= 32 && c->pf.used * 2 < c->pf.launched) {
                c->pf.off = true;
                c->pf.release();
                fprintf(stderr, "moe-cache: prefetch: only %llu of %llu guesses were used: MOE_CACHE_GPU_PREFETCH turns itself off\n", (unsigned long long) c->pf.used,
                        (unsigned long long) c->pf.launched);
            } else if (nx != c->pf.next_of.end() && s.graphs.load() >= c->pf.retry_at) {
                const std::vector<const ggml_tensor *> & want = nx->second;
                size_t w_bytes = 0;
                bool same = c->pf.buf && c->pf.src.size() == want.size();
                for (size_t i = 0; i < want.size(); i++) {
                    w_bytes += ggml_nbytes(want[i]);
                    same = same && c->pf.dst[i]->type == want[i]->type && memcmp(c->pf.dst[i]->ne, want[i]->ne, sizeof(want[i]->ne)) == 0;
                }
                size_t fr = 0, tt = 0;
                ggml_backend_dev_memory(gdev, &fr, &tt);
                if (c->pf.buf && tt != 0 && fr < ((size_t) 512 << 20)) {      // the card is filling up (llama.cpp's buffers grow with the prompt): give the room back
                    c->pf.release();
                    c->pf.retry_at = s.graphs.load() + 64;
                    same = false;
                    w_bytes = 0;
                } else if (!same) {
                    c->pf.release();
                    if (tt == 0 || fr >= w_bytes + ((size_t) 768 << 20)) {
                        ggml_init_params pp = { ggml_tensor_overhead() * (want.size() + 2) + 1024, nullptr, true };
                        c->pf.ctx = ggml_init(pp);
                        for (size_t i = 0; c->pf.ctx && i < want.size(); i++) {
                            ggml_tensor * dt = ggml_new_tensor(c->pf.ctx, want[i]->type, GGML_MAX_DIMS, want[i]->ne);
                            ggml_set_input(dt);
                            c->pf.dst.push_back(dt);
                        }
                        c->pf.buf = c->pf.ctx ? ggml_backend_alloc_ctx_tensors(c->pf.ctx, c->gpu) : nullptr;
                        if (!c->pf.buf) {
                            c->pf.release();
                            c->pf.retry_at = s.graphs.load() + 64;
                        } else {
                            c->pf.bytes = ggml_backend_buffer_get_size(c->pf.buf);
                            static std::atomic<int> said_pf{0};
                            if (said_pf++ < 3) {
                                fprintf(stderr, "moe-cache: prefetch: %.0f MiB of GPU memory for the next block's weights (%.0f MiB were free)\n", c->pf.bytes / 1048576.0, fr / 1048576.0);
                            }
                        }
                    } else {
                        c->pf.retry_at = s.graphs.load() + 64;      // no room: do nothing
                    }
                }
                if (c->pf.buf && w_bytes) {
                    c->pf.src = want;
                    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> todo;
                    for (size_t i = 0; i < want.size(); i++) {
                        todo.emplace_back(want[i], c->pf.dst[i]);
                    }
                    state_t * sp = &s;
                    gpu_prefetch_t * pfp = &c->pf;
                    c->pf.launched++;
                    c->pf.awaiting = true;
                    c->pf.filled = false;
                    c->pf.launch([sp, pfp, todo] {
                        const auto t0 = std::chrono::steady_clock::now();
                        bool ok = true;
                        for (const auto & q : todo) {
                            const ggml_tensor * t = q.first;
                            const auto it = sp->by_tensor.find(t);
                            if (it == sp->by_tensor.end()) {
                                ok = false;
                                break;
                            }
                            const tinfo_t & ti = *sp->tensors[it->second];
                            wait_experts(*sp, t, 0, ti.n_expert);
#ifdef _WIN32
                            {       // the same rule as the block's own whole-tensor upload: every expert in RAM or mapped, else nothing is copied
                                std::lock_guard<std::mutex> lk(sp->mu);
                                for (int64_t e = 0; e < ti.n_expert; e++) {
                                    const auto & nd = sp->nodes[(size_t) ti.node0 + (size_t) e];
                                    if (!(nd.resident && !nd.loading) && !nd.mapped) {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            if (!ok) {
                                break;
                            }
#endif
                            const size_t total = ggml_nbytes(q.second);
                            for (size_t o = 0; o < total; o += ((size_t) 32 << 20)) {      // slices: the device lock is held for one slice at a time
                                const size_t n = std::min<size_t>((size_t) 32 << 20, total - o);
                                ggml_backend_tensor_set(q.second, (const char *) t->data + o, o, n);
                            }
                            pfp->copied_bytes += total;
                        }
                        pfp->busy_ns += (uint64_t) std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
                        return ok;
                    });
                }
            }
        }
    }
    s.gpu_segments++;
    if (n_chunk > 1) {
        s.gpu_split_blocks++;
    }
    if (e1) {
        const int k = n_chunk > 1 ? 1 : 0;
        const uint64_t rows = pack_ids ? (uint64_t) (pack_ids->ne[0] * pack_ids->ne[1]) : 0;
        s.e1_blk_n[k]++; s.e1_parts[k] += n_chunk;
        s.e1_prep_ns[k] += (uint64_t) std::chrono::duration<double, std::nano>(t_first_up - t_fn0).count();
        s.e1_up_ns[k] += (uint64_t) up_ns; s.e1_wait_ns[k] += (uint64_t) wait_ns; s.e1_up_faults[k] += up_faults;
        s.e1_comp_ns[k] += (uint64_t) comp_ns; s.e1_get_ns[k] += (uint64_t) get_ns;
        s.e1_rows_comp[k] += rows * n_chunk; s.e1_rows_use[k] += rows;
    }
    if (seg_times) {
        static double up_ms = 0, comp_ms = 0, get_ms = 0;
        static uint64_t n = 0;
        up_ms += up_ns / 1e6;
        comp_ms += comp_ns / 1e6;
        get_ms += get_ns / 1e6;
        if ((++n & 63) == 0) {
            const double tot = up_ms + comp_ms + get_ms;
            fprintf(stderr, "moe-cache: long-prompt blocks: %llu (split %llu), upload %.0f ms (%.0f %%), compute %.0f ms (%.0f %%), read back %.0f ms\n", (unsigned long long) n,
                    (unsigned long long) s.gpu_split_blocks.load(), up_ms, 100 * up_ms / tot, comp_ms, 100 * comp_ms / tot, get_ms);
            if (pf_env) {
                fprintf(stderr, "moe-cache: prefetch: %llu guesses, %llu used, %llu failed, %.1f GB copied by the helper in %.0f ms, the graph thread waited %.0f ms, %.0f MiB of GPU memory\n",
                        (unsigned long long) c->pf.launched, (unsigned long long) c->pf.used, (unsigned long long) c->pf.failed, c->pf.copied_bytes.load() / 1e9,
                        c->pf.busy_ns.load() / 1e6, c->pf.wait_ns / 1e6, c->pf.bytes / 1048576.0);
            }
        }
    }
    return true;
}

}  // namespace
