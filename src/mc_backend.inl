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
    bool               rb_failed = false;    // no pinned host buffer type, or it could not be allocated: plain blocking read-back
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
    if (c->galloc) {
        ggml_gallocr_free(c->galloc);
    }
    if (c->hb_buf) {                  // hybrid bank (GPU memory): freed before its backend
        ggml_backend_buffer_free(c->hb_buf);
    }
    if (c->hb_ctx) {
        ggml_free(c->hb_ctx);
    }
    if (c->rb_buf) {                  // pinned read-back buffer: freed before its backend
        ggml_backend_buffer_free(c->rb_buf);
    }
    if (c->gpu) {
        ggml_backend_free(c->gpu);
    }
    delete c;
    delete backend;
}

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

// Run the expert block (MUL_MAT_ID / GLU nodes first..last) on the GPU: clone the nodes into a small graph, upload the weights and inputs
// to a reusable staging area, compute there and copy the last node's result back. Returns false (nothing changed) if anything does not fit.
bool gpu_segment(moe_cache_backend_ctx * c, state_t & s, ggml_cgraph * cg, int first, int last, std::vector<ggml_tensor *> & seg) {
    const int n_all = ggml_graph_n_nodes(cg);
    std::unordered_set<const ggml_tensor *> inside;
    for (int i = first; i <= last; i++) {
        ggml_tensor * n = ggml_graph_node(cg, i);
        if (n->op != GGML_OP_MUL_MAT_ID && n->op != g_op_glu) {
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
    std::unordered_set<const ggml_tensor *> pack_weights;
    if (pack_ids) {
        for (ggml_tensor * n : seg) {
            if (n->op == GGML_OP_MUL_MAT_ID) {
                pack_weights.insert(n->src[0]);
            }
        }
    }
    std::unordered_map<const ggml_tensor *, ggml_tensor *> clone;
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> uploads;
    auto ext = [&](const ggml_tensor * t) {
        auto it = clone.find(t);
        if (it != clone.end()) {
            return it->second;
        }
        int64_t ne[GGML_MAX_DIMS] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
        if (pack_weights.count(t)) {
            ne[2] = (int64_t) pack_used.size();
        }
        ggml_tensor * nt = ggml_new_tensor(ctx, t->type, GGML_MAX_DIMS, ne);
        ggml_set_input(nt);
        clone[t] = nt;
        uploads.emplace_back(t, nt);
        return nt;
    };
    for (ggml_tensor * n : seg) {
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
    ggml_tensor * gout = clone[out];
    ggml_set_output(gout);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, gout);
    if (!gpu_graph_supported(c, g)) {
        ggml_free(ctx);
        return false;
    }
    {
        // a whole layer of expert weights is uploaded for this block: with a GPU expert cache next to it the card can be full.
        // Check first and keep the block on the CPU instead of letting the allocation (or the CUDA pool) fail hard.
        size_t need = 0;
        for (auto & u : uploads) {
            need += ggml_nbytes(u.second);
        }
        for (ggml_tensor * n : seg) {
            need += ggml_nbytes(clone[n]);
        }
        const size_t held = ggml_gallocr_get_buffer_size(c->galloc, 0);       // already allocated for an earlier block: reused
        const size_t extra = need > held ? need - held : 0;
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(s.gpu_dev, &fr, &tot);
        const size_t headroom = (size_t) 256 << 20;                              // pool buffers, graph, fragmentation
        if (tot > 0 && fr < extra + headroom) {
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
    const auto t_up0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> pack;
    for (auto & u : uploads) {
        const ggml_tensor * t = u.first;
        ggml_tensor * d = u.second;
        if (pack_weights.count(t)) {
            // the used experts, in runs of neighbours, into their packed places
            const size_t nb2 = t->nb[2];
            for (size_t k = 0; k < pack_used.size();) {
                size_t r = 1;
                while (k + r < pack_used.size() && pack_used[k + r] == pack_used[k] + (int32_t) r) {
                    r++;
                }
                wait_experts(s, t, pack_used[k], (int64_t) r);
                ggml_backend_tensor_set(d, (const char *) t->data + (size_t) pack_used[k] * nb2, k * nb2, r * nb2);
                k += r;
            }
            s.gpu_up_bytes += pack_used.size() * nb2;
            continue;
        }
        if (t == pack_ids) {
            std::vector<int32_t> ids((size_t) (t->ne[0] * t->ne[1]));
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    ids[(size_t) (i1 * t->ne[0] + i0)] = pack_map[(size_t) *(const int32_t *) ((const char *) t->data + i0 * t->nb[0] + i1 * t->nb[1])];
                }
            }
            ggml_backend_tensor_set(d, ids.data(), 0, ids.size() * sizeof(int32_t));
            continue;
        }
        if (ggml_is_contiguous(t)) {
            wait_experts(s, t, 0, t->ne[2]);           // whole expert tensors (no-op for activations and ids)
            ggml_backend_tensor_set(d, t->data, 0, ggml_nbytes(d));
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
    // N22 (on by default; MOE_CACHE_GPU_READBACK=0 = old path): the result goes into a pinned host buffer of the GPU device, queued right after
    // the compute with one synchronize for both, then one memcpy into the CPU tensor. A blocking read into ordinary memory goes through the
    // backend's staging copy (~1.6 GB/s on the 9070 XT, 18 % of a block). Same bytes either way.
    static const bool rb_env = !getenv("MOE_CACHE_GPU_READBACK") || atoi(getenv("MOE_CACHE_GPU_READBACK")) > 0;
    const size_t out_bytes = ggml_nbytes(out);
    void * rb = nullptr;
    if (rb_env && !c->rb_failed) {
        if (!c->rb_buf || ggml_backend_buffer_get_size(c->rb_buf) < out_bytes) {
            ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(s.gpu_dev);
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
    if (rb) {
        ggml_backend_tensor_get_async(c->gpu, gout, rb, 0, out_bytes);
        ggml_backend_synchronize(c->gpu);
        memcpy(out->data, rb, out_bytes);
    } else {
        ggml_backend_tensor_get(gout, out->data, 0, out_bytes);
    }
    ggml_free(ctx);
    s.gpu_segments++;
    if (seg_times) {
        static double up_ms = 0, comp_ms = 0, get_ms = 0;
        static uint64_t n = 0;
        const auto t_end = std::chrono::steady_clock::now();
        up_ms += std::chrono::duration<double, std::milli>(t_c0 - t_up0).count();
        comp_ms += std::chrono::duration<double, std::milli>(t_g0 - t_c0).count();
        get_ms += std::chrono::duration<double, std::milli>(t_end - t_g0).count();
        if ((++n & 63) == 0) {
            const double tot = up_ms + comp_ms + get_ms;
            fprintf(stderr, "moe-cache: long-prompt blocks: %llu, upload %.0f ms (%.0f %%), compute %.0f ms (%.0f %%), read back %.0f ms\n", (unsigned long long) n,
                    up_ms, 100 * up_ms / tot, comp_ms, 100 * comp_ms / tot, get_ms);
        }
    }
    return true;
}

}  // namespace
