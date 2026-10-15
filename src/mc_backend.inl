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
std::vector<int> pin_cpus() {
    std::vector<int> out;
#ifdef __linux__
    const char * off = getenv("MOE_CACHE_CPU_PIN");
    if (off && strcmp(off, "off") == 0) {
        return out;
    }
    if (const char * m = getenv("MOE_CACHE_CPU_MASK")) {
        const size_t n = strlen(m);
        for (size_t i = 0; i < n; i++) {
            const char ch = m[n - 1 - i];
            const int v = ch >= '0' && ch <= '9' ? ch - '0' : (ch | 32) >= 'a' && (ch | 32) <= 'f' ? (ch | 32) - 'a' + 10 : 0;
            for (int b = 0; b < 4; b++) {
                if ((v & (1 << b)) && i * 4 + b < GGML_MAX_N_THREADS) {   // cpumask has GGML_MAX_N_THREADS entries
                    out.push_back((int) (i * 4 + b));
                }
            }
        }
        return out;
    }
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
    int                n_threads = 6;
    ggml_backend_t     gpu = nullptr;
    ggml_gallocr_t     galloc = nullptr;
    bool               gpu_failed = false;
    bool               pin_reported = false;
    // MOE_CACHE_HYBRID=MiB: a bank of the most used experts in GPU memory (from the usage profile); per weight tensor: bank tensor and slot of each expert
    bool               hb_tried = false;
    ggml_context *     hb_ctx = nullptr;
    ggml_backend_buffer_t hb_buf = nullptr;
    bool               hb_dropped = false;   // the bank was freed so a long prompt's GPU block fits; it is loaded again at the next decode step
    bool               seg_nomem = false;    // the last gpu_segment() call failed for lack of free GPU memory
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
    if (c->galloc) {
        ggml_gallocr_free(c->galloc);
    }
    if (c->hb_buf) {                  // hybrid bank (GPU memory): freed before its backend
        ggml_backend_buffer_free(c->hb_buf);
    }
    if (c->hb_ctx) {
        ggml_free(c->hb_ctx);
    }
    if (c->gpu) {
        ggml_backend_free(c->gpu);
    }
    delete c;
    delete backend;
}

// Run the expert block (MUL_MAT_ID / GLU nodes first..last) on the GPU: clone the nodes into a small graph, upload the weights and inputs
// to a reusable staging area, compute there and copy the last node's result back. Returns false (nothing changed) if anything does not fit.
bool gpu_segment(moe_cache_backend_ctx * c, state_t & s, ggml_cgraph * cg, int first, int last, std::vector<ggml_tensor *> & seg) {
    const int n_all = ggml_graph_n_nodes(cg);
    std::unordered_set<const ggml_tensor *> inside;
    for (int i = first; i <= last; i++) {
        ggml_tensor * n = ggml_graph_node(cg, i);
        if (n->op != GGML_OP_MUL_MAT_ID && n->op != GGML_OP_GLU) {
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
    std::unordered_map<const ggml_tensor *, ggml_tensor *> clone;
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> uploads;
    auto ext = [&](const ggml_tensor * t) {
        auto it = clone.find(t);
        if (it != clone.end()) {
            return it->second;
        }
        ggml_tensor * nt = ggml_new_tensor(ctx, t->type, GGML_MAX_DIMS, t->ne);
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
    if (!ggml_gallocr_alloc_graph(c->galloc, g)) {
        ggml_free(ctx);
        c->gpu_failed = true;
        return false;
    }
    std::vector<uint8_t> pack;
    for (auto & u : uploads) {
        const ggml_tensor * t = u.first;
        ggml_tensor * d = u.second;
        if (ggml_is_contiguous(t)) {
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
    const enum ggml_status st = ggml_backend_graph_compute(c->gpu, g);
    ggml_backend_synchronize(c->gpu);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(gout, out->data, 0, ggml_nbytes(out));
    ggml_free(ctx);
    s.gpu_segments++;
    return true;
}

}  // namespace
