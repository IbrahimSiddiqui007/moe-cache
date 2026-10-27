// moe-cache plugin part: graph compute (executor graphs, GPU block, CPU).
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

// Last check before llama.cpp's own kernels read the experts: every expert a MUL_MAT_ID of this graph uses must be complete and in RAM. Waits for
// experts still being read; an expert that is not in RAM at all (should never happen: eviction protects the whole graph) is loaded again with a
// one-time message instead of being read as decommitted memory (Windows: access violation, PC 2026-10-27) or zeros (Linux: wrong output).
void cpu_path_check(state_t & s, ggml_cgraph * cg) {
    const int n_all = ggml_graph_n_nodes(cg);
    for (int i = 0; i < n_all; i++) {
        ggml_tensor * nd = ggml_graph_node(cg, i);
        if (nd->op != GGML_OP_MUL_MAT_ID) {
            continue;            // also the ops the GPU block took over (switched to NONE)
        }
        const auto it = s.by_tensor.find(nd->src[0]);
        if (it == s.by_tensor.end()) {
            continue;
        }
#ifndef MOE_CACHE_TEST_MUTANT_NO_STEPASIDE_WAIT     // CI only: the mutant drops every wait before llama.cpp's CPU path, this one too (else it masks the missing step-aside wait)
        exec_wait_loaded(s, nd);
#endif
        const tinfo_t & ti = *s.tensors[(size_t) it->second];
        const ggml_tensor * ids = nd->src[2];
        std::vector<int> cnt((size_t) ti.n_expert, 0);
        int missing = 0;
        {
            std::lock_guard<std::mutex> lk(s.mu);
            for (int64_t t = 0; t < ids->ne[1]; t++) {
                for (int64_t k = 0; k < ids->ne[0]; k++) {
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + k * ids->nb[0] + t * ids->nb[1]);
                    if (e < 0 || e >= ti.n_expert) {
                        continue;
                    }
                    cnt[(size_t) e] = 1;
                    const node_t & n = s.nodes[(size_t) (ti.node0 + e)];
                    missing += (n.resident || n.mapped) ? 0 : 1;
                }
            }
        }
        if (missing) {
            static std::atomic<int> said{0};
            if (said++ == 0) {
                fprintf(stderr, "moe-cache: internal: %d expert(s) of %s (%lld tokens) were not in RAM when llama.cpp was about to read them; loaded again "
                        "(output stays correct; please report this line)\n", missing, nd->src[0]->name, (long long) ids->ne[1]);
            }
            s.cpu_check_reloads++;
            s.prepare_group(std::vector<int>{ it->second }, cnt, false);
        }
    }
}

enum ggml_status exec_run_graph(moe_cache_backend_ctx * c, state_t & s, ggml_cgraph * cg) {
    if (!s.pool) {
        const int n = s.exec_threads > 0 ? s.exec_threads : std::max(1, c->n_threads);
        static const std::vector<int> pins = pin_cpus();
        std::vector<int> cpus;
        for (int i = 1; i < n; i++) {
            cpus.push_back(pins.size() >= (size_t) n ? pins[(size_t) i] : -1);
        }
        int extra = 0;
#ifdef __linux__
        if (s.exec_ecores > 0) {
            std::ifstream f("/sys/devices/cpu_atom/cpus");
            std::string txt;
            if (f && std::getline(f, txt)) {
                size_t i = 0;
                while (i < txt.size() && extra < s.exec_ecores) {
                    int a = atoi(txt.c_str() + i), b = a;
                    while (i < txt.size() && isdigit((unsigned char) txt[i])) { i++; }
                    if (i < txt.size() && txt[i] == '-') {
                        i++;
                        b = atoi(txt.c_str() + i);
                        while (i < txt.size() && isdigit((unsigned char) txt[i])) { i++; }
                    }
                    for (int cpu = a; cpu <= b && extra < s.exec_ecores; cpu++) { cpus.push_back(cpu); extra++; }
                    while (i < txt.size() && !isdigit((unsigned char) txt[i])) { i++; }
                }
            }
        }
#else
        // no sysfs: the launcher passes the efficiency cores as a mask (Windows: GetLogicalProcessorInformationEx, lowest efficiency class)
        if (s.exec_ecores > 0) {
            if (const char * m = getenv("MOE_CACHE_ECORE_MASK")) {
                for (int cpu : mask_cpus(m)) {
                    if (extra >= s.exec_ecores) {
                        break;
                    }
                    cpus.push_back(cpu);
                    extra++;
                }
            }
        }
#endif
        s.pool.reset(new exec_pool_t());
        s.pool->start(cpus, (int) cpus.size());
        fprintf(stderr, "moe-cache: executor pool: %d compute threads (%d on efficiency cores)\n", (int) cpus.size() + 1, extra);
    }
    if (CPU().set_n_threads) {
        CPU().set_n_threads(c->cpu, 1);       // the small nodes between the matrix products need no pool
    }
    const int n = ggml_graph_n_nodes(cg);
    std::vector<int> batch;
    enum ggml_status st = GGML_STATUS_SUCCESS;
    for (int i = 0; i < n && st == GGML_STATUS_SUCCESS; i++) {
        ggml_tensor * nd = ggml_graph_node(cg, i);
        if (nd->op == GGML_OP_MUL_MAT_ID && s.by_tensor.find(nd->src[0]) != s.by_tensor.end() && hybrid_init(c, s)) {
            st = exec_run_nodes(c, cg, batch);
            batch.clear();
            int last = i;
            if (st == GGML_STATUS_SUCCESS && hybrid_group(c, s, cg, i, last)) {
                i = last;
                continue;
            }
        }
        if (nd->op == GGML_OP_MUL_MAT_ID && s.by_tensor.find(nd->src[0]) != s.by_tensor.end() && !stock_uses_tiled(nd)) {
            st = exec_run_nodes(c, cg, batch);
            batch.clear();
            exec_one_op(s, nd);
            if (s.exec_check) {
                const size_t nb = ggml_nbytes(nd);
                std::vector<char> mine((const char *) nd->data, (const char *) nd->data + nb);
                memset(nd->data, 0, nb);
                exec_run_nodes(c, cg, std::vector<int>{ i });
                s.exec_checked++;
                if (memcmp(mine.data(), nd->data, nb) != 0) {
                    if (++s.exec_mismatch <= 8) {   // say where: weight type, tokens in the batch, first differing value
                        const float * x = (const float *) mine.data(), * y = (const float *) nd->data;
                        size_t k = 0;
                        while (k + 1 < nb / sizeof(float) && x[k] == y[k]) { k++; }
                        fprintf(stderr, "moe-cache: executor mismatch %s: type %s, %lld tokens, %lld picks, first at float %zu: %.9g vs %.9g\n", nd->name,
                                ggml_type_name(nd->src[0]->type), (long long) nd->ne[2], (long long) nd->src[2]->ne[0], k, x[k], y[k]);
                        // recompute that one value by hand with llama.cpp's kernel: which side is the odd one out, and for which expert
                        const ggml_tensor * w = nd->src[0], * a = nd->src[1], * idt = nd->src[2];
                        const int64_t r = (int64_t) k % nd->ne[0], slot = ((int64_t) k / nd->ne[0]) % nd->ne[1], t = (int64_t) k / (nd->ne[0] * nd->ne[1]);
                        const int32_t e = *(const int32_t *) ((const char *) idt->data + slot * idt->nb[0] + t * idt->nb[1]);
                        const ggml_type_traits_cpu * tt = cpu_traits(w->type);
                        std::vector<uint8_t> col(ggml_row_size(tt->vec_dot_type, a->ne[0]) + 64);
                        const float * af = (const float *) ((const char *) a->data + (slot % a->ne[1]) * a->nb[1] + t * a->nb[2]);
                        cpu_traits(tt->vec_dot_type)->from_float(af, col.data(), a->ne[0]);
                        float v = 0;
                        tt->vec_dot((int) w->ne[0], &v, 0, (const char *) w->data + (size_t) e * w->nb[2] + (size_t) r * w->nb[1], 0, col.data(), 0, 1);
                        int same_e = 0;
                        for (int64_t tt2 = 0; tt2 < idt->ne[1]; tt2++) for (int64_t s2 = 0; s2 < idt->ne[0]; s2++)
                            same_e += *(const int32_t *) ((const char *) idt->data + s2 * idt->nb[0] + tt2 * idt->nb[1]) == e;
                        fprintf(stderr, "moe-cache:   row %lld, pick %lld, token %lld, expert %d (picked %d times in this batch); by hand %.9g -> %s; activation type %s, ne1 %lld\n",
                                (long long) r, (long long) slot, (long long) t, e, same_e, v, v == x[k] ? "matches the executor" : v == y[k] ? "matches stock" : "matches neither",
                                ggml_type_name(a->type), (long long) a->ne[1]);
                    }
                    memcpy(nd->data, mine.data(), nb);
                }
            }
        } else {
#ifndef MOE_CACHE_TEST_MUTANT_NO_TILED_WAIT    // CI only: a build without this wait must fail the delayed-read batch-8 KL step (T39)
            if (nd->op == GGML_OP_MUL_MAT_ID && s.by_tensor.find(nd->src[0]) != s.by_tensor.end()) {
                exec_wait_loaded(s, nd);   // left to llama.cpp (tiled): its experts may still be filling (reads are asynchronous in executor graphs)
            }
#endif
            batch.push_back(i);
        }
    }
    if (st == GGML_STATUS_SUCCESS) {
        st = exec_run_nodes(c, cg, batch);
    }
    s.route_ids.clear();
    return st;
}

enum ggml_status be_compute(ggml_backend_t backend, ggml_cgraph * cg) {
    auto * c = (moe_cache_backend_ctx *) backend->context;
    state_t & s = S();
    const auto t_in = std::chrono::steady_clock::now();
    struct total_t { state_t & s; std::chrono::steady_clock::time_point t; ~total_t() { s.ns_total += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count(); } } total_guard{ s, t_in };
    if (!c->cpu) {
        c->cpu = ggml_backend_dev_init(CPU().dev, nullptr);
    }
    if (c->tp_n != c->n_threads && CPU().tp_new && CPU().set_tp) {
        if (c->tp && CPU().tp_free) {
            CPU().tp_free(c->tp);
        }
        ggml_threadpool_params tpp;
        memset(&tpp, 0, sizeof(tpp));
        tpp.n_threads = c->n_threads;
        tpp.prio = GGML_SCHED_PRIO_NORMAL;
        tpp.poll = 50;
        static const std::vector<int> pins = pin_cpus();
        if (!pins.empty() && (size_t) c->n_threads <= pins.size()) {
            for (int i = 0; i < c->n_threads; i++) {
                if (pins[(size_t) i] >= 0 && pins[(size_t) i] < GGML_MAX_N_THREADS) {
                    tpp.cpumask[pins[(size_t) i]] = true;
                }
            }
            tpp.strict_cpu = true;
            if (!c->pin_reported) {
                c->pin_reported = true;
                fprintf(stderr, "moe-cache: compute threads pinned to %d performance cores (MOE_CACHE_CPU_PIN=off to disable)\n", c->n_threads);
            }
        }
        if (const char * v = getenv("MOE_CACHE_POLL")) {
            tpp.poll = (uint32_t) atoi(v);
        }
        c->tp = CPU().tp_new(&tpp);
        c->tp_n = c->n_threads;
        CPU().set_tp(c->cpu, c->tp);
    }
    if (CPU().set_n_threads) {
        CPU().set_n_threads(c->cpu, c->n_threads);
    }

    s.graphs++;
    {
        static bool reported = false;
        if (!reported) {
            reported = true;
            print_capabilities(s);
        }
    }
    if (!s.cache_on) {
        for (int i = 0; i < ggml_graph_n_nodes(cg); i++) {
            if (ggml_graph_node(cg, i)->op == GGML_OP_MUL_MAT_ID) {
                s.ops++;
            }
        }
    }
    struct group_t { int first; int last; int n_tok; };
    std::vector<group_t> groups_found;
    bool exec_graph = s.cache_on && exec_eligible(s, cg);
    // --fast (routing) and --hybrid live in the executor path: when asked for explicitly they keep it, whatever the weight type
    static const bool hybrid_asked = [] { const char * e = getenv("MOE_CACHE_HYBRID"); return e && strcmp(e, "0") != 0; }();
    if (exec_graph && !s.exec_force && !s.route_mode && !hybrid_asked) {
        // the executor only pays for weights whose dot product is compute-bound (IQ types, Q2_0: our kernels and parallel reads win); for K-quants
        // and legacy types llama.cpp's own (repacked) kernels are faster even while some experts still have to be read
        // (Qwen3.6-35B Q4_K_M at 12 GB: 7.8 tok/s with the executor, 17.2 without)
        bool heavy = false;
        const int n_all = ggml_graph_n_nodes(cg);
        for (int i = 0; i < n_all && !heavy; i++) {
            const ggml_tensor * nd = ggml_graph_node(cg, i);
            if (nd->op == GGML_OP_MUL_MAT_ID && s.by_tensor.find(nd->src[0]) != s.by_tensor.end()) {
                const ggml_type wt = nd->src[0]->type;
                heavy = wt == GGML_TYPE_IQ1_S || wt == GGML_TYPE_IQ1_M || wt == GGML_TYPE_IQ2_XXS || wt == GGML_TYPE_IQ2_XS || wt == GGML_TYPE_IQ2_S ||
                        wt == GGML_TYPE_IQ3_XXS || wt == GGML_TYPE_IQ3_S || wt == GGML_TYPE_Q2_0;
            }
        }
        if (!heavy) {
            exec_graph = false;
        }
    }
    size_t exec_jobs = 0;          // reads queued for this graph
    bool exec_heavy = false;       // weights whose dot product is compute-bound (IQ types, Q2_0)
    if (s.cache_on) {
        if (!s.preload_done) {
#ifdef _WIN32
            s.release_model_pages();
#endif
            s.preload();
        }
        s.begin_graph();       // the experts of every group below stay protected until the next graph
        const int n_nodes = ggml_graph_n_nodes(cg);
        std::vector<char> done((size_t) n_nodes, 0);
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * node = ggml_graph_node(cg, i);
            if (done[i] || node->op != GGML_OP_MUL_MAT_ID || s.by_tensor.find(node->src[0]) == s.by_tensor.end()) {
                continue;
            }
            ggml_tensor * ids = node->src[2];
            for (int k = 0; k < i; k++) {
                if (ggml_graph_node(cg, k) == ids) {
                    fprintf(stderr, "moe-cache: ids tensor is computed inside the same split, not supported\n");
                    abort();
                }
            }
            std::vector<int> tidx;
            std::vector<int> used((size_t) node->src[0]->ne[2], 0);
            for (int j = i; j < n_nodes; j++) {
                ggml_tensor * n2 = ggml_graph_node(cg, j);
                if (n2->op != GGML_OP_MUL_MAT_ID || n2->src[2] != ids) {
                    continue;
                }
                auto it2 = s.by_tensor.find(n2->src[0]);
                if (it2 == s.by_tensor.end()) {
                    continue;
                }
                if (std::find(tidx.begin(), tidx.end(), it2->second) == tidx.end()) {
                    tidx.push_back(it2->second);
                }
                done[j] = 1;
                s.ops++;
            }
            const int64_t n_used = ids->ne[0];
            const int64_t n_tok = ids->ne[1];
            for (int64_t t = 0; t < n_tok; t++) {
                for (int64_t u = 0; u < n_used; u++) {
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + u * ids->nb[0] + t * ids->nb[1]);
                    if (e >= 0 && e < (int32_t) used.size()) {
                        used[(size_t) e]++;
                    }
                }
            }
            groups_found.push_back({ i, 0, (int) n_tok });
            for (int j = 0; j < n_nodes; j++) {
                if (done[j] && ggml_graph_node(cg, j)->src[2] == ids) {
                    groups_found.back().last = j;
                }
            }
            const auto p0 = std::chrono::steady_clock::now();
            s.cur_decode = n_tok == 1;
            s.cur_scan = s.scan_tokens > 0 && n_tok >= s.scan_tokens;
            if (s.trace_f && n_tok == 1) {
                int16_t hdr[2] = { (int16_t) s.tensors[tidx[0]]->layer, (int16_t) n_used };
                fwrite(hdr, sizeof(int16_t), 2, s.trace_f);
                for (int64_t u = 0; u < n_used; u++) {
                    const int16_t e = (int16_t) *(const int32_t *) ((const char *) ids->data + u * ids->nb[0]);
                    fwrite(&e, sizeof(int16_t), 1, s.trace_f);
                }
            }
            {
                const ggml_type wt = s.tensors[(size_t) tidx[0]]->tensor->type;
                exec_heavy = exec_heavy || wt == GGML_TYPE_IQ1_S || wt == GGML_TYPE_IQ1_M || wt == GGML_TYPE_IQ2_XXS || wt == GGML_TYPE_IQ2_XS ||
                             wt == GGML_TYPE_IQ2_S || wt == GGML_TYPE_IQ3_XXS || wt == GGML_TYPE_IQ3_S || wt == GGML_TYPE_Q2_0;
            }
            if (s.route_mode && exec_graph && n_tok <= s.exec_maxcol && !tidx.empty() && n_used >= 3) {
                const tinfo_t & t0 = *s.tensors[(size_t) tidx[0]];
                // MOE_CACHE_ROUTE_DROP (default 2; --fast=2 sets 4, RESULTS 53): the lowest-ranked picks that may be skipped; at least half are always honoured
                const int keep = s.route_keep >= 0 ? (int) std::min<int64_t>(s.route_keep, n_used) : (int) std::max<int64_t>(n_used - s.route_drop, (n_used + 1) / 2);
                std::vector<int32_t> eff((size_t) (n_used * n_tok));
                for (int64_t t = 0; t < n_tok; t++) {
                    std::vector<char> taken((size_t) t0.n_expert, 0);
                    for (int64_t u = 0; u < n_used; u++) {
                        const int32_t e = *(const int32_t *) ((const char *) ids->data + u * ids->nb[0] + t * ids->nb[1]);
                        eff[(size_t) (t * n_used + u)] = e;
                        if (e >= 0 && e < (int32_t) taken.size()) taken[(size_t) e] = 1;
                    }
                    for (int64_t u = keep; u < n_used; u++) {
                        const int32_t e = eff[(size_t) (t * n_used + u)];
                        if (e < 0 || e >= (int32_t) t0.n_expert) continue;
                        s.route_considered++;
                        if (s.nodes[(size_t) (t0.node0 + e)].resident) continue;     // in RAM (or already being read): use it
                        int32_t best = -1;
                        if (s.route_mode == 2) {
                            uint64_t best_epoch = 0;
                            for (int32_t c = 0; c < (int32_t) t0.n_expert; c++) {
                                const node_t & nd2 = s.nodes[(size_t) (t0.node0 + c)];
                                if (!taken[(size_t) c] && nd2.resident && !nd2.loading && (best < 0 || (uint64_t) nd2.epoch > best_epoch)) {
                                    best = c;
                                    best_epoch = (uint64_t) nd2.epoch;
                                }
                            }
                        }
                        if (e < (int32_t) used.size() && used[(size_t) e] > 0) used[(size_t) e]--;
                        eff[(size_t) (t * n_used + u)] = best;
                        if (best >= 0) {
                            taken[(size_t) best] = 1;
                            if (best < (int32_t) used.size()) used[(size_t) best]++;
                            s.route_substituted++;
                        } else {
                            s.route_skipped++;
                        }
                    }
                }
                s.route_ids[ids] = std::move(eff);
            }
            exec_jobs += s.prepare_group(tidx, used, exec_graph);
            if (s.lookahead && n_tok == 1) {
                const ggml_tensor * xin = node->src[1];
                if (xin && xin->type == GGML_TYPE_F32 && xin->nb[0] == sizeof(float) && xin->data) {
                    std::vector<float> xv((size_t) xin->ne[0]);
                    memcpy(xv.data(), xin->data, xv.size() * sizeof(float));
                    const int layer = s.tensors[tidx[0]]->layer;
                    if (layer >= 0) {
                        s.pf_submit([&s, layer, xv = std::move(xv), n_used = (int) n_used] { s.predict_and_prefetch(layer, xv, n_used); });
                    }
                }
            }
            if (!s.profile_path.empty() && ++s.groups_since_save >= 100000) {
                s.groups_since_save = 0;
                s.save_profile();
            }
            s.ns_prep += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - p0).count();
        }
    }

    // everything is resident and the weights are bandwidth-bound (K-quants, legacy types): llama.cpp's own path is as fast and has less overhead
    if (exec_graph && exec_jobs == 0 && !exec_heavy && !s.exec_force) {
        exec_graph = false;
        if (CPU().set_n_threads) {
            CPU().set_n_threads(c->cpu, c->n_threads);
        }
        // no reads were queued for this graph, but an earlier graph may still be filling these experts in the background (gate/up/down loaded together
        // when they come in separate calls, e.g. gpt-oss): llama.cpp's own path must not read them before they are complete
        s.stepaside_graphs++;
        const int n_all = ggml_graph_n_nodes(cg);
        for (int i = 0; i < n_all; i++) {
            const ggml_tensor * nd = ggml_graph_node(cg, i);
            if (nd->op == GGML_OP_MUL_MAT_ID && s.by_tensor.find(nd->src[0]) != s.by_tensor.end()) {
#ifndef MOE_CACHE_TEST_MUTANT_NO_STEPASIDE_WAIT     // CI only: a build without this wait must fail the delayed-read gpt-oss step (proves the step covers it)
                if (exec_wait_loaded(s, nd)) {
                    s.stepaside_waits++;
                }
#endif
            }
        }
    }
    if (exec_graph) {
        const auto e0 = std::chrono::steady_clock::now();
        const enum ggml_status est = exec_run_graph(c, s, cg);
        s.ns_cpu += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - e0).count();
        return est;
    }

    // long prompts: run the expert block on the GPU (what llama.cpp's own offload does for batches of 32+ tokens)
    std::vector<std::pair<ggml_tensor *, ggml_op>> silenced;
    if (s.cache_on && s.gpu_dev && !c->gpu_failed) {
        for (const auto & gr : groups_found) {
            if (gr.n_tok < s.gpu_min_tokens) {
                continue;
            }
            std::vector<ggml_tensor *> seg;
            c->seg_nomem = false;
            bool ok = gpu_segment(c, s, cg, gr.first, gr.last, seg);
            if (!ok && c->seg_nomem && c->hb_buf) {
                hybrid_drop(c, s);         // the hybrid bank holds the memory the long-prompt block needs: free it now, load it again at the next decode step
                seg.clear();
                ok = gpu_segment(c, s, cg, gr.first, gr.last, seg);
            }
            if (ok) {
                for (ggml_tensor * n : seg) {
                    silenced.emplace_back(n, n->op);
                    n->op = GGML_OP_NONE;
                }
            } else {
                s.gpu_fallbacks++;
            }
        }
    }

    // repacked experts: tag the weight as living in llama.cpp's repack buffer type while the kernel runs, so it picks the repacked code path
    std::vector<std::pair<ggml_tensor *, ggml_backend_buffer_t>> tagged;
    if (s.cache_on && s.repack_tag) {
        const int n_all = ggml_graph_n_nodes(cg);
        for (int i = 0; i < n_all; i++) {
            ggml_tensor * node = ggml_graph_node(cg, i);
            if (node->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            auto it = s.by_tensor.find(node->src[0]);
            if (it != s.by_tensor.end() && s.tensors[it->second]->repack && node->src[0]->buffer != s.repack_tag) {
                tagged.emplace_back(node->src[0], node->src[0]->buffer);
                node->src[0]->buffer = s.repack_tag;
            }
        }
    }
    if (s.cache_on) {
        cpu_path_check(s, cg);
    }
    const auto c0 = std::chrono::steady_clock::now();
#ifdef _WIN32
    g_cpu_graph = cg;
#endif
    const enum ggml_status st = ggml_backend_graph_compute(c->cpu, cg);
#ifdef _WIN32
    g_cpu_graph = nullptr;
#endif
    for (auto & tg : tagged) {
        tg.first->buffer = tg.second;
    }
    for (auto & sl : silenced) {
        sl.first->op = sl.second;
    }
    s.ns_cpu += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - c0).count();

    return st;
}

const ggml_backend_i g_backend_i = {
    /* .get_name           = */ be_name,
    /* .free               = */ be_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .set_tensor_2d_async= */ nullptr,
    /* .get_tensor_2d_async= */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ be_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

void set_n_threads(ggml_backend_t backend, int n) {
    ((moe_cache_backend_ctx *) backend->context)->n_threads = n;
}

}  // namespace
