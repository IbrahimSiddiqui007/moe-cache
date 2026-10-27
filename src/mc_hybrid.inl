// moe-cache plugin part: hybrid hit/miss (GPU bank + CPU).
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

// ---- hybrid (MOE_CACHE_HYBRID=MiB, experimental, not token-identical): GPU computes the picks whose expert is in its bank, the CPU the rest, at the same time ----
// The bank holds the most used experts of the usage profile (static for now). Nothing is uploaded per token: only the activation and the remapped ids go up,
// the bank's partial outputs come back and replace those picks' columns of the down projection's output.
void hybrid_upload(moe_cache_backend_ctx * c, state_t & s) {
    for (auto & h : c->hb) {
        const tinfo_t & ti = *s.tensors[(size_t) h.first];
        ggml_tensor * bank = h.second.first;
        for (size_t e = 0; e < h.second.second.size(); e++) {
            const int sl = h.second.second[e];
            if (sl < 0) {
                continue;
            }
            const char * src = ti.data + e * ti.nb2;
            gpu_cache_ensure(src, bank->nb[2]);
            ggml_backend_tensor_set(bank, src, (size_t) sl * bank->nb[2], bank->nb[2]);
        }
    }
}

void hybrid_drop(moe_cache_backend_ctx * c, state_t & s) {
    ggml_backend_buffer_free(c->hb_buf);
    c->hb_buf = nullptr;
    c->hb_dropped = true;
    for (auto & h : c->hb) {
        h.second.first->data = nullptr;
        h.second.first->buffer = nullptr;
    }
    if (s.hyb_drops++ == 0) {
        fprintf(stderr, "moe-cache: hybrid: bank freed so a long prompt can use the GPU; it is loaded again when decoding starts\n");
    }
}

bool hybrid_init(moe_cache_backend_ctx * c, state_t & s);

void hybrid_adapt_setup(moe_cache_backend_ctx * c, state_t & s) {
    c->hbl.clear();
    c->hb_first = -1;
    c->hb_tok = 0;
    if (const char * v = getenv("MOE_CACHE_HYBRID_ADAPT")) {
        c->ad_every = atoi(v);
        const char * k = strchr(v, ',');
        if (k) {
            c->ad_swaps = atoi(k + 1);
        }
    }
    if (c->ad_every <= 0 || c->ad_swaps <= 0) {
        c->ad_every = 0;
        return;
    }
    if (const char * v = getenv("MOE_CACHE_HYBRID_DUEL")) {
        c->duel_w = std::max(0, atoi(v));
    }
    c->duel.clear();
    c->ha_cur = c->hb_cur = 0;
    for (auto & h : c->hb) {
        const tinfo_t & ti = *s.tensors[(size_t) h.first];
        auto & L = c->hbl[ti.layer];
        L.tidx.push_back(h.first);
        if (L.slot_exp.empty()) {
            L.slot_exp.assign((size_t) h.second.first->ne[2], -1);
            for (size_t e = 0; e < h.second.second.size(); e++) {
                if (h.second.second[e] >= 0) {
                    L.slot_exp[(size_t) h.second.second[e]] = (int) e;
                }
            }
            L.score.assign(L.slot_exp.size(), 0.0f);
            L.miss.assign(h.second.second.size(), 0.0f);
            L.base.assign(h.second.second.size(), 0);
            for (size_t e = 0; e < h.second.second.size(); e++) {
                L.base[e] = h.second.second[e] >= 0;
            }
        }
    }
    c->hb_first = c->hbl.empty() ? -1 : c->hbl.begin()->first;
}

// upload expert e of this layer into bank slot sl (all its weight tensors) and update the slot maps
void hybrid_put(moe_cache_backend_ctx * c, state_t & s, moe_cache_backend_ctx::hb_layer_t & L, int sl, int e) {
    const int old = L.slot_exp[(size_t) sl];
    for (int t : L.tidx) {
        const tinfo_t & ti = *s.tensors[(size_t) t];
        auto & h = c->hb[t];
        const char * src = ti.data + (size_t) e * ti.nb2;
        gpu_cache_ensure(src, h.first->nb[2]);
        ggml_backend_tensor_set(h.first, src, (size_t) sl * h.first->nb[2], h.first->nb[2]);
        if (old >= 0) {
            h.second[(size_t) old] = -1;
        }
        h.second[(size_t) e] = sl;
    }
    L.slot_exp[(size_t) sl] = e;
}

// between two tokens (the GPU is idle): the most missed experts replace the coldest bank expert of their own layer when they were used more often
void hybrid_adapt(moe_cache_backend_ctx * c, state_t & s) {
    const auto t0 = std::chrono::steady_clock::now();
    const float dec = powf(0.97f, (float) c->ad_every);
    if (c->duel_w > 0) {
        int sa = 0, sb = 0;
        for (const auto & d : c->duel) {
            sa += d.first;
            sb += d.second;
        }
        if (sa < sb) {
            // the profile's bank would have served more: bring profile experts back in place of adapted ones (lowest score first), no new swaps
            int done = 0;
            for (auto & lv : c->hbl) {
                auto & L = lv.second;
                for (float & sc : L.score) {
                    sc *= dec;
                }
                std::fill(L.miss.begin(), L.miss.end(), 0.0f);
                if (done >= c->ad_swaps) {
                    continue;
                }
                std::vector<char> in(L.base.size(), 0);
                for (int e : L.slot_exp) {
                    if (e >= 0) {
                        in[(size_t) e] = 1;
                    }
                }
                std::vector<std::pair<double, int>> want;     // profile experts missing from the bank, most used first
                for (size_t e = 0; e < L.base.size(); e++) {
                    if (L.base[e] && !in[e]) {
                        want.emplace_back(-s.learned.at(lv.first, (int) e), (int) e);
                    }
                }
                std::sort(want.begin(), want.end());
                for (const auto & w : want) {
                    if (done >= c->ad_swaps) {
                        break;
                    }
                    int cold = -1;
                    for (size_t sl = 0; sl < L.slot_exp.size(); sl++) {
                        const int e = L.slot_exp[sl];
                        if (e >= 0 && !L.base[(size_t) e] && (cold < 0 || L.score[sl] < L.score[(size_t) cold])) {
                            cold = (int) sl;
                        }
                    }
                    if (cold < 0) {
                        break;
                    }
                    hybrid_put(c, s, L, cold, w.second);
                    L.score[(size_t) cold] = 0.0f;
                    done++;
                }
            }
            s.hyb_swaps += (uint64_t) done;
            s.hyb_reverts += (uint64_t) done;
            s.hyb_swap_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            return;
        }
    }
    std::vector<std::tuple<float, int, int>> cand;          // misses, layer, expert
    for (auto & lv : c->hbl) {
        auto & L = lv.second;
        for (float & sc : L.score) {
            sc *= dec;
        }
        for (size_t e = 0; e < L.miss.size(); e++) {
            if (L.miss[e] >= 2.0f) {
                cand.emplace_back(L.miss[e], lv.first, (int) e);
            }
        }
    }
    std::sort(cand.begin(), cand.end(), [](const auto & x, const auto & y) { return std::get<0>(x) > std::get<0>(y); });
    int done = 0;
    for (const auto & cd : cand) {
        if (done >= c->ad_swaps) {
            break;
        }
        auto & L = c->hbl[std::get<1>(cd)];
        const int e = std::get<2>(cd);
        int cold = -1;
        float best = 0.0f;
        for (size_t sl = 0; sl < L.slot_exp.size(); sl++) {
            const float v = L.score[sl] + 0.001f * (float) s.learned.at(std::get<1>(cd), L.slot_exp[sl]);
            if (cold < 0 || v < best) {
                cold = (int) sl;
                best = v;
            }
        }
        if (cold < 0 || L.score[(size_t) cold] >= std::get<0>(cd)) {
            continue;
        }
        hybrid_put(c, s, L, cold, e);
        L.score[(size_t) cold] = std::get<0>(cd);
        done++;
    }
    for (auto & lv : c->hbl) {
        std::fill(lv.second.miss.begin(), lv.second.miss.end(), 0.0f);
    }
    s.hyb_swaps += (uint64_t) done;
    s.hyb_swap_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}

// Same bank as before the long prompt: release the long-prompt block's GPU buffer first (it holds about one layer of experts), then upload the bank again.
bool hybrid_reload(moe_cache_backend_ctx * c, state_t & s) {
    const auto t0 = std::chrono::steady_clock::now();
    c->hb_dropped = false;
    ggml_gallocr_free(c->galloc);
    c->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->gpu));
    c->hb_buf = ggml_backend_alloc_ctx_tensors(c->hb_ctx, c->gpu);
    if (!c->hb_buf) {
        fprintf(stderr, "moe-cache: hybrid: the bank no longer fits after a long prompt: choosing a smaller one\n");
        c->hb_tried = false;
        return hybrid_init(c, s);
    }
    hybrid_upload(c, s);
    s.hyb_reloads++;
    s.hyb_reload_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool hybrid_init(moe_cache_backend_ctx * c, state_t & s) {
    if (c->hb_tried) {
        if (!c->hb_buf && c->hb_dropped) {
            return hybrid_reload(c, s);
        }
        return c->hb_buf != nullptr;
    }
    c->hb_tried = true;
    const char * env = getenv("MOE_CACHE_HYBRID");
    size_t mib = env ? (strcmp(env, "auto") == 0 ? (size_t) -1 : (size_t) atoll(env)) : 0;
    if (!mib || !s.gpu_dev || c->gpu_failed) {
        return false;
    }
    {
        // never more than the GPU has free, minus room for llama.cpp's own buffers that come later (auto = all of that)
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(s.gpu_dev, &fr, &tot);
        const size_t keep = (size_t) 600 << 20;
        const size_t avail = fr > keep ? (fr - keep) >> 20 : 0;
        if (mib > avail) {
            mib = avail;
        }
        if (mib < 256) {
            fprintf(stderr, "moe-cache: hybrid: only %zu MiB of GPU memory free, no bank\n", fr >> 20);
            return false;
        }
    }
    if (s.learned.n_exp == 0) {
        fprintf(stderr, "moe-cache: hybrid: no usage profile (run moe-cache-learn or use the model once), so no bank\n");
        return false;
    }
    if (!c->gpu) {
        c->gpu = ggml_backend_dev_init(s.gpu_dev, nullptr);
        if (!c->gpu) {
            c->gpu_failed = true;
            return false;
        }
        c->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->gpu));
    }
    std::map<int, size_t> layer_bytes;                        // all expert tensors of one layer, for one expert
    for (const auto & tp : s.tensors) {
        if (tp->layer >= 0 && !tp->repack && tp->tensor) {
            layer_bytes[tp->layer] += tp->nb2;
        }
    }
    std::vector<std::tuple<double, int, int>> cand;
    for (const auto & lb : layer_bytes) {
        for (int e = 0; e < s.learned.n_exp; e++) {
            const double n = s.learned.at(lb.first, e);
            if (n > 0) {
                cand.emplace_back(n, lb.first, e);
            }
        }
    }
    std::sort(cand.begin(), cand.end(), [](const auto & x, const auto & y) { return std::get<0>(x) > std::get<0>(y); });
    for (;; mib = mib * 85 / 100) {   // allocation failed: try again with 85 % until it fits or gets too small
    if (mib < 256) {
        fprintf(stderr, "moe-cache: hybrid: could not allocate a bank on the GPU\n");
        return false;
    }
    std::map<int, std::vector<int>> chosen;
    size_t used = 0, n_exp = 0;
    const size_t budget = mib << 20;
    for (const auto & cd : cand) {
        const size_t b = layer_bytes[std::get<1>(cd)];
        if (used + b > budget) {
            continue;
        }
        chosen[std::get<1>(cd)].push_back(std::get<2>(cd));
        used += b;
        n_exp++;
    }
    if (c->hb_ctx) {
        ggml_free(c->hb_ctx);
        c->hb.clear();
    }
    ggml_init_params ip = { ggml_tensor_overhead() * (s.tensors.size() + 1), nullptr, true };
    c->hb_ctx = ggml_init(ip);
    for (size_t i = 0; i < s.tensors.size(); i++) {
        const tinfo_t & ti = *s.tensors[i];
        auto it = chosen.find(ti.layer);
        if (it == chosen.end() || ti.repack || !ti.tensor) {
            continue;
        }
        const ggml_tensor * t = ti.tensor;
        ggml_tensor * bank = ggml_new_tensor_3d(c->hb_ctx, t->type, t->ne[0], t->ne[1], (int64_t) it->second.size());
        std::vector<int> slot((size_t) ti.n_expert, -1);
        for (size_t k = 0; k < it->second.size(); k++) {
            if (it->second[k] < ti.n_expert) {
                slot[(size_t) it->second[k]] = (int) k;
            }
        }
        c->hb[(int) i] = { bank, slot };
    }
    c->hb_buf = ggml_backend_alloc_ctx_tensors(c->hb_ctx, c->gpu);
    if (!c->hb_buf) {
        continue;
    }
    hybrid_upload(c, s);
    hybrid_adapt_setup(c, s);
    fprintf(stderr, "moe-cache: hybrid: GPU bank of %zu experts in %zu layers (%.0f MiB), chosen from the usage profile\n", n_exp, chosen.size(), used / 1048576.0);
    return true;
    }
}

// One expert block (gate, up, activation, down: MUL_MAT_ID / GLU nodes sharing the same ids) starting at node `first`. Returns false to leave it to the executor.
bool hybrid_group(moe_cache_backend_ctx * c, state_t & s, ggml_cgraph * cg, int first, int & last_out) {
    const int n_all = ggml_graph_n_nodes(cg);
    const ggml_tensor * ids = ggml_graph_node(cg, first)->src[2];
    std::vector<ggml_tensor *> seg;
    for (int j = first; j < n_all; j++) {
        ggml_tensor * n = ggml_graph_node(cg, j);
        if ((n->op == GGML_OP_MUL_MAT_ID && n->src[2] == ids) || n->op == GGML_OP_GLU) {
            seg.push_back(n);
        } else {
            break;
        }
    }
    while (!seg.empty() && seg.back()->op != GGML_OP_MUL_MAT_ID) {
        seg.pop_back();
    }
    if (seg.size() < 2 || !ggml_is_contiguous(seg.back()) || seg.back()->type != GGML_TYPE_F32) {
        return false;
    }
    const std::vector<int> * slot = nullptr;
    for (ggml_tensor * n : seg) {
        if (n->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        auto it = s.by_tensor.find(n->src[0]);
        if (it == s.by_tensor.end() || !c->hb.count(it->second)) {
            return false;
        }
        slot = &c->hb[it->second].second;
    }
    const int64_t n_used = ids->ne[0], n_tok = ids->ne[1];
    const int layer = s.tensors[(size_t) s.by_tensor.at(seg[0]->src[0])]->layer;
    moe_cache_backend_ctx::hb_layer_t * L = nullptr;
    if (c->ad_every > 0) {
        if (layer == c->hb_first) {
            if (c->duel_w > 0 && c->hb_tok > 0) {      // one token has passed: keep its real and shadow hits for the comparison
                c->duel.emplace_back(c->ha_cur, c->hb_cur);
                if ((int) c->duel.size() > c->duel_w) {
                    c->duel.pop_front();
                }
            }
            c->ha_cur = c->hb_cur = 0;
            if (++c->hb_tok % (uint64_t) c->ad_every == 0) {
                hybrid_adapt(c, s);
            }
        }
        auto lt = c->hbl.find(layer);
        L = lt != c->hbl.end() ? &lt->second : nullptr;
    }
    std::vector<char> hit((size_t) (n_used * n_tok), 0);
    std::vector<int32_t> rid((size_t) (n_used * n_tok), 0);
    int64_t n_hit = 0;
    for (int64_t t = 0; t < n_tok; t++) {
        for (int64_t u = 0; u < n_used; u++) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + u * ids->nb[0] + t * ids->nb[1]);
            const int sl = (e >= 0 && (size_t) e < slot->size()) ? (*slot)[(size_t) e] : -1;
            if (sl >= 0) {
                hit[(size_t) (t * n_used + u)] = 1;
                rid[(size_t) (t * n_used + u)] = sl;
                n_hit++;
                if (L) {
                    L->score[(size_t) sl] += 1.0f;
                    c->ha_cur++;
                }
            } else if (L && e >= 0 && (size_t) e < L->miss.size()) {
                L->miss[(size_t) e] += 1.0f;
            }
            if (L && e >= 0 && (size_t) e < L->base.size() && L->base[(size_t) e]) {
                c->hb_cur++;
            }
        }
    }
    if (n_hit == 0) {
        return false;
    }
    // the GPU graph: same nodes, weights replaced by the bank, ids by the slots (misses point at slot 0; their columns are not used)
    ggml_init_params ip = { ggml_tensor_overhead() * 4 * (seg.size() + 8) + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    std::unordered_map<const ggml_tensor *, ggml_tensor *> clone;
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> uploads;
    ggml_tensor * gids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_set_input(gids);
    for (ggml_tensor * n : seg) {
        ggml_tensor * nn = ggml_new_tensor(ctx, n->type, GGML_MAX_DIMS, n->ne);
        nn->op = n->op;
        memcpy(nn->op_params, n->op_params, sizeof(nn->op_params));
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            const ggml_tensor * src = n->src[k];
            if (!src) {
                continue;
            }
            if (clone.count(src)) {
                nn->src[k] = clone[src];
            } else if (n->op == GGML_OP_MUL_MAT_ID && k == 0) {
                nn->src[k] = c->hb[s.by_tensor.at(src)].first;
            } else if (n->op == GGML_OP_MUL_MAT_ID && k == 2) {
                nn->src[k] = gids;
            } else {
                // uploaded row by row below: whole rows of F32 / I32 only (the same rule as gpu_segment)
                if (src->nb[0] != ggml_type_size(src->type) || (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_I32)) {
                    ggml_free(ctx);
                    return false;
                }
                ggml_tensor * nt = ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
                ggml_set_input(nt);
                clone[src] = nt;
                uploads.emplace_back(src, nt);
                nn->src[k] = nt;
            }
        }
        clone[n] = nn;
    }
    ggml_tensor * gout = clone[seg.back()];
    ggml_set_output(gout);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, gout);
    if (!gpu_graph_supported(c, g) || !ggml_gallocr_alloc_graph(c->galloc, g)) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(gids, rid.data(), 0, rid.size() * sizeof(int32_t));
    std::vector<uint8_t> pack;
    for (auto & up : uploads) {
        const ggml_tensor * t = up.first;
        if (ggml_is_contiguous(t)) {
            ggml_backend_tensor_set(up.second, t->data, 0, ggml_nbytes(up.second));
            continue;
        }
        const size_t es = ggml_type_size(t->type);
        pack.assign(ggml_nbytes(up.second), 0);
        size_t o = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) for (int64_t i2 = 0; i2 < t->ne[2]; i2++) for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
            memcpy(pack.data() + o, (const char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], (size_t) t->ne[0] * es);
            o += (size_t) t->ne[0] * es;
        }
        ggml_backend_tensor_set(up.second, pack.data(), 0, pack.size());
    }
    if (ggml_backend_graph_compute_async(c->gpu, g) != GGML_STATUS_SUCCESS) {
        ggml_backend_synchronize(c->gpu);    // part of it may have been submitted: let it finish before the buffers are used again
        ggml_free(ctx);
        return false;
    }
    // meanwhile the CPU computes the misses (the activation node runs for all picks; the hit picks' values are replaced below)
    if (n_hit < n_used * n_tok) {
        for (int j = 0; j < (int) seg.size(); j++) {
            if (seg[(size_t) j]->op == GGML_OP_MUL_MAT_ID) {
                exec_one_op(s, seg[(size_t) j], &hit);
            } else {
                exec_run_nodes(c, cg, std::vector<int>{ first + j });
            }
        }
    }
    ggml_backend_synchronize(c->gpu);
    ggml_tensor * out = seg.back();
    std::vector<float> res((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(gout, res.data(), 0, ggml_nbytes(gout));
    for (int64_t t = 0; t < n_tok; t++) {
        for (int64_t u = 0; u < n_used; u++) {
            if (hit[(size_t) (t * n_used + u)]) {
                memcpy((char *) out->data + u * out->nb[1] + t * out->nb[2], res.data() + (size_t) ((t * n_used + u) * out->ne[0]), (size_t) out->ne[0] * sizeof(float));
            }
        }
    }
    ggml_free(ctx);
    s.hyb_groups++;
    s.hyb_picks += (uint64_t) (n_used * n_tok);
    s.hyb_hits += (uint64_t) n_hit;
    last_out = first + (int) seg.size() - 1;
    return true;
}

}  // namespace
