// moe-cache plugin part: state_t (continued, included inside the class body by mc_state.inl): idle rebalance (BO-2).
    // Agents pause between turns while their tools run. While no graph has run for MOE_CACHE_IDLE_REBALANCE_MS, a background thread reads the experts
    // with the highest decayed use count that are not resident into free room, or into the room of the resident experts with the lowest count
    // (rb_choose in mc_rebalance.h decides, with a margin so nothing swaps back and forth). Only which experts are resident changes, never what is
    // computed. One expert (its gate, up and down tensors) is made resident and read at a time; when a graph starts the thread stops after that
    // expert: a graph that needs the expert being read waits on its "loading" flag like for any lookahead read, and the experts the thread read are
    // ordinary cache entries. Called with no lock held unless said.

    // be_compute entry / exit (the RAII guard in mc_compute.inl): a graph is running, the rebalance must not start another expert
    void graph_enter() {
        std::lock_guard<std::mutex> lk(mu);
        graph_active++;
        graph_seq++;
        if (rb_acted) {                  // a round read something since the last graph: measure this turn's first prefill and decode graph
            rb_acted = false;
            rb_armed_dec = rb_armed_pre = true;
        }
        g_saw_dec = g_saw_pre = false;
    }

    void graph_leave() {
        {
            std::lock_guard<std::mutex> lk(mu);
            graph_active--;
            if (g_saw_dec) { rb_armed_dec = false; }
            if (g_saw_pre) { rb_armed_pre = false; }
        }
        last_end_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        rb_cv.notify_all();
    }

    // text for the summary line (caller does not hold mu: the counters are read as they are, it is a report)
    std::string rb_summary() const {
        auto pct = [](uint64_t h, uint64_t m) { char t[32]; if (h + m) { snprintf(t, sizeof(t), "%.1f%%", 100.0 * h / (h + m)); } else { snprintf(t, sizeof(t), "-"); } return std::string(t); };
        return " | idle rebalance: rounds " + std::to_string(rb_rounds) + ", reads " + std::to_string(rb_reads) + " (" + std::to_string(rb_bytes_read >> 20) + " MB), swaps " +
               std::to_string(rb_swaps) + ", cancelled " + std::to_string(rb_cancelled) + ", first prefill hit " + pct(rb_pre_hits, rb_pre_miss) + ", first decode hit " +
               pct(rb_dec_hits, rb_dec_miss);
    }

    double decayed_count(const node_t & n) const {
        const uint32_t dt = tok - n.crf_tok;
        return (double) n.crf * (dt < decay_tab.size() ? (double) decay_tab[dt] : 0.0);
    }

    void rb_main() {
        std::unique_lock<std::mutex> lk(mu);
        while (!rb_stop) {
            rb_cv.wait_for(lk, std::chrono::milliseconds(std::max(50, rb_idle_ms / 4)));
            if (rb_stop) {
                break;
            }
            if (graph_active > 0 || graph_seq == 0 || rb_round_seq == graph_seq || !cache_on || tensors.empty() || idle_released) {   // never refill a cache that idle release just gave back
                continue;
            }
            const uint64_t now = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            if (now - last_end_ns.load() < (uint64_t) rb_idle_ms * 1000000ull) {
                continue;
            }
            rb_round(lk);
        }
    }

    // one rebalance round; mu held on entry and on return (released around the reads)
    void rb_round(std::unique_lock<std::mutex> & lk) {
        const uint64_t seq = graph_seq;
        rb_round_seq = seq;
        rb_rounds++;
        struct ref_t { int layer, e; };
        std::vector<rb_unit> units;
        std::vector<ref_t> refs;
        for (const auto & kv : layer_tensors) {
            if (kv.second.empty()) {
                continue;
            }
            const int64_t ne = tensors[kv.second[0]]->n_expert;
            for (int64_t e = 0; e < ne; e++) {
                rb_unit u;
                u.layer = kv.first;
                u.e = (int) e;
                for (int ti_i : kv.second) {
                    const tinfo_t & ti = *tensors[ti_i];
                    if (ti.n_expert != ne) {
                        continue;
                    }
                    const node_t & n = nodes[(size_t) (ti.node0 + e)];
                    u.score = std::max(u.score, decayed_count(n));
                    if (n.resident) { u.res_bytes += ti.nb2; } else { u.miss_bytes += ti.nb2; }
                    u.locked = u.locked || (n.resident && in_use(n));
                }
                units.push_back(u);
                refs.push_back({ kv.first, (int) e });
            }
        }
        const size_t free_b = budget > resident_bytes ? budget - resident_bytes : 0;
        const std::vector<rb_pick> plan = rb_choose(units, free_b, rb_cap_bytes, rb_gain, rb_min_score);
        if (plan.empty()) {
            return;
        }
        std::vector<char> bounce;
        char * bb = nullptr;
        size_t done = 0;
        for (const rb_pick & p : plan) {
            // a graph started (or even ran) while the previous expert was read: stop, the plan was made for a cache that has changed
            if (rb_stop || graph_active > 0 || graph_seq != seq) {
                rb_cancelled++;
                break;
            }
            const ref_t c = refs[(size_t) p.cand];
            for (int vi : p.victims) {
                const ref_t v = refs[(size_t) vi];
                for (int ti_i : layer_tensors[v.layer]) {
                    const tinfo_t & ti = *tensors[ti_i];
                    if (ti.n_expert != tensors[layer_tensors[v.layer][0]]->n_expert) {
                        continue;
                    }
                    const int s = ti.node0 + v.e;
                    if (nodes[s].resident && !in_use(nodes[s])) {
                        rb_evicted_bytes += ti.nb2;
                        evict(s);
                    }
                }
                rb_swaps++;
            }
            std::vector<std::pair<io_job_t, int>> todo;
            bool refused = false;
            for (int ti_i : layer_tensors[c.layer]) {
                tinfo_t & ti = *tensors[ti_i];
                if (ti.n_expert != tensors[layer_tensors[c.layer][0]]->n_expert) {
                    continue;
                }
                const int s = ti.node0 + c.e;
                node_t & n = nodes[s];
                if (n.resident) {
                    continue;
                }
                if (resident_bytes + ti.nb2 > budget || !commit_for(ti, c.e)) {
                    refused = true;
                    break;
                }
                const double d = decayed_count(n);
                n.cls = (uint8_t) (classlru && d >= 1.0 ? std::min(nclass - 1, std::ilogb(d)) : 0);
                n.resident = true;
                __atomic_store_n(&n.loading, true, __ATOMIC_RELEASE);      // a graph that hits this expert waits for the read; nothing evicts it
                n.epoch = 0;
                n.last_tok = tok;
                resident_bytes += ti.nb2;
                if (ti.view) { view_bytes += ti.nb2; }
                push_mru(s);
                todo.push_back({ io_job_t{ ti.data + (size_t) c.e * ti.nb2, ti.file_off + (size_t) c.e * ti.nb2, ti.nb2, ti.fdi, ti.direct, ti.repack ? ti.tensor : nullptr }, s });
            }
            if (!todo.empty()) {
                lk.unlock();
                if (!bb) {
                    bounce.assign(round_up(max_nb2_all) + 3 * PAGE, 0);
                    bb = (char *) round_up((size_t) bounce.data());      // read_job wants a page-aligned buffer
                }
                for (auto & td : todo) {
                    read_job(td.first, bb);
                }
                lk.lock();
                for (auto & td : todo) {
                    __atomic_store_n(&nodes[(size_t) td.second].loading, false, __ATOMIC_RELEASE);
                    rb_reads++;
                    rb_bytes_read += td.first.nb2;
                }
                load_cv.notify_all();
                rb_acted = true;
                done++;
            }
            if (refused) {
                break;
            }
        }
        if (done) {
            fprintf(stderr, "moe-cache: idle rebalance: %zu experts read (%.0f MB total so far), %llu swaps\n", done, rb_bytes_read / 1e6, (unsigned long long) rb_swaps);
        }
    }
