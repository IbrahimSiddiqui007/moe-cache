// moe-cache plugin part: state_t (continued, included inside the class body by mc_state.inl): background expert reads, lookahead prefetch, the stats snapshot, the reader pool.
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
#elif defined(_WIN32)
        // no sysfs: the launcher passes the efficiency cores as a mask (GetLogicalProcessorInformationEx, lowest efficiency class; processor group 0)
        if (la_ecores) {
            if (const char * m = getenv("MOE_CACHE_ECORE_MASK")) {
                DWORD_PTR set = 0;
                for (int c : mask_cpus(m)) {
                    if (c < 64) {
                        set |= (DWORD_PTR) 1 << c;
                    }
                }
                if (set) {
                    SetThreadAffinityMask(GetCurrentThread(), set);
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
            if (lt == layer_tensors.end() || lt->second.empty()) {     // empty: the layer's model was unloaded (forget_range)
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
                        while (v >= 0 && in_use(nodes[v])) {
                            v = nodes[v].prev;
                        }
                        if (v < 0) {
                            break;
                        }
                        evict(v);
                    }
                    if (resident_bytes + ti.nb2 > budget || !commit_for(ti, e)) {
                        pf_skipped++;
                        continue;
                    }
                    node_t & n = nodes[s];
                    n.resident = true;
                    __atomic_store_n(&n.loading, true, __ATOMIC_RELEASE);      // the executor reads this flag without the lock
                    n.prefetched = true;
                    n.epoch = epoch;
                    resident_bytes += ti.nb2;
                    if (ti.view) { view_bytes += ti.nb2; }
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
                 ",\"exec_mismatch\":" + std::to_string(exec_mismatch.load()) + ",\"idle_releases\":" + std::to_string(idle_releases) +
                 ",\"pool_parks\":" + std::to_string(pool ? pool->parks.load() : 0) + ",\"async_in\":" + std::to_string(async_in.load());
            j += ",\"layer_hits\":[";
            for (size_t i = 0; i < layer_hits.size(); i++) { j += (i ? "," : "") + std::to_string(layer_hits[i]); }
            j += "],\"layer_misses\":[";
            for (size_t i = 0; i < layer_misses.size(); i++) { j += (i ? "," : "") + std::to_string(layer_misses[i]); }
            j += "]}";
        }
        j = "moe-cache-stats " + j + "\n";
#ifdef _WIN32
        // one write, so llama-server's own log lines (another C runtime, same file) cannot land inside the JSON (seen in the Windows CI)
        DWORD wr = 0;
        fflush(stderr);
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), j.data(), (DWORD) j.size(), &wr, nullptr);
#else
        fputs(j.c_str(), stderr);
        fflush(stderr);
#endif
    }

    // normal teardown (the threads are alive: Linux exit, a Windows FreeLibrary without exit): stop and join them, then save and report
    ~state_t() {
        stop_threads();
        finish();
    }

    void stop_threads() {
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
        // the signals are ignored before their pipes close: the control page sends SIGUSR1 every 2 s until the process is gone, and a handler
        // writing into a closed (and reused) descriptor put a stray byte into the profile file finish() opens next (audit 2026-10-29 N7)
        struct sigaction ign;
        memset(&ign, 0, sizeof(ign));
        ign.sa_handler = SIG_IGN;
        if (g_stats_pipe[1] >= 0) {
            sigaction(SIGUSR1, &ign, nullptr);
            const int fd = g_stats_pipe[1];
            g_stats_pipe[1] = -1;
            close(fd);                       // wakes the snapshot thread, which then exits (joined below: it uses this state)
        }
        if (g_idle_pipe[1] >= 0) {
            sigaction(SIGUSR2, &ign, nullptr);
            const int fd = g_idle_pipe[1];
            g_idle_pipe[1] = -1;
            close(fd);                       // wakes the idle thread, which then exits (joined below: it uses this state)
        }
        if (stats_th.joinable()) {
            stats_th.join();
        }
#elif defined(_WIN32)
        g_stats_stop = true;
        if (g_stats_event) {
            SetEvent(g_stats_event);         // wakes the snapshot thread, which then exits
        }
        if (g_save_event) {
            SetEvent(g_save_event);          // wakes the save thread, which then exits
        }
        if (g_idle_event) {
            SetEvent(g_idle_event);          // wakes the idle thread, which then exits
        }
        // joined, not detached: neither may run on after this state (and its mutex) is gone; then the events can be closed
        if (stats_th.joinable()) {
            stats_th.join();
        }
        if (save_th.joinable()) {
            save_th.join();
        }
        if (g_stats_event) {
            CloseHandle(g_stats_event);
            g_stats_event = nullptr;
        }
        if (g_save_event) {
            CloseHandle(g_save_event);
            g_save_event = nullptr;
        }
#endif
        if (idle_th.joinable()) {
            idle_th.join();
        }
#ifdef _WIN32
        if (g_idle_event) {
            CloseHandle(g_idle_event);
            g_idle_event = nullptr;
        }
#endif
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
    }

    // the profile save and the summary lines (no thread is needed for either)
    void finish() {
        if (trace_f) {
            fclose(trace_f);
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
            if (fewer > 0) {
                fprintf(stderr, "moe-cache: approximate: fewer experts (MOE_CACHE_FEWER=%d): %llu of the lowest-ranked picks on executor layers dropped\n", fewer, (unsigned long long) fewer_dropped);
            }
            if (small_mapped || small_over) {
                fprintf(stderr, "moe-cache: cache smaller than one op: %llu experts mapped from the model file, %llu loaded over the cache size\n",
                        (unsigned long long) small_mapped, (unsigned long long) small_over);
            }
            if (commit_evictions) {
                fprintf(stderr, "moe-cache: Windows commit was short (refused, or below the headroom kept for others): %llu experts given back early, cache ended at %.2f GiB\n",
                        (unsigned long long) commit_evictions, budget / 1073741824.0);
            }
            if (cpu_check_reloads) {
                fprintf(stderr, "moe-cache: internal: %llu experts were loaded again right before llama.cpp's kernels (please report)\n", (unsigned long long) cpu_check_reloads);
            }
            fprintf(stderr, "moe-cache: reads straight into place %llu, via bounce buffer %llu, repacked %llu\n", (unsigned long long) direct_jobs, (unsigned long long) bounce_jobs, (unsigned long long) repack_jobs);
            if (async_in) {
                fprintf(stderr, "moe-cache: inputs read from the GPU asynchronously: %llu\n", (unsigned long long) async_in);
            }
            fprintf(stderr, "moe-cache: time prep=%.2fs (of which reads %.2fs) cpu_compute=%.2fs total_in_backend=%.2fs\n", ns_prep / 1e9, ns_read / 1e9, ns_cpu / 1e9, ns_total / 1e9);
            fprintf(stderr, "moe-cache: graphs=%llu groups=%llu ops=%llu hits=%llu misses=%llu (hit %.1f%%) read=%.1f MB evictions=%llu resident=%.2f GiB%s\n",
                    (unsigned long long) graphs, (unsigned long long) groups, (unsigned long long) ops, (unsigned long long) h, (unsigned long long) m,
                    h + m ? 100.0 * h / (h + m) : 0.0, bytes / 1e6, (unsigned long long) evictions, resident_bytes / (double) (1u << 30),
                    auto_mode ? (std::string(" | auto: final budget ") + std::to_string(budget / 1073741824.0).substr(0, 5) + " GiB, shrinks " + std::to_string(tune_shrinks) + ", grows " + std::to_string(tune_grows)).c_str() : "");
        }
    }

    // process exit on Windows: ExitProcess has already ended every other thread before the DLL detach runs this, so a lock may be held by a dead
    // thread and a condition variable may wait for dead waiters (llama-perplexity hung at exit forever, PC 2026-10-28, code-audit row 32). No
    // lock, notify or join: save the profile when nothing holds `mu`, report, and leave the state allocated (its threads and mutexes cannot be
    // destroyed safely). Linux never takes this path: its exit runs the destructor while the threads are alive
    void exit_without_threads() {
        if (mu.try_lock()) {
            mu.unlock();
            finish();
        } else if (trace_f) {
            fclose(trace_f);
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
        for (size_t v = 0; v < views.size(); v++) {
            if (j.dst >= views[v] && j.dst < views[v] + view_sizes[v]) {
                plat::file_view_load(j.dst, j.nb2);     // Windows file-view cache: bring the expert's file pages into the working set
                direct_jobs++;
                bytes += j.nb2;
                return;
            }
        }
        const size_t aoff = j.off / PAGE * PAGE;
        const size_t shift = j.off - aoff;
        const size_t len = round_up(shift + j.nb2);
        // Straight into place (direct) only the whole pages inside the expert are read. Its first and last partial pages are shared with the
        // neighbouring experts, which may be in use right now: they go through the bounce buffer and only this expert's bytes are copied.
        // (Before, the aligned read covered the shared pages too. The bytes it wrote there were the same, but on Windows a page under an
        // unbuffered read briefly holds other data: the executor read garbage in the last rows of a neighbouring expert, NaN router scores,
        // repeated expert ids, then the crash at request 2. PC 2026-10-27, Qwen3.5-122B from a fast NVMe drive; MOE_CACHE_DIRECT=0 fixed it.)
        char * const lo = j.dst, * const hi = j.dst + j.nb2;
        char * const in0 = (char *) round_up((uintptr_t) lo), * const in1 = (char *) ((uintptr_t) hi / PAGE * PAGE);
#ifndef MOE_CACHE_TEST_MUTANT_DIRECT_SHARED   // CI only: the old aligned read over the shared pages must fail the poisoned-reads step
        const bool direct = j.direct && !j.rtensor && in1 > in0;
#else
        const bool direct = false;
        if (j.direct && !j.rtensor) {
            if (test_poison) {
                memset(lo - shift, 0xff, len);
                std::this_thread::sleep_for(std::chrono::microseconds(test_read_delay_us));
            }
            plat::mem_commit(lo - shift, len + PAGE);
            const plat::file_t & f0 = fds[j.fdi];
            for (size_t done = 0; done < shift + j.nb2;) {
                const ptrdiff_t got = plat::file_pread(f0, lo - shift + done, len - done, (uint64_t) (aoff + done));
                if (got <= 0) { abort(); }
                done += (size_t) got;
            }
            direct_jobs++;
            bytes += len;
            return;
        }
#endif
        // the pages must exist before they are written (Windows); one extra page covers kernels that read a little past the end
        if (!plat::mem_commit(j.dst, j.nb2 + PAGE)) {
            commit_fatal("an expert read (its pages were committed when it was queued, then given back)", j.nb2);
        }
        if (test_poison) {      // test switch: what this read writes holds garbage for a while (only this expert's bytes, which nobody reads yet)
            memset(lo, 0xff, j.nb2);
            std::this_thread::sleep_for(std::chrono::microseconds(test_read_delay_us));
        }
        // each piece: a read with retries; if it keeps failing, one more try through a normal buffered read; then stop (never serve wrong data)
        const bool inject_job = fault_every > 0 && (++fault_ctr % (uint64_t) fault_every) == 0;   // test switch: this read fails every direct attempt
        auto read_all = [&](const plat::file_t & rfd, bool inject, char * buf, size_t want, size_t need, uint64_t foff) {
            size_t done = 0;
            int retries = 0;
            while (done < need) {
                ptrdiff_t got;
                if (inject && inject_job) {
                    got = -1;
                    errno = EIO;
                } else {
                    got = plat::file_pread(rfd, buf + done, want - done, foff + done);
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
        // want: aligned length asked for; need: bytes that must arrive (the end of the file can cut the last aligned page short)
        auto read_piece = [&](char * buf, size_t want, size_t need, uint64_t foff) {
            if (read_all(fds[j.fdi], true, buf, want, need, foff)) {
                return;
            }
            plat::file_t bfd;
            {
                std::lock_guard<std::mutex> lk(fds_buf_mu);
                if (!fds_buf[j.fdi].ok()) {
                    fds_buf[j.fdi] = plat::file_open(shard_paths[j.fdi].c_str(), false);
                }
                bfd = fds_buf[j.fdi];
            }
            if (!bfd.ok() || !read_all(bfd, false, buf, need, need, foff)) {
                fprintf(stderr, "moe-cache: read failed at offset %llu of %s: %s (disk error, or the model file changed after loading)\n",
                        (unsigned long long) foff, shard_paths[j.fdi].c_str(), plat::last_error().c_str());
                abort();
            }
            if (read_fallbacks++ == 0) {
                fprintf(stderr, "moe-cache: direct read failed, used a buffered read instead (output stays correct; this may be slower)\n");
            }
        };
        if (direct) {
            if (in0 > lo) {                 // first partial page: aligned read into the bounce buffer, copy this expert's part
                read_piece(bounce, PAGE, PAGE, aoff);
                memcpy(lo, bounce + shift, (size_t) (in0 - lo));
            }
            read_piece(in0, (size_t) (in1 - in0), (size_t) (in1 - in0), (uint64_t) (j.off + (size_t) (in0 - lo)));
            if (hi > in1) {                 // last partial page
                read_piece(bounce, PAGE, (size_t) (hi - in1), (uint64_t) (j.off + (size_t) (in1 - lo)));
                memcpy(in1, bounce, (size_t) (hi - in1));
            }
            direct_jobs++;
        } else {
            read_piece(bounce, len, shift + j.nb2, aoff);
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
            } else {
                memcpy(j.dst, bounce + shift, j.nb2);
                bounce_jobs++;
            }
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

