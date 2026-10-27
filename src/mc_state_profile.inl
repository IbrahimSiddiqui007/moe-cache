// moe-cache plugin part: state_t (continued, included inside the class body by mc_state.inl): tensor registration, usage profile save / warm start, the memory tuner.
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
        // every tensor of a layer is indexed with the same expert ids (lookahead, warm start, siblings check them against the layer's first
        // tensor only): a layer whose expert tensors disagree on the expert count, or an absurd count, is left to plain memory
        {
            int layer = -1;
            sscanf(t->name, "blk.%d.", &layer);
            const auto lt = layer >= 0 ? layer_tensors.find(layer) : layer_tensors.end();
            if (t->ne[2] > 65536 || (lt != layer_tensors.end() && !lt->second.empty() && tensors[(size_t) lt->second[0]]->n_expert != t->ne[2])) {
                if (unmatched++ == 0) {
                    fprintf(stderr, "moe-cache: tensor %s has %lld experts, other expert tensors of its layer another count: it is loaded normally and not cached\n",
                            t->name, (long long) t->ne[2]);
                }
                return -1;
            }
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

    // true when the profile file was written
    bool save_profile() {
        if (profile_path.empty() || session.empty()) {
            return false;
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
            return false;
        }
        const int32_t h[2] = { nl, ne };
        bool ok = fwrite("XUSE", 1, 4, f) == 4;
        ok = ok && fwrite(h, sizeof(int32_t), 2, f) == 2;
        ok = ok && fwrite(out.data(), sizeof(float), out.size(), f) == out.size();
        ok = ok && fwrite(&fingerprint, 8, 1, f) == 1;
        ok = (fclose(f) == 0) && ok;
        if (!ok) {
            fprintf(stderr, "moe-cache: writing the profile failed (disk full?); the old profile %s is kept\n", profile_path.c_str());
            plat::remove_file(tmp.c_str());
            return false;
        }
        return plat::replace_file(tmp.c_str(), profile_path.c_str());
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
        // Windows: every loaded expert is committed memory; near the commit limit (RAM + page file) a commit fails and the next write crashes.
        // Keep 1 GiB of commit free: shrink by what is missing (on Linux commit_available() is unlimited)
        const size_t commit_free = plat::commit_available(), commit_keep = (size_t) 1 << 30;
        const size_t commit_short = commit_free < commit_keep ? commit_keep - commit_free : 0;
        if (commit_short && !commit_warned) {
            commit_warned = true;
            fprintf(stderr, "moe-cache: warning: the system is close to its commit limit (%.2f GiB left of RAM + page file): the cache shrinks to "
                    "stay below it; a larger page file avoids this\n", commit_free / 1073741824.0);
        }
        if (rss > target || commit_short) {
            // over the target: the new budget is what is loaded now minus the excess, so eviction starts at once
            const size_t over = std::max(rss > target ? rss - target : 0, commit_short) + ((size_t) 64 << 20);
            size_t nb = have > over ? have - over : 0;
            nb = std::min(std::max(nb, min_budget), budget);
            budget_peak = std::max(budget_peak, budget);
            if (nb < budget) {
                budget = nb;
                tune_shrinks++;
                if (!shrink_warned && budget < budget_peak / 2) {
                    // desktop PC 2026-10-26: the cache fell from 14 to 2 GiB without a word; the user only saw a slow server. (The ~15 GB that
                    // llama-server took there was llama.cpp's model-file pages left in its working set, I37, now released at the first graph)
                    shrink_warned = true;
                    const double other = (double) (rss > have ? rss - have : 0) / 1073741824.0;
                    fprintf(stderr, "moe-cache: warning: the expert cache shrank from %.1f to %.1f GiB because the rest of llama-server now uses %.1f GiB "
                            "(limit %.1f GiB): expect many SSD reads. A smaller context, a smaller --cache-ram or closing other programs "
                            "leaves it more room\n", budget_peak / 1073741824.0, budget / 1073741824.0, other, limit_bytes / 1073741824.0);
                }
            }
            while (resident_bytes > budget) {
                int v = lru;
                while (v >= 0 && in_use(nodes[v])) {
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

