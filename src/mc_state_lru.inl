// moe-cache plugin part: state_t (continued, included inside the class body by mc_state.inl): LRU over resident experts, eviction, prepare_group, read flushing.
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
            if (ti.view) {
                plat::file_view_drop((void *) a, b - a);   // the pages leave the working set and stay in the file cache
            } else {
                plat::mem_decommit((void *) a, b - a);
            }
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
        if (ti.view) { view_bytes -= ti.nb2; }
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

    // Windows: commit the pages the read of expert e will write (the same range read_job() writes), before the expert is marked resident.
    // When Windows refuses (RAM + page file full: error 1455), evict least-recently-used experts this step does not need and try again, and lower
    // the budget to what fits. Returns false when nothing is left to evict; the caller must then not load the expert. Called with mu held.
    // (Before this, a refused commit was only retried page by page and the read went ahead: "read failed ... error 998", PC 2026-10-27.)
    bool commit_for(const tinfo_t & ti, int64_t e) {
        if (!plat::is_windows || ti.view) {     // a file-view expert uses no commit
            return true;
        }
        char * const p = ti.data + (size_t) e * ti.nb2;      // the range read_job() writes: the expert's own bytes, plus one page
        const size_t n = ti.nb2 + PAGE;
        bool evicted = false;
        // the system as a whole is low on commit (llama.cpp or other programs took it): give an LRU expert back instead of growing
        const size_t low = plat::commit_keep() / 2;
        while (plat::commit_available() < low + n) {
            int v = lru;
            while (v >= 0 && in_use(nodes[v])) {
                v = nodes[v].prev;
            }
            if (v < 0) {
                break;       // nothing to give back: load anyway (this step needs it), the commit below still has to succeed
            }
            evict(v);
            evicted = true;
            commit_evictions++;
        }
        for (;;) {
            if (plat::mem_commit(p, n)) {
                if (evicted) {
                    budget = std::min(budget, std::max(min_budget, resident_bytes + ti.nb2));   // stay at what Windows lets us commit
                }
                return true;
            }
            int v = lru;
            while (v >= 0 && in_use(nodes[v])) {
                v = nodes[v].prev;
            }
            if (v < 0) {
                return false;
            }
            if (!commit_fail_warned) {
                commit_fail_warned = true;
                fprintf(stderr, "moe-cache: warning: Windows refused memory for an expert (error %lu%s): the cache gives experts back and shrinks; "
                        "a larger page file avoids this\n", plat::commit_error(), plat::commit_error() == 1455 ? ": RAM + page file are full" : "");
            }
            evict(v);
            evicted = true;
            commit_evictions++;
        }
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
                        for (int k = 0; k < 100000 && v >= 0 && in_use(nodes[v]); k++) {
                            v = nodes[v].prev;
                        }
                        if (v < 0 || in_use(nodes[v])) {
                            break;     // no room without evicting what this group needs: leave it to the sibling's own call
                        }
                        evict(v);
                    }
                    if (resident_bytes + ti.nb2 > budget || !commit_for(ti, e)) {
                        continue;
                    }
                    n.resident = true;
                    n.epoch = epoch;
                    resident_bytes += ti.nb2;
                    if (ti.view) { view_bytes += ti.nb2; }
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
                    for (int k = 0; k < 100000 && v >= 0 && in_use(nodes[v]); k++) {
                        v = nodes[v].prev;
                    }
                    if (v < 0 || in_use(nodes[v])) {
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
                if (!commit_for(ti, e)) {
                    commit_fatal("an expert this step needs (every other expert is in use)", ti.nb2);
                }
                n.resident = true;
                n.epoch = epoch;
                resident_bytes += ti.nb2;
                if (ti.view) { view_bytes += ti.nb2; }
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
