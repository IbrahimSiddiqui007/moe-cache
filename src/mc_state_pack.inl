// moe-cache plugin part: state_t (continued, included inside the class body by mc_state.inl): the expert pack file (BO-1).
// bin/moe_cache_pack.py writes MODEL.moepack: for every (layer, expert) one 4 KiB-aligned block with that expert's slices of all its tensors (gate, up, down).
// A missing expert is then ONE large unbuffered read (the PC's SSD reads 2688 KiB at 3.47 GB/s but 896 KiB at 1.90 GB/s, RESULTS 73) followed by a copy of each
// slice to the place the per-tensor read would have written: the same bytes, so the output cannot change. Format: see the docstring of bin/moe_cache_pack.py.

    // ---- opening and checking the pack (constructor, after the model files are open) ----
    void pack_try_open() {
        // opt-in (2026-10-30): MOE_CACHE_PACK=1 (MODEL.moepack next to the model) or =PATH. A pack found next to the model is NOT used on its own:
        // in CI a leftover pack was picked up by a later step and that step's tokens differed from stock (under investigation), so nothing changes
        // for a user who did not ask for it
        const char * env = getenv("MOE_CACHE_PACK");
        if (shard_paths.empty() || !env || !*env || strcmp(env, "0") == 0) {
            return;
        }
        const bool named = env && *env && strcmp(env, "1") != 0 && strcmp(env, "auto") != 0;
        std::string path;
        if (named) {
            path = env;
        } else {
            path = shard_paths[0];
            if (path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0) {
                path.resize(path.size() - 5);
            }
            path += ".moepack";
        }
        FILE * f = plat::fopen_utf8(path.c_str(), "rb");
        if (!f) {
            if (named) {
                fprintf(stderr, "moe-cache: pack %s: cannot open it; reading experts from the model file as before\n", path.c_str());
            }
            return;
        }
        auto refuse = [&](const std::string & why) {
            fprintf(stderr, "moe-cache: pack %s not used: %s (reading experts from the model file; delete the pack or run moe_cache.py pack again)\n", path.c_str(), why.c_str());
            pack_layers.clear();
            pack_by_name.clear();
        };
        auto file_size_of = [](FILE * g) -> uint64_t {
            const long long cur = (long long) plat::ftell64(g);
            plat::fseek64(g, 0, SEEK_END);
            const long long end = (long long) plat::ftell64(g);
            plat::fseek64(g, cur, SEEK_SET);
            return end < 0 ? 0 : (uint64_t) end;
        };
        unsigned char h[256];
        if (fread(h, 1, sizeof(h), f) != sizeof(h) || memcmp(h, "MOEPACK1", 8) != 0) {
            fclose(f);
            refuse("not a pack file (or unfinished)");
            return;
        }
        auto rd32 = [&](size_t o) { uint32_t v; memcpy(&v, h + o, 4); return v; };
        auto rd64 = [&](const unsigned char * b, size_t o) { uint64_t v; memcpy(&v, b + o, 8); return v; };
        const uint32_t version = rd32(8), n_rec = rd32(12), n_files = rd32(16), header_bytes = rd32(20), crc_want = rd32(24);
        const uint64_t pack_size = file_size_of(f);
        if (version != 1 || n_files != shard_paths.size() || n_files > 8 || n_rec == 0 || n_rec > 100000 || header_bytes < 256 + 512ull * n_rec ||
            header_bytes > ((uint32_t) 64 << 20) || pack_size < header_bytes) {
            fclose(f);
            refuse("another format version, another number of model files, or a damaged header");
            return;
        }
        for (uint32_t i = 0; i < n_files; i++) {
            FILE * g = plat::fopen_utf8(shard_paths[i].c_str(), "rb");
            const uint64_t sz = g ? file_size_of(g) : 0;
            if (g) { fclose(g); }
            if (sz == 0 || sz != rd64(h, 32 + 8 * i)) {
                fclose(f);
                refuse("the model file " + shard_paths[i] + " has another size than the one the pack was made from");
                return;
            }
        }
        std::vector<unsigned char> rec((size_t) n_rec * 512);
        if (fread(rec.data(), 1, rec.size(), f) != rec.size()) {
            fclose(f);
            refuse("the layer table is cut short");
            return;
        }
        fclose(f);
        std::vector<pack_layer_t> layers;
        std::unordered_map<std::string, std::pair<int, int>> by_name;
        uint64_t last_end = header_bytes;
        for (uint32_t r = 0; r < n_rec; r++) {
            const unsigned char * b = rec.data() + (size_t) r * 512;
            pack_layer_t L;
            uint32_t layer, n_exp, nt;
            memcpy(&layer, b, 4); memcpy(&n_exp, b + 4, 4); memcpy(&nt, b + 8, 4);
            L.base = rd64(b, 16);
            L.stride = rd64(b, 24);
            L.n_expert = (int) n_exp;
            L.n = (int) nt;
            if (nt < 2 || nt > 4 || n_exp == 0 || n_exp > 65536 || L.base % PAGE || L.stride % PAGE || L.stride == 0 || L.base < last_end ||
                L.base + L.stride * n_exp > pack_size) {
                refuse("the layer table is inconsistent");
                return;
            }
            last_end = L.base + L.stride * n_exp;
            for (uint32_t k = 0; k < nt; k++) {
                const unsigned char * t = b + 32 + k * 120;
                char name[97];
                memcpy(name, t, 96);
                name[96] = 0;
                L.name[k] = name;
                L.nb2[k] = (size_t) rd64(t, 96);
                L.in_off[k] = (size_t) rd64(t, 104);
                const uint64_t src_off = rd64(t, 112);
                auto ft = file_tensors.find(L.name[k]);
                if (ft == file_tensors.end() || ft->second.first != src_off || ft->second.second != L.nb2[k] * n_exp || L.in_off[k] + L.nb2[k] > L.stride) {
                    refuse("tensor " + L.name[k] + " is not stored the same way in the model (another quantisation or file?)");
                    return;
                }
                by_name[L.name[k]] = { (int) layers.size(), (int) k };
            }
            layers.push_back(std::move(L));
        }
        // sample check: 16 pages per model file, the same offsets the packer used (a re-quantised model with the same tensor layout differs here)
        uint32_t crc = 0;
        char * pg = (char *) plat::aligned_alloc_page(PAGE);
        if (!pg) {
            refuse("out of memory");
            return;
        }
        for (size_t si = 0; si < fds.size(); si++) {
            const uint64_t size = rd64(h, 32 + 8 * si);
            const uint64_t last = size > PAGE ? (size - PAGE) / PAGE * PAGE : 0;
            for (uint64_t k = 1; k <= 16; k++) {
                const uint64_t off = std::min<uint64_t>(size * k / 17 / PAGE * PAGE, last);
                const size_t want = (size_t) std::min<uint64_t>(PAGE, size - off);
                size_t got = 0;
                int tries = 0;
                while (got < want) {
                    const ptrdiff_t r = plat::file_pread(fds[si], pg + got, PAGE - got, off + got);
                    if (r <= 0) {
                        if (r == 0 || ++tries > 3) { break; }
                        plat::sleep_ms(20);
                        continue;
                    }
                    got += (size_t) r;
                }
                if (got < want) {
                    plat::aligned_free_page(pg);
                    refuse("cannot read the model file to check the pack");
                    return;
                }
                crc = crc32_update(crc, pg, want);
            }
        }
        plat::aligned_free_page(pg);
        if (crc != crc_want) {
            refuse("sampled bytes of the model differ from the ones the pack was made from");
            return;
        }
        plat::file_t pf = plat::file_open(path.c_str(), !win_buffered, win_random);
#ifndef _WIN32
        if (!pf.ok() && !win_buffered && errno == EINVAL) {
            pf = plat::file_open(path.c_str(), false, win_random);       // a file system without O_DIRECT: the same aligned reads through the page cache
        }
#endif
        if (!pf.ok()) {
            refuse("cannot open it for unbuffered reads");
            return;
        }
        pack_fd = pf;
        pack_path = path;
        pack_layers = std::move(layers);
        pack_by_name = std::move(by_name);
        pack_on = true;
        fprintf(stderr, "moe-cache: expert pack %s: %zu layers, %.1f GiB; a missing expert is read with one unbuffered read "
                        "(file-view cache off while the pack is in use; MOE_CACHE_PACK=0 turns the pack off)\n", path.c_str(), pack_layers.size(), pack_size / 1073741824.0);
    }

    // a tensor was registered: is it in the pack with the shape the pack was made for?
    void pack_tag(tinfo_t & ti) {
        if (!pack_on) {
            return;
        }
        auto it = pack_by_name.find(ti.name);
        if (it == pack_by_name.end()) {
            return;
        }
        const pack_layer_t & L = pack_layers[(size_t) it->second.first];
        if (L.n_expert == ti.n_expert && L.nb2[(size_t) it->second.second] == ti.nb2 && !ti.repack && !ti.view) {
            ti.pack_rec = it->second.first;
            ti.pack_k = it->second.second;
        }
    }

    // the tensor that owns node s
    tinfo_t * node_tensor(int s) {
        size_t lo_t = 0, hi_t = tensors.size();
        if (hi_t == 0) {
            return nullptr;
        }
        while (lo_t + 1 < hi_t) {
            const size_t mid = (lo_t + hi_t) / 2;
            if (tensors[mid]->node0 <= s) { lo_t = mid; } else { hi_t = mid; }
        }
        return tensors[lo_t].get();
    }

    // Take the jobs of experts that have at least two slices to read out of `jobs` and make pack jobs of them (a lone slice is cheaper from the model file:
    // the pack read would carry bytes nobody wants). The caller owns the returned pack jobs.
    void pack_extract(std::vector<io_job_t> & jobs, std::vector<pack_job_t *> & out) {
        if (!pack_on || jobs.size() < 2) {
            return;
        }
        struct grp_t { int rec, e, n; int idx[4]; };
        std::unordered_map<uint64_t, size_t> where;
        std::vector<grp_t> groups;
        std::vector<char> taken(jobs.size(), 0);
        for (size_t i = 0; i < jobs.size(); i++) {
            const io_job_t & j = jobs[i];
            if (j.node < 0 || j.rtensor || j.pk) {
                continue;
            }
            const tinfo_t * ti = node_tensor(j.node);
            if (!ti || ti->pack_k < 0 || j.node < ti->node0) {
                continue;
            }
            const int64_t e = j.node - ti->node0;
            if (e >= ti->n_expert || j.nb2 != ti->nb2 || j.dst != ti->data + (size_t) e * ti->nb2) {
                continue;               // merged with a neighbour, or not this expert's own place
            }
            const uint64_t key = ((uint64_t) (uint32_t) ti->pack_rec << 32) | (uint32_t) e;
            auto it = where.find(key);
            if (it == where.end()) {
                it = where.emplace(key, groups.size()).first;
                groups.push_back({ ti->pack_rec, (int) e, 0, { -1, -1, -1, -1 } });
            }
            grp_t & g = groups[it->second];
            if (g.idx[ti->pack_k] >= 0) {
                continue;               // the same slice twice: leave the second one to the plain path
            }
            g.idx[ti->pack_k] = (int) i;
            g.n++;
        }
        for (const grp_t & g : groups) {
            if (g.n < 2) {
                continue;
            }
            const pack_layer_t & L = pack_layers[(size_t) g.rec];
            auto * pj = new pack_job_t();
            size_t lo = SIZE_MAX, hi = 0;
            for (int k = 0; k < 4; k++) {
                if (g.idx[k] < 0) {
                    continue;
                }
                pj->j[pj->n] = jobs[(size_t) g.idx[k]];
                pj->in_off[pj->n] = L.in_off[k];
                pj->n++;
                lo = std::min(lo, L.in_off[k]);
                hi = std::max(hi, L.in_off[k] + L.nb2[k]);
                taken[(size_t) g.idx[k]] = 1;
            }
            lo = lo / PAGE * PAGE;
            hi = round_up(hi);
            pj->lo = lo;
            pj->len = hi - lo;
            pj->off = L.base + (uint64_t) g.e * L.stride + lo;
            out.push_back(pj);
        }
        if (out.empty()) {
            return;
        }
        std::vector<io_job_t> rest;
        rest.reserve(jobs.size());
        for (size_t i = 0; i < jobs.size(); i++) {
            if (!taken[i]) { rest.push_back(jobs[i]); }
        }
        jobs.swap(rest);
        std::sort(out.begin(), out.end(), [](const pack_job_t * a, const pack_job_t * b) { return a->off < b->off; });
    }

    // one unbuffered read of the expert's block, then a copy of each wanted slice to its destination; if the read keeps failing, the per-tensor path takes over
    void read_pack_job(const pack_job_t & p) {
        static thread_local char * buf = nullptr;
        static thread_local size_t buf_size = 0;
        static thread_local char * fb = nullptr;
        static thread_local size_t fb_size = 0;
        if (buf_size < p.len) {
            plat::aligned_free_page(buf);
            buf = (char *) plat::aligned_alloc_page(p.len);
            if (!buf) {
                fprintf(stderr, "moe-cache: out of memory for a read buffer\n");
                abort();
            }
            buf_size = p.len;
        }
        size_t done = 0;
        int retries = 0;
        bool ok = true;
        while (done < p.len) {
            const ptrdiff_t got = plat::file_pread(pack_fd, buf + done, p.len - done, p.off + done);
            if (got < 0 || (got == 0)) {
                if (++retries <= 3) {
                    plat::sleep_ms(20);
                    continue;
                }
                ok = false;
                break;
            }
            done += (size_t) got;
        }
        if (!ok) {
            if (pack_fallbacks++ == 0) {
                fprintf(stderr, "moe-cache: a read from the pack failed (%s); using the model file for those experts (output stays correct; this may be slower)\n", plat::last_error().c_str());
            }
            const size_t need = round_up(max_nb2_all) + 2 * PAGE;
            if (fb_size < need) {
                plat::aligned_free_page(fb);
                fb = (char *) plat::aligned_alloc_page(need);
                if (!fb) {
                    abort();
                }
                fb_size = need;
            }
            for (int i = 0; i < p.n; i++) {
                read_job(p.j[i], fb);
            }
            return;
        }
        for (int i = 0; i < p.n; i++) {
            const io_job_t & j = p.j[i];
            if (!plat::mem_commit(j.dst, j.nb2 + PAGE)) {
                commit_fatal("an expert read from the pack (its pages were committed when it was queued, then given back)", j.nb2);
            }
#ifndef MOE_CACHE_TEST_MUTANT_PACK_OFFSET     // CI only: slices copied from a wrong offset inside the block must fail the pack identity step
            memcpy(j.dst, buf + (p.in_off[i] - p.lo), j.nb2);
#else
            memcpy(j.dst, buf + (p.in_off[i] - p.lo >= 64 ? p.in_off[i] - p.lo - 64 : p.in_off[i] - p.lo), j.nb2);
#endif
        }
        pack_reads++;
        pack_slices += (uint64_t) p.n;
        pack_bytes += p.len;
        bytes += p.len;
    }
