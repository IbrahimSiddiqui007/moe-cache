// moe-cache plugin part: state_t (continued, included inside the class body by mc_state.inl): peer registry (MOE_CACHE_PEERS=DIR, N2/MM2, default off, PROTOTYPE).
    // Servers that load this plugin and name the same folder write one small file each (pid, memory held, cache budget, busy or quiet) once a second and read the
    // others'. The rules are in mc_peers.h (tests/peers_test.cpp). Here: the thread, the files, and the two places they act: tune() takes the room the others
    // leave as its target (peer_ceiling), and a quiet server gives its cache back a step at a time while another one is busy and wants more.
    // MOE_CACHE_PEER_CAP_GIB=N is the memory all of them share (the launcher passes its cap); the server's own --ram is its share. Only for auto mode.
    // Not run with two real servers yet. A stock llama-server next to them is not in the registry: the launcher takes its memory off the cap (--share-ram).
    bool   peers_on = false;
    std::string peer_dir;
    size_t peer_cap = 0;
    std::atomic<size_t> peer_ceiling{(size_t) -1};      // least of what tune() may use: (size_t) -1 while no verdict exists
    std::thread peer_th;
    std::condition_variable peer_cv;
    bool   peer_stop = false;
    int    peer_live_said = -1;
    uint64_t peer_said_ms = 0;

    static std::filesystem::path peer_fs(const std::string & p) {
#ifdef _WIN32
        return std::filesystem::path(plat::widen(p.c_str()));     // a std::string path is ANSI there
#else
        return std::filesystem::path(p);
#endif
    }

    static uint64_t wall_ms() {
        return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    // called once, when the auto-mode budget is set
    void peers_start() {
        const char * d = getenv("MOE_CACHE_PEERS");
        if (!d || !*d || !auto_mode) {
            return;
        }
        std::error_code ec;
        std::filesystem::create_directories(peer_fs(d), ec);
        if (ec) {
            fprintf(stderr, "moe-cache: MOE_CACHE_PEERS=%s: cannot make the folder (%s): the peer registry is off\n", d, ec.message().c_str());
            return;
        }
#ifndef _WIN32
        {
            const std::string why = mcpeers::secure_dir(d);
            if (!why.empty()) {
                fprintf(stderr, "moe-cache: MOE_CACHE_PEERS=%s: %s: the peer registry is off\n", d, why.c_str());
                return;
            }
        }
#endif
        peer_dir = d;
        peer_cap = ram_cap() != (size_t) -1 ? ram_cap() : plat::mem_total() / 8 * 7;
        if (const char * v = getenv("MOE_CACHE_PEER_CAP_GIB")) {
            size_t x = 0;
            if (parse_gib("MOE_CACHE_PEER_CAP_GIB", v, x) && x > 0) {
                peer_cap = x;
            }
        }
        peers_on = true;
        peer_th = std::thread([this] { peer_main(); });
        fprintf(stderr, "moe-cache: peer registry (PROTOTYPE, not run with two real servers) in %s: the servers there share %.1f GiB; this one's own share is %.1f GiB\n",
                d, peer_cap / 1073741824.0, peer_share() / 1073741824.0);
    }

    size_t peer_share() const {
        const size_t guess = (size_t) 7 << 29;
        return limit_bytes > margin + guess + ((size_t) 1 << 30) ? limit_bytes - margin - guess : (size_t) 1 << 30;
    }

    // mu held
    mcpeers::rec peer_self() {
        mcpeers::rec r;
        const uint64_t now_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        const uint64_t last = last_end_ns.load();
        r.pid = (long) plat::process_id();
        r.beat_ms = wall_ms();
        r.rss = read_rss_anon() + view_bytes;
        r.margin = margin;
        r.budget = budget;
        r.share = peer_share();
        r.floor_b = std::max<size_t>(min_budget, (size_t) 1 << 30);
        r.active = graph_active > 0 || (last && now_ns - last < 10ull * 1000000000ull) ? 1 : 0;
        r.hungry = r.active && max_budget > 0 && budget < max_budget && resident_bytes + ((size_t) 512 << 20) >= budget ? 1 : 0;
        return r;
    }

    std::string peer_file(long pid) const { return peer_dir + "/" + std::to_string(pid) + ".peer"; }

    void peer_write(const mcpeers::rec & r) {
        const std::string fin = peer_file(r.pid), tmp = peer_dir + "/" + std::to_string(r.pid) + ".tmp";
#ifndef _WIN32
        if (!mcpeers::write_private(tmp, mcpeers::format(r))) {      // 0600, no symlink followed (AUD-020)
            return;
        }
#else
        {
            std::ofstream f(peer_fs(tmp), std::ios::binary | std::ios::trunc);
            f << mcpeers::format(r);
        }
#endif
        std::error_code ec;
        std::filesystem::rename(peer_fs(tmp), peer_fs(fin), ec);      // replaces the old file in one step: a reader sees the old or the new record
    }

    std::vector<mcpeers::rec> peer_read_others(long self_pid, uint64_t now_ms) {
        std::vector<mcpeers::rec> out;
        std::error_code ec;
        for (std::filesystem::directory_iterator it(peer_fs(peer_dir), ec), end; !ec && it != end; it.increment(ec)) {
            if (it->path().extension() != ".peer") {
                continue;
            }
#ifndef _WIN32
            if (!mcpeers::owned_by_me(it->path().string())) {        // a record of another user (or a link) is not a peer of ours (AUD-020)
                continue;
            }
#endif
            std::ifstream f(it->path(), std::ios::binary);
            std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            mcpeers::rec r;
            if (!mcpeers::parse(s, r) || r.pid == self_pid) {
                continue;
            }
            if (now_ms > r.beat_ms + 60000) {          // a server that died: its file is cleared by the next one that sees it
                std::error_code e2;
                std::filesystem::remove(it->path(), e2);
                continue;
            }
            out.push_back(r);
        }
        return out;
    }

    void peer_main() {
        std::unique_lock<std::mutex> lk(mu);
        while (!peer_stop) {
            peer_cv.wait_for(lk, std::chrono::milliseconds(1000));
            if (peer_stop) {
                break;
            }
            const mcpeers::rec me = peer_self();
            lk.unlock();
            std::vector<mcpeers::rec> others;
            try {
                peer_write(me);
                others = peer_read_others(me.pid, me.beat_ms);
            } catch (const std::exception &) {
                // a full disk or a folder that went away: this second is skipped, the last ceiling stays
            }
            lk.lock();
            if (peer_stop) {
                break;
            }
            const mcpeers::verdict v = mcpeers::decide(me, others, peer_cap, me.beat_ms, 5000, (uint64_t) 1 << 30);
            peer_ceiling.store(v.ceiling);
            if (v.live != peer_live_said) {
                fprintf(stderr, "moe-cache: peer registry: %d other server(s) alive, this one may hold up to %.1f GiB of the %.1f GiB\n", v.live, v.ceiling / 1073741824.0, peer_cap / 1073741824.0);
                peer_live_said = v.live;
            }
            if (v.shrink_to && v.shrink_to < budget) {
                if (me.beat_ms > peer_said_ms + 10000) {
                    fprintf(stderr, "moe-cache: peer registry: another server is busy and wants memory: cache budget %.1f -> %.1f GiB (%s)\n", budget / 1073741824.0, v.shrink_to / 1073741824.0,
                            me.active ? "above its share" : "this one is quiet");
                    peer_said_ms = me.beat_ms;
                }
                budget = v.shrink_to;
                tune_shrinks++;
                while (resident_bytes > budget) {
                    const int vi = pick_victim();
                    if (vi < 0) {
                        break;
                    }
                    evict(vi);
                }
            }
        }
    }

    void peers_stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            peer_stop = true;
        }
        peer_cv.notify_all();
        if (peer_th.joinable()) {
            peer_th.join();
        }
        if (peers_on) {
            std::error_code ec;
            std::filesystem::remove(peer_fs(peer_file((long) plat::process_id())), ec);
        }
    }
