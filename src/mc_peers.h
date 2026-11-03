// moe-cache plugin: peer registry (MOE_CACHE_PEERS=DIR, N2/MM2, default off, PROTOTYPE: decision logic checked on fake peers only, tests/peers_test.cpp).
// Several llama-servers that load this plugin write one small file each into one folder and read the others' files, so they can share ONE memory cap at
// run time instead of a fixed split. Kept free of ggml so a test builds it alone.
// Rules (decide()): the room one server may use is the cap minus the other live servers' memory (and their margins). A server that has had no graph for a while
// gives memory back, a step at a time, down to its floor, while another server is busy and wants more. When two servers are busy and one holds more than its own
// share while the other holds less and wants more, the larger one steps back toward its share. Nothing else is changed: which experts are cached is still
// each server's own choice, and the output of the model is never touched.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace mcpeers {

struct rec {
    long     pid = 0;
    uint64_t beat_ms = 0;        // wall clock, ms since the epoch, of the last write
    uint64_t rss = 0;            // what this server holds now (anonymous memory + file-view experts), bytes
    uint64_t margin = 0;         // what it keeps free for bursts (prompt cache etc.), bytes
    uint64_t budget = 0;         // its expert cache budget now
    uint64_t share = 0;          // its own fair share of the cap (its --ram)
    uint64_t floor_b = 0;        // least budget it can work with
    int      active = 0;         // a graph ran in the last few seconds
    int      hungry = 0;         // active, cache full, and the model has more experts than the cache holds
};

inline std::string format(const rec & r) {
    char t[320];
    snprintf(t, sizeof(t), "moe-cache-peer 1 %ld %llu %llu %llu %llu %llu %llu %d %d\n", r.pid, (unsigned long long) r.beat_ms, (unsigned long long) r.rss,
             (unsigned long long) r.margin, (unsigned long long) r.budget, (unsigned long long) r.share, (unsigned long long) r.floor_b, r.active, r.hungry);
    return t;
}

// false for anything that is not a whole record (a half-written file, another program's file)
inline bool parse(const std::string & s, rec & r) {
    unsigned long long beat, rss, margin, budget, share, fl;
    long pid;
    int ver, act, hun;
    if (sscanf(s.c_str(), "moe-cache-peer %d %ld %llu %llu %llu %llu %llu %llu %d %d", &ver, &pid, &beat, &rss, &margin, &budget, &share, &fl, &act, &hun) != 10 || ver != 1) {
        return false;
    }
    r.pid = pid; r.beat_ms = beat; r.rss = rss; r.margin = margin; r.budget = budget; r.share = share; r.floor_b = fl; r.active = act; r.hungry = hun;
    return true;
}

struct verdict {
    uint64_t ceiling = 0;        // most this server's memory may be now: cap - own margin - the others' memory and margins (0: no room at all)
    uint64_t shrink_to = 0;      // 0: no forced step; else lower the cache budget to this (never below the floor)
    int      live = 0;           // other servers seen alive
};

inline verdict decide(const rec & me, const std::vector<rec> & others, uint64_t cap, uint64_t now_ms, uint64_t stale_ms, uint64_t step_bytes) {
    verdict v;
    uint64_t used = me.margin;
    bool hungry_other = false, hungry_below_share = false;
    for (const rec & o : others) {
        if (o.pid == me.pid || now_ms > o.beat_ms + stale_ms) {
            continue;                                    // myself, or a server that has stopped writing (crashed, or killed): its memory is gone
        }
        v.live++;
        used += o.rss + o.margin;
        if (o.active && o.hungry) {
            hungry_other = true;
            if (o.budget + step_bytes <= o.share) {
                hungry_below_share = true;
            }
        }
    }
    v.ceiling = cap > used ? cap - used : 0;
    if (!hungry_other) {
        return v;
    }
    if (!me.active) {
        if (me.budget > me.floor_b) {
            v.shrink_to = me.budget > me.floor_b + step_bytes ? me.budget - step_bytes : me.floor_b;
        }
    } else if (hungry_below_share && me.budget > me.share && me.budget > me.floor_b) {
        const uint64_t lo = me.share > me.floor_b ? me.share : me.floor_b;
        v.shrink_to = me.budget > lo + step_bytes ? me.budget - step_bytes : lo;
    }
    return v;
}

// AUD-020: who may write the registry. A folder or record of another user could forge a peer (a fake server that holds all the memory, so this one
// shrinks its cache). The folder must be ours and closed to others (0700), records are 0600, and records of another user are not read. Windows: the
// folder inherits its parent's access rules; nothing to do here.
#ifndef _WIN32
// "" when dir is a folder of this user (made 0700 if it was open to others), else why not
inline std::string secure_dir(const std::string & dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        return "not a folder";
    }
    if (st.st_uid != geteuid()) {
        return "the folder belongs to another user";
    }
    if ((st.st_mode & 077) != 0 && chmod(dir.c_str(), 0700) != 0) {
        return "cannot restrict the folder to its owner";
    }
    return "";
}
inline bool owned_by_me(const std::string & path) {
    struct stat st;
    return lstat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_uid == geteuid();
}
// create or replace a file readable by the owner only; never follows a symlink
inline bool write_private(const std::string & path, const std::string & text) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) {
        return false;
    }
    size_t done = 0;
    while (done < text.size()) {
        const ssize_t w = write(fd, text.data() + done, text.size() - done);
        if (w <= 0) {
            close(fd);
            return false;
        }
        done += (size_t) w;
    }
    return close(fd) == 0;
}
#endif

}  // namespace mcpeers
