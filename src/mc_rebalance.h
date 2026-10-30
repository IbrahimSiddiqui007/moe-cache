// moe-cache plugin: the choice made by the idle rebalance (BO-2), kept free of ggml so tests/rebalance_test.cpp can check it without a model.
// A unit is one (layer, expert): all of its tensors (gate, up, down). score = decayed use count. The rebalance reads the best-scoring missing units
// into free room first, then into the room of the worst-scoring resident units, but only while a missing unit scores `gain` times more than
// the unit it replaces (that margin is the guard against swapping back and forth), and never more than cap_bytes in one idle period.
#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

struct rb_unit {
    int    layer = 0, e = 0;
    double score = 0.0;
    size_t miss_bytes = 0;    // bytes still to read (0: fully resident)
    size_t res_bytes = 0;     // bytes resident now (0: nothing resident)
    bool   locked = false;    // some tensor of the unit is in use (loading, or needed by the graph last run): never a victim
};

struct rb_pick {
    int              cand = -1;      // index into the units
    std::vector<int> victims;        // indices of units to evict completely before the read
};

inline std::vector<rb_pick> rb_choose(const std::vector<rb_unit> & u, size_t free_bytes, size_t cap_bytes, double gain, double min_score) {
    std::vector<int> cand, vict;
    for (size_t i = 0; i < u.size(); i++) {
        if (u[i].miss_bytes > 0 && u[i].score >= min_score) { cand.push_back((int) i); }
        if (u[i].res_bytes > 0 && !u[i].locked) { vict.push_back((int) i); }
    }
    std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return u[a].score > u[b].score; });
    std::stable_sort(vict.begin(), vict.end(), [&](int a, int b) { return u[a].score < u[b].score; });
    std::vector<rb_pick> out;
    size_t used = 0, next_v = 0;
    for (int c : cand) {
        const size_t need = u[c].miss_bytes;
        if (used + need > cap_bytes) { break; }
        rb_pick p;
        p.cand = c;
        size_t freed = 0, f = free_bytes;
        size_t nv = next_v;
        while (f + freed < need) {
            if (nv >= vict.size()) { return out; }                     // nothing left to give up
            const rb_unit & v = u[vict[nv]];
            if (v.score * gain + 0.25 > u[c].score) { return out; }    // every further candidate scores less, every further victim more
            p.victims.push_back(vict[nv]);
            freed += v.res_bytes;
            nv++;
        }
        free_bytes = f + freed - need;
        next_v = nv;
        used += need;
        out.push_back(std::move(p));
    }
    return out;
}
