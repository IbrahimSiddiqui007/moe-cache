// CACHE-623: a long prompt (scan) must not refresh the recency of already-resident sibling experts (src/mc_lru_rule.h). No ggml, no model: a small LRU list
// driven the way the sibling path in mc_state_lru.inl drives the real one.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/lru_rule_test.cpp -o /tmp/lru_rule_test && /tmp/lru_rule_test
#include "mc_lru_rule.h"
#include <algorithm>
#include <cstdio>
#include <list>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

// front = least recently used (evicted first), back = most recently used
static void sibling_touch(std::list<int> & lru, int e, bool scan) {
    if (mclru::refresh_recency_of_resident(scan)) {
        lru.remove(e);
        lru.push_back(e);
    }
}

int main() {
    CHECK(mclru::refresh_recency_of_resident(false) && !mclru::refresh_recency_of_resident(true));
    // decode left 1,2,3 resident with 3 the most recent; a 34k prompt then touches every expert 0..5 through the sibling path
    for (bool scan : { false, true }) {
        std::list<int> lru = { 0, 4, 5, 1, 2, 3 };       // 0, 4, 5 are one-use experts, 1, 2, 3 are the ones decode needs
        for (int e : { 0, 1, 2, 3, 4, 5 }) {
            sibling_touch(lru, e, scan);
        }
        if (scan) {
            CHECK((lru == std::list<int>{ 0, 4, 5, 1, 2, 3 }));          // order kept: one-use experts still go first
        } else {
            CHECK((lru == std::list<int>{ 0, 1, 2, 3, 4, 5 }));          // a normal call refreshes (decode, short prompts)
        }
    }

    // INKLING-1250 fix 2: the decode-token clock counts real tokens, not looped-layer re-entries
    {
        // visible = layers the plugin sees (managed experts); each token runs them ascending, loop layers `reps` times
        auto run = [](std::vector<int> visible, int ls, int le, int reps, int tokens) {
            int tok = 0, last = -1, pass = 0;
            for (int t = 0; t < tokens; t++) {
                int passes_done = 0;
                (void) passes_done;
                // emit the token's layer sequence: ascending, the loop range repeated reps times
                std::vector<int> seq;
                bool in_loop_emitted = false;
                for (size_t i = 0; i < visible.size(); i++) {
                    int l = visible[i];
                    bool in_loop = ls >= 0 && l >= ls && l < le;
                    if (!in_loop) { seq.push_back(l); continue; }
                    if (in_loop_emitted) { continue; }
                    in_loop_emitted = true;
                    for (int r = 0; r < reps; r++) {
                        for (int v : visible) { if (v >= ls && v < le) { seq.push_back(v); } }
                    }
                    // layers after the loop range follow after the last pass
                    for (int v : visible) { if (v >= le) { seq.push_back(v); } }
                    break;
                }
                for (int l : seq) {
                    if (mclru::starts_new_decode_token(l, last, ls, le, reps, pass)) { tok++; }
                    last = l;
                }
            }
            return tok;
        };
        std::vector<int> all24, all_to_end, loop_only, split;
        for (int l = 0; l < 24; l++) { all24.push_back(l); if (l >= 6) { all_to_end.push_back(l); } if (l >= 6 && l < 18) { loop_only.push_back(l); } }
        for (int l : { 0, 1, 2, 10, 11, 12, 20, 21, 22, 23 }) { split.push_back(l); }
        CHECK(run(all24, -1, -1, 1, 10) == 9);            // non-loop model: one wrap per token (the first token has no predecessor)
        CHECK(run(all24, 6, 18, 3, 10) == 9);             // Inkling 2 Mini layout (6-17 x3), every layer visible
        CHECK(run(all24, 6, 24, 3, 10) == 9);             // loop ends at the last layer
        CHECK(run(all_to_end, 6, 24, 3, 10) == 9);        // loop-only to the end: the 3rd wrap is the boundary
        CHECK(run(loop_only, 6, 18, 3, 10) == 9);         // only loop layers visible (user -ot keeps the rest on the GPU): was a frozen clock
        CHECK(run(split, 10, 13, 3, 10) == 9);            // graph split: layers 10-12 loop, 0-2 and 20-23 visible
        int pass = 0;
        CHECK(!mclru::starts_new_decode_token(0, -1, 6, 18, 3, pass));     // first layer after a reset: no predecessor, nothing counted
        CHECK(mclru::starts_new_decode_token(0, 23, -1, -1, 1, pass));     // last -> 0 is a new token
        CHECK(!mclru::starts_new_decode_token(18, 17, 6, 18, 3, pass));    // ascending
    }
    if (fails) {
        fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    printf("lru-rule checks passed\n");
    return 0;
}
