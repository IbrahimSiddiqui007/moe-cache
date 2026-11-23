// CACHE-623: a long prompt (scan) must not refresh the recency of already-resident sibling experts (src/mc_lru_rule.h). No ggml, no model: a small LRU list
// driven the way the sibling path in mc_state_lru.inl drives the real one.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/lru_rule_test.cpp -o /tmp/lru_rule_test && /tmp/lru_rule_test
#include "mc_lru_rule.h"
#include <algorithm>
#include <cstdio>
#include <list>

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

    // INKLING-1250 fix2: the decode-token clock counts real tokens, not looped-layer re-entries
    {
        auto run = [](int n_layer, int ls, int le, int reps, int tokens) {
            int tok = 0, last = -1;
            for (int t = 0; t < tokens; t++) {
                for (int l = 0; l < n_layer; l++) {
                    int passes = (ls >= 0 && l >= ls && l < le) ? reps : 1;
                    for (int r = 0; r < passes; r++) {
                        if (mclru::starts_new_decode_token(l, last, ls, le)) { tok++; }
                        last = l;
                    }
                }
            }
            return tok;
        };
        CHECK(run(24, -1, -1, 1, 10) == 9);   // non-loop model: one wrap per token (first token has no predecessor)
        CHECK(run(24, 6, 18, 3, 10) == 9);    // Inkling 2 Mini layout (6-17 x3): same count as without the loop
        CHECK(mclru::starts_new_decode_token(0, 23, -1, -1));          // last -> 0 is a new token
        CHECK(!mclru::starts_new_decode_token(18, 17, 6, 18));         // 17 -> 18 is ascending
        CHECK(!mclru::starts_new_decode_token(6, 17, 6, 18));          // loop re-entry 17 -> 6 is not a new token
        CHECK(mclru::starts_new_decode_token(0, 17, 6, 18));           // 17 -> 0 (loop at the end of the model) is
        CHECK(mclru::starts_new_decode_token(5, 17, 6, 18));           // dropping below the loop start is a new token
    }
    if (fails) {
        fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    printf("lru-rule checks passed\n");
    return 0;
}
