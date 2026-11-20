// TEST-1170: src/mc_test_skip.h (MOE_CACHE_TEST_SKIP_PCT): parse, determinism, share.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/test_skip_test.cpp -o /tmp/test_skip_test && /tmp/test_skip_test
#include "mc_test_skip.h"
#include <cstdio>
#include <initializer_list>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    using namespace mctestskip;
    CHECK(parse_pct(nullptr) == -1 && parse_pct("") == -1 && parse_pct("abc") == -1 && parse_pct("-1") == -1 && parse_pct("101") == -1 && parse_pct("5x") == -1);
    CHECK(parse_pct("0") == 0 && parse_pct("30") == 30 && parse_pct("50") == 50 && parse_pct("100") == 100);
    CHECK(layer_of("blk.17.ffn_gate_exps.weight") == 17 && layer_of("blk.0.ffn_down_exps.weight") == 0 && layer_of("output.weight") == -1);
    // 0 skips nothing, 100 everything; the same (layer, expert) always gives the same answer
    for (int l = 0; l < 48; l++) for (int e = 0; e < 512; e++) {
        CHECK(!skip_pick(l, e, 0) && skip_pick(l, e, 100));
        CHECK(skip_pick(l, e, 50) == skip_pick(l, e, 50));
        if (skip_pick(l, e, 30)) CHECK(skip_pick(l, e, 50));      // nested: a larger share contains the smaller one
    }
    // the share is close to the request (48 layers x 512 experts)
    for (int pct : { 30, 50 }) {
        int n = 0;
        for (int l = 0; l < 48; l++) for (int e = 0; e < 512; e++) n += skip_pick(l, e, pct);
        const double f = 100.0 * n / (48 * 512);
        CHECK(f > pct - 2 && f < pct + 2);
    }
    if (fails == 0) printf("test_skip_test: ok\n");
    return fails ? 1 : 0;
}
