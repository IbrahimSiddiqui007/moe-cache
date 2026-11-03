// Checks src/mc_poolsize.h (DECODE-137).
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/poolsize_test.cpp -o /tmp/poolsize_test && /tmp/poolsize_test
#include "mc_poolsize.h"
#include <cstdio>
#include <initializer_list>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    using namespace mcpool;
    // no pool yet: build from the graph's count, switch or not, prompt or decode (stock behaviour)
    for (bool on : { false, true }) {
        for (long long tok : { 1ll, 512ll }) {
            const decision_t d = decide(on, false, 0, false, 0, 12, tok);
            CHECK(d.act == action_t::build && d.threads == 12);
        }
    }
    // MOE_CACHE_EXEC_THREADS wins when building
    CHECK(decide(true, false, 0, false, 0 + 8, 12, 1).threads == 8);
    // switch off: a pool built for -tb 12 stays, decode with -t 6 changes nothing
    CHECK(decide(false, true, 12, false, 0, 6, 1).act == action_t::keep);
    // switch on: first decode graph with a different count rebuilds once, to the decode count
    decision_t d = decide(true, true, 12, false, 0, 6, 1);
    CHECK(d.act == action_t::rebuild && d.threads == 6);
    // after that (built from a decode graph) never again, however the counts move
    CHECK(decide(true, true, 6, true, 0, 12, 1).act == action_t::keep);
    CHECK(decide(true, true, 6, true, 0, 6, 1).act == action_t::keep);
    // a prompt graph never rebuilds
    CHECK(decide(true, true, 6, false, 0, 12, 512).act == action_t::keep);
    // same count (no explicit -tb): nothing to do, even on the first decode graph
    CHECK(decide(true, true, 6, false, 0, 6, 1).act == action_t::keep);
    // an explicit MOE_CACHE_EXEC_THREADS fixes the size
    CHECK(decide(true, true, 8, false, 8, 6, 1).act == action_t::keep);
    // odd inputs
    CHECK(decide(true, false, 0, false, 0, 0, 1).threads == 1);
    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("poolsize_test: ok\n");
    return 0;
}
