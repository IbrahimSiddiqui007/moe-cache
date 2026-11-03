// PATCH-086: the block buffer is released once, after the prompt, only when the switch is on (src/mc_release_rule.h). No ggml, no GPU.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/release_rule_test.cpp -o /tmp/release_rule_test && /tmp/release_rule_test
#include "mc_release_rule.h"
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    CHECK(!mcrelease::should_release(false, true, false));    // switch off (default): never
    CHECK(!mcrelease::should_release(true, false, false));    // no block ran: nothing to give back
    CHECK(!mcrelease::should_release(true, true, true));      // the prompt goes on: keep it for the next micro-batch
    CHECK(mcrelease::should_release(true, true, false));      // prompt over, decoding: release
    // a prompt of 3 micro-batches then 5 decode graphs: released exactly once (the caller clears blocks_ran)
    bool ran = false;
    int released = 0;
    const bool long_graph[] = { true, true, true, false, false, false, false, false };
    for (bool lg : long_graph) {
        if (mcrelease::should_release(true, ran, lg)) { released++; ran = false; }
        if (lg) { ran = true; }
    }
    CHECK(released == 1);
    if (fails) { fprintf(stderr, "%d checks failed\n", fails); return 1; }
    printf("release-rule checks passed\n");
    return 0;
}
