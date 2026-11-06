// PREFILL-638: with the GPU keeper on (MOE_CACHE_GPU_KEEP_MIB > 0) a long-prompt block uploads whole tensors, so the keeper can hold them between chunks
// (src/mc_pack_rule.h). No ggml, no GPU.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/pack_rule_test.cpp -o /tmp/pack_rule_test && /tmp/pack_rule_test
#include "mc_pack_rule.h"
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    CHECK(mcpackrule::pack_block(true, 441, 512, false));     // the real case: 441 of 512 used, no keeper: packed, as before
    CHECK(!mcpackrule::pack_block(true, 441, 512, true));     // keeper on: whole tensors, so the held copy is the whole layer
    CHECK(!mcpackrule::pack_block(true, 512, 512, false));    // every expert used: one whole copy anyway
    CHECK(!mcpackrule::pack_block(true, 512, 512, true));
    CHECK(!mcpackrule::pack_block(false, 441, 512, false));   // a block that cannot be packed stays whole
    CHECK(!mcpackrule::pack_block(false, 441, 512, true));
    CHECK(mcpackrule::pack_block(true, 1, 256, false));       // a short prompt (agent turn) still uploads only what it uses
    if (fails) { fprintf(stderr, "%d checks failed\n", fails); return 1; }
    printf("pack-rule checks passed\n");
    return 0;
}
