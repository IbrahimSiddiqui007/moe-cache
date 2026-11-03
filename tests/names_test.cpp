// QUALITY-027: only stacked expert tensors are registered as experts (src/mc_names.h). No ggml, no model.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/names_test.cpp -o /tmp/names_test && /tmp/names_test
#include "mc_names.h"
#include <cstdio>
#include <initializer_list>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    // every expert tensor name llama.cpp writes (gpt-oss, Qwen3, Mixtral, DeepSeek, merged gate+up)
    for (const char * n : { "blk.0.ffn_gate_exps.weight", "blk.12.ffn_up_exps.weight", "blk.35.ffn_down_exps.weight", "blk.3.ffn_gate_up_exps.weight" }) {
        CHECK(mcname::is_expert_tensor(n, 128));
    }
    CHECK(!mcname::is_expert_tensor("blk.0.ffn_gate_exps.weight", 1));       // one expert: nothing to slice
    CHECK(!mcname::is_expert_tensor("blk.0.ffn_gate_shexp.weight", 4));      // shared expert (2-D in practice)
    CHECK(!mcname::is_expert_tensor("blk.0.some_other_3d.weight", 8));       // another 3-D per-layer tensor is not an expert
    CHECK(!mcname::is_expert_tensor("token_embd.weight", 8) && !mcname::is_expert_tensor("output.weight", 3));
    CHECK(!mcname::is_expert_tensor("rope_freqs_exps", 8) && !mcname::is_expert_tensor(nullptr, 8));   // must start with blk.
    if (fails) { fprintf(stderr, "%d checks failed\n", fails); return 1; }
    printf("names checks passed\n");
    return 0;
}
