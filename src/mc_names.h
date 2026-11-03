// Which ggml tensors are expert weights the plugin caches (QUALITY-027). Header-only, no ggml: tests/names_test.cpp includes it.
// llama.cpp names every stacked expert tensor blk.N.ffn_{gate,up,down,gate_up}_exps.weight (shared experts are "shexp" and 2-D). Before, any 3-D tensor
// whose name began with "blk." was registered as experts, so a model with another 3-D per-layer tensor would have had it evicted slice by slice.
#pragma once
#include <cstdint>
#include <cstring>

namespace mcname {

inline bool is_expert_tensor(const char * name, int64_t ne2) {
    return ne2 > 1 && name && strncmp(name, "blk.", 4) == 0 && strstr(name, "_exps") != nullptr;
}

}  // namespace mcname
