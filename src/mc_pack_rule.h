// Whether the long-prompt GPU block uploads only the experts it uses (packed) or whole expert tensors (PREFILL-638). Header-only, no ggml: tests/pack_rule_test.cpp.
#pragma once
#include <cstddef>
namespace mcpackrule {

// ok: the block can be packed at all; n_used / n_as: experts the block's tokens use / experts in the tensor;
// keeper_on: MOE_CACHE_GPU_KEEP_MIB > 0. The keeper holds whole tensors only (a packed copy changes with every chunk), so with it on the block uploads whole
// tensors: with 441 of 512 experts used per chunk, every block was packed and the keeper never held a layer (layer-wise prefill, PC 2026-11-02:
// uploads stayed at 652 GB). Without the keeper: pack unless every expert is used (the whole upload would read the same bytes in one copy).
inline bool pack_block(bool ok, size_t n_used, size_t n_as, bool keeper_on) {
    return ok && !keeper_on && n_used != n_as;
}

}  // namespace mcpackrule
