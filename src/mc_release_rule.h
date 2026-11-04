// When the long-prompt block's GPU buffer is given back (PATCH-086). Header-only, no ggml: tests/release_rule_test.cpp includes it.
#pragma once
namespace mcrelease {

// enabled: MOE_CACHE_GPU_FREE_AFTER_PROMPT; blocks_ran: a block ran since the last release; any_long: this graph has a group long enough for a block
inline bool should_release(bool enabled, bool blocks_ran, bool any_long) {
    return enabled && blocks_ran && !any_long;
}

}  // namespace mcrelease
