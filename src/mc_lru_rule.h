// The recency rule for an expert that is already resident when a call touches it (CACHE-623). Header-only, no ggml: tests/lru_rule_test.cpp includes it.
// A long prompt (scan) uses nearly every expert, so it must not make one "recently used": the normal path (hits in mc_state_lru.inl) skips touch_use for a
// scan, and a scan's new loads go to the LRU end (push_lru). The sibling path (gpt-oss: gate/up/down arrive in separate calls) forgot this and moved
// every already-resident sibling to the MRU end, so after a 34k prompt the order was flat and decode's experts were evicted as often as one-use ones.
// Output never depends on the order (only which expert is evicted first), so this changes speed, not text.
#pragma once
namespace mclru {

inline bool refresh_recency_of_resident(bool scan) {
    return !scan;
}

// INKLING-1250 fix 2: does a decode graph at layer `dl`, right after layer `last`, start the next token? Layers ascend within a token, so a lower layer normally does.
// A loop model (arch inkling2_loop: layers loop_start..loop_end-1 run loop_count times per token with shared weights) re-enters the range at loop_start from inside it:
// that is another pass of the same token. Which wrap is the real token boundary depends on which layers the plugin sees, so the wraps that land inside the loop range
// are counted (loop_pass): the loop_count-th one is the boundary; a wrap that leaves the loop range (a layer before loop_start, or after a layer past loop_end) is a
// boundary at once and restarts the count. loop_start < 0 = not a loop model (same rule as before: any lower layer is a new token).
inline bool starts_new_decode_token(int dl, int last, int loop_start, int loop_end, int loop_count, int & loop_pass) {
    if (dl >= last) {
        return false;
    }
    if (loop_start >= 0 && loop_count > 1 && dl >= loop_start && last < loop_end) {
        if (++loop_pass < loop_count) {
            return false;
        }
    }
    loop_pass = 0;
    return true;
}

}  // namespace mclru
