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

}  // namespace mclru
