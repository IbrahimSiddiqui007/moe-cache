// TEST-1170: MOE_CACHE_TEST_SKIP_PCT=N (0-100), a TIMING-ONLY switch. Header-only, no ggml: tests/test_skip_test.cpp includes it.
//
// What it does: the executor does not compute N % of the expert picks it would run on the CPU. A pick is one (token, slot) of one expert op
// (gate, up or down); the share is chosen by a fixed hash of (layer, expert), so the SAME experts are skipped in every run, in every arm, for gate,
// up and down alike. A skipped pick gets a zeroed output row (the output buffer is cleared, nothing is read) and costs no weight reads and no dot products.
// What stays: routing, the cache bookkeeping, the reads and loads of every expert (skipped or not), the activation quantisation, the picks the
// GPU bank computes (hybrid) and everything outside the executor (attention, router, prefill batches handed to llama.cpp or the GPU).
// The output is NOT valid text. The result gives the decode speed with that share of the CPU expert work removed, i.e. the best any GPU expert bank
// could reach if it took over N % of the CPU picks for free.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace mctestskip {

// -1 = off (unset, empty, or not a number 0-100: a typo must not silently run a skip), 0..100 = percent. "0" is ON with no skipping (same code path, for arm A).
inline int parse_pct(const char * v) {
    if (!v || !*v) {
        return -1;
    }
    char * end = nullptr;
    const long n = strtol(v, &end, 10);
    if (end == v || *end != '\0' || n < 0 || n > 100) {
        return -1;
    }
    return (int) n;
}

// layer index from a tensor name "blk.<N>.ffn_..."; -1 when there is none
inline int layer_of(const char * name) {
    const char * p = strstr(name, "blk.");
    return p ? atoi(p + 4) : -1;
}

// splitmix64 over (layer, expert): fixed, platform independent
inline bool skip_pick(int layer, int expert, int pct) {
    uint64_t x = ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert;
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    return (int) (x % 100) < pct;
}

}  // namespace mctestskip
