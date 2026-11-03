// moe-cache plugin: which layers the decode tier (MOE_CACHE_DECODE_TIER, experimental, default off) keeps on the GPU, kept free of ggml so
// tests/tier_test.cpp can check it without a model. docs/DECODE_TIER_DESIGN.md says what the tier is for.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <vector>

// "38,39" -> {38, 39}: layer numbers in the order given; junk, negative numbers and duplicates are dropped; at most 32
inline std::vector<int> tier_parse(const char * s) {
    std::vector<int> v;
    while (s && *s && v.size() < 32) {
        char * end = nullptr;
        const long n = strtol(s, &end, 10);
        if (end == s) {
            s++;                       // not a number: skip one character
            continue;
        }
        if (n >= 0 && n < 100000 && std::find(v.begin(), v.end(), (int) n) == v.end()) {
            v.push_back((int) n);
        }
        s = end;
    }
    return v;
}

// The wanted layers, in the order given, that fit together in free_bytes - keep_bytes. layer_bytes: bytes of a whole layer's experts (a layer
// that does not exist in the map is skipped). The first layer that does not fit ends the list, so the order of the wish is the order of value.
inline std::vector<int> tier_fit(const std::vector<int> & want, const std::map<int, size_t> & layer_bytes, size_t free_bytes, size_t keep_bytes) {
    std::vector<int> out;
    const size_t room = free_bytes > keep_bytes ? free_bytes - keep_bytes : 0;
    size_t used = 0;
    for (int l : want) {
        const auto it = layer_bytes.find(l);
        if (it == layer_bytes.end() || it->second == 0) {
            continue;
        }
        if (used + it->second > room) {
            break;
        }
        used += it->second;
        out.push_back(l);
    }
    return out;
}
