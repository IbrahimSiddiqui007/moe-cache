// moe-cache plugin part: overflow-safe size checks for the layer table of a .moepack file (AUD-019). The numbers come from the file, so a damaged or
// hostile one must not wrap a 64-bit sum around into a value that passes. Plain C++, tested by tests/pack_check_test.cpp.
#pragma once
#include <cstdint>

namespace mcpack {

// a * b <= limit, without overflow
inline bool product_le(uint64_t a, uint64_t b, uint64_t limit) {
    return a == 0 || b <= limit / a;
}

// the layer's block [base, base + stride * n_exp) lies inside a pack of pack_size bytes
inline bool layer_fits(uint64_t base, uint64_t stride, uint64_t n_exp, uint64_t pack_size) {
    return base <= pack_size && product_le(stride, n_exp, pack_size - base);
}

// a tensor's slice [in_off, in_off + nb2) lies inside one expert stride
inline bool slice_fits(uint64_t in_off, uint64_t nb2, uint64_t stride) {
    return in_off <= stride && nb2 <= stride - in_off;
}

}   // namespace mcpack
