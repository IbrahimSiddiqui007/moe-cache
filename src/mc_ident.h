// moe-cache plugin part: does a tensor's bytes in memory match the same tensor in MOE_CACHE_GGUF (AUD-017)? Plain C++, so tests/ident_test.cpp can drive
// it with a fake file. Experts are served from that file later, so a file that only agrees on the first bytes (another fine-tune or quantisation
// with the same layout and header) would change the output without a warning. Three samples per tensor: start, middle, end.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace mcident {

constexpr size_t kSample = 4096;

// rd(offset_in_tensor, buf, n) reads n bytes of the tensor from the file; returns false when it cannot
template <class Read>
bool same_samples(const void * data, size_t size, Read rd) {
    const size_t n = std::min(size, kSample);
    const size_t offs[3] = { 0, (size - n) / 2, size - n };
    char buf[kSample];
    for (size_t k = 0; k < 3; k++) {
        if (k > 0 && offs[k] == offs[k - 1]) {
            continue;                       // small tensor: the samples overlap completely
        }
        if (!rd(offs[k], buf, n) || memcmp(buf, (const char *) data + offs[k], n) != 0) {
            return false;
        }
    }
    return true;
}

}   // namespace mcident
