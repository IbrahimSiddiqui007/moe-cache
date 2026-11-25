// AUD-019: the pack layer table checks do not wrap around at 2^64.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/pack_check_test.cpp -o /tmp/pack_check_test && /tmp/pack_check_test
#include "mc_pack_check.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    const uint64_t P = 4096;
    using namespace mcpack;
    CHECK(layer_fits(P, P, 8, 9 * P));                      // exactly full
    CHECK(!layer_fits(P, P, 8, 9 * P - 1));
    CHECK(!layer_fits(P, P, 9, 9 * P));
    CHECK(layer_fits(0, 0, 8, 0));
    CHECK(!layer_fits(10 * P, P, 1, 5 * P));                // base beyond the file
    // hostile: stride * n_exp wraps to a small number (2^63 * 2 = 0 mod 2^64); the old sum base + stride * n_exp passed
    CHECK((P + (1ull << 63) * 2ull) <= 16 * P);             // what the unchecked arithmetic saw
    CHECK(!layer_fits(P, 1ull << 63, 2, 16 * P));
    CHECK(!layer_fits(P, ~0ull, 65536, ~0ull));
    CHECK(!layer_fits(~0ull - 10, P, 4, ~0ull));            // base + ... wraps
    CHECK(slice_fits(0, P, P));
    CHECK(slice_fits(P / 2, P / 2, P));
    CHECK(!slice_fits(P / 2, P, P));
    CHECK(!slice_fits(~0ull - 5, 10, P));                   // in_off + nb2 wraps to 4: the old sum passed
    CHECK(!slice_fits(10, ~0ull, P));
    CHECK(product_le(0, ~0ull, 0));
    CHECK(!product_le(1ull << 40, 1ull << 40, ~0ull));
    // CACHE-695: payload hash. CRC-32 matches the zlib value ("123456789" -> 0xCBF43926), and chunked == one piece
    CHECK(crc32_bytes(0, "123456789", 9) == 0xCBF43926u);
    CHECK(crc32_bytes(crc32_bytes(0, "1234", 4), "56789", 5) == 0xCBF43926u);
    CHECK(HASH_FLAG == 1u);
    {   // slice-by-8 against a bit-by-bit reference: every length 0..80 at every start offset 0..8 (unaligned), and a long buffer split at odd places
        const auto ref = [](uint32_t crc, const unsigned char * p, size_t n) {
            uint32_t c = crc ^ 0xffffffffu;
            for (size_t i = 0; i < n; i++) { c ^= p[i]; for (int k = 0; k < 8; k++) { c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; } }
            return c ^ 0xffffffffu;
        };
        std::vector<unsigned char> buf(20000);
        uint32_t x = 12345;
        for (auto & b : buf) { x = x * 1664525u + 1013904223u; b = (unsigned char) (x >> 24); }
        bool same = true;
        for (size_t off = 0; off <= 8; off++) {
            for (size_t n = 0; n <= 80; n++) { same = same && crc32_bytes(0, buf.data() + off, n) == ref(0, buf.data() + off, n); }
        }
        CHECK(same);
        CHECK(crc32_bytes(0, buf.data() + 3, buf.size() - 3) == ref(0, buf.data() + 3, buf.size() - 3));
        uint32_t c = 0;
        const size_t cuts[] = {1, 7, 8, 9, 4093, 4096, 11111, 11112, buf.size() - 3};     // continuing from the previous value, pieces of any size
        size_t at = 0;
        for (size_t cut : cuts) { c = crc32_bytes(c, buf.data() + 3 + at, cut - at); at = cut; }
        CHECK(c == ref(0, buf.data() + 3, buf.size() - 3));
    }
    {
        std::vector<unsigned char> pack(5 * 4096);
        for (size_t i = 0; i < pack.size(); i++) { pack[i] = (unsigned char) (i * 131 + (i >> 8)); }
        const auto rd = [&](uint64_t off, void * b, size_t n) -> size_t {
            if (off >= pack.size()) { return 0; }
            const size_t k = (size_t) std::min<uint64_t>(n, pack.size() - off);
            memcpy(b, pack.data() + off, k);
            return k;
        };
        const std::function<size_t(uint64_t, void *, size_t)> fn = rd;
        const uint32_t want = crc32_bytes(0, pack.data() + P, 3 * P);
        std::vector<char> buf(P + 100);                     // chunk size that does not divide the layer: the loop must still cover every byte
        std::string why;
        CHECK(payload_crc_ok(fn, P, 3 * P, want, buf.data(), buf.size(), why));           // good pack
        pack[P + 2 * P + 7] ^= 1;                                                          // one corrupted byte inside the layer
        CHECK(!payload_crc_ok(fn, P, 3 * P, want, buf.data(), buf.size(), why));
        CHECK(why.find("differ") != std::string::npos);
        pack[P + 2 * P + 7] ^= 1;
        pack[0] ^= 1;                                                                      // a byte outside the layer does not matter for it
        CHECK(payload_crc_ok(fn, P, 3 * P, want, buf.data(), buf.size(), why));
        why.clear();
        CHECK(!payload_crc_ok(fn, 3 * P, 3 * P, want, buf.data(), buf.size(), why));      // layer runs past the end: short read, no crash
        CHECK(why.find("cannot read") != std::string::npos);
        CHECK(!payload_crc_ok(fn, P, 3 * P, 0u, buf.data(), buf.size(), why));            // a missing hash (0 in an old pack) never matches by accident
    }
    if (fails) return 1;
    printf("pack_check_test: ok\n");
    return 0;
}
