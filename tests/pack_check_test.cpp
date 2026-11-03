// AUD-019: the pack layer table checks do not wrap around at 2^64.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/pack_check_test.cpp -o /tmp/pack_check_test && /tmp/pack_check_test
#include "mc_pack_check.h"
#include <cstdio>

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
    if (fails) return 1;
    printf("pack_check_test: ok\n");
    return 0;
}
