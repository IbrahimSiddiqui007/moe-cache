// AUD-021: prefill_lend gives back only what tune() did not already take away meanwhile.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/lend_test.cpp -o /tmp/lend_test && /tmp/lend_test
#include "mc_lend.h"
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    const size_t M = 1 << 20;
    // budget 10000 MiB, lends 2000 -> 8000. Nothing changed: all 2000 come back (10000 again)
    CHECK(mclend::restore_amount(2000 * M, 8000 * M, 8000 * M) == 2000 * M);
    // tune() cut the budget to 7000 during the long prompt: restore adds 1000, not 2000 (the old code gave 9000, above what tune decided)
    CHECK(8000 * M - 1000 * M + mclend::restore_amount(2000 * M, 8000 * M, 7000 * M) == 8000 * M);
    // tune() cut by more than the lend: nothing is added
    CHECK(mclend::restore_amount(2000 * M, 8000 * M, 5000 * M) == 0);
    // tune() grew the budget: the whole amount (the caller still caps at max_budget)
    CHECK(mclend::restore_amount(2000 * M, 8000 * M, 9000 * M) == 2000 * M);
    CHECK(mclend::restore_amount(0, 8000 * M, 8000 * M) == 0);
    if (fails) return 1;
    printf("lend_test: ok\n");
    return 0;
}
