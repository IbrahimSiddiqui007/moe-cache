// Checks src/mc_shared_gpu.h (CACHE-149): the sizing arithmetic for a shared-memory (integrated) GPU. The rest of the feature (device detection, the real
// buffers, the budget of a live cache) needs an iGPU and is NOT covered here.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/shared_gpu_test.cpp -o /tmp/shared_gpu_test && /tmp/shared_gpu_test
#include "mc_shared_gpu.h"
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    using namespace mcshared;
    // cap: 1/8 of the RAM the plugin may use, 1..4 GiB; the user's MiB value wins
    CHECK(stage_cap(28 * GiB, 0) == 3 * GiB + GiB / 2);            // the Claw: 28 GiB limit -> 3.5 GiB
    CHECK(stage_cap(64 * GiB, 0) == 4 * GiB);                      // big machine: capped at 4 GiB
    CHECK(stage_cap(16 * GiB, 0) == 2 * GiB);
    CHECK(stage_cap(6 * GiB, 0) == GiB);                           // small: at least 1 GiB ...
    CHECK(stage_cap(2 * GiB, 0) == GiB / 2);                       // ... but never more than 1/4 of the RAM
    CHECK(stage_cap(28 * GiB, 1500) == 1500 * MiB && stage_cap(2 * GiB, 9000) == 9000 * MiB);
    CHECK(stage_cap(28 * GiB, -5) == 3 * GiB + GiB / 2);           // junk value: the default
    // free memory the fit test sees: the driver's figure (shared RAM, huge) limited by what the cap leaves after the held buffer
    CHECK(free_for_block(20 * GiB, 0, 3 * GiB) == 3 * GiB);
    CHECK(free_for_block(20 * GiB, GiB, 3 * GiB) == 2 * GiB);
    CHECK(free_for_block(20 * GiB, 5 * GiB, 3 * GiB) == 0);        // already holding more than the cap: only blocks that fit in what is held run
    CHECK(free_for_block(GiB, 0, 3 * GiB) == GiB);                 // the driver is the tighter limit
    // fits(need) = free >= (need - held) + headroom  <=>  need + headroom <= cap (when the driver has the room)
    {
        const size_t cap = 3 * GiB, headroom = 256 * MiB, held = GiB;
        for (size_t need : { 2 * GiB, 2 * GiB + 768 * MiB, 2 * GiB + 769 * MiB, 3 * GiB }) {
            const bool fits = free_for_block(20 * GiB, held, cap) >= (need > held ? need - held : 0) + headroom;
            CHECK(fits == (need + headroom <= cap));
        }
    }
    // counting: growth lowers the budget and the autotuner's ceiling, shrinking gives the ceiling back exactly
    {
        account_t a;
        size_t budget = 20 * GiB, max_budget = 40 * GiB;
        CHECK(account(a, 2 * GiB, budget, max_budget, GiB) && budget == 18 * GiB && max_budget == 38 * GiB && a.reserved == 2 * GiB);
        CHECK(!account(a, 2 * GiB, budget, max_budget, GiB) && budget == 18 * GiB);                    // unchanged: nothing moves
        CHECK(account(a, 3 * GiB, budget, max_budget, GiB) && budget == 17 * GiB && max_budget == 37 * GiB);
        CHECK(!account(a, GiB, budget, max_budget, GiB) && budget == 17 * GiB && max_budget == 39 * GiB);   // shrink: ceiling back, budget regrows by tuning
        CHECK(!account(a, 0, budget, max_budget, GiB) && max_budget == 40 * GiB && a.max_taken == 0);
    }
    {   // the autotuner has not set a ceiling yet (0): only the budget moves; nothing is given back later that was never taken
        account_t a;
        size_t budget = 10 * GiB, max_budget = 0;
        CHECK(account(a, GiB, budget, max_budget, GiB) && budget == 9 * GiB && max_budget == 0);
        CHECK(!account(a, 0, budget, max_budget, GiB) && max_budget == 0);
    }
    {   // never below the floor
        account_t a;
        size_t budget = 2 * GiB, max_budget = 3 * GiB;
        CHECK(account(a, 5 * GiB, budget, max_budget, GiB) && budget == GiB && max_budget == GiB && a.max_taken == 2 * GiB);
        CHECK(!account(a, 0, budget, max_budget, GiB) && max_budget == 3 * GiB);
        size_t b2 = GiB / 2, m2 = GiB / 2;                              // already under the floor: untouched
        account_t a2;
        CHECK(!account(a2, GiB, b2, m2, GiB) && b2 == GiB / 2 && m2 == GiB / 2);
    }
    if (fails) {
        fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    printf("shared-gpu checks passed\n");
    return 0;
}
