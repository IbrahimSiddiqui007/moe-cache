// Checks src/mc_livecap.h (RAM-LIVE): the real decision functions, no stubs.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/livecap_test.cpp -o /tmp/livecap_test && /tmp/livecap_test
#include "mc_livecap.h"
#include <cstdio>

static int fails = 0;
#define CHECK(c, w) do { if (c) { printf("  ok   %s\n", w); } else { printf("  FAIL %s\n", w); fails++; } } while (0)
using namespace mclive;

int main() {
    const size_t keep = 4 * GIB;
    // off
    CHECK(ceiling(20 * GIB, 10 * GIB, 0, keep) == 20 * GIB, "no available figure: the target stays");
    CHECK(ceiling(20 * GIB, 10 * GIB, 1 * GIB, 0) == 20 * GIB, "keep 0 (off): the target stays");
    // shrink: 10 GiB held, 1 GiB available, 4 GiB kept -> give back 3 GiB
    CHECK(ceiling(20 * GIB, 10 * GIB, 1 * GIB, keep) == 7 * GIB, "available 1, keep 4: gives back 3 GiB (10 -> 7)");
    CHECK(ceiling(20 * GIB, 2 * GIB, 0 + 1, keep) == 0, "shortfall larger than what is held: 0, not an underflow");
    CHECK(ceiling(6 * GIB, 10 * GIB, 1 * GIB, keep) == 6 * GIB, "the ordinary target is the lower line: it wins");
    // room: 10 GiB held, 9 GiB available, keep 4 -> may hold up to 15; target 12 wins
    CHECK(ceiling(12 * GIB, 10 * GIB, 9 * GIB, keep) == 12 * GIB, "room 5 GiB: the target 12 is the line");
    CHECK(ceiling(20 * GIB, 10 * GIB, 9 * GIB, keep) == 15 * GIB, "room 5 GiB, target 20: 15");
    CHECK(ceiling(20 * GIB, 10 * GIB, keep, keep) == 10 * GIB, "available == keep: holds what it holds");
    // grow rules
    CHECK(!grow_allowed(keep + GIB - 1, keep, 10000, 0), "under keep + 1 GiB: no grow");
    CHECK(grow_allowed(keep + GIB, keep, 10000, 0), "at keep + 1 GiB: grow");
    CHECK(!grow_allowed(keep + 5 * GIB, keep, 10500, 10000), "a grow half a second ago: wait");
    CHECK(grow_allowed(keep + 5 * GIB, keep, 11000, 10000), "a second after the last grow: allowed");
    CHECK(!grow_allowed(keep + 5 * GIB, 0, 11000, 0), "keep 0: never");
    CHECK(grow_step(keep + 5 * GIB, keep) == (size_t) 256 << 20, "plenty of room: 256 MiB");
    CHECK(grow_step(keep + GIB + ((size_t) 100 << 20), keep) == (size_t) 100 << 20, "100 MiB over the margin: 100 MiB");
    CHECK(grow_step(keep + GIB, keep) == 0 && grow_step(keep, keep) == 0, "no room: 0");
    // a squeeze and recovery in steps: cache 10 GiB held; other programs take RAM then free it. Never over the ceiling, never growing under the line.
    size_t held = 10 * GIB, avail = 6 * GIB;
    avail = 2 * GIB;                                   // the others take 4 GiB
    held = ceiling(12 * GIB, held, avail, keep);       // evict down
    avail += 2 * GIB;                                  // the cache gave back 2 GiB (10 -> 8)
    CHECK(held == 8 * GIB, "squeeze: 10 -> 8 GiB");
    avail = keep + 3 * GIB;                            // the others free RAM
    uint64_t t = 100000, last = 0; int steps = 0;
    while (held < 12 * GIB && steps < 100 && grow_allowed(avail, keep, t, last)) {
        const size_t g = grow_step(avail, keep);
        if (!g) { break; }
        held += g; avail -= g; last = t; t += 1000; steps++;
    }
    CHECK(held > 8 * GIB && avail >= keep + GIB - 1 && steps >= 2, "recovery: grows in steps and stops 1 GiB over keep");
    printf(fails ? "livecap test: FAIL (%d)\n" : "livecap test: PASS\n", fails);
    return fails ? 1 : 0;
}
