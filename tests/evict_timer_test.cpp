// DECODE-123: the always-on evict() timer (src/mc_evict_timer.h). No ggml, no GPU.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/evict_timer_test.cpp -o /tmp/evict_timer_test && /tmp/evict_timer_test
#include "mc_evict_timer.h"
#include <cstdio>
#include <thread>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    // add() accumulates into ns; decode steps also split into ns_dec / calls_dec, non-decode steps do not
    mcevict::timer_t t;
    t.add(100, false);
    t.add(200, true);
    t.add(300, true);
    CHECK(t.ns.load() == 600);
    CHECK(t.ns_dec.load() == 500);
    CHECK(t.calls_dec.load() == 2);
    t.add(50, false);
    CHECK(t.ns.load() == 650);
    CHECK(t.ns_dec.load() == 500);          // the split is unchanged by prefill-side time
    CHECK(t.calls_dec.load() == 2);

    // a decode scope records at least the time slept (2 ms)
    mcevict::timer_t ts;
    {
        mcevict::scope_t scope{ts, true};
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    CHECK(ts.ns.load() >= 2000000);
    CHECK(ts.ns_dec.load() >= 2000000);     // scope ns == decode ns when every call is a decode call
    CHECK(ts.calls_dec.load() == 1);
    CHECK(ts.ns_dec.load() <= ts.ns.load());

    // a non-decode scope leaves the decode split at zero
    mcevict::timer_t tp;
    {
        mcevict::scope_t scope{tp, false};
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    CHECK(tp.ns.load() >= 2000000);
    CHECK(tp.ns_dec.load() == 0);
    CHECK(tp.calls_dec.load() == 0);

    if (fails) { fprintf(stderr, "%d checks failed\n", fails); return 1; }
    printf("evict-timer checks passed\n");
    return 0;
}
