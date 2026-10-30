// Fake-trace test of the idle rebalance choice (src/mc_rebalance.h): no model. g++ -std=c++17 -Isrc tests/rebalance_test.cpp -o /tmp/rebalance_test && /tmp/rebalance_test
#include "mc_rebalance.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <random>
static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)
static rb_unit U(double score, size_t miss, size_t res, bool locked = false) { rb_unit u; u.score = score; u.miss_bytes = miss; u.res_bytes = res; u.locked = locked; return u; }
int main() {
    // 1. free room first: no victim is touched while there is space
    { std::vector<rb_unit> u = { U(9, 10, 0), U(8, 10, 0), U(0.1, 0, 10) };
      auto p = rb_choose(u, 20, 1000, 2.0, 0.75);
      CHECK(p.size() == 2 && p[0].cand == 0 && p[1].cand == 1 && p[0].victims.empty() && p[1].victims.empty()); }
    // 2. full cache: the worst resident unit makes room for the best missing one, only with the margin
    { std::vector<rb_unit> u = { U(10, 10, 0), U(6, 10, 0), U(1, 0, 10), U(4, 0, 10), U(5.9, 0, 10) };
      auto p = rb_choose(u, 0, 1000, 2.0, 0.75);
      CHECK(p.size() == 1 && p[0].cand == 0 && p[0].victims.size() == 1 && p[0].victims[0] == 2);   // 6 vs victim 4: 4*2+.25 > 6, stop
    }
    // 3. margin: a missing unit that scores about the same as a resident one does not swap (no thrash)
    { std::vector<rb_unit> u = { U(5, 10, 0), U(4, 0, 10) };
      CHECK(rb_choose(u, 0, 1000, 2.0, 0.75).empty()); }
    // 4. locked units are never victims; cap bounds the bytes; min score filters noise
    { std::vector<rb_unit> u = { U(10, 10, 0), U(0.5, 0, 10, true) };
      CHECK(rb_choose(u, 0, 1000, 2.0, 0.75).empty()); }
    { std::vector<rb_unit> u = { U(10, 10, 0), U(9, 10, 0), U(8, 10, 0) };
      CHECK(rb_choose(u, 100, 20, 2.0, 0.75).size() == 2); }
    { std::vector<rb_unit> u = { U(0.5, 10, 0) };
      CHECK(rb_choose(u, 100, 100, 2.0, 0.75).empty()); }
    // 5. a candidate needing two victims takes the two worst
    { std::vector<rb_unit> u = { U(20, 20, 0), U(1, 0, 10), U(2, 0, 10), U(15, 0, 10) };
      auto p = rb_choose(u, 0, 1000, 2.0, 0.75);
      CHECK(p.size() == 1 && p[0].victims.size() == 2 && p[0].victims[0] == 1 && p[0].victims[1] == 2); }
    // 6. fake trace: a cache of 30 units filled with last turn's experts, a new turn that favours a hot set; the plan must raise the total
    // score held in the cache, never exceed the room, and never evict a unit it also loads
    { std::mt19937 g(7); std::vector<rb_unit> u(100);
      for (int i = 0; i < 100; i++) { u[i].score = (i < 20 ? 8.0 : 0.3) + (g() % 100) / 200.0; u[i].layer = i / 10; u[i].e = i % 10; }
      std::vector<bool> res(100, false); size_t held = 0;
      for (int i = 70; i < 100; i++) { res[i] = true; held++; }          // resident: 30 cold units
      for (int i = 0; i < 100; i++) { u[i].res_bytes = res[i] ? 10 : 0; u[i].miss_bytes = res[i] ? 0 : 10; }
      double before = 0; for (int i = 0; i < 100; i++) if (res[i]) before += u[i].score;
      auto p = rb_choose(u, 0, 100000, 2.0, 0.75);
      CHECK(p.size() == 20);
      std::vector<int> evicted(100, 0);
      for (auto & k : p) { for (int v : k.victims) { CHECK(res[v] && !evicted[v]); evicted[v] = 1; res[v] = false; held--; } CHECK(!res[k.cand]); res[k.cand] = true; held++; CHECK(held <= 30); }
      double after = 0; for (int i = 0; i < 100; i++) if (res[i]) after += u[i].score;
      CHECK(after > before * 3); CHECK(held == 30); }
    if (fails) { fprintf(stderr, "%d failures\n", fails); return 1; }
    printf("rebalance_test: ok\n");
    return 0;
}
