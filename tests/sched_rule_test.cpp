// WINDOWS-013: which scheduling switches are on (src/mc_sched_rule.h). Default on: EcoQoS off + CPU priority above normal + GPU priority above normal (the measured
// bundle, +2.0 % prompt on the PC, both pairs won); MOE_CACHE_SCHED_TUNING=0 and the single variables opt out. No Windows calls here.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/sched_rule_test.cpp -o /tmp/sched_rule_test && /tmp/sched_rule_test
#include "mc_sched_rule.h"
#include <cstdio>
#include <initializer_list>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    using mcsched::choose;
    auto d = choose(nullptr, nullptr, nullptr, nullptr);                 // nothing set: the default is ON
    CHECK(d.eco_off && d.proc == 1 && d.gpu == 3);
    d = choose("", "", "", "");                                          // empty text counts as not set
    CHECK(d.eco_off && d.proc == 1 && d.gpu == 3);
    for (const char * off : { "0", "off", "false", "no", "none" }) {     // the opt-out: nothing is changed
        d = choose(off, nullptr, nullptr, nullptr);
        CHECK(!d.eco_off && d.proc == 0 && d.gpu == 0);
    }
    d = choose("1", nullptr, nullptr, nullptr);
    CHECK(d.eco_off && d.proc == 1 && d.gpu == 3);
    d = choose(nullptr, "0", nullptr, nullptr);                          // one switch off, the others stay on
    CHECK(!d.eco_off && d.proc == 1 && d.gpu == 3);
    d = choose(nullptr, nullptr, "off", nullptr);
    CHECK(d.eco_off && d.proc == 0 && d.gpu == 3);
    d = choose(nullptr, nullptr, nullptr, "none");
    CHECK(d.eco_off && d.proc == 1 && d.gpu == 0);
    d = choose(nullptr, nullptr, "high", "high");                        // the old values still work
    CHECK(d.proc == 2 && d.gpu == 4);
    d = choose(nullptr, nullptr, "above", "realtime");
    CHECK(d.proc == 1 && d.gpu == 5);
    d = choose("0", "1", "high", "above");                               // master off, but single variables set explicitly still act (the old behaviour)
    CHECK(d.eco_off && d.proc == 2 && d.gpu == 3);
    d = choose("0", nullptr, nullptr, "r");
    CHECK(!d.eco_off && d.proc == 0 && d.gpu == 5);
    if (fails == 0) printf("sched rule test: PASS\n");
    return fails ? 1 : 0;
}
