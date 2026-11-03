// AUD-013: the exit guard leaves the state alone while a graph is active, even when the cache lock happens to be free.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/exit_test.cpp -o /tmp/exit_test && /tmp/exit_test
#include "mc_exit.h"
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    using mcexit::action;
    CHECK(mcexit::choose(false, true, 0) == action::delete_state);          // idle server: threads joined, profile saved, state deleted
    CHECK(mcexit::choose(false, true, 1) == action::leave_state);           // lock free between two steps of a running graph (the old code deleted here)
    CHECK(mcexit::choose(false, true, 3) == action::leave_state);
    CHECK(mcexit::choose(false, false, 0) == action::leave_state);          // lock held
    CHECK(mcexit::choose(true, true, 0) == action::leave_state);            // Windows process teardown
    if (fails) return 1;
    printf("exit_test: ok\n");
    return 0;
}
