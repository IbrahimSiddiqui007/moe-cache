// QUALITY-027: mccat::cat never writes past its buffer, also when a line is far longer than what is left (src/mc_cat.h). No ggml, no model.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/cat_test.cpp -o /tmp/cat_test && /tmp/cat_test
#include "mc_cat.h"
#include <cstdio>
#include <cstring>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    // a guard band on both sides of the buffer: nothing may be written there
    char mem[64 + 32 + 32];
    memset(mem, 'G', sizeof(mem));
    char * buf = mem + 32;
    const size_t cap = 64;
    int n = 0;
    n = mccat::cat(buf, cap, n, "abc %d", 42);
    CHECK(n == 6 && std::string(buf, 6) == "abc 42");
    n = mccat::cat(buf, cap, n, "; %s", std::string(300, 'x').c_str());          // far longer than the room left
    CHECK(n == (int) cap - 1 && buf[cap - 1] == 0 && strlen(buf) == cap - 1);
    n = mccat::cat(buf, cap, n, " and more %d", 1);                                // already full: still safe, nothing added
    CHECK(n == (int) cap - 1 && strlen(buf) == cap - 1);
    n = mccat::cat(buf, cap, (int) cap + 50, "x");                                 // a fill position past the end (the old wrap-around case)
    CHECK(n == (int) cap - 1);
    bool guard_ok = true;
    for (int i = 0; i < 32; i++) {
        guard_ok = guard_ok && mem[i] == 'G' && mem[32 + 64 + i] == 'G';
    }
    CHECK(guard_ok);
    CHECK(mccat::cat(buf, cap, -5, "ok") == 2);                                   // a negative fill counts as 0
    if (fails) { fprintf(stderr, "%d checks failed\n", fails); return 1; }
    printf("cat checks passed\n");
    return 0;
}
