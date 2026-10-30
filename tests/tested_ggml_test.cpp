// Checks src/mc_tested.h: a ggml version in the tested list matches every build of it, "version@commit" only that build.
//   g++ -std=c++17 -Wall -Werror -Isrc tests/tested_ggml_test.cpp -o /tmp/tested_ggml_test && /tmp/tested_ggml_test
#include "mc_tested.h"
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    using mctested::tested;
    const char * L = "0.26.0, 0.27.0, 0.28.0@ff0d396e3";
    CHECK(tested(L, "0.26.0", "23b0202a1") && tested(L, "0.26.0", nullptr) && tested(L, "0.27.0", "620d2225c") && tested(L, "0.27.0", ""));   // bare versions: any build
    CHECK(tested(L, "0.28.0", "ff0d396e3"));                                   // the build CI runs
    CHECK(tested(L, "0.28.0", "ff0d396e3dda0f33e7ce5ad4267fce4f7690cfb1"));    // a full commit starts with the entry
    CHECK(!tested(L, "0.28.0", "a34de8618"));                                  // another 0.28.0 build: untested
    CHECK(!tested(L, "0.28.0", nullptr) && !tested(L, "0.28.0", ""));          // 0.28.0 with no commit known: untested
    CHECK(!tested(L, "0.29.0", "ff0d396e3") && !tested(L, "0.25.1", "7fe450e19") && !tested(L, "", "ff0d396e3"));
    CHECK(!tested(L, "0.28.0", "ff0d39"));                                     // too short a commit prefix is not a match
    CHECK(tested("0.25.1", "0.25.1", "7fe450e19") && !tested("0.25.1", "0.25.10", nullptr) && !tested("", "0.25.1", nullptr));
    CHECK(tested("0.27.0,0.28.0@ab12cd34e", "0.28.0", "ab12cd34e56f"));        // no spaces
    CHECK(!tested("0.28.0@", "0.28.0", "ff0d396e3"));                          // an empty commit entry matches nothing
    if (fails) {
        fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    printf("tested-ggml checks passed\n");
    return 0;
}
