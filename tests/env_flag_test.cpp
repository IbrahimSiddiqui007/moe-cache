// Checks src/mc_env.h (AUD-004/005/006): one parse for every boolean MOE_CACHE_* switch, and the size parse of MOE_CACHE_HYBRID.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/env_flag_test.cpp -o /tmp/env_flag_test && /tmp/env_flag_test
#include "mc_env.h"
#include <cstdio>
#include <cstdlib>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    using mcenv::flag_value;
    // unset and empty give the default, either way round (docker-compose "VAR=" must not flip a switch)
    CHECK(flag_value(nullptr, true) && !flag_value(nullptr, false));
    CHECK(flag_value("", true) && !flag_value("", false) && flag_value("  ", true) && !flag_value("  ", false));
    // off words and 0 are false for default-on and default-off switches (diagnostics: =0 must NOT turn them on)
    for (const char * v : { "0", "off", "OFF", "false", "False", "no", " 0 ", "-1", "00" }) {
        CHECK(!flag_value(v, true));
        CHECK(!flag_value(v, false));
    }
    // on words and numbers > 0 are true (default-on features must stay on for "true"/"on"/"yes")
    for (const char * v : { "1", "on", "On", "true", "TRUE", "yes", "2", "17", " 1" }) {
        CHECK(flag_value(v, true));
        CHECK(flag_value(v, false));
    }
    // some other word: on (a default-off switch set to a word was always on, a default-on one stays on)
    CHECK(flag_value("enabled", false) && flag_value("enabled", true));
    // DECODE-104: MOE_CACHE_GPU_MIN_TOKENS=1 repeats a token on the GPU: 1 becomes 2 (and says so), other values pass, unset keeps the default
    {
        bool cl = false;
        CHECK(mcenv::gpu_min_tokens_value("1", 8, &cl) == 2 && cl);
        CHECK(mcenv::gpu_min_tokens_value("2", 8, &cl) == 2 && !cl);
        CHECK(mcenv::gpu_min_tokens_value("16", 8, &cl) == 16 && !cl);
        CHECK(mcenv::gpu_min_tokens_value("0", 8, &cl) == 0 && !cl);
        CHECK(mcenv::gpu_min_tokens_value(nullptr, 8, &cl) == 8 && !cl);
        CHECK(mcenv::gpu_min_tokens_value("1", 8) == 2);
    }
    // through the environment
    setenv("MOE_CACHE_LOOKAHEAD", "", 1);
    CHECK(!mcenv::env_on("MOE_CACHE_LOOKAHEAD"));                  // MOE_CACHE_LOOKAHEAD= used to switch the lookahead on (and the executor off)
    setenv("MOE_CACHE_HYBRID", "off", 1);
    CHECK(!mcenv::env_on("MOE_CACHE_HYBRID"));
    setenv("MOE_CACHE_SEG_TIMES", "0", 1);
    CHECK(!mcenv::env_on("MOE_CACHE_SEG_TIMES"));                  // used to add a synchronize after every upload and compute
    setenv("MOE_CACHE_EXEC_FUSE", "true", 1);
    CHECK(mcenv::env_flag("MOE_CACHE_EXEC_FUSE", true));           // atoi("true") == 0 used to switch this default-on feature off
    unsetenv("MOE_CACHE_EXEC_FUSE");
    CHECK(mcenv::env_flag("MOE_CACHE_EXEC_FUSE", true) && !mcenv::env_flag("MOE_CACHE_EXEC_FUSE", false) && !mcenv::env_on("MOE_CACHE_EXEC_FUSE"));
    // MOE_CACHE_HYBRID is a size; the same function answers "asked" and "how many MiB"
    using mcenv::hybrid_mib;
    CHECK(hybrid_mib(nullptr) == 0 && hybrid_mib("") == 0 && hybrid_mib("0") == 0 && hybrid_mib("off") == 0 && hybrid_mib("false") == 0 && hybrid_mib("no") == 0);
    CHECK(hybrid_mib("-5") == 0 && hybrid_mib("junk") == 0);
    CHECK(hybrid_mib("4096") == 4096 && hybrid_mib(" 512 ") == 512 && hybrid_mib("auto") == (size_t) -1);
    if (fails) {
        fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    printf("env-flag checks passed\n");
    return 0;
}
