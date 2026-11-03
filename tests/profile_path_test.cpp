// AUD-016: a profile that belongs to another model is kept; this model uses "<path>.<fingerprint>" next to it.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/profile_path_test.cpp -o /tmp/profile_path_test && /tmp/profile_path_test
#include "mc_profile_path.h"
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    CHECK(mcprofile::own_path("/p/use.bin", 0x1234abcdull) == "/p/use.bin.00000000" "1234abcd");
    CHECK(mcprofile::own_path("use.bin", 0xffffffffffffffffull) == "use.bin.ffffffffffffffff");
    CHECK(mcprofile::own_path("a", 1) != mcprofile::own_path("a", 2));      // two models, two files
    CHECK(mcprofile::own_path("a", 7) == mcprofile::own_path("a", 7));      // the same model finds its file again
    CHECK(mcprofile::own_path("a", 7) != "a");                              // never the other model's file
    if (fails) return 1;
    printf("profile_path_test: ok\n");
    return 0;
}
