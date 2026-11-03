// Model-free test of the decode tier's layer choice (src/mc_tier.h). g++ -std=c++17 -O1 -Wall -Werror -Isrc tests/tier_test.cpp -o /tmp/tier_test && /tmp/tier_test
#include "mc_tier.h"
#include <cstdio>
#include <string>
static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)
using V = std::vector<int>;
int main() {
    // parse: order kept, duplicates, junk and negatives dropped, empty and null are empty
    CHECK(tier_parse("38,39") == (V{ 38, 39 }));
    CHECK(tier_parse(" 39 , 38,39;x7") == (V{ 39, 38, 7 }));
    CHECK(tier_parse("-1,5") == (V{ 5 }));
    CHECK(tier_parse("") .empty() && tier_parse(nullptr).empty() && tier_parse("abc").empty());
    { std::string big; for (int i = 0; i < 100; i++) big += std::to_string(i) + ","; CHECK(tier_parse(big.c_str()).size() == 32); }
    // fit: Flash-Next numbers, 676 MiB a layer
    const size_t MiB = 1048576, L = 676 * MiB;
    std::map<int, size_t> lb; for (int l = 0; l < 48; l++) lb[l] = L;
    CHECK(tier_fit({ 38, 39 }, lb, 1500 * MiB, 100 * MiB) == (V{ 38, 39 }));          // 1400 MiB room, 1352 needed
    CHECK(tier_fit({ 38, 39 }, lb, 1400 * MiB, 100 * MiB) == (V{ 38 }));              // 1300 MiB room: one layer
    CHECK(tier_fit({ 38, 39 }, lb, 700 * MiB, 100 * MiB).empty());                    // 600 MiB room: none
    CHECK(tier_fit({ 38, 39 }, lb, 100 * MiB, 512 * MiB).empty());                    // less free than kept: none, no underflow
    CHECK(tier_fit({ 99, 5, 6 }, lb, 2000 * MiB, 0) == (V{ 5, 6 }));                  // a layer the model does not have is skipped
    lb[7] = 2000 * MiB;
    CHECK(tier_fit({ 7, 5 }, lb, 2500 * MiB, 0).size() == 1 && tier_fit({ 7, 5 }, lb, 2500 * MiB, 0)[0] == 7);   // the first that does not fit ends the list, not before
    CHECK(tier_fit({ 5, 7, 6 }, lb, 2500 * MiB, 0) == (V{ 5 }));
    if (fails) { fprintf(stderr, "%d failed\n", fails); return 1; }
    printf("tier_test: ok\n");
    return 0;
}
