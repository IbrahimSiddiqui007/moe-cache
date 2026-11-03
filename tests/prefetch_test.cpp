// R-02 (src/mc_prefetch.h): env parse, distance cap, lines per row, nothing asked past the last row. Model-free.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/prefetch_test.cpp -o /tmp/prefetch_test && /tmp/prefetch_test
#include "mc_prefetch.h"
#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    using namespace mcpf;
    // off by default and for every non-number
    for (const char * v : { (const char *) nullptr, "", "0", "-3", "off", "false", "no", "abc", "4x" }) CHECK(parse_rows(v) == 0);
    // numbers, the cap, the words
    CHECK(parse_rows("1") == 1 && parse_rows("4") == 4 && parse_rows("16") == 16 && parse_rows("17") == 16 && parse_rows("99999") == 16);
    CHECK(parse_rows("on") == 4 && parse_rows("true") == 4 && parse_rows("yes") == 4);
    // lines per row: 1.5 KB -> 24, 4 KB -> 64, a 64 KB row is capped
    CHECK(lines_for(1) == 1 && lines_for(64) == 1 && lines_for(65) == 2 && lines_for(1536) == 24 && lines_for(4096) == 64 && lines_for(65536) == kMaxLines);
    // nothing is asked past the last row, nor with distance 0
    CHECK(lines_asked(1536, 0, 4, 100) == 24);
    CHECK(lines_asked(1536, 95, 4, 100) == 24);        // row 99 is the last one
    CHECK(lines_asked(1536, 96, 4, 100) == 0);         // row 100 does not exist
    CHECK(lines_asked(1536, 0, 0, 100) == 0 && lines_asked(1536, 0, -1, 100) == 0);
    // rows_ahead on a real buffer runs for every row without touching memory outside it (a prefetch cannot fault, but the address must stay inside)
    std::vector<char> w(1536 * 100, 1);
    for (int64_t r = 0; r < 100; r++) rows_ahead(w.data(), 1536, r, 8, 100);
    CHECK(true);
    if (fails) return 1;
    printf("prefetch_test: all ok\n");
    return 0;
}
