// CACHE-1230: the long-prompt block copies back only its last node; a reader outside the range of another node of it, or a graph-output flag on
// one, must make the block stay off the GPU. No ggml: a plain node list.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/seg_readers_test.cpp -o /tmp/seg_readers_test && /tmp/seg_readers_test
#include "mc_seg_readers.h"
#include <cstdio>
#include <set>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

struct g_t { std::vector<std::vector<int>> srcs; std::set<int> outputs; };
static bool safe(const g_t & g, int first, int last) {
    return mcseg::inner_nodes_private((int) g.srcs.size(), first, last, 3,
        [&](int i, int k) { return k < (int) g.srcs[i].size() ? g.srcs[i][k] : -1; }, [&](int i) { return g.outputs.count(i) > 0; });
}

int main() {
    // 0 = norm, 1 = up (reads 0), 2 = gate (reads 0), 3 = glu (reads 1, 2), 4 = down (reads 3), 5 = add (reads 4), 6 = next layer (reads 5)
    g_t g;
    g.srcs = { {}, { 0 }, { 0 }, { 1, 2 }, { 3 }, { 4 }, { 5 } };
    CHECK(safe(g, 1, 4));                       // the normal block: only the last node is read from outside
    g_t bad = g;
    bad.srcs[6].push_back(2);                   // a later node also reads the gate output (inner node)
    CHECK(!safe(bad, 1, 4));
    g_t early = g;
    early.srcs[0] = { 3 };                      // a node before the block reading an inner node (cannot be in a valid graph, still refused)
    CHECK(!safe(early, 1, 4));
    g_t out = g;
    out.outputs.insert(3);                      // an inner node flagged as graph output
    CHECK(!safe(out, 1, 4));
    g_t lastout = g;
    lastout.outputs.insert(4);                  // the last node may be an output: it is copied back
    CHECK(safe(lastout, 1, 4));
    g_t lastread = g;
    lastread.srcs[6].push_back(4);              // reading the last node from outside is the normal case
    CHECK(safe(lastread, 1, 4));
    CHECK(safe(g, 4, 4));                       // single-node block
    if (fails) { return 1; }
    printf("seg_readers_test: ok\n");
    return 0;
}
