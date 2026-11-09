// CACHE-988: the GPU block's ggml context must hold the worst case of a block (a Nemotron-style 512-expert layer included) even when the loaded ggml
// charges more per graph than the headers say, and a pool that runs out must end in "no GPU block, the CPU runs it", never in a null dereference.
// No ggml: a fake pool that returns null when full, with the same bookkeeping as ggml_new_object.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/ctx_size_test.cpp -o /tmp/ctx_size_test && /tmp/ctx_size_test
#include "mc_ctx_size.h"
#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

struct pool_t {
    size_t cap, used = 0;
    explicit pool_t(size_t c) : cap(c) {}
    bool take(size_t n) {                // ggml_new_object: null when the object does not fit
        if (used + n > cap) { return false; }
        used += n;
        return true;
    }
};
constexpr size_t TENSOR_OH = 368;                                     // object header + tensor
constexpr size_t MAX_SRC = 10;
static size_t graph_oh(size_t n, size_t extra) { return n * 8 * 2 + (n * 2 + 1) * 8 + n / 8 + 256 + extra; }    // nodes, leafs, hash keys, bitset, header

// the block as gpu_segment builds it: one tensor per node, one per distinct source, then one graph; returns false when the pool ran out
static bool build_block(size_t n_nodes, size_t n_srcs_each, size_t pool_extra, size_t real_extra, bool old_formula) {
    size_t slots = mcctx::slots(n_nodes, MAX_SRC);
    size_t cap = old_formula ? TENSOR_OH * 4 * (n_nodes + 8) + graph_oh(2048, 0) + 4096 : mcctx::bytes(slots, TENSOR_OH, graph_oh(slots, 0));
    pool_t p(cap + pool_extra);
    for (size_t i = 0; i < n_nodes * (1 + n_srcs_each); i++) {
        if (!p.take(TENSOR_OH)) { return false; }
    }
    return p.take(graph_oh(old_formula ? 2048 : slots, real_extra));
}

int main() {
    // every block size the plugin can see, with the loaded ggml charging up to 8 KiB more per graph than its headers say
    for (size_t n : { 1, 2, 3, 4, 6, 8, 16, 40, 128, 512 }) {
        for (size_t extra : { 0, 688, 4096, 8192 }) {
            CHECK(build_block(n, 4, 0, extra, false));                // new sizing: fits with 16 KiB slack
            CHECK(build_block(n, MAX_SRC, 0, 0, false));              // worst case: every node has the maximum number of sources
        }
    }
    // the old sizing: enough only while the tensors stay at 4 per node (a block that needs more, or a graph that costs more, ran out)
    CHECK(build_block(3, 3, 0, 0, true));
    CHECK(!build_block(3, 3, 0, 40000, true));
    // a pool that is too small is reported as false (the caller frees and returns false), never partially used
    CHECK(!build_block(512, 4, 0, 4 << 20, false));
    if (fails) { return 1; }
    printf("ctx_size_test: ok\n");
    return 0;
}
