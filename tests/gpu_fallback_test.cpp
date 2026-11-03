// PREFILL-172: a GPU error in the middle of a block leaves no half-written state, and the CPU result is the one stock would give. No ggml, no GPU, no model:
// the fault is injected the way the plugin gets it (an exception thrown out of the GPU call, after the GPU has written part of the output), through the
// same mcguard::guarded / make_undo the plugin's gpu_segment_guarded, hybrid_group_guarded, be_compute and exec_run_nodes use.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/gpu_fallback_test.cpp -o /tmp/gpu_fallback_test && /tmp/gpu_fallback_test
#include "mc_guard.h"
#include <cstdio>
#include <string>
#include <system_error>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

enum op_t { OP_NONE = 0, OP_MUL_MAT_ID = 1, OP_GLU = 2 };
struct node_t { op_t op; std::vector<float> out; };
struct graph_t { std::vector<node_t> nodes; bool gpu_done = false; };

static graph_t make_graph() {
    graph_t g;
    g.nodes = { { OP_MUL_MAT_ID, std::vector<float>(8, -1.f) }, { OP_GLU, std::vector<float>(8, -1.f) }, { OP_MUL_MAT_ID, std::vector<float>(8, -1.f) } };
    return g;
}
// the CPU path: computes every node (value = node index * 10 + element), what stock llama.cpp would produce
static void cpu_compute(graph_t & g) {
    for (size_t i = 0; i < g.nodes.size(); i++) {
        for (size_t k = 0; k < g.nodes[i].out.size(); k++) {
            g.nodes[i].out[k] = (float) (i * 10 + k);
        }
    }
}
static bool same(const graph_t & a, const graph_t & b) {
    for (size_t i = 0; i < a.nodes.size(); i++) {
        if (a.nodes[i].out != b.nodes[i].out || a.nodes[i].op != b.nodes[i].op) {
            return false;
        }
    }
    return true;
}

enum fault_t { NONE, STD_EXCEPTION, SYSTEM_ERROR, FOREIGN, HANDLER_THROWS };

// one block as be_compute runs it: GPU first (guarded), the nodes it took over are silenced, then the CPU path computes the rest; on a fault the whole
// block goes to the CPU. Returns the number of log lines the failure handler wrote.
static int run_block(graph_t & g, int mode, bool & injected, fault_t fault, bool & stayed_cpu) {
    int logged = 0;
    std::vector<std::pair<node_t *, op_t>> silenced;
    auto undo = mcguard::make_undo([&] { for (auto & s : silenced) { s.first->op = s.second; } silenced.clear(); });
    const bool ok = mcguard::guarded(mode, injected, [&]() -> bool {
        // the GPU wrote the first node's output and then failed
        for (size_t k = 0; k < g.nodes[0].out.size(); k++) { g.nodes[0].out[k] = 999.f; }
        switch (fault) {
            case STD_EXCEPTION: throw std::runtime_error("vk::OutOfDeviceMemoryError");
            case SYSTEM_ERROR:  throw std::system_error(std::make_error_code(std::errc::not_enough_memory));
            case FOREIGN:       throw 42;
            case HANDLER_THROWS: throw std::runtime_error("vk::DeviceLostError");
            case NONE: break;
        }
        for (size_t k = 0; k < g.nodes[0].out.size(); k++) { g.nodes[0].out[k] = (float) k; }   // GPU result (same numbers as the CPU: node 0)
        g.gpu_done = true;
        return true;
    }, [&](const char *) {
        logged++;
        stayed_cpu = true;
        if (fault == HANDLER_THROWS) { throw std::runtime_error("the GPU is gone, synchronize failed too"); }
    });
    if (ok) {
        silenced.emplace_back(&g.nodes[0], g.nodes[0].op);       // the GPU block's node is switched off for the CPU pass
        g.nodes[0].op = OP_NONE;
        for (size_t i = 1; i < g.nodes.size(); i++) {
            for (size_t k = 0; k < g.nodes[i].out.size(); k++) { g.nodes[i].out[k] = (float) (i * 10 + k); }
        }
    } else {
        cpu_compute(g);                                          // fall back: everything on the CPU
    }
    undo.run();
    return logged;
}

int main() {
    graph_t ref = make_graph();
    cpu_compute(ref);

    // no fault: the GPU block is used and the final state equals the CPU result (ops restored)
    {
        graph_t g = make_graph(); bool inj = false, cpu = false;
        CHECK(run_block(g, 1, inj, NONE, cpu) == 0 && g.gpu_done && !cpu && same(g, ref));
    }
    // every kind of fault: one log line, CPU result, nothing from the GPU's half-written output left, ops intact
    for (fault_t f : { STD_EXCEPTION, SYSTEM_ERROR, FOREIGN, HANDLER_THROWS }) {
        graph_t g = make_graph(); bool inj = false, cpu = false;
        int logged = -1;
        bool escaped = false;
        try { logged = run_block(g, 1, inj, f, cpu); } catch (...) { escaped = true; }
        CHECK(!escaped && logged == 1 && cpu && !g.gpu_done && same(g, ref));
    }
    // env-driven injection (MOE_CACHE_GPU_CATCH=2): first block faults and equals CPU, the second block uses the GPU again
    {
        graph_t g1 = make_graph(), g2 = make_graph(); bool inj = false, cpu = false;
        CHECK(run_block(g1, 2, inj, NONE, cpu) == 1 && cpu && same(g1, ref));
        cpu = false;
        CHECK(run_block(g2, 2, inj, NONE, cpu) == 0 && g2.gpu_done && same(g2, ref));
    }
    // =0 opts out: the throw leaves, but the silenced ops still come back (scope undo)
    {
        graph_t g = make_graph(); bool inj = false, cpu = false, threw = false;
        try { run_block(g, 0, inj, STD_EXCEPTION, cpu); } catch (const std::exception &) { threw = true; }
        CHECK(threw && g.nodes[0].op == OP_MUL_MAT_ID && g.nodes[1].op == OP_GLU);
    }
    // the undo runs when an exception unwinds a scope, and not twice after run()
    {
        std::vector<int> v = { 1, 2, 3 };
        int undone = 0;
        try {
            auto u = mcguard::make_undo([&] { v.assign(3, 0); undone++; });
            throw std::runtime_error("x");
        } catch (const std::exception &) {}
        CHECK(undone == 1 && v[0] == 0);
        {
            auto u = mcguard::make_undo([&] { undone++; });
            u.run();
        }
        CHECK(undone == 2);
        // an undo that throws inside the destructor must not terminate
        bool alive = true;
        try { auto u = mcguard::make_undo([&] { throw std::runtime_error("y"); }); } catch (...) { alive = false; }
        CHECK(alive);
    }
    if (fails) {
        fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    printf("gpu-fallback checks passed\n");
    return 0;
}
