// CACHE-694: what the cache keeps after a model is unloaded, and what stays valid while `nodes` grows. No model, no GPU: the plugin is included whole and
// a state_t is driven with hand-made tensors over a small temp file. Needs the ggml headers only (the ggml calls are never reached, so the link ignores
// the undefined ones):
//   g++ -std=c++17 -Wno-unused-function -Ithird_party/ggml-v3/include -Ithird_party/ggml-v3/src -Isrc tests/unload_state_test.cpp -o /tmp/unload_state_test -lpthread -ldl -no-pie -Wl,--unresolved-symbols=ignore-all && /tmp/unload_state_test
#include "../src/moe_cache_plugin.cpp"
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

extern "C" const char * ggml_version(void) { return "test"; }
extern "C" const char * ggml_commit(void) { return "test"; }

static const size_t NB2 = 8192;                      // bytes per expert

// one expert tensor "blk.<layer>.<kind>.weight" of n_exp experts, data pointing into `buf`
static ggml_tensor make_tensor(const char * name, int n_exp, char * data) {
    ggml_tensor t;
    memset(&t, 0, sizeof(t));
    snprintf(t.name, sizeof(t.name), "%s", name);
    t.ne[0] = 64; t.ne[1] = 64; t.ne[2] = n_exp; t.ne[3] = 1;
    t.nb[0] = 1; t.nb[1] = 128; t.nb[2] = NB2; t.nb[3] = NB2 * (size_t) n_exp;
    t.data = data;
    return t;
}

int main() {
    const std::string path = std::string("/tmp/unload_state_model.") + std::to_string((long) getpid()) + ".bin";
    std::vector<char> file(NB2 * 64, 0);
    for (size_t i = 0; i < file.size(); i++) { file[i] = (char) (i * 7 + 3); }
    { FILE * f = fopen(path.c_str(), "wb"); if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return 2; } fwrite(file.data(), 1, file.size(), f); fclose(f); }

    auto * sp = new state_t();
    state_t & s = *sp;
    s.win_view = false;
    s.shard_paths = { path };
    s.fds_buf.resize(1);
    s.limit_bytes = (size_t) 64 << 30;

    // model A: two layers, 8 experts each, in buffer bufA; model B: one layer, 24 experts, in bufB
    std::vector<char> bufA(file.begin(), file.begin() + (long) (NB2 * 16)), bufB(file.begin() + (long) (NB2 * 32), file.begin() + (long) (NB2 * 56));
    ggml_tensor a0 = make_tensor("blk.0.ffn_up_exps.weight", 8, bufA.data());
    ggml_tensor a1 = make_tensor("blk.1.ffn_up_exps.weight", 8, bufA.data() + NB2 * 8);
    ggml_tensor b0 = make_tensor("blk.0.ffn_up_exps.weight", 24, bufB.data());
    s.file_tensors["blk.0.ffn_up_exps.weight"] = { 0, NB2 * 8 };
    s.file_shard["blk.0.ffn_up_exps.weight"] = 0;
    s.file_tensors["blk.1.ffn_up_exps.weight"] = { NB2 * 8, NB2 * 8 };
    s.file_shard["blk.1.ffn_up_exps.weight"] = 0;

    CHECK(s.register_tensor(&a0, a0.data) == 0);
    // the address of a flag the executor / wait_experts hold while a graph runs (mc_exec.inl:388, :573; mc_backend.inl:362)
    const bool * flag = &s.nodes[0].loading;
    CHECK(s.register_tensor(&a1, a1.data) == 1);
    CHECK(&s.nodes[0].loading == flag);              // CACHE-694 part 1: registering more tensors must not move the nodes that are already in use
    for (int i = 0; i < 200; i++) { s.nodes.resize(s.nodes.size() + 64); s.node_owner.resize(s.nodes.size(), 1); }
    CHECK(&s.nodes[0].loading == flag);

    // the tuner fixes the ceiling from the registered tensors (here 16 experts), and the warm start runs once
    s.tune();
    CHECK(s.max_budget == NB2 * 16);
    s.preload();
    CHECK(s.preload_done);
    s.session.assign(2, std::vector<double>(8, 1.0));

    // unload model A: its buffer is freed
    s.forget_range(bufA.data(), bufA.size());
    // part 2: nothing of A is left, so the next model starts from a clean state
    CHECK(s.max_budget == 0);
    CHECK(!s.preload_done);
    CHECK(s.session.size() == 2);                    // kept: finish() still saves the picks at exit

    // reload: model B (a different size) registers afresh and the tuner sees B's size, not A's
    s.file_tensors["blk.0.ffn_up_exps.weight"] = { NB2 * 32, NB2 * 24 };
    CHECK(s.register_tensor(&b0, b0.data) == 2);
    s.tune();
    CHECK(s.max_budget == NB2 * 24);
    s.max_budget = 0;

    // a model that is only partly unloaded keeps its state (another buffer of the same model is still alive)
    s.preload_done = true;
    s.max_budget = 12345;
    s.forget_range(bufA.data(), 1);                  // a range that holds none of B's tensors: B is still registered
    CHECK(s.max_budget == 12345 && s.preload_done);

    if (fails) { fprintf(stderr, "%d checks failed\n", fails); remove(path.c_str()); return 1; }
    printf("unload-state checks passed\n");
    fflush(stdout);
    remove(path.c_str());
    _exit(0);                                       // skip the state's destructor (threads, files): the checks are done
}
