// How concentrated is the work inside one expert? (neuron-level sparsity probe; measures only, changes nothing)
//   g++ -O2 -std=c++17 -fPIC -shared -Ithird_party/ggml-v3/include tests/sparsity_probe.cpp -o /tmp/sparsity_probe.so -ldl
//   LD_PRELOAD=/tmp/sparsity_probe.so SPARSITY_OUT=/tmp/sparsity.txt llama-server -m MODEL.gguf ...   (then send some requests, stop the server with SIGINT)
// For every expert activation vector h = act(gate x) * (up x) of every MoE layer (the "ffn_moe_swiglu" tensors, shape [n_ff, n_expert_used, n_tokens])
// it records which fraction of the squared magnitude sits in the largest p % of the neurons, for p = 5, 10, 25, 50. If 25 % of the neurons carry 90 %
// of the energy, a sparse computation could skip 75 % of the rows of gate, up and down for a small error. Decode steps (1 token) and prompt batches
// are counted separately. Energy is a proxy: it does not weigh a neuron by the size of its down-projection column.
// It interposes ggml_backend_sched_new to install an evaluation callback, which makes the scheduler split the graph at every such node: it is slow
// (never use it for timing).
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>
#include "ggml.h"
#include "ggml-backend.h"

namespace {
const double PCT[4] = { 0.05, 0.10, 0.25, 0.50 };
struct acc_t { double sum[4] = {0, 0, 0, 0}; double n = 0; };
std::mutex mu;
std::map<int, acc_t> dec_layer, pre_layer;      // layer -> accumulators
acc_t dec_all, pre_all;
std::vector<float> host;

int layer_of(const char * name) {
    const char * d = strrchr(name, '-');
    return d ? atoi(d + 1) : -1;
}

void add(acc_t & a, const double f[4]) {
    for (int i = 0; i < 4; i++) a.sum[i] += f[i];
    a.n += 1;
}

bool eval_cb(struct ggml_tensor * t, bool ask, void *) {
    if (strncmp(t->name, "ffn_moe_swiglu", 14) != 0) return ask ? false : true;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || t->ne[3] != 1) return true;
    const int64_t nff = t->ne[0], nsel = t->ne[1], ntok = t->ne[2];
    std::lock_guard<std::mutex> lk(mu);
    host.resize((size_t) (nff * nsel * ntok));
    ggml_backend_tensor_get(t, host.data(), 0, host.size() * sizeof(float));
    const int il = layer_of(t->name);
    std::vector<float> e((size_t) nff);
    for (int64_t k = 0; k < nsel * ntok; k++) {
        const float * h = host.data() + (size_t) k * nff;
        double total = 0;
        for (int64_t i = 0; i < nff; i++) { e[i] = h[i] * h[i]; total += e[i]; }
        if (!(total > 0)) continue;
        std::sort(e.begin(), e.end(), std::greater<float>());
        double f[4]; double run = 0; int64_t pos = 0;
        for (int q = 0; q < 4; q++) {
            const int64_t upto = std::max<int64_t>(1, (int64_t) (PCT[q] * nff));
            while (pos < upto) run += e[(size_t) pos++];
            f[q] = run / total;
        }
        if (ntok == 1) { add(dec_all, f); add(dec_layer[il], f); } else { add(pre_all, f); add(pre_layer[il], f); }
    }
    return true;
}

void report() {
    const char * path = getenv("SPARSITY_OUT");
    FILE * o = path ? fopen(path, "w") : stderr;
    if (!o) o = stderr;
    auto print = [&](const char * what, const acc_t & a, const std::map<int, acc_t> & per) {
        if (a.n == 0) return;
        fprintf(o, "%s: %.0f expert activations\n  share of squared magnitude in the top  5%% / 10%% / 25%% / 50%% of neurons: %.3f %.3f %.3f %.3f\n", what, a.n,
                a.sum[0] / a.n, a.sum[1] / a.n, a.sum[2] / a.n, a.sum[3] / a.n);
        double lo = 2, hi = -1; int lo_l = -1, hi_l = -1;
        for (auto & kv : per) {
            const double v = kv.second.sum[2] / kv.second.n;     // top 25 %
            if (v < lo) { lo = v; lo_l = kv.first; }
            if (v > hi) { hi = v; hi_l = kv.first; }
        }
        fprintf(o, "  top-25%% share by layer: lowest %.3f (layer %d), highest %.3f (layer %d)\n", lo, lo_l, hi, hi_l);
    };
    print("decode (1 token)", dec_all, dec_layer);
    print("prompt batches", pre_all, pre_layer);
    if (o != stderr) fclose(o);
}

struct reporter_t { ~reporter_t() { report(); } } reporter;
}

// llama.cpp installs its own (null) callback right after creating the scheduler: keep ours.
extern "C" void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback, void *) {
    typedef void (*fn_t)(ggml_backend_sched_t, ggml_backend_sched_eval_callback, void *);
    static fn_t orig = (fn_t) dlsym(RTLD_NEXT, "ggml_backend_sched_set_eval_callback");
    orig(sched, eval_cb, nullptr);
}

extern "C" ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload) {
    typedef ggml_backend_sched_t (*fn_t)(ggml_backend_t *, ggml_backend_buffer_type_t *, int, size_t, bool, bool);
    static fn_t orig = (fn_t) dlsym(RTLD_NEXT, "ggml_backend_sched_new");
    return orig(backends, bufts, n_backends, graph_size, parallel, op_offload);
}
