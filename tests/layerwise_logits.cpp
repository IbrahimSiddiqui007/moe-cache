// Logit check for LLAMA_LAYERWISE_PREFILL (patches/llama.cpp/0050). Build against the patched llama.cpp tree, from its root:
//   g++ -O2 -std=c++17 -Iinclude -Iggml/include /path/to/layerwise_logits.cpp -Lbuild/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build/bin -o build/bin/layerwise_logits
// Run (the same model and arguments twice, the second time with the switch):
//   layerwise_logits run MODEL OUT.bin [n_tokens=300] [ub=32] [batch=256] [all|last=all] [n_gpu_layers=0] [seed=1]
//   LLAMA_LAYERWISE_PREFILL=3 layerwise_logits run MODEL OUT_lw.bin 300 32 256 all
//   layerwise_logits cmp OUT.bin OUT_lw.bin
// "run" decodes n_tokens random tokens in batches of `batch` (so a batch holds batch/ub chunks), then writes the logits of the
// marked tokens (all, or the last of the prompt) as raw floats. "cmp" prints the largest difference, the NMSE and how many rows
// have the same best token.
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int run(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: run MODEL OUT.bin [n_tokens ub batch all|last n_gpu_layers seed]\n");
        return 2;
    }
    const char * path   = argv[2];
    const char * out    = argv[3];
    const int n_tokens  = argc > 4 ? atoi(argv[4]) : 300;
    const int ub        = argc > 5 ? atoi(argv[5]) : 32;
    const int n_batch   = argc > 6 ? atoi(argv[6]) : 256;
    const bool all      = argc > 7 ? strcmp(argv[7], "last") != 0 : true;
    const int ngl       = argc > 8 ? atoi(argv[8]) : 0;
    const int seed      = argc > 9 ? atoi(argv[9]) : 1;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", path);
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = (uint32_t) n_tokens + 16;
    cp.n_batch  = (uint32_t) n_batch;
    cp.n_ubatch = (uint32_t) ub;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot create the context\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::mt19937 rng(seed);
    std::vector<llama_token> tok(n_tokens);
    for (auto & t : tok) {
        t = (llama_token) (rng() % (uint32_t) n_vocab);
    }

    std::vector<float> all_logits;
    llama_batch b = llama_batch_init(n_batch, 0, 1);
    for (int i0 = 0; i0 < n_tokens; i0 += n_batch) {
        const int n = std::min(n_batch, n_tokens - i0);
        b.n_tokens = n;
        for (int i = 0; i < n; i++) {
            b.token[i]     = tok[i0 + i];
            b.pos[i]       = i0 + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = all || (i0 + i == n_tokens - 1);
        }
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "llama_decode failed at %d\n", i0);
            return 1;
        }
        for (int i = 0; i < n; i++) {
            if (b.logits[i]) {
                const float * l = llama_get_logits_ith(ctx, i);
                all_logits.insert(all_logits.end(), l, l + n_vocab);
            }
        }
    }
    FILE * f = fopen(out, "wb");
    fwrite(&n_vocab, sizeof(int), 1, f);
    fwrite(all_logits.data(), sizeof(float), all_logits.size(), f);
    fclose(f);
    printf("wrote %zu rows of %d logits to %s\n", all_logits.size() / (size_t) n_vocab, n_vocab, out);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static std::vector<float> load(const char * path, int & n_vocab) {
    FILE * f = fopen(path, "rb");
    std::vector<float> v;
    if (!f || fread(&n_vocab, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    float x;
    while (fread(&x, sizeof(float), 1, f) == 1) {
        v.push_back(x);
    }
    fclose(f);
    return v;
}

static int cmp(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: cmp A.bin B.bin\n");
        return 2;
    }
    int va = 0, vb = 0;
    auto a = load(argv[2], va);
    auto b = load(argv[3], vb);
    if (va != vb || a.size() != b.size() || a.empty()) {
        printf("DIFFERENT SHAPE: %zu vs %zu floats\n", a.size(), b.size());
        return 1;
    }
    const size_t rows = a.size() / (size_t) va;
    double se = 0, ss = 0, worst = 0;
    size_t same = 0, nan = 0;
    for (size_t r = 0; r < rows; r++) {
        size_t ia = 0, ib = 0;
        for (int j = 0; j < va; j++) {
            const double x = a[r * va + j], y = b[r * va + j];
            if (std::isnan(x) || std::isnan(y)) { nan++; continue; }
            se += (x - y) * (x - y);
            ss += x * x;
            worst = std::max(worst, std::fabs(x - y));
            if (a[r * va + j] > a[r * va + ia]) ia = (size_t) j;
            if (b[r * va + j] > b[r * va + ib]) ib = (size_t) j;
        }
        same += ia == ib;
    }
    printf("rows %zu  max|diff| %.3g  NMSE %.3g  same best token %zu/%zu  NaN %zu\n", rows, worst, ss > 0 ? se / ss : 0.0, same, rows, nan);
    return nan ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && !strcmp(argv[1], "run")) return run(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "cmp")) return cmp(argc, argv);
    fprintf(stderr, "usage: run ... | cmp A B\n");
    return 2;
}
