// Layer-wise prefill check (PREFILL-638): decode one long prompt of pseudo-random tokens with a small n_ubatch and write the logits of its last token and the
// next 8 greedy steps to a file; run it with and without LLAMA_LAYERWISE_PREFILL=N (separate processes) and compare the two files (tests/layerwise_compare.py).
//   g++ -std=c++17 -O1 -I$LLAMA/include -I$LLAMA/ggml/include tests/layerwise_check.cpp -L$LLAMA/build/bin -lllama -lggml -lggml-base -o layerwise_check
//   LD_LIBRARY_PATH=$LLAMA/build/bin ./layerwise_check MODEL.gguf OUT.bin [n_tokens=600] [n_ubatch=128]
#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL OUT [n_tokens] [n_ubatch]\n", argv[0]); return 2; }
    const int n_tok = argc > 3 ? atoi(argv[3]) : 600, n_ub = argc > 4 ? atoi(argv[4]) : 128;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_tok + 64; cp.n_batch = n_tok + 64; cp.n_ubatch = n_ub;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create the context\n"); return 1; }
    std::vector<llama_token> toks(n_tok);
    unsigned s = 12345;
    for (auto & t : toks) { s = s * 1664525u + 1013904223u; t = (llama_token) (3 + (s >> 8) % (unsigned) (n_vocab - 3)); }
    FILE * f = fopen(argv[2], "wb");
    llama_batch b = llama_batch_get_one(toks.data(), n_tok);
    int rc = llama_decode(ctx, b);
    if (rc != 0) { fprintf(stderr, "decode failed: %d\n", rc); return 1; }
    for (int step = 0; step < 9; ++step) {
        const float * lg = llama_get_logits(ctx);
        fwrite(lg, sizeof(float), (size_t) n_vocab, f);
        const llama_token next = (llama_token) (std::max_element(lg, lg + n_vocab) - lg);
        printf("step %d best token %d\n", step, next);
        llama_token one = next;
        llama_batch b1 = llama_batch_get_one(&one, 1);
        if (llama_decode(ctx, b1) != 0) { fprintf(stderr, "decode failed at step %d\n", step); return 1; }
    }
    fclose(f);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
