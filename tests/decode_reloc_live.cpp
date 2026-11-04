// Tiny-model check for llama.cpp patch 0070 (LLAMA_DECODE_RELOC, DECODE-630). Not part of tools/run_quick_tests.sh: it needs a patched llama.cpp
// build (CUDA) and a small MoE GGUF. tests/decode_reloc_live.sh builds and runs it.
//   decode_reloc_live MODEL.gguf [twoctx|late2|long]
// Prints token ids only (greedy), so two runs (LLAMA_DECODE_RELOC unset / =1) can be compared with diff:
//   phase 1: prompt of 300 tokens in chunks of 256, then 30 single-token steps (the decode phase starts after LLAMA_DECODE_RELOC_AFTER of them)
//   phase 2: save the sequence state (warm-session slot save), 20 more tokens (G2), clear, restore, 20 tokens again (G3); G3 must equal G2
//   phase 3: a 100-token batch (more than the decode chunk: the layers move back), then 10 tokens
// "twoctx": a second context on the same model is created first (MTP / draft stand-in); the patch must refuse, the tokens must not change.
// "late2":  the second context is created AFTER the first one moved the layers (prompt, 30 steps), runs its own prompt and 30 steps, then the first
//           context goes on for 20 steps: its reloc_step sees two live contexts and moves back; no crash, tokens equal to the run with the switch off.
// "long":   greedy identity, 4 prompts (50 / 700 / 300 / 1500 tokens, one batch each) x 600 generated tokens = 2400 tokens, memory cleared between
//           prompts, default LLAMA_DECODE_RELOC_UB (512) and AFTER (64): the 700 and 1500 token batches arrive in the decode phase (more than 512:
//           the layers move back). Prints the token ids and the text (non-printable bytes escaped) per prompt.
// LLAMA_DECODE_RELOC_FAIL=1|2 (test only, in the patch) makes the first / second reserve of the move throw: default mode, tokens must equal the off run.
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

static int argmax(const float * l, int n) { int b = 0; for (int i = 1; i < n; i++) if (l[i] > l[b]) b = i; return b; }

static int step(llama_context * ctx, std::vector<llama_token> & t, int n_vocab, std::vector<int> & out) {
    llama_batch b = llama_batch_get_one(t.data(), (int32_t) t.size());
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return -1; }
    int id = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
    out.push_back(id);
    return id;
}

static void dump(const char * n, const std::vector<int> & v) { printf("%s:", n); for (int x : v) printf(" %d", x); printf("\n"); }

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [twoctx|late2|long]\n", argv[0]); return 2; }
    const bool twoctx = argc > 2 && !strcmp(argv[2], "twoctx");
    const bool late2  = argc > 2 && !strcmp(argv[2], "late2");
    const bool lng    = argc > 2 && !strcmp(argv[2], "long");
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    // the experts of layers 0..2 stay in host memory (plain CPU buffer, as --n-cpu-moe 3 does)
    llama_model_tensor_buft_override ov[2] = { { "blk\\.[0-2]\\.ffn_.*_exps", ggml_backend_cpu_buffer_type() }, { nullptr, nullptr } };
    mp.tensor_buft_overrides = ov;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 256; cp.n_ubatch = 256; cp.n_seq_max = 1;
    if (lng) { cp.n_ctx = 4096; cp.n_batch = 2048; cp.n_ubatch = 2048; }
    llama_context * ctx2 = twoctx ? llama_init_from_model(model, cp) : nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    auto tok = [&](int i) { return (llama_token) (5 + (i * 7) % 50 % n_vocab); };
    if (lng) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int lens[4] = { 50, 700, 300, 1500 };
        const int n_gen = 600;
        for (int k = 0; k < 4; k++) {
            llama_memory_clear(llama_get_memory(ctx), true);
            const int seed = 3 + 4 * k;
            std::vector<llama_token> t(lens[k]);
            for (int i = 0; i < lens[k]; i++) t[i] = (llama_token) ((5 + (i * seed + (i * i) % 97) % 400) % n_vocab);
            std::vector<int> g, sink;
            int last = step(ctx, t, n_vocab, sink);
            if (last < 0) return 1;
            g.push_back(last);
            for (int i = 1; i < n_gen; i++) { std::vector<llama_token> u = { (llama_token) last }; last = step(ctx, u, n_vocab, g); if (last < 0) return 1; }
            char name[16]; snprintf(name, sizeof name, "P%d", k);
            dump(name, g);
            printf("T%d: ", k);
            for (int id : g) {
                char buf[256]; int n = llama_token_to_piece(vocab, id, buf, sizeof buf, 0, true);
                for (int j = 0; j < n; j++) { unsigned char c = buf[j]; if (c >= 32 && c < 127 && c != 92) putchar(c); else printf("\\x%02x", c); }
            }
            printf("\n");
        }
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }
    if (late2) {
        std::vector<int> a1, a2, a3, a4, sk;
        int p = 0, l = -1;
        while (p < 300) {
            int n = std::min(256, 300 - p);
            std::vector<llama_token> t(n);
            for (int i = 0; i < n; i++) t[i] = tok(p + i);
            l = step(ctx, t, n_vocab, sk);
            if (l < 0) return 1;
            p += n;
        }
        for (int i = 0; i < 30; i++) { std::vector<llama_token> t = { (llama_token) l }; l = step(ctx, t, n_vocab, a1); if (l < 0) return 1; }
        llama_context * c2 = llama_init_from_model(model, cp);   // the layers of ctx are moved now
        if (!c2) { fprintf(stderr, "second context failed\n"); return 1; }
        int l2 = -1; p = 0;
        while (p < 300) {
            int n = std::min(256, 300 - p);
            std::vector<llama_token> t(n);
            for (int i = 0; i < n; i++) t[i] = tok(p + i);
            l2 = step(c2, t, n_vocab, sk);
            if (l2 < 0) return 1;
            p += n;
        }
        for (int i = 0; i < 30; i++) { std::vector<llama_token> t = { (llama_token) l2 }; l2 = step(c2, t, n_vocab, a2); if (l2 < 0) return 1; }
        for (int i = 0; i < 20; i++) { std::vector<llama_token> t = { (llama_token) l }; l = step(ctx, t, n_vocab, a3); if (l < 0) return 1; }
        for (int i = 0; i < 10; i++) { std::vector<llama_token> t = { (llama_token) l2 }; l2 = step(c2, t, n_vocab, a4); if (l2 < 0) return 1; }
        dump("A1", a1); dump("A2", a2); dump("A3", a3); dump("A4", a4);
        printf("ctx2_matches_ctx1: %s\n", a1 == a2 ? "yes" : "NO");
        llama_free(c2);
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }
    std::vector<int> g1, g2, g3, g4, sink;
    int pos = 0, last = -1;
    while (pos < 300) {                                       // phase 1: prompt chunks
        int n = std::min(256, 300 - pos);
        std::vector<llama_token> t(n);
        for (int i = 0; i < n; i++) t[i] = tok(pos + i);
        last = step(ctx, t, n_vocab, sink);
        pos += n;
    }
    for (int i = 0; i < 30; i++) { std::vector<llama_token> t = { (llama_token) last }; last = step(ctx, t, n_vocab, g1); }
    // phase 2: slot save / restore in the decode phase
    std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
    llama_state_seq_get_data(ctx, state.data(), state.size(), 0);
    const int last_saved = last;
    for (int i = 0; i < 20; i++) { std::vector<llama_token> t = { (llama_token) last }; last = step(ctx, t, n_vocab, g2); }
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_state_seq_set_data(ctx, state.data(), state.size(), 0);
    last = last_saved;
    for (int i = 0; i < 20; i++) { std::vector<llama_token> t = { (llama_token) last }; last = step(ctx, t, n_vocab, g3); }
    // phase 3: a new prompt chunk bigger than the decode chunk, then decode again
    {
        std::vector<llama_token> t(100);
        for (int i = 0; i < 100; i++) t[i] = tok(1000 + i);
        last = step(ctx, t, n_vocab, sink);
    }
    for (int i = 0; i < 10; i++) { std::vector<llama_token> t = { (llama_token) last }; last = step(ctx, t, n_vocab, g4); }

    dump("G1", g1); dump("G2", g2); dump("G3", g3); dump("G4", g4);
    printf("restore_matches: %s\n", g2 == g3 ? "yes" : "NO");
    llama_free(ctx);
    if (ctx2) llama_free(ctx2);
    llama_model_free(model);
    return g2 == g3 ? 0 : 3;
}
