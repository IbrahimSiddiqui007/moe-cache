// Bit-exactness and speed check of the plugin's Q2_0 kernel.
//   g++ -O2 -std=c++17 -Isrc tests/kernel_check.cpp -o /tmp/kernel_check -ldl && /tmp/kernel_check [/path/to/libggml-cpu.so.0]
// With a library path it compares against the real ggml_vec_dot_q2_0_q8_0 of that library; without one (or when it cannot be loaded) it compares
// against a plain C++ copy of llama.cpp's loop (same operations, same order, fused multiply-add as the compiled library uses).
#include "fast_q2_0.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <random>
#include <vector>

using dot_fn = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

// llama.cpp's generic loop, written out; the compiled library fuses the two multiply-adds, so this does too
static void ref_loop(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    const fastk::block_q2_0 * x = (const fastk::block_q2_0 *) vx;
    const fastk::block_q8_0 * y = (const fastk::block_q8_0 *) vy;
    float sumf = 0.0f;
    for (int i = 0; i < n / 64; i++) {
        const float d0 = fastk::half_to_float(x[i].d);
        float sumi = 0.0f;
        for (int k = 0; k < 2; k++) {
            const fastk::block_q8_0 & yb = y[i * 2 + k];
            const float d1 = fastk::half_to_float(yb.d);
            int sb = 0;
            for (int b = 0; b < 8; b++) {
                const uint8_t byte = x[i].qs[k * 8 + b];
                for (int j = 0; j < 4; j++) {
                    sb += ((int) ((byte >> (2 * j)) & 3) - 1) * yb.qs[b * 4 + j];
                }
            }
            sumi = __builtin_fmaf(d1, (float) sb, sumi);
        }
        sumf = __builtin_fmaf(d0, sumi, sumf);
    }
    *s = sumf;
}

__attribute__((target("f16c"))) static float hw_half(unsigned short h) { return _cvtsh_ss(h); }
int main(int argc, char ** argv) {
    dot_fn ref = nullptr;
    if (argc > 1) {
        void * h = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
        if (h) {
            if (auto init = (void (*)()) dlsym(h, "ggml_cpu_init")) init();   // fills the fp16 -> fp32 table the library kernel reads
            ref = (dot_fn) dlsym(h, "ggml_vec_dot_q2_0_q8_0");
        }
    }
    printf("reference: %s\n", ref ? "the real library function" : "built-in copy of llama.cpp's loop");
    if (!ref) ref = ref_loop;
    // the half -> float conversion must match the hardware one for all 65536 values
    {
        long bad = 0;
        for (uint32_t v = 0; v < 65536; v++) {
            float a = fastk::half_to_float((uint16_t) v), b = hw_half((unsigned short) v);
            uint32_t ai, bi; memcpy(&ai, &a, 4); memcpy(&bi, &b, 4); bad += ai != bi;
        }
        printf("half -> float: %ld of 65536 values differ from the hardware conversion\n", bad);
    }
    std::mt19937_64 rng(12345);
    long bad_fma = 0, bad_nofma = 0, total = 0;
    for (int n : { 64, 128, 640, 2560, 4096, 12288 }) {
        const int nb = n / 64;
        std::vector<fastk::block_q2_0> x(nb);
        std::vector<fastk::block_q8_0> y(nb * 2);
        for (int trial = 0; trial < 3000; trial++) {
            for (auto & b : x) { b.d = (uint16_t) (0x2000 + rng() % 0x1800); for (auto & q : b.qs) q = (uint8_t) rng(); }   // positive, normal halves
            for (auto & b : y) { b.d = (uint16_t) (0x2000 + rng() % 0x1800); for (auto & q : b.qs) q = (int8_t) rng(); }
            if (trial % 7 == 0) { for (auto & b : y) for (auto & q : b.qs) q = (rng() & 1) ? 127 : -128; }              // extreme values
            float r = 0, a = 0, c = 0;
            ref(n, &r, 0, x.data(), 0, y.data(), 0, 1);
            fastk::vec_dot_q2_0_q8_0_avx2<true>(n, &a, x.data(), y.data());
            fastk::vec_dot_q2_0_q8_0_avx2<false>(n, &c, x.data(), y.data());
            uint32_t ri, ai, ci; memcpy(&ri, &r, 4); memcpy(&ai, &a, 4); memcpy(&ci, &c, 4);
            bad_fma += ri != ai; bad_nofma += ri != ci; total++;
        }
    }
    printf("trials %ld: mismatches with fused multiply-add %ld, with separate multiply/add %ld\n", total, bad_fma, bad_nofma);
    // speed: one expert-row sized dot (n = 2560) repeated, data in L2
    const int n = 2560, nb = n / 64, reps = 400000;
    std::vector<fastk::block_q2_0> x(nb); std::vector<fastk::block_q8_0> y(nb * 2);
    for (auto & b : x) { b.d = 0x3000; for (auto & q : b.qs) q = (uint8_t) rng(); }
    for (auto & b : y) { b.d = 0x3000; for (auto & q : b.qs) q = (int8_t) rng(); }
    volatile float sink = 0; float s = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) { ref(n, &s, 0, x.data(), 0, y.data(), 0, 1); sink += s; }
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) { fastk::vec_dot_q2_0_q8_0_avx2<true>(n, &s, x.data(), y.data()); sink += s; }
    auto t2 = std::chrono::steady_clock::now();
    const double a = std::chrono::duration<double>(t1 - t0).count(), b = std::chrono::duration<double>(t2 - t1).count();
    printf("n=%d: library %.1f ns per dot, avx2 %.1f ns per dot, speed-up %.1fx\n", n, a / reps * 1e9, b / reps * 1e9, a / b);
    return (bad_fma && bad_nofma) ? 1 : 0;
}
