// Bit-exactness and speed check of the plugin's Q2_0 kernel.
//   g++ -O2 -std=c++17 -Isrc tests/kernel_check.cpp -o /tmp/kernel_check -ldl && /tmp/kernel_check [/path/to/libggml-cpu.so.0]
//   Windows (Windows CI): x86_64-w64-mingw32-g++ -O2 -std=c++17 -Isrc tests/kernel_check.cpp -o kernel_check.exe -static (built-in reference only)
// With a library path it compares against the real ggml_vec_dot_q2_0_q8_0 of that library; without one (or when it cannot be loaded) it compares
// against a plain C++ copy of llama.cpp's loop (same operations, same order, fused multiply-add as the compiled library uses).
#include "fast_q2_0.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <random>
#include <vector>

using dot_fn = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

// llama.cpp's generic loop, written out; the compiled library fuses the two multiply-adds, so this does too. target("fma"): the fused multiply-add
// instruction, not the C runtime's fmaf (Windows CI's UCRT fmaf is not exactly fused: half the trials differed from every kernel by the same count)
__attribute__((target("fma"))) static void ref_loop(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
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
#ifndef _WIN32
    if (argc > 1) {
        void * h = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
        if (h) {
            if (auto init = (void (*)()) dlsym(h, "ggml_cpu_init")) init();   // fills the fp16 -> fp32 table the library kernel reads
            ref = (dot_fn) dlsym(h, "ggml_vec_dot_q2_0_q8_0");
        }
    }
#else
    (void) argc; (void) argv;
#endif
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
    long bad_fma = 0, bad_nofma = 0, bad_perm = 0, bad_ys = 0, bad_up_ref = 0, total = 0;
    for (int n : { 64, 128, 192, 640, 2560, 4096, 12288 }) {      // 192: an odd block count (the tail of upstream's two-accumulator loop)
        const int nb = n / 64;
        std::vector<fastk::block_q2_0> x(nb);
        std::vector<fastk::block_q8_0> y(nb * 2);
        for (int trial = 0; trial < 3000; trial++) {
            for (auto & b : x) { b.d = (uint16_t) (0x2000 + rng() % 0x1800); for (auto & q : b.qs) q = (uint8_t) rng(); }   // positive, normal halves
            for (auto & b : y) { b.d = (uint16_t) (0x2000 + rng() % 0x1800); for (auto & q : b.qs) q = (int8_t) rng(); }
            if (trial % 7 == 0) { for (auto & b : y) for (auto & q : b.qs) q = (rng() & 1) ? 127 : -128; }              // extreme values
            float r = 0, a = 0, c = 0, pm = 0;
            ref(n, &r, 0, x.data(), 0, y.data(), 0, 1);
            {
                std::vector<fastk::block_q8_0_perm> yp(nb * 2);
                fastk::permute_q8_0(y.data(), yp.data(), nb * 2);
                fastk::vec_dot_q2_0_perm_avx2(n, &pm, 0, x.data(), 0, yp.data(), 0, 1);
                uint32_t ri2, pi2; memcpy(&ri2, &r, 4); memcpy(&pi2, &pm, 4); bad_perm += ri2 != pi2;
            }
            {   // N20: the column-sum kernel against the written-out upstream #30447 kernel, and that one against the reference
                float u = 0, v = 0;
                std::vector<fastk::block_q8_0_ys> ys(nb * 2);
                fastk::prepare_q8_0_ys(y.data(), ys.data(), nb * 2);
                fastk::vec_dot_q2_0_up_avx2(n, &u, x.data(), y.data());
                fastk::vec_dot_q2_0_ys_avx2(n, &v, 0, x.data(), 0, ys.data(), 0, 1);
                uint32_t ui, vi, ri3; memcpy(&ui, &u, 4); memcpy(&vi, &v, 4); memcpy(&ri3, &r, 4); bad_ys += ui != vi; bad_up_ref += ui != ri3;
            }
            fastk::vec_dot_q2_0_q8_0_avx2<true>(n, &a, x.data(), y.data());
            fastk::vec_dot_q2_0_q8_0_avx2<false>(n, &c, x.data(), y.data());
            uint32_t ri, ai, ci; memcpy(&ri, &r, 4); memcpy(&ai, &a, 4); memcpy(&ci, &c, 4);
            bad_fma += ri != ai; bad_nofma += ri != ci; total++;
        }
    }
    printf("trials %ld: mismatches with fused multiply-add %ld, with separate multiply/add %ld\n", total, bad_fma, bad_nofma);
    printf("permuted-activation kernel: %ld mismatches in %ld trials\n", bad_perm, total);
    printf("column-sum kernel vs upstream #30447 kernel: %ld mismatches in %ld trials\n", bad_ys, total);
    printf("upstream #30447 kernel vs the reference: %ld mismatches (0 = the library is #30447; the plain loop differs)\n", bad_up_ref);
    const bool ref_plain = !(bad_fma && bad_nofma), ref_up = bad_up_ref == 0;
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
    {
        std::vector<fastk::block_q8_0_perm> yp(nb * 2);
        fastk::permute_q8_0(y.data(), yp.data(), nb * 2);
        auto t3 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) { fastk::vec_dot_q2_0_perm_avx2(n, &s, 0, x.data(), 0, yp.data(), 0, 1); sink += s; }
        const double c = std::chrono::duration<double>(std::chrono::steady_clock::now() - t3).count();
        printf("n=%d: permuted-activation kernel %.1f ns per dot (%.1fx the library, %.1fx the first kernel)\n", n, c / reps * 1e9, a / c, b / c);
    }
    {
        std::vector<fastk::block_q8_0_ys> ys(nb * 2);
        fastk::prepare_q8_0_ys(y.data(), ys.data(), nb * 2);
        auto t4 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) { fastk::vec_dot_q2_0_up_avx2(n, &s, x.data(), y.data()); sink += s; }
        auto t5 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) { fastk::vec_dot_q2_0_ys_avx2(n, &s, 0, x.data(), 0, ys.data(), 0, 1); sink += s; }
        const double u = std::chrono::duration<double>(t5 - t4).count(), v = std::chrono::duration<double>(std::chrono::steady_clock::now() - t5).count();
        printf("n=%d: upstream #30447 kernel %.1f ns per dot, column-sum kernel %.1f ns per dot (%.2fx)\n", n, u / reps * 1e9, v / reps * 1e9, u / v);
    }
    // the reference must be one of the two kinds the plugin knows; each kind's own kernels must match it bit for bit
    return (bad_ys || (!ref_plain && !ref_up) || (ref_plain && bad_perm)) ? 1 : 0;
}
