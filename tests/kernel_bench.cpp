// Speed of llama.cpp's CPU dot product for every quantisation type (one core, data in cache): where are the slow kernels?
//   g++ -O2 -std=c++17 -Isrc tests/kernel_bench.cpp -o /tmp/kernel_bench -ldl && /tmp/kernel_bench /usr/lib/llama.cpp-cuda
#include "fast_q2_0.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <random>
#include <string>
#include <vector>

struct traits_cpu { void (*from_float)(const float *, void *, int64_t); void (*vec_dot)(int, float *, size_t, const void *, size_t, const void *, size_t, int); int vec_dot_type; int64_t nrows; };
int main(int argc, char ** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    void * hb = dlopen((dir + "/libggml-base.so.0").c_str(), RTLD_NOW | RTLD_GLOBAL);
    void * hc = dlopen((dir + "/libggml-cpu.so.0").c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!hb || !hc) { fprintf(stderr, "cannot load libggml-base / libggml-cpu from %s\n", dir.c_str()); return 2; }
    auto row_size = (size_t (*)(int, int64_t)) dlsym(hb, "ggml_row_size");
    auto q_init = (void (*)(int)) dlsym(hb, "ggml_quantize_init");
    auto q_chunk = (size_t (*)(int, const float *, void *, int64_t, int64_t, int64_t, const float *)) dlsym(hb, "ggml_quantize_chunk");
    auto traits = (const traits_cpu * (*)(int)) dlsym(hc, "ggml_get_type_traits_cpu");
    auto cpu_init = (void (*)()) dlsym(hc, "ggml_cpu_init");
    auto type_name = (const char * (*)(int)) dlsym(hb, "ggml_type_name");
    if (cpu_init) cpu_init();
    struct T { int id; };
    const int ids[] = { 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 34, 35, 39, 42 };
    const int64_t n = 4096, nrows = 64;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> src((size_t) n * nrows), act(n), ones(n, 1.0f);
    for (auto & v : src) v = nd(rng);
    for (auto & v : act) v = nd(rng);
    printf("%-10s %12s %14s\n", "type", "ns per dot", "G weights/s");
    for (int id : ids) {
        const traits_cpu * t = traits(id);
        if (!t || !t->vec_dot) continue;
        const size_t rs = row_size(id, n);
        std::vector<uint8_t> w(rs * nrows);
        q_init(id);
        size_t got = q_chunk(id, src.data(), w.data(), 0, nrows, n, ones.data());
        if (got != rs * nrows) {                       // type cannot be produced by the quantiser: fill with random bytes and a sane scale
            std::mt19937_64 r2(7);
            for (auto & b : w) b = (uint8_t) r2();
            if (id == 42) for (int64_t r = 0; r < nrows; r++) for (int64_t b = 0; b < n / 64; b++) { w[r * rs + b * 18] = 0x00; w[r * rs + b * 18 + 1] = 0x30; }
        }
        const size_t as = row_size(t->vec_dot_type, n);
        std::vector<uint8_t> a(as + 64);
        auto tf = traits(t->vec_dot_type);
        if (tf && tf->from_float) tf->from_float(act.data(), a.data(), n);
        volatile float sink = 0; float s = 0;
        const int reps = 20000;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) { t->vec_dot((int) n, &s, 0, w.data() + (size_t) (i % nrows) * rs, 0, a.data(), 0, 1); sink += s; }
        const double ns = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps * 1e9;
        printf("%-10s %12.0f %14.2f%s\n", type_name(id), ns, n / ns, id == 42 ? "   <- llama.cpp scalar loop" : "");
        if (id == 42) {
            std::vector<uint8_t> a2 = a;
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; i++) { fastk::vec_dot_q2_0_q8_0_avx2<true>((int) n, &s, w.data() + (size_t) (i % nrows) * rs, a2.data()); sink += s; }
            const double ns2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() / reps * 1e9;
            printf("%-10s %12.0f %14.2f   <- plugin AVX2 kernel\n", "Q2_0 fast", ns2, n / ns2);
        }
    }
    return 0;
}
