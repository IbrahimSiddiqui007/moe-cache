// moe-cache plugin part: interposed llama.cpp functions: Q2_0 / AVX-512 kernels, GPU-cache uploads.
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

#if defined(__x86_64__) && defined(__linux__)
// ---- faster Q2_0 kernel (bit-identical to llama.cpp's) ----
// llama.cpp up to 23b0202a1 has no SIMD version of the Q2_0 x Q8_0 dot product on x86: it runs a plain C loop. This definition replaces it when the
// plugin is preloaded (LD_PRELOAD), because the function table of libggml-cpu is filled by symbol name. Results are bit-identical
// (tests/kernel_check.cpp compares them with the real library function). MOE_CACHE_FAST_KERNELS=0 switches it off.
#include <dlfcn.h>
#include "fast_q2_0.h"
extern "C" MOE_CACHE_EXPORT void ggml_vec_dot_q2_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

bool q2_0_library_is_plain();   // mc_exec.inl

// preloaded into a llama-server whose CPU backend is a separate library loaded later (RTLD_LOCAL: --load-method preload on a build with
// GGML_BACKEND_DL): that library's table still binds to these symbols, but RTLD_NEXT cannot find its own functions. Say so instead of calling
// a null pointer, or (Q2_0) replacing a kernel that could not be compared with the library's (audit 2026-10-29)
[[noreturn]] static void interpose_no_orig(const char * fn) {
    fprintf(stderr, "moe-cache: %s of llama.cpp's CPU backend is not visible to the preloaded plugin (its CPU backend is a separate library): "
                    "load the plugin with --load-method path (GGML_BACKEND_PATH) instead of LD_PRELOAD\n", fn);
    abort();
}

extern "C" MOE_CACHE_EXPORT void ggml_vec_dot_q2_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    typedef void (*fn_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    static fn_t orig = (fn_t) dlsym(RTLD_NEXT, "ggml_vec_dot_q2_0_q8_0");
    static const bool fast = [] {
        const char * e = getenv("MOE_CACHE_FAST_KERNELS");
        return !(e && strcmp(e, "0") == 0) && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && q2_0_library_is_plain();
    }();
    if (!orig) {
        interpose_no_orig("ggml_vec_dot_q2_0_q8_0");
    }
    if (fast && nrc == 1 && n > 0 && n % 64 == 0) {
        fastk::vec_dot_q2_0_q8_0_avx2<true>(n, s, vx, vy);
        return;
    }
    orig(n, s, bs, vx, bx, vy, by, nrc);
}

// ---- AVX-512 IQ2_XXS / IQ3_XXS dot products (MOE_CACHE_AVX512=1; off by default) ----
// See src/fast_iq_avx512.h. They are only used after a self-check that compares them bit-for-bit with llama.cpp's own function on random data;
// a single mismatch switches them off for the whole run. MOE_CACHE_AVX512=1 on a CPU without AVX-512 does nothing.
#include <random>
#include "fast_iq_avx512.h"
namespace {
typedef void (*iq_dot_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

int iq512_decide(iq_dot_t o2, iq_dot_t o3) {      // 0 off, 1 AVX-512, 2 AVX-512 + VNNI
    const char * e = getenv("MOE_CACHE_AVX512");
    if (!(e && atoi(e) != 0) || !o2 || !o3) {
        return 0;
    }
    if (!fastk512::cpu_has_avx512()) {
        fprintf(stderr, "moe-cache: MOE_CACHE_AVX512=1 but this CPU has no AVX-512 (BW, VL, DQ): not used\n");
        return 0;
    }
    const bool vnni = fastk512::cpu_has_avx512_vnni();
    std::mt19937 rng(12345);
    alignas(64) static fastk512::block_iq2_xxs w2[8];
    alignas(64) static fastk512::block_iq3_xxs w3[8];
    alignas(64) static fastk512::block_q8_K a[8];
    bool ok = true;
    for (int trial = 0; trial < 200 && ok; trial++) {
        for (auto & b : w2) { for (auto & q : b.qs) q = (uint16_t) rng(); b.d = (uint16_t) (0x2800 + rng() % 0x1000); }
        for (auto & b : w3) { for (auto & q : b.qs) q = (uint8_t) rng(); b.d = (uint16_t) (0x2800 + rng() % 0x1000); }
        for (auto & b : a) { b.d = 0.001f * (float) (1 + rng() % 100); for (auto & q : b.qs) q = (int8_t) ((int) (rng() % 255) - 127); }
        const int n = 8 * 256;
        float r2, r3, f2, f3;
        o2(n, &r2, 0, w2, 0, a, 0, 1);
        o3(n, &r3, 0, w3, 0, a, 0, 1);
        fastk512::dot_iq2_xxs(n, &f2, w2, a);
        fastk512::dot_iq3_xxs(n, &f3, w3, a);
        ok = memcmp(&r2, &f2, 4) == 0 && memcmp(&r3, &f3, 4) == 0;
        if (ok && vnni) {
            fastk512::dot_iq2_xxs_vnni(n, &f2, w2, a);
            fastk512::dot_iq3_xxs_vnni(n, &f3, w3, a);
            ok = memcmp(&r2, &f2, 4) == 0 && memcmp(&r3, &f3, 4) == 0;
        }
    }
    if (!ok) {
        fprintf(stderr, "moe-cache: AVX-512 IQ kernels FAILED the bit-exact self-check: using llama.cpp's kernels (please report this CPU)\n");
        return 0;
    }
    fprintf(stderr, "moe-cache: AVX-512 IQ2_XXS / IQ3_XXS kernels on (%s, self-check passed)\n", vnni ? "AVX-512 + VNNI" : "AVX-512");
    return vnni ? 2 : 1;
}

int iq512_mode() {
    static const int mode = iq512_decide((iq_dot_t) dlsym(RTLD_NEXT, "ggml_vec_dot_iq2_xxs_q8_K"), (iq_dot_t) dlsym(RTLD_NEXT, "ggml_vec_dot_iq3_xxs_q8_K"));
    return mode;
}
}  // namespace

static int iq512_probe_trampoline() { return iq512_mode(); }
static const int iq512_registered = (moe_cache_iq512_probe = &iq512_probe_trampoline, 0);

extern "C" MOE_CACHE_EXPORT void ggml_vec_dot_iq2_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    static iq_dot_t orig = (iq_dot_t) dlsym(RTLD_NEXT, "ggml_vec_dot_iq2_xxs_q8_K");
    if (!orig) {
        interpose_no_orig("ggml_vec_dot_iq2_xxs_q8_K");
    }
    const int m = iq512_mode();
    if (m && nrc == 1 && n > 0 && n % 256 == 0) {
        if (m == 2) fastk512::dot_iq2_xxs_vnni(n, s, vx, vy); else fastk512::dot_iq2_xxs(n, s, vx, vy);
        return;
    }
    orig(n, s, bs, vx, bx, vy, by, nrc);
}

extern "C" MOE_CACHE_EXPORT void ggml_vec_dot_iq3_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    static iq_dot_t orig = (iq_dot_t) dlsym(RTLD_NEXT, "ggml_vec_dot_iq3_xxs_q8_K");
    if (!orig) {
        interpose_no_orig("ggml_vec_dot_iq3_xxs_q8_K");
    }
    const int m = iq512_mode();
    if (m && nrc == 1 && n > 0 && n % 256 == 0) {
        if (m == 2) fastk512::dot_iq3_xxs_vnni(n, s, vx, vy); else fastk512::dot_iq3_xxs(n, s, vx, vy);
        return;
    }
    orig(n, s, bs, vx, bx, vy, by, nrc);
}

// ---- llama.cpp's GPU expert cache (--moe-cache-mib) on top of this cache: see gpu_cache_mode() ----
// Its uploads read the expert bytes straight from the weight memory, where only the experts we hold in RAM exist. Make them resident first.
extern "C" MOE_CACHE_EXPORT void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    typedef void (*fn_t)(ggml_backend_t, struct ggml_tensor *, const void *, size_t, size_t);
    static fn_t orig = (fn_t) dlsym(RTLD_NEXT, "ggml_backend_tensor_set_async");
    if (gpu_cache_mode()) {
        gpu_cache_ensure(data, size);
    }
    if (!orig) {
        interpose_no_orig("ggml_backend_tensor_set_async");
    }
    orig(backend, tensor, data, offset, size);
}
#endif
