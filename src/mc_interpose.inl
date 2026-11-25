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
