// Faster IQ3_XXS x Q8_K dot product for the expert executor (GLM-5.3-Flash's down projections; about 80 % of its expert time).
// Grid values and sign bytes are fetched with gather instructions instead of eight scalar lookups. Bit-identical to llama.cpp's AVX2 function:
// the integer part is exact in any order, and the float part is the same fused multiply-add per 256-weight block with the same horizontal sum
// (tests/kernel_variants.cpp: 0 mismatches in 5,120 dots; 1.14x on an i7-13620H with AVX-VNNI, 1.12x without).
// Used only on Intel CPUs: gathers are slow on AMD Zen 2/3, where the library function stays.
#pragma once
#include "fast_iq_avx512.h"   // block layouts and lookup tables

namespace fastk3 {

#define FASTK3 __attribute__((target("avx2,fma,f16c")))
#define FASTK3_VNNI __attribute__((target("avx2,fma,f16c,avxvnni")))

inline const uint64_t * signs_table() {   // 8 sign bytes (+1 / -1) per 7-bit index, same as llama.cpp's keven_signs_q2xs
    static const struct T {
        uint64_t v[128];
        T() {
            for (int i = 0; i < 128; i++) {
                uint64_t x = 0;
                for (int j = 0; j < 8; j++) x |= (uint64_t) (((fastk::KSIGNS_IQ2XS[i] >> j) & 1) ? 0xff : 0x01) << (8 * j);
                v[i] = x;
            }
        }
    } t;
    return t.v;
}

#define FASTK3_IQ3_BODY(VNNI) \
    const fastk512::block_iq3_xxs * x = (const fastk512::block_iq3_xxs *) vx; \
    const fastk512::block_q8_K    * y = (const fastk512::block_q8_K *) vy; \
    const uint64_t * SIGNS = signs_table(); \
    const int nb = n / 256; \
    const __m128i shifts = _mm_setr_epi32(0, 7, 14, 21), m127 = _mm_set1_epi32(127); \
    __m256 accumf = _mm256_setzero_ps(); \
    for (int i = 0; i < nb; ++i) { \
        const float d = _cvtsh_ss(x[i].d) * y[i].d; \
        const uint8_t * q3 = x[i].qs; \
        const uint8_t * gas = x[i].qs + 64; \
        const int8_t * q8 = y[i].qs; \
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256(); \
        for (int ib32 = 0; ib32 < 8; ib32 += 2) { \
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32; \
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32; \
            uint32_t aux32[2]; \
            memcpy(aux32, gas, 8); gas += 8; \
            const __m256i g1 = _mm256_i32gather_epi32((const int *) fastk::IQ3XXS_GRID, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) q3)), 4); \
            const __m256i g2 = _mm256_i32gather_epi32((const int *) fastk::IQ3XXS_GRID, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) (q3 + 8))), 4); \
            const __m256i s1 = _mm256_i32gather_epi64((const long long *) SIGNS, _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int) aux32[0]), shifts), m127), 8); \
            const __m256i s2 = _mm256_i32gather_epi64((const long long *) SIGNS, _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int) aux32[1]), shifts), m127), 8); \
            q3 += 16; \
            const __m256i dot1 = _mm256_maddubs_epi16(g1, _mm256_sign_epi8(q8_1, s1)); \
            const __m256i dot2 = _mm256_maddubs_epi16(g2, _mm256_sign_epi8(q8_2, s2)); \
            const __m256i sc1 = _mm256_set1_epi16((short) (2 * (aux32[0] >> 28) + 1)), sc2 = _mm256_set1_epi16((short) (2 * (aux32[1] >> 28) + 1)); \
            if (VNNI) { \
                sumi1 = _mm256_dpwssd_avx_epi32(sumi1, dot1, sc1); \
                sumi2 = _mm256_dpwssd_avx_epi32(sumi2, dot2, sc2); \
            } else { \
                sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1)); \
                sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2)); \
            } \
        } \
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf); \
    } \
    __m128 r = _mm256_extractf128_ps(accumf, 1); \
    r = _mm_add_ps(r, _mm256_castps256_ps128(accumf)); \
    r = _mm_add_ps(r, _mm_movehl_ps(r, r)); \
    r = _mm_add_ss(r, _mm_movehdup_ps(r)); \
    *s = 0.25f * _mm_cvtss_f32(r);

FASTK3_VNNI inline void dot_iq3_xxs_vnni(int n, float * s, const void * vx, const void * vy) { FASTK3_IQ3_BODY(true) }
FASTK3 inline void dot_iq3_xxs_avx2(int n, float * s, const void * vx, const void * vy) { FASTK3_IQ3_BODY(false) }

// executor signature (ggml_vec_dot_t): one row, nrc = 1
inline void vec_dot_iq3_xxs_vnni(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) { dot_iq3_xxs_vnni(n, s, vx, vy); }
inline void vec_dot_iq3_xxs_avx2(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) { dot_iq3_xxs_avx2(n, s, vx, vy); }

// Intel CPU with AVX2 + FMA + F16C (and AVX-VNNI for the faster version); MOE_CACHE_FAST_KERNELS=0 turns it off with the other kernels.
// any_vendor (MOE_CACHE_IQ3_KERNEL=1) drops the Intel check: the result is bit-identical anyway, only the speed on AMD is unmeasured (Zen 4 gathers
// are much faster than Zen 2/3's; GitHub issue #1, idea I36).
inline bool usable(bool & vnni, bool any_vendor = false) {
    __builtin_cpu_init();
    vnni = __builtin_cpu_supports("avxvnni");
    return (any_vendor || __builtin_cpu_is("intel")) && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
}

}  // namespace fastk3
