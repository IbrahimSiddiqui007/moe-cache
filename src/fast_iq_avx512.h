// AVX-512 dot products for IQ2_XXS x Q8_K and IQ3_XXS x Q8_K (llama.cpp's block layouts), written for CPUs with AVX-512BW (Zen 4, Ice Lake and newer).
//
// STATUS: written without access to an AVX-512 machine. They are never used unless MOE_CACHE_AVX512=1, and even then only after a self-check that
// compares them bit-for-bit with llama.cpp's own function on random data (see avx512_selfcheck in the plugin). tests/kernel_variants.cpp and
// tools/friend_test.sh run the same comparison and time both versions.
//
// Why the result is identical to llama.cpp's AVX2 function: the integer arithmetic (unsigned grid x signed activation, summed in 16 bit, times the
// block scale, summed in 32 bit) is exact in any order; the 32-bit sums of one 256-weight block are brought to the same eight lanes as the AVX2
// version and enter the float accumulator with the same fused multiply-add and the same final horizontal sum.
//
// What is different: eight grid entries are fetched with one gather instead of two; the 64 sign bits of 64 weights become one mask register and the
// signs are applied with a masked subtraction (the AVX2 code does eight table lookups and a byte-sign instruction per 64 weights).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include "iq_tables.h"

#define FASTK512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma,f16c")))
#define FASTK512_VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx2,fma,f16c")))

namespace fastk512 {

struct block_iq2_xxs { uint16_t d; uint16_t qs[32]; };           // 256 weights, 66 bytes
struct block_iq3_xxs { uint16_t d; uint8_t qs[96]; };            // 256 weights, 98 bytes: 64 grid indices, then 32 bytes of signs and scales
struct block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };

FASTK512 static inline float hsum8(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// 64-bit mask of the negative weights of 32 weights: four 7-bit sign fields in the low 28 bits of w, each expanded by ksigns (bit j = weight j negative)
static inline uint32_t sign_mask32(uint32_t w) {
    return (uint32_t) fastk::KSIGNS_IQ2XS[w & 127] | ((uint32_t) fastk::KSIGNS_IQ2XS[(w >> 7) & 127] << 8) |
           ((uint32_t) fastk::KSIGNS_IQ2XS[(w >> 14) & 127] << 16) | ((uint32_t) fastk::KSIGNS_IQ2XS[(w >> 21) & 127] << 24);
}

#define FASTK_ACC_PLAIN(acc, dot, sc) _mm512_add_epi32((acc), _mm512_madd_epi16((dot), (sc)))
#define FASTK_ACC_VNNI(acc, dot, sc) _mm512_dpwssd_epi32((acc), (dot), (sc))

#define FASTK_DEFINE_IQ2_XXS(NAME, ATTR, ACC)                                                                                                         \
ATTR static void NAME(int n, float * s, const void * vx, const void * vy) {                                                                           \
    const block_iq2_xxs * x = (const block_iq2_xxs *) vx;                                                                                             \
    const block_q8_K * y = (const block_q8_K *) vy;                                                                                                   \
    const int nb = n / 256;                                                                                                                           \
    const __m128i pick = _mm_setr_epi8(0, 1, 2, 3, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1);                                                     \
    __m256 accumf = _mm256_setzero_ps();                                                                                                              \
    for (int i = 0; i < nb; i++) {                                                                                                                    \
        const float d = _cvtsh_ss(x[i].d) * y[i].d;                                                                                                   \
        const uint16_t * q2 = x[i].qs;                                                                                                                \
        const int8_t * q8 = y[i].qs;                                                                                                                  \
        __m512i acc = _mm512_setzero_si512();                                                                                                         \
        for (int ib = 0; ib < 4; ib++) {                                                                                                              \
            uint32_t a[4];                                                                                                                            \
            memcpy(a, q2, 16);                                                                                                                        \
            q2 += 8;                                                                                                                                  \
            const __m128i raw = _mm_loadu_si128((const __m128i *) a);                                                                                 \
            const __m256i idx = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(raw, pick));                                                                    \
            const __m512i grid = _mm512_i32gather_epi64(idx, (const long long *) fastk::IQ2XXS_GRID, 8);                                              \
            const __mmask64 neg = (__mmask64) sign_mask32(a[1]) | ((__mmask64) sign_mask32(a[3]) << 32);                                              \
            const __m512i q = _mm512_loadu_si512((const void *) q8);                                                                                  \
            q8 += 64;                                                                                                                                 \
            const __m512i qs = _mm512_mask_sub_epi8(q, neg, _mm512_setzero_si512(), q);                                                               \
            const __m512i dot = _mm512_maddubs_epi16(grid, qs);                                                                                       \
            const __m512i sc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16((short) (2 * (a[1] >> 28) + 1))),                          \
                                                  _mm256_set1_epi16((short) (2 * (a[3] >> 28) + 1)), 1);                                              \
            acc = ACC(acc, dot, sc);                                                                                                                  \
        }                                                                                                                                             \
        const __m256i sum = _mm256_add_epi32(_mm512_castsi512_si256(acc), _mm512_extracti64x4_epi64(acc, 1));                                         \
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum), accumf);                                                                 \
    }                                                                                                                                                 \
    *s = 0.125f * hsum8(accumf);                                                                                                                      \
}

#define FASTK_DEFINE_IQ3_XXS(NAME, ATTR, ACC)                                                                                                         \
ATTR static void NAME(int n, float * s, const void * vx, const void * vy) {                                                                           \
    const block_iq3_xxs * x = (const block_iq3_xxs *) vx;                                                                                             \
    const block_q8_K * y = (const block_q8_K *) vy;                                                                                                   \
    const int nb = n / 256;                                                                                                                           \
    __m256 accumf = _mm256_setzero_ps();                                                                                                              \
    for (int i = 0; i < nb; i++) {                                                                                                                    \
        const float d = _cvtsh_ss(x[i].d) * y[i].d;                                                                                                   \
        const uint8_t * q3 = x[i].qs;                                                                                                                 \
        const uint8_t * gas = x[i].qs + 64;                                                                                                           \
        const int8_t * q8 = y[i].qs;                                                                                                                  \
        __m512i acc = _mm512_setzero_si512();                                                                                                         \
        for (int ib = 0; ib < 4; ib++) {                                                                                                              \
            uint32_t a[2];                                                                                                                            \
            memcpy(a, gas, 8);                                                                                                                        \
            gas += 8;                                                                                                                                 \
            const __m512i idx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) q3));                                                          \
            q3 += 16;                                                                                                                                 \
            const __m512i grid = _mm512_i32gather_epi32(idx, (const int *) fastk::IQ3XXS_GRID, 4);                                                    \
            const __mmask64 neg = (__mmask64) sign_mask32(a[0]) | ((__mmask64) sign_mask32(a[1]) << 32);                                              \
            const __m512i q = _mm512_loadu_si512((const void *) q8);                                                                                  \
            q8 += 64;                                                                                                                                 \
            const __m512i qs = _mm512_mask_sub_epi8(q, neg, _mm512_setzero_si512(), q);                                                               \
            const __m512i dot = _mm512_maddubs_epi16(grid, qs);                                                                                       \
            const __m512i sc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16((short) (2 * (a[0] >> 28) + 1))),                          \
                                                  _mm256_set1_epi16((short) (2 * (a[1] >> 28) + 1)), 1);                                              \
            acc = ACC(acc, dot, sc);                                                                                                                  \
        }                                                                                                                                             \
        const __m256i sum = _mm256_add_epi32(_mm512_castsi512_si256(acc), _mm512_extracti64x4_epi64(acc, 1));                                         \
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum), accumf);                                                                 \
    }                                                                                                                                                 \
    *s = 0.25f * hsum8(accumf);                                                                                                                       \
}

FASTK_DEFINE_IQ2_XXS(dot_iq2_xxs, FASTK512, FASTK_ACC_PLAIN)
FASTK_DEFINE_IQ2_XXS(dot_iq2_xxs_vnni, FASTK512_VNNI, FASTK_ACC_VNNI)
FASTK_DEFINE_IQ3_XXS(dot_iq3_xxs, FASTK512, FASTK_ACC_PLAIN)
FASTK_DEFINE_IQ3_XXS(dot_iq3_xxs_vnni, FASTK512_VNNI, FASTK_ACC_VNNI)

static inline bool cpu_has_avx512() {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
}
static inline bool cpu_has_avx512_vnni() {
    return cpu_has_avx512() && __builtin_cpu_supports("avx512vnni");
}

}  // namespace fastk512
