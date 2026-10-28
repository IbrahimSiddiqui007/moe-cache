// Q2_0 x Q8_0 dot product for x86-64 with AVX2.
//
// llama.cpp (commit 7fe450e19) has no SIMD version of this kernel: on x86 it runs the plain C loop `ggml_vec_dot_q2_0_q8_0_generic`.
// Models that store the expert down projection as Q2_0 (Qwen3.8-Flash-Next "IQ2_XS" and "Q2_0" files) spend most of their CPU time there.
//
// This version gives bit-identical results: the integer part of every 32-weight group is exact in both versions, and the floating point
// part is done in the same order with the same operations as the C loop (checked against the real library function by tests/kernel_check).
//   C loop:  sumi (float) += d1 * (float) sumi_block    for the two Q8_0 groups of a Q2_0 block, then   sumf += d0 * sumi
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace fastk {

struct block_q2_0 { uint16_t d; uint8_t qs[16]; };   // 64 weights, 2 bits each, values 0..3 mean -1..2
struct block_q8_0 { uint16_t d; int8_t qs[32]; };    // 32 weights

static inline float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else {                       // subnormal half -> normal float
            int e = -1;
            do { e++; man <<= 1; } while (!(man & 0x400));
            out = sign | ((uint32_t) (127 - 15 - e) << 23) | ((man & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (man << 13) | (man ? 0x00400000u : 0u);   // NaN: quiet bit set like the hardware conversion
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

// exact integer sum of (value - 1) * q8 over 32 weights; q = 8 bytes of Q2_0 (32 two-bit values, value t in bits 2t..2t+1 of the little-endian word)
__attribute__((target("avx2")))
static inline int group_dot_avx2(const uint8_t * q, const int8_t * a) {
    const __m128i lut_lo = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);   // n & 3
    const __m128i lut_hi = _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);   // n >> 2
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128i bytes = _mm_loadl_epi64((const __m128i *) q);                            // 8 bytes = 32 weights
    const __m128i ln = _mm_and_si128(bytes, m4);
    const __m128i hn = _mm_and_si128(_mm_srli_epi16(bytes, 4), m4);
    const __m128i f0 = _mm_shuffle_epi8(lut_lo, ln);   // weight 4b+0
    const __m128i f1 = _mm_shuffle_epi8(lut_hi, ln);   // weight 4b+1
    const __m128i f2 = _mm_shuffle_epi8(lut_lo, hn);   // weight 4b+2
    const __m128i f3 = _mm_shuffle_epi8(lut_hi, hn);   // weight 4b+3
    const __m128i u01 = _mm_unpacklo_epi8(f0, f1);
    const __m128i u23 = _mm_unpacklo_epi8(f2, f3);
    const __m128i w_lo = _mm_unpacklo_epi16(u01, u23);   // weights 0..15
    const __m128i w_hi = _mm_unpackhi_epi16(u01, u23);   // weights 16..31
    const __m256i vals = _mm256_set_m128i(w_hi, w_lo);
    const __m256i av = _mm256_loadu_si256((const __m256i *) a);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(vals, av), ones16);                       // sum of value * a
    const __m256i s = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), av), ones16);        // sum of a
    __m256i d = _mm256_sub_epi32(p, s);
    __m128i r = _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
    r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 0x4e));
    r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 0xb1));
    return _mm_cvtsi128_si32(r);
}

// fma = true reproduces a C loop that the compiler contracted to fused multiply-add, false a loop with separate multiply and add
template <bool FMA>
__attribute__((target("avx2,fma")))
static inline void vec_dot_q2_0_q8_0_avx2(int n, float * s, const void * vx, const void * vy) {
    const block_q2_0 * x = (const block_q2_0 *) vx;
    const block_q8_0 * y = (const block_q8_0 *) vy;
    const int nb = n / 64;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d0 = half_to_float(x[i].d);
        float sumi = 0.0f;
        for (int k = 0; k < 2; k++) {
            const block_q8_0 * yb = &y[i * 2 + k];
            const float d1 = half_to_float(yb->d);
            const int sb = group_dot_avx2(&x[i].qs[k * 8], yb->qs);
            if (FMA) { sumi = __builtin_fmaf(d1, (float) sb, sumi); } else { sumi += d1 * (float) sb; }
        }
        if (FMA) { sumf = __builtin_fmaf(d0, sumi, sumf); } else { sumf += d0 * sumi; }
    }
    *s = sumf;
}


// ---- second version: activations in a kernel-friendly layout (only the expert executor uses it) ----
// The executor quantises one activation column and reuses it for thousands of weight rows, so it can afford to rearrange that column once.
// Per Q8_0 block of 32 values: the values are stored in the order (field j, byte b) = original index 4*b + j, plus their integer sum.
// A Q2_0 byte b of the block holds the weights 4b..4b+3 in its four 2-bit fields, so with this layout one shift and one mask give a vector
// that lines up with the activations and the whole group is two instructions of multiply-add: no unpacking, no table lookups.
// The integer result is the same, and the floating point part is the same as in vec_dot_q2_0_q8_0_avx2 (checked in tests/kernel_check.cpp).
struct block_q8_0_perm { uint16_t d; int16_t sum; int8_t qs[32]; };   // 36 bytes

static inline void permute_q8_0(const block_q8_0 * y, block_q8_0_perm * out, int nblocks) {
    for (int i = 0; i < nblocks; i++) {
        out[i].d = y[i].d;
        int sum = 0;
        for (int b = 0; b < 8; b++) {
            for (int j = 0; j < 4; j++) {
                const int8_t v = y[i].qs[4 * b + j];
                out[i].qs[j * 8 + b] = v;
                sum += v;
            }
        }
        out[i].sum = (int16_t) sum;
    }
}

__attribute__((target("avx2,fma,f16c")))
static inline void vec_dot_q2_0_perm_avx2(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    const block_q2_0 * x = (const block_q2_0 *) vx;
    const block_q8_0_perm * y = (const block_q8_0_perm *) vy;
    const int nb = n / 64;
    const __m256i shifts = _mm256_setr_epi64x(0, 2, 4, 6);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i ones16 = _mm256_set1_epi16(1);
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        uint64_t q0, q1;
        memcpy(&q0, &x[i].qs[0], 8);
        memcpy(&q1, &x[i].qs[8], 8);
        const __m256i v0 = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x((long long) q0), shifts), m3);
        const __m256i v1 = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x((long long) q1), shifts), m3);
        const __m256i a0 = _mm256_loadu_si256((const __m256i *) y[2 * i].qs);
        const __m256i a1 = _mm256_loadu_si256((const __m256i *) y[2 * i + 1].qs);
        const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(v0, a0), ones16);
        const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(v1, a1), ones16);
        const __m128i r0 = _mm_add_epi32(_mm256_castsi256_si128(p0), _mm256_extracti128_si256(p0, 1));
        const __m128i r1 = _mm_add_epi32(_mm256_castsi256_si128(p1), _mm256_extracti128_si256(p1, 1));
        const __m128i h = _mm_hadd_epi32(_mm_hadd_epi32(r0, r1), _mm_setzero_si128());   // lane 0 = sum of r0, lane 1 = sum of r1 (exact integers)
        const int sb0 = _mm_cvtsi128_si32(h) - (int) y[2 * i].sum;
        const int sb1 = _mm_extract_epi32(h, 1) - (int) y[2 * i + 1].sum;
        const float d0 = _cvtsh_ss(x[i].d), d1a = _cvtsh_ss(y[2 * i].d), d1b = _cvtsh_ss(y[2 * i + 1].d);
        float sumi = __builtin_fmaf(d1a, (float) sb0, 0.0f);
        sumi = __builtin_fmaf(d1b, (float) sb1, sumi);
        sumf = __builtin_fmaf(d0, sumi, sumf);
    }
    *s = sumf;
}

// ---- third version: exact match for llama.cpp's own AVX2 kernel (ggml-org/llama.cpp#30447; in release b11669 = 620d2225c and master) ----
// llama.cpp now has a SIMD Q2_0 dot of its own that adds up in another order than the plain loop, so the kernels above no longer match it.
// up_avx2 is upstream's function written out (the plugin compares it with the library's at start, bit for bit, before using ys_avx2).
// ys_avx2 does the same integer lanes and the same multiply-add order, but takes the activation's per-lane sums madd(maddubs(1, y), 1) from a
// column prepared once by the executor instead of computing them in every row's dot (desk: 1.24x, 0 mismatches; idea N20).
// madd(maddubs(v, y) - maddubs(1, y), 1) = madd(maddubs(v, y), 1) - madd(maddubs(1, y), 1) exactly: no 16-bit saturation (|pair| <= 2*3*128).
__attribute__((target("avx2,fma,f16c")))
static inline float up_hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
__attribute__((target("avx2,fma,f16c")))
static inline __m256i up_unpack_32(const uint8_t * qs) {
    __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) qs));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 12));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 6));
    return _mm256_and_si256(v, _mm256_set1_epi32(0x03030303));
}
__attribute__((target("avx2,fma,f16c")))
static inline __m256 up_acc_block(const block_q2_0 * x, const block_q8_0 * y, __m256 acc) {
    const float dx = _cvtsh_ss(x->d);
    const __m256 d0 = _mm256_set1_ps(dx * _cvtsh_ss(y[0].d));
    const __m256 d1 = _mm256_set1_ps(dx * _cvtsh_ss(y[1].d));
    const __m256i one = _mm256_set1_epi8(1);
    const __m256i y0 = _mm256_loadu_si256((const __m256i *) y[0].qs), y1 = _mm256_loadu_si256((const __m256i *) y[1].qs);
    const __m256i q0 = _mm256_sub_epi16(_mm256_maddubs_epi16(up_unpack_32(x->qs + 0), y0), _mm256_maddubs_epi16(one, y0));
    const __m256i q1 = _mm256_sub_epi16(_mm256_maddubs_epi16(up_unpack_32(x->qs + 8), y1), _mm256_maddubs_epi16(one, y1));
    const __m256 p0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(q0, _mm256_set1_epi16(1)));
    const __m256 p1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(q1, _mm256_set1_epi16(1)));
    acc = _mm256_fmadd_ps(d0, p0, acc);
    return _mm256_fmadd_ps(d1, p1, acc);
}
__attribute__((target("avx2,fma,f16c")))
static inline void vec_dot_q2_0_up_avx2(int n, float * s, const void * vx, const void * vy) {
    const block_q2_0 * x = (const block_q2_0 *) vx;
    const block_q8_0 * y = (const block_q8_0 *) vy;
    const int nb = n / 64;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 1 < nb; i += 2) {
        acc0 = up_acc_block(x + i + 0, y + 2 * i + 0, acc0);
        acc1 = up_acc_block(x + i + 1, y + 2 * i + 2, acc1);
    }
    if (i < nb) {
        acc0 = up_acc_block(x + i, y + 2 * i, acc0);
    }
    *s = up_hsum_float_8(_mm256_add_ps(acc0, acc1));
}

struct block_q8_0_ys { int32_t ysum[8]; int8_t qs[32]; float d; };   // 68 bytes: lane sums, values, scale as float

__attribute__((target("avx2,fma,f16c")))
static inline void prepare_q8_0_ys(const block_q8_0 * y, block_q8_0_ys * out, int nblocks) {
    for (int j = 0; j < nblocks; j++) {
        const __m256i yy = _mm256_loadu_si256((const __m256i *) y[j].qs);
        _mm256_storeu_si256((__m256i *) out[j].ysum, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), yy), _mm256_set1_epi16(1)));
        memcpy(out[j].qs, y[j].qs, 32);
        out[j].d = _cvtsh_ss(y[j].d);
    }
}

__attribute__((target("avx2,fma,f16c")))
static inline __m256 ys_acc_block(const block_q2_0 * x, const block_q8_0_ys * y, __m256 acc) {
    const float dx = _cvtsh_ss(x->d);
    const __m256 d0 = _mm256_set1_ps(dx * y[0].d);
    const __m256 d1 = _mm256_set1_ps(dx * y[1].d);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i y0 = _mm256_loadu_si256((const __m256i *) y[0].qs), y1 = _mm256_loadu_si256((const __m256i *) y[1].qs);
    const __m256 p0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(up_unpack_32(x->qs + 0), y0), ones16),
                                                          _mm256_loadu_si256((const __m256i *) y[0].ysum)));
    const __m256 p1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(up_unpack_32(x->qs + 8), y1), ones16),
                                                          _mm256_loadu_si256((const __m256i *) y[1].ysum)));
    acc = _mm256_fmadd_ps(d0, p0, acc);
    return _mm256_fmadd_ps(d1, p1, acc);
}
__attribute__((target("avx2,fma,f16c")))
static inline void vec_dot_q2_0_ys_avx2(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    const block_q2_0 * x = (const block_q2_0 *) vx;
    const block_q8_0_ys * y = (const block_q8_0_ys *) vy;
    const int nb = n / 64;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 1 < nb; i += 2) {
        acc0 = ys_acc_block(x + i + 0, y + 2 * i + 0, acc0);
        acc1 = ys_acc_block(x + i + 1, y + 2 * i + 2, acc1);
    }
    if (i < nb) {
        acc0 = ys_acc_block(x + i, y + 2 * i, acc0);
    }
    *s = up_hsum_float_8(_mm256_add_ps(acc0, acc1));
}

}  // namespace fastk
