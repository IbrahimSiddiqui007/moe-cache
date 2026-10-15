// Candidate faster AVX2 / AVX-VNNI versions of the IQ2_XXS x Q8_K and IQ3_XXS x Q8_K dot products, checked bit-for-bit against the real
// library function and timed A-B-A-B against it (one core, data in cache).
//   g++ -O2 -std=c++17 -Isrc tests/kernel_variants.cpp -o /tmp/kernel_variants -ldl
//   /tmp/kernel_variants <dir with libggml-base.so.0 and libggml-cpu.so.0> [check]
// Block layouts and lookup tables come from src/ (no llama.cpp source tree needed). On CPUs with AVX-512 it also checks and times src/fast_iq_avx512.h.
//
// Why the results are identical: every variant does the same integer arithmetic (u8 grid x s8 activation -> s16 -> x scale -> s32; sums of
// integers are exact in any order), and the float part is the same fused multiply-add per 256-weight block in the same order with the same
// horizontal sum as the library.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "fast_iq_avx512.h"
#include "fast_iq3.h"
using block_iq2_xxs = fastk512::block_iq2_xxs;
using block_iq3_xxs = fastk512::block_iq3_xxs;
using block_q8_K = fastk512::block_q8_K;
#define QK_K 256
#define iq2xxs_grid fastk::IQ2XXS_GRID
#define iq3xxs_grid fastk::IQ3XXS_GRID
#define ksigns_iq2xs fastk::KSIGNS_IQ2XS
#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <dlfcn.h>
#include <immintrin.h>
#include <random>
#include <string>
#include <vector>

#define TGT __attribute__((target("avx2,fma,f16c,avxvnni")))
using dot_fn = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

static uint64_t SIGNS[128];   // 8 sign bytes (+1/-1) per 7-bit index; the 8th keeps the parity even (same as llama.cpp's keven_signs_q2xs)
static void init_signs() {
    for (int i = 0; i < 128; i++) {
        uint64_t v = 0;
        for (int j = 0; j < 8; j++) v |= (uint64_t) (((ksigns_iq2xs[i] >> j) & 1) ? 0xff : 0x01) << (8 * j);
        SIGNS[i] = v;
    }
}
static TGT inline float hsum8(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}
static TGT inline __m256i acc16(__m256i acc, __m256i dot, __m256i scale, bool vnni) {
    return vnni ? _mm256_dpwssd_avx_epi32(acc, dot, scale) : _mm256_add_epi32(acc, _mm256_madd_epi16(dot, scale));
}

// ---- IQ2_XXS -------------------------------------------------------------------------------------------------------------------------
template <bool GATHER, bool VNNI>
static TGT void iq2xxs(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    const block_iq2_xxs * x = (const block_iq2_xxs *) vx;
    const block_q8_K    * y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    const __m128i shifts = _mm_setr_epi32(0, 7, 14, 21), m127 = _mm_set1_epi32(127);
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = _cvtsh_ss(x[i].d) * y[i].d;
        const uint16_t * q2 = x[i].qs;
        const int8_t * q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            uint32_t aux32[4];
            memcpy(aux32, q2, 16); q2 += 8;
            const uint8_t * aux8 = (const uint8_t *) aux32;
            __m256i g1, g2, s1, s2;
            if (GATHER) {
                const __m128i i1 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) aux32[0]));
                const __m128i i2 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) aux32[2]));
                g1 = _mm256_i32gather_epi64((const long long *) iq2xxs_grid, i1, 8);
                g2 = _mm256_i32gather_epi64((const long long *) iq2xxs_grid, i2, 8);
                s1 = _mm256_i32gather_epi64((const long long *) SIGNS, _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int) aux32[1]), shifts), m127), 8);
                s2 = _mm256_i32gather_epi64((const long long *) SIGNS, _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int) aux32[3]), shifts), m127), 8);
            } else {
                g1 = _mm256_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
                g2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
                s1 = _mm256_set_epi64x(SIGNS[(aux32[1] >> 21) & 127], SIGNS[(aux32[1] >> 14) & 127], SIGNS[(aux32[1] >> 7) & 127], SIGNS[aux32[1] & 127]);
                s2 = _mm256_set_epi64x(SIGNS[(aux32[3] >> 21) & 127], SIGNS[(aux32[3] >> 14) & 127], SIGNS[(aux32[3] >> 7) & 127], SIGNS[aux32[3] & 127]);
            }
            const __m256i dot1 = _mm256_maddubs_epi16(g1, _mm256_sign_epi8(q8_1, s1));
            const __m256i dot2 = _mm256_maddubs_epi16(g2, _mm256_sign_epi8(q8_2, s2));
            sumi1 = acc16(sumi1, dot1, _mm256_set1_epi16((short) (2 * (aux32[1] >> 28) + 1)), VNNI);
            sumi2 = acc16(sumi2, dot2, _mm256_set1_epi16((short) (2 * (aux32[3] >> 28) + 1)), VNNI);
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = 0.125f * hsum8(accumf);
}

// ---- IQ3_XXS -------------------------------------------------------------------------------------------------------------------------
template <bool GATHER, bool VNNI>
static TGT void iq3xxs(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    const block_iq3_xxs * x = (const block_iq3_xxs *) vx;
    const block_q8_K    * y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    const __m128i shifts = _mm_setr_epi32(0, 7, 14, 21), m127 = _mm_set1_epi32(127);
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = _cvtsh_ss(x[i].d) * y[i].d;
        const uint8_t * q3 = x[i].qs;
        const uint8_t * gas = x[i].qs + QK_K / 4;
        const int8_t * q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            uint32_t aux32[2];
            memcpy(aux32, gas, 8); gas += 8;
            __m256i g1, g2, s1, s2;
            if (GATHER) {
                g1 = _mm256_i32gather_epi32((const int *) iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) q3)), 4);
                g2 = _mm256_i32gather_epi32((const int *) iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) (q3 + 8))), 4);
                s1 = _mm256_i32gather_epi64((const long long *) SIGNS, _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int) aux32[0]), shifts), m127), 8);
                s2 = _mm256_i32gather_epi64((const long long *) SIGNS, _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int) aux32[1]), shifts), m127), 8);
            } else {
                g1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                      iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
                g2 = _mm256_set_epi32(iq3xxs_grid[q3[15]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[12]],
                                      iq3xxs_grid[q3[11]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[9]], iq3xxs_grid[q3[8]]);
                s1 = _mm256_set_epi64x(SIGNS[(aux32[0] >> 21) & 127], SIGNS[(aux32[0] >> 14) & 127], SIGNS[(aux32[0] >> 7) & 127], SIGNS[aux32[0] & 127]);
                s2 = _mm256_set_epi64x(SIGNS[(aux32[1] >> 21) & 127], SIGNS[(aux32[1] >> 14) & 127], SIGNS[(aux32[1] >> 7) & 127], SIGNS[aux32[1] & 127]);
            }
            q3 += 16;
            const __m256i dot1 = _mm256_maddubs_epi16(g1, _mm256_sign_epi8(q8_1, s1));
            const __m256i dot2 = _mm256_maddubs_epi16(g2, _mm256_sign_epi8(q8_2, s2));
            sumi1 = acc16(sumi1, dot1, _mm256_set1_epi16((short) (2 * (aux32[0] >> 28) + 1)), VNNI);
            sumi2 = acc16(sumi2, dot2, _mm256_set1_epi16((short) (2 * (aux32[1] >> 28) + 1)), VNNI);
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = 0.25f * hsum8(accumf);
}

static void w512_iq2(int n, float * s, size_t, const void * x, size_t, const void * y, size_t, int) { fastk512::dot_iq2_xxs(n, s, x, y); }
static void w512_iq2v(int n, float * s, size_t, const void * x, size_t, const void * y, size_t, int) { fastk512::dot_iq2_xxs_vnni(n, s, x, y); }
static void w512_iq3(int n, float * s, size_t, const void * x, size_t, const void * y, size_t, int) { fastk512::dot_iq3_xxs(n, s, x, y); }
static void w512_iq3v(int n, float * s, size_t, const void * x, size_t, const void * y, size_t, int) { fastk512::dot_iq3_xxs_vnni(n, s, x, y); }

struct traits_cpu { void (*from_float)(const float *, void *, int64_t); dot_fn vec_dot; int vec_dot_type; int64_t nrows; };
struct Variant { const char * name; dot_fn fn; };

int main(int argc, char ** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    const bool check_only = argc > 2 && std::string(argv[2]) == "check";
    init_signs();
    void * hb = dlopen((dir + "/libggml-base.so.0").c_str(), RTLD_NOW | RTLD_GLOBAL);
    void * hc = dlopen((dir + "/libggml-cpu.so.0").c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!hc) {
        hc = dlopen((dir + "/libggml-cpu.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
    }
    if (!hc) {
        // GGML_CPU_ALL_VARIANTS builds (install.sh) have only libggml-cpu-<variant>.so: take the one ggml itself would pick (highest backend score)
        std::string best;
        int best_score = 0;
        if (DIR * d = opendir(dir.c_str())) {
            while (dirent * e = readdir(d)) {
                const std::string f = e->d_name;
                if (f.rfind("libggml-cpu-", 0) != 0 || f.find(".so") == std::string::npos) continue;
                void * h = dlopen((dir + "/" + f).c_str(), RTLD_NOW | RTLD_LOCAL);
                if (!h) continue;
                auto score = (int (*)()) dlsym(h, "ggml_backend_score");
                const int sc = score ? score() : 0;
                if (sc > best_score) { best_score = sc; best = f; }
                dlclose(h);
            }
            closedir(d);
        }
        if (!best.empty()) {
            hc = dlopen((dir + "/" + best).c_str(), RTLD_NOW | RTLD_LOCAL);
            printf("cpu library: %s (variant chosen by score)\n", best.c_str());
        }
    }
    if (!hb || !hc) { fprintf(stderr, "cannot load libggml-base / libggml-cpu from %s\n", dir.c_str()); return 2; }
    auto row_size = (size_t (*)(int, int64_t)) dlsym(hb, "ggml_row_size");
    auto q_init = (void (*)(int)) dlsym(hb, "ggml_quantize_init");
    auto q_chunk = (size_t (*)(int, const float *, void *, int64_t, int64_t, int64_t, const float *)) dlsym(hb, "ggml_quantize_chunk");
    auto traits = (const traits_cpu * (*)(int)) dlsym(hc, "ggml_get_type_traits_cpu");
    auto cpu_init = (void (*)()) dlsym(hc, "ggml_cpu_init");
    if (cpu_init) cpu_init();

    struct Case { const char * name; int type; std::vector<Variant> v; };
    std::vector<Case> cases = {
        { "iq2_xxs", 16, { { "scalar-lookup+madd", iq2xxs<false, false> }, { "gather", iq2xxs<true, false> }, { "vnni", iq2xxs<false, true> }, { "gather+vnni", iq2xxs<true, true> } } },
        { "iq3_xxs", 18, { { "scalar-lookup+madd", iq3xxs<false, false> }, { "gather", iq3xxs<true, false> }, { "vnni", iq3xxs<false, true> }, { "gather+vnni", iq3xxs<true, true> }, { "plugin fast_iq3 vnni", fastk3::vec_dot_iq3_xxs_vnni }, { "plugin fast_iq3 avx2", fastk3::vec_dot_iq3_xxs_avx2 } } },
    };
    printf("cpu: avx2=%d avx_vnni=%d avx512(bw,vl,dq)=%d avx512_vnni=%d\n", (int) __builtin_cpu_supports("avx2"), (int) __builtin_cpu_supports("avxvnni"),
           (int) fastk512::cpu_has_avx512(), (int) fastk512::cpu_has_avx512_vnni());
    if (fastk512::cpu_has_avx512()) {
        cases[0].v.push_back({ "avx512", w512_iq2 });
        cases[1].v.push_back({ "avx512", w512_iq3 });
        if (fastk512::cpu_has_avx512_vnni()) {
            cases[0].v.push_back({ "avx512+vnni", w512_iq2v });
            cases[1].v.push_back({ "avx512+vnni", w512_iq3v });
        }
    } else {
        printf("no AVX-512 on this CPU: the AVX-512 variants are skipped\n");
    }
    const int64_t n = 4096, nrows = 64;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> src((size_t) n * nrows), ones(n, 1.0f), act(n);
    for (auto & v : src) v = nd(rng);
    int bad_total = 0;
    for (auto & c : cases) {
        const traits_cpu * t = traits(c.type);
        const size_t rs = row_size(c.type, n), as = row_size(t->vec_dot_type, n);
        q_init(c.type);
        std::vector<uint8_t> w(rs * nrows), wr(rs * nrows), a(as + 64);
        if (q_chunk(c.type, src.data(), w.data(), 0, nrows, n, ones.data()) != rs * nrows) { fprintf(stderr, "quantiser failed for %s\n", c.name); return 3; }
        std::mt19937_64 r2(7);
        for (auto & b : wr) b = (uint8_t) r2();                       // random bytes: every grid index / sign pattern / scale
        for (int64_t r = 0; r < nrows; r++) for (int64_t b = 0; b < n / 256; b++) { uint8_t * blk = wr.data() + r * rs + b * (rs / (n / 256)); blk[0] = 0x00; blk[1] = 0x30; }   // finite fp16 d
        auto tf = traits(t->vec_dot_type);
        // ---- bit-exactness against the library function ----
        for (auto & v : c.v) {
            int bad = 0; long cmp = 0;
            for (int it = 0; it < 40; it++) {
                for (auto & x : act) x = nd(rng);
                tf->from_float(act.data(), a.data(), n);
                for (int pass = 0; pass < 2; pass++) for (int64_t r = 0; r < nrows; r++) {
                    const uint8_t * row = (pass ? wr.data() : w.data()) + r * rs;
                    float ref = 0, got = 0;
                    t->vec_dot((int) n, &ref, 0, row, 0, a.data(), 0, 1);
                    v.fn((int) n, &got, 0, row, 0, a.data(), 0, 1);
                    cmp++;
                    if (memcmp(&ref, &got, 4) != 0) bad++;
                }
            }
            printf("%-8s %-20s bit-exact vs library: %s (%ld dots, %d mismatches)\n", c.name, v.name, bad ? "NO" : "yes", cmp, bad);
            bad_total += bad;
        }
        if (check_only) continue;
        // ---- timing: library function and variants, interleaved rounds ----
        tf->from_float(act.data(), a.data(), n);
        std::vector<Variant> all = { { "library", t->vec_dot } };
        for (auto & v : c.v) all.push_back(v);
        const int reps = 20000, rounds = 7;
        std::vector<std::vector<double>> ns(all.size());
        volatile float sink = 0; float s = 0;
        for (int r = 0; r < rounds; r++) for (size_t k = 0; k < all.size(); k++) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; i++) { all[k].fn((int) n, &s, 0, w.data() + (size_t) (i % nrows) * rs, 0, a.data(), 0, 1); sink += s; }
            ns[k].push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps * 1e9);
        }
        printf("%-8s %-20s %10s %10s %12s\n", c.name, "variant", "median ns", "min ns", "G weights/s");
        double base = 0;
        for (size_t k = 0; k < all.size(); k++) {
            std::sort(ns[k].begin(), ns[k].end());
            const double med = ns[k][rounds / 2];
            if (k == 0) base = med;
            printf("%-8s %-20s %10.0f %10.0f %12.2f   x%.2f\n", c.name, all[k].name, med, ns[k][0], n / med, base / med);
        }
    }
    return bad_total ? 1 : 0;
}
