// E3 bench: how close does the executor's Q2_0 expert phase get to the memory bus, and what do the per-op barriers cost?
//   g++ -O2 -std=c++17 -mavx2 -mfma -pthread -Isrc tests/exec_bw_bench.cpp -o /tmp/exec_bw_bench && /tmp/exec_bw_bench [threads] [GiB]
// Shape of one decode step's expert phase: 10 experts, 2048 rows x 4096 columns of Q2_0 each, three ops (gate, up, down) with one activation column.
// Runs (each ~1 s): read-only sum over the same bytes (the bus ceiling for this access pattern), the Q2_0 dot with one pool job per op (what the plugin does),
// and the same dots with gate+up in one job (MOE_CACHE_EXEC_FUSE). Weights come from a pool of random experts bigger than the cache so reads come from DRAM.
#include "fast_q2_0.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static const int NE00 = 4096, ROWS = 2048, CHUNK = 128, PICKS = 10;
static const size_t ROWB = (size_t) NE00 / 64 * sizeof(fastk::block_q2_0);

struct task_t { const char * w; float * dst; };
struct job_t { const task_t * t; int n_tasks; int nch; std::atomic<int> next{0}, done{0}; int mode; const fastk::block_q8_0_perm * act; };

static inline void cpu_relax() { _mm_pause(); }

struct pool_t {
    std::vector<std::thread> th;
    std::atomic<uint32_t> gen{0};
    std::atomic<bool> stop{false};
    std::atomic<job_t *> cur{nullptr};
    std::atomic<int> inside{0};
    static void work(job_t * j) {
        for (;;) {
            const int c = j->next.fetch_add(1, std::memory_order_relaxed);
            if (c >= j->n_tasks * j->nch) return;
            const task_t & t = j->t[c / j->nch];
            const int r0 = (c % j->nch) * CHUNK, r1 = r0 + CHUNK;
            if (j->mode == 0) {                 // read-only: sum every 32-byte word
                __m256i acc = _mm256_setzero_si256();
                for (int r = r0; r < r1; r++) {
                    const char * p = t.w + (size_t) r * ROWB;
                    for (size_t o = 0; o < ROWB; o += 32) acc = _mm256_add_epi8(acc, _mm256_loadu_si256((const __m256i *) (p + o)));
                }
                alignas(32) uint8_t tmp[32];
                _mm256_store_si256((__m256i *) tmp, acc);
                t.dst[r0] = (float) tmp[0];
            } else {
                for (int r = r0; r < r1; r++) fastk::vec_dot_q2_0_perm_avx2(NE00, &t.dst[r], 0, t.w + (size_t) r * ROWB, 0, j->act, 0, 1);
            }
            j->done.fetch_add(1, std::memory_order_release);
        }
    }
    void start(int n) {
        for (int i = 0; i < n; i++) th.emplace_back([this] {
            uint32_t seen = gen.load();
            for (;;) {
                while (gen.load(std::memory_order_acquire) == seen) { if (stop.load()) return; cpu_relax(); }
                seen = gen.load();
                inside.fetch_add(1);
                job_t * j = cur.load(std::memory_order_acquire);
                if (j) work(j);
                inside.fetch_sub(1);
            }
        });
    }
    void run(job_t * j) {
        while (inside.load() > 0) cpu_relax();
        j->next = 0; j->done = 0;
        cur.store(j, std::memory_order_release);
        gen.fetch_add(1);
        work(j);
        while (j->done.load(std::memory_order_acquire) < j->n_tasks * j->nch || inside.load() > 0) cpu_relax();
    }
    ~pool_t() { stop = true; for (auto & t : th) t.join(); }
};

int main(int argc, char ** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 16;
    const double gib = argc > 2 ? atof(argv[2]) : 1.5;
    const size_t expert_b = (size_t) ROWS * ROWB, n_exp = (size_t) (gib * 1073741824.0) / expert_b;
    std::vector<char> big(n_exp * expert_b);
    std::mt19937_64 rng(7);
    for (size_t i = 0; i + 8 <= big.size(); i += 8) { uint64_t v = rng(); memcpy(&big[i], &v, 8); }
    for (size_t b = 0; b + sizeof(fastk::block_q2_0) <= big.size(); b += sizeof(fastk::block_q2_0)) ((fastk::block_q2_0 *) &big[b])->d = 0x2c00;   // sane scale
    std::vector<fastk::block_q8_0> a8(NE00 / 32);
    for (auto & b : a8) { b.d = 0x2c00; for (auto & q : b.qs) q = (int8_t) (rng() % 200 - 100); }
    std::vector<fastk::block_q8_0_perm> act(NE00 / 32);
    fastk::permute_q8_0(a8.data(), act.data(), NE00 / 32);
    std::vector<float> out(3 * PICKS * ROWS + 64);
    pool_t pool; pool.start(T - 1);
    printf("threads %d, %zu experts of %.2f MB (%.2f GiB), %d picks per op, chunk %d rows\n", T, n_exp, expert_b / 1e6, big.size() / 1073741824.0, PICKS, CHUNK);
    auto bench = [&](const char * name, int mode, int fuse) {
        std::mt19937 pr(3);
        const auto t0 = std::chrono::steady_clock::now();
        double bytes = 0; int steps = 0;
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(1200)) {
            task_t tk[3][PICKS];
            for (int op = 0; op < 3; op++) for (int p = 0; p < PICKS; p++) tk[op][p] = { &big[(pr() % n_exp) * expert_b], &out[((size_t) op * PICKS + p) * ROWS] };
            job_t j[3];
            for (int op = 0; op < 3; op++) { j[op].t = tk[op]; j[op].n_tasks = PICKS; j[op].nch = ROWS / CHUNK; j[op].mode = mode; j[op].act = act.data(); }
            if (fuse) {   // gate+up: one job over 2*PICKS tasks (tk[0] and tk[1] are adjacent in memory)
                j[0].n_tasks = 2 * PICKS;
                pool.run(&j[0]); pool.run(&j[2]);
            } else {
                for (int op = 0; op < 3; op++) pool.run(&j[op]);
            }
            bytes += 3.0 * PICKS * expert_b; steps++;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("%-34s %7.2f GB/s   %6.2f steps/s (%.2f ms per step)\n", name, bytes / s / 1e9, steps / s, 1000.0 * s / steps);
    };
    {   // exactness: the same tasks as 3 jobs and as gate+up fused must give bit-identical outputs
        task_t tk[3][PICKS];
        std::mt19937 pr(11);
        std::vector<float> o1(out.size()), o2(out.size());
        auto fill = [&](std::vector<float> & o) { for (int op = 0; op < 3; op++) for (int p = 0; p < PICKS; p++) tk[op][p].dst = &o[((size_t) op * PICKS + p) * ROWS]; };
        for (int op = 0; op < 3; op++) for (int p = 0; p < PICKS; p++) tk[op][p].w = &big[(pr() % n_exp) * expert_b];
        job_t j[3];
        for (int op = 0; op < 3; op++) { j[op].t = tk[op]; j[op].n_tasks = PICKS; j[op].nch = ROWS / CHUNK; j[op].mode = 1; j[op].act = act.data(); }
        fill(o1); for (int op = 0; op < 3; op++) pool.run(&j[op]);
        fill(o2); j[0].n_tasks = 2 * PICKS; pool.run(&j[0]); pool.run(&j[2]);
        printf("fused vs separate outputs: %s\n", memcmp(o1.data(), o2.data(), 3 * PICKS * ROWS * sizeof(float)) == 0 ? "bit-identical" : "DIFFERENT");
    }
    bench("read-only sum, 3 jobs per step", 0, 0);
    bench("read-only sum, gate+up fused", 0, 1);
    bench("Q2_0 dot, 3 jobs per step", 1, 0);
    bench("Q2_0 dot, gate+up fused (2 jobs)", 1, 1);
    bench("Q2_0 dot, 3 jobs per step (again)", 1, 0);
    bench("Q2_0 dot, gate+up fused (again)", 1, 1);
    return 0;
}
