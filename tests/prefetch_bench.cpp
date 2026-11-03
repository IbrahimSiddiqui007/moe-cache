// R-02 bench: does asking for the next rows ahead lift the read rate of a row-by-row dot product? Not part of the quick set (it needs RAM and time).
//   g++ -O2 -march=native -std=c++17 -Isrc tests/prefetch_bench.cpp -o /tmp/prefetch_bench -lpthread && /tmp/prefetch_bench [MiB=1024] [row_bytes=2048] [threads=4]
// Each thread takes a chunk of rows of one big buffer, like the executor's chunks, and sums int8 products (memory bound). Prints GB/s for ahead = 0,1,2,4,8,16.
#include "mc_prefetch.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static int64_t dot(const int8_t * a, const int8_t * b, size_t n) { int64_t s = 0; for (size_t i = 0; i < n; i++) s += (int) a[i] * (int) b[i]; return s; }

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? (size_t) atoll(argv[1]) : 1024, row = argc > 2 ? (size_t) atoll(argv[2]) : 2048;
    const int nth = argc > 3 ? atoi(argv[3]) : 4;
    const int64_t rows = (int64_t) (mib * 1048576 / row);
    std::vector<int8_t> w(rows * row), x(row);
    for (size_t i = 0; i < w.size(); i++) w[i] = (int8_t) (i * 2654435761u >> 24);
    for (size_t i = 0; i < row; i++) x[i] = (int8_t) (i % 7 - 3);
    printf("%zu MiB, rows of %zu B, %d threads\nahead  GB/s (best of 5)\n", mib, row, nth);
    for (int ahead : { 0, 1, 2, 4, 8, 16 }) {
        double best = 0; int64_t chk = 0;
        for (int rep = 0; rep < 5; rep++) {
            // flush: touch another buffer so the weights are not in cache
            static std::vector<char> junk(256u << 20, 1); for (size_t i = 0; i < junk.size(); i += 64) junk[i]++;
            std::vector<int64_t> sums(nth, 0);
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> th;
            for (int t = 0; t < nth; t++) th.emplace_back([&, t] {
                int64_t s = 0;
                for (int64_t c0 = t * 16; c0 < rows; c0 += 16 * nth)
                    for (int64_t r = c0; r < std::min<int64_t>(c0 + 16, rows); r++) {
                        if (ahead) mcpf::rows_ahead((const char *) w.data(), row, r, ahead, rows);
                        s += dot(&w[r * row], x.data(), row);
                    }
                sums[t] = s;
            });
            for (auto & h : th) h.join();
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            best = std::max(best, w.size() / sec / 1e9);
            chk = 0; for (auto s : sums) chk += s;
        }
        printf("%5d  %6.1f   (checksum %lld)\n", ahead, best, (long long) chk);
    }
    return 0;
}
