// mc_stable_vec: address-stable growable array (src/mc_stable_vec.h). No ggml, no GPU.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/stable_vec_test.cpp -o /tmp/claude-1000/c694/svt -lpthread
#include "mc_stable_vec.h"
#include <atomic>
#include <cstdio>
#include <thread>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    // address stability of v[0] while growing to 1,000,000 elements
    mcstable::stable_vec<int> v;
    v.resize(1);
    v[0] = 42;
    int* first = &v[0];
    v.resize(1000000);
    CHECK(&v[0] == first);
    CHECK(v[0] == 42);

    // size
    CHECK(v.size() == 1000000);

    // default values kept after growth: resize() must not disturb existing data
    for (size_t i = 1; i < v.size(); ++i) CHECK(v[i] == 0);
    v[123456] = 7;
    v.resize(2000000);
    CHECK(v.size() == 2000000);
    CHECK(v[123456] == 7);
    for (size_t i = 1000000; i < 2000000; ++i) CHECK(v[i] == 0);

    // grow-only: shrinking is a no-op
    v.resize(5);
    CHECK(v.size() == 2000000);

    // reader thread reads v[0..99] in a loop while main thread grows 100 times
    mcstable::stable_vec<long> w;
    w.resize(100);
    for (size_t i = 0; i < 100; ++i) w[i] = (long)i;
    std::atomic<bool> stop{false};
    std::atomic<long> reader_bad{0};
    long reads = 0;
    std::thread reader([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            for (size_t i = 0; i < 100; ++i) {
                if (w[i] != (long)i) reader_bad.fetch_add(1, std::memory_order_relaxed);
            }
            reads++;
        }
    });
    for (int g = 1; g <= 100; ++g) {
        w.resize(100 + (size_t)g * mcstable::stable_vec<long>::CHUNK);
        w[99 + (size_t)g * mcstable::stable_vec<long>::CHUNK] = 0;
    }
    stop.store(true);
    reader.join();
    CHECK(reader_bad.load() == 0);
    CHECK(reads > 0);
    CHECK(w.size() == 100 + 100 * mcstable::stable_vec<long>::CHUNK);

    if (fails) { fprintf(stderr, "%d checks failed\n", fails); return 1; }
    printf("stable-vec checks passed\n");
    return 0;
}
