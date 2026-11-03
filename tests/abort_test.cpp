// SERVER-012: the plugin's abort callback (src/mc_abort.h). No ggml, no model: the chunk loop below is the shape of exec_pool_t::work (claim a chunk,
// skip it when a stop is requested, count it as done so the barrier ends).
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/abort_test.cpp -o /tmp/abort_test -lpthread && /tmp/abort_test
#include "mc_abort.h"
#include <cstdio>
#include <thread>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

static std::atomic<bool> g_cancel{false};
static std::atomic<int> g_asked{0};
static bool cb_flag(void * d) { g_asked++; return ((std::atomic<bool> *) d)->load(); }

// returns chunks computed; `done` ends at `total` also when chunks were skipped
static int run_job(mcabort::state_t & a, int total, int threads, int cancel_after, std::atomic<int> & done) {
    std::atomic<int> next{0}, computed{0};
    auto work = [&] {
        int fin = 0;
        for (;;) {
            const int c = next.fetch_add(1);
            if (c >= total) { break; }
            if (mcabort::stop_requested(a)) { fin++; continue; }
            if (computed.fetch_add(1) + 1 == cancel_after) { g_cancel = true; }     // the user presses stop during chunk `cancel_after`
            fin++;
        }
        done.fetch_add(fin);
    };
    std::vector<std::thread> th;
    for (int i = 1; i < threads; i++) { th.emplace_back(work); }
    work();
    for (auto & t : th) { t.join(); }
    return computed.load();
}

int main() {
    mcabort::state_t a;
    CHECK(!mcabort::stop_requested(a));                          // no callback: never stops
    mcabort::set(a, cb_flag, &g_cancel);
    CHECK(a.gen.load() == 1 && !mcabort::stop_requested(a));     // callback says no
    g_cancel = true;
    CHECK(mcabort::stop_requested(a));
    g_cancel = false;
    CHECK(mcabort::stop_requested(a));                           // sticky within the graph: every thread gets the same answer
    const int asked = g_asked.load();
    CHECK(mcabort::stop_requested(a) && g_asked.load() == asked); // and the callback is not called again
    mcabort::begin_graph(a);
    CHECK(!mcabort::stop_requested(a));                          // next graph: asked afresh
    mcabort::set(a, nullptr, nullptr);
    CHECK(a.gen.load() == 2 && !mcabort::stop_requested(a));     // llama.cpp clears the callback with nullptr

    // a job of 1000 chunks on 4 threads; the stop comes during chunk 10: far fewer than 1000 chunks run, the barrier still ends, the next graph runs fully
    mcabort::set(a, cb_flag, &g_cancel);
    g_cancel = false;
    std::atomic<int> done{0};
    const int computed = run_job(a, 1000, 4, 10, done);
    CHECK(done.load() == 1000);
    CHECK(computed >= 10 && computed < 100);
    CHECK(mcabort::stop_requested(a));
    g_cancel = false;
    mcabort::begin_graph(a);
    done = 0;
    CHECK(run_job(a, 1000, 4, 1 << 30, done) == 1000 && done.load() == 1000);   // not cancelled: every chunk runs
    if (fails) {
        fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    printf("abort checks passed (%d of 1000 chunks ran after the stop)\n", computed);
    return 0;
}
