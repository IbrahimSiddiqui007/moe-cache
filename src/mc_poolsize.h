// DECODE-137: size of the executor's thread pool when -tb differs from -t. Header-only, no ggml (tests/poolsize_test.cpp).
//
// Stock: the pool is built once from the first graph's thread count. llama-server's first graph is usually a prompt, so with an explicit -tb
// the pool gets -tb threads and every decode step then runs on that many, not on -t. With MOE_CACHE_EXEC_POOL_DECODE=1 (default off) the pool is
// built as before for a first prompt graph (it has to work now), and rebuilt ONCE when the first single-token graph shows a different count.
// MOE_CACHE_EXEC_THREADS=N still wins over both. The thread count only decides how rows are shared out; every row is summed in the same order.
#pragma once

namespace mcpool {

enum class action_t { keep, build, rebuild };

struct decision_t {
    action_t act = action_t::keep;
    int threads = 0;        // pool size to build (compute threads, including the calling thread)
};

// have_pool / pool_n / pool_from_decode: what exists now. exec_threads: MOE_CACHE_EXEC_THREADS (0 = unset). n_threads: what llama.cpp asks for this
// graph. graph_tokens: the largest token count of a MUL_MAT_ID in the graph (1 = a decode step).
inline decision_t decide(bool enabled, bool have_pool, int pool_n, bool pool_from_decode, int exec_threads, int n_threads, long long graph_tokens) {
    decision_t d;
    const int want = exec_threads > 0 ? exec_threads : (n_threads > 0 ? n_threads : 1);
    if (!have_pool) {
        d.act = action_t::build;
        d.threads = want;
        return d;
    }
    if (enabled && !pool_from_decode && exec_threads <= 0 && graph_tokens == 1) {
        if (want != pool_n) {
            d.act = action_t::rebuild;
            d.threads = want;
        }
    }
    return d;
}

}  // namespace mcpool
