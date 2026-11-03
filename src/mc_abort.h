// The abort callback of the plugin backend (ledger SERVER-012). Header-only, no ggml: tests/abort_test.cpp includes it.
// llama.cpp hands a callback to every backend (llama_set_abort_callback, found through the registry as "ggml_backend_set_abort_callback") and the server
// sets it for a cancelled request. The stock CPU backend asks it between graph nodes and between chunks. The plugin's executor runs the expert matrix
// products itself, so it must ask too: without this a cancelled long prompt finished the plugin's share of the current micro-batch (all expert layers).
// Rule: once the callback says "stop" the answer sticks until the next graph starts (so every thread sees the same answer), the remaining chunks are
// skipped (the barrier still completes) and the graph returns GGML_STATUS_ABORTED, which llama.cpp turns into "decode aborted" and discards the output.
#pragma once
#include <atomic>

namespace mcabort {

typedef bool (*callback_t)(void * data);

struct state_t {
    std::atomic<callback_t> cb{nullptr};
    std::atomic<void *>     data{nullptr};
    std::atomic<unsigned>   gen{0};          // bumped at every set: a backend that forwards the callback to another one re-sends it when this changed
    std::atomic<bool>       tripped{false};  // sticky within one graph
};

inline state_t & global() {
    static state_t s;
    return s;
}

inline void set(state_t & s, callback_t cb, void * data) {
    s.cb.store(nullptr);
    s.data.store(data);
    s.cb.store(cb);
    s.tripped.store(false);
    s.gen.fetch_add(1);
}

// A new graph starts: the previous answer is forgotten (the server clears its flag for the next request).
inline void begin_graph(state_t & s) {
    s.tripped.store(false, std::memory_order_relaxed);
}

// True when the work should stop. Cheap when no callback is set (one load) and after it tripped (one load); otherwise one call of the callback.
inline bool stop_requested(state_t & s) {
    if (s.tripped.load(std::memory_order_relaxed)) {
        return true;
    }
    const callback_t cb = s.cb.load(std::memory_order_acquire);
    if (!cb) {
        return false;
    }
    if (cb(s.data.load(std::memory_order_relaxed))) {
        s.tripped.store(true, std::memory_order_relaxed);
        return true;
    }
    return false;
}

}  // namespace mcabort
