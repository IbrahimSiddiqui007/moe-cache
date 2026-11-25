// DECODE-123: always-on timer around evict() (how much of the prep time eviction itself costs).
// Header-only, no ggml, no GPU. Included by mc_state_livestats.h.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace mcevict {

struct timer_t {
    std::atomic<uint64_t> ns{0};        // total ns spent inside evict()
    std::atomic<uint64_t> ns_dec{0};    // of which during decode steps
    std::atomic<uint64_t> calls_dec{0}; // how many decode-step evict() calls
    void add(uint64_t d_ns, bool decode) {
        ns += d_ns;
        if (decode) {
            ns_dec += d_ns;
            calls_dec++;
        }
    }
};

struct scope_t {
    timer_t & t;
    bool dec;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~scope_t() {
        t.add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(), dec);
    }
};

} // namespace mcevict
