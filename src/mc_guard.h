// The exception guard around the long-prompt GPU block (audit AUD-002). Header-only, no ggml: tests/gpu_guard_test.cpp includes it with a throwing mock.
// ggml-vulkan reports a failed allocation (staging or pool memory another program took in the meantime) or a lost device by throwing a C++ exception.
// Nothing in the plugin's GPU block caught it, so the exception unwound through llama_decode and ended the server mid-request. The guard turns it into
// "this block stays on the CPU" (same tokens, slower) and remembers it so no further GPU block is tried.
#pragma once
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include "mc_env.h"

namespace mcguard {

// MOE_CACHE_GPU_CATCH: unset or empty -> 1 (catch, the default); 0/off/false/no -> 0 (do not catch, as before 0.3.6); 2 -> 2 (catch and inject one
// fault at the first block: for test runs); any other on-value -> 1.
inline int catch_mode(const char * v) {
    if (!v || !*v) {
        return 1;
    }
    if (!mcenv::flag_value(v, true)) {
        return 0;
    }
    char * end = nullptr;
    const long n = strtol(v, &end, 10);
    return (end != v && n >= 2) ? 2 : 1;
}

// Runs `run()` (returns bool: block done on the GPU). mode 0: no guard. Otherwise a std::exception out of it (or the injected one, once, in mode 2)
// is caught: `failed(what)` is called and the result is false. `injected` is the caller's once-only flag for mode 2.
template <class Failed>
void report(Failed && failed, const char * what) noexcept;

template <class Run, class Failed>
bool guarded(int mode, bool & injected, Run && run, Failed && failed) {
    if (mode <= 0) {
        return run();
    }
    try {
        if (mode >= 2 && !injected) {
            injected = true;
            throw std::runtime_error("injected by MOE_CACHE_GPU_CATCH=2");
        }
        return run();
    } catch (const std::exception & e) {
        report(failed, e.what());
        return false;
    } catch (...) {
        report(failed, "unknown exception");
        return false;
    }
}

// The failure handler must not throw out of the catch (it syncs and talks to the same GPU that just failed): anything it throws is dropped.
template <class Failed>
void report(Failed && failed, const char * what) noexcept {
    try {
        failed(what);
    } catch (...) {
    }
}

// Runs `undo` when the scope ends, also when an exception unwinds it, unless release() ran. Used to put graph nodes back (op switched to NONE for the GPU,
// buffer tagged for the repack kernel) so an exception can never leave a node of llama.cpp's graph changed.
template <class Undo>
struct scope_undo {
    Undo undo;
    bool armed = true;
    explicit scope_undo(Undo u) : undo(std::move(u)) {}
    scope_undo(const scope_undo &) = delete;
    scope_undo & operator=(const scope_undo &) = delete;
    void run() {            // undo now (the normal path); the destructor then does nothing
        if (armed) {
            armed = false;
            undo();
        }
    }
    ~scope_undo() {
        if (armed) {
            try { undo(); } catch (...) {}
        }
    }
};
template <class Undo>
scope_undo<Undo> make_undo(Undo u) { return scope_undo<Undo>(std::move(u)); }

}  // namespace mcguard
