// CLEANUP-002: cache budget, memory limit and eviction-protection members
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_budget_t {
    size_t budget = 0;
    bool   commit_warned = false;      // Windows: the commit-limit warning was printed (tune())
    bool   commit_fail_warned = false; // Windows: a refused expert commit was reported (commit_for())
    uint64_t commit_evictions = 0;     // experts evicted because Windows refused a commit or system commit ran low
    size_t budget_peak = 0;          // largest budget so far; a fall below half of it is reported once (shrink_warned)
    bool shrink_warned = false;
    bool   auto_mode = false;          // MOE_CACHE_SIZE_GIB=auto: size the cache from the real memory limit
    size_t limit_bytes = 0;
    size_t margin = (size_t) 1 << 30;  // keep this much below the limit (MOE_CACHE_MARGIN_GIB, default 1 GiB)
    size_t pending_bytes = 0;          // expert bytes marked resident but not read yet
    size_t max_budget = 0;             // all registered expert bytes
    size_t min_budget = 0;
    uint64_t tune_shrinks = 0, tune_grows = 0;
    size_t resident_bytes = 0;
    uint64_t epoch = 0;
    // experts with epoch >= protect_epoch are used by the graph being prepared/computed: no eviction may take them. All groups of a graph are
    // prepared before any is computed, so protecting only the current group (epoch == epoch) let a later group (or tune() shrinking the budget)
    // evict an earlier group's experts before llama.cpp read them: decommitted memory on Windows (access violation, PC 2026-10-27), zeros on Linux
    uint64_t protect_epoch = UINT64_MAX;
    uint64_t cpu_check_reloads = 0;                     // experts cpu_path_check() had to load again (should stay 0)
};
