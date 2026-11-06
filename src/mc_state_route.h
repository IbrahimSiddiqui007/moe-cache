// CLEANUP-002: approximate routing and fewer-experts members
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_route_t {
    // approximate routing (MOE_CACHE_ROUTE=skip|subst, decode and small batches, executor only; off by default; quality-gated, see docs/internal/EXPERIMENTS.md):
    // the lowest-ranked picks of a token that are NOT in RAM are dropped (skip) or replaced by the most recently used resident expert of that layer (subst).
    // Ranked picks are in order of the router's weights, so the experts given up are the ones that contribute least.
    int  route_mode = 0;                                // 0 off, 1 skip, 2 subst
    int  route_keep = -1;                               // MOE_CACHE_ROUTE_KEEP: picks always honoured (default: all but the last route_drop)
    int  route_drop = 2;                                // MOE_CACHE_ROUTE_DROP: lowest-ranked picks that may be skipped (--fast=2: 4)
    uint64_t route_skipped = 0, route_substituted = 0, route_considered = 0;
    // fewer experts (MOE_CACHE_FEWER=N, approximate, B3): on layers the executor computes (CPU-held), decode and small batches use only the top n_used-N picks
    // (never fewer than half), whether they are in RAM or not. Prompt batches keep all picks. The router's weights are not renormalised (the dropped
    // experts are the lowest-ranked ones); MOE_CACHE_FEWER_RENORM=1 is not implemented, quality is judged by the gate (docs/QUALITY_GATE.md, level fewer).
    int  fewer = 0;
    uint64_t fewer_dropped = 0, fewer_considered = 0;
    std::unordered_map<const ggml_tensor *, std::vector<int32_t>> route_ids;   // ids tensor -> effective expert per (token, pick); -1 = skipped
};
