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
    // GUI-043: the last decode steps (one token's pass through the layers) that ran through the routing change above, with how many picks it changed in each.
    // The page maps them to the tokens of an answer ("at risk" marks). A step ends when the layer number does not grow any more (the next token starts) or when
    // nothing touched it for a while (the answer ended). Time: milliseconds on this process's own steady clock, taken when the step began; the stats line
    // carries "now_ms" from the same clock.
    struct step_rec_t { uint32_t ms; uint16_t changed; uint16_t n_tok; };
    static constexpr size_t STEP_RING = 512;
    step_rec_t step_ring[STEP_RING] = {};
    uint64_t steps_total = 0;                           // steps closed so far (the ring holds the last STEP_RING of them)
    uint32_t cur_step_changed = 0, cur_step_ms = 0, cur_step_last = 0;
    uint16_t cur_step_ntok = 0;
    int cur_step_layer = -1;
    bool cur_step_open = false;
    std::mutex step_mu;                                 // the compute thread writes, the snapshot thread reads
    static uint32_t step_now_ms() {
        static const auto t0 = std::chrono::steady_clock::now();
        return (uint32_t) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    void step_close_locked() {
        if (!cur_step_open) return;
        step_ring[steps_total % STEP_RING] = { cur_step_ms, (uint16_t) std::min<uint32_t>(cur_step_changed, 65535u), cur_step_ntok };
        steps_total++;
        cur_step_open = false; cur_step_changed = 0; cur_step_ntok = 0; cur_step_layer = -1;
    }
    void step_touch(int64_t n_tok, int layer) {         // the routing block ran for this layer
        std::lock_guard<std::mutex> lk(step_mu);
        const uint32_t now = step_now_ms();
        if (cur_step_open && layer <= cur_step_layer) step_close_locked();
        if (!cur_step_open) { cur_step_open = true; cur_step_ms = now; }
        cur_step_layer = layer; cur_step_last = now;
        if (n_tok > cur_step_ntok) cur_step_ntok = (uint16_t) std::min<int64_t>(n_tok, 65535);
    }
    void step_add(uint32_t n) { std::lock_guard<std::mutex> lk(step_mu); cur_step_changed += n; }
    void step_close_if_idle(uint32_t idle_ms) {         // the snapshot asks: a step nothing touched for a while is finished
        std::lock_guard<std::mutex> lk(step_mu);
        if (cur_step_open && step_now_ms() - cur_step_last > idle_ms) step_close_locked();
    }
};
