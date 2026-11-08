// CLEANUP-002: GPU device, hybrid and long-prompt block counters
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_gpu_t {
    ggml_backend_dev_t gpu_dev = nullptr;               // first GPU backend device, used for long prompts
    ggml_backend_dev_t gpu_dev2 = nullptr;              // MOE_CACHE_GPU_DEV2 (default off, PROTOTYPE): a second card for long-prompt blocks that do not fit on the first
    std::atomic<uint64_t> gpu_alt_blocks{0};            // blocks that ran on the second card
    int    gpu_min_tokens = 32;                         // batches of at least this many tokens run their expert block on the GPU (0 = never)
    std::atomic<uint64_t> hyb_groups{0}, hyb_picks{0}, hyb_hits{0};   // MOE_CACHE_HYBRID: layers split between GPU bank and CPU, picks, picks served by the bank
    std::atomic<uint64_t> hyb_drops{0}, hyb_reloads{0}, hyb_reload_ns{0};
    std::atomic<uint64_t> hyb_swaps{0}, hyb_swap_ns{0}, hyb_reverts{0};   // MOE_CACHE_HYBRID_ADAPT: experts swapped into the bank, time spent uploading them   // bank freed for a long prompt's GPU block, loaded again at the next decode step
    std::atomic<uint64_t> hyb_ph_ns[7] = {};            // hybrid_group phase time: scan, build graph, input copies, launch, CPU misses, GPU wait, read-back (ns, summed over layer steps)
    std::atomic<uint64_t> gpu_segments{0}, gpu_fallbacks{0}, gpu_nomem{0}, gpu_split_blocks{0};   // gpu_nomem: blocks kept on the CPU because the GPU had too little free memory; gpu_split_blocks: blocks run in parts (N23)
    std::atomic<uint64_t> gpu_up_bytes{0}, gpu_up_saved{0};     // gpu_up_saved: idea P11, bytes whose upload a held GPU copy made unnecessary
};
