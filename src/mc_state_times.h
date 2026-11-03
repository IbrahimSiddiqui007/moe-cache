// CLEANUP-002: timing counters (gaps, E1 times, executor times)
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_times_t {
    // MOE_CACHE_SEG_TIMES=1 (N23-F6): where the time outside our own work goes. gap = from the end of one of our graph splits to the start of the next
    // (llama.cpp's other splits, the scheduler, copies, GPU waits), sync = the wait for inputs read asynchronously from the GPU (it contains the GPU
    // time of the split before us), gpu_seg = long-prompt GPU blocks incl. uploads. gap buckets: < 0.2 ms, < 1 ms, < 5 ms, >= 5 ms.
    std::atomic<uint64_t> ns_gap{0}, ns_sync{0}, ns_gpu_seg{0}, n_gap{0}, last_exit_ns{0};
    std::atomic<uint64_t> gap_b_ns[4]{}, gap_b_n[4]{};
    std::atomic<uint64_t> gap_faults{0}, in_faults{0}, last_exit_faults{0}, entry_faults{0}, top_gap_ns[4]{}, top_gap_graph[4]{}, top_gap_faults[4]{};   // faults of the process inside the gaps / inside our backend; the 4 longest gaps
    // MOE_CACHE_E1_TIMES=1 (with MOE_CACHE_SEG_TIMES=1; E1 step 1): what the gaps and the long-prompt GPU blocks are made of.
    // cpu: CPU time of the whole process (all threads) and of the graph thread alone, inside the gaps and inside our backend (a gap whose graph-thread CPU time is
    // far below its wall time is waiting, not computing). blk[0] = blocks run in one part, blk[1] = blocks split into parts:
    // wait = wait_experts inside the upload, up_faults = page faults during the upload, prep = from entering the block to the first upload,
    // rows_comp = expert-matmul output rows the GPU computed, rows_use = rows kept (one per token and expert slot).
    std::atomic<uint64_t> e1_gap_cpu_ns{0}, e1_in_cpu_ns{0}, e1_gap_tcpu_ns{0}, e1_in_tcpu_ns{0}, e1_last_cpu{0}, e1_last_tcpu{0}, e1_ent_cpu{0}, e1_ent_tcpu{0};
    std::atomic<uint64_t> e1_blk_n[2]{}, e1_parts[2]{}, e1_up_ns[2]{}, e1_comp_ns[2]{}, e1_get_ns[2]{}, e1_wait_ns[2]{}, e1_up_faults[2]{}, e1_prep_ns[2]{}, e1_rows_comp[2]{}, e1_rows_use[2]{};
    std::atomic<uint64_t> ex_quant_ns{0}, ex_build_ns{0}, ex_run_ns{0}, ex_part_ns[3]{}, ex_part_bytes[3]{}, ex_part_n[3]{};   // MOE_CACHE_EXEC_TIMES: executor phases; part 0 gate, 1 up, 2 down
};
