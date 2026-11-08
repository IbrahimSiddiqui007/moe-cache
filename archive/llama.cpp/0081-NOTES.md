# 0081: dynamic row chunks for decode-time `mul_mat_id` on the CPU (R5, 2026-11-08)

`GGML_CPU_MMID_ROWS=N` (unset / 0 = stock): for one row of src1 (a decode token) each expert is cut into chunks of N rows that the threads claim from an atomic counter, instead of one static chunk per thread. Per-row maths unchanged, so the output is bit-identical (greedy text identical in 9 of 9 paired server runs).

## Result (laptop i7-13620H + RTX 4060 8 GB, Qwen3.6-35B-A3B IQ3_XXS, `-ngl 99 --n-cpu-moe 30 -ub 2048`, 6 pinned threads)
| Test | Stock | N = 16 | Plugin executor (for reference) |
|---|---|---|---|
| llama-bench, noisy phase, 4 pairs | 12.8 | 15.1 (+17 %) | - |
| server probe, quiet phase, 5 runs each | 28.06 | 27.96 | - |
| server probe, controlled quiet, A B B A | 24.2 | 23.1 | 27.3 (+13 %) |
| server probe, one pinned core kept 100 % busy by another process, A B B A | 13.4 | 13.4 | 18.8 (+40 %) |
The first row was the machine's slow phase, not the patch: with a controlled disturbance the patch gives nothing. Strict thread pinning plus the barrier between nodes still waits for the thread that shares its core; the plugin's pool waits for chunks, not threads.

## Restore
`patch -p1 < archive/llama.cpp/0081-cpu-mmid-dynamic-rows.patch` in a llama.cpp checkout at b12064 (or copy to `patches/llama.cpp/`). Retest if: threads are no longer strictly pinned, or the barrier is made chunk-based like the plugin pool's.
