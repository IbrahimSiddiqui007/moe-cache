# 0080: CPU gate+up+SwiGLU fusion for decode-time MoE layers, and per-split / per-op timers (R4, 2026-11-08)

Two things in one patch (base b12064, both files `ggml/src/ggml-backend.cpp` and `ggml/src/ggml-cpu/ggml-cpu.c`):

1. **Timers (attribution only, never for speed numbers).** `GGML_R4_PROF=1`: wall clock of the host calls per scheduler split; `=2`: also synchronises after every device split (true GPU time); `=3`: per-op time, barrier time and the split of `mul_mat_id` into "prep + first barrier" and "own expert loop" on thread 0 of the CPU pool. Prints one `R4PROF` / `R4CPU` line per 100 decode tokens to stderr. Useful again for any hybrid CPU/GPU decode question.
2. **`GGML_CPU_FUSE_MOE=1`**: `MUL_MAT_ID(gate)` + `MUL_MAT_ID(up)` + `GLU(swiglu)` in one CPU node for one decode token (chunks start at multiples of 8 rows so the SwiGLU vector body sees the same elements as the whole-row version). One quantisation of the activation, one row mapping, one barrier instead of five.

## Result (laptop i7-13620H + RTX 4060 8 GB, CUDA, Qwen3.6-35B-A3B IQ3_XXS, `-ngl 99 --n-cpu-moe 30 -ub 2048`, q8_0 KV, 6 pinned threads)
| Check | Result |
|---|---|
| Identical output | yes: greedy text identical on 3 prompts x 200 tokens, both repeats |
| Speed, fast phase of the machine (server probe, 2 pairs) | unfused 22.6 / 22.3, fused 24.2 / 23.2 tok/s (+4 to +7 %) |
| Speed, slow phase (llama-bench A B B A, 6 pairs) | unfused 11.2 mean, fused 10.4 mean (-7 %); CPU node time unchanged (42-48 vs 43-53 ms per token) |
| Why it is not a clear win | thread 0 spends about 45 % of the CPU time per token waiting at barriers for the slowest thread (foreign load on the laptop); fewer barriers did not remove the skew |

## Restore
`git apply archive/llama.cpp/0080-cpu-moe-glu-fuse-and-split-profile.patch` in a llama.cpp checkout at b12064 (plain `patch -p1` also works); build as usual; run with `GGML_CPU_FUSE_MOE=1`.
Retest if: the CPU pool stops losing time at barriers (quiet machine, or a barrier-free expert-parallel scheme where each thread does whole experts), or a CPU with more cores than the laptop's.
