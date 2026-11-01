# Phase memory: -ub 2048 room during the prompt, more GPU layers during decode (design note, 2026-11-01)

Source: handover/parts/scout_round2_20261101.md section 3 (R2-08 to R2-12). Nothing here is measured; gains are the scout's guesses.

## The problem
A big `-ub` helps the prompt and costs decode: the compute buffer (about 3 GB of VRAM at `-ub 2048`, ctx 128k) and the plugin's block buffer (about 1.1 GB, held after the first block, never shrunk) stay allocated in decode, where `--n-cpu-moe` has already fixed which layers sit on the GPU. Each GPU layer is worth about 1.5-4 % of decode (two noisy rows). Four fewer layers at `-ub 2048` is the visible loss.

## What can be changed at run time, and what cannot
| Piece | Who owns it | Can it change after load? |
|---|---|---|
| Which layers' experts are on the GPU (`--n-cpu-moe`, `-ot`) | llama.cpp loader | No. Weights are allocated once; moving a layer means freeing and re-uploading a weight buffer and re-reserving the scheduler. |
| Compute buffer (`sched_reserve`, sized for `min(n_ctx, n_ubatch)`) | llama.cpp context | Only by building a new scheduler (`sched_reserve` again, which `llama-context.cpp` already does when `sched_need_reserve` is set). The batch splitter and the plugin assume a fixed `cparams.n_ubatch`. |
| Plugin block buffer (`c->galloc`) | plugin | Yes: it can be freed and re-created (the gallocr only grows). |
| Plugin-held whole expert tensors (`MOE_CACHE_GPU_KEEP_MIB`) | plugin | Yes, but only the GPU block reads them, and the block is not used in decode (`MOE_CACHE_GPU_MIN_TOKENS=1` is refused: wrong output for single-token batches, T25). |

So freed VRAM in decode has **no consumer** today. Freeing the block buffer (R2-11) or shrinking the compute buffer (R2-09) alone gains nothing but host RAM.

## Options
| Option | What | Gain by guess | Size | Verdict |
|---|---|---|---|---|
| A. Free the block buffer at the first decode graph, re-create at the next prompt chunk (R2-11) | plugin, `mc_backend.inl` | 0 VRAM benefit alone; returns about 1.1 GB for a consumer | 0.5 day | Build only together with a consumer |
| B. Lower the effective `n_ubatch` for decode (R2-09) | patch in `llama-context.cpp` / server: when all slots decode, re-reserve for a small ubatch; raise it when a prompt chunk arrives | gives back up to 3 GB VRAM (1.5-4 layers) if a consumer exists | 2-3 days, risky: batch splitter, plugin sizes | Not small |
| C. A decode tier of whole layers kept by the plugin (R2-10) | after a prompt keep the most-missed layers (R2-08) on the GPU and run them through the GPU block in decode | +2 to +15 % decode | 1-2 weeks; needs the single-token bug understood first and the block to be cheap per token | The only real consumer; large |
| D. Choose the GPU layers by misses (R2-08) | launcher: `-ot` for the layers with the most `layer_misses` instead of `--n-cpu-moe` | +1-3 % if the spread is 1.5x | 0.5-1 day, launcher only | Cheapest; needs the stats first, not a patch |
| E. Do not trade: run `-ub 1024` with the layer count the planner gives for it | planner table fix (R2-02) | the scout's 2 x 2 test decides | 10 lines | Run the 2 x 2 test before any of A-C |

## Recommendation
1. Run T2 (the 2 x 2 test: `-ub` 1024/2048 x `--n-cpu-moe` 37/41) before building anything. If `-ub 1024` with 41 CPU layers also reaches the 2048 prompt speed, the gain was the emptier card, not the chunk size, and A-C are moot.
2. If the chunk size does matter, build D first (launcher, no patch), then C behind a switch. B and A are only useful as C's prerequisites.
3. Not built in this round: nothing in R2 section 3 is small. The 20-line diagnostic R2-05 (minimum free VRAM and median versus worst block time in one end-of-prompt line) would make the next sweep readable and is the cheapest next code change.

## Update 2026-11-01 (prompt-memory session)
- The compute buffer is mostly logits, not activations: on LLaDA at `-ub 2048` the logits matrix is 1228 MiB of 1492 MiB (patch 0040 prints it; Flash-Next would be about 1.9 GiB of 3 GiB by arithmetic). Patch 0042 (`LLAMA_N_OUTPUTS_MAX`) shrinks it with no weight movement; this is a cheaper consumer-free gain than options A-C above and should be tried first.
- The output head is 417 MiB on Flash-Next (half a layer); `tools/head_lend_math.py` gives the arithmetic. Lending it is not built (patches/llama.cpp/README.md, section U09).
- P19 scratch sharing is built as 0041 (prototype, off); on LLaDA it saves 0 MiB.
