# Archive: built, tested, did not win

Code that lost or gained nothing in a paired run leaves the program and stays here, so it can come back if the situation changes. Each file is a patch that holds the code and where it goes.

## Restore
Plugin: `git apply archive/plugin/NAME.patch` from the top of the repo. llama.cpp: copy the file back to `patches/llama.cpp/`. Later changes to the same lines may need a small manual fix.

## What is here
Results: Flash-Next Q2_0, 34k prompt, -ub 1024, RX 9070 XT + 7950X, Windows, 2026-11-01, one baseline against each arm in mirrored order. Baseline 252.1 / 251.9 tok/s prompt, decode about 27.7. Keep-or-drop runs, not chart numbers.

| File | Switch | What it did | Prompt tok/s | Decode | Why archived | Retest if |
|---|---|---|---|---|---|---|
| plugin/gpu_keep_pin.patch | `MOE_CACHE_GPU_KEEP_PIN=1` with `MOE_CACHE_GPU_KEEP_MIB=N` (in `src/mc_backend.inl`) | keeps the first expert tensors on the GPU between blocks, so they are not uploaded again | 251.3 / 251.5 | 27.6 | no gain | much more free VRAM, or uploads become the largest cost |
| plugin/gpu_whole_runs.patch | `MOE_CACHE_GPU_WHOLE_RUNS=N` (`src/mc_backend.inl`) | uploads a whole expert tensor in runs, each after its own wait | 247.4 / 249.9 | 27.7 | no gain | cold-disk waits inside uploads become large |
| plugin/gpu_rb_pipe.patch | `MOE_CACHE_GPU_RB_PIPE=N` (`src/mc_backend.inl`) | reads a block's result back in slices, overlapping host copy and transfer | 247.5 / 250.8 | 27.4 / 26.9 | no gain, decode lower | read-back becomes the largest cost |
| plugin/timer_1ms.patch | `MOE_CACHE_TIMER_1MS=1` (`src/platform.h`, `src/mc_state.inl`) | asks Windows for a 1 ms timer tick | 249.4 / 250.2 vs 248.7 / 250.7 | 27.4 vs 27.3 | no gain | a wait is shown to depend on the tick |
| llama.cpp/0001-vulkan-fence-spin-iters.patch | `GGML_VK_FENCE_SPIN_ITERS=N` (`ggml-vulkan.cpp`) | polls the GPU fence more often in the last 20 % of a graph | 247.4 / 247.2 | 27.8 / 28.0 | prompts about 2 % slower | never expected to gain much; another GPU or driver |

### Night run 2026-11-02 (quiet PC, Flash-Next Q2_0 34k, -ub 2048, GPU_SUM, 41 CPU layers; baseline prompt 501-539, decode 26.3-26.8)

| File | Switch | Result | Why archived | Retest if |
|---|---|---|---|---|
| llama.cpp/0021-qwen4exp-no-cont.patch | `LLAMA_QWEN4EXP_NO_CONT` | 511, decode 26.4 | no gain seen | upstream adopts it |
| llama.cpp/0023-vulkan-matmul-tile-switch.patch | `GGML_VK_AMD_LARGE_TILE`, `GGML_VK_MMID_SMALL_TILE` | 505, 519; large tile changes text | no gain | another GPU |
| llama.cpp/0024-vulkan-submit-flops-div.patch | `GGML_VK_SUBMIT_FLOPS_DIV` | 508 / 551 | inside noise | - |
| llama.cpp/0025-vulkan-gdn-pipeline.patch | `GGML_VK_GDN_PIPELINE` | 528 / 512 | inside noise | - |
| llama.cpp/0041-vulkan-prealloc-alias.patch | `GGML_VK_PREALLOC_ALIAS` | 524 | inside noise | 128k+ with a large scratch |
| llama.cpp/0060-vulkan-q2_0-int8-coopmat.patch | `GGML_VK_Q2_0_INT8` | 531, base 526-540 | no gain, approximate | - |
| llama.cpp/0061-vulkan-gdn-subgroup-size.patch | `GGML_VK_GDN_SUBGROUP` | 540 | no gain, not bit-identical | - |
| llama.cpp/0062-vulkan-rdna-split-k-large.patch | `GGML_VK_RDNA_SPLIT_K_LARGE` | 539 | no gain | - |
| llama.cpp/0063-vulkan-rdna-split-k-small-n.patch | `GGML_VK_RDNA_SPLIT_K_SMALL_N` | 516 | no gain | MTP verify batches |
| plugin/gpu_compute_async.patch | `MOE_CACHE_GPU_COMPUTE_ASYNC` | 503, decode 26.5 | no gain | - |
| plugin/pool_where_lite.patch | `MOE_CACHE_POOL_WHERE_LITE` | 514 / 541 | inside noise | - |
| plugin/pred_fast.patch, stats_lite.patch, pool_chunk_adapt.patch, sibling_order.patch | `MOE_CACHE_PRED_FAST`, `STATS_LITE`, `POOL_CHUNK_ADAPT`, `SIBLING_ORDER` | together with two others: 560 / 579 vs 557 / 575 | group gained nothing | - |

### PC 2026-11-02 late morning (owner's update running; pairs compared inside each pair)

| File | Switch | Result | Why archived | Retest if |
|---|---|---|---|---|
| llama.cpp/0026-vulkan-top-k-mask.patch | `GGML_VK_TOP_K_MASK` | 560 / 569 off, 565 / 571 on; decode same | no gain at 34k; it saves VRAM, which may matter at 256k | VRAM is short at 256k |
| llama.cpp/0050-layerwise-prefill.patch (design: llama.cpp/0050-LAYERWISE_DESIGN.md) | `LLAMA_LAYERWISE_PREFILL=N` | 2026-11-02: 344 / 347 vs 395 / 366 and 402 / 407 vs 448 / 448; upload stayed 652 GB with KEEP_MIB 1200, 2400, 4000 | 10 % slower and saved no upload: the plugin's held tensors never cut uploads in layer order | the plugin gets a keeper built for layer order (a day or two of work) |

Layer-wise prefill, port to b12064 (2026-11-06, PREFILL-638): `llama.cpp/0050-layerwise-prefill-b12064.patch` is the patch moved to the pin b12064 (four small context fixes, the `decode()` loop wrapped as before, `self_k_rot` / `self_v_rot` are now lists). It applies after 0004, 0020, 0030, 0040 and 0042 in name order, `libllama` builds on the laptop (CPU), and on the tiny random qwen4exp model (8 layers, 600 tokens, `-ub 128`, greedy) the logits of the last token and the next 8 steps equal the stock run exactly (max |diff| 0) for `LLAMA_LAYERWISE_PREFILL` = 2, 3 and 4; 1 is refused with one line. Checker: `llama.cpp/0050-tests/layerwise_check.cpp` and `layerwise_compare.py`. It is archived, not shipped: the PC re-test (A / B2 / B3 / B2 / A, `MOE_CACHE_GPU_KEEP_MIB` on, packing now off by itself) decides whether it comes back. To use it, copy the patch into `patches/llama.cpp/` on the `ci/llama-patched` branch and push (the workflow builds the Windows server).
| plugin/gpu_prefetch.patch (branch claude/block-prefetch-52b3eb, never merged) | `MOE_CACHE_GPU_PREFETCH=1` | 34k 598 / 593 off, 598 / 597 on; 128k 414 vs 418 | no gain: the gaps did not hide the upload | uploads become slower than compute |
| plugin/pin_compute.patch | `MOE_CACHE_PIN_COMPUTE=1` | two pairs, both drifted (381 -> 548, 511 -> 638 within the pair) | nothing provable in four tries | a steady PC |
| llama.cpp/0042-n-outputs-max.patch | `LLAMA_N_OUTPUTS_MAX=8` | at 39 CPU layers 585 / 598 off, 606 / 590 on | inside noise | VRAM is short at 128k+ |
| llama.cpp/0002-server-adaptive-draft-length.patch | `LLAMA_SPEC_ADAPT` | MTP decode 16.35 both | no gain | another draft model |
| llama.cpp/0003-vulkan-mmid-vec-max-tokens.patch | `GGML_VK_MMID_VEC_MAX_TOKENS` | 16.45 vs 16.2-16.35 | no gain | - |
| plugin/decode_tier.patch (design: docs/internal/DECODE_TIER_DESIGN.md; also deletes `src/mc_tier.h`, `tests/tier_test.cpp`) | `MOE_CACHE_DECODE_TIER=38,39` (+ `_CHECK`, `_KEEP_MIB`) | 2026-11-03, 64k ctx, 34k prompt, restarted PC: check mode exact (0 groups above 0.05); decode 26.71 / 26.22 vs 26.72 / 26.65, prompt 311 / 313 vs 488 / 585 | kill check: no decode gain, prompt -41 % (letting llama.cpp place 2 more layers gives +15 %: the gain needs the loader, see the engine re-check) | a llama.cpp patch keeps a decode-only GPU copy of layers |

### PC 2026-11-05 (IQ3_XXS, 64k prompt, timers off, A B B A + C + R)

| File | Switch | Result | Why archived | Retest if |
|---|---|---|---|---|
| llama.cpp/0070-decode-reloc.patch (design: llama.cpp/0070-DECODE_RELOC_DESIGN.md) + plugin/decode_reloc_launcher.patch (launcher hook `planner/decode_reloc.py`, the stage in `bin/moe_cache.py`, `tests/decode_reloc_test.py`, `tests/decode_reloc_live.*`, the CI steps, the `MOE_CACHE_DECODE_RELOC` row) | `LLAMA_DECODE_RELOC=1` (launcher: `MOE_CACHE_DECODE_RELOC=1`), DECODE-630 | moving 2 layers back to the GPU after the prompt: decode 11.41 vs 12.07 tok/s (-5.5 %), prompt 171.6 vs 184.0 (-6.8 %); shrink-only mode -21 %. The gain came from the smaller `-ub`, not from the layers (C and R +39.7 % / +48.6 % decode, prompts -16 / -21 %) | all three forms lost; the 64k decode loss is GPU memory spilling into system RAM at `-ub 2048` | a GPU with much more free VRAM after the prompt |

Restore: copy `llama.cpp/0070-decode-reloc.patch` back to `patches/llama.cpp/` and `llama.cpp/0070-DECODE_RELOC_DESIGN.md` to `docs/DECODE_RELOC_DESIGN.md`, then `git apply --exclude=README.md archive/plugin/decode_reloc_launcher.patch` from the top of the repo (the README hunk no longer applies since the docs restructure of 2026-11-06: leave it out; docs/CONFIGURATION.md rows and `docs/DECODE_RELOC_DESIGN.md` paths are the old ones, `docs/internal/` and `docs/results-archive/` hold the moved files) (it restores the launcher hook, tests, CI steps and the doc rows; later changes to `.github/workflows/llama_patched.yml`, `bin/moe_cache.py` or the README may need a small manual fix). The patch was checked with `git apply --check` on the tree it was removed from.

### Switches that stayed off, removed on 2026-11-06 (v0.3.8 cleanup; the Critic marked each safe on 2026-11-05)

Each patch is the removal read backwards: `git apply archive/plugin/NAME.patch` from the top of the repo puts the switch, its code and its `docs/CONFIGURATION.md` row back (checked with `git apply --check` on the tree each one was removed from; later edits of the same lines may need a small manual fix).

| File | Switch | Why archived | Retest if |
|---|---|---|---|
| plugin/reader_affinity.patch | `MOE_CACHE_READER_AFFINITY` (TEST-004) | off, no gain seen; ideas triage 2026-11-05 | reads queue behind compute threads on a many-core CPU |
| plugin/warm_sort.patch | `MOE_CACHE_WARM_SORT` (CACHE-129) | PC test 2026-11-05 (cache switch bundle): no gain, prompt -3.4 % | a disk that reads in file order much faster than at random |
| plugin/async_coalesce.patch | `MOE_CACHE_ASYNC_COALESCE` (CACHE-062) | same bundle, no gain. The merged-read code (`io_job_t::more`, `coalesced`) is shared with the synchronous path and stays | - |
| plugin/demand_first.patch | `MOE_CACHE_DEMAND_FIRST` and the `demand_pending` counter (CACHE-112) | same bundle, no gain | guessed reads are shown to delay needed ones |
| plugin/layer_share.patch | `MOE_CACHE_LAYER_SHARE` (CACHE-110) | built, never measured, off | one layer is shown to hold most of the cache |
| plugin/evict_siblings.patch | `MOE_CACHE_EVICT_SIBLINGS` (CACHE-110) | same | - |
| plugin/residency_check.patch | `MOE_CACHE_RESIDENCY_CHECK` (CACHE-110) | a diagnostic only (mincore sample at exit), off | the OS is suspected of reclaiming cached experts |
| plugin/refill_after_prompt.patch | `MOE_CACHE_REFILL_AFTER_PROMPT` (CACHE-110) | built, never measured, off | - |
| plugin/preload_skip_tokens.patch | `MOE_CACHE_PRELOAD_SKIP_TOKENS` (CACHE-110) | built, never measured, off | - |
| plugin/gpu_upload_threads.patch | `MOE_CACHE_GPU_UPLOAD_THREADS` (PREFILL-187) | PC test 2026-11-05: no gain; unsafe without resizable BAR | uploads are shown to be limited by one thread |
| plugin/gpu_stage_threads.patch | `MOE_CACHE_GPU_STAGE_THREADS` (CACHE-051) | PC test 2026-11-05: its parent `GPU_STAGE` lost; this only worked with it. `GPU_STAGE` itself is still in the tree (PREFILL-060 is only partly archived) | - |
| plugin/plan_cost.patch | `MOE_CACHE_PLAN_COST` (ENGINE-003) | off; ideas triage | the planner must pick `-ub 2048` at 64k+ with no table row |
| plugin/compute_db.patch | `MOE_CACHE_COMPUTE_DB`, `gguf_plan.py --record-buffer` (PATCH-010) | off, never used | per-model compute buffers must be recorded |
| plugin/idle_adapt.patch | `MOE_CACHE_IDLE_ADAPT` (AGENTS-015) | off; its own estimate is under 1 % | tool pauses are long and varied |
| plugin/proxy_keepalive.patch | `MOE_CACHE_PROXY_KEEPALIVE` (AGENTS-019) | off, setup cost per request is small against a model reply | many very short agent calls |
| llama.cpp/0005-tokenize-cache.patch (notes: llama.cpp/0005-tokenize-cache-NOTES.md) | `LLAMA_TOKENIZE_CACHE` (PATCH-022) | exact match only; an agent loop changes the text every request | the same long prompt arrives again and again |

Not touched on purpose: `MOE_CACHE_WIN_OPEN_HINT` (WINDOWS-672) stays until the 30-minute Windows read-path gate has run, and every row the Critic marked "SHIPPED, DO NOT REMOVE" or "archive only part" (PREFILL-184, PREFILL-060, CACHE-115, CACHE-131, MULTIGPU-001, -008, -011, LINUX-006, DECODE-059). DECODE-045, DECODE-057, PATCH-001 and PATCH-126 had no code left to move.

### Night chain 2026-11-07 (PC, kit k39c8e0b / v037rc, timers off; the chain's own keep/kill rules)

Removed on 2026-11-07; each patch is the removal read backwards (`git apply archive/plugin/NAME.patch` from the top of the repo; checked with `git apply --check` on the tree it was removed from).

| File | Switch | Result | Why archived | Retest if |
|---|---|---|---|---|
| plugin/gpu_extra_ops.patch | `MOE_CACHE_GPU_EXTRA_OPS` (PREFILL-054) | prompt -0.3 %, decode +0.4 %, text identical | no gain | a model with clamp/scale between the expert matrices on the GPU path |
| plugin/pool_spin_us.patch | `MOE_CACHE_POOL_SPIN_US` (DECODE-075) | 300 us: prompt +6.2 % (pairs +18.4 / -3.8), decode -0.1 %; 1000 us: +0.9 % / +0.1 % | inside the pair spread, decode flat | workers are shown to sleep between layers (`pool_parks` high) |
| plugin/prefill_lend_mib.patch (also removes `src/mc_lend.h`, `tests/lend_test.cpp`) | `MOE_CACHE_PREFILL_LEND_MIB` (PREFILL-032) | LEND_MIB=2048: prompt -2.6 % (pairs -1.2 / -4.0), decode +1.3 %, text identical | prompt lost | RAM pressure during prompts on a smaller machine |
| plugin/exec_pool_decode.patch (also removes `src/mc_poolsize.h`, `tests/poolsize_test.cpp`) | `MOE_CACHE_EXEC_POOL_DECODE` (DECODE-137) | `-t 16 -tb 32`: decode +0.6 % (pairs +0.3 / +0.8) | no gain | a CPU where `-tb` and `-t` differ much |
| plugin/probe_cache.patch | `MOE_CACHE_PROBE_CACHE` (ENGINE-021) | start-up saves 367 ms (off 372 ms, on 4.6 ms) | the rule needed 1000 ms | the probe becomes slow on some driver |
| llama.cpp/0050-layerwise-prefill-b12064.patch (second attempt, keeper fix) | `LLAMA_LAYERWISE_PREFILL=N` + `MOE_CACHE_GPU_KEEP_MIB` (PREFILL-638) | IQ3_XXS 34k prompt: N=8 prompt -51.9 % (241 vs 501), uploads 684 vs 652 GB (did not fall); N=34 uploads 0 GB but prompt -77.9 %, decode -10.4 % | the 8-layer mode does not cut uploads, the 34-layer mode costs far more in graph machinery than it saves | the machinery (one graph per layer and chunk, hidden state through host memory) becomes much cheaper |
| llama.cpp/0042-n-outputs-max.patch | `LLAMA_N_OUTPUTS_MAX` (PATCH-128) | compute buffers identical with and without it: 2,021 MiB at `-ub 2048`, 4,660 MiB at `-ub 4096` | inert by design at b12064: llama-server sets `n_outputs_max` = 4 itself (n_parallel auto 4) and 0042 acts only when it is 0. The real top tensor is the lightning-indexer score tensor (512 MiB at `-ub 2048`), which 0042 does not touch | the indexer score tensor is chunked and the logits reserve grows again |

### Merged upstream (2026-11-03, pin move b11966 -> b12064)

| File | Switch | Result | Why archived | Retest if |
|---|---|---|---|---|
| llama.cpp/0022-vulkan-count-experts-fast.patch | `GGML_VK_COUNT_EXPERTS_FAST` | +5 % prompts at 34k, +7 % at 64k (PC) | merged upstream as PR 31266, in the pin from b12064 (always on there; no switch) | - |
| llama.cpp/0027-batched-tensor-copy.patch | `GGML_BACKEND_BATCHED_COPY` | slot save / restore faster on Vulkan (see PRs) | merged upstream as PR 30881 + 31418, in the pin from b12064 (always on there; our patch only added the switch) | - |

## Notes on 0001
`ggml_vk_wait_for_fence` waits for the last 20 % of a graph by polling `getFenceStatus` with 1000 `pause` instructions between polls. The patch made the count an env value (N blocks of 10 pauses, unset = 100 = stock). Only the poll interval changes, so output cannot change. Tested at N = 10.
