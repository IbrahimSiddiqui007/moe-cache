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
| llama.cpp/0002-server-adaptive-draft-length.patch | `LLAMA_SPEC_ADAPT` | MTP decode 16.35 both | no gain | another draft model |
| llama.cpp/0003-vulkan-mmid-vec-max-tokens.patch | `GGML_VK_MMID_VEC_MAX_TOKENS` | 16.45 vs 16.2-16.35 | no gain | - |

## Notes on 0001
`ggml_vk_wait_for_fence` waits for the last 20 % of a graph by polling `getFenceStatus` with 1000 `pause` instructions between polls. The patch made the count an env value (N blocks of 10 pauses, unset = 100 = stock). Only the poll interval changes, so output cannot change. Tested at N = 10.
