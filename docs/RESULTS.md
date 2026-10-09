# Results (measured 2026-10-08, one laptop)

Machine: i7-13620H (6P+4E), 30 GB RAM, RTX 4060 Laptop 8 GB, WD SN560 NVMe, Linux. llama.cpp commit 7fe450e19.
Method: "clean" = CPU <= 58 C at start, no background load, fresh page cache, process memory limit via cgroup (`MemoryMax`, swap off),
4 prompts, temperature 0, seed 42, forward and reverse order; "steady" = mean of requests 2-4. Unclean runs are not reported.
"Stock" = the same llama.cpp without the plugin. Raw data and scripts: `work/t7-plugin/` of the research repo.

## 1. Token identity (4 prompts x 250 tokens vs stock)

| Model | Variant | Result |
|---|---|---|
| Qwen3-Coder 30B-A3B Q4_K_M | 6 plugin variants incl. 68,903 evictions | all identical |
| Qwen3.6 35B-A3B Q4_K_M | cold, learning, warm start, with MTP | all identical |
| Qwen3-Coder requantized to IQ4_XS, MXFP4_MOE, Q5_0 | 10 GiB cache, heavy eviction | 3/3 identical |
| Granite 3.1 MoE 3B | large and small cache | 2/2 identical |
| DeepSeek-V2-Lite 16B | large and small cache | 2/2 identical |
| gpt-oss-20b (native MXFP4) | large and small cache; auto cache | 3/3 identical |
| Qwen3-Next-80B-A3B Q4_K_M (48.5 GB) | 16 GiB cache | identical |
| Hunyuan-A13B Q4_K_M (48.8 GB) | short test (2 prompts x 32 tokens) | identical |

## 2. Qwen3.6 35B-A3B, steady tok/s (ncmoe 34, ctx 32768; fixed cache = limit - 3 GiB; auto sizing came later)

| Test | Limit | Stock | Plugin |
|---|---|---|---|
| Cold cache | 28 GB | 30.7 | 28.4 |
| Cold cache | 12 GB | 10.2 | 21.7 |
| Warm start | 28 GB | 29.0 | 32.3 |
| Warm start | 12 GB | 8.1 | 21.7 |
| Warm start + MTP | 28 GB | 39.5 | 38.2 |
| Warm start + MTP | 12 GB | 8.8 | 24.0 |
| Prompt processing, 2.6k tokens (prompt tok/s) | 12 GB | 8.4 / 5.8 (older log) | 55.1 |

## 2b. Qwen3.6 with the automatic cache size (final design, ncmoe 34, ctx 32768; clean)

| Test | Limit | Stock | Plugin |
|---|---|---|---|
| Cold cache | 12 GB | 10.2 | 23.6 (23.6, 23.5, 23.6, 23.8 over four runs) |
| Warm start | 12 GB | 8.1 | 23.5 |
| Warm start | 28 GB | 29.0 | 31.1 |

## 2c. Where the experts live matters (Qwen3.6, 12 GB, plugin, ctx 8192, cold)

| Placement | VRAM used | tok/s |
|---|---|---|
| All experts on the CPU (`--n-cpu-moe 99`) | ~2.6 GB | 16.1, 16.2 |
| Planner placement (8 layers of experts on the GPU, `--n-cpu-moe 32`) | ~6.2 GB | 25.5, 26.2 |

The plugin manages the experts that stay on the CPU; the GPU placement is chosen once at start by the planner. Using the GPU well is worth about +60 % on top.
Stock llama-bench (everything in RAM): 0 / 4 / 8 / 10 / 12 layers of experts on the GPU = 20.4 / 22.7 / 27.1 / 29.5 / 31.1 tok/s.

## 2d. KAT Coder 35B-A3B, 28 GB (steady tok/s)

| Config | Stock | Plugin, cold |
|---|---|---|
| ctx 32768, ncmoe 34 | 30.7, 30.7 | 27.0 |
| ctx 8192, ncmoe 33 | 32.3 | 27.9 |

An earlier note that the cache beats stock by 45 % on KAT (stock 19.7) was an outlier and is withdrawn.

## 2d2. KAT Coder with its warm-start profile (28 GB, clean, ctx 8192 / ncmoe 33)

Profile learned from 14 varied prompts (not the benchmark prompts). Plugin cold 27.9 steady (first request 17.9) -> plugin warm **31.8** (first request 31.9); stock 32.3. A second run in the other configuration (ctx 32768) gave 30.4 (flagged unclean, stock 30.7).

## 2e. Huge pages (Qwen3.6, 28 GB, plugin cold)

Default (system THP always): 27.3, 24.5. Huge pages off for the expert memory: 27.5, 26.6. No benefit from huge pages.

## 3. Other models, steady tok/s, all experts on the CPU (ctx 8192, cold cache)

| Model | Limit | Stock | Plugin | Notes |
|---|---|---|---|---|
| Qwen3-Next-80B (48.5 GB) | 28 GB | 4.0 | 12.9 | fixed cache |
| Qwen3-Next-80B | 20 GB | 1.3 | 9.3 | auto cache (fixed margins hit the memory limit twice) |
| Hunyuan-A13B (short test) | 28 GB | 1.0 | 1.3 | 13B active params, bandwidth-bound |
| gpt-oss-20b | 28 GB | 15.2 | 14.7 | fits in RAM |
| gpt-oss-20b | 10 GB | 14.0 | 13.1 | auto cache (fixed 3 GiB margin: 10.0) |
| DeepSeek-V2-Lite | 28 GB | 19.4 | 18.9 | fits in RAM |
| DeepSeek-V2-Lite | 9 GB | 19.3 | 18.3 | auto cache (fixed 3 GiB margin: 10.6) |
| Granite 3B | 28 GB / 5 GB | 48.6 / 48.5 | 47.9 / 48.2 | fits in RAM |

Caveats: the zoo tests force all experts onto the CPU (`--n-cpu-moe 99`) so the GPU holds only attention/shared weights (~2.6 GB of 8 GB VRAM);
real use keeps as many layers' experts on the GPU as fit, which speeds up both stock and plugin and shrinks the relative gain.
Section 2 numbers use the earlier fixed cache margin; they have not been re-measured with `auto`.

## 4. CPU-only machine (GPU hidden from llama.cpp), DeepSeek-V2-Lite, model fits in RAM

Measured 2026-10-09 with the repack support, clean runs (GPU hidden, 28 GB limit, ctx 8192, 4 prompts x 300 tokens, forward and reverse order, true stock binary):

| | Stock (repacked weights) | Plugin (repack on, 7.5 GiB cache) |
|---|---|---|
| Steady tok/s (requests 2-4) | 17.4 and 17.2 (mean 17.3) | 16.7 and 16.7 (mean 16.7) |
| First request | 18.2 | 16.5 (cache is cold: prompt phase 10 tok/s vs 48) |

The plugin is 3.5 % slower when the model fits in RAM on a CPU-only machine; identity is 4/4. The earlier figures (stock 8.4, plugin 6.4, -24 %, identity 3 of 4) were measured before repack support
and under different conditions; they are replaced by this table.

## 5. No dedicated GPU, memory limited (Qwen3.6 35B, 21.7 GB model, GPU hidden from llama.cpp, ctx 8192)

Stock needs `--no-repack` here (default repacking makes a full copy in anonymous memory).

| Limit | Stock (--no-repack) | Plugin |
|---|---|---|
| 12 GB | 3.1 | 9.9 |
| 16 GB | 6.4 | **10.3** (clean rerun, requests 2-4 = 10.6, 10.1, 10.2; stock 6.6, 6.4, 6.1) |

## 6. Two-times-RAM model on a small machine (Qwen3-Next-80B, 48.5 GB, 24 GB limit, about 2.6 GB of GPU use)

Stock 2.3 tok/s, plugin 10.6 tok/s (4.6x).

## 7. Prompt processing (long prompts, 2.6k to 3.2k tokens)

| Setup | Stock | moe-cache, CPU does the expert work | moe-cache, GPU-assisted (default) |
|---|---|---|---|
| DeepSeek-V2-Lite, fits in RAM, GPU present | 188, 166 tok/s | 42, 43 | **166, 183** |
| Qwen3.6-35B, 12 GB limit | 6 to 8 (thrashing, older measurement) | 48, 48 | **62, 60** |

Long-prompt identity vs stock: 2 of 2 prompts identical on DeepSeek with the GPU path active (`tests/identity.sh --long`).

## 8. Forced repacking with a GPU present (DeepSeek-V2-Lite, all experts on CPU)

Token generation: repack off 12.7 to 14.6 tok/s, repack on 11.8 to 13.3 tok/s: no decode gain. Repack stays automatic (on only when stock would repack).

## 9. Memory-limit test (`tests/limit.sh`, hard cgroup limit, no swap, short + long prompts)

| Model | Limit | Result |
|---|---|---|
| gpt-oss-20b | 10 GB | PASS (long prompt 110 tok/s, generation up to 16.7) |
| DeepSeek-V2-Lite | 9 GB | PASS (long prompt 280 tok/s, generation up to 18.0) |
| Qwen3-Next-80B (48.5 GB) | 20 GB | PASS (generation 6 to 11 tok/s) |
| Qwen3.6-35B | 12 GB | PASS (long prompt 67 tok/s, generation up to 20) |

A first version of the memory controller let a long first prompt grow the cache past the limit (the budget raced ahead of what was loaded and the correction was too weak); fixed by tying the budget to the actually loaded bytes.

## 10. Robustness tests (Granite 3.1 MoE and DeepSeek-V2-Lite, token identity vs stock)

| Test | Result |
|---|---|
| `-np 2`, requests sent one after the other | 4/4 identical |
| `-np 2`, requests sent two at a time | server healthy and answers both. Output depends on when the requests arrive, **for stock llama.cpp too** (stock differs from its own delay-0 output in 17 of 20 timing variants; the plugin in 20 of 20): batches of 32+ tokens switch from CPU to GPU math. Identity cannot be promised for concurrent use. |
| Long context: about 12,000-token prompts, ctx 16384 (Granite) | 4/4 identical |
| Long context, DeepSeek-V2-Lite, 9 GB limit | 4/4 identical |
| Failing direct reads (every 40th expert read fails all its direct attempts; test switch `MOE_CACHE_FAULT_EVERY`) | 320 reads fell back to a buffered read, output 4/4 identical. If the buffered read fails too, the server stops with a message (it never serves wrong data). |
| Memory-bandwidth probe (CPU and GPU reading host memory at the same time) | CPU alone 40-41 GB/s, GPU alone 12.5 GB/s, together 39-41 GB/s in total: the two share one memory bus and do not add up |

