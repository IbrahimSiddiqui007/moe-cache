# Results (measured 2026-10-08, one laptop)

Note (2026-10-14): the `bench/` scripts and `moe-cache-bench` named below moved to [moe-cache-benchmark](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark); `bench/gguf_plan.py` is now `planner/gguf_plan.py`.

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

## 11. CPU thread pinning on a hybrid CPU (i7-13620H: 6 performance cores with 2 threads each, 4 efficiency cores)

Qwen3.6 35B, 6 layers of experts on the GPU, ctx 32768, 28 GB limit (the model fits), steady tok/s, forward and reverse order, clean runs (start 57-59 C, no background load).

**Stock llama.cpp** (no plugin):

| Setting | Runs | Mean | vs default |
|---|---|---|---|
| default (threads float over all cores) | 31.1, 30.8 | 30.9 | |
| `-C 0x555 --cpu-strict 1` (one thread per performance core) | 36.2, 35.1 | 35.7 | **+15 %** |
| `-t 12 -C 0xFFF --cpu-strict 1` (all threads of the performance cores) | 37.5, 35.4 | 36.5 | **+18 %** |
| `--poll 100` | 31.0, 31.9 | 31.5 | +2 % (noise) |

**With the plugin** (cold cache, auto size; the plugin has its own thread pool, so llama.cpp's flags do not reach it):

| Setting | Runs | vs unpinned plugin |
|---|---|---|
| plugin not pinned (`MOE_CACHE_CPU_PIN=off`) | 26.4, 25.8 | |
| plugin pins its threads automatically (default on hybrid Intel CPUs) | 31.3, 30.8 | **+19 %** |
| plugin pinned and `-C 0x555 --cpu-strict 1` for llama.cpp | 31.9, 31.9 | **+22 %** |
| same with polling off in both pools (`MOE_CACHE_POLL=0 --poll 0`) | 31.4, 32.1 | no effect |

`moe-cache-server` now adds `-C MASK --cpu-strict 1` on hybrid Intel CPUs (`--no-pin` to disable). Pinning does not change the output (identity 4/4).

**What closed the gap to stock (2026-10-09):** the plugin was slower than pinned stock only with a *cold* cache (experts loaded one by one on demand). With a warm start the pinned plugin reached 37.5, 37.2, 34.3 tok/s (mean 36.3). So when every CPU expert fits in the cache and no profile exists yet, the plugin now loads all of them in order at start (`MOE_CACHE_PRELOAD_ALL=0` turns this off; startup takes about 6 s for 15 GiB). Result, same setup, plugin pinned, no profile:

| | Runs (steady tok/s) | Mean | First request |
|---|---|---|---|
| plugin before (cold cache) | 31.7, 30.2, 30.8, 32.2, 31.9, 31.3, 30.8 | about 31.3 | about 18 |
| plugin after (loads all experts at start) | 34.9, 36.7, 34.8 | **35.5** | 36-38 |
| stock, pinned (for comparison) | 36.2, 35.1, 37.5, 35.4 | 36.0 | 38 |
| stock, defaults | 31.1, 30.8 | 30.9 | |

So with pinning (automatic on hybrid Intel CPUs) the plugin is at parity with pinned stock and about 15 % above stock with default settings, when the model fits in RAM. Identity 4/4. The memory-limit tests (gpt-oss 10 GB, DeepSeek 9 GB, Qwen3-Next 20 GB, Qwen3.6 12 GB) still pass, and none of those cases preloads (they do not fit).
Also fixed: the memory tuner re-read `/proc/self/status` after every layer even when nothing was read; no measurable speed change.

**With MTP (speculative decoding, `--spec-type draft-mtp --spec-draft-n-max 2`) and pinning, same model and limit** (steady tok/s, order stock, plugin, plugin, stock; both pinned with `-C 0x555 --cpu-strict 1`): stock 43.2 and 40.0 (mean 41.6), plugin 41.8 and 41.4 (mean 41.6). Draft acceptance is the same in all four runs (734 of 919), so the output matches. Speed falls within a run (48 to 37 tok/s) as the CPU heats up.

## 12. A model 2x bigger than RAM: gpt-oss-120b (MXFP4, 63.4 GB, 128 experts, top 4)

All experts on the CPU, ctx 8192, 4 prompts x 100 tokens, both pinned to the performance cores, hard cgroup limit with swap off, true stock binary, clean starts (50-57 C, no background load).
**One run per cell** (the user chose a short plan), forward order only, so treat the figures as indicative.

| Memory limit | Stock | moe-cache | Ratio |
|---|---|---|---|
| 24 GB | 0.34 tok/s (requests: 0.98, 0.46, 0.27, 0.30) | **2.71 tok/s** (2.36, 2.83, 2.60, 2.69) | **8.0x** |
| 28 GB | 0.96 tok/s (1.50, 1.40, 0.68, 0.79) | **3.10 tok/s** (2.72, 3.37, 2.97, 2.98) | **3.2x** |

- Token identity at the 24 GB limit (4 prompts x 24 tokens): **4/4 identical**.
- Plugin statistics: 24 GB limit hit rate 81.3 %, 156 GB read from the SSD, 29,960 evictions; 28 GB limit hit rate 84.6 %, 128 GB read, 22,561 evictions.
- Stock slows down during a run (0.98 to 0.27 tok/s) as the page cache thrashes and the CPU heats up; the plugin stays flat. The plugin is limited by SSD reads (about 4 GB read per generated token at 24 GB).
- An earlier attempt to run the stock side without a memory limit got the whole desktop session killed by `systemd-oomd`; every big-model run now uses a hard limit (`tests/identity.sh --limit`).

### 12b. gpt-oss-120b again, experts split between GPU and RAM, with the deep metrics (moe-cache-bench, 2026-10-09)

24 GB limit, `--n-cpu-moe 34` (2 of 36 layers of experts on the 8 GB GPU, 6.0 GB VRAM in use), ctx 8192, 4 prompts x 100 tokens, both pinned to the performance cores, one run each.
**The tool flagged both runs as not clean**: starting temperature was fine (55 and 56 C) but background CPU was 38 % of one core (the desktop, a browser pane and a compositor were running). Treat these as indicative; they agree with the earlier clean runs of section 12.

| | Stock | moe-cache |
|---|---|---|
| Generation speed, steady (requests 2-4) | **0.37 tok/s** (1.17, 0.42, 0.34, 0.34) | **3.01 tok/s** (2.84, 3.12, 2.82, 3.08) = **8.1x** |
| SSD read per generated token | 325 MB | 330 MB |
| Major page faults per token | 78,891 | 772 |
| CPU time per token | 1,100 ms | 540 ms |
| Peak memory (cgroup) | 25.8 GB | 25.2 GB |
| Cache hit rate, decode steps | n/a | 86.3 % |
| Evictions per token / idle time of an evicted expert | n/a | 73.6 / 3,749 groups |
| Time per token: SSD reads / expert compute / bookkeeping / everything else | n/a | 178 / 70 / 11 / 128 ms (includes the prompt phase) |

What this shows: both read **the same amount** from the SSD per token (about 325 MB), but stock does it with about 79,000 small page-fault reads per token and moe-cache with about 770 large reads. The 8x is read efficiency, not less data.
Compared with all experts on the CPU (section 12: 0.34 and 2.71 tok/s), moving 2 of 36 layers to the GPU gave +9 % (stock) and +11 % (moe-cache), about what 2/36 of the traffic predicts.

### 12c. What speeds up gpt-oss-120b (24 GB limit, 2 of 36 expert layers on the GPU, moe-cache only; moe-cache-bench, 2026-10-09)

Same setup as 12b, 4 prompts x 100 tokens, one run per row, tool-flagged clean with the background-CPU allowance set to 40 % of one core (the desktop was running). Steady = requests 2-4.

| Setting | tok/s | SSD reads per token | Note |
|---|---|---|---|
| default before (bounce buffer, 6 reader threads) | 3.02 | 175 ms | |
| 12 reader threads | 3.02 | 173 ms | no effect |
| 24 reader threads | 2.94 | 175 ms | no effect |
| **read straight into place** | **3.70** | 144 ms | +22 %, identical tokens |
| read straight into place, 12 threads | 3.62 | 144 ms | |
| new default (automatic for models that do not fit) | **3.72** | 143 ms | +23 % |

- The SSD streams 3.9 GB/s with parallel direct reads (2.8 GB/s with one reader); moe-cache gets about 2.3 GB/s effective, so the read path still has headroom.
- Token identity vs the stock run: 4/4 for every row.
- Read-into-place is switched on automatically when the model file is larger than 85 % of the memory limit (`MOE_CACHE_DIRECT=0/1` forces it). It restores what the old llama.cpp patch did. On Qwen3.6 at a 12 GB limit (reads are 20x smaller) old and new path are the same within noise: 27.8 and 25.2 tok/s (old, mean 26.5) against 27.5 and 26.4 (new, mean 27.0).

### 12d. Speed ideas tried on the SSD-bound models (2026-10-10; moe-cache-bench, GPU holds some expert layers, one run per cell unless stated)

**Run-to-run variation is about +-10 %** on these models (the same setting gave 3.6 to 4.3 tok/s in different hours), so changes below 10 % cannot be seen without repeated paired runs.

| Idea | Result | Verdict |
|---|---|---|
| **Sibling batch load** (load the gate, up and down experts of a layer together; gpt-oss hands them over in three separate calls) | gpt-oss-120b, 24 GB: **4.47 tok/s** (4.31, 4.63) against 3.50 (3.54, 3.46) with it off, paired in both orders: **+28 %**; SSD read time per token 144 -> 110 ms (about 2.9 GB/s effective, up from 2.2); tokens identical to stock 4/4 | **kept, on by default** (`MOE_CACHE_SIBLINGS=0` turns it off). No effect on models that already group the three operations (Qwen3.6, Qwen3-Next) |
| Read experts straight into place | +22 % on the 120B (see 12c); none on models that fit | on when the model does not fit |
| More reader threads | no effect | the batches were too small to need them |
| Huge pages for the cache arena (`MOE_CACHE_THP`) | 3.97 and 4.26 with, 4.32 without | no gain |
| **Lookahead prefetch** (predict next layer's experts from the router weights, read them during compute) | Qwen3.6-35B 12 GB: 20.7 vs 27.5 tok/s (-25 %); Qwen3-Next-80B 24 GB: 11.8 vs 13.4 (-12 %); gpt-oss-120b: 0.94 vs 3.63 (-74 %), later 2.0 with a cap of 2 per layer. Prediction recall for gpt-oss-120b is only 10.4 % (random: 3 %) | **does not work**; code stays, off by default (`MOE_CACHE_LOOKAHEAD=1`) |
| Smarter eviction (2Q, LFU, pinning the hottest experts from a profile) | replay of 13,464 real decode layer-steps of gpt-oss-120b at the real cache size: LRU 84.2 %, 2Q 83.6 %, LFU 84.1-84.2 %, LRU + pinned hottest 84.3-84.8 %; offline optimum 92.5 % | **no policy beats LRU by more than 0.6 points**; dropped |
| Read-path microbenchmark (6 threads, direct reads) | 3 scattered 4.25 MB reads into freshly released pages 2.72 GB/s; same into reused pages 3.84; one packed 12.7 MB read 3.61 | page re-faulting costs about 30 %; a packed file cannot be read into three separate memory places with direct I/O, so it was not built |
| **N-gram speculative decoding** on a code-editing workload (the answer mostly copies the prompt), Qwen3.6-35B 24 GB, 2 runs each | off 39.4 and 36.9; `ngram-mod` 35.7 and 36.3 (about -5 %); `ngram-simple` 28.5 and 29.4 (-25 %, and one of three answers differed from the non-speculative tokens) | no gain on MoE models: verifying several drafted tokens wakes extra experts |
| **L3 cache helper thread** (an efficiency-core thread loads the up/down experts into L3 while gate computes), Qwen3.6-35B 28 GB, 4 clean runs each | off 38.4 tok/s (39.8, 37.6, 36.8, 39.3); on 31.2 (34.2, 30.7, 29.8, 30.0) | **-19 %**; off by default (`MOE_CACHE_L3PREFETCH=1`) |

Where the 120B stands now (24 GB limit, 2 of 36 expert layers on the GPU, 4 prompts x 100 tokens): stock llama.cpp 0.37 tok/s, moe-cache 4.47 tok/s (12x).

## 13. Small models, CPU only, hard memory limit (Raspberry-Pi-class test, 2026-10-10)

GPU hidden (`CUDA_VISIBLE_DEVICES=`), 4 threads pinned to 4 performance cores, every run in its own cgroup with the stated limit and swap off, start at 51 to 55 C
(the gate is 58 C), `llama-bench` pp64 / tg64 with 3 repetitions (mean ± standard deviation), one run per cell, stock = same llama.cpp without the plugin.
**These are not Raspberry Pi numbers**: this CPU is several times faster than a Pi's. They show how the plugin behaves relative to stock with 1 to 3 GB of memory.

| Model (total / active) | File | Limit | Stock tg64 tok/s | moe-cache tg64 tok/s | Note |
|---|---|---|---|---|---|
| Granite 3.1 MoE 1B-A400M Q4_K_M | 0.82 GB | 3 GB | 107.5 ± 3.4 | 98.7 ± 3.5 | fits: plugin about 8 % slower |
| Granite 3.1 MoE 1B-A400M Q8_0 | 1.42 GB | 3 GB | 68.2 ± 0.9 | 66.9 ± 1.5 | fits: about 2 % slower |
| Granite 3.1 MoE 3B-A800M Q4_K_M | 2.0 GB | 3 GB | 55.2 ± 0.4 | 51.6 ± 1.4 | fits: about 7 % slower |
| SmallThinker 4B-A0.6B Q4_K_M | 2.65 GB | 3 GB | 54.0 ± 0.2 | 46.3 ± 1.5 | fits: about 14 % slower |
| Ministral 3 3B Q4_K_M (dense, no experts) | 2.15 GB | 3 GB | 16.3 ± 0.3 | 15.6 ± 0.3 | the plugin has nothing to manage; about 4 % slower |
| Granite 3.1 MoE 3B-A800M Q4_K_M | 2.0 GB | 1.2 GB | killed (out of memory) | 5.3 ± 0.0 | stock repacks the weights into anonymous memory |
| Granite 3.1 MoE 1B-A400M Q8_0 | 1.42 GB | 0.9 GB | 2.2 ± 0.0 | 4.5 ± 2.0 | |
| OLMoE-1B-7B-0125 Q4_0 (6.9B / 1.3B) | 3.93 GB | 3 GB | killed (out of memory) | 19.4 ± 0.9 | file larger than the limit |
| OLMoE-1B-7B-0125 Q4_0 | 3.93 GB | 2 GB | killed (out of memory) | 9.2 ± 0.1 | |

Stock has to be run with `--no-repack` when the model does not fit. Same cells with `llama-server`, 4 requests x 48 tokens, mean of requests 2 to 4, stock `--no-repack` against the plugin with `MOE_CACHE_REPACK=off` (like for like):

| Model | Limit | Stock `--no-repack` | moe-cache (repack off) | moe-cache (repack auto) |
|---|---|---|---|---|
| Granite 3B-A800M Q4_K_M | 1.2 GB | 1.4 | 5.4 | 5.3 |
| Granite 1B-A400M Q8_0 | 0.9 GB | 2.0 | 6.5 | 6.5 |
| OLMoE-1B-7B Q4_0 | 3 GB | 7.6 | 16.6 | 13.7 |
| OLMoE-1B-7B Q4_0 | 2 GB | 1.8 | 6.1 | 6.1 |

Findings: the plugin is not faster when the whole model fits (about 2 to 14 % slower on these small CPU-only runs, SmallThinker the most); it keeps models usable that do not fit, 2 to 4 times faster than stock with repacking switched off, and stock with default settings is killed. On CPU-only machines the automatic repacking was slower than repack off in the 3 GB OLMoE cell (13.7 against 16.6): not understood yet. Ministral Q8_0 (3.65 GB, dense) in the 3 GB limit ran at 0.8 tok/s for stock.

## 14. Qwen3.8-Flash-Next IQ2_XS, 24 GB limit, 3 expert layers on the GPU: kernel and executor (2026-10-10)

10 requests of 48 tokens, mean of requests 6 to 10, one run per row, not gated (desktop programs were using CPU: read the ratios, not the absolute values). Rows of one table were run in the same session.
Token identity against stock: 4/4 for every configuration listed.

| Configuration | tok/s | Note |
|---|---|---|
| plugin 0.1.4 (llama.cpp's scalar Q2_0 kernel) | 6.9 | before |
| fast Q2_0 kernel | 10.3 | paired run, see docs/EXPERIMENTS.md |
| fast kernel, executor off | 8.2 | second session |
| fast kernel + executor, 6 threads | 10.9 | +33 % over the row above |
| fast kernel + executor, 6 threads + 4 efficiency cores | 12.4 | +52 % over the row above |

Second round, same configuration with the permuted-activation Q2_0 kernel (the executor rearranges each quantised activation column once; 5.9x the library's dot product, bit-identical):

| Configuration (same session) | tok/s |
|---|---|
| executor off | 11.2 |
| executor on, 6 threads | 12.3 |
| executor on + 4 efficiency cores | 13.5 |

Identity 4/4 and 0 byte mismatches over 12,420 expert ops with `MOE_CACHE_EXEC_CHECK=1`. The same configuration measured 8.2 tok/s in a busier session: absolute values on this laptop move by about 30 % with
desktop load and temperature, so compare only rows measured together.


## 15. New llama.cpp pin (23b0202a1, ggml 0.26.0) and upstream's own MoE cache (2026-10-11)

What changed: llama.cpp moved the buffer-type interface of ggml (backend API 2 -> 3). The plugin has one new build mode for it (`GGML_API=3 ./build.sh`) and refuses to load into a mismatching ggml. Upstream also merged `--moe-cache-mib` (PR 29887): an LRU cache of experts in GPU memory for layers whose experts stay in host RAM.

Token identity vs stock, plugin built for API 3, 4 prompts x 48 tokens: Granite 3B 4/4, Granite 1B 4/4, OLMoE 4/4, Qwen3.6-35B (24 GB limit) 4/4, Flash-Next IQ2_XS (24 GB limit) 4/4.

Decode speed, steady-state mean of requests 4-10 (48 tokens each), every run started at <= 58 C with the machine idle, ABAB order, two rounds:

| Qwen3.6-35B-A3B, 24 GB limit, ctx 4096 | round 1 | round 2 |
|---|---|---|
| stock, 8 expert layers on the GPU (`--n-cpu-moe 32`) | 32.8 | 33.8 |
| stock + `--moe-cache-mib 5000`, all experts on the CPU side | 42.9 | 43.0 |
| plugin, 8 expert layers on the GPU | 38.8 | 39.4 |
| plugin + `--moe-cache-mib 5000` | 29.6 | 28.4 (cache is disabled: the plugin's buffer is not a host buffer, so no layer is cached and no layer is on the GPU) |

Token identity of the upstream cache itself: 3 of 4 prompts identical to the same model without it (one prompt diverges at token 8: experts computed on the GPU instead of the CPU give slightly different floats, so a greedy choice flips). It is a valid but not bit-identical result.

| Flash-Next IQ2_XS, 24 GB limit, ctx 2048, same VRAM for experts in every row | round 1 | round 2 |
|---|---|---|
| stock, 3 expert layers on the GPU | 5.4 | 5.7 |
| stock + `--moe-cache-mib 2200` | 7.8 | 11.5 (last two requests 8.3 and 13.4) |
| plugin, 3 expert layers on the GPU | 12.0 | 12.2 |

The stock-with-cache row moved a lot between rounds; a third round is needed before quoting it. The plugin's rows are stable.

Kernel candidates (IQ2_XXS and IQ3_XXS dot products, `tests/kernel_variants.cpp`, one core, data in cache, library function and variants interleaved over 7 rounds, two runs, start 54-55 C): all variants are bit-identical to the library function. IQ2_XXS: no variant is faster (0.90-0.95x). IQ3_XXS: gather-based lookups 1.13-1.18x, the other variants 0.97-1.01x. Both kernels are limited by assembling table lookups, not by arithmetic.


## 16. Warm start, read speed and page faults on Qwen3.8-Flash-Next IQ2_XS (2026-10-11)

Setup: plugin built for API 3 on llama.cpp 23b0202a1, 24 GB hard limit, swap off, 3 expert layers on the GPU, 4 requests of 48 tokens, ABAB twice, every run started at <= 58 C with the machine idle. The profile was learned from `tools/learn_prompts.txt` (14 prompts, 60 tokens each); the test prompts are different.

| Request | no profile (2 runs) | `MOE_CACHE_WARM` profile (2 runs) |
|---|---|---|
| 1st: tok/s | 6.1 / 7.4 | 15.0 / 14.5 |
| 1st: prompt time | 1.8 s / 1.5 s | 0.7 s / 0.7 s |
| 2nd: tok/s | 8.2 / 10.2 | 11.9 / 13.7 |
| 3rd: tok/s | 11.0 / 10.8 | 12.9 / 12.7 |
| 4th: tok/s | 11.8 / 12.6 | 13.4 / 12.6 |

The profile removes the cold ramp of the first requests; the steady speed is the same.

Major page faults of the server during one request (plugin, warm): 23, 8, 5 and 3 for prompts of 8-10 tokens. The n-gram table is not what makes the prompt phase take about 1 s (that is expert work), so prefetching it during prefill has nothing to gain.

Read speed (`tests/read_bench.cpp`, O_DIRECT random reads from the 39 GB shard, 4 reader threads, numbers confirmed against the SSD's own read counters in `/proc/diskstats`; the benchmark's own "fresh page" figures over-count and are not quoted):

| Case | Result |
|---|---|
| Device ceiling (WD SN560) | about 3.6-3.8 GB/s |
| 2.1 MB slices, destination pages reused | 3.65 GB/s at the device |
| 2.1 MB slices, destination pages freshly unmapped (what eviction does) | 2.9 GB/s at the device (about 20 % lower) |
| same, pre-faulted with `MADV_POPULATE_WRITE` in the reading thread | 3.58 GB/s |
| 0.48 MB slices, 4 threads: pread vs io_uring batches of 4 (pages reused) | 2.6 vs 3.4 GB/s; 1 thread: 1.0 vs 2.1 GB/s |
| 2.1 MB and 7 MB slices | pread and io_uring both reach the device ceiling |

Reading: for large slices (GLM-sized experts) the only loss is the fresh-page cost, which pre-faulting recovers; io_uring helps only for slices of about 0.5 MB or less, and Flash-Next is limited by CPU compute, not by reads, so neither changes its tok/s.


## 17. Settings sweep, prefill and pre-fault (2026-10-11, Qwen3.6-35B-A3B and Flash-Next, same machine and method as section 15)

Stock llama.cpp 23b0202a1 with `--n-cpu-moe 99 --moe-cache-mib N` on Qwen3.6-35B (24 GB limit), single clean runs, decode tok/s (repeat runs of the same setting spread by about +-3 %):

| Setting | tok/s |
|---|---|
| cache 3500 MiB | 37.6 |
| cache 5000 MiB | 42.1 (44.1 in a repeat) |
| cache 5500 MiB | 44.1 |
| cache 6000 / 6500 MiB | out of GPU memory (8 GB card) |
| 5000 MiB with `-t 6` / `-t 8` / `-t 12` | 43.7 / 43.4 / 44.2 (no effect) |
| 5000 MiB with `-ctk q8_0 -ctv q8_0` | 43.5 (no effect) |

Prompt processing, 2168-token prompt, second request, tok/s:

| Setting | `-ub 512` (default) | `-ub 1024` | `-ub 2048` |
|---|---|---|---|
| cache 4000 MiB, all experts on the CPU side | 461 | 646 | **812** |
| static 8 GPU layers (`--n-cpu-moe 32`) | 448 | n/a | 783 |

A larger micro-batch means expert weights cross PCIe fewer times per prompt: about +70-76 % on this machine, at the cost of a bigger compute buffer in VRAM.

Pre-fault of replacement pages (`MOE_CACHE_PREFAULT=1`, `MADV_POPULATE_WRITE` in the reader thread), Flash-Next, plugin, cold start, 6 requests, ABAB twice: mean of requests 4-6 was 12.7 / 12.9 tok/s without and 11.5 / 12.6 with it, first-request prompt time 1.5 s vs 1.7 s. No gain: Flash-Next is limited by CPU compute, not by reads. The option stays off by default; it is untested on an SSD-bound model.

### Estimates for GLM-5.3-Flash (not measured: no model file on this machine)

Inputs: 42 MoE layers, 8 of 288 experts, about 8.5 G routed weights per token; UD-IQ2_XXS file 101.8 GB, UD-Q2_K_XL 108.7 GB; per-core dot-product speed measured in section 15 and `tests/kernel_bench.cpp`; RAM cache of about 20 GB, hit rate assumed from the Flash-Next trace (about 80 % at a 20 % cache fraction); SSD 3.6 GB/s measured on the laptop.

| Machine | UD-IQ2_XXS | UD-Q2_K_XL | Main limit |
|---|---|---|---|
| i7-13620H, RTX 4060 8 GB, 31 GB, SN560 | 3-5 tok/s | 4-6 tok/s | IQ types: CPU compute; K-quants: SSD |
| Ryzen 7 7800X3D, RTX 4070, 32 GB DDR5-6000, PCIe 4 NVMe (assumed 5-6 GB/s) | 4-6.5 tok/s | 5.5-8 tok/s | CPU compute / SSD |

Uncertainty about +-40 %. The first measurement on a real GLM file replaces these numbers.


## 18. GPU expert cache with the plugin, quant comparison, MTP and neuron sparsity (2026-10-11, Qwen3.8-Flash-Next and Qwen3.6 / Qwen3-Coder)

Setup as in section 15: llama.cpp 23b0202a1, plugin built for API 3, 24 GB hard limit (swap off), ctx 2048, 8 requests of 48 tokens, mean of requests 4-8, every run started at <= 58 C with the machine idle.

### Plugin together with llama.cpp's GPU expert cache (`MOE_CACHE_GPU_CACHE=1`, `--moe-cache-mib 2200`)

Token identity: plugin + cache equals stock + cache (same cache size), 4 prompts x 48 tokens, on Qwen3-Coder-30B (12 GB limit) and Flash-Next IQ2_XS (24 GB limit).

| Flash-Next IQ2_XS, tok/s | round 1 | round 2 |
|---|---|---|
| plugin, 3 static GPU layers | 11.7 | 11.4 |
| plugin + GPU cache 2200 MiB (all experts managed by the plugin) | 12.0 | 13.5 |
| stock + GPU cache 2200 MiB | 4.9 | 4.0 |

The GPU cache adds 0-19 % on top of the plugin here (the two rounds disagree; a third is needed before quoting a number); stock with the cache is 2.5-3x slower than the plugin with or without it, with a 10-18 s first-request stall.

### Same model, two quantisations (plugin, static 3 GPU layers)

| File | round 1 | round 2 | size |
|---|---|---|---|
| IQ2_XS (ISTA, mostly Q2_0 down projections, which have the fast kernel) | 12.3 | 12.65 | 68.0 GB |
| UD-Q2_K_XL (unsloth) | 10.75 | 11.5 | 78.9 GB |

The K-quant file is about 12 % slower: it is 16 % bigger, and its kernel advantage does not show on a model whose IQ file already uses the fast Q2_0 kernel. This does not answer the question for GLM-5.3-Flash (plain IQ2_XXS / IQ3_XXS, no fast kernel).

### MTP speculative decoding (unsloth `MTP/mtp-...-Q4_K_M.gguf`, `--spec-type draft-mtp --spec-draft-n-max 3`)

| Config | Result |
|---|---|
| 3 static GPU layers + MTP | does not start: the head's own MoE block needs 2.3 GiB of VRAM |
| all experts on CPU + MTP, head on GPU | does not start: compute buffer for the MTP context does not fit |
| 3 static GPU layers + MTP, head experts on CPU | does not start (556 MiB more than fits) |
| all experts on CPU + MTP, head experts on CPU (`--spec-draft-cpu-moe`) | 4.3 / 4.3 tok/s vs 11.5 / 11.4 / 11.0 without MTP (-62 %) |

Draft acceptance rose from 45 % to 73 %, mean accepted length about 3.1 tokens. Token identity MTP vs no MTP: 2 of 4 prompts identical (batched verification uses different float order). Likely reasons for the slowdown (not isolated): verification batches of 3-4 tokens bypass the executor, and expert math, not weight reads, is the limit on this CPU.

### Neuron-level sparsity (`tests/sparsity_probe.cpp`, share of squared activation magnitude of an expert's intermediate vector in its largest p % of neurons)

| Model | activations | top 5 % | top 10 % | top 25 % | top 50 % |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B, decode | 60,200 | 0.568 | 0.709 | 0.890 | 0.980 |
| Qwen3-Coder-30B-A3B, decode | 72,232 | 0.638 | 0.758 | 0.910 | 0.984 |

Per layer the top-25 % share ranges from 0.86 to 0.98. Reading: skipping the weakest half of the neurons drops about 2 % of the energy (about 14 % relative error of that expert's output vector); skipping three quarters drops about 10 % (about 33 %). Gate and up must be computed before the activation is known, so without a trained predictor only the down projection (one third of the weights) can skip neurons. Not pursued.


## 19. Small-batch executor, GPU cache third and fourth round (2026-10-11)

### Plugin + llama.cpp GPU cache, Flash-Next IQ2_XS, 24 GB limit, 10 requests of 48 tokens (mean of requests 4-10), all runs at <= 58 C and idle

| Round | plugin, 3 static GPU layers | plugin + `--moe-cache-mib 2200` |
|---|---|---|
| 1 | 11.7 | 12.0 |
| 2 | 11.4 | 13.5 |
| 3 | 12.6 | 14.0 |
| 4 | 12.1 | 14.1 |
| mean | **11.95** | **13.4 (+12 %)** |

Prompt time of the short prompts halves (about 1.1 s to 0.6 s).

### Small-batch executor (up to 8 tokens per step; tokens that pick the same expert share its weight rows)

Exactness: tokens identical to stock and 0 byte mismatches in every op compared with llama.cpp's own result, with batches forced to 1-4 tokens (`-ub 4`):
OLMoE 9,894 ops, Granite 3B 19,830 ops, Flash-Next IQ2_XS 27,945 ops (Q2_0 fast kernel and IQ types). `MOE_CACHE_EXEC=2` forces the executor on for tests.

Flash-Next, all experts on the CPU side, 8 requests:

| | before (round 1 / 2 / 3) | new executor (round 1 / 2) |
|---|---|---|
| no MTP, tok/s | 11.55 / 11.36 / 11.0 | 12.4 / 11.9 |
| no MTP, prompt time of short prompts | 1.1-1.4 s | 0.5-0.8 s |
| MTP (head experts on CPU), tok/s | 4.3 / 4.3 | **9.5 / 10.7** (still 17 % below no MTP) |

### Quality harness (`bench/quality.py`)

`compare` (prompt bank with checkable answers, judge for same meaning, degeneration checks, optional long-context retrieval), `longctx` (needle retrieval at growing sizes, reports prefill and decode speed), `soak` (long session with periodic probes, tracks answers, tok/s and memory).

### Caveat for `-ub 2048` (2026-10-11)

With a 128k context (`-c 131072`, q8_0 KV cache) the compute buffer of a 2048-token micro-batch is 3.06 GiB on the GPU and does not fit an 8 GB card next to the model,
so the server fails to start. The +70 % prompt-processing gain of section 17 applies to short contexts; at long contexts use the default `-ub 512` (or 1024).


## 20. Long context, long sessions and prompt processing on Flash-Next (2026-10-11)

Plugin on Flash-Next IQ2_XS, 24 GB hard limit, all experts on the CPU side (the only layout where a 128k KV cache fits next to the model on 8 GB), q8_0 KV cache,
`-c 131072`, `bench/quality.py longctx` (a fact hidden in filler text at 10 / 50 / 90 % depth, asked at the end).

| Test | Result |
|---|---|
| 8k nominal (6.4k tokens), 3 depths | 3 of 3 found; prefill 77-89 s (72-84 tok/s); decode 9.1-9.7 tok/s |
| 32k nominal (25.5k tokens), 3 depths x 2 passes on one server for 75 min | **6 of 6 found**; prefill 459 s to 691 s (56 to 37 tok/s, settled at about 37); decode 7.4-7.9 tok/s |
| 32k with the default 8 GiB `--cache-ram` | server **OOM-killed** by the 24 GB limit when the second long request started (llama-server saves the finished prompt into host RAM in one burst) |
| Same with `--cache-ram 2048` and the plugin keeping that memory free | no kill, stable RSS about 21 GB |

Prompt processing, 5,948-token prompt, same setup at 16k context, tok/s:

| `-ub` | prefill | decode after the prompt (first request) |
|---|---|---|
| 512 | 87.0 / 90.4 | 8.8 / 8.7 |
| 1024 | 111.9 | 8.6 |
| 2048 | 140.5 | 8.8 |

A prompt batch touches nearly every expert of every layer, so each micro-batch re-reads whatever the RAM cache does not hold; prefill time scales with the number of micro-batches.
At 128k context a 2048-token micro-batch needs a 3.06 GiB compute buffer on the GPU (does not fit in 8 GB); 1024 should need about half (not measured).

Scan-resistant loading of prompt batches (`MOE_CACHE_SCAN_TOKENS=32`, experts of prompt batches enter the cache at the cold end), two rounds:

| | off | on |
|---|---|---|
| prefill | 87.0 / 90.4 | 68.9 / 64.7 (-25 %) |
| first decode request after the prompt | 8.8 / 8.7 | 11.6 / 11.5 (+31 %) |

A net loss for prompts (about 22 s more per 6k-token prompt for one faster request), so it is off by default (`MOE_CACHE_SCAN_TOKENS=N` to try).
**Superseded (2026-10-15, section 48):** with prompt reads merged (section 38) and a 512-token threshold it now wins: Flash-Next prefill +27 %, SSD reads -58 %; on by default since then.


## 21. Approximate routing (opt-in), SSD-bound setting (2026-10-11)

What it is: with `MOE_CACHE_ROUTE=skip|subst` the executor looks at the two lowest-ranked picks of every token (the ones with the smallest router weights). If such an
expert is not in RAM it is dropped (skip) or replaced by the most recently used resident expert of that layer (subst), so no read from the SSD is needed.
Exact routing stays the default. Setting: Flash-Next IQ2_XS, 24 GB limit, RAM cache **6 GiB** (18 % of the experts, similar to a 100 GB model on 32 GB), 8 requests of 48 tokens.

| | exact | skip | substitute |
|---|---|---|---|
| decode tok/s (mean of requests 4-8) | 6.6 | **7.4 (+12 %)** | **7.3 (+10 %)** |
| low-ranked picks looked at / given up or replaced | n/a | 299,970 / 97,682 given up | 301,590 / 96,661 replaced |
| RAM cache hit rate | 74.6 % | 79.1 % | n/a |
| data read from the SSD (whole run) | 578 GB | 454 GB (-21 %) | n/a |

Quality (28 prompts of `bench/quality.py`, answers compared with the exact run, `compare-files`):

| | skip | substitute |
|---|---|---|
| checkable prompts answered correctly (26) | 26 (exact: 26) | 26 (exact: 26) |
| new degenerate answers | 0 | 0 |
| mean text similarity to the exact answers | 0.89 | 0.88 |
| answers identical to the exact run | 17 of 30 | 20 of 30 |
| judged "same meaning" by the model itself | 57 % | 67 % |

The judge figures are **not reliable**: the judge (the 2-bit model itself) marked pairs with text similarity 0.95-0.98 as different. Use `bench/quality.py judge-calibrate`
to check whether a server is a usable judge (9 of 10 known pairs). The result to rely on is the checkable-prompt accuracy and the absence of degenerate output.

Conclusion: a 10-12 % speed-up on an SSD-bound setting for no measured loss on 26 factual / arithmetic / format prompts; open-ended answers change wording. Not enabled by default.
Untested: models with a lower cache hit rate, long outputs, code generation beyond the 3 code prompts.

## 22. Coding-session benchmark: prompt reuse on a 20k-token context (2026-10-11, Flash-Next IQ2_XS)

What it is: `bench/coding_session.py` puts the first four files of this repository (44,989 characters = **19,949 tokens**) in front of every request, asks five
different questions, then edits the middle of the context and asks once more. All requests use `cache_prompt=true` and the same slot.
Setting: `moe-cache-server --ctx 20000 --gpu-cache auto` (tuner: `-ub 1024`, llama.cpp GPU cache 2459 MiB, `--n-cpu-moe 999`, `--cache-ram 2048`), 24 GB limit, swap 0,
KV f16, plugin warm-started from a learned profile, executor with 4 E-cores. llama.cpp 23b0202a1, plugin build with API 3.
**Not a clean-gate measurement**: no temperature gate (package 74 C at the end), one run, so read the ratios, not the absolute numbers.

| # | Request | Tokens processed | Tokens reused | Time to first token (s) | Decode tok/s | Wall (s) |
|---|---|---|---|---|---|---|
| 1 | first, cold | 19,949 | 0 | 785.0 | 9.2 | 791.8 |
| 2-5 | same context, new question | 1,018-1,026 | 18,921 | 41.6-43.4 | 7.3-9.2 | 48.4-51.5 |
| 6 | after an edit in the middle | **19,964** | **0** | 805.3 | 9.5 | 812.0 |

Findings:
- Reuse works for an unchanged prefix: a follow-up question costs about 50 s instead of 790 s.
- About 1,020 tokens are re-processed on every follow-up, not only the new question (it is probably the checkpoint granularity of the hybrid model; not verified).
- **An edit anywhere inside the context invalidated all of it** (0 tokens reused, 805 s): this model keeps recurrent state that cannot be cut back to an arbitrary position.
  For editing workflows the context should be ordered so that the changing part comes last.
- Prefill ran at 24-25 tok/s, about 4x below the plugin-only 82-140 tok/s of section 20. The cause was found afterwards: the long-prompt GPU block could not allocate VRAM next to the
  GPU expert cache and silently stayed on the CPU (section 23).
- `bench/coding_session.py` printed "about 13,232 tokens" for the context: its estimate (3.4 characters per token) is off by 1.5x for this tokenizer on code (2.25 characters per token measured).

## 23. Why prefill was 4x slower with the GPU cache, and the fix (2026-10-12, Flash-Next IQ2_XS, ctx 8192, `-ub 1024`)

What was wrong: for prompts of 32 tokens or more the plugin uploads one whole layer of expert weights to the GPU and computes the expert block there
(`gpu_segment`). With llama.cpp's GPU expert cache sized to fill the card, that upload (about 923 MiB per layer for this model: 704 MiB of weights plus the
batch's intermediate results) did not fit. Two things happened: a failed allocation disabled the block for the rest of the session (the coding-session job, 24 tok/s),
or the CUDA pool aborted the server (`cuMemCreate` out of memory, two crashes at 2459 MiB). The warning line is new.

Fix: the plugin checks free VRAM before the upload and keeps the block on the CPU with a one-time message (no abort, no permanent disable); the tuner keeps
`1.35 x layer + 160 MiB` free when it sizes the cache (Flash-Next, ctx 8192: 2911 -> 1800 MiB; at ctx 20000 the cache would be under 1.5 GiB, so it uses 3 static layers).

Gated runs (start <= 65 C, 24 GB limit, swap 0, two prompts of about 3.7k tokens, then six 48-token requests; one run per row, first-to-last decode requests include the cache warming):

| Server | Start C | Prefill tok/s | Decode tok/s (mean of last 4 of 6) | Long-prompt GPU blocks |
|---|---|---|---|---|
| cache off | 53 | 50.7 / 48.9 | 13.4 | 376 on the GPU |
| cache 2459 MiB (old auto-like) | 54 | 24.1 / 24.2 | 15.6 | 0 on the GPU, 376 on the CPU for lack of memory |
| cache 1800 MiB (new auto) | 55 | **55.8 / 59.6** | 14.8 | 376 on the GPU |
| cache 2100 MiB, old plugin (step 1) | 55 | 54.9 / 58.6 | not measured | worked (about 1165 MiB free) |

Reading: prefill is 2.3x faster than with the oversized cache; decode is about 5 % lower than with it and about 10 % above cache-off (single runs, +-5 %).
Token identity with the new plugin vs stock + the same cache (`--moe-cache-mib 1800 -ub 1024`, `tests/identity.sh --long --tokens 24`): 2/2 identical, 188 GPU blocks, 0 fallbacks.
Caveats: only 2 long prompts in the identity run; the second prefill prompt returned no text in every run (also with the cache off, unexplained, does not affect timing).

## 24. Prompt reuse, saved contexts, where decode time goes (2026-10-12, Flash-Next IQ2_XS, 24 GB limit, ctx 8192 / 4096)

All runs gated (start <= 65 C), one run per row, plugin build with the section 23 fix.

### 24.1 What llama-server re-processes on a follow-up (coding session, `bench/coding_session.py`, context of 5.4-5.9k tokens)
| `-ub` | extra flags | cold prefill (s) | follow-up processed / reused | follow-up TTFT (s) | edit in the middle |
|---|---|---|---|---|---|
| 1024 | default | 86 | 1,018-1,026 / 4,902 | 15.7 | 1,043 processed, 4,902 reused (the edit happened to lie after the checkpoint) |
| 256 | `-cms 256 -ctxcp 64` | 133 | 250-258 / 5,670 | 9.0 | **5,945 processed, 0 reused** |
| 512 | warm-up in 4 growing sections (`--sections 4`) | 29 + 29 + 38 | 506-514 / 5,427 | 11.1 | **886 processed, 5,072 reused** |

- The tail that is processed again on every follow-up is about one micro-batch (`-ub`): this hybrid model keeps one recurrent-state checkpoint per request, at the
  prompt's end minus `-ub`. `--checkpoint-min-step` and `--ctx-checkpoints` did not add checkpoints inside a prompt.
- An edit before that checkpoint reuses nothing (0 reused at `-ub 256`, 0 reused in the 20k-token run of section 22).
- Warming the context in a few growing prefixes (each one a request with `n_predict` 1) puts a checkpoint near each prefix's end; the later edit then reuses everything
  before the nearest checkpoint. Total warm-up time is the same as one cold prefill. `bench/coding_session.py --sections N` does this.
- Smaller `-ub` makes follow-ups cheaper (258 vs 1,026 tokens) and the cold prefill slower (133 vs 86 s).

### 24.2 Saving and restoring a context (llama-server `--slot-save-path`, `/slots/0?action=save|restore`)
| Step | Result |
|---|---|
| first request, 2,722-token context | 56 s |
| save | 0.15 s, 439 MB file |
| restart the server, restore | 0.25 s, 2,761 tokens restored; the same question then processed 4 tokens |
| generated text after restore vs the unsaved run | identical (192 characters) |
`moe-cache-server --project NAME` keeps one folder per project, restores the saved state at start, and `bin/moe-cache-project save NAME` saves it.

### 24.3 Where the expert time goes (executor counters, 12 prompts x 64 tokens, all experts on the CPU, cache 21 GiB)
| Weight type | Ops | Time (s) | Share | Expert data (GB) |
|---|---|---|---|---|
| IQ2_S | 52,490 | 17.7 | 49 % | 285.7 |
| Q2_0 | 37,053 | 12.1 | 33 % | 176.8 |
| IQ2_XXS | 16,984 | 5.5 | 15 % | 74.1 |
| IQ1_M | 4,632 | 1.45 | 4 % | 17.1 |
Time includes waiting for experts that were still loading. IQ2_S has not been tried in the kernel-variant tests (only IQ2_XXS and IQ3_XXS were): it is the first candidate.

### 24.4 Cache policy and next-layer prediction (offline replay of the same 769-token trace, `tools/trace_replay.py`; second half evaluated)
| Cache (GiB) | LRU | 2Q | 25 % pinned + LRU | 50 % pinned | 75 % pinned |
|---|---|---|---|---|---|
| 4 | 68.4 % | 67.6 % | 68.0 % | 66.9 % | 62.9 % |
| 8 | 83.3 % | 82.6 % | 83.1 % | 82.8 % | 82.7 % |
| 12 | 91.5 % | 91.5 % | 91.5 % | 91.3 % | 91.4 % |
| 16 | 95.5 % | 95.6 % | 95.3 % | 95.2 % | 95.3 % |
| 19.5 | 97.2 % | 97.0 % | 97.2 % | 97.3 % | 97.3 % |
Policy does not matter (at most 0.5 points apart from the 4 GiB pinned rows): LRU stays. Cheap next-layer predictors from the trace alone catch 36-57 % of the
picks with 20 predictions per 10 picks. The plugin's own lookahead predictor (router weights of the next layer, `MOE_CACHE_LOOKAHEAD=1`) reached **68.1 % recall**
(241,479 of 354,760 picks; 38,667 experts prefetched, 25,809 used, 9,413 wasted, 8,006 late), but decode with it was 9.2-10.6 tok/s against 11.8-13.6 without
(same prompts, one run each, helper threads compete for the CPU). This is the number that sets the ceiling of a GPU-side prefetch (GPU_TIER_DESIGN.md step 2).
Caveat: 12 short prompts, 769 decode tokens, cold start.

## 25. Quality evidence: quantisations, thinking, approximate routing, code generation (2026-10-12, Flash-Next, 24 GB limit)

Gated runs (start <= 65 C). Quality numbers are on small banks: read them as "no large difference found", not as proof of equality.

### 25.1 The three quantisations on disk (all experts on the CPU, ctx 4096, `-ub 512`)
| File | Size | Checkable prompts correct (of 26) | Perplexity (6 x 512 tokens of this repository's docs) | Collect wall time (30 prompts x 150 tokens) |
|---|---|---|---|---|
| IQ2_XS (ISTA) | 68.0 GB | 26 | 5.789 +- 0.381 | 319 s |
| Q2_0 (ISTA) | 66.4 GB | 25 | 5.796 +- 0.381 | 314 s |
| UD-Q2_K_XL (unsloth) | 78.9 GB | 26 | 5.775 +- 0.373 | 403 s |
The perplexities are inside each other's error bars (6 chunks); mean text similarity of the open answers to IQ2_XS is 0.81 (Q2_0) and 0.75 (Q2_K_XL), no new degenerate answers.
Decode speed with the plugin was 12.5 (IQ2_XS) vs 11.1 tok/s (Q2_K_XL) in section 18. Nothing measured here argues for the larger file on this machine.

### 25.2 Thinking (IQ2_XS, same bank, chat template)
| Setting | Wall time for the bank | Correct | Notes |
|---|---|---|---|
| default | 319 s | 26 / 26 | |
| `--reasoning off` | 226 s (-29 %) | 25 / 26 | mean similarity 0.56 to the default answers |
| `--reasoning-budget 256` | 314 s | 26 / 26 | answers identical: the 150-token answer cap in this bank never reaches the budget, so this row says nothing about the budget |

### 25.3 Approximate routing against exact (KL divergence, 3 x 512 tokens, batch 8 so the executor handles every step, 6 GiB RAM cache)
| | exact | skip | substitute |
|---|---|---|---|
| perplexity | 4.498 | 4.560 (+1.4 %) | 4.597 (+2.2 %) |
| mean KL divergence to exact | 0 | 0.0253 +- 0.0018 | 0.0355 +- 0.0028 |
| same top token as exact | 100 % | 93.6 % +- 0.9 | 93.1 % +- 0.9 |
Code generation (10 small Python tasks run against assertions, `quality.py codegen`, thinking off, same 6 GiB cache): exact 10/10, skip 10/10, substitute 9/10 (`roman_to_int`).
Together with section 21 (26/26 checkable prompts, +10-12 % speed) this is the first distribution-level number for the approximate modes: roughly one in fifteen
top-1 tokens changes, and the loss in perplexity is 1-2 %. 3 chunks and 10 tasks are small; the perplexity text is technical English, not code or chat.

## 26. GLM-5.3-Flash UD-IQ2_XXS (101.8 GB) on the 31 GB / 8 GB laptop (2026-10-14)

llama.cpp 23b0202a1, plugin built for API 3, 24 GB hard limit (swap off), ctx 4096, all experts on the CPU side (`--n-cpu-moe 999`; the non-expert part fills the 8 GB card,
so no GPU expert cache fits: 1500 MiB failed to allocate), 4 prompts x 24 tokens, **not temperature-gated** (read ratios, not absolute values). One run per row.

| Run | Identical to stock | Decode tok/s, requests 1-4 |
|---|---|---|
| Stock llama.cpp | - | 0.33, 0.30, 0.33, 0.32 |
| Plugin 0.2.3.2 (executor takes up to 8-token batches) | **3/4** (one prompt differs at its first token) | 1.64, 1.91, 1.70, 1.79 |
| Plugin, executor single-token only (`MOE_CACHE_EXEC_MAXCOL=1`) | **4/4** | 1.69, 1.93, 1.67, 1.70 |
| + 4 efficiency cores | 4/4 | 1.74, 2.00, 1.66, 1.78 |
| + faster IQ3_XXS kernel (1.24x on the dot product, bit-exact) | 4/4 | 1.69, 1.83, 1.66, 1.71 |
| + 8 reader threads | 4/4 | 1.72, 1.92, 1.75, 1.80 |
| + pre-faulting replacement pages | 4/4 | 1.80, 1.78, 1.68, 1.76 |
| `--fast` (route skip) | 2/4 (approximate by design) | **2.11, 2.31, 2.03, 2.08** |
| 28 GB limit (capped by the plugin to the 24.4 GB actually available) | 4/4 | 1.69, 2.04, 1.67, 1.78 |

- **About 5.5x stock**, token-identical, on a model 3.3x larger than the memory it may use.
- Why the identity failed first: with the byte check on (`MOE_CACHE_EXEC_CHECK=1`) 3 of 12,348 executor ops differed from llama.cpp in the last bit, all in one 8-token prompt
  batch of layer 17 (IQ2_XXS gate/up and IQ3_XXS down); single-token decode was exact. Stock vs stock was identical (noise floor PASS). **Cause found:** in all 3 cases one expert was picked by all 8
  tokens of the batch, and llama.cpp 23b0202a1 then uses its tiled routine (ggml-cpu/tiled, taken at 8 or more rows per expert), which sums in another order; recomputing
  the value by hand with the one-row kernel matched the executor. The executor now leaves such ops to llama.cpp: 0 mismatches in 12,345 checked ops, 4/4 identical,
  multi-token batches still on.
- Where the time goes: cache hit rate 56 %, about 1.5 GB read from the SSD per token, about 2.6 GB/s of the drive's 3.6 GB/s. Decode is limited by SSD reads, which is why the
  kernel, more readers, efficiency cores and pre-faulting did not change the speed; only reading less (`--fast`: 21 % fewer bytes) did. The ceiling at this hit rate is about
  2.4 tok/s exact. More RAM (higher hit rate) is the lever: see the bigger-machine list.
- IQ3_XXS takes about 80 % of the executor's expert time on this model; its new kernel pays only once the reads stop dominating.
- A 28 GB cgroup limit was OOM-killed by the kernel at 26.3 GB because only about 25 GB was free with the desktop running; the plugin now caps its limit at what is available.

## 27. Executor in a build with separately loaded backends (2026-10-14)

llama.cpp 7fe450e19 built with Vulkan and loadable CPU backends (how official releases are built), plugin loaded with `GGML_BACKEND_PATH`, Qwen3.8-Flash-Next IQ2_XS,
24 GB limit, all experts on the CPU side, 4 prompts x 24 tokens, not temperature-gated. The executor now finds llama.cpp's CPU kernels in the separately loaded
`libggml-cpu-*.so` (before, it switched itself off in this kind of build).

| | Identical to stock | Decode tok/s, requests 1-4 (cache warming up) |
|---|---|---|
| Stock (Vulkan build) | - | 1.43, 2.27, 3.00, 3.18 |
| Plugin (executor on, Q2_0 and IQ3_XXS kernels inside the executor) | **4/4** | 2.74, 3.93, 6.28, 7.15 |

## 28. Flash-Next settings on the 8 GB card (2026-10-14)

Qwen3.8-Flash-Next IQ2_XS, `moe-cache-server` (llama.cpp 23b0202a1, API 3 build), 24 GB limit, ctx 4096, 8 requests x 48 tokens, clean-run gate (<= 65 C) before each run,
one run per row; request-to-request speed varies 8-14 tok/s, so differences under about 10 % are noise.

| Setup | Decode tok/s, mean of requests 4-8 | Same tokens as row 1 |
|---|---|---|
| `--gpu-cache auto` (tuner: 1956 MiB, 1110 MiB kept free for long prompts) | 13.09 | - |
| same with `--ub 512` (cache size unchanged: 1957 MiB) | 11.59 | 8/8 |
| `--gpu-cache 2900` (no room kept for long prompts) | 12.25 | 8/8 |
| `--gpu-cache off` (3 static GPU layers, executor + kernels on the CPU) | 12.36 | 6/8 (GPU vs CPU rounding) |

- With the GPU expert cache, llama.cpp computes every expert on the GPU and uploads the ones it does not hold; the plugin's executor runs 0 ops (it only keeps experts in RAM).
  So CPU kernels cannot speed this setup up; it is limited by PCIe uploads (about 285 MiB per token).
- A faster IQ2_S kernel (gathers for the grid lookups, bit-exact) measured 0.99x the library function and was dropped.
- What is left on this card: let the CPU compute the experts the GPU cache misses, in parallel, instead of uploading them (GPU_TIER_DESIGN.md step 3, MULTI_DEVICE_DESIGN.md).

## 29. Fits in RAM vs not, with llama.cpp's GPU expert cache in both (Qwen3.6-35B-A3B Q4_K_M, 21.7 GB; 2026-10-14)

llama.cpp 23b0202a1, same arguments for both (`--n-cpu-moe 999 --moe-cache-mib 5000 -ub 1024 -C 0x555 --cpu-strict 1 --cache-ram 2048`, ctx 4096), plugin with
`MOE_CACHE_GPU_CACHE=1` and 4 efficiency cores; 8 requests x 48 tokens, clean-run gate before each run, order stock, plugin, plugin, stock, then 12 GB.

| Limit | Run | Decode tok/s per request | Mean of requests 4-8 |
|---|---|---|---|
| 24 GB (fits) | stock | 30.6 38.0 40.7 39.3 47.1 45.7 40.0 39.7 | **42.3** |
| 24 GB | plugin | 9.6 13.7 20.9 27.8 31.2 31.5 28.3 31.3 | 30.0 |
| 24 GB | plugin | 9.8 14.0 22.2 31.7 34.9 35.8 32.2 36.3 | 34.2 |
| 24 GB | stock (page cache emptied by the plugin runs: re-reads from the SSD) | 4.4 7.2 12.0 26.1 23.3 28.2 20.8 27.2 | 25.1 (not comparable) |
| 12 GB (does not fit) | plugin | 11.8 11.8 11.9 11.5 22.0 18.2 12.4 14.4 | **15.7** |
| 12 GB | stock | 4.4 4.6 5.1 4.5 7.5 6.6 4.3 4.7 | 5.5 |

- When the model fits, stock with a warm page cache is faster (42 vs 30-34 tok/s): after its margins the plugin's own cache gets 17.5 GiB and holds 90 % of what is
  used, while stock's page cache holds the whole 21.7 GB file. Use stock llama.cpp when the model fits (README "When to use something else").
- When it does not fit (12 GB), the plugin is 2.9x stock.

## 30. Hybrid hit/miss prototype: GPU bank + CPU executor in parallel (Flash-Next, 8 GB card; 2026-10-14)

`MOE_CACHE_HYBRID=MiB|auto` (opt-in, experimental, not token-identical): a bank of the most used experts (from the usage profile) lives in GPU memory; for every
decode layer the GPU computes the picks whose expert is in the bank while the executor computes the others on the CPU, then the bank's outputs replace those picks'
columns. Nothing is uploaded per token except the activation and the ids. Same setup as section 28 (`--n-cpu-moe 999 --gpu-cache off`, 24 GB limit, 8 requests x 48 tokens,
clean-run gate, one run per row); "same tokens" is against the `--gpu-cache auto` run of section 28.

| Setup | Bank | Picks computed from the bank | Decode tok/s, mean of requests 4-8 | Same tokens |
|---|---|---|---|---|
| llama.cpp GPU cache (`--gpu-cache auto`, section 28) | - | - | 13.09 | - |
| `MOE_CACHE_HYBRID=3000` | 2,175 experts, 2,999 MiB | 47.6 % | **16.81** | 7/8 |
| `MOE_CACHE_HYBRID=4500` / `5500` | could not be allocated: all picks on the CPU | 0 % | 11.08 / 10.88 | 6/8 |
| `MOE_CACHE_HYBRID=auto` (free GPU memory minus 600 MiB) | 1,828 experts, 2,519 MiB | 50.8 % | **17.40 (+33 %)** | 5/8 |

- In line with the estimate from the measured costs (+25-40 %, GPU_TIER_DESIGN.md).
- Not yet measured: quality (KL divergence / answer comparison), prompt processing (the bank takes the memory the long-prompt GPU block would use), a bank that adapts
  to the session instead of the static profile choice, and other models.

## 31. Hybrid quality, and an exactness race at 8-token batches (Flash-Next IQ2_XS, 8 GB card, 24 GB limit; 2026-10-14)

Method: `llama-perplexity` over 20 chunks x 512 tokens (chunks 1-10 prose from llama.cpp's docs, 11-20 C++ and Python from its sources), batch 8 (`-b 8 -ub 8`) so every
step goes through the executor and the hybrid, `--n-cpu-moe 999 -ngl 99`, KL divergence against the exact plugin (exact = stock: section 26, identity tests).
KL runs are quality runs, not speed runs (not temperature-gated).

### 31.1 A race found on the way: exact runs were not repeatable at batch 8
The first exact-vs-exact control gave mean KL 0.0038 (98.4 % same top token) instead of 0. Controls on Qwen3-Coder-30B-A3B Q4_K_M (fits in RAM; 3 chunks, batch 8):

| Run | Mean KL vs stock | Same top token |
|---|---|---|
| stock vs stock | 0.000000 | 100 % |
| plugin, model fits (no evictions, executor steps aside) | 0.000000 | 100 % |
| plugin, 10 GB (evictions), executor off | 0.000000 | 100 % |
| plugin, no evictions, executor forced on | 0.000000 | 100 % |
| **plugin, 10 GB (evictions) + executor forced on** | **0.0063** | 98.2 % |
| same, with the fix | 0.000000 | 100 % |

Cause: in executor graphs the expert reads are asynchronous (the executor waits for each expert's flag itself). Since 804cd6f the executor hands ops back to llama.cpp
when its tiled routine would run (8+ tokens of the batch on one expert); those ops did not wait, so llama.cpp could read an expert that was still being filled.
The executor's byte check (`MOE_CACHE_EXEC_CHECK=1`) did not see it because its recompute runs after the executor has waited. Fix: wait for the op's experts before handing
it back. Only `experimental` had it (no release); in the server it needs a batch of exactly 8 tokens (decode is 1 token), so it was rare there.
New test `tests/kl_batch.sh` (logits at batch 8, stock vs plugin, must be KL 0) and a CI step running it with forced eviction and the forced executor.

### 31.2 Hybrid vs exact (fixed plugin)
| Run | Picks computed from the GPU bank | Mean KL vs exact | 99 % KL | Same top token | PPL ratio |
|---|---|---|---|---|---|
| exact again (noise floor) | - | **0.000000** | - | 100 % | 1.000 |
| `MOE_CACHE_HYBRID=auto`, bank chosen for this text (3,640 MiB, 2,651 experts) | 56.9 % | **0.0043 +- 0.0003** | 0.066 | 98.4 % | 1.0027 +- 0.0020 |
| `--fast` (route skip; at 24 GB it gives up 3.2 % of the low-ranked picks) | - | 0.0053 +- 0.0003 | 0.088 | 98.1 % | 1.0026 +- 0.0021 |
| `--fast` with a 6 GiB cache (gives up 31 % of the low-ranked picks) | - | 0.0115 +- 0.0006 | 0.167 | 97.5 % | 1.0034 +- 0.0030 |

- The bank here is larger than in the server (3.6 vs 2.5 GB: perplexity at ctx 512 leaves more VRAM free) and was chosen from this text's own usage, so 57 % of the picks
  ran on the GPU (server: 51 %). With the chat profile only 14 % did; a bank that does not match the work does little, for speed and for quality.
- Differences come from the GPU adding up in another order than the CPU (same reason llama.cpp's own GPU expert cache is not bit-identical). About 1 top token in 60
  changes; the perplexity change is inside its error.
- `--fast` at 24 GB (KL 0.0053) is much closer to exact than at a 6 GiB cache (0.025, section 25.3): it only skips experts that are not in RAM.
- Yardsticks not run: llama.cpp's own GPU expert cache (`llama-perplexity --moe-cache-mib` at batch 8 stopped with `GGML_ASSERT(buffer)`; see 31.3) and two 2-bit quants
  of the same model (the Q2_0 / UD-Q2_K_XL copies are no longer on disk; on hold).
- Section 25.3 (approximate routing, KL 0.025 / 0.036 at a 6 GiB cache) was measured at batch 8 with evictions, so the race above is in those numbers too (both in the
  exact base and in the approximate runs). The 24 GB and 6 GiB rows here replace it (6 GiB: 0.0115 instead of 0.025).

### 31.3 Answers and code (`moe-cache-server --hybrid`, launcher, 24 GB limit)
Reference: the exact answers of section 25.1 (same model, same bank of prompts, 150 tokens). The hybrid server judged "same meaning" itself.

| | Exact | `--hybrid` |
|---|---|---|
| checkable prompts correct | 26 / 26 | **26 / 26** |
| judged "same meaning" (2-bit model as the judge) | - | 21 / 30; read by hand: **30 / 30** (the 9 "different" ones are punctuation, rewording such as "looks" for "appears", or the 150-token cut falling elsewhere) |
| mean text similarity / new degenerate answers | - | 0.91 / 0 |
| code tasks, thinking off (as in 25.3) | 10 / 10 | **10 / 10** |
| code tasks, thinking on (1,200 tokens) | not run | 9 / 10 (`word_count`: empty answer, the thinking used all tokens) |

Verdict: the hybrid passes as an opt-in mode (KL below `--fast`, no lost answers); `moe-cache-server --hybrid` added.

### 31.4 Plugin + llama.cpp's GPU expert cache in llama-perplexity at batch 8
`llama-perplexity --moe-cache-mib 1000 -b 8` runs with stock llama.cpp but stops with `GGML_ASSERT(buffer)` (ggml-backend.cpp:273) at the first compute with the plugin
(`MOE_CACHE_GPU_CACHE=1`), on Qwen3-Coder and Flash-Next. The server with the same setup works (section 28); `llama-perplexity` does a warm-up decode that the launcher
switches off, which is the first suspect. Open.

## 32. Prompt speed with the hybrid (Flash-Next IQ2_XS, 8 GB card, 24 GB limit, ctx 8192; 2026-10-14)

Launcher defaults, clean-run gate (start 53-56 C), one short request first (as in a chat, so the hybrid bank is sized before any long prompt), then two prompts of
about 2k and two of about 5k tokens (different text, no prompt cache).

| Setup | Prefill tok/s | Long-prompt blocks on the GPU |
|---|---|---|
| `--gpu-cache auto` (`-ub 1024`, cache 1799 MiB) | 63.5 / 60.9 / 59.8 / 61.9 | 752 of 752 |
| `--gpu-cache off` (`-ub 2048` from `--ub auto`) | 21.3 / 17.6 / 18.0 / 17.8 | 66 of 414: no room left for the block next to `-ub 2048` |
| `--hybrid`, bank kept (section 30 code) | 17.9 / 17.4 / 16.8 / 17.3 | 0 of 423: the bank holds the memory |
| **`--hybrid`, bank freed for the long prompt, loaded again when decoding starts** | **73.5 / 75.2 / 66.5 / 71.5** | 423 of 423 |

- Freeing and reloading the 2.4 GB bank cost 0.96 s per long prompt (2.85-5.6 s over a 30-prompt session, including short chat prompts of about 35 tokens, where
  running the block on the CPU would take longer than the reload).
- `--gpu-cache off` with `--ub auto` is a launcher problem of its own (the tuner picks `-ub 2048` without keeping room for the long-prompt block): to fix.

## 33. Adaptive hybrid bank (Flash-Next IQ2_XS, 8 GB card, 24 GB limit; 2026-10-14)

`MOE_CACHE_HYBRID_ADAPT=N,K` (default 8,16 whenever the hybrid is on; 0 = off): every N tokens up to K of the experts the bank missed most replace the coldest expert
of their own layer's bank, when they were used more often (decayed hit counts). Uploads happen between tokens, when the GPU is idle.

Offline first (`tools/bank_replay.py`, trace of 6,542 decode tokens recorded during section 31.3, 1,827-expert bank, profile as before the session):
| Policy | Picks served by the bank | Experts uploaded per token |
|---|---|---|
| fixed (profile) | 19.6 % | 0 |
| best static bank for this trace | 38.1 % | 0 |
| adaptive 8,16 (per-layer slots) | **40.5 %** | 2.0 |
| adaptive 8,64 | 44.5 % | 8.0 |

Live, clean-run gate, one run per row:
| Run | Bank hit | Decode tok/s |
|---|---|---|
| 8 requests x 48 tokens, fixed bank (profile updated by today's runs) | 20.3 % | 14.91 (mean of 4-8) |
| same, adaptive | 30.7 % | 15.21 |
| 30 mixed-topic prompts x 96 tokens (quality bank), fixed bank | 20.2 % | 13.10 (mean of 6-30) |
| same, adaptive | **39.3 %** | **14.43 (+10 %)** |
| 8 requests x 48 tokens, fixed bank, profile as before today's runs (the one section 30 used) | 51.0 % | **17.58** (section 30: 17.40, so no regression) |
| same, adaptive | 47.2 % | 16.24 |

- When the session drifts from the profile (the normal case: the profile is the sum of past work) the adaptive bank doubles the bank's hits and gains 10 %.
- When the profile already matches the prompts exactly (here: the profile was learned on these 8 benchmark prompts) the swaps cost 4 points of hits (replay: 1.5 points).
  Tried offline and dropped: starting slot scores from the profile, a lasting profile prior (helps the matching case by 1 point, costs 5 points in the drifting one),
  fewer or rarer swaps, a higher minimum miss count; all trade the two cases against each other.
- Upload cost is small: 2 experts per token (about 2.8 MB), 0.26-1.7 s per run in total.

### 33.1 The bank switches itself (shadow comparison, `MOE_CACHE_HYBRID_DUEL=W`, default 32)
The plugin also counts what the profile's own bank would have served (a shadow, one table lookup per pick). Every swap round it compares the two over the last W tokens:
while the shadow is ahead (the session matches the profile) it makes no new swaps and brings profile experts back; otherwise it keeps adapting.
Replay: matching profile 48.2 % (fixed) / 46.7 % (adaptive) / **48.0 %** (with the shadow); drifting session 19.6 / 40.5 / **40.4 %**.

| Live run (clean-run gate) | Fixed bank | Adaptive + shadow |
|---|---|---|
| 8 x 48 tokens, profile matching the prompts | 17.58 tok/s, 51.0 % hits (above) | **17.05**, 50.2 % (323 profile experts brought back) |
| 30 mixed prompts, same profile snapshot for both | 12.77, 15.0 % | **14.88 (+16.5 %)**, 31.6 % |

The loss in the matching case is now inside the noise (-3 %), the drifting gain stays; on by default whenever the hybrid is on.

## 34. Hybrid on GLM-5.3-Flash and on Qwen3.6-35B at 12 GB (8 GB card; 2026-10-14)

Clean-run gate, one run per row.

| Model, limit | Run | Picks on the GPU | Decode tok/s | Identical to stock |
|---|---|---|---|---|
| GLM-5.3-Flash UD-IQ2_XXS, 24 GB, 4 x 24 tokens | plugin exact (learned a usage profile) | - | 1.72 / 1.89 / 1.78 / 1.90 | 4/4 |
| same | `MOE_CACHE_HYBRID=auto` (only 716 MiB free next to the non-expert part: 99 experts) | 15.9 % | 1.84 / 2.06 / 2.00 / 2.03 | 2/4 |
| Qwen3.6-35B-A3B Q4_K_M, 12 GB, 8 x 48 tokens | plugin + llama.cpp GPU cache 5000 MiB (section 29) | - | 15.7 (mean of 4-8) | - |
| same | `MOE_CACHE_HYBRID=auto` (5,230 MiB bank, adaptive) | 55.2 % | **8.74** | - |

- GLM: the bank is too small on 8 GB to matter, and decode is limited by SSD reads (57 % cache hits); a 12-16 GB card (BIGGER_MACHINES B12).
- Qwen3.6 (Q4_K experts): the hybrid loses to llama.cpp's GPU cache by almost half. The CPU computes Q4_K fast and the experts are small, so the hybrid's per-layer
  GPU round trip (graph, activation upload, sync, copy back, 40 layers per token) costs more than it saves. The hybrid helps models whose experts are slow on the CPU
  (Flash-Next: IQ2_XS / Q2_0 with the plugin's executor); for Q4_K-class models keep `--gpu-cache`.

## 35. Prompt speed and time to first token on the default path (Flash-Next IQ2_XS, 8 GB card, 24 GB limit, ctx 8192; 2026-10-14)

Clean-run gate, one run per row; same prompts as section 32.

| Run | Prefill tok/s (2k / 2k / 5k / 5.4k) | Note |
|---|---|---|
| launcher default (`--gpu-cache off`, `--ub auto` = 2048) before | 21.3 / 17.6 / 18.0 / 17.8 | the long-prompt block lacked 10 MiB (needs 1,107 MiB per layer at `-ub 2048`, 1.57x the layer) |
| **same after the tuner fix** (static expert layers and `-ub` from one decision, block reserved per `-ub`) | **74.1 / 75.9 / 64.1 / 73.0** | about 4x for every default user |
| `--hybrid --ub 4096` (with `-b 4096`; the launcher passed `-b 2048` before, which capped `-ub`) | 95.7 / 77.3 / 26.0 / 24.2 | +30 % on the first 2k prompt, but the 5k prompts ran out of GPU memory on 8 GB and fell back to the CPU: for 12-16 GB cards (BIGGER_MACHINES B13) |
| `--hybrid`, next layer's experts read in the background during long prompts (`MOE_CACHE_PROMPT_PREFETCH=1`) | 52.9 / 75.4 / 65.5 / 70.5 | no gain: reader time moved to the helper threads (127 -> 41 s) but the wait did not shrink; off |

Where the time goes (section 32's hybrid run, about 210 s of prompts): about 100 s waiting for 242 GB of expert reads (46 % of what the prompt touched was in RAM),
54 s of GPU blocks (319 GB uploaded, about 6 GB/s), 21 s of CPU expert compute. Reads run at about 2 GB/s on a 3.6 GB/s drive: the next lever, not yet understood.

Time to first token, launcher default, by the threshold above which a prompt batch runs on the GPU (`MOE_CACHE_GPU_MIN_TOKENS`):
| Prompt tokens | 32 (default, as llama.cpp) | 128 | 512 |
|---|---|---|---|
| 36-37 | 5.17 / 4.02 s | **3.34 / 3.32 s** | 3.24 / 3.38 s |
| 109-113 | **4.73 / 4.26 s** | 6.92 / 6.84 s | 7.50 / 6.50 s |
| 180-186 | 5.86 / 5.53 s | 6.51 / 5.49 s | 12.35 / 12.27 s |
| 414-473 | 9.23 / 13.98 s | 8.67 / 13.68 s | 26.0 / 31.4 s |

The crossover is near 64 tokens. The exact default keeps 32 (stock llama.cpp also moves batches of 32+ tokens to the GPU, so the results stay identical);
`--hybrid` and `--fast`, which are approximate anyway, now use 64 (short chat prompts about 4.6 -> 3.3 s).

## 36. Identity on llama.cpp 23b0202a1: six more models and a second race (2026-10-15, overnight; 24 GB hard limit)

Each model: `tests/identity.sh` with a forced-eviction cache, the same with the executor forced on and byte-checked (`--no-repack`), and `tests/kl_batch.sh`
(batch-8 logits). Plugin 4c0a5e9 (gpt-oss: also 68f63e3 + the fix below).

| Model (architecture) | Identity, eviction | Executor forced + byte check | Batch-8 logits |
|---|---|---|---|
| Granite 3.1 1B-A400M (granitemoe) | 4/4 | 4/4, 0 of 6,708 ops differ | KL 0 |
| OLMoE-1B-7B (olmoe) | 4/4 | 4/4, 0 of 4,476 | KL 0 |
| SmallThinker-4BA0.6B (smallthinker) | 4/4 | 4/4, 0 of 8,940 | KL 0 |
| Granite-4.0-H Tiny (granitehybrid) | 4/4 | 4/4, 0 of 11,742 | KL 0 |
| LFM2-8B-A1B (lfm2moe) | 4/4 | 4/4, 0 of 6,402 | KL 0 |
| Ling-mini-2.0 (bailingmoe2) | 4/4 | 4/4, 0 of 5,313 | KL 0 |
| DeepSeek-V2-Lite 16B (deepseek2) | 4/4 | 4/4, 0 of 7,266 | KL 0 |
| gpt-oss-20b MXFP4 (gpt-oss) | **0/4** -> **4/4 with the fix** | 4/4, 0 of 7,053 | KL 0 |
| Qwen3-Next-80B-A3B Q4_K_M (qwen3next, 46 GB at 24 GB) | 4/4 (8 GiB cache) | 4/4, 0 of 14,100 | not run (stock would take hours at this limit) |
| Hunyuan-A13B Q4_K_M (hunyuan-moe, 46 GB at 24 GB) | 4/4 (8 GiB cache) | 4/4, 0 of 8,940 | not run |

**Second race (in releases v0.2.0-v0.2.3.2):** gpt-oss hands the gate, up and down products to the plugin in separate calls (bias operations in between).
When the gate's experts are loaded in an executor graph, the up and down experts are read along with them in the background. For the next call those experts
are "resident, still loading", so no read is queued, the executor steps aside (everything looks resident) and llama.cpp computed with experts that were
still being filled. The output differed from stock at the first token of every prompt at a 1 GiB cache. With the executor forced on it never stepped aside,
which is why the byte-checked run passed. Fix: before stepping aside, wait for the graph's experts. Models whose gate and up come in one call are not affected.

## 37. OLMoE-1B-7B Q4_0 at a 3 GB limit: repacking (T23; 2026-10-15, clean-run gate, 8 x 48 tokens, all experts on the CPU side)

| Run | Decode tok/s, mean of requests 4-8 | Cache | Hit rate |
|---|---|---|---|
| stock llama.cpp | 4.86 | - | - |
| plugin, CPU repacking on (default) | **9.97** | 1.65 GiB | 73.3 % |
| plugin, `--no-repack` | 8.77 | 1.25 GiB | 61.8 % |

The older observation (repacking slower in a 3 GB OLMoE run) does not reproduce. The plugin repacked no experts in either run; the difference is the cache size the
automatic sizing reaches.

## 38. Prompt reads merged (T38; Flash-Next IQ2_XS, 8 GB card, 24 GB limit, ctx 8192, launcher default `--gpu-cache off`; 2026-10-15)

Why the reads ran at about 2.5 GB/s: in a long prompt every missing expert tensor was its own read of about 0.46 MB, into freshly evicted pages; section 16 had
measured 2.6 GB/s for 0.48 MB reads at 4 threads and the drive's ceiling (3.6 GB/s) only for reads of 2 MB and more. Neighbouring experts of a tensor are neighbours
in the file and in memory, so the synchronous reads of a prompt are now merged into reads of up to 16 MiB (`MOE_CACHE_COALESCE_MIB`, default 16, 0 = off).
Clean-run gate, one run per row, same prompts as section 32.

| Run | Prefill tok/s (2.4k / 1.9k / 5.0k / 5.4k) | Waiting for reads | Read speed |
|---|---|---|---|
| merging off | 80.7 / 83.7 / 75.2 / 81.0 | 98 s for 245 GB | 2.5 GB/s |
| **16 MiB** | **91.9 / 102.7 / 93.0 / 100.2** | 72 s for 244 GB | 3.4 GB/s |
| 64 MiB | 85.5 / 89.9 / 82.2 / 86.6 | 75 s | 3.3 GB/s (fewer, larger jobs for the 6 readers) |
| 16 MiB + pre-faulting the pages (`MOE_CACHE_PREFAULT=1`) | 82.1 / 90.2 / 80.6 / 86.4 | 72 s | no further gain |
| **16 MiB as the new default** (launcher defaults only) | **90.5 / 99.2 / 90.2 / 98.2** | 73 s | 3.3 GB/s |

Identity with merging active (DeepSeek-V2-Lite, 1 GiB cache, direct reads forced): long prompts 2/2 (25,486 reads merged), short prompts 4/4 (8,276 merged).
Over the session the default prefill on Flash-Next went from 18 tok/s (section 32) to about 95 tok/s. The reads are now near the drive's limit; the next levers are
the uploads (319 GB at about 6 GB/s, pinned memory) and fewer uploads per token (larger prompt chunks on bigger cards, BIGGER_MACHINES B13).

## 39. Prompt-lookup drafting on code edits (T17; Flash-Next IQ2_XS, 8 GB card, 24 GB limit, launcher defaults; 2026-10-15)

4 requests, each a ~120-line Python file plus a small edit ("rename `tune` everywhere ... reply with the complete file"), thinking skipped, 320 tokens, clean-run
gate, one run per row. All drafted runs gave the same tokens as the run without drafting (4/4).

| Drafting (`-- --spec-type ...`) | Decode tok/s per edit | Mean | Drafts accepted |
|---|---|---|---|
| off | 11.00 / 11.88 / 8.19 / 10.10 | 10.29 | - |
| `ngram-mod` (drafts of about 55-59 tokens) | 12.17 / 12.69 / 12.53 / 6.78 | 11.04 | 100 / 100 / 100 / 57 % |
| `ngram-mod --spec-ngram-mod-n-max 8` (9-token verify batches) | 7.19 / 7.46 / 7.33 / 7.65 | 7.41 | 100 / 100 / 100 / 88 % |
| **`ngram-mod --spec-ngram-mod-n-max 7`** (8-token verify batches) | 11.56 / 14.38 / 8.99 / 9.35 | **11.07** | 100 / 100 / 100 / 90 % |
| `ngram-map-k` | 5.86 / 8.86 / 7.87 / 3.62 | 6.55 | 86 % |

- The gain is small (+8 %, single runs, at the noise level) even with every draft accepted: a verify step of 8 tokens wakes the experts of all 8, so it costs
  much more than one ordinary step (the reason the n-gram test of section 15 lost on Qwen3.6).
- Keep every speculative draft at 7 tokens or fewer with this plugin: the verify batch (draft + 1) must fit the executor's 8 tokens; at 9 every verify step
  falls back to llama.cpp's own path (-28 %). Long drafts (above 31 tokens) go to the long-prompt GPU block and collapse when a draft is rejected.
- Option for code-edit work: `moe-cache-server MODEL -- --spec-type ngram-mod --spec-ngram-mod-n-max 7` (exact, small gain). Not a default.

## 40. GLM-5.3-Flash with its own MTP layer (T46; UD-IQ2_XXS, 24 GB limit, 8 GB card, clean-run gate, 4 x 24 tokens; 2026-10-15)

The GGUF carries one next-token layer (`blk.45.nextn.*`); llama.cpp 23b0202a1 loads it with `--spec-type draft-mtp`.

| Run | Decode tok/s per request | Drafts accepted | Identical to stock |
|---|---|---|---|
| no drafting | 1.79 / 1.88 / 1.76 / 1.80 | - | 4/4 |
| `--spec-draft-n-max 2` | 1.24 / 1.46 / 1.31 / 1.30 | 48 % | 3/4 |
| `--spec-draft-n-max 3` | 1.24 / 1.16 / 1.44 / 1.38 | 39-46 % | 2/4 |

About -25 %: fewer than half the drafts are accepted, and every verify step reads the experts of all its tokens on a model that is limited by SSD reads.
(Verified tokens can differ from one-token decoding because the batch shape changes the GPU part's rounding, as with stock llama.cpp.) Dropped.

## 41. GLM-5.3-Flash decode reads: the drive is idle, not slow (T47; 24 GB limit, clean-run gate, 4 x 24 tokens; 2026-10-15)

During decode the 4 reader threads are busy about half the time (161 s of reader time in 81 s): the experts of a layer can only be read once its router has
chosen them, so reads come in bursts with gaps between layers. Filling the gaps needs reads for the next layer before it routes (a guess):

| Run | Decode tok/s per request | Mean | Identical |
|---|---|---|---|
| executor, no lookahead | 1.79 / 1.88 / 1.76 / 1.80 | 1.81 | 4/4 |
| lookahead, executor off (`MOE_CACHE_LOOKAHEAD=1`, the old behaviour) | 1.80 / 1.88 / 1.76 / 1.79 | 1.81 | 4/4 |
| lookahead + executor (`MOE_CACHE_LOOKAHEAD=2`, new), 2 guesses per layer | 1.92 / 1.94 / 1.90 / 1.98 | 1.94 | 4/4 |
| same, 4 guesses per layer | 1.79 / 1.87 / 1.85 / 1.85 | 1.84 | 4/4 |
| executor, no lookahead (again) | 1.82 / 2.00 / 1.88 / 1.98 | 1.92 | 4/4 |
| lookahead + executor (again) | 1.83 / 1.36 / 1.86 / 1.95 | 1.75 | 4/4 |

The prediction is right 54 % of the time on GLM (2 guesses per layer: 64 % of the prefetched experts used), but the wasted reads cost as much as the useful
ones save on a drive that is the bottleneck: no reliable gain (the two baseline runs differ by 6 %). With MTP (section 40) and read merging (decode reads are
single experts) also out, GLM on this laptop stays at about 1.8-1.9 tok/s exact and 2.0-2.3 with `--fast`: none of the approaches tried (four short prompts each) gave a reliable gain, which does not prove nothing else can; more RAM (BIGGER_MACHINES B6) is the best-supported lever.
`MOE_CACHE_LOOKAHEAD=2` stays as an experimental switch (exact).

## 42. Section 2 re-measured, and the executor slowing down K-quant models (T26; Qwen3.6-35B-A3B Q4_K_M, `--n-cpu-moe 34`, ctx 32768; 2026-10-15)

llama.cpp 23b0202a1, plugin with automatic sizing, clean-run gate, 8 x 48 tokens, mean of requests 4-8. MTP: `--spec-type draft-mtp --spec-draft-n-max 2`
(83 % of the drafts accepted).

| Limit | Run | tok/s | Section 2 (llama.cpp 7fe450e19, before the executor) |
|---|---|---|---|
| 28 GB | stock | 32.19 | 29.0-30.7 |
| 28 GB | plugin | 31.64 | 31.1-32.3 |
| 28 GB | stock + MTP | 28.31 (still warming up: the page cache was emptied by the plugin run before it) | 39.5 |
| 28 GB | plugin + MTP | 36.47 | 38.2 |
| 12 GB | stock + MTP | 8.38 | 8.8 |
| 12 GB | plugin + MTP | 13.64 | 24.0 |
| 12 GB | plugin | **7.82** | 23.5 (section 2b) |
| 12 GB | plugin, executor off (`MOE_CACHE_EXEC=0`) | 17.16 | |
| 12 GB | **plugin with the fix below** | **16.61** (identical to stock 4/4) | |

The executor (0.2.0 and later) was slower than llama.cpp for Q4_K experts: it stepped aside only when every expert of a step was already in RAM, so at
12 GB (9 % misses) it computed every step with the generic one-row kernel instead of llama.cpp's faster ones. Graphs whose expert weights are not compute-bound
(anything but IQ types and Q2_0) now go llama.cpp's own way from the start (`MOE_CACHE_EXEC=2` still forces the executor). Flash-Next (IQ2_XS / Q2_0) keeps the
executor: 12.95 vs 12.28 tok/s, unchanged. Releases 0.2.0-0.2.3.3 have the slow-down for K-quant models that do not fit in RAM.

Limits of this section (audit 2026-10-15): the 28 GB stock + MTP row was still warming up and is not a comparison. The fix is an isolated executor result
(7.82 -> 16.61 at 12 GB, same build otherwise); 16.61 is still below section 2b's 23.5 (another llama.cpp and plugin version, cache settings that differ),
and plugin + MTP was not re-measured with the fix, so the gap to the historic 24.0 is not explained yet. One run per row.

## 43. Quant yardstick for `--fast`: two 2-bit quants of the same model (T8; Flash-Next, 24 GB limit, 2026-10-15)

`llama-perplexity` with `--kl-divergence` against the same base file as section 31 (exact Flash-Next IQ2_XS, 20 chunks x 512 tokens of prose and code,
batch 8 so the executor handles every step). The second run is the exact Q2_0 file of the same model (ISTA-DASLab GSQ-RCO, shard 1 sha256-checked; shard 2 is
byte-identical to the IQ2_XS one). Plugin built from 847296a (byte check of every expert tensor against the file passed), 126,699 expert ops by the executor.
KL runs are deterministic in exact mode (a rerun of the base gave KL 0), so one run per row.

| Compared with exact IQ2_XS | Mean KLD | Same top token | PPL (base 1.970) |
|---|---|---|---|
| `--fast` at a 24 GB limit (section 31) | 0.0053 | 98.1 % | |
| `--fast` at a 6 GiB cache (section 31) | 0.0115 | 97.5 % | |
| **Q2_0 file of the same model, exact** | **0.2434 ± 0.0101** | **88.9 %** | 2.031 (+3.1 %) |

Reading: choosing between two 2-bit files of the same model moves the next-token distribution 21-46x more than `--fast` does. `--fast`'s change is small next to
what users already accept when they pick a quantisation. Not measured: the UD-Q2_K_XL file (not on disk), and quality on answers for Q2_0 (KL and PPL only).

## 44. Re-measured for the README (2026-10-15 afternoon; llama.cpp 23b0202a1, plugin from 02121c6, clean-run gate, 8 x 48 tokens, mean of requests 4-8)

**A measurement trap, fixed:** a memory-limited stock run can read the model from page cache that an earlier run left behind and that is charged to another
cgroup, so it does not count against the limit (stock Qwen3.6 "at 12 GB" ran at 31.8-32.2 tok/s that way). Every limited stock run below first drops the
model's pages from the page cache (`posix_fadvise(DONTNEED)`, work/zoo/evict_cache.py). The Qwen3.5-122B download ran in the background (~37 MB/s) during
the Qwen3.6 and Flash-Next runs.

### 44.1 Qwen3.6-35B-A3B Q4_K_M fits in RAM (24 GB limit, ctx 4096, all experts on the CPU side + llama.cpp GPU cache 5000 MiB, as section 29)

| Run | tok/s | Note |
|---|---|---|
| plugin, cold (no profile) | 33.41, 33.41, 33.24 | executor ran 0 ops (K-quant fix works); cache 17.5 GiB, 90.5 % hits, still filling |
| plugin, warm start from a profile | 36.70, 36.54 | 96.8 % hits |
| stock, page cache warm | 42.89 (section 29: 42.3) | the whole 21.7 GB file in the page cache |

When the model fits, stock is faster even against a warm plugin (the plugin's cache gets 17.5 GiB after its margins; stock's page cache holds everything and
stays warm across restarts). **The launcher now runs stock llama-server when the model fits** (file + 1.5 GB <= the memory the server may use).

### 44.2 Qwen3.6-35B-A3B at 12 GB (`--n-cpu-moe 34`, ctx 32768, as sections 2 / 42): the drop from 23.5 explained (audit E02)

| Run | tok/s | Note |
|---|---|---|
| plugin, `--cache-ram 2048` (launcher default until today) | 16.37, 17.50 | cache 5.5 -> 7.5 GiB, 91.4 % hits |
| plugin + MTP (`--spec-type draft-mtp --spec-draft-n-max 2`) | 18.69, 18.47 | MTP now +10 % (it lost with the old K-quant executor) |
| plugin, `--cache-ram 0` (no prompt-cache reserve) | **22.18, 23.03** | cache 7.5 -> 10.3 GiB, 95.5 % hits: back at section 2b's 23.5 |
| plugin, `--cache-ram 512` | 18.92, 19.54 | still -15 % against 0 |
| stock, page cache dropped first | 7.00, 6.81 | section 2 (older llama.cpp): 10.2 |

The prompt-cache reserve (2 GiB, added in 0.2.2 against an out-of-memory crash at 128k) took a large share of a 12 GB limit; even 512 MiB costs 15 %.
The launcher now passes `--cache-ram 0` below 16 GB (a single chat reuses its slot; only switching between conversations loses the host prompt cache)
and 2048 MiB above: at 12 GB that is **22.6 vs 6.9 tok/s stock (3.3x)**.

### 44.3 Qwen3.8-Flash-Next IQ2_XS at 24 GB (8 GB GPU, ctx 4096, launcher defaults: `--n-cpu-moe 46 -ub 2048 --cache-ram 2048`; stock with the same arguments)

| Run | tok/s |
|---|---|
| stock (page cache dropped first) | 5.31, 5.06 |
| moe-cache | 12.28, 12.66 (2.4x) |

## 45. Qwen3.5-122B-A10B at three bit widths (T27; unsloth UD quants, 24 GB limit, 8 GB GPU, launcher defaults, 2026-10-15)

llama.cpp 23b0202a1, plugin from 02121c6 (`build-light`), command lines from the launcher (`moe-cache-server --ram 24 --ctx 4096 [--fast] --dry-run`: 1 of 48 expert
layers on the GPU, `-ub 2048`), clean-run gate, stock runs with the same arguments after the model was dropped from the page cache. Identity: `tests/identity.sh
--limit 24 --tokens 24` (4 prompts, stock vs plugin). moe-cache and `--fast`: 8 x 48 tokens, mean of requests 4-8, paired. **Stock: 3 x 24 tokens, mean of
requests 2-3** (8 x 48 would take ~16 min per run at under 2 tok/s), one run: a steady-state mean like the others, but not the same requests.

| Quant (file) | Identity | Stock tok/s | moe-cache tok/s | `--fast` tok/s |
|---|---|---|---|---|
| UD-IQ2_XXS (36.6 GB) | 4/4 | 1.89 | **7.70, 7.73 (4.1x)** | 8.71, 9.23 (+16 %) |
| UD-IQ3_XXS (44.7 GB) | 4/4 | 1.10 | **5.64, 5.78 (5.2x)** | 6.86, 6.70 (+19 %) |
| UD-IQ4_XS (60.2 GB, 3 shards) | 4/4 | 0.91 | **3.82, 3.50 (4.0x)** | 4.23, 4.70 (+22 %) |

Each step up in width means more expert bytes per token (8 of 256 experts in each of 48 layers: about 1.0 / 1.25 / 1.7 GB per token) and a smaller share of the
model in 24 GB, so speed falls with file size; the gap to stock stays 4-5x because stock reads its misses through small page faults.

## 46. Prompt processing, stock vs moe-cache (2026-10-15 evening; launcher command lines, clean-run gate, page cache dropped before stock runs)

`work/zoo/prefill_req.py` (one short request, then prompts of ~2.4k / 1.9k / 5.0k / 5.4k tokens, cache_prompt off, then 4 decode requests); stock runs used
`prefill_req_short.py` (the short request and the first 2.4k prompt only: stock decode is slow). One run each.

| Model, limit | Stock prefill (2.4k prompt) | moe-cache prefill (2.4k / 1.9k / 5.0k / 5.4k) | Decode after the prompts, stock / moe-cache |
|---|---|---|---|
| Qwen3.6-35B Q4_K_M, 12 GB (ctx 8192) | 132.8 tok/s | 136.5 / **202.8** / 186.1 / **205.5** tok/s | 3.31 / 18.6-25.1 tok/s |
| Qwen3.8-Flash-Next IQ2_XS, 24 GB | 65.9 tok/s (ctx 4096) | 90.5-91.9 / 99.2 / 90.2 / 98.2 tok/s (section 38, v0.2.3.3, ctx 8192) | 2.13 / ~12 tok/s |

Reading: for long prompts llama.cpp already sends the expert work to the GPU, so on a model whose experts fit in RAM most of the time (Qwen3.6 at 12 GB)
stock prompt speed is close to moe-cache's on the first prompt; moe-cache gains on later prompts (its cache is warm) and above all in decode. On Flash-Next
(68 GB at 24 GB) moe-cache's prompt speed is 1.4x stock. The moe-cache Flash-Next run of this session failed (the 5-6k prompts exceed ctx 4096 in that run),
so its row uses section 38's numbers, same launcher settings at ctx 8192.

## 47. Long contexts, moe-cache only (owner's choice: no stock runs; 2026-10-15 evening, 24 GB limit, launcher command lines, clean-run gate)

| Model, context | Placement chosen by the launcher | Decode tok/s (8 x 48, mean of 4-8, paired) | Prefill |
|---|---|---|---|
| Flash-Next IQ2_XS, ctx 32768 | 0 of 48 expert layers on the GPU (the 32k buffers leave no room), `-ub 2048` | 12.46, 11.35 (4k context: 12.28, 12.66) | 29,564-token prompt at **38.4 tok/s** (12.8 min); decode right after it 8.1-11.2 |
| GLM-5.3-Flash UD-IQ2_XXS, ctx 16384 | 0 of 46 expert layers on the GPU, `-ub 512` | 1.89, 1.94 (4k context: ~1.75) | 10,453-token prompt at **5.7 tok/s** (30.6 min) |

Reading: decode does not get slower with the longer context on either model. Prefill does: Flash-Next 38 tok/s on a 30k prompt against ~95 on 2-5k prompts
(attention over the whole prompt, and no expert layers fit on the GPU at 32k); GLM re-reads most of its ~95 GB of experts from the SSD for every 512-token
chunk, so its long prompts run at under 6 tok/s. Ideas I3 (keep a share of the cache across chunks) and B13 (bigger chunks on a bigger card) target this.

## 48. Long prompts keep part of the cache (idea I3, scan-resistant loading; Flash-Next IQ2_XS, 24 GB, launcher defaults, ctx 8192, 2026-10-15 ~21:00)

`MOE_CACHE_SCAN_TOKENS=512`: experts loaded by prompt batches of 512+ tokens go to the evict-first end of the LRU list and hits do not move experts to the
front, so a prompt that needs more experts than the cache holds recycles its own slots instead of evicting everything cached before it. With plain LRU a
cyclic pass over 64 GB of experts through a ~20 GB cache keeps nothing between micro-batches. Paired runs, `pf_run.sh` (prompts of 2.4k / 1.9k / 5.0k /
5.4k tokens, then 4 decode requests), same build (`build-i3`), order off, on, on, off:

| Run | Prefill tok/s | Cache hit rate | Read from SSD | Decode after |
|---|---|---|---|---|
| off a | 55.6 / 78.0 / 69.9 / 75.2 | 43.7 % | 241.8 GB | 7.9-10.2 |
| on a | 72.8 / 98.1 / 81.7 / 106.1 | 78.0 % | 102.8 GB | 8.7-12.5 |
| on b | 85.9 / 103.4 / 88.7 / 102.0 | 78.1 % | 102.5 GB | 8.1-10.6 |
| off b | 74.8 / 73.8 / 74.5 / 79.7 | 43.8 % | 241.8 GB | 8.8-11.2 |

Mean prefill: off 72.7, **on 92.4 tok/s (+27 %)**; SSD reads -58 %. Output cannot change (only which expert is evicted differs; CI runs a long-prompt
identity step with it on). Now the default. The off runs are lower than section 38's 90-103 (another cache state at the start of this build's runs);
the comparison is paired. X8 (the same idea, October) lost 25 % before prompt reads were merged (T38).

GLM-5.3-Flash UD-IQ2_XXS (24 GB, ctx 8192, `-ub 512`, one 2.2k-token prompt, one run each): off 6.0 tok/s, 25.8 % hits, 453 GB read; **on 6.5 tok/s, 37.5 % hits,
384 GB read**. Smaller gain than Flash-Next (the cache holds only ~20 of GLM's ~95 GB of experts, so less survives between chunks); +8 % is inside the
run-to-run noise, the 15 % fewer reads are not. Not slower, so the default stays on for GLM too.

## 49. First run on real Windows, and the iGPU experiments E1/E2 (MSI Claw 8 AI+, 2026-10-15; from the Claw session's report, handover/claw_results.md)

Machine: Core Ultra 7 258V (4P + 4LPE), Arc 140V iGPU, 32 GB, Windows 11. llama.cpp official release b11669 (Vulkan, ggml 0.27.0). Test model OLMoE-1B-7B Q4_0;
for E1/E2 requantized with `--pure` to each type (speed only).

**T5 (identity test): not reached.** The plugin DLL did not load (`load_backend: failed to load ...dll:` with an empty reason): it imported
`libwinpthread-1.dll` (a MinGW runtime not present on Windows) and bound `ggml_backend_dev_by_type / dev_count / dev_get / register` to `ggml-base.dll`
although Windows exports them from `ggml.dll`. Fixed in `build.sh --windows` (0b7bbaa: two import libraries, static winpthread); rerun pending.
Stock decode on this machine (experts on the CPU, attention on the iGPU): 45-46 tok/s.

**Launcher on Windows:** check, dry run, real start and a chat request work (stock mode: the model fits). Found and fixed the same day: a killed launcher left
llama-server running (job object now), a plugin that did not load was reported as loaded (warning now), no memory detection without /proc (GlobalMemoryStatusEx
now), the tuner's "no VRAM information" answer overrode the planner, `check` gave no reason for a load failure. Open: no core pinning on Windows.

**E1 (iGPU kernel speed and handoff): no stop rule hit.** Whole-token decode, tok/s:

| Type | CPU (best of 4 / 8 threads) | iGPU (Vulkan) | iGPU / CPU |
|---|---|---|---|
| IQ2_S (true iq2_s) | 52.4 | 43.6 | 0.83x (the Vulkan iq2_s kernel is 1.9x slower than iq2_xs for the same shape) |
| IQ2_XS | 43.0 | 71.5 | 1.66x |
| IQ2_XXS | 52.6 | 87.1 | 1.66x |
| IQ3_XXS | 43.1 | 91.1 | 2.11x |
| Q2_0 | 55.7 | 156.9 | 2.82x |

One CPU/iGPU handoff costs about 0.2 ms (at most 0.46-0.64 ms), far from the feared 3 ms. But llama.cpp's own per-layer split (experts on the CPU, the rest
on the iGPU: 34 handoffs per token) is slower than the CPU alone for every type: a scheduler must not do a round trip per layer (SCHEDULER_DESIGN.md).
Side finding: for a model that fits in RAM, experts on the iGPU (`-ngl 99` without `--n-cpu-moe`) are 1.7-2.8x faster than on the CPU on this machine.

**E2 (both busy at once): no stop rule hit.** The busy iGPU costs the CPU 12-28 %, the busy CPU costs the iGPU 1-30 %; both together give 1.08-1.46x the
faster one alone (two whole-model streams; a per-layer split of one token will gain less, which E3/E4 measure).

## 50. Windows round 2 (MSI Claw) and the first AMD machine (mini PC, T7) (2026-10-16; from handover/claw_results.md "Round 2" and handover/moe-cache-t7/)

**MSI Claw, round 2 (rebuilt kit, llama.cpp b11669 = ggml 0.27.0, OLMoE Q4_0).** Both DLL fixes of section 49 work: no `libwinpthread-1.dll` import,
the four registry functions bind to `ggml.dll`, `LoadLibrary` succeeds, and ggml 0.27.0 accepts the API 3 plugin (the API 2 DLL is refused by ggml with
"incompatible API version", as it should be). **T5 still failed, on a new, Windows-only bug:** llama-server died in the first graph in every cache-mode
variant (with and without Vulkan, executor, direct reads, repacking, mmap, or a cache that holds the whole model) and never in pass-through mode.
The Claw session caught it with a small ctypes debugger: a write by an ggml-cpu OpenMP worker into a page that was reserved but not committed, 4.13 MiB
into a 52.12 MiB range, which is exactly the size of the plugin's compute buffer (`sched_reserve: MOE_CACHE compute buffer size = 52.12 MiB`).
Cause: in cache mode every buffer of the plugin's buffer type is reserved address space only. On Linux an untouched page of an `mmap(MAP_NORESERVE)`
range appears zero-filled on first touch, so the compute buffer just worked; on Windows a `MEM_RESERVE` page faults until `MEM_COMMIT`, and ggml's kernels
write node outputs through `tensor->data` without any buffer call. Linux tests cannot see it (guard mode only protects evicted expert pages).
Fix (cffa6f9): `buf_init_tensor` commits every non-expert tensor's range, once per 64 KiB chunk (a per-buffer flag array, so the per-token graph
allocation costs a few lookups, not ~900 `VirtualAlloc` calls).

**Windows CI (idea I8, new `.github/workflows/windows.yml`) found the second half the same night.** A GitHub Windows runner with the official b11669
CPU release still crashed at the first request after cffa6f9. From the runner's crash dump (minidump read with a small script; return addresses mapped
with addr2line on the CI's debug-info DLL): the plugin's reader thread (`do_reads` -> CPU repack `set_tensor` -> memcpy of one `block_q4_Kx8`) wrote
into a reserved page. `read_job` commits each expert plus one extra page; for the last expert of a buffer that range runs past the end of the
reservation, and `VirtualAlloc(MEM_COMMIT)` then fails as a whole (error 487, `ERROR_INVALID_ADDRESS`) and commits nothing. Fix (87dd800):
`mem_commit` falls back to page-by-page commits up to the end of the reservation. Next run:

| Windows CI step (Server 2025 runner, b11669 CPU, Granite 1B, 0.2 GiB cache, 24 tokens) | Result |
|---|---|
| platform layer natively | all checks pass, including the new commit-past-the-end case (also under Wine, which reports error 487 too) |
| launcher `check` / dry run / core detection | plugin loads; `--gpu-cache` dropped on path loading; 2 cores, one efficiency class: no pinning; API 3 read from `ggml-base.dll` |
| identity, cache forced to evict | **4/4 PASS** (first passing identity test on Windows with the cache on): 35,811 evictions, 11.1 GB re-read; the one-time "commit failed (error 487)" note appears, confirming the cause |
| identity, executor forced on and byte-checked (`--no-repack` on both servers) | first try 1/4: a third Windows bug, the plugin did not see `--no-repack` (it read `/proc/self/cmdline` only) and repacked while stock did not. Fixed (`GetCommandLineA`); then **4/4 PASS, 6,765 executor ops checked against the stock kernels, 0 mismatches** |

To read the counters on Windows (where the test kills the server, so the summary at exit never appears), `MOE_CACHE_STATS_SIGNAL=1` now also works
there: setting the named event `Local\moe-cache-stats-<pid>` prints one `moe-cache-stats {json}` line (Linux: SIGUSR1), now with executor counters.

Plugin decode in that step is ~4 tok/s against stock's ~70: a 0.2 GiB cache for a 0.8 GB model re-reads experts all the time on a CI disk (a stress
setting for correctness, not a speed test).

Launcher rechecks on the Claw: the launcher's death now ends llama-server (job object), a plugin that does not load is reported, `check` and the dry run
read Windows memory, the dry run is PowerShell. Fixed after this round: the launcher now names the exit code when llama-server crashes
(`0xC0000005 (access violation)` and the last plugin line instead of silence), planner errors give the real reason (not "pip install gguf"), the
API pre-warning works on Windows (version string in `ggml-base.dll`), a missing `MOE_CACHE_LIB` is named as such, `check` prints GB throughout and says
the cache first runs at the first request. Core pinning on Windows (Claw finding 5) is implemented (f7721a6, efficiency class from
`GetLogicalProcessorInformationEx` -> `-C` mask) and not yet run on Windows.

**Mini PC (T7): Ryzen 7 5800H (Zen 3, AVX2), Vega 8 iGPU (3 GiB carve-out) via Vulkan (RADV), 32 GB DDR4, Ubuntu 26.04, llama.cpp 23b0202a1
built by `install.sh --vulkan`; plugin from 022c674 plus the fixes below.** The first AMD CPU and the first AMD GPU.

| Test | Result |
|---|---|
| `moe-cache-check` (Granite 1B) | 0 failed, 2 warnings (swap on; model fits) |
| friend test, stock vs plugin | 4/4 |
| identity, Granite, forced eviction (0.2 GiB): CPU only / Vega 8 / executor forced + byte-checked | 4/4 / 4/4 / 4/4, 0 mismatches in 6,708 expert ops (q4_K, q6_K) |
| identity, Qwen3.6-35B-A3B Q4_K_M at a 12 GB limit, Vega 8 | 4/4 (cache grew to 10.5 GiB; no eviction in that short test) |
| launcher start, chat, stop (Granite) | fine; the AMD VRAM path (audit A13) reads the 3 GiB carve-out and ignores the 14.9 GB GTT |
| decode, Qwen3.6 at 12 GB, paired (stock / mc / mc / stock), cool machine, page cache dropped | stock 8.12 / 9.08, moe-cache 10.76 / 10.43 tok/s: **1.23x** |

No expert layer fits on the 3 GiB iGPU next to the other weights and buffers, so this is the no-GPU case. moe-cache's 10.6 tok/s is close to the laptop's
no-GPU 9.9 (section 44), but stock is much faster here than on the laptop (8.6 vs 3.1 tok/s at the same limit); why (SSD, readahead, the DDR4 / Zen 3
caches) was not investigated. The two stock runs differ by 12 %, so read +23 % as roughly +15 to +30 %.

Bugs found there (all fixed in cffa6f9):

| # | Bug | Effect | Fix |
|---|---|---|---|
| B1 | `MOE_CACHE_GPU_CACHE=1` (`--gpu-cache`) with the plugin loaded through `GGML_BACKEND_PATH` | **wrong tokens** (0/4 identical, one token repeated): the hook that loads experts before llama.cpp's GPU cache uploads them is symbol interposition, which only works with LD_PRELOAD; nothing warned | the plugin refuses the mode unless preloaded; the launcher drops `--gpu-cache` on path loading; the friend test skips that case; two CI checks. Affects every install.sh / Vulkan / official / Windows build |
| B2 | the long-prompt GPU path accepted only `GPU` devices; ggml-vulkan reports APUs as `IGPU` | plugin prefill 363 vs stock 632-703 tok/s | discrete GPU first, else the iGPU: 569-597 tok/s, identity 2/2 |
| F2 | friend-test kernel check opened `libggml-cpu.so.0`; `GGML_CPU_ALL_VARIANTS` builds have only `libggml-cpu-<variant>.so` | section 2 always failed on install.sh builds | picks the variant with the highest backend score, as ggml does (checked on the laptop: `alderlake`, all bit-exact) |
| F3 | `install.sh --vulkan` on a headless server without the `render` group | Vulkan silently saw only llvmpipe | warning after the build, with the `usermod -aG render,video` hint |
| F5 | `wait_cool.sh` without lm-sensors on AMD (no thermal zones, k10temp in hwmon only) | read 99 C forever | reads k10temp / zenpower / coretemp from hwmon |

Machine notes (not moe-cache bugs): the `render` group was missing for Vulkan; a KDE bluedevil / obexd D-Bus loop kept the idle load near 2 (fixed for that
session by unloading bluedevil); a NoMachine desktop session alone kept the load at 2.6 and 75 C, above the clean-run gate.

## 51. Coding agents on moe-cache: OpenCode and Hermes Agent, agent mode and warm sessions (2026-10-26, laptop)

Setup: RTX 4060 8 GB, Qwen3.6-35B-A3B Q4_K_M (21.7 GB) under a hard 16 GB cgroup limit (no swap), launcher `--agent --ram 14` (plugin, 9 GiB cache,
38 layers of experts on the CPU, `-ub 1024`, context 65536, 4 slots sharing it), llama.cpp 23b0202a1. OpenCode 1.19.4 and Hermes Agent v0.22.5 in
one-shot mode, task "fix the bug in ./calc.py". Not a timed benchmark (the agents run in parallel with the server): the numbers are llama-server's
own per-request timings.

| Session | First request: new prompt tokens / time | Later turns | Whole session | Result |
|---|---|---|---|---|
| OpenCode 1, cold server | 7,438 / 45.6 s (163 tok/s) | 79 / 3.4 s, 20 / 0.8 s (prefix reused) | 78 s | file fixed |
| OpenCode 2, same server | **29 / 0.7 s** (preamble reused from the checkpoint at the last user message) | 80 / 3.4 s, 20 / 0.8 s | 30 s | file fixed |
| OpenCode A, cold (warm-session cycle) | 7,438 / 44.6 s | - | - | model asked for /calc.py (root), auto-rejected; preamble still processed |
| OpenCode B, after a server restart with the saved conversation restored (0.2 s) | **1,028 / 8.5 s** (rolled back to the checkpoint 4 + n_ubatch tokens before A's prompt end) | 90 / 3.5 s | 39 s | file fixed |
| Hermes, file tools only | 3,687 / 23.0 s | 76 / 2.9 s, 264 / 4.7 s | 55 s | file fixed |

Findings:
- Tool calls work through the plugin with the models' own templates (`--jinja`, the default at this commit); one raw tool call: correct name and
  arguments, 17 tok/s.
- The first turn is what agents wait for; llama-server already reuses every later turn and, for a new session on the same server, the preamble.
  Warm sessions (idea I29) carry that across restarts: 5.3x faster first turn. One saved file per model and context: the last agent used wins
  (Hermes's 4,113-token conversation replaced OpenCode's 7,497).
- Hybrid models keep up to 8 checkpoints per slot here (about 60 MiB each on Qwen3.6); the plugin's measured-memory tuner absorbs them, with
  512 MiB kept free. Hard limit never reached; no out-of-memory events.
- OpenCode's parallel title request runs in another slot without disturbing the main one, but both decode slower while they overlap (8-11 tok/s
  instead of 17).
- `opencode run` waits on an open stdin (a script must give it `< /dev/null`).

## 52. Long-prompt block fell back to the CPU at context 8192 (tuner fix), Flash-Next prefill at 8192 (PF8K), kill checks I2 / I28 / B1 (2026-10-26, laptop)

**Bug found and fixed:** with the launcher's defaults at `--ctx 8192` (no GPU cache: `-ub 2048`, 2 expert layers on the GPU) the long-prompt GPU block
found 423 MiB of VRAM free where it needed 455 MiB (`moe-cache: a long-prompt block needs ...: it stays on the CPU`), so most blocks ran on the CPU and
Flash-Next prefill fell to about 20 tok/s. Cause: the tuner kept 1.6 x 704 + 160 MiB free for the block, but the plugin also requires 256 MiB of
headroom before it allocates one (gpu_segment), leaving 24 MiB of slack that CUDA's own buffers at ctx 8192 used up. Fix (planner/gguf_plan.py): the
block reserve includes those 256 MiB. Effect: one expert layer fewer on the GPU (`--n-cpu-moe 47` instead of 46 on Flash-Next at 4k/8k context).

| Flash-Next IQ2_XS, 24 GB limit, ctx 8192, launcher defaults after the fix, clean-run gate | 2.4k prompt | 1.9k prompt | 5.0k prompt | 5.4k prompt | decode (4 x 48) |
|---|---|---|---|---|---|
| run a | 100.5 | 131.4 | 123.9 | 134.9 | 11.8, 10.0, 11.7, 12.0 |
| run b | 93.2 | 115.8 | 105.7 | 112.6 | 11.1, 9.2, 11.3, 11.9 |

No block stayed on the CPU in either run. Runs a and b differ by about 15 % on average (8 / 13 / 17 / 20 % per prompt: partly outside the usual +-10 % band); decode at ctx 8192 is 9-12 tok/s (12.4-13 at
ctx 4096, section 44).

**I2 (overlap upload with compute in the long-prompt block): killed on the laptop.** Timers (`MOE_CACHE_SEG_TIMES=1`), 128 blocks at -ub 2048: upload
70 %, compute 21 %, read back 9 % of block time. Under the 25 % line: overlap could hide at most ~21 % of block time. Re-check on a PCIe 5 card.

**I28 (`--gpu-cache` with path loading): cheap exact fallback killed.** Flash-Next, 22 GB, `--gpu-cache auto` (2047 MiB), 8 x 48 tokens, counter
`MOE_CACHE_GPU_ENSURE_STATS=1`: 13,140 of 289,053 experts that llama.cpp's GPU cache uploaded (4.55 %) were not in RAM, equal to the plugin's miss rate
(95.5 % hits; every pick goes through the GPU cache in this mode). A "resident-or-skip" fallback would skip 4.6 % of picks (approximate). Decode 13.30
tok/s at 22 GB. Exact `--gpu-cache` for Windows / official builds needs an upstream hook.

**B1 (plugin-owned GPU tier) simulation gate at 12-14 GiB: passed against whole layers, marginal against an LRU cache.** work/hottier/hotsim12.py with
12/14 GiB, Qwen3.6 traces (22 sessions), the simulator's cost model for both sides: at 12 GiB a warm LRU tier (admit >= 2, <= 4 copies per token) 2.64x
all-CPU, whole layers 1.73x (+53 %; about +19 % after the model's 28 % under-rating of whole layers seen at 8 GiB), plain LRU (admit >= 3) 2.31x
(+14 %). llama.cpp's own GPU cache is close to plain LRU, so its measured curve on a 16 GB card (BIGGER_MACHINES / R1) decides B1.

**I34 (one warm file per slot, the two longest restored into their own slots), Qwen3.6, 16 GB limit:** OpenCode then Hermes on one server (both files
fixed), saves of slot 1 (4,131 tokens, 546 MB, 0.2 s) and slot 2 (7,729 tokens, 685 MB, 1.0 s), restart, both restored (0.3 s, 0.2 s), each agent's
next session picked its own slot by prompt similarity: first request 1,028 tokens in 9.0 s (OpenCode, cold 7,438 in ~45 s) and 1,028 in 6.2 s (Hermes,
cold 3,687 in 23 s); whole sessions 79 -> 46 s and 54 -> 35 s; all four files fixed.


**Flash-Next as an agent model (64 GB IQ2_XS, 23 GB hard limit, `--agent --ram 22`: ctx 65536, `-ub 512`, all 48 expert layers on the CPU side),
OpenCode, warm-session cycle:** session A (cold) first turn 7,461 tokens in 186 s (40 tok/s), whole session 242 s, file fixed; saved 7,752 tokens with 7
checkpoints (1,183 MB, 1.5 s); after a restart and restore (0.7 s) session B's first turn processed 516 tokens in 47 s (in parallel with OpenCode's
610-token title request; both pay whole-layer uploads), whole session **112 s instead of 242 s**, file fixed. An 89-token follow-up took 6.7 s.
At 64k context the 8 GB card only fits `-ub 512` next to the KV cache, so prefill is 40 tok/s instead of 93-135 at ctx 8192.

**I4 (the long-prompt block uploads only the experts its tokens use), now the default:** before, every batch of 32+ tokens uploaded each layer's whole
expert tensors to the GPU, as stock llama.cpp does. Now the used experts are packed into a smaller tensor and the ids renumbered (src/mc_backend.inl,
`MOE_CACHE_GPU_PACK=0` turns it off). Identity on Qwen3.6 (plugin vs stock, 16 GB limit): 4/4 identical at ~74-token prompts (146-174 of 256 experts
uploaded) and ~147-token prompts (157-198 of 256). Speed, Qwen3.6 `--ram 14`, ctx 8192, prompts without prompt cache, clean-run gate, off/on/on/off:

| Prompt | packing off | packing on | |
|---|---|---|---|
| ~70 tokens (3 prompts) | 1,621 / 1,631 / 1,634 ms; 1,611 / 1,613 / 1,625 ms | 896 / 896 / 908 ms; 903 / 906 / 923 ms | **1.8x** |
| ~170 tokens (3 prompts) | 1,729 / 1,744 / 1,762 ms; 1,720 / 1,722 / 1,753 ms | 1,112 / 1,098 / 1,141 ms; 1,133 / 1,108 / 1,144 ms | **1.55x** |

This is the agent follow-up case (tool results of 30-300 tokens). Stock llama.cpp uploads whole layers here, so it should gain the same way, but stock was not timed in this test. Batches that use
90 % or more of the experts upload the whole tensor as before.

**I15 (stronger `--fast` levels), quality: kill check passed.** `kl_run.sh` (llama-perplexity, 20 x 512 tokens prose + code, Flash-Next, fixed 6 GiB cache,
16 GB limit, against the exact base.kld), `MOE_CACHE_ROUTE=skip` with `MOE_CACHE_ROUTE_KEEP` = picks always honoured (Flash-Next uses 10 per token):

| Setting | Mean KL | Same top token | Low-ranked picks given up |
|---|---|---|---|
| keep 8 (today's `--fast`, section 43) | 0.0115 | 97.5 % | 304,442 of 972,990 |
| keep 7 | 0.0166 +- 0.0009 | 97.0 % | 436,008 of 1,459,485 |
| keep 6 | 0.0224 +- 0.0011 | 96.4 % | 558,383 of 1,945,980 |
| yardstick: Flash-Next Q2_0 vs IQ2_XS | 0.243 | 88.9 % | - |

Keep 6 still sits 11x below the quant-to-quant difference while giving up 83 % more SSD reads than keep 8. Speed: below.

**I15 speed: no gain at 24 GB.** Flash-Next, 24 GB limit, launcher `--fast` (auto cache ~15 GiB), 8 x 48 tokens, clean-run gate, order 8, 7, 6, 6, 7, 8:
keep 8 14.57 / 16.08 tok/s, keep 7 15.10 / 13.72, keep 6 15.13 / 15.25 (all inside the +-10 % band). With 97 % hits only 4-10 k picks per run were
given up (42-85 k looked at), so there is little left to skip; the levels can only pay where the cache is much smaller than the model (12 GB and
below): not made an option.

## 53. Laptop ideas under a day (2026-10-26 afternoon): I15 at 12 GB, I18 starter profiles, I11 split, I35 by calculation

**I15 at 12 GB: the stronger `--fast` levels pay where the cache is small.** Flash-Next, 12 GB limit, launcher `--fast`, 8 x 48 tokens, clean-run gate,
order 8, 7, 6, 6, 7, 8 (runs drifted downward over time; keep 6 beats keep 8 in both halves):

| `MOE_CACHE_ROUTE_KEEP` | run 1 | run 2 | mean | picks given up | hit rate |
|---|---|---|---|---|---|
| 8 (today's `--fast`) | 10.30 | 8.79 | 9.55 | 9.5 k of 42.3 k | 90.7 % |
| 7 | 11.59 | 10.00 | 10.80 | 14.5 k of 63.5 k | 92.2 % |
| 6 | 13.09 | 14.28 | **13.69 (+43 %)** | 18.9 k of 84.6 k | 94 % |

With KL 0.0224 for keep 6 (section 52; yardstick 0.243) a stronger opt-in level for small memory looks worth it; not shipped: needs an answer
comparison and the owner's OK (a new approximate option).

**I18 starter usage profiles: first request 2.1x faster.** `profiles/<arch>-<layers>x<experts>.usage` (written by tools/make_starter_profile.py
without the model fingerprint, so every quant of a family can use it): qwen4exp-48x512 (Flash-Next), glm5-next-46x288 (GLM-5.3-Flash),
qwen35moe-48x256 (Qwen3.5-122B), qwen35moe-41x256 (Qwen3.6). The launcher copies the matching one to the user's profile path when there is none
(`--no-starter` to skip). Flash-Next, 24 GB, fresh profile name, launcher real start, order cold, starter, starter, cold, clean-run gate:

| Run | first request | mean of requests 4-8 |
|---|---|---|
| cold (no profile) | 5.96, 6.18 | 10.78, 10.98 |
| starter profile | **13.70, 11.81** | 14.35, 11.98 |

The starter came from the developer's own sessions; the test prompts (speed_req.py) are different text.

**I11 split of state_t: done, compiled code byte-identical.** src/mc_state.inl now holds the settings and fields; the methods continue in
mc_state_io.inl, mc_state_profile.inl and mc_state_lru.inl, included inside the class body. `.text` and `.rodata` of the built plugin compare
equal with the single-file build (225,330 bytes of code). The Windows cross-build compiles.

**I35 (real KV size in the tuner): killed by calculation.** From the GGUF shapes: Flash-Next has attention in 12 of 48 layers, K and V of 512 each
(24.6 KB per token, 1.50 GiB at 64k) plus indexer keys (~0.19 GiB) plus the recurrent state of 36 layers (~117 MB per slot, 0.44 GiB for 4 slots):
2.13 GiB against the tuner's 2.44 GiB. The ~0.3 GiB gap is far from the ~0.7 GiB that `-ub 1024` at 64k needs. (Qwen3.6: ~0.8 GiB gap, worth about
one expert layer, not the micro-batch step.)

**I6 (`--fast` also uses a q8_0 KV cache): done.** Quality of the q8_0 KV cache alone (kl_run.sh, Flash-Next, 6 GiB, exact routing): **KL 0.0045,
same top token 98.2 %** (today's `--fast` routing alone: 0.0115; yardstick 0.243). It halves the KV memory, which at agent contexts is what a bigger
micro-batch needs: in agent mode (ctx 65536) the tuner now picks `-ub 1024` instead of 512 for Flash-Next. Fit check, Flash-Next, `--agent`, 23 GB
limit, one run each (prefill_req.py): exact 49.6 / 48.0 / 52.2 / 52.3 tok/s on 2.4k / 1.9k / 5.0k / 5.4k prompts; `--fast` 69.5 / 80.7 / 81.5 / 78.8
(about +55 %); no block fell back to the CPU, no out-of-memory; decode 8.2-11.6 vs 8.2-10.8. A `-ctk` / `-ctv` given after `--` wins.

**`--fast=2` shipped (opt-in, I15).** `MOE_CACHE_ROUTE_DROP` (default 2) sets how many of the lowest-ranked picks may be skipped when not in RAM;
`--fast=2` sets 4 with at least half the picks always kept (Flash-Next: keep 6 of 10; 8-pick models: keep 4). Answer comparison, Flash-Next, 12 GB,
6 questions at temperature 0 (arithmetic, code, explanation, time arithmetic, TCP/UDP, translation): `--fast` matched exact word for word on 5 of 6
(TCP/UDP reworded, same points); `--fast=2` matched word for word on 2, and on the other 4 gave the same results and points in different wording
(17 x 23 = 391 with one step fewer, the same Fibonacci code with a shorter error message, the sky answer naming Rayleigh scattering). All 18 answers
correct. Picks given up in that session: 20,439 (`--fast`) and 43,367 (`--fast=2`).

## 54. Windows 11 desktop: Ryzen 9 7950X + RX 9070 XT (2026-10-26; from the PC session, windows-kit/pc_results.md)

First real run on the owner's new desktop, driven over SSH. **Every speed here is PRELIMINARY**: the owner's apps were open (Dia, Claude, ChatGPT,
Discord: 18.6 GB of working sets, 14.2 GB of RAM free at the start), and Windows' file cache could not be cleared (needs admin rights, RAMMap).
Clean reruns: Flash-Next is done in section 55 (file cache flushed without admin rights); 122B and GLM follow. Identity and bit-exact results do not depend on any of this and are final.

| Item | Value |
|---|---|
| Machine | Windows 11, Ryzen 9 7950X (AVX-512, built-in Radeon iGPU), RX 9070 XT 16 GB (driver 32.0.23034.4), 32 GB DDR5 (31.1 GB visible) |
| llama.cpp | b11669 (ggml 0.27.0, commit 620d2225c), official Vulkan zip |
| Plugin | DLL cross-built from ff73ec1 (API 3); source clone at e36344b |
| Models | Flash-Next IQ2_XS 64 GB, GLM-5.3-Flash 95 GB, Qwen3.5-122B UD-IQ2_XXS 36.6 GB, Qwen3.6 Q4_K_M 21.7 GB, Granite 1B (all 8 downloads verified, ~208 GB in 61 min) |
| Vulkan devices | Vulkan0 = built-in Radeon (reports 16,204 MiB shared), Vulkan1 = RX 9070 XT; llama.cpp uses only Vulkan1 |
| HIP zip (rocm-10.0) | lists no device; has `amdhip64_7.dll` and `ggml-hip.dll` but no hipBLAS / rocBLAS: probably needs AMD's HIP SDK (ideas R13 / I31, later) |
| Stale NVIDIA driver | `nvidia-smi.exe` and an RTX 4060 Laptop entry in the registry (Windows install carried over from the laptop); harmless, but it fooled the launcher (below) |

### WP1: identity (final)

| Test | Result |
|---|---|
| Granite 1B, cache 0.5 GiB (evicting) | **PASS 4/4**; 10,105 evictions, 3.6 GB re-read |
| Granite, `MOE_CACHE_EXEC=2` + `EXEC_CHECK=1` | **PASS 4/4**; executor ops 9,069, checked 9,069, **0 mismatches** |
| Qwen3.6 Q4_K_M, cache 4 GiB | **PASS 4/4**; 35,265 evictions, 25.7 GB re-read (executor steps aside on K-quants: expected) |
| Qwen3.5-122B UD-IQ2_XXS (bigger than RAM), cache 4 GiB | **PASS 4/4**; 75,951 evictions, 67.9 GB re-read, executor ops 19,134 |
| `check --run` on Qwen3.6 | 0 failed, 2 warnings (Windows experimental, ggml version); GPU = RX 9070 XT; "cache runs: yes" |
| Launcher real start + chat, Flash-Next | ready after 32 s, correct answer (below) |

Known note in the logs: "commit ... failed (Windows error 487), committing page by page" is the handled case from earlier Windows runs.
On the 122B run `MOE_CACHE_AVX512=1` did nothing: the plugin's own CPU kernels (Q2_0, IQ3_XXS gather, AVX-512) are compiled for Linux only
(`__linux__` guards; AVX-512 and Q2_0 work by symbol interposition = LD_PRELOAD, which Windows does not have). Speeds in the identity runs are not
measurements (forced small cache, downloads running).

### Bug found: the planner saw no GPU on Windows with an AMD card (fixed in e36344b)

The planner read free VRAM from nvidia-smi or Linux sysfs only. On Windows with an AMD or Intel card it saw no GPU and kept every expert layer on
the CPU; `check` named the first listed device (the iGPU). Fix: when nvidia-smi reports no GPU, the launcher reads `llama-server --list-devices` and
picks the adapter with the most dedicated VRAM in the registry (9070 XT 17.1 GB vs iGPU 0.5 GB). Dry-run, Qwen3.6:

| | before | after |
|---|---|---|
| GPU | none | RX 9070 XT (Vulkan1), 15.07 GiB free |
| expert layers on the CPU | 40 / 40 | 18 / 40 |
| `-ub` | 512 | 2048 |

### Crash: AMD's Windows Vulkan driver dies when llama-server saves a context checkpoint of Flash-Next (launcher workaround, 7a1b86f)

First launcher start on Flash-Next: ready after 36 s, then llama-server died at the end of the first prompt (access violation 0xc0000005 in
`amdvlk64.dll`). Narrowed down with **stock** llama-server (no plugin), one request each, Flash-Next IQ2_XS on the 9070 XT:

| Stock llama-server | Result |
|---|---|
| launcher's arguments (`--n-cpu-moe 36 -ub 2048 -c 8192 --cache-ram 2048`) | crash |
| same with `-ub 512` | crash |
| all experts on the CPU (`--n-cpu-moe 999`) | crash |
| `-fa off` | crash |
| `GGML_VK_DISABLE_COOPMAT=1` / `..._COOPMAT2=1` | crash |
| `-ngl 0` (CPU only) | works |
| `llama-bench` pp64 / tg8 (no server, no checkpoints) | works (6.9 / 2.95 tok/s) |
| `-np 1 --ctx-checkpoints 0 -c 2048` | works |
| **`--ctx-checkpoints 0` alone** (`-c 8192`, 4 slots) | **works** |

So saving a checkpoint of Flash-Next's recurrent state crashes the driver. It is not moe-cache (stock crashes the same way), not seen with Qwen3.6
or Qwen3.5-122B (also hybrid) on the same machine, nor on Linux / RADV. Launcher: Windows + AMD Vulkan GPU + Flash-Next family gets
`--ctx-checkpoints 0` with a message (`learn` and agent mode use the same helper, b1fc10b). Cost: a hybrid model cannot jump back to an earlier point
of a conversation (plain continuation still reuses the cache); matters for agents (Task 4b checks it). With the workaround, launcher + plugin: ready
after 32 s, 36 CPU expert layers, warm start 8.49 GiB in 8.1 s; first cold request 145 tokens, correct answer, **2.64 tok/s (PRELIMINARY)**.
For an upstream report: driver 32.0.23034.4, llama.cpp b11669 (620d2225c), the table above.

### Kernels on a real AVX-512 CPU (WP4, check mode; kernel_variants built for Windows, 88adca5)

Library picked: `ggml-cpu-zen4.dll`. AVX-VNNI variants are skipped on CPUs without AVX-VNNI (they crashed with an illegal instruction: Zen 4 has
AVX-512 VNNI, not the VEX form). Bit-exact check, 5,120 dots each:

| Kernel | Result |
|---|---|
| iq2_xxs AVX-512 / AVX-512+VNNI (written earlier without an AVX-512 machine) | **bit-exact, 0 mismatches / 0 mismatches** |
| iq3_xxs AVX-512 / AVX-512+VNNI | **bit-exact, 0 / 0** |
| iq3_xxs plugin gather kernel, AVX2 (I36) | **bit-exact, 0** |
| iq2_xxs / iq3_xxs gather, scalar-lookup | bit-exact |

Speed against llama.cpp's own zen4 function, **PRELIMINARY** (apps open, downloads running; one core, data in cache, two runs):

| Kernel | run 1 | run 2 |
|---|---|---|
| iq2_xxs gather | 0.64x | 0.59x |
| iq2_xxs AVX-512 / AVX-512+VNNI | 0.64x / 0.72x | 0.73x / 0.61x |
| iq3_xxs gather / plugin AVX2 kernel (I36) | 0.56x / 0.58x | 0.70x / 0.64x |
| iq3_xxs AVX-512 / AVX-512+VNNI | 0.60x / 0.61x | 0.77x / 0.76x |

Every gather-based and every AVX-512 kernel is 0.56-0.77x on Zen 4 (gathers are slow there, as the original Intel-only note said). Decisions: **I36 stays off on AMD
by default** (`MOE_CACHE_IQ3_KERNEL=1` kept for testing; 8659920); **R12 (AVX-512 kernels) killed on Zen 4** (kill rule: slower than AVX2). A quiet-machine
rerun is unlikely to flip them (every ratio is well under 1.0 in both runs). The AVX-512 kernels are also Linux-only (interposition), so using them would have needed a change in
the executor.

### WP2: Flash-Next IQ2_XS speed, stock vs plugin (PRELIMINARY: apps open, Windows file cache warm for stock)

Script `pc_speed.ps1`: warm-up, 3 decode requests of 256 tokens (a request that ends at once, 0.00, is ignored), 1 prefill of 3,533 tokens. Order ABBA.
Two layouts: the old launcher default (36 CPU expert layers, 12 on the GPU) and the new default (all 48 on the CPU, `--n-cpu-moe 48`).

| Layout / run | decode tok/s | prefill tok/s |
|---|---|---|
| old default, stock A | 10.98 / 13.10 | 62.4 |
| old default, plugin A | 2.87 / 3.03 | 85.1 |
| old default, plugin B | 2.78 / 3.05 | 82.4 |
| old default, stock B | 11.53 / 13.98 | 66.3 |
| new default, stock A | 13.60 / 15.42 | 44.9 |
| new default, plugin A (first session) | 4.92 / 5.60 | 105.3 |
| new default, plugin B | 12.16 / 12.16 | 120.0 |
| new default, stock B | 13.91 / 16.26 | 56.8 |
| new default, plugin stats run (warm profile, 256 tokens) | **13.52** | - |

Stats run, new default: cache 13.3 GiB (start budget 16.05), decode hit rate 95.6 %, 8.2 GB read for 256 tokens. Reading the table:
**decode a little below stock (plugin 12-13.5 after the first session vs stock 14-16), prefill about 2x stock (105-120 vs 45-57)**. Plugin A's 5 tok/s is the first session after a cold start:
the plugin reads the file unbuffered, while stock ran with Windows' file cache already warm from earlier runs (idea I38).

### The finding: on AMD's Windows driver, VRAM in use costs system RAM (launcher rule, db8e634)

Why the plugin decoded about 4x slower than stock with 12 expert layers on the GPU:

| Measurement | Value |
|---|---|
| Plugin cache budget during the run | started 14.13 GiB, fell to 2.1-3.0 GiB; hit rate 56 %; 26 GB read from the SSD for 64 tokens; read time 44.8 s vs compute 12.4 s |
| Stock llama-server alone, private memory | 15.3 GiB with 12 expert layers on the GPU, 6.8 GiB with none |
| System-wide available memory when stock starts | 22.8 -> 7.9 GB (-14.9 GB) while the GPU holds 12.0 GiB dedicated |

So the tuner's shrink was right: the VRAM in use is also taken from system RAM, and the planner over-committed (12 GPU expert layers = 12.0 GiB of
VRAM and ~14.9 GB of RAM, leaving the cache ~3 GiB). Launcher rule: **Windows + AMD GPU + a model bigger than RAM: keep all experts on the CPU.** Stock
gains from it too (11-14 -> 14-16 tok/s). The plugin now also prints one warning when its cache shrinks below half of its peak, with how much the rest
of llama-server uses (2dccf17); on the PC it had fallen 14 -> 2 GiB without a word. Untested: allocation switches, Smart Access Memory off, HIP (idea I37).

### Qwen3.5-122B UD-IQ2_XXS (36.6 GB), new default (all experts on the CPU), ABBA (PRELIMINARY)

Three decode requests per run. A first series lost both plugin runs to a script bug (`powershell -File` drops an empty `-Extra ""`); fixed in
pc_series.ps1 and rerun.

| Run | decode tok/s | prefill tok/s |
|---|---|---|
| stock A | 11.52 / 13.15 / 14.46 | 82.4 |
| plugin A (first session) | 3.35 / 3.32 / 3.44 | 114.1 |
| plugin B | 6.05 / 6.07 / 6.73 | 128.6 |
| stock B | 11.46 / 13.56 / 12.64 | 86.9 |
| extra stock runs (first series) | 10.46-14.29 / 10.89-13.93 | 83.5 / 73.3 |

Plugin prefill 1.4-1.5x stock; decode half of stock or less, possibly stock's warm Windows file cache (idea I38); the clean run in section 55 shows the gap stays with a cold cache, so that explanation is not enough. The 122B diagnosis with stats is still to do.

### GLM-5.3-Flash UD-IQ2_XXS (102 GB, 4 shards), new default, ABBA (PRELIMINARY, incomplete)

| Run | decode tok/s | prefill tok/s |
|---|---|---|
| stock A | 1.55 / 3.35 / 1.44 | 14.9 |
| plugin A | crashed (one decode request done: 1.14 tok/s) | - |
| plugin B | crashed | - |

After the fix (2dccf17, CI-built DLL), same layout:

| Run | decode tok/s | prefill tok/s |
|---|---|---|
| stock B | 1.60 / 3.61 / 1.34 | 14.7 |
| plugin A, fixed DLL | **no crash**; 1.09 / 1.05 / 0.99 | 42.5 (2.9x stock) |
| plugin B, fixed DLL | to be confirmed | to be confirmed |

Decode is below stock here (stock's file cache is warm, idea I38). The new shrink warning fired: the cache fell 15.3 -> 3.8 GiB because the rest of
llama-server used 15.0 GiB even with all experts on the CPU (GLM's non-expert parts on the GPU; idea I37).

### GLM-5.3-Flash: crash in the long-prompt GPU block, fixed (2dccf17)

Both plugin runs above died at the long prompt with an access violation in `VCRUNTIME140.dll` (cache 4.59 GiB; the memory limit was only
9.09 GiB at start). Cause: `gpu_segment` uploaded the
whole expert tensor when at least 90 % of a layer's experts were used, including evicted ones. On Windows an evicted expert is decommitted memory
(access violation); on Linux it is zeros (never used, harmless). Now it packs unless every expert is used, and a whole-tensor upload
(`MOE_CACHE_GPU_PACK=0`) first checks that all experts are in RAM, else the block stays on the CPU. Linux behaviour is unchanged. Confirmed on the PC with the CI-built DLL: plugin A
ran the long prompt without a crash (table above).

### Not measured yet

| Item | State |
|---|---|
| 122B decode diagnosis with stats | to do |
| GLM: plugin B with the fixed DLL | to be confirmed |
| `--fast`, `--hybrid` rows of WP2 | not run |
| Clean timings (apps closed, file cache flushed) | Flash-Next done (section 55); 122B and GLM running |
| Windows job-object memory limit (WP3) on the PC | untested |
| HIP build | needs AMD's HIP SDK |

## 55. Windows desktop, clean WP2: apps closed, file cache flushed before every run (2026-10-26; complete: Flash-Next, 122B, GLM)

Repeat of the speed runs in section 54, whose numbers are PRELIMINARY (apps open; Windows' file cache warm for stock and cold for the plugin). WP2 is done
for the three big models; the plugin stats runs and `--fast` / `--hybrid` rows are not measured (cells marked "-"). The owner's Claude app was open
during the series, so these numbers are still not a perfectly quiet-machine result.

| Item | Value |
|---|---|
| Machine | Windows 11, Ryzen 9 7950X, RX 9070 XT 16 GB (driver 32.0.23034.4), 32 GB DDR5; same as section 54 |
| llama.cpp | b11669 (ggml 0.27.0), official Vulkan zip |
| Plugin | DLL built from 2dccf17 (CI build; includes the long-prompt crash fix) |
| Layout | launcher defaults: all experts on the CPU (`--n-cpu-moe 48`), `--ctx-checkpoints 0` for Flash-Next |
| Owner's apps | mostly closed; the owner's Claude app was still open, so the numbers are marked preliminary in that one respect (what else ran is not recorded) |

**Method.**
- Script `pc_series.ps1 -Flush "fileA;fileB;..."`: before every run it reads the other model files (70-99 GB in total, more than the 31 GB of RAM) from the SSD,
  so Windows' file cache no longer holds the model under test. This replaces dropping the cache with RAMMap, which needs admin rights. Check afterwards that
  the first stock run is not faster than its repeat for a reason other than the cache (to be filled).
- Order ABBA (stock, plugin, plugin, stock), each run a fresh llama-server started by PID and stopped by PID.
- `pc_speed.ps1` per run: warm-up, 3 decode requests of 256 tokens (a request that ends at once is ignored), 1 prefill request of about 3.5k tokens.
- Stats run per model for the plugin (cache budget, hit rate, SSD reads), to explain decode where it differs from stock.

### Flash-Next IQ2_XS (64 GB) (clean run: apps closed, 99 GB of other model files read in 41 s before every run)

Launcher default on Windows + AMD (all experts on the CPU), DLL 2dccf17. Two decode requests per run counted; the first decode request of every run
is a prompt that ends at once (0.00) and is excluded.

| Run | decode tok/s | prefill tok/s |
|---|---|---|
| stock A | 12.13 / 14.94 | 53.0 |
| plugin A | 12.30 / 12.34 | 120.3 |
| plugin B | 12.16 / 12.17 | 119.7 |
| stock B | 13.92 / 16.05 | 56.6 |
| plugin stats run (cache budget, hit rate, GB read) | - | - |

Reading: **plugin decode about 12.2 tok/s in all four requests (very steady) against stock 12.1-16.1**; stock's faster requests are 20-30 % above the
plugin's, its slowest equals it. **Prefill 2.2x stock** (120 vs 53-57). Both agree with the preliminary numbers in section 54 (decode a little below stock,
prefill about 2x), now with a cold file cache for every run: the plugin's first-session gap seen there (5 tok/s) did not appear (plugin A 12.3).
Note: the runner's `speed.log` failed to open ("Stream was not readable", a log-file lock); the numbers above are from the runner's captured output.

### Qwen3.5-122B UD-IQ2_XXS (36.6 GB) (clean run: apps closed, other model files read before every run)

Three decode requests per run, launcher default (all experts on the CPU), DLL 2dccf17.

| Run | decode tok/s | prefill tok/s |
|---|---|---|
| stock A | 11.12 / 13.64 / 14.22 | 80.0 |
| plugin A | 6.61 / 6.54 / 7.04 | 129.3 |
| plugin B | 6.08 / 6.09 / 6.57 | 127.7 |
| stock B | 10.60 / 13.65 / 14.03 | 86.8 |
| plugin stats run (cache budget, hit rate, GB read) | - | - |

Reading: **decode plugin about 6.5 tok/s against stock about 11-14 (half)**; **prefill 1.5-1.6x stock** (128-129 vs 80-87). Unlike Flash-Next, the decode
gap stays with a fair cold start for both, so it is not only stock's warm file cache (the explanation offered in section 54 does not hold for this
model). The cause is not known yet; the stats run (cache budget, hit rate, SSD reads) is still to do.

### GLM-5.3-Flash UD-IQ2_XXS (102 GB, 4 shards) (cold start each run: file cache flushed by reading Flash-Next's 68 GB)

Three decode requests per run, launcher default (all experts on the CPU), DLL 2dccf17. No crash in either plugin run.

| Run | decode tok/s | prefill tok/s |
|---|---|---|
| stock A | 1.38 / 3.70 / 1.70 | 15.3 |
| plugin A | 1.18 / 1.05 / 1.03 | 44.4 |
| plugin B | 1.07 / 1.01 / 1.03 | 45.1 |
| stock B | 1.21 / 3.60 / 1.65 | 14.9 |
| plugin stats run (cache budget, hit rate, GB read) | - | - |

Reading: **prefill 2.9-3.0x stock** (44-45 vs 15). **Decode: plugin about 1.05 tok/s in every request against stock 1.2-3.7**; stock's second request is about 3.5x
the plugin's (3.6-3.7 vs 1.0-1.2; the cause of that one fast request is not known), the other two are within about 1.0-1.7x of it. The plugin's cache was squeezed to about 3.7 GiB by the AMD
driver's RAM cost (the shrink warning fired, ideas I37 / I40), so it reads from the SSD for most picks. The model is 102 GB on a 32 GB machine, so both
sides are bound by reading; the plugin's decode is not better here.

### Against section 54

| Model | Section 54 (PRELIMINARY): stock / plugin decode, plugin prefill vs stock | Section 55 (clean) |
|---|---|---|
| Flash-Next | 14-16 / 12-13.5 tok/s; prefill about 2x | 12.1-16.1 / 12.2 tok/s; prefill 2.2x |
| Qwen3.5-122B | 11-14 / 3-7 tok/s; prefill 1.4-1.5x | 11-14 / 6.1-7.0 tok/s; prefill 1.5-1.6x |
| GLM-5.3-Flash | 1.3-3.6 / 1.0-1.1 tok/s; prefill 2.9x | 1.2-3.7 / 1.05 tok/s; prefill 2.9-3.0x |

What the clean series shows: (1) with a cold file cache for both, the plugin's first-session decode gap (idea I38) is gone on Flash-Next (12.3 vs stock 12.1-16.1)
but not on 122B (about half of stock), so unbuffered reads alone do not explain it; (2) prefill is 2.2x (Flash-Next), 1.5-1.6x (122B) and 2.9-3.0x (GLM) stock on
this machine; (3) on GLM the shrink from the AMD driver's RAM cost still happens (cache about 3.7 GiB), as in section 54. Open: the 122B decode diagnosis with
stats (`handover` note 122B_DIAG_PLAN) and the budget idea I40.

## 56. Windows desktop experiments (2026-10-26/27; SKELETON: known facts filled, empty cells "-" are not measured yet)

Machine, llama.cpp and DLL as in sections 54-55 (Ryzen 9 7950X, RX 9070 XT 16 GB, driver 32.0.23034.4, 32 GB RAM; llama.cpp b11669 Vulkan). Scripts are in
`windows-kit/` (`pc_runlib.ps1`, `pc_prefill_lib.ps1`, `pc_r*.ps1`). Every speed is PRELIMINARY until the section says otherwise; quote only filled tables.

### I37: VRAM in use "costs" system RAM on Windows: it is llama.cpp's file mapping, not the AMD driver

**Resolved 2026-10-27.** The first measurements (below) looked like the AMD driver taking RAM for VRAM. A trim test on the new driver 32.0.32015.7029 (owner's apps
closed) shows the RAM is the model file's pages in llama-server's working set. Stock llama-server, Flash-Next, 12 GPU expert layers, b11669:

| Moment | RAM taken (system-wide Available drop) | llama-server working set | private memory | GPU dedicated |
|---|---|---|---|---|
| after load | 13.3 GB | 12.8 GiB | 15.3 GiB | 13.2 GiB |
| after `EmptyWorkingSet` on its PID | 0.6 GB | 0.0 | - | 13.2 GiB (unchanged) |
| after one 16-token request | 5.3 GB (the CPU experts it read) | - | - | 13.2 GiB |

The request still answered correctly after the trim. Cause (code reading of b11669, `handover/I37_MMAP_FINDING_20261027.md`): llama.cpp uploads GPU tensors straight from the
file mapping and, on Windows, never unmaps it afterwards (`unmap_fragment` is a no-op there), so those file pages stay in the working set, and Windows'
"Available" counter excludes working-set pages (Linux's MemAvailable counts them). The 13.5-14.9 GB seen earlier on driver 32.0.23034.4 fit the same cause. `--load-mode none`
is worse for stock (private memory 40 GB, 23.8 GB taken: it copies the CPU experts). The upstream angle is a llama.cpp Windows loader issue (the owner writes it), not an AMD report.
Fix in progress in the plugin: trim once after load, before the warm start. After it works, the RAM-budget layout of I40 (killed below on the old accounting) is worth a rerun.

First measurements (driver 32.0.23034.4; system RAM taken at startup, 12 expert layers on the GPU) — these only showed that the llama.cpp Vulkan switches change nothing:

| Setting | system RAM taken | private memory | GPU dedicated |
|---|---|---|---|
| baseline | 13.5 GB | 15.3 GiB | 12.0 GiB |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | 14.3 GB | 15.6 GiB | 12.5 GiB |
| `GGML_VK_VISIBLE_DEVICES=1` (built-in Radeon hidden) | 14.4 GB | 15.3 GiB | 12.0 GiB |
| `GGML_VK_ENABLE_MEMORY_PRIORITY=1` | 14.6 GB | 15.3 GiB | 12.0 GiB |
| host-visible off + built-in Radeon hidden | 14.3 GB | 15.6 GiB | 12.5 GiB |

With `GGML_VK_MEMORY_LOGGER=1` ggml-vulkan allocates 13.13 GiB of device memory and only 169 MiB of host memory (consistent with the mapping explanation: the RAM is not driver memory).

| Option | Result |
|---|---|
| Newer Adrenalin than 32.0.23034.4 (32.0.32015.7029) | RAM taken unchanged (13.3 GB): not the driver; it does fix the Flash-Next checkpoint crash (I44) |
| Trim the working set after load (`EmptyWorkingSet`) | RAM taken 0.6 GB, GPU memory unchanged: confirmed |
| Hardware-accelerated GPU scheduling off, HIP build, Linux / RADV | not needed any more |

Earlier text in sections 54-55 and the I40 rows that say "VRAM costs RAM on AMD's Windows driver" describe the symptom correctly; the cause is the file mapping.

### Qwen3.5-122B decode: why the plugin is at half of stock (plan: shared note `122B_DIAG_PLAN.md`)

Clean-run numbers are in section 55 (plugin about 6.5 vs stock about 11-14 tok/s). Stats run: plugin, cold start, one 256-token request; counters read
before and after (`MOE_CACHE_STATS_SIGNAL=1`, named event) and subtracted. PRELIMINARY (the owner's Claude app was open).

| Measurement | Default (executor on) | `MOE_CACHE_EXEC=0` |
|---|---|---|
| Decode speed | 7.02 tok/s | 5.84 tok/s |
| Cache budget at start -> end, `shrinks` | 16.89 -> 13.6 GiB, 5 | 13.2 GiB at the end |
| Decode hit rate | 87.3 % | 86.7 % |
| `read_bytes` | 35.9 GB (about 140 MB per token) | - |
| `ns_read` (summed over the 4 readers) | 67.1 s | - |
| `ns_cpu` (includes waiting for loads) / `ns_prep` / `ns_total` | 26.7 s / 3.9 s / 30.6 s | - |
| `exec_ops` | 36,864 (144 per token) | - |

Second series, same model and request, cache budget about 11 GiB (a smaller cache than the run above; PRELIMINARY):

| Run (122B, plugin, cold) | decode tok/s | decode hit rate | read | `ns_read` |
|---|---|---|---|---|
| default (4 readers) | 5.97 | 84.2 % | 44.3 GB | 85 s |
| `MOE_CACHE_EXEC_READERS=8` | 5.86 | 83.9 % | 45.3 GB | 120 s |
| `MOE_CACHE_EXEC_READERS=16` | 5.81 | 83.7 % | 45.7 GB | 135 s |
| `MOE_CACHE_LOOKAHEAD=2` | 6.35 | 93.1 % | 54.5 GB | 58 s |
| `MOE_CACHE_SIZE_GIB=14`, `--ram 18` | - | - | - | - |

More readers do not help: they only queue (random reads of about 1 MB, unbuffered, deliver about 1.2 GB/s, against about 2.4 GB/s for sequential reads), and `ns_read` grows
with the number of readers. Lookahead gives +6 % decode for 23 % more bytes read. The lever that is left is the cache size (I38, buffered reads: next).

Reading, against the three causes in the plan:
- **Executor overhead (cause 3): no.** Turning the executor off is slower (5.84), and `ns_prep` is only 3.9 s of the 30.6 s.
- **A collapsed cache (cause 1): only partly.** The budget fell from 16.9 to 13.6 GiB (5 shrinks), well short of Flash-Next's or GLM's squeeze, but the decode hit rate is
  only 87 %, so about 49 of the 384 picks per token miss.
- **Cost of the misses (cause 2): the likely main cost.** About 140 MB per token has to be read. With 4 readers at about 2.1 GB/s that is at least
  65 ms per token, which is the gap to stock (about 75 ms per token in total). Stock serves the same misses from Windows' warm file cache. Flash-Next has about 3x fewer active bytes per
  token, so its reads hide behind compute.
- The plugin's dot products are not the problem (the same zen4 functions as stock, 32-row chunks over 16 threads, from the code reading in the shared note
  `qwen122b_windows_gap.md`). Still to test: more readers and lookahead (above), and buffered reads (I38).

Conclusion (preliminary): the 122B gap is read cost for a 12-16 % miss rate through unbuffered reads, not compute; more readers do not help, a bigger cache (or cheaper reads, I38) would.

### I40: all experts on the CPU vs a RAM-budgeted number of GPU expert layers (Windows + AMD)

Branch `claude/i40-amd-ram-budget`: the launcher plans with (available RAM - cache floor) / 1.1 as the VRAM budget instead of keeping every expert on the CPU.

| Model | Layout | GPU expert layers | Plugin decode tok/s | Stock decode tok/s | Cache budget at the end |
|---|---|---|---|---|---|
| Flash-Next | all on CPU (today), A | 0 | 12.94 / 12.91 (prefill 121.5) | - | - |
| Flash-Next | RAM budget (I40), A | - | 8.55 / 10.55 (prefill 129.3) | - | - |
| Flash-Next | RAM budget (I40), B | - | 8.66 / 10.57 (prefill 127.9) | - | - |
| Flash-Next | all on CPU (today), B | 0 | 12.70 / 12.68 (prefill 120.9) | - | - |
| Qwen3.5-122B | all on CPU (today), A | 0 | 5.85 / 5.86 / 6.41 (prefill 125.6) | - | - |
| Qwen3.5-122B | RAM budget (I40), A | - | 3.99 / 4.20 / 4.24 (prefill 135.8) | - | - |
| Qwen3.5-122B | RAM budget (I40), B | - | 4.15 / 4.29 / 4.20 (prefill 134.7) | - | - |
| Qwen3.5-122B | all on CPU (today), B | 0 | 5.64 / 5.79 / 6.37 (prefill 126.3) | - | - |
| GLM-5.3-Flash | all on CPU | 0 | - | - | - |
| GLM-5.3-Flash | RAM budget | - | - | - | - |

Plugin, cold start each run, ABBA (all-CPU A, I40 A, I40 B, all-CPU B), PRELIMINARY (the owner's Claude app was open):

| Model | Decode, all on CPU (mean) | Decode, RAM budget (mean) | Change | Prefill, all on CPU / RAM budget | Change |
|---|---|---|---|---|---|
| Flash-Next | 12.81 tok/s | 9.58 tok/s | **-25 %** | 121.2 / 128.6 tok/s | +6 % |
| Qwen3.5-122B | 5.99 tok/s | 4.18 tok/s | **-30 %** | 126.0 / 135.3 tok/s | +7 % |
| GLM-5.3-Flash | - | - | - | - | - |

**Verdict: I40 KILLED.** Giving GPU expert layers back by a RAM budget costs 25-30 % of decode for 6-7 % more prefill on both models; the all-CPU default stays.
Branch `claude/i40-amd-ram-budget` is not merged. (Both orders of each pair agree, so this is not a drift effect; the GPU-layer counts the budget chose are not recorded here.)

### Disk reads on Windows (`win_read_bench`, CI build c852720; PRELIMINARY)

Qwen3.5-122B file on the PC's NVMe drive (Sabrent, C:, `stornvme` driver, BitLocker off, file neither sparse nor compressed). **Windows Defender real-time protection
was ON** during these runs (the owner turned it off afterwards; re-test pending). The "x N" is the benchmark's parallelism setting. "Unbuffered" is what the plugin does
today (`FILE_FLAG_NO_BUFFERING`, overlapped); "buffered" goes through the file cache.

| Read size x N | Unbuffered GB/s (ops/s) | Other unbuffered variants | Buffered GB/s |
|---|---|---|---|
| 1 MiB x 4 | 3.02 (2,878) fresh buffers | reused 3.21; bounce buffer 3.11; commit only 15.5 | 3.96 |
| 1 MiB x 16 | 1.67 (1,597) | - | 8.54 |
| 256 KiB x 4 | 0.69 (2,630) | - | 3.23 |
| 256 KiB x 8 | 0.68 (2,609) | - | 6.82 |
| 128 KiB x 8 | 0.38 (2,893) | - | 3.59 |

Reading: unbuffered reads are capped at about 2,600-2,900 operations per second whatever the size or the number of threads, so smaller reads and more threads
do not help (they get slower in GB/s). Fresh and reused buffers give the same rate, which rules out page commit as the cost. The plugin already opens its files
with `FILE_FLAG_OVERLAPPED`. Suspects, not yet tested: Defender's file-system filter on each unbuffered operation (re-test with real-time protection off), and the
drive running at a queue depth of one per thread (a queue-depth benchmark is pending). Buffered reads are faster on every row here, up to 8.5 GB/s.

### I38: buffered reads (cold and warm file cache)

| Condition | First-session decode tok/s | Second-session decode tok/s |
|---|---|---|
| Unbuffered reads (today), cold cache | 5.89 (hit rate 83.2 %, read 47.1 GB, cache budget 10.8 GiB) | - |
| Buffered reads, cold cache | - (the run was cut off: the PC went offline) | - |
| Unbuffered reads, warm cache | - | - |
| Buffered reads, warm cache | - | - |

### R2: `-ub` at long context (Flash-Next, `--ctx 65536`; `pc_r2_ub_long.ps1`)

Prefill tok/s for a ~15k and a ~30k-token prompt. Kill rule: `-ub` 2048 within 10 % of 1024.

| `-ub` | 15k stock | 15k plugin | 30k stock | 30k plugin |
|---|---|---|---|---|
| 512 | - | - | - | - |
| 1024 | - | - | - | - |
| 2048 | - | - | - | - |

Verdict: -

### R5: bigger prompt chunks (Qwen3.6, stock, same `--n-cpu-moe`; `pc_r5_ub_stock.ps1`)

Prefill tok/s. Kill rule: `-ub` 4096 within 10 % of 2048.

| `-ub` | ~2k prompt | ~8k prompt | ~32k prompt |
|---|---|---|---|
| 2048 | - | - | - |
| 4096 | - | - | - |
| 8192 | - | - | - |

Verdict: -

### R11: VRAM freed by a q8_0 KV cache at 64k context (Flash-Next, plugin; `pc_r11_kv_vram.ps1`)

GPU dedicated memory of llama-server in GiB (the quality side, KL 0.0045 for q8_0 KV alone, is section 53).

| Variant | idle | after a ~2k prompt | after a ~20k prompt | adapter after 20k | 20k prefill tok/s |
|---|---|---|---|---|---|
| A: f16 KV, `-ub 512` | - | - | - | - | - |
| B: q8_0 KV, `-ub 512` | - | - | - | - | - |
| C: `--fast` (q8_0 KV, tuner's `-ub`) | - | - | - | - | - |

VRAM freed by q8_0 KV (A - B): -

### R7 (MTP head on the GPU; `pc_r7_mtp.ps1`) and R3 (`--hybrid` on Flash-Next; `pc_r3_hybrid.ps1`)

| R7: Qwen3.6, 8 runs (palindrome) | decode tok/s | prefill tok/s | draft accepted / drafted |
|---|---|---|---|
| stock | - | - | - |
| stock + MTP | - | - | - |
| plugin | - | - | - |
| plugin + MTP | - | - | - |

Kill rule R7: no gain from MTP on stock AND on the plugin. Verdict: -

| R3: Flash-Next | decode tok/s | prefill tok/s | cache budget at the end |
|---|---|---|---|
| exact (plugin default) | - | - | - |
| `--fast` | - | - | - |
| `--hybrid` 6 GB bank | - | - | - |
| `--hybrid` 10 GB bank | - | - | - |

Kill rule R3: hybrid under +10 % decode over exact. Verdict: -

### Task 4b: Hermes Agent on Windows through moe-cache (with the API key), and WP3 (`--ram` job memory limit)

OpenCode was first only installed as a desktop app on the PC (no CLI); the owner then installed the CLI (v2.1.4, npm), so both agents were tested (rows below).

| Check | Result |
|---|---|
| `--agent` server starts on the PC (`moe_cache.py server qwen3.6 --agent --ram 20 --always-plugin`); the launcher prints that a key is in use and where it is stored | yes |
| A request without the key gets 401, with `moe_cache.py key` as Bearer 200 | yes: 401 / 200 |
| `connect hermes --write` finds the stored key (Hermes config on Windows) | yes: the key reached Hermes Agent v0.22.5's `config.yaml` (not printed); the owner's Hermes config was restored afterwards |
| Hermes one-shot task (edit a scratch file): first turn tokens / time, whole session | task done correctly in 81 s (whole session); first-turn tokens not recorded |
| OpenCode CLI v2.1.4 through moe-cache (`server qwen3.6 --agent --ram 20 --always-plugin`, `connect opencode --write`: `%USERPROFILE%\.config\opencode\opencode.json` with the API key; the owner's file was restored afterwards) | `opencode run` task "fix add() in calc.py" done correctly in 114 s (whole session); first prompt 8,728 tokens at 124.5 tok/s; decode 7.0-9.5 tok/s |
| Launching OpenCode from a script on Windows | start `opencode.cmd`, not the npm `opencode.ps1` shim (`Start-Process` cannot run a .ps1) |
| Warm session after a restart: first turn tokens / time | - |
| Flash-Next with `--ctx-checkpoints 0`: cost for an agent that edits history | - |
| WP3: `--ram 20` sets a job memory limit; the log line; the server stays under it | log line "Windows job memory limit 20.0 GB on llama-server"; the cache was sized from it (start budget 12.5 GiB, margin 4 GiB). "Stays under it" under load: not measured |
| WP3: a run that hits the limit behaves like the Linux `--ram` (message, no crash) | - |

What to conclude (to be written from the numbers): whether the plugin's first-session decode gap (idea I38, unbuffered reads) survives a cold file cache
for stock too; whether the shrink on GLM (idea I37) still happens with the apps closed.

## 57. Agent-shaped benchmark (idea I30): replayed OpenCode and Hermes sessions, stock vs moe-cache (2026-10-26 evening, laptop)

What agents wait for is the first token of each turn and the whole task, not decode tok/s. `tools/agent_bench.py` recorded two real sessions through a
proxy and replays them (same request bodies, same overlap, same pauses; greedy, capped at the recorded token counts); `tools/agent_bench_pair.sh` runs
the pairs. Recordings (`tests/agent_sessions/`, made with moe-cache `--agent` on Qwen3.6): **OpenCode 1.19.4** "the tests in test_stats.py fail: run
them, fix stats.py, run them again" (5 requests, first prompt 7.45k tokens, tests pass) and **Hermes Agent v0.22.5** (`-t file`) "fix median() and
add docstrings, then stop" (7 requests incl. its title request, first prompt 3.7k tokens, file fixed).

| Item | Value |
|---|---|
| Machine | laptop, i7-13620H, RTX 4060 8 GB, 31 GB RAM; llama.cpp 23b0202a1; plugin build-light (v0.2.3.5) |
| moe-cache | `moe-cache-server MODEL --agent --ram R -- --alias A --slot-save-path EMPTY_DIR` (context 65536, 4 slots, checkpoints, jinja; usage profile from earlier use) |
| stock | the same llama-server command the launcher prints for `--agent --dry-run`, without the plugin (Qwen3.6: `--n-cpu-moe 38 -ub 1024`; Flash-Next: `--n-cpu-moe 48 -ub 512`) |
| Limit | hard cgroup limit, no swap: Qwen3.6 16 GB (`--ram 14`), Flash-Next 23 GB (`--ram 22`) |
| Per run | model dropped from the page cache and **checked** (mmap + mincore: 0.00 GB), fresh server, /health + 15 s, clean-run gate (`wait_cool.sh 65`), OpenCode replay then Hermes replay on the same server, stop |
| Same work | prompt tokens processed and tokens generated were identical per request in every run on both sides (exact output) |

**Qwen3.6-35B-A3B Q4_K_M (21.7 GB), 16 GB limit, 3 runs per side, order stock, mc, mc, stock, mc, stock (min / median of the runs):**

| Session | Side | First-turn TTFT (s) | Follow-up TTFT, median per run (s) | Whole task (s) | Decode median (tok/s) |
|---|---|---|---|---|---|
| OpenCode | stock | 49.7 / 62.4 | 7.1 / 7.5 | 176.9 / 182.7 | 8.3-9.9 |
| OpenCode | moe-cache | **44.8 / 48.9** | **4.6 / 4.8** | **105.4 / 111.6 (1.64x)** | 14.8-15.8 |
| Hermes (second session on the server) | stock | 165.5 / 244.5 | 22.7 / 31.4 | 476.5 / 623.5 | 3.5-5.2 |
| Hermes (second session on the server) | moe-cache | **23.0 / 27.5** | **5.7 / 6.2** | **124.1 / 127.0 (4.9x)** | 10.0-14.8 |

Single runs: OpenCode task stock 183.1 / 176.9 / 182.7 s, moe-cache 113.6 / 105.4 / 111.6 s; Hermes stock 476.5 / 623.5 / 686.7 s, moe-cache
124.1 / 171.3 / 127.0 s (the 171 s run decoded at 10.0 instead of 12.6-14.8 tok/s; cause not found). No out-of-memory kill in any run; stock hit
the cgroup ceiling 554-655 k times per run, moe-cache 37-87 k.

**Flash-Next IQ2_XS (68 GB), 23 GB limit, one pair only (PRELIMINARY: stopped at the owner's request, see the idle-spin bug below):**

| Session | stock (run 1) | moe-cache (run 2) |
|---|---|---|
| OpenCode: first-turn TTFT / follow-up median / task | 263.4 s / 34.6 s / 817.2 s (decode 2.05 tok/s) | **173.1 s / 8.5 s / 277.6 s (2.9x)** (9.3 tok/s) |
| Hermes: first-turn TTFT / follow-up median / task | 914.4 s / 103.8 s / 2063.5 s (1.81 tok/s) | **82.7 s / 13.4 s / 270.9 s (7.6x)** (7.4 tok/s) |

The moe-cache run started after a failed clean-run gate (74 C, load 9 after 30 min) because of the bug below, so it ran hot: if anything its numbers
are low. Runs 3-4 were not done.

Findings:
- **The first turn is the agent's biggest wait, and moe-cache shortens it less than everything else.** On Qwen3.6 the OpenCode first turn (7.45k
  tokens) is 49 vs 62 s: long prompts already go to the GPU in stock llama.cpp (section 46). Follow-up turns (100-800 new tokens each) are where the
  gap opens: 4.8 vs 7.5 s per turn (median) (the I4 packed upload, section 52), plus decode at 1.7x.
- **Stock gets much worse for the second session on the same server; moe-cache does not.** Hermes after OpenCode: stock first turn 165-255 s for a
  3.7k prompt (OpenCode's 7.45k took 50-65 s), decode 3.5-5 tok/s: the first conversation's KV checkpoints (host RAM) shrink the room left for the
  memory-mapped model, so stock faults more. moe-cache's cache sits in its own budget and keeps 23-31 s. That is the agent case: one server, several
  sessions in a row.
- **Measurement trap (new): page-cache eviction silently fails on the ollama blob** `/var/lib/ollama/.../sha256-d372de8e...` (21.7 GB stays cached
  after `posix_fadvise(DONTNEED)`, no visible process maps it; on files I own eviction works). A first set of 4 runs on that blob let stock read
  ~5.6 GB outside its cgroup (stock scope charged 16.1 GB file pages) and was discarded; the runs above use a byte-identical copy in work/zoo/qwen36/
  under the same file name. Earlier limited stock runs on that blob may be optimistic for stock. The pair script now checks residency and stops.
- **Plugin bug found: an idle Flash-Next moe-cache server burns ~720 % CPU** (9-10 executor threads at 65-90 % each, mostly futex_wait) for 30+ min
  with no request; the warm start is not it (9.85 GiB in 4.0 s). Qwen3.6 servers idle normally. Logs kept for the robustness session
  (work/agents/i30/fn/server-2-mc.log, server-3-mc.log).
- A recorded Hermes session with file tools only ran into a loop (the task was done by request ~15, then ~85 identical 177-token tool calls trying to
  verify, until the 25 min timeout); it is kept out of the benchmark (work/agents/i30/rec/old/hermes_loop.json). The prompt now tells it to stop after
  editing.
- Follow-up turns process 100-250 tokens more than in the recording (the replay re-sends the agent's original assistant message; the model now
  writes something else at temperature 0); the same on both sides. The warm-session keeper logged no save or restore during any run (it saves after
  20 s idle; each server was stopped sooner).

## 58. Windows desktop, 2026-10-27: fixes confirmed, MTP, Flash-Next IQ2_XS vs IQ3_XXS, GLM page file

PC: Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5, Windows 11, llama.cpp Vulkan **ggml 0.27 (620d2225c)** (no longer b11669: numbers before this
section are not strictly comparable), AMD driver 32.0.32015.7029. Every run cold (other big model files read first). Decode = 256-token greedy
requests; one prompt ("CPU cache") now ends after 1 token on Flash-Next with stock and plugin alike (newer llama.cpp), so it is left out.

**Fixes confirmed on the PC**
| Bug | Cause | Fix (merged in experimental) | PC check |
|---|---|---|---|
| Crash at request 2 (122B from the fast D: drive) | an unbuffered read of one expert covered pages it shares with a neighbour; Windows shows other data there while the read runs, so the executor read garbage, router scores went NaN, a token listed one expert twice and the task's column arrays overflowed | direct reads write only the expert's own whole pages (head/tail through the bounce buffer); task bound check; poison-reads CI step + mutant (1299479, 6f72b0b) | correct text at both requests, 0 mismatches in `MOE_CACHE_EXEC_CHECK=1`, no crash |
| Wrong 122B output ("3333...") once the crash was gone | same cause | same | as above; `MOE_CACHE_DIRECT=0` (bounce reads only) was also correct, 13.46 tok/s |
| Read into uncommitted memory (Windows error 998) after a refused commit (1455); commit at an invalid address (487) | failed commits went on page by page and the reader wrote into them | a refused commit shrinks the cache; 487 clamps to the reservation (ed70328) | no 998/487 lines in later runs |
| The cache starved Windows of commit (GLM + MTP: a new PowerShell could not start) | the start budget after the I37 trim ignored llama.cpp's own commit; on Windows Vulkan VRAM allocations are charged to system commit (~24 GB with GLM) | start budget capped at free commit minus max(2 GiB, 8 %), re-checked by tune() (87323e5) | GLM + MTP: budget 4.25 GiB instead of 19.07, lowest free commit 2.9 GB, Windows stayed usable |
| `--ram` GB passed as GiB | launcher units | B1 (ebc946f) | - |

**Flash-Next (Qwen3.8-Flash-Next-GSQ-RCO), decode tok/s on two prompts ("CPU cache 2" / "water cycle"), prefill of 3.5k tokens; ABBA order; DLL 1299479**
| Quant | Setup | Decode | Prefill |
|---|---|---|---|
| IQ2_XS (68 GB, C:) | stock, launcher's 12 GPU expert layers | 15.23 / 19.36, 15.03 / 19.22 | 109.5, 107.9 |
| IQ2_XS | plugin default (12 layers) | 15.09 / 15.22, 15.01 / 14.95 | 120.2, 120.5 |
| IQ2_XS | **plugin, `MOE_CACHE_WIN_BUFFERED=1` + `--n-cpu-moe 35` (13 layers)** | **18.51 / 19.57, 18.49 / 19.44** | **152.7, 152.7** |
| IQ3_XXS (76 GB, D:) | stock, launcher's 9 GPU expert layers | 14.65 / 17.89, 14.75 / 17.51 | 96.4, 102.0 |
| IQ3_XXS | plugin default (9 layers) | 9.97 / 10.89, 9.90 / 10.85 | 93.9, 93.7 |
| IQ3_XXS | plugin, buffered + 13 layers | 11.94 / 13.16, 12.01 / 13.15 | 108.0, 109.4 |
Text: plugin = stock (first 200 characters of both requests) on IQ2_XS and IQ3_XXS. The issue #1 user (Linux, same GPU, 13 layers) had 18.8 (IQ2_XS) and
16.9 (IQ3_XXS). On IQ3_XXS the plugin loses decode to stock on Windows; executor on/off test: see below.

**Qwen3.6-35B-A3B (fits in RAM), MTP head on the GPU (`--spec-type draft-mtp --spec-draft-n-max 2`)**
| | no MTP | MTP | drafts accepted |
|---|---|---|---|
| stock | 46.01 / 46.10 | 68.28 / 68.92 (+49 %) | ~78 % |
| plugin (forced) | 45.57 / 45.62 | **70.29 / 70.20 (+54 %)** | |
Prefill -5..-8 % with MTP. Flash-Next and the 122B quants have no head in their files; GLM has one (blk.45.nextn) but with MTP its cache fell to 4.25 GiB
(commit) and decode to 0.76-0.99 tok/s: MTP stays off for GLM on this PC.

**GLM-5.3-Flash (102 GB), one cold run per setting (preliminary)**
| Page file | Setting | Decode | Cache |
|---|---|---|---|
| 6.9 GB | default | 1.52 | 16.1 -> 9.6 GiB (commit limit) |
| 14 GB | default | 2.02 | 16.2 -> 14.9 |
| 14 GB | executor off | 1.71 | 15.2 -> 15.8 |
| 14 GB | buffered reads | **2.20** | 16.1 -> 14.8 |
| 14 GB | executor off + buffered | 2.02 | 15.1 -> 15.8 |

**Qwen3.5-122B UD-IQ2_XXS from D: (request 2 of the crash sequence)**: stock 12.35-13.06; plugin after the fix 12.89 (measured while a download wrote
to D:, not clean); `MOE_CACHE_DIRECT=0` 13.46. A clean ABBA series is still to do.

**IQ3_XXS executor on / off** (buffered + 13 layers): on 11.93-12.01 / 13.05-13.16, off 11.39-11.49 / 12.67-12.68: the executor helps a little;
the gap to stock (14.7 / 17.5-17.9) is elsewhere (stock reads IQ3's experts through Windows' file cache; to investigate).

**Quality yardstick 2.0** (tools/quality_yardstick.py, branch claude/quality-yardstick; 24 open prompts x 256 tokens + 20 exact-answer tasks, greedy),
Flash-Next through the plugin (buffered + 13 layers), IQ3_XXS as the reference:
| Quant | Tasks right | Loops | Time | vs IQ3_XXS |
|---|---|---|---|---|
| IQ3_XXS | 18/20 | 1/44 | 850 s | reference |
| IQ2_XS | 17/20 | 0/44 | 475 s | every answer worded differently (median first difference at token 22, 18 % same prefix); 3 tasks changed |
A small set: IQ2_XS is not measurably worse here; a bigger task set is needed to settle "IQ2 answers worse".

**Flash-Next MTP head as a separate file** (unsloth `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, `-md ... --spec-type draft-mtp --spec-draft-n-max 2 -ngld 99`), IQ2_XS:
| | no head | head | drafts accepted |
|---|---|---|---|
| stock (12 layers) | 15.10 / 19.22, prefill 110 | 13.06 / 15.85 (-15 %), prefill 79 | 74 % |
| plugin (buffered + 13 layers) | 18.58 / 19.55, prefill 153 | 11.02 / 12.39 (-38 %), prefill 103 | 73 % |
Killed: verifying 3 tokens at once on CPU-held experts costs more than the drafts save. MTP pays only when the experts are in RAM/VRAM (Qwen3.6).

**Flash-Next IQ3_S (83.6 GB, C:), first look, one cold run each:** stock (launcher's 7 GPU expert layers, `--n-cpu-moe 41`) decode 10.62 / 12.66 / 12.22,
prefill 49.3; plugin buffered + 13 layers decode 8.73 / 9.19 / 9.81, prefill 88.6 (1.8x). As on IQ3_XXS, the plugin's decode lags stock on IQ3 types on
Windows while its prefill is far ahead. (IQ3_S file 2 is byte-identical to IQ3_XXS file 2, same sha256.)

## 59. Windows desktop, 2026-10-27 evening: the cache moves to read-only views of the model file (no commit) and beats stock

PC: Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5, page file 14 GB, Windows 11, llama.cpp Vulkan ggml 0.27 (620d2225c), launcher 65facd1.
Every run cold (tools/win_flush_cache.ps1 reads 38.5 GB of other files first). Decode = the 256-token greedy "water cycle" request after a
64-token warm-up, `cache_prompt` off. Stock = llama-server with the same `-ngl 99 --n-cpu-moe N -c 8192 -fa on --load-mode mmap -ub 2048`.

**Why the plugin lost on IQ3 (stats of the measured request, plugin with private memory, buffered reads, 13 GPU layers)**
| Model | decode | hit % | disk read | cache | resizes in one request |
|---|---|---|---|---|---|
| Flash-Next IQ2_XS | 18.37 | 96.0 | 5.5 GB | 10.85 GiB | 75 / 74 |
| Flash-Next IQ3_XXS | 12.61 | 88.0 | 19.5 GB | 8.31 GiB | 226 / 226 |
Windows charges private memory to commit (RAM + page file), and Vulkan's GPU allocations too: the cache was held at 8-11 GiB (start budget cut
from 17.4 to 11.5 GiB, "free commit 15.13 GiB") while stock's mmap pages are file-backed and never charged. Not a kernel issue: on Windows stock
and plugin both use ggml-cpu-zen4.dll (the plugin's own kernels are Linux-only).

**Feasibility (C:\dl\locktest.exe, wstest.exe; 15 GB of the IQ3_XXS file)**
| Method | system commit | load speed | under 14 GB extra pressure |
|---|---|---|---|
| VirtualLock on a read-only view | +15 GB | 1.94 GB/s | - |
| hard working-set minimum, no lock | **+0** | 2.67 GB/s | 87 % stayed, rest in standby |

**The fix** (branch claude/win-file-cache, now the Windows default; `MOE_CACHE_WIN_VIEW=0` = private memory): expert tensors point into the
plugin's read-only view of the model file; load = PrefetchVirtualMemory + touch, evict = drop from the working set (pages stay in the file
cache), a hard working-set minimum of budget + rest of the process. Cache policy, prefetch and executor unchanged (same as Linux).

**Cold ABBA, stock vs plugin, launcher defaults (decode tok/s)**
| Model | GPU expert layers | stock | plugin (file view) | change |
|---|---|---|---|---|
| Flash-Next IQ2_XS (68 GB, C:) | 12 | 15.35, 14.43 (14.89) | 22.00, 21.87 (**21.94**) | **+47 %** |
| Flash-Next IQ3_XXS (76 GB, D:) | 9 | 15.10, 15.18 (15.14) | 19.80, 19.91 (**19.86**) | **+31 %** |
| Flash-Next IQ3_XXS | 13 | 13.46, 13.29 (13.38) | 16.10, 16.02 (16.06) | +20 % |
| Flash-Next IQ3_S (84 GB, C:) | 7 | 12.93, 12.91 (12.92) | 17.27, 17.29 (**17.28**) | **+34 %** |
| Qwen3.5-122B UD-IQ2_XXS (34 GB, D:) | 12 | 13.70, 14.05 (13.88) | 16.31, 15.95 (**16.13**) | **+16 %** |
| GLM-5.3-Flash UD-IQ2_XXS (102 GB, C:), one pair | 3 | 2.21 | **4.44** | **2.0x** |
IQ3_XXS stats with the view (13 layers): hit 97.8 %, 3.6 GB read, cache 16.21 GiB, resizes 1 / 0. The launcher's default layer count is the
best of the two tried (19.9 at 9 layers vs 16.1 at 13): no extra GPU layers needed.

**Text**: plugin = stock, byte for byte, on Flash-Next IQ2_XS (1041 characters), IQ3_S (1032), the 122B (1027) and GLM (1148). Granite on the PC with a 0.2 GiB
cache (35929 evictions through the views): identity 4/4, executor 6765 ops checked, 0 mismatches.

**Another program needs memory while the server runs** (C:\dl\pc_pressure.ps1, Flash-Next IQ2_XS, working-set minimum 21.9 GiB):
| Other program | got its memory in | new program starts in | server decode before / during / after |
|---|---|---|---|
| 8 GB (8.2 GB free) | 1.0 s | 17, 7 ms | 22.33 / 23.41 / 23.98 |
| 14 GB (8.4 GB free) | 3.8 s (1.3 s without the server) | 15, 8 ms | 22.35 / 23.67 / 22.88 |
No freeze, no errors: Windows found the memory elsewhere (file cache, page file) and the cache kept its pages.

**Also tonight**: `--moe-cache-mib` (llama.cpp's own GPU expert cache, needed by I28) aborts at load on this Vulkan build even without the plugin
(`ggml-backend.cpp:1089: pre-allocated tensor (moe_cache.0.blk.0.ffn_gate_exps.weight) in a buffer (Vulkan1) that cannot run the operation (NONE)`),
with or without the iGPU and op offload. Cause: each cache bank is one tensor (~2970 MiB at 8909 MiB) and Vulkan refuses tensors over its
~2 GiB storage-buffer limit; `--moe-cache-mib 5900` (~1967 MiB banks) and 1500 load. With 5900 MiB, I28 (hook, plugin loaded by path, branch
claude/gpu-cache-path without the file view), IQ2_XS cold: stock + GPU cache 6.93 tok/s (prefill 3.1), plugin --gpu-cache 9.69 (5.5),
plugin --gpu-cache off 13.71 (12.6); 112948 uploads routed, text = stock. llama.cpp's GPU expert cache loses on this Vulkan PC: I28 parked.

## 60. Windows desktop, 2026-10-27 night: agent sessions (I43 / T8), stock vs plugin, with and without MTP

Same PC as RESULTS 59, experimental 46ed596 (file-view cache), tools\agent_bench_pair.ps1 (fixed in 3a54f00): the recorded OpenCode (5
requests) and Hermes (7 requests) sessions replayed, cold before every run, order stock, plugin, plugin, stock; means of the two runs of
each kind. Both kinds get the launcher's `--agent` arguments (context 65536, --jinja, the same MTP choice). 0 errors in every replay.

| Model | MTP | Session | task s stock / plugin | first token s stock / plugin | follow-up first token (median) s stock / plugin | decode tok/s stock / plugin |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (20 GB, fits in RAM) | on | OpenCode | 27.77 / 26.38 | 13.53 / 13.41 | 1.067 / **0.670** | 72.4 / 72.7 |
| | on | Hermes | 25.34 / 22.95 | 5.20 / 5.11 | 1.061 / **0.846** | 71.5 / 72.2 |
| | off | OpenCode | 31.79 / 29.46 | 12.87 / 10.94 | 1.076 / **0.717** | 43.6 / 43.1 |
| | off | Hermes | 33.29 / 31.03 | 5.03 / 4.89 | 1.048 / **0.809** | 43.7 / 43.3 |
| Flash-Next IQ2_XS (68 GB, bigger than RAM) | (no head) | OpenCode | 174.8 / **73.4** | 116.8 / **38.6** | 7.30 / **2.14** | 21.5 / 21.2 |
| | | Hermes | 163.9 / **81.6** | 66.2 / **21.8** | 8.76 / **3.01** | 21.1 / 21.3 |

- Bigger than RAM (Flash-Next): a whole agent session is **2.0-2.4x faster** with the plugin; the gain is prompt processing (first token 3x,
  follow-up first token ~3x), decode is equal at this context.
- Fits in RAM (Qwen3.6): sessions 5-9 % faster, follow-up first token 20-37 % faster, decode equal.
- MTP (both kinds): decode 43.5 -> 72 tok/s, a session 13-25 % shorter; keep `--mtp` on in agent mode for models with a head.

**GPU expert layers with the file-view cache** (one cold run each, plugin, the 256-token request): the launcher's planner already picks the
best count; one more layer than the planner loses ~23 % (VRAM runs short next to the long-prompt block).
| Flash-Next | GPU expert layers (--n-cpu-moe) | decode tok/s |
|---|---|---|
| IQ2_XS | 8 (40) / 10 (38) / **12 (36, default)** / 14 (34) | 20.94 / 21.31 / **22.12** / 16.91 |
| IQ3_XXS | 5 (43) / 7 (41) / **9 (39, default)** / 11 (37) | 18.68 / 19.06 / **19.82** / 15.27 |

**MTP on GLM-5.3-Flash with the file-view cache** (one cold run, the 256-token request; GLM has an MTP head, blk.45.nextn): stock 2.21,
plugin 4.44, **plugin + `--mtp` 6.92 tok/s (3.1x stock)**, text byte-identical to stock, drafts accepted 61 % / 100 % (two requests).
The cache kept its 20 GiB with free system commit at 3.47 GiB (the MTP draft context's GPU memory is charged to commit); before the file view
the same setup cut the cache to 4.25 GiB and decoded at 0.8-1.0 tok/s (RESULTS 58). MTP on GLM: yes, now.

**Quality, 202 exact-answer tasks** (tools/quality_tasks.json from branch claude/quality-100 ad775f5; `run --tasks-only --no-think
--task-n-predict 128`, greedy; plugin, launcher defaults, file view):
| Flash-Next | tasks right | loops | time | tasks changed vs IQ3_XXS |
|---|---|---|---|---|
| IQ3_XXS (76 GB) | **193 / 202 (95.5 %)** | 0 | 242 s | reference |
| IQ2_XS (68 GB) | 187 / 202 (92.6 %) | 0 | 186 s | 14: 10 right only on IQ3_XXS, 4 only on IQ2_XS |
IQ3_XXS leads by ~3 points; 10 vs 4 changed tasks is a lean, not a proof (paired sign test p ~ 0.18). IQ2_XS is ~10 % faster (21.9 vs
19.9 tok/s, RESULTS 59) and 8 GB smaller. Recommendation (owner's D4): list both; IQ3_XXS when accuracy matters, IQ2_XS for speed.

**llama.cpp threads** (Flash-Next IQ2_XS, file view, one cold run each; the executor pool follows -t): 8: 20.99, 12: 22.81, **16 (default)
22.13**, 24: 21.60, 32: 18.68 tok/s. Flat from 12 to 24 (inside run-to-run noise), SMT threads (32) lose 16 %: the default stays.

**GLM-5.3 GPU expert layers** (file view, no MTP, cold): ABBA decode `--n-cpu-moe 45` (planner default) 5.52 / 5.58 vs 43 (+2 layers)
5.99 / 6.02 (+8 %; single runs had shown 4.44 vs 6.01, 41 layers 4.58). Long prompt (3020 tokens, one cold run each): 45: **54.8 tok/s**,
84 long-prompt GPU blocks; 43: 47.8 tok/s (-13 %), only 2 GPU blocks (no VRAM left for the block). The planner's block reserve trades the
+8 % decode for prompt speed, which is what agent sessions wait on: the default stays.

**Memory budget (`--ram`)** (Flash-Next IQ2_XS, file view, one cold run each): 24 GB 21.81, 20 GB 21.00, 16 GB **failed to load**:
`ggml_vulkan: Device memory allocation of size 717930704 failed ... ErrorOutOfDeviceMemory`. Cause: the launcher put a Windows job memory
limit of `--ram` on llama-server; a job limit caps committed memory, and Windows commits Vulkan's GPU allocations. With the file view the
experts use no commit, so the limit only capped llama.cpp's GPU memory (a Linux cgroup does not count GPU memory). Fix (branch
claude/win-job-limit c20cfb2): no job limit with the file view (kept for MOE_CACHE_WIN_VIEW=0). After the fix: 16 GB **20.03 tok/s**
(cache 11.55 GiB), 92 % of the 24 GB speed.

**Start-up** (Flash-Next IQ2_XS, launcher defaults; cold = after win_flush_cache, warm = restart right after, no flush):
| | ready after | first 64-token request | ready + first answer |
|---|---|---|---|
| stock, cold / warm | 22.9 / 20.4 s | 8.5 s (10.2 tok/s) / 10.5 s (8.7 tok/s) | 31.4 / 30.9 s |
| plugin (file view), cold / warm | 28.5 / 21.0 s | 3.6 s (20.8 tok/s) / 3.5 s (21.0 tok/s) | 32.1 / **24.5 s** |
The plugin's cold start loads the profile's hot experts before it reports ready (+5.6 s), which the first request gets back; a warm restart
finds the experts in the file cache (the view's evicted pages stay there) and answers 6 s sooner than stock.

**Two agents at once on one server** (Flash-Next IQ2_XS, `--agent` arguments, the OpenCode and Hermes replays started together, cold,
ABBA; C:\dl\pc_twoagents.ps1): both sessions done after stock 333.8 / 336.1 s, plugin **145.5 / 145.4 s (2.3x)**; first token stock
177-197 s vs plugin 54-61 s; follow-up first token (median) stock 4.8-10.6 s vs plugin 2.5-3.7 s; 0 errors. The gain holds with two
agents sharing the server.

**Kill check N1 (GPU hot bank chosen by use instead of whole layers)**: Flash-Next IQ2_XS, default 12 GPU layers, stats of the measured
256-token request (11.57 s at 22.12 tok/s): time in the plugin's CPU expert backend 5.02 s (43 %), of which CPU expert compute 4.91 s (42 %),
read waits ~1 % (hit 98.9 %); the other 57 % is the GPU layers, attention and sync. Under the 50 % kill line: even a perfect bank moving the
CPU's share of picks from 75 % to 40 % saves ~20 % of a token (ceiling ~+25 % decode) before its VRAM's system-RAM cost on AMD. Parked.

**KV cache q8_0 (R11)** (Flash-Next IQ2_XS, file view, one cold run each): decode 22.01 tok/s (f16: 22.12); 2950-token prompt 170.6 tok/s
(f16: 164.6). No speed difference at this context; q8_0 halves the KV memory, which only matters at long agent contexts (--fast already uses it).

## 61. Windows desktop, 2026-10-27/28: list A (I32, B13, F5, B11/B12, N13, N8, N3, N5)

Same PC as RESULTS 59-60 (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5, page file 14 GB, Windows 11), llama.cpp Vulkan 620d2225c, launcher 65facd1,
plugin DLL 681f3bc (file view), unless a row says otherwise.

**I32: Vulkan iq2_s vs iq2_xs matrix-vector kernel on RDNA4** (facts for the owner's upstream note, with I25). Stock llama-server, `--device Vulkan1`
(the 9070 XT; the iGPU left out), `GGML_VK_PERF_LOGGER=1`, Flash-Next IQ3_XXS (D:), which mixes expert types of one shape: gate/up of layers
12/15/17/21 (iq2_s), 0/2/4/5 (iq2_xs), 1/3/6/7 (iq2_xxs), 35/36/41/44 (iq3_xxs) on the GPU with `-ot`, the other experts on the CPU; 64 decode
tokens after a 16-token warm-up (C:\dl\pc_i32.ps1, log C:\dl\i32.err.txt). Shape per call: m=640, k=2560, 10 experts of 512 (one token).
| Tensor type | bits/weight | `MUL_MAT_ID_VEC` median (624 calls) | p10 | effective read | vs iq2_xs |
|---|---|---|---|---|---|
| iq2_xxs | 2.06 | 42.75 us | 42.32 us | 99 GB/s | 0.93x |
| iq2_xs | 2.31 | 46.07 us | 36.48 us | 103 GB/s | 1 |
| **iq2_s** | 2.5 | **75.57 us** | 74.40 us | 68 GB/s | **1.64x** (p10 2.04x) |
| iq3_xxs | 3.06 | 73.66 us | 57.96 us | 85 GB/s | 1.60x |
Prompt batches (`MUL_MAT_ID`, 2 / 4 tokens): iq2_s 115 / 142 us vs iq2_xs 41 / 61 us (2.8x / 2.3x).
Verdict: the gap seen on the Arc iGPU (1.9x, I25) is also on AMD RDNA4 (1.64x median, over the 1.3x kill line): iq2_s is as slow as iq3_xxs
with 18 % fewer bits. Not checked: whether 620d2225c changed either shader (that commit is not on the laptop; no fetch).

**B13 / R5: `-ub` 4096 / 8192 vs 2048 with all experts in RAM** (Qwen3.6-35B-A3B, 20 GB, fits: the launcher runs stock llama-server; `--ub N -- -c 20480`,
the planner keeps 15 expert layers on the CPU for all three; cold, ABBA; C:\dl\pc_ub.ps1: a 14710-token and a 2950-token prompt, then the 256-token request)
| `-ub` | 14.7k prompt tok/s (two runs) | mean | 2.95k prompt tok/s | decode tok/s |
|---|---|---|---|---|
| **2048 (default)** | 658.6, 657.8 | **658** | 632.8, 636.1 | 40.86, 40.31 |
| 4096 | 504.8, 513.6 | 509 (-23 %) | 614.0, 616.3 | 41.11, 41.16 |
| 8192 | 421.9, 407.8 | 415 (-37 %) | 580.5, 583.8 | 40.66, 41.06 |
Verdict: bigger micro-batches are slower on this Vulkan PC (same layer split, so it is the batch itself); decode unchanged. The default 2048 stays; killed.

**F5 rest: robustness** (Flash-Next IQ2_XS, plugin with the file view, launcher defaults; C:\dl\pc_f5.ps1, pc_f5_user.ps1; memory from the Windows counters)
| Test | Result |
|---|---|
| kill llama-server 5 s into the 256-token request (cold start before) | client gets "connection closed"; the launcher exits and says the server ended with 0xFFFFFFFF; 3 s after the kill: commit 30.5 -> **14.8 GB** (the idle level), available RAM 8.4 -> 25.5 GB, the expert pages in standby (24.4 GB): no stale working-set minimum (it belongs to the process) |
| restart right after the kill (no flush) | ready after 20.7 s (cold start 28.8 s), decode **22.08 tok/s**, text byte-identical to the reference run, no warnings |
| on-demand, `--idle 1`, two rounds (launcher experimental bbcec55 in C:\moe-cache-pla, own cache dir C:\dl\f5cache so the owner's remembered choice is untouched) | request -> model started, ready after 34 s, first 64-token answer after 39.6 / 39.3 s, decode 19.9 / 21.1; stopped after 1 idle minute both times; commit back to 14.8 GB and available RAM to 25.2 GB each time |
| on-demand listener start | its port opens 16.5 s after start (a request in that window gets "connection refused") |
| as a non-admin user: `runas /trustlevel:0x20000` (Administrators deny-only, the token keeps only SeChangeNotifyPrivilege; a normal desktop session also has SeIncreaseWorkingSetPrivilege) | the working-set minimum is refused (error 1314, privilege not held); the plugin keeps caching: decode **21.58 tok/s**, text identical, working set 18.5 GB (no other memory pressure, so nothing was trimmed). **Bug**: the launcher printed "the plugin loaded but disabled itself ... llama-server runs stock" (it took any plugin line with "refus" as disabled); fixed on claude/pc-list-a 1c94caf |
| page file (read only, not changed) | fixed 14336 MB on C: (not system-managed), in use 1.8 GB, peak 2.96 GB |
Note: the SSH sessions used for every PC run so far are elevated (High Mandatory Level, all admin privileges), including the RESULTS 59 working-set
test labelled "as a normal user"; the run above is the first one without them.

**N8: drive speed probe in `check`** (branch claude/pc-list-a 1c94caf + 80baa9c): 4 s of random 1 MiB reads that bypass the file cache
(FILE_FLAG_NO_BUFFERING / O_DIRECT, 4 threads) on the model's largest file. Same 122B file on both PC drives, and plugin runs from each (cold,
ABBA C, D, D, C, launcher defaults, file view):
| Drive | probe GB/s (3 runs) | plugin decode tok/s | measured request: read / reader time / rate | hit % (measured request) |
|---|---|---|---|---|
| C: Sabrent (PCIe 3) | 3.02, 3.22, 3.21 | 19.78, 19.84 (19.81) | 2.86 GB / 4.03 s / 0.71 GB/s | 97.8 |
| D: SanDisk X3N (PCIe 4) | 4.53, 4.54, 4.53 | 20.31, 20.22 (20.27) | 2.83 GB / 2.97 s / 0.95 GB/s | 97.8 |
| C: / D: | **0.70** | 0.98 | **0.74** | - |
Laptop NVMe: 2.9-3.1 GB/s (drive ceiling 3.6). The probe is repeatable (±4 %) and ranks the drives like the plugin's own reads (0.70 vs
0.74), but at 98 % hits the faster drive gives only +2.3 % decode; the "20-40 % loss" behind N8 was from the private-memory cache era.
Verdict: kept as a `check` line that warns only below 1.0 GB/s (SATA, hard disk, USB); NVMe drives get an OK line with the number.
Also: the 122B decodes at 19.8-20.3 tok/s here vs 16.1 in RESULTS 59 (same DLL and launcher; cause not investigated: hit rate 97.8 % here).

**B11 / B12 / R3: `--hybrid` GPU bank of 6 / 10 GB vs the launcher default** (plugin, file view, cold; order default, 6, 10, 10, 6, default;
`--hybrid` with MOE_CACHE_HYBRID=6144 / 10240 MiB; C:\dl\pc_ub.ps1 -Reps 70: a ~3k-token prompt, then the 256-token request). The bank is chosen
from the usage profile; with `--hybrid` all experts go to the plugin (`--n-cpu-moe 999`) and the output is no longer identical to stock.
| Model | Run | bank | 3k prompt tok/s | decode tok/s (two runs) | decode vs default |
|---|---|---|---|---|---|
| GLM-5.3-Flash (C:) | default (`--n-cpu-moe 45`) | - | 54.8, 55.3 | 5.40, 5.32 (5.36) | - |
| | `--hybrid` 6 GB | 842 experts, 42 layers, 6138 MiB | 54.1, 54.7 | 5.89, 5.87 (5.88) | **+9.7 %** |
| | `--hybrid` 10 GB | capped at free VRAM: 959 experts, 6996 MiB | 53.2, - | 5.70, 5.68 (5.69) | +6.2 % |
| Qwen3.5-122B (D:) | default (`--n-cpu-moe 36`) | - | 259.7, 265.8 | 21.23, 21.14 (21.19) | - |
| | `--hybrid` 6 GB | 2449 experts, 47 layers, 6142 MiB | 189.0, 187.6 | 17.46, 17.54 (17.50) | **-17 %** |
| | `--hybrid` 10 GB | 3935 experts, 9868 MiB | 159.7, 159.1 | 17.54, 17.41 (17.48) | -18 % (prompt -40 %) |
The GLM bank is freed for long prompts and loaded again for decoding, so its prompt speed stays; on the 122B the bank costs prompt speed too.
Verdict: under the +10 % kill line on GLM (and approximate), a loss on the 122B: killed on this PC. For GLM, MTP (+25 % over the plugin, exact,
RESULTS 60) is the better lever.

**N13 / B4 / R2: `-ub` 512-4096 at ~34k tokens, and a launcher bug** (Flash-Next IQ2_XS, plugin with the file view, `--ctx 40960 --ub N`, cold, ABBA;
C:\dl\pc_ub.ps1 -PromptFile C:\dl\long30k.txt: the first 90 KB of this file, 33979 tokens, then the 256-token request).
**Bug**: the launcher ran its tuner (micro-batch, GPU expert layers, VRAM kept free for the plugin's long-prompt GPU block) only with `--ub auto`. A
chosen `--ub N` (or `-- -ub N`) fell back to the plain planner, which fills the card with expert layers (36 at any -ub): every long-prompt
block found 0 MiB free ("needs 1293 MiB ... only 0 MiB are free: it stays on the CPU") and the whole prompt ran on the CPU. Fix
(claude/pc-list-a 1d2d5a8): `tune(..., ub=N)` plans the layers around the chosen micro-batch (512/1024: 38, 2048: 41, 4096: 46 CPU expert
layers at 40960; auto picks 1024 with 38).
| `-ub` | before the fix (launcher 65facd1): prompt tok/s, decode | after the fix (1d2d5a8): prompt tok/s (two runs) | GPU blocks | decode tok/s |
|---|---|---|---|---|
| 512 | 141.3, 23.03 | 215.4, 215.4 | 2546 | 22.31, 22.30 |
| **1024 (auto at this context)** | 121.0, 18.73 | **275.7, 278.8** | 1292 | 22.39, 22.24 |
| 2048 | 132.6, 19.26 | 244.4, 249.2 | 697 | 20.29, 20.20 |
| 4096 | 119.9 / 119.3, 17.53 / 17.67 | 231.6, 232.0 | 414 | 18.05, 18.28 |
Verdict: with the fix every `-ub` is 1.5-2.3x faster at 34k tokens than before; 1024 (what the tuner already picks at this context) is best,
+12 % over 2048 and +28 % over 512; bigger batches cost decode (fewer GPU layers). The default (`--ub auto`) was not affected by the bug.
Not run (wrap-up): MOE_CACHE_GPU_MIN_TOKENS 8 / 16 / 32 for short follow-up prompts (script C:\dl\pc_mintok.ps1 ready).

**Left over from list A** (stopped at the owner's wrap-up, 2026-10-28)
| Item | Done | What remains |
|---|---|---|
| N13 short prompts | - | MOE_CACHE_GPU_MIN_TOKENS 8 / 16 / 32, ABBA (32, 16, 8, 8, 16, 32): `C:\dl\pc_mintok.ps1 -Tag t -MinTok N` (cached ~3k context, 10 cumulative follow-up turns of ~10-60 tokens, prompt ms per turn); ~40 min |
| N3 idle release | design only | measure first: `C:\dl\pc_n3.ps1` (running server, its hard minimum dropped and working set emptied from outside, next request timed vs the on-demand restart: 34 s ready, 39 s to first answer). Build only if the next request is fast: plugin named event Local\moe-cache-idle-<pid> (drop the minimum, empty the working set, ws_min_cur = 0 so the next graph sets it again) + an earlier idle stage in on-demand. Note: an idle on-demand server is already stopped after --idle (memory back, RESULTS 61 F5) |
| N5 serve before the warm start ends | measured cost only | the warm start runs inside llama.cpp's load (a graph before "model loaded"), 3.9-4.0 s for 13-15 GiB here. Measure first: `C:\dl\pc_n5.ps1 -Tag a` vs `-EnvList "MOE_CACHE_PRELOAD_FRAC=0"` (ready + first answer). Build only on a gain: preload in a background thread per chunk with loading flags (shared Linux/Windows) |

**pc-bench: the list-A left-overs and more PC checks** (2026-10-28 afternoon; same PC, llama.cpp Vulkan 620d2225c, plugin DLL 681f3bc unless a row says
otherwise; one job at a time under the PC lock, every run cold; scripts C:\dl\pcb_*.ps1 and C:\dl\pcb\, logs C:\dl\pcb_*.log; part handover/parts/pc-bench.md)

**N13 rest: `MOE_CACHE_GPU_MIN_TOKENS` 8 / 16 / 32 for short follow-up turns** (Flash-Next IQ2_XS, launcher 65facd1 defaults; C:\dl\pc_mintok.ps1: an
~3k-token context cached in slot 0, then 10 cumulative turns; ABBA 32, 16, 8, 8, 16, 32). Prompt ms of each new turn, both runs per value:
| New tokens | 32 (default, as llama.cpp) | 16 | 8 |
|---|---|---|---|
| 14-15 | 410-475 | 433-473 (+0-6 %) | 551-663 (+35-39 %) |
| 21 | 519-527 | 702-712 (+34-36 %) | 690-709 (+31-36 %) |
| 31 | 656-713 | 826-915 (+26-28 %) | 818-904 (+25-27 %) |
| 46 / 63 | 901-1201 | 908-1228 (same) | 913-1271 (same) |
Verdict: **killed**. Below 32 tokens the GPU block (upload of whole expert layers) is slower than the CPU; 46+ tokens go to the GPU at every
setting. The default 32 also keeps the output identical to stock llama.cpp (which moves batches of 32+ tokens to the GPU itself).

**N3: idle release of the working-set minimum** (measure: C:\dl\pc_n3.ps1, Flash-Next IQ2_XS, twice: a running plugin server's hard minimum dropped and
its working set emptied from outside, 30 s idle, then the next request)
| | run 1 | run 2 |
|---|---|---|
| busy: 64-token request, 256-token decode | 3.9 s, 22.01 tok/s | 4.0 s, 21.73 tok/s |
| busy: working set / minimum / available RAM | 18.7 / 22.8 / 6.8 GB | 18.7 / 22.8 / 6.8 GB |
| released: working set / available RAM | 0.0 / **25.0 GB** | 0.0 / **24.9 GB** |
| next 64-token request, then 256-token decode | **4.4 s**, 21.54 tok/s | **4.5 s**, 21.10 tok/s |
(The decode after the release ran without a minimum: the outside release left the plugin thinking its minimum was still set. The build below sets
it again at the next graph.) An on-demand restart instead takes 34 s to ready, 39 s to the first answer (F5). Verdict: **pays: built** (claude/pc-bench
a92a45f): on-demand asks the plugin after `--idle-release` minutes (default 5, 0 = off; before the `--idle` stop) to give the cache's RAM back and
keeps the server. Same logic on Linux and Windows: SIGUSR2 / event `Local\moe-cache-idle-<pid>`; file-view experts (Windows) stay marked resident
(pages to standby, the minimum set again at the next graph), private-memory experts (Linux, `MOE_CACHE_WIN_VIEW=0`) are evicted. Linux check
(Granite, all experts on the CPU): 0.19 GiB released, RSS -190 MB, next answer byte-identical. CI green (identity 40675256728, Windows 40675313675).
PC check of the build (on-demand `--idle-release 0.5`, DLL 570d351 = N3 + the exit-hang fix below, two rounds; C:\dl\pcb\pcb_n3v.py):
| | round 1 | round 2 |
|---|---|---|
| first request (starts the server, 64 tokens) / then 256 tokens | 45.1 s / 20.18 tok/s | 39.5 s / 20.07 tok/s |
| busy: available RAM, working set, minimum | 6.4 GB, 18.3 GB, 20.86 GiB | 6.6 GB, 18.5 GB, 21.24 GiB |
| after 30 s idle the listener asks; plugin: "idle: N GiB of experts given back" | 17.39 GiB | 17.55 GiB |
| idle: available RAM, working set | **24.4 GB**, 0.0 GB | **24.3 GB**, 0.0 GB |
| next request (256 tokens) after the release | 14.5 s, 18.67 tok/s, text identical | 14.6 s, 18.46 tok/s, text identical |
| the request after that (no release) | 11.8 s, **22.37 tok/s** | 11.9 s, **22.12 tok/s** |
The minimum is set again at the next graph (20.38 / 20.76 GiB); the first request after a release pays ~2.7 s while its pages come back from the
standby list (no SSD reads: nothing else needed the memory), against ~40 s for a restart. Built and merged.

**N5: serve while the warm start runs?** (C:\dl\pc_n5.ps1, Flash-Next IQ2_XS, ABBA: default, `MOE_CACHE_PRELOAD_FRAC=0`, same, default)
| | ready | first 64-token request | ready + first | the next 256-token request |
|---|---|---|---|---|
| warm start (default; 14.7 GiB in 4.1-4.2 s) | 43.4, 43.1 s | 4.0, 3.9 s | **47.4, 47.0 s** | 21.70, 21.56 tok/s |
| no warm start | 39.0, 39.1 s | 5.5, 5.5 s | **44.5, 44.6 s** | 18.87, 18.93 tok/s (-13 %) |
Verdict: **not built**. Serving during the warm start can save at most ~2.7 s of a 47 s cold start (6 %), and the first requests would then run on a
cold cache (the next request after a skipped warm start decodes 13 % slower, which costs more than it saves).

**T2b: why the 122B ran 16.1 tok/s in RESULTS 59 and 19.8-21.2 in list A** (same DLL 681f3bc, launcher 65facd1, scripts pc_iq3_stats.ps1 / pc_stock.ps1,
D:). Logs first, then one ABBA: plugin, plugin + 2.8 GB held by C:\dl\hog.exe (RESULTS 59's smaller free memory), stock, stock, plugin + hog, plugin.
| Measured 256-token request | RESULTS 59 (2026-10-27 18:41) | list A N8 (23:47) | today plugin | today plugin + 2.8 GB hog |
|---|---|---|---|---|
| memory limit the plugin saw (after load) | 21.69 GiB | 24.87 GiB | 25.30, 25.41 | 22.66, 22.65 |
| misses / evictions | 5034 / 3129 | 3228 / 0 | 3048 / 0 | 4127 / 1446 |
| plugin read wait / CPU expert compute | 5.27 / 7.95 s | 2.97 / 6.53 s | 2.84 / 6.51 s | 3.63 / 6.70 s |
| time outside the plugin (GPU layers, attention) | 7.21 s | 5.86 s | 6.24 s | 6.27 s |
| decode tok/s | 16.31 (pair 16.13) | 20.31 | **19.77, 19.69** | **19.22, 19.25** |
| stock in the same session | 13.70, 14.05 | - | **15.88, 16.02** | - |
Explained: the smaller free memory of that evening is real but costs only **~3 %** (19.24 vs 19.73: more misses, 1446 evictions). The rest was a
slowdown of the whole PC during the 122B pair at 18:40-18:44 on 2026-10-27: the same CPU expert work took 22 % longer, the GPU part 16 % longer, and
stock was 13 % slower too. The Flash-Next IQ2_XS pair a few minutes earlier in the same series matches today's speed (21.94 then, 22.0 today). The
cause of that slowdown is not in the logs (nothing else of ours wrote files in that window). **Correction to RESULTS 59**: the 122B is **+24 %**
over stock on this PC (today's ABBA: stock 15.95, plugin 19.73), not +16 %.

**Exit hang (found by R11): llama-perplexity with the plugin never exited on Windows; fixed** (claude/pc-bench 570d351; details handover/parts/pc-bench.md)
R11's first run (DLL 681f3bc) finished its work at 15:22 and then stayed inside ExitProcess for good: one thread left in a user-mode wait, 23.5 GB
working set, the process could not be killed (Stop-Process, taskkill /F); only a reboot cleared it. ExitProcess ends every other thread before the
DLL detach runs the static state_t destructor, which then locked mutexes, notified condition variables and joined threads that were already gone
(code-audit row 32). It never showed before because llama-server is always killed. Fix: the state lives on the heap behind a static guard; when the
process is exiting on Windows (RtlDllShutdownInProgress) the guard only saves the profile (when nothing holds `mu`) and prints the summary, with no
lock, notify or join; otherwise (threads alive: Linux exit, FreeLibrary) the old teardown runs. Same rule on both systems: never join a thread the
system has ended. CI green (identity 40686766503, Windows 40686766623, sanitize 40686792661).
| Check (PC, Granite 1B, llama-perplexity, 1 chunk, 0.2 GiB cache; C:\dl\pcb_exit.ps1) | exit | cache | profile saved at exit | summary line |
|---|---|---|---|---|
| no plugin | 3.2 s | - | - | - |
| DLL 570d351, twice | **2.1 s, 1.7 s** | on | yes | yes |
| DLL 681f3bc (R11, Flash-Next) | never (reboot) | on | no | no |
Linux (laptop, Granite, 0.2 GiB cache, 99,278 evictions): llama-perplexity and llama-completion exit, profile saved, logits identical.

**I42 PC check: the control page's "Compare with the other side" vs the pc_speed.ps1 method** (Flash-Next IQ2_XS, page and launcher at 7c47642, DLL
681f3bc, both cold; ABBA: page with moe-cache first, script stock, plugin, plugin, stock, page with stock first; C:\dl\pcb\pcb_i42.py. The script times
three 256-token prompts, the page the first two, both with `ignore_eos`)
| | page (2 runs) | script, the same two prompts | script, mean of its 3 prompts |
|---|---|---|---|
| moe-cache decode tok/s | 21.57, 20.92 (21.25) | 21.10, 21.43 (21.27) | 21.30, 21.55 |
| stock decode tok/s | 16.38, 16.26 (16.32) | 16.59, 16.57 (16.58) | 17.75, 17.69 |
| moe-cache prefill tok/s | 294.1, 307.2 (3053 tokens) | 282.7, 284.8 (3533 tokens) | - |
| stock prefill tok/s | 88.8, 84.0 | 101.7, 108.7 | - |
| speed-up shown | **1.30x** | 1.28x | 1.21x |
Verdict: **decode within 0.1 % (moe-cache) and 1.6 % (stock) of the script on the same prompts: the page's number stands.** Stock's first 256 tokens
run on a cold file cache (13.0 tok/s) and its next ones on a warmer one (19.5-20.2), so the script's third prompt lifts stock's mean and the ratio
(1.21x). Prefill: moe-cache within 6 %, stock 17 % lower on the page (shorter prompt, colder cache): label the page's prefill "indicative".

**N9 PC check: two models in on-demand, does the file cache keep the other one warm on 32 GB?** (Qwen3.6-35B, 21.7 GB, + Flash-Next IQ2_XS, both on C:;
on-demand of 7c47642 with its own cache dir; cold file cache at the start of each round, then A B A B A, every request after the first a swap;
time = request sent to the 32-token answer; C:\dl\pcb\pcb_n9.py)
| | round 1 | round 2 |
|---|---|---|
| Qwen3.6 cold / after a swap back | 30.5 / 31.1, 31.1 s | 30.5 / 31.2, 30.7 s |
| Flash-Next cold (first swap) / after a swap back | 45.5 / 46.7 s | 45.9 / 46.2 s |
| standby (normal priority) while Qwen3.6 / Flash-Next runs | 10.8-10.9 / 7.8-8.0 GB | 10.9-11.0 / 8.2-8.7 GB |
Verdict: **killed**: a swap back costs as much as a cold start (+0-2 %). With one model's cache and working set in RAM only 8-11 GB of standby is
left, less than either model, so the other model's pages are gone by the time it is asked for again. Two models in one endpoint still work (N9 is
a usability feature); each swap is a full start.

**N16 gate: upload speed of the long-prompt GPU block** (idea-scout; Flash-Next IQ2_XS, launcher 7c47642 `--ctx 40960` (auto `-ub` 1024), DLL 681f3bc,
`MOE_CACHE_SEG_TIMES=1`, the 33,979-token prompt twice, cold then again; C: disk reads sampled every second; C:\dl\pcb_n16.ps1)
| | cold | again |
|---|---|---|
| prompt tok/s, wall | 267.6, 127.0 s | 283.0, 120.1 s |
| GPU blocks, bytes uploaded | 1292, 689.3 GB | 1292, 689.3 GB |
| upload / compute / read back (cumulative) | ~29.1 / 7.4 / 8.3 s | ~25.8 / 7.4 / 8.1 s |
| upload speed | **23.7 GB/s** | **26.7 GB/s** |
| C: disk reads during the prompt | 18.1 GB (~5.8 s at 3.1 GB/s) | 11.8 GB |
Verdict: **N16 killed** (its line: upload already over 18 GB/s): the copy runs near what PCIe 4.0 x16 carries, and the drive is ~20 % of the upload
time. The blocks are ~35 % of the prompt's wall time; their read back takes 18 % of the block time (6.3 ms per block for ~10 MB of activations,
~1.6 GB/s), the slowest part relative to its size.

**N15 gate: the GPU <-> plugin hand-off per CPU expert layer** (idea-scout's handoff_bench.cpp, built with mingw against the ggml 0.27 headers, run with
the PC's llama.cpp DLLs, 3000 iterations, two repeats, medians; A = today's scheduler path, B = async reads + one sync, C = floor, D = B + async copy
back from the GPU's pinned host buffer)
| Device | A | B | C | D | A - B | A - D |
|---|---|---|---|---|---|---|
| RX 9070 XT | 209.7, 211.8 us | 136.1, 139.2 | 64.3, 64.8 | 149.7, 150.9 | **73 us** | 60 us |
| Radeon iGPU | 162.1, 165.9 | 118.2, 112.1 | 55.4, 52.6 | 59.6, 62.5 | 49 us | 103 us |
GPU kernel time per decoded token (stock, `--n-cpu-moe 36`, GGML_VK_PERF_LOGGER=1, 64-token request): **20.8 ms** in 2981 kernel calls (2487 of them
under 10 us, 7.2 ms together); dense matrix-vector products 3.38 GB of weights in 11.8 ms (286 GB/s, the card's peak is ~640): bf16 hyper-connection
matrices 388-508 GB/s, iq4_xs ones 260-315 GB/s. Verdict (idea-scout, ledger 5k): on the 9070 XT the best mode is async reads alone (B), 73 us x 36
layers = ~2.6 ms of a ~45 ms token, **~+6 %**; the GPU kernels are ~79 % of the ~26 ms outside the plugin.

**R11: `--fast` with a q8_0 KV cache on the PC: KL against f16 KV** (llama-perplexity with the plugin, Flash-Next IQ2_XS, `--n-cpu-moe 38`, `-fa on`,
batch 8 (every step through the executor, as in decode), 16 chunks of 512 tokens of C:\dl\long30k.txt; base logits f16 KV with DLL 681f3bc, the
runs with DLL 570d351 (same compute, plus the exit-hang fix); C:\dl\pcb_r11.ps1, ~4.6 min per run)
| Run | KV cache | routing | Mean KLD | 99 % KLD | Same top token | PPL (base 6.2143) |
|---|---|---|---|---|---|---|
| control | f16 | exact | **0.000000** | 0.000037 | 100.000 % | - |
| q8 KV alone | q8_0 | exact | **0.0120** | 0.1114 | 95.12 % | 6.2323 |
| `--fast` as shipped | q8_0 | skip (`MOE_CACHE_ROUTE=skip`) | **0.0145** | 0.1271 | 94.56 % | 6.2302 |
| laptop, q8 KV alone (RESULTS 53, CUDA) | q8_0 | exact | 0.0045 | - | 98.2 % | - |
Verdict: **over the 0.01 kill line on this PC**: the q8_0 KV cache alone costs KL 0.0120 on Vulkan (RDNA4), 2.7x the laptop's CUDA figure, with no
run-to-run noise (the f16 control is 0 to six places, every top token the same). The skipping that `--fast` adds is small here (+0.0025: 98 %
hits). Since the same default must hold on Linux and Windows, the choice is to drop q8_0 KV from `--fast` everywhere (it stays one `-- -ctk q8_0
-ctv q8_0` away, and agent mode then picks a smaller `-ub` at 64k, I6) or keep it with this number in the docs. **Owner's decision (2026-10-28): keep
q8_0 KV in `--fast`** (it saves about half the KV memory, ~0.8 GB at a 64k Flash-Next context; `--fast` is the approximate mode anyway); no code change.

## 62. GLM max on the Windows PC (pc-glm, 2026-10-28): drive, GPU layers, MTP, long prompts, and why code is slow

PC as RESULTS 59-61 (7950X, RX 9070 XT 16 GB, 32 GB). GLM-5.3-Flash UD-IQ2_XXS (102 GB, 4 shards) copied C: -> D: (robocopy 30 s;
sha256 of the 4 copies = the download's). llama.cpp Vulkan 620d2225c, launcher C:\moe-cache-pla (1d2d5a8, launcher code = experimental),
plugin DLL 681f3bc (file view). Script C:\dl\glm\glmrun.ps1: cold (win_flush_cache), ready time, 64-token warm-up, then two greedy requests
with stats deltas per request: **water** = the usual 256-token "Describe the water cycle step by step for a ten-year-old.", **code** =
192 tokens of "Write a Python function that reads a CSV file and prints the average of every numeric column, then explain ...".

**Desk estimate first** (handover/parts/pc-glm.md): reads were ~half of a token at 90 % hits, so the D: drive was expected to give +15-20 %,
+2 GPU layers +8 %, MTP +25 % on predictable text and ~0 below ~65 % draft acceptance; ceiling ~10 tok/s if every read were free.

**Memory confound found and removed.** With the owner's desktop apps open (Steam, Discord, Claude) the auto memory limit moved between runs
(20.8 / 22.6 GiB; last night 25.7) and the cache with it. Sweeps below use a fixed `--ram 22` (20.5 GiB); the final pair uses auto memory
after the owner closed the apps.

**Drive, fixed `--ram 22`, ABBA (C, D, D, C), launcher default 45 CPU expert layers, no MTP**
| Drive | water tok/s | code tok/s | ready |
|---|---|---|---|
| C: Sabrent (3.1 GB/s) | 4.16, 4.19 (4.18) | 2.33, 2.32 (2.33) | 47.8, 47.4 s |
| D: SanDisk (4.5 GB/s) | 4.35, 4.30 (4.33) | 2.63, 2.62 (2.63) | 36.8, 36.8 s |
| D: vs C: | **+3.6 %** | **+13 %** | **-23 %** |
The more a request misses, the more the drive matters: water (72 % hits) +3.6 %, code (57 % hits, 1.2 GB read per token) +13 %.

**Settings on D:, fixed `--ram 22`, mirrored order (each setting twice, except MTP draft 1)**
| CPU expert layers | MTP | water tok/s | code tok/s | water hit % | draft acceptance water / code |
|---|---|---|---|---|---|
| 45 (default) | off | 4.35, 4.30 | 2.63, 2.62 | 72.4 | - |
| 45 | on (draft 2) | 4.76, 5.33 | 2.62, 2.67 | 60.3, 66.5 | 100 % / 79 % |
| 45 | on (draft 1) | 4.48 | 2.66 | 65.8 | 100 % / 85 % |
| **43** | off | **5.06, 5.10** | **2.83, 2.85** | 78.3 | - |
| 43 | on | load abort once (`GGML_ASSERT(ctx->mem_buffer != NULL)` creating the MTP draft context), 3.94 once | 2.48 | 48.1 | 100 % / 78 % |
| 41 | off | 3.82, 3.87 | 2.66, 2.66 | 63.0 | - |
| 41 | on | load abort twice (same assert) | - | - | - |
MTP's draft context costs memory that the cache needed (water hits 72 -> 60-67 %): on water it gains +10-23 % (two runs 12 % apart) at 100 %
acceptance, on code 0 % at 79 %. MTP with 2 more GPU layers does not fit (aborts or squeezes the cache to 48 % hits; llama-server also
logged "failed to allocate memory for prompt cache state"). 43 without MTP is the best decode, and repeatable.

**Best vs default with launcher auto memory** (apps closed: 25.7 GB available, memory limit 25.6-25.8 GiB, start budget ~19.2 GiB), D:, ABBA,
each run: water, code, then a 3020-token prompt
| CPU expert layers | water tok/s | code tok/s | 3k prompt tok/s | water hit % |
|---|---|---|---|---|
| 45 (default) | 5.89, 5.89 (5.89) | 2.80, 2.79 (2.80) | 67.2, 67.9 (67.6) | 88.8 |
| 43 | 6.35, 6.21 (6.28) | 3.05, 3.02 (3.04) | 58.8, 59.3 (59.1) | 89.9 |
| 43 vs 45 | +6.6 % | +8.6 % | **-12.6 %** | |
Same trade as RESULTS 60 (+8 % decode, -13 % long prompt). One agent turn of 3k prompt tokens + 300 output tokens: 45 = 44.4 + 50.9 = 95 s,
43 = 51.1 + 47.8 = 99 s: break-even to slightly worse for agents, better for chat.

**Why code is slow: is the cache still learning, or is code's set too big?** (`--ram 22`, D:, default layers; one server: code A, a
different code request B (JavaScript class), code A again; two runs)
| Request | decode tok/s | hit % |
|---|---|---|
| code A | 2.66, 2.65 | 57.0, 57.1 |
| code B | 2.32, 2.34 | 49.9, 50.3 |
| code A again | 2.62, 2.66 | 56.2, 56.8 |
The same request hits no better the second time: GLM's code working set is larger than a 14-19 GiB cache, which cycles within one request.
A per-mode starting profile ("code mode" preloading code experts) would only change the start state: no gain for GLM on 32 GB.

**Verdict.** Best GLM setting on this PC: the model on the faster drive (D:), launcher default layers, no MTP for code/agent use
(water 5.89, code 2.80 tok/s; stock was 2.21 on C: in RESULTS 59, not re-run on D:). `--n-cpu-moe 43` is +7-9 % decode for -13 % long prompts: worth it for chat, not for agents.
Launcher default: **no change** (the planner's long-prompt reserve is right for agent sessions; a chat-only mode could take 43, same rule on
Linux). MTP stays as is (on in agent mode): it does not lose on GLM (code +1 %, water +10-23 %), but it gives nothing on code; a small
follow-up: the launcher should not let MTP combine with extra GPU layers on a 16 GB card (it aborts at load). *Done (fixes, 2026-10-28)*: llama.cpp logs
"failed to fit params to free device memory" at that load (the card is over-committed; on Windows the spill then starves the host allocation of the
draft context); the launcher now warns when `--n-cpu-moe` is below the planned number while MTP is on (the planned 45 loads with MTP).

## 63. Flash-Next quant choice on the Windows PC (pc-glm, 2026-10-28): Q2_0, IQ3_S, Coder IQ1_M vs IQ2_XS / IQ3_XXS

Same PC, plugin with launcher defaults (C:\moe-cache-pla 1d2d5a8 launcher, DLL 681f3bc file view), cold runs. Owner's rule: a slower file must
earn it with clearly better quality; testing kept minimal (UD-IQ4_XS and UD-Q4_K_XL dropped after quant-research's desk review: both predicted
under ~14 tok/s with at most ~4 tasks of headroom on this set; no KL run). Downloads: Q2_0 shard 1 and Coder IQ1_M shard 1 from Hugging Face
(sha256 = LFS oid); the second shard of Q2_0, Coder IQ1_M and IQ2_XS is **the same file** (sha256 316b46f3...), copied locally.

**Quality, 202 exact-answer tasks** (as RESULTS 60: `quality_yardstick.py run --tasks-only --no-think --task-n-predict 128`, greedy; 0 loops in every run).
Paired sign test on the tasks that changed (two-sided, exact):
| File | tasks right | vs IQ2_XS: right only here / only in IQ2_XS, p | vs IQ3_XXS: p |
|---|---|---|---|
| IQ2_XS (68.0 GB) | 187 | - | 4 / 10, p 0.18 |
| IQ3_XXS (75.8 GB) | **193** | 10 / 4, p 0.18 | - |
| IQ3_S (83.6 GB) | 192 | 8 / 3, p 0.23 | 3 / 4, p 1.00 |
| Q2_0 (66.4 GB) | 191 | 8 / 4, p 0.39 | 4 / 6, p 0.75 |
| Coder IQ1_M (58.4 GB, 256 of 512 experts) | 189 | 6 / 4, p 0.75 | 1 / 5, p 0.22 |
No difference between any two files is proven (all p > 0.05): this set separates them only by a few tasks.

**Q2_0 vs IQ2_XS speed** (cold, mirrored order IQ2_XS, Q2_0, Q2_0, IQ2_XS; both get 36 CPU expert layers, -ub 2048; auto memory ~23 GiB)
| File | water decode tok/s | code decode tok/s | 3k prompt tok/s | 34k prompt tok/s (`--ctx 40960`) | ready |
|---|---|---|---|---|---|
| IQ2_XS | 21.05, 20.96 (21.01) | 21.57, 21.51 (21.54) | 251.0, 255.4 (253) | 228.0, 256.4 (242) | 45.7 s (C:) / 33.7 s (D:) |
| Q2_0 | 21.95, 22.36 (22.16) | 22.40, 22.91 (22.66) | 283.4, 278.2 (281) | 283.3, 291.5 (287) | 33.6 s |
| Q2_0 vs IQ2_XS | **+5.5 %** | +5.2 % | **+11 %** | **+19 %** | |
Caveat: in the decode / 3k pair Q2_0 was on D: and IQ2_XS on C: (reader time 2.9 vs 3.6 s per request, so part of the +5 % decode may be the
drive); the 34k pair had both on D:. Hit rates equal (98.4-98.8 %). The model card's +33 % decode / 3.4x prompt (GPU-only setup) does not
reproduce here: decode is CPU-bound on this PC, as quant-research predicted (0 to +7 %).

**Stock vs plugin on Q2_0** (the owner's new default Flash-Next file; D:, cold; stock = launcher `--stock` defaults, two runs 18:39-18:43;
plugin = the two Q2_0 runs above)
| Q2_0 | water decode tok/s | code decode tok/s | 3k prompt tok/s |
|---|---|---|---|
| stock llama.cpp | 18.26, 18.55 (18.41) | 20.13, 19.88 (20.01) | 120.1, 121.5 (120.8) |
| moe-cache | 21.95, 22.36 (22.16) | 22.40, 22.91 (22.66) | 283.4, 278.2 (281) |
| plugin vs stock | **+20 %** | **+13 %** | **2.33x** |
Stock decodes Q2_0 much faster than IQ2_XS (18.4 vs 14.9 tok/s, RESULTS 59), so the plugin's decode gain is smaller on Q2_0 (+20 % vs +47 %);
the prompt gain (2.3x) is what agent sessions wait on.

**Coder IQ1_M in an agent session** (OpenCode recording, one plugin run, cold; vs RESULTS 60's IQ2_XS plugin runs 74.04 / 72.84 s)
| File | task s | first token s | follow-up first token (median) s | decode tok/s |
|---|---|---|---|---|
| IQ2_XS (RESULTS 60, 2026-10-27) | 73.4 | 38.6 | 2.14 | 21.2 |
| Coder IQ1_M | **55.5 (-24 %)** | 23.7 | 1.51 | 21.24 |
More than 10 % apart, so no ABBA (owner's rule). Its server took ~150 s to come up (IQ2_XS ~60 s): not looked into. *Later (N21)*: that was the benchmark's own wait before it starts, plus the old launcher's slow header reads (fixed in cbe46b0);
with experimental's launcher the Coder server is ready 18.6 s after start (launcher 4.6 s, llama.cpp's load 10.8 s; pc-bench, timestamped), and
the Q2_0 server 20.6-22.7 s (33.6 s with the old launcher).

**Table for the owner** (decode: today's water request, or RESULTS 59 ABBA where marked)
| File | GB | decode tok/s | 3k prompt | tasks /202 | vs IQ2_XS | vs IQ3_XXS | verdict |
|---|---|---|---|---|---|---|---|
| Q2_0 | 66.4 | 22.16 | 281 | 191 | +1.2 tok/s (+5 %), +4 tasks (not proven) | +2.3 tok/s, -2 tasks (not proven) | **worth it: new speed pick**, replaces IQ2_XS: not slower anywhere, prompts 11-19 % faster, smaller, same or better tasks |
| IQ2_XS | 68.0 | 21.01 (R59: 21.94) | 253 | 187 | - | | superseded by Q2_0 |
| IQ3_XXS | 75.8 | R59: 19.86 | - | 193 | -2 tok/s (-9 %), +6 tasks (not proven) | - | **keep as the accuracy pick** (most tasks right; costs ~10 % decode) |
| IQ3_S | 83.6 | R59: 17.28 | - | 192 | | -2.6 tok/s (-13 %), -1 task | **not worth it**: slower and bigger than IQ3_XXS for no gain (rule needed +5 tasks) |
| Coder IQ1_M | 58.4 | 21.24 (agent) | - | 189 | agent session -24 % | -4 tasks (not proven) | **worth it for coding agents only** (half the experts; general knowledge not checked beyond these tasks) |
| UD-IQ4_XS / UD-Q4_K_XL | 93.7 / 111.3 | not run | | | | | dropped (desk: under ~14 tok/s, little room to gain on this set) |

**Addendum (session fixes, 2026-10-28 evening): N20 exact faster Q2_0 kernel, and N20b executor forced.** llama.cpp b11669 and master have their
own AVX2 Q2_0 dot (#30447), so the plugin's old kernels (exact to the plain C loop) stepped aside, on Linux too. claude/q2_0-windows adds a
kernel with #30447's integer lanes and multiply-add order that takes the activation's lane sums from a column prepared once (idea-scout's
prototype): bit-identical, 1.25x per dot in cache (tests/kernel_check.cpp: 0 mismatches in 21,000 trials). At start the plugin compares the
CPU library's Q2_0 function with the plain loop and with a #30447 copy and uses the matching kernel (PC: "#30447", column-sum kernel chosen);
the executor kernels now run on Windows too (only the LD_PRELOAD override stays Linux-only). PC, Flash-Next Q2_0 on D:, cold, launcher defaults:

| Run (mirrored) | water decode tok/s | code decode tok/s | 3k prompt tok/s |
|---|---|---|---|
| old DLL (experimental 8cc6c7c) | 22.54, 22.53 | 22.85, 22.97 | 282.0, 285.7 |
| new DLL (column-sum kernel) | 22.15, 22.32 | 22.63, 22.66 | 284.4, 272.7 |
| new DLL, executor default (MOE_CACHE_EXEC=1) | 22.66, 22.35 | 22.93, 22.80 | 285.9, 276.3 |
| new DLL, executor forced (MOE_CACHE_EXEC=2) | 22.34, 22.40 | 22.77, 22.70 | 276.3, 284.0 |

Text identical in all pairs (temperature 0); `MOE_CACHE_EXEC_CHECK=1`: 0 mismatches in 55,404 executor ops. The executor ran the same 55,404
ops with and without forcing, so it already computes the decode expert ops; the faster kernel shows no gain because decode's expert dots
stream their weights from RAM (bandwidth-bound) while the 1.25x was measured with data in cache. **Verdict: N20 kept (exact, same rule on
Linux and Windows, no speed change on the PC); N20b killed (forcing the executor changes nothing).**

## 64. Long context on the Windows PC (pc-bench, 2026-10-28 evening): 64k and 128k+ on Flash-Next Q2_0, the 122B and Qwen3.6

Owner: the main use is 64k-256k context. Same PC (7950X, RX 9070 XT 16 GB, 32 GB), llama.cpp Vulkan 620d2225c, launcher experimental 28b022b
(`server MODEL --agent --no-warm --ctx N`, own cache dir with the usage profile copied), plugin DLL built from experimental f478564 (exit-hang fix,
Windows executor kernels). Text: the moe-cache repo's docs and code concatenated (C:\dl\pcb\lc_text.txt), fed as appended chunks into slot 0
(`cache_prompt`, 4k, then 16k each) so every chunk's prefill is timed; decode = 128 greedy tokens at that depth. Each point **one cold run** (no
ABBA: single); free RAM is "available" from Windows; Steam was running on the PC. Driver C:\dl\pcb\pcb_lc.py, log C:\dl\pcb_lc.log.

**Planner choices** (dry runs, all models; no load failed at any setting)
| Model | `--ctx 65536` | `--ctx 262144` |
|---|---|---|
| Flash-Next Q2_0 (D:) | 39 expert layers on the CPU, `-ub 1024` | all 48 on the CPU, `-ub 512` |
| Qwen3.5-122B IQ2_XXS (D:) | 39 on the CPU, `-ub 1024` | all 48 on the CPU, `-ub 512` |
| Qwen3.6-35B (C:, fits: stock llama-server) | `--n-cpu-moe 22`, `-ub 1024` | `--n-cpu-moe 38`, `-ub 512` |
Stock llama-server, Flash-Next Q2_0, `-ngl 99 --n-cpu-moe 48`, Vulkan1 buffers (MiB; `-lv 4`): 256k f16 KV 6144 + 1536 (indexer), q8_0 3264 + 816,
recurrent state 450; compute 1403 (`-ub 512`), 2983 (1024), 3979 (2048); 128k `-ub 2048` compute 3040 (f16) / 3072 (q8_0). All loaded.

**Prefill (tok/s) by depth and decode at depth**
| Point | ready | prefill: 0-4k / 4-21k / 21-37k / 37-53k / 53-63k / beyond | decode at 4k | decode at ~63.5k | decode deeper |
|---|---|---|---|---|---|
| Flash-Next Q2_0, moe-cache, 64k | 19.2 s | 231 / 266 / 215 / 203 / 195 | 22.89 (hits 98.5 %) | **20.65** (97.0 %) | - |
| Flash-Next Q2_0, **stock**, 64k | 16.2 s | 72 / 171 / 125 / 112 / 106 | 21.38 | **16.83** | - |
| Flash-Next Q2_0, moe-cache, 256k setting | 15.1 s | 143 / 142 / 112 / 109 / 111 / 105, 113, 108 up to 138k | 18.52 (97.0 %) | - | **15.93 at 138k** (95.3 %) |
| Qwen3.5-122B, moe-cache, 64k | 19.2 s | 209 / 266 / 255 / 237 / 217 | 19.69 (99.1 %) | **17.26** (95.3 %) | - |
| Qwen3.6-35B (stock, fits), 64k | 16.2 s | 477 / 848 / 814 / 792 / 739 | 67.21 | **59.09** | - |
| *Added from the overnight chain (RESULTS 67, same driver and builds, cold, single runs; 128k points: the 5th chunk ends at 68.5k, then 85k / 101k / 118k / 130.6k)* | | | | | |
| Flash-Next Q2_0, moe-cache, 128k (46 CPU layers, `-ub 2048`) | 16.2 s | 197 / 221 / 197 / 191 / 179 / 170, 172, 170, 161 up to 130.6k | 18.97 (96.6 %) | - | **15.27 at 130.6k** (93.9 %) |
| Qwen3.5-122B, moe-cache, 128k (47 CPU layers, `-ub 2048`) | 15.2 s | 190 / 215 / 209 / 198 / 193 / 180, 187, 180, 176 up to 130.6k | 15.54 (95.9 %) | - | **11.27 at 130.6k** (89.3 %) |
| Qwen3.6-35B (stock, fits), 128k | 14.1 s | 474 / 936 / 1014 / 955 / 895 / 798, 812, 704, 697 up to 130.5k | 55.08 | - | **45.76 at 130.5k** |
| Qwen3.5-122B, moe-cache, 256k setting | 15.1 s | 113 / 115 / 104 / 103 / 103 / 97, 99, 96, 97 up to 138k | 13.92 (93.3 %) | - | 12.65 at 138k (93.6 %); **timed out at 35 min** after the 138k save (most likely the whole prompt processed again, finding 5) |
| Qwen3.6-35B (stock, fits), 256k setting (`-ub 512`) | 13.1 s | 182 / 263 / 260 / 266 / 255 / 259, 249, 234, 235 up to 138k, 217-225 up to 227.8k | 49.03 | - | 35.75 at 138k, **41.74 at 227.8k** |
| GLM-5.3-Flash IQ2_XXS, moe-cache, 64k | 26.1 s | 35.4 / 35.2 (to 20.3k), then **timed out at 35 min** (15 min of it waiting for a cool-down break) | 2.08 (32.7 %) | - | - |
Whole 63.5k prompt: Flash-Next moe-cache ~289 s vs stock ~525 s (**1.8x**); decode at 63.5k **+23 %** (20.65 vs 16.83). Free RAM during the runs:
2.8-4.6 GB (moe-cache), 0.3 GB (stock: the file cache fills the rest), 10.5-14.6 GB at ready.

**Saving the agent's conversation (N27: llama-server's `/slots/0?action=save`; an 8-token request on slot 1 sent 0.3 s into the save)**
| Point | depth | save | file (incl. context checkpoints) | the 8-token request |
|---|---|---|---|---|
| Flash-Next Q2_0 64k | 63.5k | 2.4 s | 2.78 GB (6 checkpoints, 675 MiB) | 3.6 s (waits for the save) |
| Qwen3.5-122B 64k | 63.5k | 1.9 s | 2.66 GB (894 MiB) | 2.4 s |
| Qwen3.6 64k | 63.5k | 1.7 s | 1.90 GB (377 MiB) | 2.2 s |
| Flash-Next Q2_0 256k setting | 138k | 5.5 s | 5.31 GB | 6.6 s |
| Flash-Next Q2_0 128k (RESULTS 67) | 130.6k | 5.2 s | 5.08 GB (8 checkpoints, 901 MiB) | 6.1 s |
| Qwen3.5-122B 128k (RESULTS 67) | 130.6k | 4.0 s | 4.62 GB (1193 MiB) | 4.9 s |
| Qwen3.6 128k (RESULTS 67) | 130.5k | 3.0 s | 3.54 GB (503 MiB) | 3.2 s |
| Qwen3.5-122B 256k setting (RESULTS 67) | 138k | 4.0 s | 4.81 GB | 5.1 s |
| Qwen3.6 256k setting (RESULTS 67) | 138k | 3.1 s | 3.71 GB | 3.2 s; restoring this file later: 1.7 s |

**Findings**
| # | Finding | Next |
|---|---|---|
| 1 | At the 256k setting the planner moves every expert layer to the CPU and drops `-ub` to 512 (f16 KV + long-prompt block reserve): Flash-Next prefill halves (143 vs 231-266 tok/s at the same 4-21k depth) and decode at 4k is 19 % lower (18.52 vs 22.89) than at the 64k setting | fixes: N26 (KV from the header: the tuner's flat 40 KB/token over-reserved; corrected) and N29 (`-ub 1024` without the block at 128k+): ABBA queued (below) |
| 2 | Decode falls with depth: Flash-Next -10 % at 63.5k, -14 % from 4k to 138k at the 256k setting; 122B -12 % at 63.5k; Qwen3.6 -12 % at 63.5k | expected (attention + indexer); the plugin's share is unchanged (hits 95-97 %) |
| 3 | moe-cache vs stock at 64k (Flash-Next Q2_0): prefill 1.8x, decode +23 %; stock at 63.5k needs the whole file cache (0.3 GB free) | - |
| 4 | A slot save at 64k blocks the server 1.7-2.4 s (5.5 s at 138k), and a request sent meanwhile waits | quant-research's N27 change (save later, less often) |
| 5 | The 256k point's final decode after the 138k save and the slot-1 request did not finish in 7 min (likely the whole 138k prompt being processed again: unified KV, recurrent state); stopped at the 30-min limit | to check: whether a request on another slot or a save drops slot 0's recurrent state at 128k+ |
| 6 | No load abort, no KV that did not fit, no checkpoint crash on the AMD driver at any point | - |
Not run tonight (time; owner's stop 23:30): Qwen3.6 and the 122B at 256k, the 122B stock at 64k (its C: copy was gone, the D: one was queued
too late). Coder IQ1_M and GLM: removed from the PC by the owner. Queued for the next PC session: N29 256k ABBA (fixes), N15 (shader), N22
(laptop-h1-s9), N30 (fixes), with ready scripts (handover/parts/pc-bench.md).
**Update 2026-10-29 (overnight chain, RESULTS 67):** Flash-Next Q2_0, the 122B and Qwen3.6 at 128k and Qwen3.6 at 256k are in the tables above.
Finding 5 reproduced on Qwen3.6 at 256k: after the 138k save and the slot-1 request, the next chunk processed all 157,806 tokens again (625 s);
the 122B at 256k reached 138k in ~23 min and then timed out at the chain's 35-min limit, most likely in the same re-processing. GLM at 64k timed
out too (35 tok/s prompts; 15 of its 35 min went to a cool-down break). Still open: GLM 64k and the 122B at 256k with a longer limit.

**Addendum: N25 executor chunk size** (MOE_CACHE_EXEC_CHUNK, DLL 31ac159, Flash-Next Q2_0 on D:, launcher defaults, cold, ABBA 32 / 128 / 512 / 512 / 128 / 32,
256-token water and code decodes, a 3.5k-token prompt; C:\dl\pcb_chunk.ps1)
| Chunk (rows) | water decode tok/s | code decode tok/s | 3.5k prompt tok/s | text |
|---|---|---|---|---|
| 32 (old default) | 23.30, 23.58 | 23.64, 23.99 | 328.6, 330.6 | reference |
| **128** | 24.40, 24.29 (**+3.9 %**) | 24.83, 24.69 (**+3.9 %**) | 334.0, 328.3 | identical |
| 512 | 23.56, 23.72 (+0.9 %) | 24.09, 24.28 (+1.6 %) | 330.4, 323.9 | identical |
128 wins in every pair with identical text: the owner made it the default (fixes). Memory read speed of the PC (N25 / N25c, membw.exe, 8 GB):
~50-55 GB/s from 4 threads up, the same through a read-only mapped view of the model file (51.5-55.3) as from private memory (50.3-55.1), so the
CPU expert phase's ~26 GB/s is the executor's, not the view's.

## 65. New models on the Windows PC (pc-glm, 2026-10-28 evening): Nemotron 3.5 Bolt 74B-A9B; DeepSeek V4-Flash (desk only)

Picks from quant-research (handover/parts/model-research.md); owner: "see what's the realistic performance we can get with these", minimal
testing, one cold run per point. Same PC (7950X, RX 9070 XT 16 GB, 32 GB). Nemotron needs a llama.cpp newer than the PC's 620d2225c (tokenizer
fix #30960): official release **b11966** (ff0d396e3, Vulkan x64 zip, sha256 977dc456... = the release digest) in C:\llama-new. The moe-cache
DLL 681f3bc loads into b11966's ggml (backend API 3; 100 % cache hits when forced). Text checked: normal English and working code.

**Nemotron 3.5 Bolt 74B-A9B** (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF, NVIDIA's 2-bit QAT file, 22.0 GB, sha256 91d63363... = LFS oid;
C:\models). It fits in RAM + VRAM, so the launcher runs **stock** llama-server by default (the plugin only for models bigger than memory).
| Point (cold, launcher default = stock unless marked) | prompt | prompt tok/s | decode tok/s |
|---|---|---|---|
| water 256 / code 192 (ctx 64k) | short | - | 27.91 / 28.11 |
| ~57k-token prompt + 128 tokens (`--ctx 65536`) | 56,661 | **595.4** (95 s) | 27.79 at that depth |
| same, plugin forced (`--always-plugin`) | 56,661 | 329.0 (172 s) | 22.12 (water 22.82) |
| ~114k-token prompt + 128 tokens (`--ctx 131072`) | 113,932 | **720.8** (158 s) | 24.85 at that depth (water 25.41) |
| ~228k-token prompt + 128 tokens (`--ctx 262144`) | 227,709 | 279.6 (814 s) | 23.22 at that depth (water 25.82) |

| Nemotron, other measures | Result |
|---|---|
| 202 tasks (launcher default, `--no-think`, cap 128) | **190 / 202** (Q2_0 191: 8 / 9 changed, p 1.00; IQ3_XXS 193: p 0.58); 0 truncated answers. Took 701 s (Q2_0 169 s): ~3.5 s per short request, not investigated |
| OpenCode agent replay, stock (= launcher default), cold | **47.75 s** (first token 18.3 s, follow-up 1.53 s, decode 28.4) vs Flash-Next IQ2_XS 73.4 s (RESULTS 60), Coder IQ1_M 55.5 s (RESULTS 63) |
| same, plugin forced (agent_bench_pair's "mc" kind) | 59.17 s (first token 25.2 s, decode 24.2) |

**DeepSeek V4-Flash-0731 UD-IQ2_XXS (90.9 GB): desk only, not downloaded** (main session: its predicted 7-9 tok/s decode and 50-90 tok/s prefill,
i.e. a 12-20 min first turn at 64k, are useless for agents). Runs on 620d2225c; KV 1.8 GB at 256k (quant-research's estimate, not measured).

**Verdict.** Nemotron 3.5 Bolt is **worth it**: against Flash-Next Q2_0 (the default) it decodes +26 % (27.9 vs 22.2), reads long prompts about
2x faster (595 tok/s at 57k vs 287 at 34k) and keeps 23-25 tok/s at 114k-228k, the 202 tasks are equal (190 vs 191), and an OpenCode session is
the fastest measured (47.8 s). It fits in RAM + VRAM, so moe-cache adds nothing here (forced plugin: -18 % decode, -45 % prompt) and the
launcher's default (stock) is right. Needs llama.cpp newer than 620d2225c (b11966 used). General knowledge beyond the 202 tasks not checked.

## 66. Linux file-view cache made the default (laptop, measured 2026-10-27 night; merged 2026-10-28 under the owner's new rule)

`MOE_CACHE_LINUX_VIEW` was a prototype (branch claude/linux-file-cache, killed then for being under +5 %). The owner's rule of 2026-10-28 keeps every
gain that repeats in all ABBA pairs with exact output, so it is now the Linux default, as the file view already is on Windows (section 59);
`MOE_CACHE_LINUX_VIEW=0` goes back to private memory and O_DIRECT reads. Experts point into a shared read-only mmap of the model; load =
MADV_WILLNEED + MADV_POPULATE_READ, evict = MADV_COLD + MADV_DONTNEED. Qwen3.6 35B Q4_K_M under `MemoryMax=10G MemorySwapMax=0`, ncmoe 34,
ctx 8192, cold every run, both arms the plugin, ABBABAAB:
| | private + O_DIRECT | file view | change |
|---|---|---|---|
| steady decode (mean of requests 4-8, 4 runs each) | 10.60 | 11.10 | **+4.7 %** (every pair: +0.8, +8.4, +8.3, +1.8 %) |
| first (cold) request | 8.34 | 12.43 | **+49 %** |
| expert read time / CPU compute time | 19.5 s / 9.0 s | 11.5 s / 17.4 s | reads -41 %, compute +93 % |
| major faults | ~500 | ~27 000 | the kernel reclaims resident view pages at the cgroup limit |
Tokens identical in all 8 runs; granite identity 4/4 (GPU, CPU, CPU --no-repack) and 2/2 long through the view. Never map it MADV_RANDOM (Linux >= 6.3
reclaim ignores accessed bits of such mappings: 3.4-4.0 tok/s). Open lead: the compute time doubles because reclaimed view pages fault back;
a private cache read as fast as page-cache read-ahead would keep both gains.

## 67. Overnight chain on the Windows PC (pc-bench, 2026-10-28 23:32 -> 10-29 08:01): N29, N22, N15, N30, N26, N27, N23, long context, Bolt, HIP, N25c, N18

Same PC (7950X, RX 9070 XT 16 GB, 32 GB, AMD driver 32.0.32015.7029), llama.cpp Vulkan 620d2225c unless named, Flash-Next Q2_0 on D: unless named.
Scripts C:\dl\pcb_overnight.ps1 (+ pcb_ovlib.ps1) and the follow-up C:\dl\pcb_followup.ps1; one run = C:\dl\pcb\pcb_ov_run.py (launcher C:\moe-cache-ov
= experimental 15e09b1 unless named, `server MODEL`, own cache dir), cold (file cache flushed before every run), under the PC's timed lock, nothing
else running ("others running: none" before every run, 25.0-26.0 GB available). ABBA = runs A, B, B, A; text = each run's replies compared byte for
byte with run 1. Raw files: C:\dl\overnight\ (summary.csv, chain.log, followup.log, `<item>.jsonl`, `<item>_N.*`). RAM peak = the highest "in use"
of the log-only 28 GiB watch (C:\dl\overnight\ramcap.log, started 01:04, so N29 and N22 runs 1-2 are not covered; "< 28" = never logged over 28 GiB).
Cool-down breaks (owner): 03:13-03:28 (inside N27 run 3's lock wait) and 06:06-06:21 (inside GLM 64k's time limit); this PC exposes no CPU temperature.

**Summary**
| Item | Result | Verdict |
|---|---|---|
| N15 async reads of the plugin split's inputs | decode **+12 %** (22.7 -> 25.6 after 34k), water +12 %, code +12 %, 34k prompt +2 %, every pair, text identical | **merged** (f310c28) |
| N26 256k: old plan vs head 15e09b1 | head 72 vs old 139 tok/s prefill, decode 16.2 vs 19.1 | **T0a**: regression of the merged N26 plan at 256k; fixed on head (378cc26, 2963cc5), below |
| N29 256k: `-ub 1024`, block off | prefill +3.5 % at 4k, equal at 34k, decode at 4k -13 % | **killed** (but both arms ran every prompt on the CPU, below) |
| N22 pinned / async read-back | 34k prompt +3 % (+2.4, +3.6 % per pair), 128k mixed (-2.0, +5.1 %), read-back time per block unchanged | not merged; session "abba" repeats it |
| N27 warm saves at 64k / 128k | prefill 149 -> 201 (64k), 92 -> 176 (128k); follow-up after 90 s idle equal | already merged (cd625ba); the prefill gain is a layer-plan difference, not the save (below) |
| N30 executor pool spin | spin 2-5 ms: parks -97 %, decode -0.8 to -2 %, CPU time +14-23 % | **killed** |
| N23 where a long prompt's time goes | 34k 233 tok/s, 128k 113 tok/s with the Vulkan per-op logger on | logs for session "n23" |
| Long context (RESULTS 64 gaps) | FN Q2_0 / 122B / Qwen3.6 at 128k, Qwen3.6 at 256k done; 122B 256k and GLM 64k **timed out at 35 min** | in RESULTS 64's tables |
| Bolt | OpenCode replay at 256k 70.7 s (47.8 s at the default context); tiny-request test did not run (harness bug) | recorded; session "bolt" reruns |
| I31 stock HIP vs stock Vulkan | HIP decode 7.7 vs 18.7 tok/s, 3.5k prompt 45 vs 122 | **killed** |
| moe-cache under HIP | the plugin loads and runs; HIP is slow anyway. Vulkan: plugin 22.9 vs stock 17.5 tok/s (water) | Vulkan stays |
| N25c executor timers | pool matmul 26.4 s of the plugin's 52.9 s CPU time (32 GB/s incl. waits), cold short run | open: split the other half (below) |
| N18 split reads | not run: the script looked for the GLM shard on C: (it is on D:) | stays parked |
| Driver timeout of ~21:40 | event logs dumped to C:\dl\overnight\tdr_dump.txt | session "tdr" analyses it |

**N15: async reads of the plugin split's inputs** (DLL C:\dl\n15 = claude/n15-async, `--ctx 40960`; A = `MOE_CACHE_ASYNC_IN=0`, the old blocking
copies; B = default; 256-token water and code decodes, then a 34k-token prompt + decode; 37 CPU expert layers)
| Run | arm | water tok/s | code tok/s | 34k prompt tok/s | decode after 34k | async reads | RAM peak GiB |
|---|---|---|---|---|---|---|---|
| 1 | A | 22.02 | 22.40 | 286.6 | 22.69 | 0 | 28.57 |
| 2 | B | 24.64 | 25.06 | 297.2 | 25.46 | 18,944 | 28.53 |
| 3 | B | 24.72 | 25.08 | 292.8 | 25.67 | 18,944 | 28.53 |
| 4 | A | 22.11 | 22.49 | 290.3 | 22.80 | 0 | 28.55 |
| B vs A | | **+11.9 %** | **+11.7 %** | +2.3 % | **+12.4 %** | | |
Text identical in all 4 runs. Twice the +6 % the microbench predicted (ledger N15). Every pair wins with identical text: merged into experimental
(f310c28, `MOE_CACHE_ASYNC_IN`, on by default for AMD Vulkan). Every run passed 28 GiB in use (log only).

**N26 at 256k: old layer plan vs the merged one (T0a)** (`--ctx 262144 --agent`, 4k and 34k prompts; A = launcher 28b022b + DLL f478564, the plan of
RESULTS 64; B = experimental 15e09b1, which has N26's KV from the header)
| Run | arm | plan | 4k prompt tok/s | decode after | 34k prompt tok/s | decode after | GPU blocks (34k) | RAM peak GiB |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 48 CPU expert layers, `-ub 512` | 138.9 | 19.17 | 153.1 | 18.49 | 3,149 | < 28 |
| 2 | B | 45 CPU expert layers, `-ub 512` | 71.7 | 16.25 | 73.7 | 15.45 | **0** | < 28 |
| 3 | B | same | 73.1 | 16.21 | 75.6 | 15.39 | **0** | 30.35 |
| 4 | A | same as 1 | 139.8 | 18.97 | 154.6 | 18.30 | 3,149 | 29.10 |
| B vs A | | | **-48 %** | -15 % | **-51 %** | -16 % | | |
Text: A vs A identical; B differs from A (different layer split). Main's reading (T0a): the slow arm is the **new** plan at 15e09b1. It had no
compute-buffer row for 256k and used a 0.55 GB fallback, so it put 3 expert layers on the GPU; the long-prompt block then found 0 MiB free and every
prompt ran on the CPU (0 GPU blocks). 378cc26 (the N29 data: measured 256k compute rows) already makes head plan 48 CPU layers at 256k like the
fast arm, and 2963cc5 scales the largest measured row for contexts beyond the table instead of the flat 0.55 GB. RESULTS 64's 256k numbers (old
plan) stay the right ones for head.

**N29 at 256k: `-ub 1024` with the long-prompt block off** (DLL and launcher 15e09b1, `--ctx 262144 --agent`; A = defaults: `-ub 512`, block on,
45 CPU expert layers; B = `--ub 1024`, `MOE_CACHE_GPU_MIN_TOKENS=999999`, 47 CPU expert layers)
| Run | arm | 4k prompt tok/s | decode after | 34k prompt tok/s | decode after | GPU blocks |
|---|---|---|---|---|---|---|
| 1 | A | 81.3 | 19.89 | 86.7 | 19.29 | 0 |
| 2 | B | 86.2 | 17.66 | 87.1 | 19.57 | 0 |
| 3 | B | 85.9 | 17.80 | 88.2 | 19.53 | 0 |
| 4 | A | 85.0 | 20.77 | 89.0 | 19.45 | 0 |
| B vs A | | +3.5 % | **-12.8 %** | -0.2 % | +0.9 % | |
Text: A vs A identical, B differs from A. RAM: before the watch started. **Killed** (main): no prompt gain, decode at 4k lower. Caveat: this ran on
the same 15e09b1 plan as N26's slow arm, so arm A's block also found no GPU memory (0 blocks) and the ABBA compared two CPU-only prompt paths; what
`-ub 1024` does next to working GPU blocks at 256k was not measured (head now plans 48 CPU layers with `-ub 512` at 256k, 378cc26).

**N22: pinned + async read-back of the long-prompt block** (DLL 77e6e09 = claude/n22-readback, launcher 15e09b1, `--ctx 131072`: 44 CPU expert
layers, `-ub 2048`; A = `MOE_CACHE_GPU_READBACK=0`, the old path; B = default; both `MOE_CACHE_SEG_TIMES=1`)
| Run | arm | 34k prompt tok/s | decode after | 128k prompt tok/s | decode after | GPU blocks (128k) | read back per block | RAM peak GiB |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 241.5 | 19.86 | 129.8 | - (reply ended after 3 tokens) | 1,358 | 13.1 ms | not logged |
| 2 | B | 247.4 | 19.40 | 127.2 | 15.39 | 1,242 | 13.8 ms | not logged |
| 3 | B | 253.7 | 19.77 | 123.4 | 15.10 | 1,180 | 13.5 ms | 29.74 |
| 4 | A | 244.8 | 19.69 | 117.4 | - (reply ended after 3 tokens) | 978 | 13.1 ms | 28.87 |
| B vs A | | +3.0 % (pairs +2.4, +3.6) | | +1.4 % (pairs -2.0, +5.1) | | | unchanged | |
Read back per block = the block timer's total "read back" ms / blocks over the whole run (upload ~59 %, compute ~16 % of block time in both arms).
The number of 128k blocks that found GPU memory varies from run to run (978-1,358), which moves the 128k prompt more than the arm does. Text:
**DIFFERENT in all three comparisons, including A vs A** (runs 1 and 4 already differ in the 34k reply's thinking): flagged for session "abba",
not investigated here. Verdict: no clear gain and the read-back time it targets did not fall; not merged, "abba" repeats the ABBA.

**N27: warm-session saves** (`--ctx 131072 --agent --warm`, a 64k-token conversation, 90 s idle, a follow-up request, then the conversation grown
to 128k (58,975 new tokens), 90 s idle, a follow-up; A = experimental 15e09b1; B = launcher claude/n27-warm-save 2188d79; same DLL 15e09b1)
| Run | arm | plan | 64k prompt tok/s | follow-up after idle | 128k step tok/s | follow-up after idle | GPU blocks 64k / 128k | RAM peak GiB |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 44 CPU layers, `-ub 2048` | 147.8 | 0.41 s | 91.7 | 1.13 s | 879 / **0** | 28.48 |
| 2 | B | 46 CPU layers, `-ub 2048` | 203.2 | 0.42 s | 175.0 | 1.18 s | 1,426 / 1,334 | 28.79 |
| 3 | B | same | 199.4 | 0.43 s | 176.9 | 1.12 s | 1,426 / 1,334 | 28.73 |
| 4 | A | same as 1 | 151.1 | 0.42 s | 91.7 | 1.25 s | 878 / **0** | 28.50 |
2188d79 branches from before N26 (3cd1023), so B ran the **old** flat-KV plan (2 more CPU expert layers, so more free VRAM) and A the N26 plan:
the +35 % (64k) and +92 % (128k) prefill come from the long-prompt blocks finding GPU memory in B and not (128k) or only partly (64k) in A, not
from the warm save. The follow-up after idle is the same in both arms. N27 is merged already (cd625ba). The 128k case is T0a's pattern at 128k: head
changed the 128k plan again (378cc26: no `-ub 2048` at 128k+), so a head check at 128k in agent mode is still needed. Text: DIFFERENT in all
three comparisons, including A vs A (flagged for "abba").

**N30: executor pool spin** (DLL 8451a87 = claude/n30-pool-spin, launcher 28b022b defaults, `MOE_CACHE_POOL_SPIN_US` unset = the old 40,000
`pause`s; order unset / 500 / 2000 / 5000 / 5000 / 2000 / 500 / unset; water and code decodes, a 3.5k prompt; C:\dl\pcb_spin.ps1)
| Spin | water tok/s | code tok/s | 3.5k prompt tok/s | pool parks over the two decodes | llama-server CPU s |
|---|---|---|---|---|---|
| unset | 23.45, 23.50 | 23.71, 23.79 | 332.8, 320.1 | 265,920, 265,705 | 279, 281 |
| 500 us | 23.13, 23.27 | 23.51, 23.68 | 330.4, 326.0 | 288,825, 288,475 | 269, 267 |
| 2000 us | 22.97, 23.06 | 23.31, 23.45 | 325.7, 318.0 | 7,744, 7,756 | 327, 319 |
| 5000 us | 23.17, 23.02 | 23.31, 23.37 | 328.8, 329.8 | 7,680, 7,701 | 344, 345 |
Text identical in all 8 runs; no run logged over 28 GiB. 500 us changes nothing (parks +9 %); 2-5 ms removes 97 % of the parks, but decode is
0.8-2 % lower than unset in every pair and the server uses 14-23 % more CPU time. Workers sleeping between layers are not the N25 gap: **killed**.

**N23: where a long prompt's time goes** (DLL 15e09b1, `--ctx 131072`, `GGML_VK_PERF_LOGGER=1`, `MOE_CACHE_SEG_TIMES=1`, one run each)
| Run | prompt | prompt tok/s | decode after (logger on) | GPU blocks | block time: upload / compute / read back | RAM peak GiB |
|---|---|---|---|---|---|---|
| 1 | 34k | 233.0 | 10.59 | 748 | 58 / 16 / ~25 % | 28.23 |
| 2 | 128k | 113.1 | 10.18 | 837 | 60 / 16 / ~24 % | 29.14 |
The per-op logs (n23_perf_1/2.err.txt, 20 and 24 MB) are analysed by session "n23"; the decode numbers are slowed by the logger and are not speeds.

**Long context** (pcb_lc.ps1, launcher 28b022b + DLL f478564 as in RESULTS 64, 35-min limit per point): the numbers are in RESULTS 64's tables.
| Point | Result | RAM peak GiB |
|---|---|---|
| Flash-Next Q2_0 128k | done, 768 s; decode 18.97 at 4k, 15.27 at 130.6k | 28.67 |
| Qwen3.5-122B 128k | done, 735 s; decode 15.54 at 4k, 11.27 at 130.6k | < 28 |
| Qwen3.6 128k (stock) | done, 196 s; decode 55.08 / 45.76 | 28.37 |
| Qwen3.5-122B 256k setting | **timed out at 35 min** after reaching 138k (decode 12.65 there) and its save | < 28 |
| Qwen3.6 256k setting (stock) | stopped feeding at its 1500 s budget at 227.8k (decode 41.74); after the 138k save the next chunk processed all 157.8k tokens again | < 28 |
| GLM-5.3-Flash 64k | **timed out at 35 min** (15 min waiting for the 06:06 cool-down break; 35 tok/s prompts, decode 2.08 at 4k, hit 32.7 %) | < 28 |
Qwen3.6 (C:) runs warned that the flush's other files (22.8 GB) were smaller than RAM, so its file may have been partly cached.

**Bolt (Nemotron 3.5 Bolt 74B-A9B, llama.cpp b11966, stock = the launcher's default for a model that fits)**
| Measure | 256k (`--ctx 262144`, launcher plan `--n-cpu-moe 88 -ub 512`) | default context (RESULTS 65) |
|---|---|---|
| OpenCode replay, whole task | **70.7 s** | 47.8 s |
| first token (7,634-token first prompt) | 35.0 s (218 tok/s) | 18.3 s |
| follow-up first token, median | 2.09 s | 1.53 s |
| decode, median | 25.5 tok/s | 28.4 tok/s |
One cold run, errors 0; in 4 of 5 requests the model stopped 11-26 tokens before the recorded length (replay warning).
The Hermes replay did not run: C:\dl\glm\sess_oc holds only the OpenCode recording. **Tiny requests** (default / `--reasoning-budget 0` / no
thinking, on 620d2225c and on b11966): none ran. The harness started a stock run (`--largs=--stock`) through the launcher in C:\moe-cache-ov,
which looked for the plugin DLL in C:\moe-cache-ov\build\win and stopped ("plugin not built"). Session "bolt" fixes it and reruns.

**I31: stock HIP vs stock Vulkan** (raw llama-server, Flash-Next Q2_0, `-ngl 99 --n-cpu-moe 39 -c 8192 -fa on -ub 2048`; A = C:\llama-vk
(Vulkan 620d2225c), B = C:\llama-hip (official b11669 ROCm/HIP zip, no SDK installed))
| Run | arm | water tok/s | 3.5k prompt tok/s | decode after | RAM peak GiB |
|---|---|---|---|---|---|
| 1 | Vulkan | 18.70 | 122.7 | 16.98 | 28.92 |
| 2 | HIP | 7.74 | 45.5 | 5.99 | 30.85 |
| 3 | HIP | 7.68 | 45.8 | 5.84 | 30.85 |
| 4 | Vulkan | 18.65 | 122.0 | 16.53 | 28.84 |
Text: Vulkan vs Vulkan identical, HIP differs. HIP is 2.4x slower in decode and 2.7x in prompts on this card: **killed**.

**moe-cache under HIP** (follow-up item 14: the plugin DLL 15e09b1 in C:\llama-hip's llama-server, `--ctx 65536`, water + 34k prompt)
| Run | arm | plan | water tok/s | 34k prompt tok/s | decode after | RAM peak GiB |
|---|---|---|---|---|---|---|
| 0 | plugin, HIP (load check) | 48 CPU layers, `-ub 512` | 7.82 | - | - | not logged |
| 1 | stock, HIP | `-ub 512` | 7.66 | 30.0 | 6.74 | 30.86 |
| 2 | plugin, HIP | 48 CPU layers, `-ub 512` | 7.79 | 39.8 | 6.60 | 28.23 |
| 3 | stock, Vulkan | `-ub 1024` | 17.52 | 179.9 | 23.37 | 30.85 |
| 4 | plugin, Vulkan | 38 CPU layers, `-ub 1024` | **22.88** | **199.0** | 23.28 | 28.64 |
The plugin loads and runs under HIP ("moe-cache: active"), but the launcher reads no GPU memory from the HIP build and plans every expert layer on
the CPU; HIP itself is slow either way. Vulkan: the plugin is +31 % on the short decode and +11 % on the 34k prompt, equal after 34k. Vulkan stays
the only GPU route on Windows AMD.

**N25c: executor timers on decode-like batches** (llama-perplexity, DLL 15e09b1, 6 chunks of 512 tokens at `-b 8 -ub 8`, `--n-cpu-moe 39`, cold,
`MOE_CACHE_STATS=1`): 2.0 min, PPL 5.2407; 15,015 graphs, hits 97.7 % (46,833 misses, 21.6 GB read, budget 24.2 GiB). The plugin's CPU time
(`cpu_compute`, the whole executor graph) 52.90 s of 53.97 s in the backend; inside it the pool's q2_0 matmuls (`executor by type`) **26.39 s**
for 854.2 GB of expert rows = 32.4 GB/s including waits (the 7950X reads ~50-55 GB/s, RESULTS 64 addendum). So half the executor's time is outside
its matmuls (waits for the cold run's 46,833 misses, routing, the other ops). Verdict: the timers work now; to tell waits from work, the next
run should be warm and longer, or the executor needs a wait timer. Not a decode result (batches of 8 tokens).

**N18: split expert reads** (C:\dl\pcb_n18.ps1): did not run. It flushes and reads `C:\models\GLM-5.3-Flash-UD-IQ2_XXS-00002-of-00004.gguf` first,
which does not exist (GLM is on D:), and the flush script stopped there. N18 stays parked (killed on Linux; owner parked it on Windows).

**Text check (flag only).** A vs A differs in N22 (runs 1 and 4, ctx 131072, already in the 34k reply) and N27 (64k+); it is identical in N15, N26,
N29, N30 and I31. Session "abba" owns the question whether this is the check or the runs.

**RAM watch.** Past 28 GiB in use (log only, nothing stopped): every logged N15, N22, N26 (runs 3-4), N27, N23, I31 and HIP-item run, Flash-Next
and Qwen3.6 at 128k; peak **30.86 GiB** (hip_plugin run 1). Never past 28 GiB: N30, the 122B at 128k and 256k, Qwen3.6 at 256k, GLM, Bolt, N25c. The product's hard cap (RAM minus max(4 GiB, 12.5 %)) was merged since (e708462).
