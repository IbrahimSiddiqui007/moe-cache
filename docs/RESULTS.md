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

No block stayed on the CPU in either run. Runs a and b differ by about 12 % (the known +-10 % band); decode at ctx 8192 is 9-12 tok/s (12.4-13 at
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

This is the agent follow-up case (tool results of 30-300 tokens) and it is faster than stock llama.cpp, which uploads whole layers. Batches that use
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
