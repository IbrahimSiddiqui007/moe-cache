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
