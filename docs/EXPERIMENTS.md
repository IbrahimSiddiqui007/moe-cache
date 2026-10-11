# Experiment ideas for large models (Qwen3.8-Flash-Next class)

Status: ideas with a way to test each one. Nothing here is built or measured unless a "Measured" column says so.
Every idea that changes the generated tokens is opt-in and needs a quality measurement before it goes anywhere near a default.

## What we know about the target (Qwen3.8-Flash-Next IQ2_XS, this laptop, 24 GB limit)
| Fact | Value | Source |
|---|---|---|
| Layers / experts / top-k | 48 / 512 / 10 (+1 shared) | HF config |
| Expert size | about 1.44 MB per expert (three tensors, about 2.3 bits per weight) | planner |
| Experts total | 33.0 GiB; dense core 3.5 GiB; n-gram table 26.8 GiB (stays in the file) | planner |
| Cache at 24 GB | 21 GiB = 68 % of the CPU-side experts (3 layers of experts on the GPU) | planner |
| Quick test (not gated, 48-token answers) | 5.1 to 7.0 tok/s while the cache filled, 88.6 % hit rate, no evictions yet | `fn_speed` run |
| n-gram table access | 16 rows of 90 bytes per token, at one layer | `qwen4exp.cpp` |

## Ideas, ranked by (expected gain x low cost)
| # | Idea | What it does | Why it might work here | Changes tokens? | How to test | Cost |
|---|---|---|---|---|---|---|
| 1 | **Measure first: routing trace of Flash-Next** | record expert ids per decode step with `MOE_CACHE_TRACE_IDS` on 30 varied prompts, then run the replay and skew scripts | everything below depends on how skewed and how repetitive routing is with 512 fine-grained experts; the 120B had only 10 % lookahead recall, Qwen3.6 had 77 % | no | trace, `work/hottier`, `analyze_model.py` | 1 hour |
| 2 | **Learned warm start** (`moe-cache-learn` on Flash-Next) | profile of which experts the user's prompts need, loaded at start | first request was 5.1 vs 7.0 tok/s; the prompt phase waited 13 s for cold experts | no | A/B first-request time with and without profile | 30 min |
| 3 | **MTP speculative decoding** | the model ships an MTP head (4.1 GB file); verifying k drafted tokens reuses each loaded expert for several tokens | +20 % on Qwen3.6 when fitting; for SSD-bound models fewer reads per accepted token. n-gram speculative failed because it wakes extra experts, MTP drafts are closer to the real path | no (same model, verified) | `--spec-type draft-mtp`, check pinned llama.cpp supports qwen4exp MTP | 1 hour |
| 4 | **Cache-aware routing (expert substitution)** | when the router picks an expert that is not resident and its weight is small, use the best resident expert of that layer instead | misses are the bottleneck; each avoided miss saves 1.44 MB of SSD reads; with 10 of 512 experts the 9th and 10th picks carry little weight | **yes** (opt-in) | replay on the trace: miss reduction vs mass of the substituted weights; then KL divergence / perplexity on a text file | 1 to 2 days |
| 5 | **Adaptive top-k ("fast mode" for 10 experts)** | skip experts whose routing weight is below a threshold, or the last N | Qwen3.6 trace: dropping the 2 smallest of 8 removed 25 % of reads and 9.1 % of the output weight | **yes** (opt-in) | same replay and KL measurement as 4 | 1 day |
| 6 | **n-gram row prefetch for prompts** | during prompt processing all rows of the n-gram table are known up front: issue the reads in parallel (`io_uring` / `madvise(WILLNEED)`) instead of 16 page faults per token | 13 prompt tokens took 13 s cold; part may be table faults, not only experts | no | count page faults of the table file during prefill, then prefetch | 1 day |
| 7 | **`io_uring` batched reads with registered buffers** | submit all misses of a layer in one syscall, fewer thread wake-ups | batches are only 1 to 2 missing experts per layer today, so only 3 to 6 reads are in flight | no | read microbenchmark first (`work/rdbench`), then plugin | 2 days |
| 8 | **One `preadv` per expert** (scatter into gate/up/down) | one IO instead of three, no packed copy of the model needed | removes the old objection "a packed read cannot land in three places" | no | microbenchmark | 1 day |
| 9 | **Pre-fault replacement pages** (`MADV_POPULATE_WRITE` ahead of the read) | avoid the 30 % page re-fault cost measured on fresh pages | microbench: 2.72 GB/s fresh pages vs 3.84 GB/s reused | no | microbenchmark | 1 day |
| 10 | **Intra-layer overlap** | compute the resident experts of a layer while the missing ones are still loading | up to about 20 % on the 120B (estimate) | no | prototype; identity must hold | 3 to 5 days |
| 11 | **GPU hot tier from VRAM left over** (opt-in, "same quality, not same tokens") | copies of the hottest experts in VRAM, computed on the GPU | 512 fine-grained experts may be more skewed than Qwen3.6; the earlier 4 GB result was +7 % over whole layers | **yes** | experiment 1 gives the skew; then `hotsim12.py` for the real VRAM left (about 1 GiB after 3 expert layers) | 3 days |
| 12 | **Compressed RAM cache** | keep cold experts compressed in RAM, decompress on use | IQ2 data is close to incompressible, so probably useless | no | `zstd -3` on a 1 GB slice of shard 1, one command | 5 min |

## Status (2026-10-11)
| # | Result |
|---|---|
| 1 | Done: routing trace (section above). |
| 2 | **Measured: first request 2.1-2.4x faster with a learned profile**, steady speed unchanged (RESULTS.md section 16). |
| 3 | llama.cpp 23b0202a1 has an MTP graph for qwen4exp; the head is a separate file (`unsloth/Qwen3.8-Flash-Next-GGUF`, `MTP/mtp-...-Q4_K_M.gguf`, 2.79 GB); not downloaded yet, not tested. |
| 4, 5, 11 | Not done on purpose (change tokens). |
| 6 | **Dead**: 3-23 major faults per request. |
| 7 | io_uring: same device ceiling; +30 % to 2x only for slices of 0.5 MB or less. No effect on Flash-Next (compute-bound). |
| 8 | Not possible on the stock file layout (one expert's gate, up and down are three separate regions); would need a repacked copy. Slices of 2 MB and more already reach the device ceiling. |
| 9 | Fresh pages cost about 20 % of read throughput; pre-faulting in the reader thread recovers it. Worth doing when a model is SSD-bound (GLM-sized experts), not for Flash-Next. |
| 10 | Already in v0.2.0 (the executor computes ready experts first and waits for the others). |
| 12 | Dead (zstd saves 2.6 %). |

## Order I would run them in
1. Idea 12 (five minutes, may kill a whole branch of ideas), then idea 1 (data for 4, 5, 11 and for the cache policy).
2. Ideas 2 and 3 (no change to tokens, cheap, directly useful for a "just run the model" user).
3. Ideas 6 to 9 as microbenchmarks before touching the plugin.
4. Ideas 4 and 5 only with a written quality measurement plan, as clearly labelled opt-in modes.

## Measured (2026-10-10, Qwen3.8-Flash-Next IQ2_XS, this laptop, 24 GB limit, 3 expert layers on the GPU)

**Routing trace** (12 varied prompts, 948 decode tokens, 45 CPU layers, 512 experts, top-10; `MOE_CACHE_TRACE_IDS`):

| Statistic | Value |
|---|---|
| Top 5 / 10 / 25 / 50 % of (layer, expert) pairs cover | 30.5 / 47.0 / 76.1 / 95.2 % of picks (uniform would be 5 / 10 / 25 / 50 %) |
| Distinct pairs used in 948 tokens | 18,325 of 23,040 (80 %) |
| Global LRU hit rate, cold start, whole run / second half | 4 GiB 70 / 71 %, 8 GiB 85 / 85 %, 12 GiB 91 / 92 %, 16 GiB 94 / 95 %, **19.5 GiB 95 / 97 %**, 24 GiB 96 / 98 % |
| Miss rate by pick rank at 19.5 GiB | rank 0: 1.0 %, rank 5: 3.3 %, rank 9: 7.1 % (the last picks miss 7 times as often as the first) |
| Token-to-token overlap per layer | 36.1 % (random 2.0 %) |
| Static top set from the first half, evaluated on the second half | 61 % at 8 GiB, 87 % at 19.5 GiB (LRU does better, so a profile is a warm start, not a replacement) |

**Time per token** (plugin counters, 384 tokens): about 99 ms expert compute on the CPU and about 32 ms SSD reads (cache still warming); the rest is small.

**Threads** (same 8 requests, cold file cache, 24 GB limit):

| Setting | Mean of requests 5-8 |
|---|---|
| 6 threads on the 6 performance cores (baseline) | 5.57 tok/s |
| 12 threads (performance cores and their siblings) | 5.40 tok/s |
| 16 threads, no pinning (includes efficiency cores) | 5.49 tok/s |

**Compressibility** of 1 GiB of expert bytes: zstd gives 97.4 % of the original size, so a compressed RAM cache is out.

What this changes in the ranking above:
- Decode here is limited by **expert math on the CPU**, not by the SSD: thread count does not matter (hyper-threads share the same execution units, efficiency cores are slower and the rows are split statically) and reads are about a quarter of the time. Ideas 6 to 9 (IO tuning) can give at most the read share, so roughly 10 to 20 %.
- The levers that act on the compute are: fewer experts evaluated per token (ideas 4 and 5, both change tokens), moving compute to the GPU (idea 11), and a **format the CPU kernels run faster**: the model's authors report 1.9x lower latency for their Q2_0 variant against IQ2_XS (66.4 GB file, not downloaded).
- The last picks of a token miss 7 times as often as the first and carry the smallest weights, which is exactly where idea 4 (substitution) and idea 5 (adaptive top-k) would act.

## Result: the bottleneck was one kernel (2026-10-10)

The "IQ2_XS" file of Qwen3.8-Flash-Next is not pure IQ2_XS: gate and up are IQ2_S / IQ2_XXS / IQ1_M, and **all 48 down projections are Q2_0**. llama.cpp (7fe450e19) has
no SIMD implementation of the Q2_0 x Q8_0 dot product on x86; it runs a scalar C loop. From the two files' compute times the Q2_0 weights cost about 5 times as much per
weight as the IQ2 weights and made up about 70 % of the expert math.

| Step | Result |
|---|---|
| Kernel written (`src/fast_q2_0.h`, AVX2, integer part exact, float part in the C loop's order with fused multiply-add) | 3.1x faster than the library function on a 2560-weight dot |
| Bit-exactness (`tests/kernel_check.cpp` against the installed libggml-cpu) | 0 mismatches in 18,000 random inputs incl. extreme values; half to float identical for all 65,536 values |
| Token identity, Flash-Next IQ2_XS, 4 prompts x 48 tokens, 24 GB limit | 4/4 identical to stock |
| Speed, same session, 10 requests of 48 tokens, mean of requests 6 to 10 | kernel on **10.26 tok/s**, kernel off (`MOE_CACHE_FAST_KERNELS=0`) 6.94 tok/s |
| Expert compute time over the run | 29.3 s with the kernel, 45.8 s without |
| Q2_0 file of the same model | 4.03 tok/s before, 8.35 tok/s with the kernel |

One paired run each, not gated (desktop apps were using CPU), so read the ratio (about 1.5x), not the absolute numbers. How it is applied: the plugin exports
`ggml_vec_dot_q2_0_q8_0`; with `LD_PRELOAD` the dynamic linker resolves libggml-cpu's function table to it. In `GGML_BACKEND_PATH` mode it is not used.


## Measured (2026-10-11): kernels for the GLM-5.3-Flash types and upstream's expert cache

- GLM-5.3-Flash UD-IQ2_XXS (header read, no download): gate/up IQ2_XXS (IQ2_S in one layer), down IQ3_XXS (IQ4_XS in three layers), 288 experts, 8 used, 42 MoE layers. Per-core dot-product speed on this CPU: IQ2_XXS 16.2, IQ3_XXS 12.5 G weights/s versus Q4_K 46 and Q2_K 50. By arithmetic alone that is roughly 5-6 tok/s of expert compute (estimate, not measured on the model).
- AVX2 / AVX-VNNI rewrites of those two dot products: no gain for IQ2_XXS, 13-18 % for IQ3_XXS with gathers (not worth shipping: gathers are slow on some AMD CPUs). See RESULTS.md section 15.
- Upstream's `--moe-cache-mib` is the "GPU hot tier" idea (idea 11) implemented in llama.cpp itself: +29 % on Qwen3.6-35B when the model fits in RAM. It does not work together with this plugin yet (see RESULTS.md section 15).
