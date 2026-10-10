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

## Order I would run them in
1. Idea 12 (five minutes, may kill a whole branch of ideas), then idea 1 (data for 4, 5, 11 and for the cache policy).
2. Ideas 2 and 3 (no change to tokens, cheap, directly useful for a "just run the model" user).
3. Ideas 6 to 9 as microbenchmarks before touching the plugin.
4. Ideas 4 and 5 only with a written quality measurement plan, as clearly labelled opt-in modes.
