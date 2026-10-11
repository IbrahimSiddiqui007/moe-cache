# Where a decoded token's time goes, and what could still be gained (2026-10-11)

Machine: i7-13620H (6P+4E), RTX 4060 Laptop 8 GB on PCIe 4.0 x8, 31 GB RAM, WD SN560. llama.cpp 23b0202a1. Model: Qwen3.8-Flash-Next IQ2_XS (68 GB, 24 GB limit)
unless said otherwise. "Measured" = a number from a clean run on this machine; "Read" = found by reading the source, not measured.

## What a token costs (Flash-Next, plugin, GPU expert cache 2200 MiB, steady state about 18 tok/s = 55 ms)

| Piece | Evidence | Size |
|---|---|---|
| Scheduler splits | Measured (llama.cpp log): 50 splits per decoded token, one per layer | one synchronisation per layer: the router's expert ids must be read back before the next step |
| Expert uploads over PCIe | Measured: llama.cpp cache hit rate 53.6 % at 2200 MiB, about 285 MiB uploaded per token; host-to-GPU copies run at 9-11 GB/s from pageable memory (`tests/h2d_bench.cu`) | about 25-30 ms of the 55 ms |
| Plugin RAM cache | Measured: 97.7 % hit rate at 19.5 GiB | reads are not the limit once warm |
| GPU compute, attention, shared experts | not isolated (no `nsys`/`ncu` on this machine) | the rest |

## What was tried and what it gave (all measured, clean runs)

| Change | Result |
|---|---|
| llama.cpp GPU expert cache on top of the plugin (2200 MiB) | +12 % over static GPU layers (11.95 -> 13.4 tok/s, 4 rounds), prompt time halves |
| The same cache at only 500 MiB | **-27 %** (8.6 vs 11.95): nearly every expert misses and is uploaded every layer |
| `-ub 2048` instead of the default 512 | prompt processing +70-76 % |
| Executor for batches of up to 8 tokens | exact (0 of 57,669 byte comparisons differ); short-prompt processing about 2x faster; makes MTP 2.3x faster than before, still 17 % below no MTP |
| Pre-fault of replacement pages | no gain (compute-bound) |
| MTP head on the GPU | does not fit in 8 GB next to the model (needs about 2.9 GiB incl. buffers) |

## Read in the source (not measured)

| Finding | Where | Consequence |
|---|---|---|
| CPU `mul_mat_id` runs experts one after another with two thread barriers per op; the plugin's executor replaces this with one shared work counter and starts on resident experts first | `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_compute_forward_mul_mat_id` | the executor is the reason decode on partly-resident models is faster than stock; upstreaming the idea is possible |
| The GPU cache reads the router ids back and synchronises the backend at every layer before it uploads | `src/llama-moe-cache.cpp`, `copy()` | uploads are serialised with compute; they cannot overlap unless the experts are predicted earlier |
| The cache does not serve batches above 32 tokens; they use the normal copy path | `get_slot_map()`, `max_batch = 32` | prompts do not benefit from the cache |
| x86 IQ2_XXS / IQ3_XXS dot products are table-lookup bound on one shuffle port | `ggml/src/ggml-cpu/arch/x86/quants.c` | AVX2 rewrites gave 0-18 %; AVX-512 permutes may do better (implemented, untested: `src/fast_iq_avx512.h`) |

## Best leads, in order of expected gain

| # | Idea | Why it could pay | Risk |
|---|---|---|---|
| 1 | **Predict and upload the next layer's experts while this layer computes** (the plugin's lookahead predictor exists) | uploads are 25-30 ms of 55 ms and currently wait for compute; a hit hides them | prediction quality depends on the model (recall 77 % on Qwen3.6, 10 % on gpt-oss-120b); needs a hook before llama.cpp's callback |
| 2 | **Hybrid hit/miss**: GPU computes cached experts while the CPU computes the misses from RAM, no upload | removes PCIe traffic; CPU math for 5 misses is about as long as their upload (0.6 ms vs 0.6-0.9 ms per layer) and runs in parallel | needs a split MUL_MAT_ID per layer inside the plugin |
| 3 | Larger effective cache: quantise the cached copies harder, or cache only the most reused expert tensors (down projection) | more hits per MiB | quality-gated |
| 4 | Pinned slot arena for the plugin's RAM cache (uploads from pinned memory: 12.9 vs 9-11 GB/s) | at most +15-30 % on the upload part | breaks stock addresses; large redesign; small gain |
| 5 | AVX-512 IQ kernels on Zen 4 / Intel AVX-512 | removes eight sign lookups per 64 weights | unknown until a tester runs `tools/friend_test.sh` |

## 128k context (what the numbers say)

Flash-Next's KV cache is small: 60 MiB per 2,048 tokens in f16, so about 3.8 GiB at 128k (1.9 GiB with q8_0). On an 8 GB card that leaves little for a GPU expert cache,
so a long context and a big cache compete for the same VRAM. `bench/quality.py longctx` measures retrieval and speed at growing lengths, and `soak` checks
that answers, speed and memory stay stable over a long session.

## Long sessions: memory accounting (found by testing, 2026-10-11)

A 128k-context server under a 24 GB limit was OOM-killed when a *second* long request started: llama-server saves the finished prompt's state (KV cache and
checkpoints) into host RAM (`--cache-ram`, 8 GiB by default) in one burst, and the plugin's RAM cache had been sized to "limit minus a 1 GiB margin". The plugin now
adds `--cache-ram` to its margin, and `moe-cache-server` / the benchmark tool default it to 2 GiB.

## Prompt processing is a streaming problem (measured 2026-10-11)

With the model bigger than RAM, every micro-batch of a prompt passes over (almost) all experts of every layer, so prefill time is about
`(prompt tokens / ub) x (bytes of experts not in RAM) / SSD speed`. Raising `-ub` from 512 to 2048 gave +58 % at a 6k prompt (87 to 140 tok/s). The ceiling is the GPU compute
buffer, which grows with the context (3.06 GiB at `-ub 2048` and 128k), so the best `-ub` depends on how much context you run. Inserting prompt experts at the cold end of the
cache (scan resistance) lost 25 % of prefill and is off by default. Remaining idea: choose `-ub` automatically from the free VRAM and the context size in the launcher.
