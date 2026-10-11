# Design: a GPU expert tier owned by the plugin (ideas 1 and 3)

Status: **design only, not built.** Written 2026-10-11 from measurements on an RTX 4060 Laptop (PCIe 4.0 x8), Flash-Next IQ2_XS.

## Why llama.cpp's own cache cannot be extended from outside

llama.cpp's `--moe-cache-mib` decides slots itself (an LRU inside `llama_moe_cache`), reads the router's expert ids back and synchronises the GPU **at every layer**,
uploads the misses, then computes. Nothing outside can learn which slot an expert got, so nothing outside can upload an expert *before* the router has chosen it.
Two ideas therefore need a cache that the plugin owns:

1. **Prefetch**: while layer N computes, upload the experts that layer N+1 is predicted to need.
2. **Hybrid hit/miss**: the GPU computes the experts it holds, the CPU computes the missing ones from RAM in parallel, nothing is uploaded for a miss.

## What the measurements say it is worth

| Quantity (Flash-Next, 18 tok/s steady = 55 ms per token, 45 expert layers) | Value | Source |
|---|---|---|
| GPU cache hit rate at 2200 MiB | 53.6 % | llama.cpp log |
| Bytes uploaded per token | about 285 MiB | llama.cpp log |
| Host-to-GPU copy speed | 9-11 GB/s pageable, 12.9 pinned | `tests/h2d_bench.cu` |
| Upload time per token | about 25-30 ms (about half of the token) | arithmetic from the two lines above |
| CPU time for the same misses (IQ2 / Q2_0 kernels, 6 P-cores) | about 0.6 ms per layer, similar to the upload | `kernel_bench`, executor timings |
| Synchronisations per token | 50 (one per layer) | llama.cpp log |

Prefetch can hide the part of the upload that is predicted correctly. With a predictor recall of 50-75 % the token could go from 55 ms to roughly 38-45 ms (+25-45 %);
at recall 10 % (gpt-oss-120b) it gains nothing. Hybrid compute removes the upload but needs the CPU free at the same time as the GPU: estimated +10-20 %.
Both estimates are arithmetic, not measurements.

## Components

| Part | What it does | Reuses |
|---|---|---|
| GPU bank | VRAM slots per tensor shape and type, allocated through the GPU backend's buffer type | `gpu_dev`, as `gpu_segment` |
| Slot map and LRU | expert -> slot per layer; scan-resistant admission is optional | the plugin's node table |
| Upload engine | pinned staging ring, one copy stream, events instead of backend-wide sync | `h2d_bench.cu` results |
| Predictor | layer N+1's experts from layer N's router input and the profile | `predict_and_prefetch` (lookahead) |
| Decode block | gate, up, activation, down, router weighting and sum for the hit experts on the GPU (own small ggml graph with slot indices as ids) | `gpu_segment` graph cloning |
| Miss path | executor computes the missing experts on the CPU; the two partial sums are added | `exec_one_op` |
| Switch | `MOE_CACHE_GPU_TIER=1`, off by default; falls back to the executor on any failure | |

## Order of work (each step must keep the exact-identity test passing, or the quality gate for approximate steps)

1. Bank + slot map + upload on miss, GPU computes all experts of cached layers (should equal llama.cpp's cache: same speed, a baseline).
2. Predictor-driven prefetch on a separate copy stream (the gain in "ideas 1").
3. Hybrid hit/miss with concurrent CPU execution (idea 3).
4. Hot-expert quantisation and per-layer sizing (idea 3 of the lead list).

Main risks: the router weighting and the expert sum live in llama.cpp's graph after the matrix products, so a partial sum has to be reproduced exactly for identity;
CUDA stream ownership next to llama.cpp's own streams; VRAM sharing with `--n-cpu-moe` layers and the KV cache.

## Update 2026-10-12: measured inputs for step 2 (prefetch) and the order of work

- **Predictor recall on Flash-Next** (plugin lookahead, router weights of the next layer, 12 prompts x 64 tokens): 68.1 % of the next layer's picks predicted
  (241,479 of 354,760), 67 % of the prefetched experts were used (25,809 of 38,667), 8,006 arrived late. Cheap predictors from past picks alone reach 36-57 % with
  about 28 % precision (RESULTS.md section 24.4). So a prefetch with the router-based predictor is in the range that makes the estimate (+25-45 %) plausible; the
  cheap ones would double the upload traffic for a gain that is not there.
- **Cache policy is not a lever**: LRU, 2Q and pinned+LRU are within 0.5 points (RESULTS.md section 24.4), so the tier can use plain LRU.
- **VRAM is the constraint that came out of the prefill work** (RESULTS.md section 23): the long-prompt GPU block needs about one layer (923 MiB) free next to any GPU
  expert cache. A plugin-owned bank must reserve that too, or give the memory back during prompt processing.
- **Not started**: step 1 (bank + upload on miss) is only a baseline equal to llama.cpp's cache; the gain starts at step 2, which needs the router-based predictor
  to run before llama.cpp's callback and a separate copy stream. That is days of work and a new exact-identity argument for the partial sum, so it was left as the
  next large item rather than started inside a short window.
