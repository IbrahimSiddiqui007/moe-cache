# Do we need our own inference engine?

Short answer: **not now.** Stay a layer on top of ggml through v1.0, and make the decision on measurements, with one gate (below).

## What the others do

Strata and Maya (from their README pages; their code was not read) are dedicated engines for one model each (a 125B and a 321B MoE), built on components of
llama.cpp/ggml plus their own CUDA/AMD kernels, with a hot-expert tier in VRAM, all experts in RAM and the rest on SSD. They need 32 GB RAM and 12 GB VRAM (Strata) or a V100-class GPU (Maya), run on Windows and Linux, and report high speeds
(Strata: 94 tok/s on an RTX 5070 for its 125B model at 2-bit).

## What an own engine would buy

| Benefit | How much, from our own data |
|---|---|
| GPU hot-expert tier with custom kernels | Emulation (Qwen3.6): a profile-chosen tier is about equal to simply putting whole layers on the GPU (1.41x vs 1.45x at 4.5 GiB). A dynamic tier might add +10 to +25 % in the best case; copying an expert to the GPU costs 0.18 ms, so it can also lose. Largest relative gain at small VRAM (4 GB and below). |
| CPU and GPU computing the same layer in parallel | Not measured. Decode is limited by memory bandwidth, about 75 % of a token is expert reading on the CPU; overlap could help but the ceiling is unknown. |
| One implementation on Windows, AMD and NVIDIA | Real, but llama.cpp already has CUDA, HIP, Vulkan and SYCL backends; the plugin rides on them. |
| No graph-split overhead between CPU and GPU parts | Unmeasured. |

## What it costs

- Every model architecture must be re-implemented and validated token by token (llama.cpp has 100+). Strata and Maya each cover one model.
- Tokenizers, chat templates, sampling, quantization formats, server API, long-term maintenance.
- A much higher hardware floor in the reference engines (32 GB RAM, 12 GB VRAM). Our target (24 GB RAM, 4 to 8 GB VRAM, any MoE) is a different market.

## What the plugin already delivers without an engine

- 2x to 7x stock when the model does not fit in RAM (the biggest lever by far), identical output, works with any MoE llama.cpp supports (8 architectures tested).
- Using the GPU well is worth +60 % (all experts on CPU 16.2 -> planner placement 25.9 tok/s) and is done by llama.cpp's own placement.

## Decision rule

1. Keep building on ggml (plugin) for v0.2 to v1.0.
2. **Gate for an engine**: prototype the GPU tier inside the plugin (a backend can use the GPU through ggml's CUDA/Vulkan backends). If it gives **at least +20 %** on a small-VRAM setup (about 4 GB, the interesting case) with identical output, keep it as a plugin feature.
3. Consider an own engine only if that prototype shows large gains **and** the missing piece is something ggml's backend interface cannot express (custom kernels reading experts through a slot table, intra-layer CPU/GPU overlap).
4. If none of that happens, an engine would add maintenance cost for an unmeasured benefit.

## What we cannot claim

The emulation ran on one machine with one model; the GPU tier has never been built; Strata/Maya's own performance depends on their narrow model choice and quantizations and is not comparable to our numbers.

## Gate result (2026-10-09, simulation and a small identity probe; no tier was built)

- Simulation on 22 recorded Qwen3.6 decode traces: at about 1 GiB of GPU expert memory (a 4 GB card) a profile-chosen static tier gives 1.13x over all experts on the CPU (whole-layer placement: 1.06x); a dynamic tier needs about 90 to 100 expert copies per token and loses (0.83x to 0.88x) once each copy costs 0.178 ms. The +20 % bar is not reached.
- Identity probe (Granite 3B): computing the expert block on the GPU for short batches changes the generated tokens within 3 to 20 tokens, because GPU and CPU kernels round differently. A tier that mixes both cannot be token-identical to stock.
- Decision: no GPU hot tier; GPU use stays with llama.cpp's own placement (`--n-cpu-moe`, chosen by the planner). Caveats: one model's traces, a cost model fitted on an emulation, no real 4 GB card.
