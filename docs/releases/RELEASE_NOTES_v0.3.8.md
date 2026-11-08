# moe-cache v0.3.8 (release date: pending)

**What this release is.** Long prompts are read faster by default, a model that fits in RAM but keeps expert layers on the CPU now runs through the plugin, dense models get a sensible GPU split, the kits bundle our patched llama.cpp build with every patch off, and the benchmark tools were folded into the launcher. Flash-Next IQ3_XXS is the default model again. **Every speed number below was measured on the v0.3.8 head kit on the Windows PC, 2026-11-08, unless it says laptop.** The release-night re-measurement is not in yet: PC night numbers: pending

llama.cpp pin: PIN: b12207+ pending

## Measured

**What these numbers are.** Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM at 6000 MT/s since 2026-11-07, Vulkan), llama.cpp b12064, context 65,536, cold file cache before every run, order stock, moe-cache, moe-cache, stock, two runs per arm, a real seeded 65k conversation with real answers (`tools/conversation_bench.py`). Records from before 2026-11-07 ran at 4800 MT/s and are not comparable. Source: `benchmarks/runs/` and `benchmarks/claims/`; the per-depth tables are in `docs/RESULTS_CURRENT.md`.

| Run | New prompt tokens, stock to moe-cache | Decode, stock to moe-cache |
|---|---|---|
| Flash-Next IQ3_XXS (the default), whole run | 54.4 to 188.6 tok/s | 14.25 to 19.1 tok/s |
| Flash-Next Q2_0 (the faster option), whole run | 227.6 to 434.8 tok/s | 25.4 to 26.9 tok/s (equal to stock at 48k-64k depth, 0.98x) |
| Qwen3.6-35B, fits in RAM, plugin forced on | about the same (1.06x) | about the same (1.07x) |

PC night numbers: pending

## What changed

| Where | Change | Measured |
|---|---|---|
| Long-prompt mode, **on by default** | From 32k context on a graphics card with room the launcher reads prompts in chunks of 4096 tokens (`-ub 4096`) with a q8_0 KV cache and `MOE_CACHE_GPU_MIN_TOKENS=64`; all three are part of what was measured. `--no-long-prompt` goes back to `-ub 2048` and f16 KV; `--exact` gives plain llama.cpp's answers. | Flash-Next IQ3_XXS, 65k conversation, previous default against long-prompt: new prompt tokens 187.4 to 331.8 tok/s (+77.0 %), decode 19.01 to 19.35 tok/s (+1.8 %); 42 expert layers on the CPU before, 45 now. Answers are not byte-identical to stock: KL 0.014-0.015 against f16, same top token in 94-95 % of positions, meaning kept in 10 of 10 checks at 55k |
| Models that fit in RAM | A MoE model that fits in RAM but has expert layers on the CPU runs through the plugin's expert executor; `--stock` or `MOE_CACHE_STOCK_WHEN_FITS=1` opts out. A model whose experts all fit the GPU still runs stock. With no GPU it stays stock. | Laptop with an 8 GB GPU (Qwen3.6-35B-A3B IQ3_XXS): decode +10 to +29 % in six paired band comparisons (mean +15 %), prompt about 10 % lower. 16 GB PC, plugin forced, 65k conversation: 1.06x prompt, 1.07x decode. Without a GPU the launcher stays on stock llama.cpp (checked with a dry run on the laptop with the GPU hidden) |
| Dense models | A model without experts gets llama.cpp's `--fit` with `--fit-target 256`; one bigger than the GPU keeps attention and the KV cache on the GPU and puts the FFN weights of the last layers in RAM (`-ot`; `--no-ffn-ram` undoes it); on Linux hybrid Intel CPUs, when part of the dense model runs on the CPU, 8 decode threads go on 4 performance cores. The planner says "not a MoE" for a dense file. | Laptop 8 GB, Qwen3-14B: 8.2 against 6.3 tok/s with `--fit`; 32k chat decode 4.3 / 3.9 / 3.8 against 4.0 / 1.3 / 1.0 tok/s with FFN in RAM; 5.2 / 5.0 / 5.2 against 4.0 / 3.9 / 3.8 with the thread rule. One laptop, one run each |
| Default model | Flash-Next IQ3_XXS is the default again; Q2_0 is the faster option. | Prompt 3.5x, decode 1.34x of stock in the 65k conversation (IQ3_XXS) |
| llama.cpp kit | The Windows kit and the Linux `-ggml0.26` tarball bundle our patched llama.cpp build (b12207 plus patches 0004, 0020, 0030, 0040). **Every patch is built in and off.** The launcher prints which patches are in at start-up. CI checks that the build answers like the official one at temperature 0. The official build still works. | no speed change claimed |
| Benchmarks | The benchmark tools now live in the launcher: `bench-report --chat`, the Fair comparison button, a ready-to-measure panel, a noise verdict, the results view, `--limit-gb`, and a submit flow (`bench-record ingest-issue`, an issue workflow that checks submitted records). | tooling, no speed claim |
| CI | Downloads in the CI workflows retry and use the cache first, so a download-host outage does not fail a run; the patched-build jobs retry their downloads. | none |
| GPU expert cache | The experimental GPU switches (`MOE_CACHE_HYBRID_ASYNC`, `MOE_CACHE_HYBRID_PERSIST`) are **off by default and not part of this release's measured behaviour**; no GPU tier is built. | none |
| Windows | The scheduling tuning (EcoQoS off, raised CPU and GPU priority) is removed: two paired repeats measured decode -6.8 % and -7.1 %, prompt -0.7 % and +0.4 %. `--no-sched-tuning` is accepted and does nothing for one release. | decode -6.8 % / -7.1 % with it on |
| Archived | Switches and patches that did not win a paired run moved to `archive/` with restore notes (patch 0070, fifteen plugin switches, five more from the 2026-11-07 chain, patch 0080). | none |

## Known limits

The README "Known issues" table lists them. In short: the plugin-for-fits default is measured on one 8 GB-GPU laptop and one PC model (Without a GPU the launcher stays on stock llama.cpp (checked with a dry run on the laptop with the GPU hidden)); the dense-model rules are laptop only, one run each; the `-ub 4096` memory estimate comes from one model (Flash-Next IQ3_XXS at 64k), so another model may need the one automatic retry or `--no-long-prompt`.

## Update

`moe_cache.py update --apply`, or run `install.sh` / `install.ps1` from this release.
