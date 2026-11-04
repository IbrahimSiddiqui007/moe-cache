# Results: Windows desktop, MSI Claw and mini PC, 2026-10-15 to 2026-11-03 (checked 2026-11-04)

Note (2026-10-14): the `bench/` scripts and `moe-cache-bench` named below moved to [moe-cache-benchmark](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark); `bench/gguf_plan.py` is now `planner/gguf_plan.py`.

The Linux laptop results (Results 1-48, 51-53, 57, 66, 70) moved to [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) on 2026-11-04. **Result numbers are the same in both files and were not renumbered**, so "section 44" or "RESULTS 60" anywhere in the repository still means that number; the Contents table below says which file holds it. Newest result first.

## Machines

| Machine | CPU | GPU / VRAM | RAM | OS | Drive | Notes |
|---|---|---|---|---|---|---|
| Windows 11 desktop (the owner's PC) | Ryzen 9 7950X (16 cores, AVX-512) | RX 9070 XT 16 GB (the built-in Radeon is hidden from llama.cpp) | 32 GB DDR5 (31.1 GB visible) | Windows 11 | C: Sabrent NVMe; D: a second drive (model not stated in this file) | Results 54-56, 58-65, 67-69, 71-79. llama.cpp Vulkan b11669 (up to Result 56 and some later runs), then 620d2225c (ggml 0.27), then the official b11966; each result lists what its text names |
| MSI Claw 8 AI+ | Core Ultra 7 258V (4P + 4LPE) | Arc 140V iGPU | 32 GB | Windows 11 | not stated | Result 49-50 (2026-10-15/16): llama.cpp b11669 (Vulkan, ggml 0.27.0); test model OLMoE-1B-7B Q4_0 |
| Mini PC | Ryzen 7 5800H (Zen 3, AVX2) | Vega 8 iGPU (3 GiB carve-out), Vulkan (RADV) | 32 GB DDR4 | Ubuntu 26.04 | not stated | Result 50 (2026-10-16): llama.cpp 23b0202a1 |

## Current state (2026-11-04): where the numbers that apply to the next release are

| Question | Section | Note |
|---|---|---|
| **What is true now, per machine, model and setting; every published number is a claim on recorded runs** | [benchmarks/BOARD.md](../benchmarks/BOARD.md) | new results go to `benchmarks/` (runs, claims, board); this file stays as the numbered history up to Result 79, which is also converted there |
| v0.3.5 against v0.3.4, long prompts, official llama.cpp b11966 server (the numbers to quote) | 79 | prompts +42 % at 34k, +38 % at 64k and 128k; decode -6 % at 34k, -10 % at 64k (Windows PC, Flash-Next Q2_0) |
| The same comparison with a patched server and patch 0022 (not shipped) | 78 | prompts overstated by about 5-7 %; its 64k follow-up shows 39 CPU layers fix the placement (decode 26.1-26.8) |
| The 2026-11-04 overnight chain (decode stock vs moe-cache on six models, 34k / 64k prompts on three Flash-Next files; v0.3.5 kit, llama.cpp b11966, every run not clean) | none yet (records and claims `chain1103-*`) | [benchmarks/BOARD.md](../benchmarks/BOARD.md), the README card and [benchmark_card.html](benchmark_card.html); not written up as a numbered result |
| v0.3.6 (bundles llama.cpp b12064) | none | its speed has not been re-measured; the `chain1103-*` claims show "stale" on the board until a run on v0.3.6 is recorded |
| Windows desktop, stock vs plugin, short context, per model | 59-63 | file-view cache, GLM 2.0x paired (4.44 vs 2.21 tok/s, section 59) |
| Agent sessions, stock vs plugin | 57, 60 | recorded OpenCode / Hermes replays |
| Approximate modes and their quality | 72, and QUALITY_GATE.md | KV q8_0, fewer experts |
| Linux laptop (8 GB card), older versions (v0.1 to v0.2.3.5) | 1-48, 51-53, 57, 66, 70 | now in [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md), kept as history; many were superseded, see the notes at the top of results 20, 54-56, 58, 64, 76 and 78 |

Dates differ between sections on purpose: a result is true for the date, version and machine in its heading. Rules for reading it: a later section on the same machine and model replaces an earlier one when it says so.

Every result starts with a small table: date (from the heading, else from the git history of this file), machine (taken from the old header's list of sections per machine, not re-checked per result), model and quant, kit or commit, llama.cpp build, settings and whether the run was clean. A cell says "not stated" when the result's text does not give it; nothing was filled in by guessing. The result's old heading is kept in small print under the table.

Method: "clean" = CPU <= 58 C at start, no background load, fresh page cache, process memory limit via cgroup (`MemoryMax`, swap off),
4 prompts, temperature 0, seed 42, forward and reverse order; "steady" = mean of requests 2-4. Unclean runs are not reported.
"Stock" = the same llama.cpp without the plugin. Raw data and scripts: `work/t7-plugin/` of the research repo.

## Contents

| Result | Date | Machine | Subject | File |
|---|---|---|---|---|
| [79](#result-79-v035-vs-v034-on-long-prompts-official-b11966-server) | 2026-11-03 | Windows 11 desktop | v0.3.5 vs v0.3.4 on long prompts, official b11966 server | this file |
| [78](#result-78-experimental-vs-v034-on-long-prompts) | 2026-11-02 | Windows 11 desktop | Experimental vs v0.3.4 on long prompts | this file |
| [77](#result-77-baseline-guard-deepseek-glm-mtp-host_out-at-64k-and-the-batch-slowdown-stair) | 2026-10-31 | Windows 11 desktop | Baseline guard, DeepSeek, GLM MTP, HOST_OUT at 64k, and the batch slowdown stair | this file |
| [76](#result-76-long-prompts-plugin-outputs-skip-the-slow-copy) | 2026-10-31 | Windows 11 desktop | Long prompts: plugin outputs skip the slow copy | this file |
| [75](#result-75-buffered-reads-on-the-private-memory-path-and-the-deepseek-v41-stop-after-10-tokens) | 2026-10-31 | Windows 11 desktop | Buffered reads on the private-memory path, and the DeepSeek V4.1 "stop after 10 tokens" | this file |
| [74](#result-74-idle-cache-refill-and-the-expert-pack-with-file-views) | 2026-10-30 evening | Windows 11 desktop | Idle cache refill and the expert pack with file views | this file |
| [73](#result-73-v032-checks-on-the-windows-pc) | 2026-10-30 afternoon | Windows 11 desktop | v0.3.2 checks on the Windows PC | this file |
| [72](#result-72-approximate-package-b-behind-a-quality-gate-q8_0-kv-and-fewer-experts-on-flash-next-q2_0) | 2026-10-30 night | Windows 11 desktop | Approximate package B behind a quality gate: q8_0 KV and fewer experts on Flash-Next Q2_0 | this file |
| [71](#result-71-bigfix-desktop-gpu-memory-time-between-our-graph-splits-executor-phases-depth-decode) | 2026-10-29 night | Windows 11 desktop | bigfix: desktop GPU memory, time between our graph splits, executor phases, depth decode | this file |
| 70 | 2026-10-29 night | Linux laptop | Diffusion MoE: prefill proxy and LLaDA-MoE | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| [69](#result-69-afternoon-pc-queue) | 2026-10-29 10:11-17:00 | Windows 11 desktop | Afternoon PC queue | this file |
| [68](#result-68-nemotron-35-bolt-the-35-s-per-tiny-request-explained-and-short-prompts-on-stock-runs) | 2026-10-29 | Windows 11 desktop | Nemotron 3.5 Bolt: the "3.5 s per tiny request" explained, and short prompts on stock runs | this file |
| [67](#result-67-overnight-chain-on-the-windows-pc-n29-n22-n15-n30-n26-n27-n23-long-context-bolt-hip-n25c-n18) | 2026-10-28 23:32 -> 10-29 08:01 | Windows 11 desktop | Overnight chain on the Windows PC: N29, N22, N15, N30, N26, N27, N23, long context, Bolt, HIP, N25c, N18 | this file |
| 66 | 2026-10-27 night | Linux laptop | Linux file-view cache made the default | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| [65](#result-65-new-models-on-the-windows-pc-nemotron-35-bolt-74b-a9b-deepseek-v4-flash) | 2026-10-28 evening | Windows 11 desktop | New models on the Windows PC: Nemotron 3.5 Bolt 74B-A9B; DeepSeek V4-Flash | this file |
| [64](#result-64-long-context-on-the-windows-pc-64k-and-128k-on-flash-next-q2_0-the-122b-and-qwen36) | 2026-10-28 evening | Windows 11 desktop | Long context on the Windows PC: 64k and 128k+ on Flash-Next Q2_0, the 122B and Qwen3.6 | this file |
| [63](#result-63-flash-next-quant-choice-on-the-windows-pc-q2_0-iq3_s-coder-iq1_m-vs-iq2_xs--iq3_xxs) | 2026-10-28 | Windows 11 desktop | Flash-Next quant choice on the Windows PC: Q2_0, IQ3_S, Coder IQ1_M vs IQ2_XS / IQ3_XXS | this file |
| [62](#result-62-glm-max-on-the-windows-pc-drive-gpu-layers-mtp-long-prompts-and-why-code-is-slow) | 2026-10-28 | Windows 11 desktop | GLM max on the Windows PC: drive, GPU layers, MTP, long prompts, and why code is slow | this file |
| [61](#result-61-windows-desktop-list-a) | 2026-10-27/28 | Windows 11 desktop | Windows desktop: list A | this file |
| [60](#result-60-windows-desktop-agent-sessions-stock-vs-plugin-with-and-without-mtp) | 2026-10-27 night | Windows 11 desktop | Windows desktop: agent sessions, stock vs plugin, with and without MTP | this file |
| [59](#result-59-windows-desktop-the-cache-moves-to-read-only-views-of-the-model-file-and-beats-stock) | 2026-10-27 evening | Windows 11 desktop | Windows desktop: the cache moves to read-only views of the model file and beats stock | this file |
| [58](#result-58-windows-desktop-fixes-confirmed-mtp-flash-next-iq2_xs-vs-iq3_xxs-glm-page-file) | 2026-10-27 | Windows 11 desktop | Windows desktop: fixes confirmed, MTP, Flash-Next IQ2_XS vs IQ3_XXS, GLM page file | this file |
| 57 | 2026-10-26 evening | Linux laptop | Agent-shaped benchmark: replayed OpenCode and Hermes sessions, stock vs moe-cache | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| [56](#result-56-windows-desktop-experiments) | 2026-10-26/27 | Windows 11 desktop | Windows desktop experiments | this file |
| [55](#result-55-windows-desktop-clean-wp2-apps-closed-file-cache-flushed-before-every-run) | 2026-10-26 | Windows 11 desktop | Windows desktop, clean WP2: apps closed, file cache flushed before every run | this file |
| [54](#result-54-windows-11-desktop-ryzen-9-7950x--rx-9070-xt) | 2026-10-26 | Windows 11 desktop | Windows 11 desktop: Ryzen 9 7950X + RX 9070 XT | this file |
| 53 | 2026-10-26 afternoon | Linux laptop | Laptop ideas under a day: I15 at 12 GB, I18 starter profiles, I11 split, I35 by calculation | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 52 | 2026-10-26 | Linux laptop | Long-prompt block fell back to the CPU at context 8192, Flash-Next prefill at 8192, kill checks I2 / I28 / B1 | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 51 | 2026-10-26 | Linux laptop | Coding agents on moe-cache: OpenCode and Hermes Agent, agent mode and warm sessions | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| [50](#result-50-windows-round-2-and-the-first-amd-machine) | 2026-10-16 | MSI Claw 8 AI+ | Windows round 2 and the first AMD machine | this file |
| [49](#result-49-first-run-on-real-windows-and-the-igpu-experiments-e1e2) | 2026-10-15 | MSI Claw 8 AI+ | First run on real Windows, and the iGPU experiments E1/E2 | this file |
| 48 | 2026-10-15 ~21:00 | Linux laptop | Long prompts keep part of the cache | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 47 | 2026-10-15 evening | Linux laptop | Long contexts, moe-cache only | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 46 | 2026-10-15 evening | Linux laptop | Prompt processing, stock vs moe-cache | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 45 | 2026-10-15 | Linux laptop | Qwen3.5-122B-A10B at three bit widths | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 44 | 2026-10-15 afternoon | Linux laptop | Re-measured for the README | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 43 | 2026-10-15 | Linux laptop | Quant yardstick for `--fast`: two 2-bit quants of the same model | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 42 | 2026-10-15 | Linux laptop | Section 2 re-measured, and the executor slowing down K-quant models | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 41 | 2026-10-15 | Linux laptop | GLM-5.3-Flash decode reads: the drive is idle, not slow | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 40 | 2026-10-15 | Linux laptop | GLM-5.3-Flash with its own MTP layer | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 39 | 2026-10-15 | Linux laptop | Prompt-lookup drafting on code edits | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 38 | 2026-10-15 | Linux laptop | Prompt reads merged | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 37 | 2026-10-15 | Linux laptop | OLMoE-1B-7B Q4_0 at a 3 GB limit: repacking | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 36 | 2026-10-15 | Linux laptop | Identity on llama.cpp 23b0202a1: six more models and a second race | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 35 | 2026-10-14 | Linux laptop | Prompt speed and time to first token on the default path | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 34 | 2026-10-14 | Linux laptop | Hybrid on GLM-5.3-Flash and on Qwen3.6-35B at 12 GB | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 33 | 2026-10-14 | Linux laptop | Adaptive hybrid bank | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 32 | 2026-10-14 | Linux laptop | Prompt speed with the hybrid | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 31 | 2026-10-14 | Linux laptop | Hybrid quality, and an exactness race at 8-token batches | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 30 | 2026-10-14 | Linux laptop | Hybrid hit/miss prototype: GPU bank + CPU executor in parallel | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 29 | 2026-10-14 | Linux laptop | Fits in RAM vs not, with llama.cpp's GPU expert cache in both | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 28 | 2026-10-14 | Linux laptop | Flash-Next settings on the 8 GB card | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 27 | 2026-10-14 | Linux laptop | Executor in a build with separately loaded backends | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 26 | 2026-10-14 | Linux laptop | GLM-5.3-Flash UD-IQ2_XXS on the 31 GB / 8 GB laptop | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 25 | 2026-10-12 | Linux laptop | Quality evidence: quantisations, thinking, approximate routing, code generation | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 24 | 2026-10-12 | Linux laptop | Prompt reuse, saved contexts, where decode time goes | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 23 | 2026-10-12 | Linux laptop | Why prefill was 4x slower with the GPU cache, and the fix | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 22 | 2026-10-11 | Linux laptop | Coding-session benchmark: prompt reuse on a 20k-token context | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 21 | 2026-10-11 | Linux laptop | Approximate routing, SSD-bound setting | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 20 | 2026-10-11 | Linux laptop | Long context, long sessions and prompt processing on Flash-Next | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 19 | 2026-10-11 | Linux laptop | Small-batch executor, GPU cache third and fourth round | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 18 | 2026-10-11 | Linux laptop | GPU expert cache with the plugin, quant comparison, MTP and neuron sparsity | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 17 | 2026-10-11 | Linux laptop | Settings sweep, prefill and pre-fault | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 16 | 2026-10-11 | Linux laptop | Warm start, read speed and page faults on Qwen3.8-Flash-Next IQ2_XS | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 15 | 2026-10-11 | Linux laptop | New llama.cpp pin and upstream's own MoE cache | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 14 | 2026-10-10 | Linux laptop | Qwen3.8-Flash-Next IQ2_XS, 24 GB limit, 3 expert layers on the GPU: kernel and executor | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 13 | 2026-10-10 | Linux laptop | Small models, CPU only, hard memory limit | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 12 | not recorded (git: 2026-10-09) | Linux laptop | A model 2x bigger than RAM: gpt-oss-120b | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 11 | not recorded (git: 2026-10-09) | Linux laptop | CPU thread pinning on a hybrid CPU | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 10 | not recorded (git: 2026-10-09) | Linux laptop | Robustness tests | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 9 | not recorded (git: 2026-10-08) | Linux laptop | Memory-limit test | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 8 | not recorded (git: 2026-10-08) | Linux laptop | Forced repacking with a GPU present | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 7 | not recorded (git: 2026-10-08) | Linux laptop | Prompt processing | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 6 | not recorded (git: 2026-10-08) | Linux laptop | Two-times-RAM model on a small machine | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 5 | not recorded (git: 2026-10-08) | Linux laptop | No dedicated GPU, memory limited | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 4 | not recorded (git: 2026-10-08) | Linux laptop | CPU-only machine, DeepSeek-V2-Lite, model fits in RAM | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 3 | not recorded (git: 2026-10-08) | Linux laptop | Other models, steady tok/s, all experts on the CPU | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 2e | not recorded (git: 2026-10-08) | Linux laptop | Huge pages | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 2d2 | not recorded (git: 2026-10-08) | Linux laptop | KAT Coder with its warm-start profile | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 2d | not recorded (git: 2026-10-08) | Linux laptop | KAT Coder 35B-A3B, 28 GB | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 2c | not recorded (git: 2026-10-08) | Linux laptop | Where the experts live matters | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 2b | not recorded (git: 2026-10-08) | Linux laptop | Qwen3.6 with the automatic cache size | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 2 | not recorded (git: 2026-10-08) | Linux laptop | Qwen3.6 35B-A3B, steady tok/s | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |
| 1 | not recorded (git: 2026-10-08) | Linux laptop | Token identity | [RESULTS_OLD_LAPTOP.md](RESULTS_OLD_LAPTOP.md) |

## Windows summary notes (moved from the README on 2026-10-30)

**Long context (Windows PC, RESULTS 64; agent mode, one cold run per point, not ABBA).** Flash-Next Q2_0 at a 64k setting: moe-cache reads a 63.5k-token
prompt 1.8x faster than stock (about 289 vs 525 s; prefill 266 / 195 tok/s at 4-21k / 53-63k vs 171 / 106) and decodes 20.7 vs 16.8 tok/s at that depth
(+23 %). Qwen3.5-122B with moe-cache at 64k: prefill 266 / 217, decode 17.3 at 63.5k. Qwen3.6 (fits, stock): 848 / 739, decode 59.1. At the 256k setting the
planner then kept the long-prompt GPU block its room (RESULTS 67 / 69, paired runs): Flash-Next prefill 108-110 tok/s at 4-34k and decode **21.9 vs 18.5 tok/s** at 4k (+18 %) over the old plan; decode at 138k depth was 15.9 with the old plan (not re-measured). **Superseded for v0.3.5 (2026-11-03):** the launcher now uses `-ub 2048` above 32k context, including 256k; prompts read at 421-448 tok/s at 64k and 421 at 128k (section 79), and a 222k-token prompt at 256k reads at 334 tok/s (section 79).

**Other models (Windows PC, RESULTS 65, one cold run per point).** Nemotron 3.5 Bolt 74B-A9B (NVIDIA's 2-bit QAT file, 22.0 GB) fits in RAM + VRAM,
so the launcher runs stock llama.cpp for it (needs llama.cpp b11966 or newer): decode 27.9 tok/s (23-25 at 114k-228k tokens), prompts 595 tok/s at
57k and 721 at 114k, 190/202 tasks (Flash-Next Q2_0 191, not proven), OpenCode session 47.8 s (the fastest measured). moe-cache forced on it is slower
(decode 22.1, prompt 329 at 57k): the plugin is for models bigger than memory. DeepSeek V4-Flash: not tested.

## Result 79. v0.3.5 vs v0.3.4 on long prompts, official b11966 server

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-11-03 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | Flash-Next Q2_0 (D:) | versions and commits named in the text: v0.3.5, v0.3.4, 1309fcd | mentioned in the text: b11966 | clean: PC restarted at 02:03, baseline guard PASS at 49.4 tok/s, runs 02:30-03:25 | Cleanliness not recorded: the old heading says clean, but the guard log's “not clean: owner work running” line (02:09, 2026-11-03) is fixed harness text, not a check, so neither is proven; the record has no other-programs field |

<sub>Correction (2026-11-04): the heading below called this run clean; the guard log's “not clean” line is fixed harness text, so cleanliness is not recorded (see the Clean cell).</sub>

<sub>Old heading: 79. v0.3.5 vs v0.3.4 on long prompts, official b11966 server (Windows PC, 2026-11-03; clean: PC restarted at 02:03, baseline guard PASS at 49.4 tok/s, runs 02:30-03:25)</sub>

Windows 11, Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5, Flash-Next Q2_0 on D:, cold file cache before each run. Both versions ran on the **official llama.cpp b11966 server**, without patch 0022 (section 78 used a patched server, so its prompt numbers overstate v0.3.5 by about 5-7 %; these replace them). Order new old old new, two runs per cell, shown as first / second; the averages are my arithmetic.

| Context | Prompt v0.3.4 | Prompt v0.3.5 | Change | Decode v0.3.4 | Decode v0.3.5 | Change |
|---|---|---|---|---|---|---|
| 34k | 408.6 / 403.1 | 577.0 / 572.6 | +42 % | 28.43 / 28.59 | 27.04 / 26.38 | -6 % |
| 64k | 313.8 / 314.1 | 421.3 / 448.3 | +38 % | 27.42 / 27.04 | 25.69 / 23.44 | -10 % |
| 128k | 304.8 / 305.5 | 420.7 / 421.9 | +38 % | not measured (the answer ended after 2 tokens) | not measured | - |

**The decode loss is real.** After a long prompt v0.3.5 decodes 6-10 % slower than v0.3.4. Cause: the bigger prompt chunk (`-ub 2048`) takes GPU memory, so 2 fewer expert layers fit on the GPU. At 64k the launcher keeps 39 expert layers on the CPU on a nearly empty card and 40 when other programs hold about 1 GB; v0.3.4 had 37-38. Check run (64k): 2 more GPU layers with `-ub 512` (38 CPU layers) decode 26.63 / 26.92, against 23.74 / 22.91 with `-ub 2048` and 40 layers.

Other runs the same night (not paired with v0.3.4):

| Item | Result |
|---|---|
| 256k setting (222k-token prompt), prompt tok/s | `-ub 2048` 334.8 / 333.2 vs `-ub 1024` 263.4 / 265.8 (+26 %). The planner now picks 2048 at 256k (commit 1309fcd) |
| Qwen3.5-122B IQ2_XXS, 64k, prompt tok/s | new default 538.9 / 551.5 vs `-ub 1024` 438.7 / 437.3 (+24 %) |
| GLM-5.3-Flash, 30k, prompt tok/s | 72.8 / 73.1 vs 44.9 / 45.0 (+62 %); decode 2.88 vs 3.06 (-6 %) |
| Planner rule `MOE_CACHE_PLAN_RULE=1`, 128k, prompt tok/s | 403.5 / 403.6 vs 418.7 / 423.4 without (-4 %): stays off by default |
| Windows priority switches, 34k | prompt +2.0 % (598.2 vs 586.4, average over two pairs), decode the same |

Text was not compared in this run. Only these models and depths are measured. Source: PC overnight run of 2026-11-03 (A B B A per row).

## Result 78. Experimental vs v0.3.4 on long prompts

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-11-02 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | Flash-Next Q2_0 (D:) | unreleased experimental code (next release, later v0.3.5) against the released v0.3.4 | b11966 with patch 0022 (a patched server, not shipped) | next release; clean: PC restarted at 11:47, baseline guard PASS at 48.6 tok/s, runs 13:19-14:13 | Cleanliness not recorded: the old heading says clean, but the guard log's “not clean: owner work running” line (12:02, 2026-11-02) is fixed harness text, not a check, so neither is proven; the record has no other-programs field |

<sub>Correction (2026-11-04): the heading below called this run clean; the guard log's “not clean” line is fixed harness text, so cleanliness is not recorded (see the Clean cell).</sub>

<sub>Old heading: 78. Experimental (next release) vs v0.3.4 on long prompts (Windows PC, 2026-11-02; clean: PC restarted at 11:47, baseline guard PASS at 48.6 tok/s, runs 13:19-14:13)</sub>

**Superseded by section 79:** this run used a patched server with patch 0022 (not shipped in v0.3.5), so its prompt numbers are about 5-7 % too high; section 79 repeats the comparison on the official server. The 64k follow-up below (39 CPU layers) is still the placement evidence.

Windows 11, Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5, Flash-Next Q2_0 on D:, cold file cache before each run, order new old old new. "v0.3.4" is the released kit with its own llama-server. "Experimental" is the unreleased launcher defaults (prompt chunk 2048 above 32k context, GPU_SUM on) plus llama.cpp patch 0022 (`GGML_VK_COUNT_EXPERTS_FAST=1`) on the patched b11966 server; these defaults are not in any release yet. Two runs per cell, shown as first / second; the averages are my arithmetic.

| Context, prompt | v0.3.4 prompt tok/s | Experimental prompt tok/s | Prompt change (averages) | v0.3.4 decode tok/s | Experimental decode tok/s | Decode change (averages) |
|---|---|---|---|---|---|---|
| 40960, 34k tokens | 392.2 / 411.0 | 637.2 / 619.7 | +56 % | 28.55 / 28.48 | 27.29 / 27.67 | -3.6 % |
| 65536, 64k | 311.9 / 333.9 | 434.4 / 436.6 | +35 % | 26.29 / 26.10 | 22.95 / 21.72 | **-15 %** |
| 131072, 128k | 306.8 / 306.0 | 436.3 / 437.5 | +43 % | not recorded | not recorded | - |

Decode after a 64k prompt is 15 % lower with the experimental defaults than with v0.3.4, in both pairs; the cause is not found. Prompts are faster at all three depths. Text was not compared in this run. Only this model and these three depths are measured.

**Follow-up, same boot, 16:05-16:26 (64k, `-ub 2048`, patch 0022, one run per arm except two for 39 and 40):** 41 CPU layers 454.3 / decode 22.34; 40 layers 464.3, 476.5 / 23.69, 25.40; **39 layers 457.7, 454.6 / 26.82, 26.13**; `-ub 1024` with the launcher's 38 layers 415.4 / 25.60. So the 64k decode loss was the placement. The planner now reserves for 39 at 64k.

## Result 77. Baseline guard, DeepSeek, GLM MTP, HOST_OUT at 64k, and the batch slowdown stair

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-31 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | DeepSeek, GLM-5.3-Flash | not stated | not stated in this section | healthy PC, baseline 48.1 tok/s at 12:29, PC up 1-5 h; **not chart numbers** | Not stated |

<sub>Old heading: 77. Baseline guard, DeepSeek, GLM MTP, HOST_OUT at 64k, and the batch slowdown stair (Windows PC, 2026-10-31; healthy PC, baseline 48.1 tok/s at 12:29, PC up 1-5 h; **not chart numbers**)</sub>

Source: handover section 4 of 2026-11-01. One run per row unless stated; these are checks, not chart numbers.

| Item | Result | State |
|---|---|---|
| B1 baseline guard | `baseline_guard.ps1`: the faster of 2 stock Qwen3.6 runs, limit 44 tok/s; a fail stops the queue. First real run passed | done |
| B2 DeepSeek | Not a bug: the raw `/completion` call hits the end token, chat is correct (114 tokens, 6.1 tok/s, one run). The 0xFFFFFFFF exit code was our own Stop-Process | done, the launcher names it |
| B3 GLM MTP, ABBA | water prompt 5.91 -> 7.83 tok/s (+32 %), code prompt 2.65 -> 2.59 (-2.3 %). The code text was **different** with the same split, so the launcher's "same output" was wrong | default unchanged (on in `--agent`); wording now says the output can differ slightly for some models |
| B4 buffered reads, private path | see section 75 | merged, Windows only |
| HOST_OUT (E1 fix) | Flash-Next: 34k prompt 286 -> 395 / 396 tok/s, wall 119 -> 86.5 s, text identical; 64k prompt 229 -> 299 / 306 tok/s. Section 76 covers the 34k row only | merged, default on, `MOE_CACHE_HOST_OUT=0` turns it off |
| Open worry | Decode after a 64k prompt: 25.39 tok/s off vs 24.59 / 24.33 on (-3.5 %), one off-run. At 34k it was +0.7 %. To repeat in the next chain | not settled |
| Batch slowdown 1.7 %, stair | **D08** (pool counters on separate cache lines) cost **-1.3 / -1.5 %** in both pairs; D12 about 0; up to D03 about 0; C15 + B05 about -0.3 % from one clean run; D10, D16, D07 about 0 | D08 reverted |

## Result 76. Long prompts: plugin outputs skip the slow copy

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-31 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | not in the heading; mentioned in the text: Flash-Next, Q2_0 | not stated | not stated in this section | baseline 48.1 tok/s that day, PC up about 4 h, so not chart numbers yet | Not stated |

<sub>Old heading: 76. Long prompts: plugin outputs skip the slow copy (Windows PC, 2026-10-31; baseline 48.1 tok/s that day, PC up about 4 h, so not chart numbers yet)</sub>

**Superseded for numbers to quote by sections 78-79** (this run was "not chart numbers": PC up about 4 h).

A sampler on the graph thread found 35 % of the time between plugin graphs in `ggml_backend_tensor_copy`'s slowest path (malloc, get, set, free), because the plugin's buffer type did not report host. The table is the first version, which changed the type after the first copy. The shipped version picks the type when the buffer is created, through the public API (compute buffers are host, weights are not); it has not been re-measured on the PC.

| Flash-Next Q2_0, ctx 40960, same DLL, 34k-token prompt | Prompt tok/s | Prompt wall s | Decode after, tok/s | Text |
|---|---|---|---|---|
| Fix on | 395.4 | 86.6 | 28.56 | identical |
| Fix off | 286.3 | 119.3 | 28.34 | identical |
| Fix on | 396.4 | 86.3 | 28.53 | identical |

One model, 34k tokens only. Other models and depths are not measured.

## Result 75. Buffered reads on the private-memory path, and the DeepSeek V4.1 "stop after 10 tokens"

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-31 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | DeepSeek V4.1 | versions and commits named in the text: v0.3.3 | mentioned in the text: b11966 | see the section | Not stated |

<sub>Old heading: 75. Buffered reads on the private-memory path, and the DeepSeek V4.1 "stop after 10 tokens" (Windows PC, 2026-10-31)</sub>

Same PC. The buffered row comes from the night chain of 10-31, when the PC ran about 18 % slow (fixed by a restart): stock and moe-cache alike, so only the relative result counts, no absolute number here is for a chart.

| Test | Result | Kept |
|---|---|---|
| Flash-Next Q2_0, `MOE_CACHE_WIN_VIEW=0`, ABBA: unbuffered vs `MOE_CACHE_WIN_BUFFERED=1` | reader speed 0.26 / 0.26 -> 0.84 / 0.84 GB/s per reader thread; decode +39 %, prompts about 2x | **yes**: the default whenever the file views are not used |
| Qwen3.6-35B, default views + repacked weights, same switch | reader speed 8.3-8.6 GB/s in all four runs: no difference | nothing changes on the default path |
| DeepSeek-V4.1-Flash-1017 IQ1_M + Engram, v0.3.3 kit, b11966, ctx 4096, one run on the restarted PC | a chat request gives a correct, complete answer (114 tokens, `finish=stop`). The "stop after 10 tokens" was the model's own end token: the check sent raw `/completion` text with no chat template, and the top-5 next tokens show that end token first (probability 0.44) on the water prompt. The server's exit code 0xFFFFFFFF was the test script's own `Stop-Process` | **no bug**: use chat requests for this model. The launcher now names that exit code |

## Result 74. Idle cache refill and the expert pack with file views

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-30 evening | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | Flash-Next Q2_0 | not stated | mentioned in the text: b11669 | BO-2; BO-1b; Windows PC; "not clean": the owner's programs were running | No (heading says not clean) |

<sub>Old heading: 74. Idle cache refill (BO-2) and the expert pack with file views (BO-1b) (Windows PC, 2026-10-30 evening; "not clean": the owner's programs were running)</sub>

Same PC (7950X, RX 9070 XT 16 GB, 32 GB), llama.cpp Vulkan b11669, Flash-Next Q2_0, ABBA pairs, text compared in every run.

| Test | Result | Kept |
|---|---|---|
| BO-2 idle refill (`MOE_CACHE_IDLE_REBALANCE`), the recorded OpenCode session with its tool pauses x4 | decode 12.6 -> 13.0 and 13.2 -> 13.5 tok/s; wait for the follow-up answers 26.3 -> 26.1 and 32.0 -> 23.9 s; text identical | **yes**: on by default since 0.3.3 (it only changes which experts sit in RAM) |
| BO-1 expert pack, private-memory cache | reader speed 0.21 -> 1.19 GB/s per thread, decode 1.7-2.1 -> 5.7-6.6 tok/s | opt-in only: the private-memory cache is not the default |
| BO-1b expert pack with the default file-view cache | reader speed 1.11 -> 1.17, decode 15.6-16.1 -> 14.5-15.1 tok/s, short prompts slower | **no**: views already read near full speed |

## Result 73. v0.3.2 checks on the Windows PC

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-30 afternoon | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | not in the heading; mentioned in the text: Flash-Next, Q2_0 | versions and commits named in the text: v0.3.2, b3a6e45, e212dfa | mentioned in the text: b11669, b11966 | all "not clean": the owner's programs held ~15 GB RAM and some VRAM; one queue, kits e212dfa / b3a6e45 | No (heading says not clean) |

<sub>Old heading: 73. v0.3.2 checks on the Windows PC (2026-10-30 afternoon; all "not clean": the owner's programs held ~15 GB RAM and some VRAM; one queue, kits e212dfa / b3a6e45)</sub>

Same PC (7950X, RX 9070 XT 16 GB, 32 GB), llama.cpp Vulkan b11669 (C:\llama-vk) unless noted. Decode = tok/s after the prompt, 256 greedy tokens. ABBA = A, B, B, A; pair 1 = runs 1/2, pair 2 = runs 4/3.

| Test | Result |
|---|---|
| Crash A/B (old 673fa23 vs new e212dfa DLL, Flash-Next Q2_0, ctx 40960, 2 requests) | no crash in 8 runs; text identical in every pair; decode 17.1-17.4 both. The buffer-layout fix (e212dfa) changes nothing on b11669 |
| README numbers (item 96/98za, one cold run each) | **not usable**: at 5-7 GB "available" llama-server's prompt cache failed to allocate ("bad allocation", ~230 MB) and llama.cpp then wrote through NULL (memcpy in ggml-base from llama.dll; the plugin's crash handler: outside the expert cache and its graphs) at the 3rd request: Windows commit (RAM + page file) was exhausted. 256k: prompts 77-82 tok/s, decode 11 (README 108-110 / 21.9) with a 11.9 GiB cache. README tables unchanged; clean rerun pending |
| b11966 (C:\llama-new, ggml 0.28) | new launcher picks the API 3 DLL and the cache runs (granite, check --run); the 673fa23 launcher reported the misleading "DLL not found" |
| D5 lookahead 122B, ctx 40960 (MOE_CACHE_LOOKAHEAD=2) | pair 1 9.36 -> 9.77, pair 2 13.32 -> 10.86: not kept |
| E3 fused gate+up (MOE_CACHE_EXEC_FUSE) | water 17.31 -> 17.62 / 17.29 -> 17.34; code 17.61 -> 18.06 / 17.57 -> 17.69; timing run 17.26 -> 17.64 / 17.22 -> 17.52: **all 6 pairs**, text identical: on by default |
| ClassLRU (MOE_CACHE_POLICY=classlru) | 122B: water 10.93 -> 11.21 / 10.89 -> 11.02, code 10.45 -> 10.75 / 10.40 -> 10.60, hit +1.5-2.1 points; Flash-Next (95 % hits): -0.4 to +1.4 %, mixed; text identical everywhere: opt-in |
| BO-1 read size (tests/win_read_bench.cpp, random unbuffered reads, Flash-Next Q2_0 file on D:) | 896 KiB 1.90 GB/s (~2,070 ops/s at 4 and 8 threads), 1792 KiB 3.04, 2688 KiB 3.47 GB/s: whole-expert reads +83 % |
| Final kit (b3a6e45 CI zip, unzipped fresh) | version, control page, pick, server without a model, on-demand status, check --run (granite), bench-report (granite, forced, warm): 7/7 pass |

## Result 72. Approximate package B behind a quality gate: q8_0 KV and fewer experts on Flash-Next Q2_0

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-30 night | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | Flash-Next Q2_0 | not stated | not stated in this section | B4; B3; Windows PC, , all "not clean": owner desktop used VRAM; branch claude/approx | No (heading says not clean) |

<sub>Old heading: 72. Approximate package B behind a quality gate: q8_0 KV (B4) and fewer experts (B3) on Flash-Next Q2_0 (Windows PC, 2026-10-30 night, all "not clean": owner desktop used VRAM; branch claude/approx)</sub>

Rule change (owner 2026-10-29): token identity to stock is no longer required; the meaning has to match. Thresholds were written before measuring (`docs/QUALITY_GATE.md`).
Gate = `tools/quality_gate.py`: mean KL divergence and same-top-token against stock (3 x 512 tokens of long30k.txt, batch 8, f16-KV stock as base) and the 202 tasks of `tools/quality_tasks.json`.
Flash-Next Q2_0, `--n-cpu-moe 46`, ctx 4096, thinking off.

| Level | Settings | Mean KLD (limit) | Same top token (limit) | Tasks right (stock 190/202; limit) | Loops | Verdict |
|---|---|---|---|---|---|---|
| kv (B4) | q8_0 KV | **0.0079** +- 0.0005 (0.015) | **96.7 %** +- 0.6 (95) | 190 (>= 188) | 0 | **PASS** |
| fewer (B3) | `--fast` (route skip) + q8_0 KV + 2 lowest-ranked experts dropped on CPU-held layers | **0.0338** +- 0.0021 (0.060) | **92.0 %** +- 1.0 (88) | 192 (>= 185) | 0 | **PASS** |

Speed, ABBA with the text check, ctx 262144, `--gpu-margin 1.5` on both arms (tok/s; pair = A, B, B, A):
| B4: default (A) vs `--fast=kv` (B) | A | B | Verdict |
|---|---|---|---|
| prefill 4k | 84.9, 86.3 | 169.2, 167.4 | B 2.0x (the tuner puts the freed KV into more GPU layers and -ub 1024: 184 instead of 376 long-prompt segments) |
| prefill 64k | 72.7, 72.9 | 144.0, 145.7 | B 2.0x |
| prefill 128k | > 30 min (timed out twice) | 136.1, 135.0 (about 16 min) | B at least 2x |
| decode 4k | 16.95, 17.23 | 17.05, 17.0 | equal (pair 1 +0.6 %, pair 2 -1.3 %: noise) |
| decode 64k | 14.17, 14.05 | 15.68, 15.92 | B +11 % / +13 % |
| decode 128k | not measured (A timed out) | - | - |

| B3: `--fast` (A) vs `--fast=fewer` (B) | A | B | Verdict |
|---|---|---|---|
| decode 4k | 17.97, 18.51 | 19.05, 18.95 | B +6 % / +2 % |
| decode 64k, 128k | 16.54, 16.82 / 6.7 | no value (the continuation ended at once: 0 tokens) / 10.73, 10.39 | no clean pair: unproven |
| prefill | 156 / 135 / 129 | 158 / 136 / 129 | equal (only decode batches are affected) |

Reading: B4 passes the gate with room to spare and doubles prefill at ctx 262144 (that comes from the tuner's layer/micro-batch plan, not from the q8 arithmetic); decode gain +11-13 % at 64k, equal at 4k.
B3 passes the gate (KLD in the `--fast` range, tasks not worse) but its speed is shown only at 4k (+2-6 %); the 64k / 128k pairs lost their decode value, so it stays opt-in and **unproven at 64k-256k**.
The A arms at 128k run into the 30-minute limit: the default plan at ctx 262144 is slow there, which is part of what B4 fixes.
CI (`quality.yml`, granite): `exact` enforced (KLD 0, 100 %, 93/93 tasks); stock llama-perplexity at the API 3 pin aborts with `-ctk q8_0` on the CPU-only build (stock too), so approximate levels there are report-only smoke without KV.
Runner bug found on the way: `runner.ps1` `$o = Get-CimInstance` overwrites `$O`, which kills `Script1` items at once.

## Result 71. bigfix: desktop GPU memory, time between our graph splits, executor phases, depth decode

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-29 night | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | Flash-Next Q2_0 (D:) | not stated | not stated in this section | all "not clean": owner work took 3-6 GB of RAM; branch claude/bigfix | No (heading says not clean) |

<sub>Old heading: 71. bigfix: desktop GPU memory, time between our graph splits, executor phases, depth decode (Windows PC, 2026-10-29 night, all "not clean": owner work took 3-6 GB of RAM; branch claude/bigfix)</sub>

Flash-Next Q2_0 (D:), ctx 131072, launcher defaults, a separate Edge window open (about 1.2 GiB of VRAM). Raw logs on the PC: C:\dl\abba\bigfix_*.

**1. Long-prompt block with the desktop holding GPU memory** (A = experimental 5e6ecd0, B = bigfix ff30110, ABBA, prefill tok/s):
| Prompt | A | B | GPU blocks A -> B |
|---|---|---|---|
| p34k pair 1 | 68.3 | 110.2 | 78 -> 1428 |
| p34k pair 2 | 67.4 | 109.0 | 14 -> 1428 |
| p128k pair 1 | 75.4 | 94.6 | 2092 -> 4956 |
| p128k pair 2 | 66.6 | 79.4 | 0 -> 4956 |
B is faster in every pair. Cause (A's log): "a long-prompt block needs 256 MiB of GPU memory but only 171 MiB are free: it stays on the CPU". B splits the block into 2-3 parts of 211-238 experts (329-408 MiB free). Text: B vs B identical at both sizes; A vs A already differs (its GPU/CPU split differs run to run); all answers summarise the same document with the same meaning. Absolute speeds are far below the clean ~290 tok/s (memory pressure): read the ratios, not the numbers. The launcher change (margin 0.25 + 0.5 x VRAM that other programs already hold, 0.5-2.5 GB) is unit-tested only (tests/desk_margin_test.py); this run does not separate it from the split.

**2. Where the time outside our backend goes (p34k, B, 101.8 tok/s, wall 335 s, MOE_CACHE_SEG_TIMES):**
| Part | Seconds | Share of 346 s |
|---|---|---|
| inside our backend | 210 | 61 % |
| of which GPU blocks (upload+compute+read-back) | 180 | |
| of which cache fill (prep) | 26 | |
| of which llama.cpp's own CPU path | 4 | |
| between our splits (gaps) | 136 | 39 % |
| of which gaps >= 5 ms | 131 | |
Page faults of the process in the prompt: 125M, 88M inside our backend (uploads read the mapped model file) and 37M inside the gaps. The longest gaps (0.58 s) carry 31-32k faults each, which is the PLE table (about 16 rows per token x 2048 tokens = 32k lookups per micro-batch, deepdive row SM1): at 17 micro-batches that is about 10 s, 3 % of the prompt. The other 120 s in the gaps is llama.cpp's other splits and GPU waiting (the earlier perf-logger run named 22 s of Vulkan kernels at 34k): not resolved. The first run (51) was polluted (hit rate 45 %, 128k timed out at 45 min) and is not used.

**3. Executor phases in decode (4k run, 16 threads, MOE_CACHE_EXEC_TIMES; hit rate 66 %, so many experts were still being read):**
| Phase | Share |
|---|---|
| dot products (busy) | 36 % of thread time inside the pool |
| spinning for experts still being read | 32 % |
| idle at the barrier / wake-up | 32 % (caller waits 68 us per op for the slowest worker, wake-up 11 us per start) |
| quantise + task setup | under 1 % of the pool time |
Per part: gate 408 us per op, up 195 us, down 187 us for the same 4.7 MB: gate absorbs the waits for experts still loading. Up and down run at 24-25 GB/s. So the missing half of the bandwidth is not quantising or dispatch; it is (a) waiting for reads on a RAM-limited machine and (b) the barrier: ops are small (about 40 chunks of 128 rows over 16 threads). Idea, not built: run gate and up in one pool job (same input, independent, same arithmetic) to halve the barriers.

**4. Decode at depth (N41b).** Flash attention per call: 35-42 us at 4k (k = 4096-4352) vs 154 us at 120k (k = 120832): the whole KV is scanned (123 MB per layer at about 800 GB/s), nothing is skipped. With 12 attention layers that is about +1.4 ms per token (about +2-3 % of a token), not the -14 % in the README. The 128k decode follow-up generated 1 token (EOS), so no 128k decode figure from this run. The rest of the -14 % is probably the ctx-dependent plan (131072 ctx leaves fewer GPU expert layers than the 8k-ctx run it was compared with); same-ctx clean runs would show it. Sparse-skip in Vulkan flash attention is upstream, not ours.

Timers added (off by default): MOE_CACHE_SEG_TIMES (gaps, faults, longest gaps in the stats JSON and exit lines), MOE_CACHE_EXEC_TIMES ("ex" in the stats JSON, "exec-times" exit lines).

## Result 69. Afternoon PC queue

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-29 10:11-17:00 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | not in the heading; mentioned in the text: Flash-Next, Q2_0 | not stated | not stated in this section | all runs "not clean": the owner's own work was running, ~7 GiB | No (heading says not clean) |

<sub>Old heading: 69. Afternoon PC queue (Windows PC, 2026-10-29 10:11-17:00; all runs "not clean": the owner's own work was running, ~7 GiB)</sub>

Runner `C:\dl\afternoon\runner.ps1`, paired ABBA, Flash-Next Q2_0, abba's fixed text check (per prompt; "differs, split differed" allowed only where
the GPU/CPU split of the long-prompt blocks changed). The first hour produced nothing (helper `pcb_t0_run.py` had no `--port` option; every run
stopped after 11 s), and the runner lost its own log lines until 14:47 (`$o` and `$O` are one variable in PowerShell); both fixed.

| Item | A | B | Runs (prefill tok/s; decode tok/s) | Text | Verdict |
|---|---|---|---|---|---|
| T0a confirm, ctx 262144 | old plan (28b022b + f478564) | f310c28 (378cc26 + 2963cc5) | 4k: A 109.4 / 102.9, B 110.0 / 109.0; 34k: A 107.3 / 102.5, B 108.6 / 107.1; decode A 18.5 / 18.0, **B 21.9 / 21.6** | identical in every pair | **no regression; decode +18 %** (T0a closed) |
| n23-pipe (c05b9a0) vs n23-split alone | split (MOE_CACHE_GPU_PIPE=0) | split + pipelined block | 34k: A 148.4 / 154.2, **B 165.3 / 166.1**; 128k: A 132.7 / 136.9, **B 147.2 / 147.8**; decode equal | identical | +10 % prefill, but it sits on n23-split |
| n23-split (d4d381d) vs today's code | MOE_CACHE_GPU_CHUNKS=0 | default | 34k: A 154.0 / 156.2, B 155.4 / 155.2; 128k: A 136.7 / 137.0, B 137.0 / 136.2 | identical | **no repeatable gain** (B lost pair 2): not merged |
| N22 read-back (6465d7d) | MOE_CACHE_GPU_READBACK=0 | default | 34k: A 191.5, B 200.3 / 195.1; 128k: A 168.6, B 169.4 / 169.5; **run 4 (A) never ran**: a CHECK job left two log files in C:\dl\pc_checks at 15:23, and the lock waits while that folder holds any file | no check file | **undecided** (3 of 4 runs): rerun |
| n23 pipe-only (809505c, pipelining without split) | MOE_CACHE_GPU_PIPE=0 | default | queued 16:57; results in `C:\dl\afternoon\summary.csv` | - | next session |
| 122B 256k, GLM 64k, Flash-Next 256k | - | - | queued after it (until 22:00 at the latest) | - | next session |

Peak RAM in use on the PC during the 128k runs: 28.7-29.5 GiB including the owner's ~7 GiB (logged; nothing stopped).

**Evening additions to 69 (2026-10-29 20:16-21:32, not clean: the owner's desktop held ~2.8 GB VRAM: dwm 1.2 GB, a browser 0.9 GB):**
| Item | A | B | Runs | Verdict |
|---|---|---|---|---|
| G3 MTP head on Flash-Next Q2_0 (ctx 40960, f310c28 DLL; unsloth `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, `-md ... --spec-type draft-mtp --spec-draft-n-max 2 -ngld 99`) | no head | head | decode water/code: A 17.7 / 17.65 and 19.36 / 19.67, **B 10.43 / 9.78 and 11.34 / 10.32**; p34k prefill A 72.9 / 80.4, B 57.4 / 57.5; expert hits A 95-97 %, **B 4-54 %** | **killed again (-40 % decode)**: the head and its draft context take RAM from the expert cache and hits collapse; MTP is not the lever on Flash-Next at 32 GB |
| Prefill 73-80 tok/s at 34k in both arms (vs ~290 on a clean PC) | - | - | "a long-prompt block needs 791 MiB of GPU memory but only 697 MiB are free: it stays on the CPU" | **desktop VRAM problem**: the planner's reserve is taken by other processes after the plan; fix in progress (session bigfix: block split / re-plan, margin from the desktop's own use) |
| n23-pipe-only (809505c) | - | - | every valid pair won (+8-11 % prefill, text identical); the one lost pair ran out of RAM | **merged 5e6ecd0** without a rerun (owner) |
| N22 read-back | - | - | +3-5 % at 34k, mixed at 128k | **parked** (owner: no more reruns for small gains) |

## Result 68. Nemotron 3.5 Bolt: the "3.5 s per tiny request" explained, and short prompts on stock runs

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-29 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | Nemotron 3.5 Bolt 74B-A9B | not stated | mentioned in the text: b11966 | T5, Windows PC | Not stated in the heading; the text mentions runs that were not clean |

<sub>Old heading: 68. Nemotron 3.5 Bolt: the "3.5 s per tiny request" explained, and short prompts on stock runs (T5, Windows PC, 2026-10-29)</sub>

Same PC (7950X, RX 9070 XT 16 GB, 32 GB), llama.cpp b11966, Bolt stock (launcher default: it fits), cold (file cache flushed), one server per arm.
**Not clean: owner work running** (Edge open; 22.7-24.4 GB free RAM at each start). Data: PC C:\dl\t5\out\.

**The 202-task run of RESULTS 65 (701 s):** 431 s were one request stuck from 21:36:42 to 21:43:51, the window of the PC's driver timeout and
network drop (T6); 227 s prompt processing; 27 s decoding (4 tokens per answer, as Q2_0; thinking was off); 16 s other. Without the stall 1.34 s
per request (Q2_0 0.84 s). Prompts of 33-35 tokens took ~0.4 s, prompts of 36+ tokens ~1.2 s: llama.cpp moves an op to the GPU for batches of
32+ tokens (`GGML_OP_OFFLOAD_MIN_BATCH`, default 32; Vulkan, CUDA, SYCL, Metal, CANN), which for Bolt copies every CPU-held expert to the GPU
(~1 s), and llama-server processes a prompt as (n - 4) + 4 tokens for its checkpoint.

| Arm (tiny requests) | chat, template default (thinking on) | chat, enable_thinking off | 202-task requests (first 60) | tokens vs default |
|---|---|---|---|---|
| default | 3.36 s (68 tokens, reasoning) | 0.71 s | 1.32 s | - |
| `--reasoning-budget 0` | 0.87 s | - | 1.31 s | 60/60 identical |
| `GGML_OP_OFFLOAD_MIN_BATCH` 64 / 128 / 256 | - | 0.83-0.85 s | 0.61 / 0.59 / 0.60 s | 60/60 identical |

Prompt time (ms) by size, nothing cached:
| Prompt tokens | 32 (llama.cpp) | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| 51 | 1223 | 435 | 421 | 437 | - |
| 105 | 1377 | 1391 | 588 | 586 | - |
| 159 | 1439 | 1482 | 1468 | 683 | - |
| 303 | 1748 | - | - | 1787 | **962** |
| 375 | 1613 | - | - | 1763 | **1102** |
| 465 | 1657 | - | - | 1817 | **1266** |
| 555 / 735 | 1697 / 1784 | - | - | 1733 / 1796 | 1688 / 1774 |
Answers identical in every arm (90 prompts). **Launcher default since this section:** stock runs whose model fits and that keep expert layers on
the CPU get `GGML_OP_OFFLOAD_MIN_BATCH=512` (a user's value wins; plugin runs unchanged). Not proven bit-exact (CPU and GPU sums may round
differently); tokens were identical everywhere it was checked. Thinking stays as the model card sets it (on for agents).

## Result 67. Overnight chain on the Windows PC: N29, N22, N15, N30, N26, N27, N23, long context, Bolt, HIP, N25c, N18

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-28 23:32 -> 10-29 08:01 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | not in the heading; mentioned in the text: Flash-Next, Qwen3.6, Qwen3.5-122B, GLM-5.3, Nemotron, Q2_0, IQ2_XXS | versions and commits named in the text: 28b022b, f478564, 15e09b1, 77e6e09, 8451a87 | mentioned in the text: 620d2225c, b11966, b11669 | pc-bench | Not stated |

<sub>Old heading: 67. Overnight chain on the Windows PC (pc-bench, 2026-10-28 23:32 -> 10-29 08:01): N29, N22, N15, N30, N26, N27, N23, long context, Bolt, HIP, N25c, N18</sub>

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

## Result 65. New models on the Windows PC: Nemotron 3.5 Bolt 74B-A9B; DeepSeek V4-Flash

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-28 evening | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | Nemotron 3.5 Bolt 74B-A9B; DeepSeek V4-Flash (desk estimate only) | versions and commits named in the text: 681f3bc | mentioned in the text: 620d2225c, b11966, ff0d396e3 | pc-glm; desk only | Not stated |

<sub>Old heading: 65. New models on the Windows PC (pc-glm, 2026-10-28 evening): Nemotron 3.5 Bolt 74B-A9B; DeepSeek V4-Flash (desk only)</sub>

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

## Result 64. Long context on the Windows PC: 64k and 128k+ on Flash-Next Q2_0, the 122B and Qwen3.6

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-28 evening | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | Flash-Next Q2_0, Qwen3.5-122B, Qwen3.6 | versions and commits named in the text: 28b022b, f478564, 31ac159 | mentioned in the text: 620d2225c | pc-bench | Not stated in the heading; the text mentions runs that were not clean |

<sub>Old heading: 64. Long context on the Windows PC (pc-bench, 2026-10-28 evening): 64k and 128k+ on Flash-Next Q2_0, the 122B and Qwen3.6</sub>

**Superseded for Flash-Next Q2_0 prompt speed by sections 78-79 (2026-11-02/03):** 64k now reads at 421-448 tok/s (this section: 195-266) and the 256k setting uses `-ub 2048`. The 122B and Qwen3.6 rows here are still the only measurements of those models at 64k with these versions.

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
| Qwen3.5-122B, moe-cache, 256k setting (overnight build: launcher 28b022b, DLL f478564) | 15.1 s | 113 / 115 / 104 / 103 / 103 / 97, 99, 96, 97 up to 138k | 13.92 (93.3 %) | - | 12.65 at 138k (93.6 %); **timed out at 35 min** after the 138k save (most likely the whole prompt processed again, finding 5) |
| Qwen3.5-122B, moe-cache, 256k setting, **rerun on experimental f310c28** (46 CPU layers, `-ub 512`; hold 1 of a split run, T4) | 17.8 s | 96 / 100 / 88.5 / 89 / 90, 87, 82, 84, 85 up to 135k | 13.50 (90.2 %) | **12.85 at 62.7k** (88.2 %) | 12.01 at 130.5k (85.4 %); hold 1 stopped at 135k by its 29-min budget; **holds 2-3 (to ~250k) queued** (C:\dl\afternoon item 70) |
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
out too (35 tok/s prompts; 15 of its 35 min went to a cool-down break). **T4 (2026-10-29, f310c28, split into holds of 29 min or less, slot 0 saved and restored between holds, no slot-1 request):** why 35 min was
too little: the 122B at 256k fills ~254k tokens at 82-100 tok/s = ~50 min of prefill alone, and GLM's 63.5k prompt at 35 tok/s = ~30 min plus a
2 tok/s decode; neither was a stall (the 122B's last stretch is the known re-processing, finding 5). Hold 1 of the 122B (cap 21 GiB because the owner's
own work used 6.9 GiB of RAM: not clean) reached 135k in 27.1 min, slot save 4.4 s / 4.73 GB. Still open, queued on the PC for the afternoon runner
(items 70-72, results in C:\dl\afternoon\ and C:\dl\t4\t4.log; method and commands: handover/parts/longctx_20261029.md): 122B 256k holds 2-3, GLM 64k, Flash-Next Q2_0 256k on the current build.

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

## Result 63. Flash-Next quant choice on the Windows PC: Q2_0, IQ3_S, Coder IQ1_M vs IQ2_XS / IQ3_XXS

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-28 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | Flash-Next Q2_0, IQ3_S, Coder IQ1_M, IQ2_XS, IQ3_XXS | versions and commits named in the text: 681f3bc, 8cc6c7c | mentioned in the text: b11669 | pc-glm | Not stated |

<sub>Old heading: 63. Flash-Next quant choice on the Windows PC (pc-glm, 2026-10-28): Q2_0, IQ3_S, Coder IQ1_M vs IQ2_XS / IQ3_XXS</sub>

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

## Result 62. GLM max on the Windows PC: drive, GPU layers, MTP, long prompts, and why code is slow

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-28 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | GLM-5.3-Flash | versions and commits named in the text: 1d2d5a8, 681f3bc | mentioned in the text: 620d2225c | pc-glm | Not stated |

<sub>Old heading: 62. GLM max on the Windows PC (pc-glm, 2026-10-28): drive, GPU layers, MTP, long prompts, and why code is slow</sub>

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

## Result 61. Windows desktop: list A

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-27/28 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | not in the heading; mentioned in the text: Flash-Next, Qwen3.6, GLM-5.3, Qwen3.5-122B, Granite, IQ3_XXS, IQ2_XS | versions and commits named in the text: 65facd1, 681f3bc, bbcec55, 570d351, 7c47642 | mentioned in the text: 620d2225c | I32, B13, F5, B11/B12, N13, N8, N3, N5 | Not stated |

<sub>Old heading: 61. Windows desktop, 2026-10-27/28: list A (I32, B13, F5, B11/B12, N13, N8, N3, N5)</sub>

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

## Result 60. Windows desktop: agent sessions, stock vs plugin, with and without MTP

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-27 night | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | not in the heading; mentioned in the text: Qwen3.6, Flash-Next, GLM-5.3, IQ2_XS, IQ3_XXS | not stated | not stated in this section | I43 / T8 | Not stated |

<sub>Old heading: 60. Windows desktop, 2026-10-27 night: agent sessions (I43 / T8), stock vs plugin, with and without MTP</sub>

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

## Result 59. Windows desktop: the cache moves to read-only views of the model file and beats stock

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-27 evening | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | not in the heading; mentioned in the text: Flash-Next, Qwen3.5-122B, GLM-5.3, Granite, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_XXS | versions and commits named in the text: 65facd1 | mentioned in the text: 620d2225c | no commit | Not stated |

<sub>Old heading: 59. Windows desktop, 2026-10-27 evening: the cache moves to read-only views of the model file (no commit) and beats stock</sub>

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

## Result 58. Windows desktop: fixes confirmed, MTP, Flash-Next IQ2_XS vs IQ3_XXS, GLM page file

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-27 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | Qwen3.6, Flash-Next IQ2_XS and IQ3_XXS, GLM | versions and commits named in the text: ebc946f | mentioned in the text: 620d2225c, b11669 | see the section | Not stated in the heading; the text mentions runs that were not clean |

<sub>Old heading: 58. Windows desktop, 2026-10-27: fixes confirmed, MTP, Flash-Next IQ2_XS vs IQ3_XXS, GLM page file</sub>

**Superseded in part by section 59 (same evening):** the IQ3_XXS loss to stock found here was fixed by the file-view cache.

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

## Result 56. Windows desktop experiments

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-26/27 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | not in the heading; mentioned in the text: Flash-Next, Qwen3.5-122B, GLM-5.3, Qwen3.6 | not stated | mentioned in the text: b11669 | SKELETON: known facts filled, empty cells "-" are not measured yet | Skeleton: not every cell is measured |

<sub>Old heading: 56. Windows desktop experiments (2026-10-26/27; SKELETON: known facts filled, empty cells "-" are not measured yet)</sub>

**Status:** a skeleton from 2026-10-26/27; cells marked "-" were never filled. Later results on the same questions are in sections 58-61 (I37 and the 122B diagnosis were resolved by the file-view cache, section 59).

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

## Result 55. Windows desktop, clean WP2: apps closed, file cache flushed before every run

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-26 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | Flash-Next, Qwen3.5-122B, GLM | versions and commits named in the text: 2dccf17 | mentioned in the text: b11669 | complete: Flash-Next, 122B, GLM | Yes (heading says clean) |

<sub>Old heading: 55. Windows desktop, clean WP2: apps closed, file cache flushed before every run (2026-10-26; complete: Flash-Next, 122B, GLM)</sub>

**Superseded for speed by section 59 (2026-10-27):** these runs are from before the file-view cache and the I37 fix, when the plugin was at or below stock on Windows. Kept as the "before" measurement.

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

## Result 54. Windows 11 desktop: Ryzen 9 7950X + RX 9070 XT

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-26 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | not in the heading; mentioned in the text: Flash-Next, GLM-5.3, Qwen3.5-122B, Qwen3.6, Granite, IQ2_XS, IQ2_XXS, Q4_K_M, Q2_0, IQ3_XXS | not stated (the text names fix commits ff73ec1, 7a1b86f, db8e634) | mentioned in the text: b11669, 620d2225c | see the section (source: the PC session's windows-kit/pc_results.md) | Preliminary: the section says every speed is preliminary (owner's apps open, warm file cache); identity and bug findings are final |

<sub>Old heading: 54. Windows 11 desktop: Ryzen 9 7950X + RX 9070 XT (2026-10-26; from the PC session, windows-kit/pc_results.md)</sub>

**Superseded:** every speed in this section is preliminary (apps open, warm file cache). The identity, bug and driver findings stand; for speeds use sections 55, 59-63 and 79.

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

## Result 50. Windows round 2 and the first AMD machine

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-16 | MSI Claw 8 AI+ (Windows 11) and mini PC (Ryzen 7 5800H, Vega 8 iGPU, 32 GB DDR4, Ubuntu 26.04) | OLMoE-1B-7B Q4_0 (Claw); Granite and Qwen3.6-35B-A3B Q4_K_M (mini PC) | versions and commits named in the text: 022c674 | mentioned in the text: b11669, 23b0202a1 | MSI Claw; mini PC, T7; from handover/claw_results.md "Round 2" and handover/moe-cache-t7/ | Not stated |

<sub>Old heading: 50. Windows round 2 (MSI Claw) and the first AMD machine (mini PC, T7) (2026-10-16; from handover/claw_results.md "Round 2" and handover/moe-cache-t7/)</sub>

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

## Result 49. First run on real Windows, and the iGPU experiments E1/E2

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings (from the old heading) | Clean |
|---|---|---|---|---|---|---|
| 2026-10-15 | MSI Claw 8 AI+ (Core Ultra 7 258V, Arc 140V iGPU, 32 GB, Windows 11) | OLMoE-1B-7B Q4_0 | not stated | mentioned in the text: b11669 | MSI Claw 8 AI+; from the Claw session's report, handover/claw_results.md | Not stated |

<sub>Old heading: 49. First run on real Windows, and the iGPU experiments E1/E2 (MSI Claw 8 AI+, 2026-10-15; from the Claw session's report, handover/claw_results.md)</sub>

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
