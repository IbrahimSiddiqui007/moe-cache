# 2026-11-02: the next release (v0.3.5 candidate) against v0.3.4 on long prompts, patched server

**Converted from `docs/RESULTS.md` section 78 and the evening notes of the same day (private, not in the repo). VRAM, process RAM and CPU were not recorded on this day: every such cell is `not recorded`. There are no `records/` files for this day.**

**Superseded by [2026-11-03](../2026-11-03/README.md):** this run used a patched llama.cpp server with patch 0022 (not shipped in v0.3.5), so its prompt numbers are about 5-7 % too high. The 64k follow-up (39 CPU layers) is still the placement evidence.

Machine: Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5). Clean: PC restarted at 11:47, baseline guard PASS at 48.6 tok/s, runs 13:19-14:13; follow-up in the same boot 16:05-16:26.

**Headline:** prompts +35 % to +56 % with the unreleased defaults, but decode after a 64k prompt was 15 % lower than v0.3.4; the follow-up showed the cause is placement (CPU layers), not the code.

## Hardware

| Part | Value |
|---|---|
| CPU | AMD Ryzen 9 7950X |
| RAM | 32 GB DDR5 (speed not recorded) |
| GPU | AMD Radeon RX 9070 XT, 16 GB VRAM (driver not recorded) |
| OS | Windows 11 (build not recorded) |
| Model drive | D: (drive model and interface not recorded) |
| Power plan | not recorded |

## Software and versions

| Part | Value |
|---|---|
| "v0.3.4" arm | the released kit with its own llama-server |
| "Experimental" arm | the unreleased launcher defaults (prompt chunk 2048 above 32k context, GPU_SUM on) plus llama.cpp patch 0022 (`GGML_VK_COUNT_EXPERTS_FAST=1`) on the patched b11966 server |
| commits, plugin sha256, Python | not recorded |

## Method

- Order new old old new, two runs per cell, shown as first / second; averages are my arithmetic. Cold file cache before each run.
- Model: Flash-Next Q2_0 on D: (file hash and source URL not recorded). Prompts of 34k, 64k and 128k tokens; prompt hash, n_predict, temperature and seed: not recorded.
- Clean: yes for the main table (restart 11:47, baseline guard PASS 48.6 tok/s).
- Text was not compared.

## What changed since the last run

The previous comparison is RESULTS 77 (2026-10-31). New here: prompt chunk 2048 above 32k context, GPU_SUM on, patch 0022 on the patched b11966 server. These defaults were not in any release yet.

## Results per model and config

### Flash-Next Q2_0, main table (RESULTS 78)

CPU expert layers (`--n-cpu-moe`) and GPU expert layers per cell: not recorded for this table. VRAM, process RAM, CPU: not recorded.

| Context, prompt | v0.3.4 prompt tok/s | Experimental prompt tok/s | Prompt change | v0.3.4 decode tok/s | Experimental decode tok/s | Decode change |
|---|---|---|---|---|---|---|
| 40960, 34k tokens | 392.2 / 411.0 | 637.2 / 619.7 | +56 % | 28.55 / 28.48 | 27.29 / 27.67 | -3.6 % |
| 65536, 64k | 311.9 / 333.9 | 434.4 / 436.6 | +35 % | 26.29 / 26.10 | 22.95 / 21.72 | **-15 %** |
| 131072, 128k | 306.8 / 306.0 | 436.3 / 437.5 | +43 % | not recorded | not recorded | - |

Decode after a 64k prompt is 15 % lower in both pairs.

### 64k follow-up, same boot, 16:05-16:26 (`-ub 2048`, patch 0022)

One run per arm except two for 39 and 40 layers.

| CPU expert layers | Prompt tok/s | Decode tok/s |
|---|---|---|
| 41 | 454.3 | 22.34 |
| 40 | 464.3 / 476.5 | 23.69 / 25.40 |
| **39** | 457.7 / 454.6 | **26.82 / 26.13** |
| 38 with `-ub 1024` | 415.4 | 25.60 |

So the 64k decode loss was the placement. The planner now reserves for 39 at 64k.

### 64k default on the PC, evening check (kit from commit a23f8b6)

**Not clean:** the owner's programs were open and the PC had been up about 5 h. The launcher picked `-ub 2048` with **40** CPU layers (15.06 GiB free; 39 layers need about 15.35 GiB free: the planner keeps 1.46 GiB for the long-prompt block plus 0.25 GiB and half of what other apps hold).

| Arm | Prompt tok/s | Decode tok/s |
|---|---|---|
| Official server, default | 441 / 436 | 25.6 / 24.7 |
| Patched server + 0022 | 471 (average; +7 % on the official server's, my arithmetic) | not recorded |
| `-ub 1024`, 38 CPU layers | 389 / 388 | 27.7 / 23.3 |

The text was the same for the official and the patched server. The planner was kept as it is (the owner agreed). Quick and Windows CI were green on a23f8b6 (Windows run id not published here).

### Decode graph reuse check

A log check (no timing run) showed the decode graph is reused 126 times in 128 tokens, so the graph was already being reused; the idea (graph reuse for decode) was killed.

## What each flag did

| Flag or setting | A | B | Prompt tok/s | Decode tok/s | Source |
|---|---|---|---|---|---|
| CPU expert layers at 64k, `-ub 2048` | 41 / 40 / 39 | | 454.3 / 464-477 / 455-458 | 22.34 / 23.69-25.40 / 26.13-26.82 | follow-up table above |
| `-ub` 1024 (38 layers) against 2048 (40 layers) at 64k, evening check | 2048 | 1024 | 441 / 436 against 389 / 388 | 25.6 / 24.7 against 27.7 / 23.3 | evening table above |
| patch 0022 | off | on | about +7 % on average | same text | evening table above; section 78 estimates the overstatement at 5-7 % |

## Limitations and what was not measured

- VRAM, process RAM, CPU, TTFT, wall time: not recorded for any run.
- Decode at 128k: not recorded. Text not compared in the main table.
- The main table used a patched server: its prompt numbers are too high (see the 2026-11-03 page for the official-server numbers).
- The evening check was not clean (owner programs open, about 5 h uptime).
- One machine, one model, three depths. Cause of the main-table 64k loss was found only afterwards (placement).

## Reproduce

Not recorded as exact commands for this converted page. Launcher defaults of the release kit and of the experimental build, cold file cache, order new old old new, contexts 40960 / 65536 / 131072. New runs: `bench-record` ([docs/BENCHMARK_TEMPLATE.md](../../docs/BENCHMARK_TEMPLATE.md)).

## Raw data

No `records/` files (converted page). Source: [docs/RESULTS.md](../../docs/RESULTS.md) section 78 and the private notes of the day.
