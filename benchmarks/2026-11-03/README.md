# 2026-11-03: v0.3.5 against v0.3.4 on the official llama.cpp server, and the overnight checks

**Converted from `docs/RESULTS.md` section 79 and the overnight run notes (private, not in the repo). VRAM, process RAM and CPU were not recorded on this day: every such cell is `not recorded`. There are no `records/` files for this day yet; the numbers below are the ones written down at the time.**

Machine: Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5). Clean: PC restarted at 02:03, baseline guard PASS at 49.4 tok/s (stock llama.cpp speed check before the run). Runs 02:30-03:25 for the release comparison (RESULTS 79); the whole overnight chain ran 02:08-06:55 (overnight notes).

**Headline:** on long prompts v0.3.5 is 38-42 % faster than v0.3.4 (34k, 64k, 128k), and decodes 6-10 % slower after the long prompt (34k, 64k) because the bigger prompt chunk leaves 2 fewer expert layers on the GPU.

## Hardware

| Part | Value |
|---|---|
| CPU | AMD Ryzen 9 7950X |
| RAM | 32 GB DDR5 (speed not recorded) |
| GPU | AMD Radeon RX 9070 XT, 16 GB VRAM (driver not recorded) |
| OS | Windows 11 (build not recorded) |
| Model drive | D: (drive model and interface not recorded) |
| Power plan | not recorded |

Source: RESULTS 79 heading and first paragraph.

## Software and versions

| Part | Value |
|---|---|
| moe-cache | v0.3.5 (new) and v0.3.4 (old, the released kit); commits for the two arms not recorded. Planner change for 256k: commit 1309fcd |
| plugin sha256 | not recorded |
| llama.cpp | official b11966 server for both versions; patch 0022 not used (RESULTS 79) |
| Backend | Vulkan (RESULTS "Machines" line: llama.cpp Vulkan b11966) |
| Python | not recorded |

## Method

- Protocol: two arms (v0.3.4, v0.3.5), order new old old new, two runs per cell, shown as first / second. Averages and percentages on this page are my arithmetic from those pairs.
- Cold file cache before each run. Uptime at start: about 0.5 h after the 02:03 restart (my arithmetic from the restart and run times).
- Prompts: 34k, 64k and 128k tokens (contexts 40960, 65536 and 131072 as in the 2026-11-02 table; RESULTS 79 names only the depths); the 256k test used a 222k-token prompt (an earlier 280k-token prompt did not fit the 262k context and was dropped). Prompt text hash: not recorded.
- Model: Flash-Next Q2_0 (file hash and source URL not recorded), except the cells that name another model.
- n_predict, temperature, seed: not recorded for this day.
- Text was not compared (identical-to-baseline: not checked).
- Clean: yes (restart, baseline guard PASS 49.4 tok/s). Other programs: not recorded, but the baseline guard passed.
- The first 5 minutes of the chain and the first 256k attempt were lost to two script bugs (fixed before the numbers below).

## What changed since the last run

Previous page: [2026-11-02](../2026-11-02/README.md).

- The comparison now runs on the **official b11966 server for both versions**. 2026-11-02 used a patched server with patch 0022, which overstated v0.3.5 prompts by about 5-7 %; this page replaces those numbers.
- The planner picks `-ub 2048` at 256k context (commit 1309fcd).
- v0.3.5 is built without patch 0022 (the fix is merged upstream and arrives with the next llama.cpp pin move).

## Results per model and config

### Flash-Next Q2_0: v0.3.5 against v0.3.4 (RESULTS 79)

Prompt chunk `-ub 2048` for v0.3.5 (the v0.3.4 `-ub` is not recorded). CPU expert layers (`--n-cpu-moe`): at 64k v0.3.5 keeps 39 on a nearly empty card and 40 when other programs hold about 1 GB, v0.3.4 had 37-38; for 34k and 128k not recorded. GPU expert layers: not recorded (the model's layer count is not in the source). VRAM peak, process RAM peak, CPU: not recorded.

| Context | Prompt v0.3.4 (tok/s) | Prompt v0.3.5 | Change | Decode v0.3.4 | Decode v0.3.5 | Change |
|---|---|---|---|---|---|---|
| 34k | 408.6 / 403.1 | 577.0 / 572.6 | +42 % | 28.43 / 28.59 | 27.04 / 26.38 | -6 % |
| 64k | 313.8 / 314.1 | 421.3 / 448.3 | +38 % | 27.42 / 27.04 | 25.69 / 23.44 | -10 % |
| 128k | 304.8 / 305.5 | 420.7 / 421.9 | +38 % | not measured (the answer ended after 2 tokens) | not measured | - |

The decode loss is real. Check run at 64k: 2 more GPU layers with `-ub 512` (38 CPU layers) decode 26.63 / 26.92, against 23.74 / 22.91 with `-ub 2048` and 40 layers (+15 % on the averages, my arithmetic; RESULTS 79).

### Other models and depths (not paired with v0.3.4)

| Model, depth | New default | Compared with | Prompt tok/s change | Decode | Source |
|---|---|---|---|---|---|
| Flash-Next Q2_0, 256k setting (222k-token prompt) | `-ub 2048`: 334.8 / 333.2 | `-ub 1024`: 263.4 / 265.8 | +26 % | not recorded | RESULTS 79 |
| Qwen3.5-122B IQ2_XXS, 64k | 538.9 / 551.5 | `-ub 1024`: 438.7 / 437.3 | +24 % | 0.0 cell: the answer ended after 2 tokens, not measured | RESULTS 79, overnight notes |
| GLM-5.3-Flash, 30k | 72.8 / 73.1 | `-ub 1024`: 44.9 / 45.0 | +62 % | 2.88 vs 3.06 tok/s (-6 %) | RESULTS 79 |

The 122B and GLM runs were guards: no regression in prompts; the GLM decode loss is the same size as Flash-Next's.

### Decode-tier kill check (Windows PC, prototype `MOE_CACHE_DECODE_TIER`)

Design: [DECODE_TIER_DESIGN.md](../../docs/DECODE_TIER_DESIGN.md). The plan was context 65536 and a 34k-token prompt with a 256-token greedy answer; the run's own record was not kept, so the exact context, prompt and run time are not recorded, and the date is filed here as requested.

| Arm | Setting | Prompt tok/s (first / second) | Decode tok/s |
|---|---|---|---|
| Check mode | tier on layers 38,39, `..._CHECK=1` | not recorded | passed: 0 groups above 0.05 |
| TIER | tier on 2 layers (38, 39) | 311.4 / 313.3 | 26.71 / 26.22 |
| BASE | no tier | 488.0 / 585.4 | 26.72 / 26.65 |

Verdict (rule in the design page: kill if decode gain is under +3 % or prompts drop more than 3 %): **killed**. Decode did not improve (tier 26.46 against base 26.69 on the averages, my arithmetic), and prompts were far slower (tier about 312 against base about 537). The BASE prompt pair (488.0 / 585.4) varies by 20 %, which was not explained.

## What each flag did

| Flag or setting | A | B | Prompt tok/s | Decode tok/s | Source |
|---|---|---|---|---|---|
| `-ub` at 64k (decode-room check) | 2048, 40 CPU layers | 512, 38 CPU layers | not recorded | 23.74 / 22.91 against 26.63 / 26.92 (+15 %) | RESULTS 79 |
| `-ub` at 256k | 1024 | 2048 | 263.4 / 265.8 against 334.8 / 333.2 (+26 %) | not recorded | RESULTS 79 |
| `-ub` at 64k, 122B | 1024 | default (2048) | 438.7 / 437.3 against 538.9 / 551.5 (+24 %) | not recorded | RESULTS 79 |
| `MOE_CACHE_PLAN_RULE=1` at 128k | off | on | 418.7 / 423.4 against 403.5 / 403.6 (-4 %) | not recorded | RESULTS 79. Stays off by default: it loses |
| Windows priority switches (EcoQoS off and process priorities) at 34k | default | on | 586.4 against 598.2 (+2.0 %, average of two pairs) | the same | RESULTS 79. Default-on is the owner's decision |
| `MOE_CACHE_DECODE_TIER=38,39` | off | on | 488.0 / 585.4 against 311.4 / 313.3 | 26.72 / 26.65 against 26.71 / 26.22 | table above. Killed |

## Limitations and what was not measured

- VRAM, process RAM, CPU, TTFT and wall time: not recorded for any run on this page (the record collector was not used yet).
- Hardware details (driver, OS build, RAM speed, drive model), plugin hash, commits, model hash and source, prompt hash, n_predict, seed: not recorded.
- Text was not compared between versions or arms.
- Decode at 128k and the 122B default cell is not measured (the answer ended after 2 tokens).
- One machine, one main model (Flash-Next Q2_0) for the release comparison; 122B and GLM are single-depth guards.
- Two runs per cell: the first/second spread is shown, no more statistics.
- The decode-tier check has no recorded context, prompt length or time, and an unexplained BASE spread.

## Reproduce

Not recorded as exact commands for this converted page. The comparison is: launcher defaults of each release kit on the official b11966 server, `--ctx` 40960 / 65536 / 131072, cold file cache, order new old old new. For new runs use `bench-record` ([docs/BENCHMARK_TEMPLATE.md](../../docs/BENCHMARK_TEMPLATE.md)), which writes the command into the record.

## Raw data

No `records/` files for this day (converted page). The numbers come from [docs/RESULTS.md](../../docs/RESULTS.md) section 79 and the private overnight notes.
