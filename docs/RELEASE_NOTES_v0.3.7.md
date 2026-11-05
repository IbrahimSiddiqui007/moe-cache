# moe-cache v0.3.7 (2026-11-05)

**What this release is.** A default-model change (Flash-Next **Q2_0** again), the first multi-turn conversation benchmark, home-network serving that is off by default and key-protected, safer installers and key handling, and a lot of tidying of the numbers we publish. **Every speed number below was measured on the v0.3.6 kit; v0.3.7 itself has not been measured on the PC yet** (see "Not measured").

## Measured

**What these numbers are.** Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), 2026-11-05, plugin v0.3.6 (kit "head5"), launcher defaults, context 65,536, cold file cache before every run. Records: `benchmarks/runs/2026-11-05_pc_*`, claims `conv65k-*` and `quant-abba-p64k`, board `benchmarks/BOARD.md`.

**A real 65k conversation, stock llama.cpp against moe-cache.** A seeded conversation (code pastes, logs, follow-ups; `tools/conversation_bench.py`, seed 42, 128 tokens per answer) is sent turn by turn with the whole history, prompt cache on, temperature 0, until the context is full. Order stock, moe-cache, moe-cache, stock; two runs per arm; the tables show the mean of the two runs. Prompt tok/s is new tokens read divided by prompt time (turns with at least 64 new tokens); decode is over turns with at least 16 generated tokens; the band is the depth of the prompt before the answer.

*Flash-Next Q2_0 (the default), 46 turns to 65,379 tokens.* Whole run: moe-cache reads new prompt tokens at 424.5 against 201.4 tok/s and decodes 27.09 against 22.37 tok/s.

| Depth band | Turns per run | Prompt tok/s, stock | Prompt tok/s, moe-cache | Decode tok/s, stock | Decode tok/s, moe-cache | First token s, stock | First token s, moe-cache |
|---|---|---|---|---|---|---|---|
| 0k-16k | 6 | 75.3 | 390.6 | 22.1 | 27.0 | 1.23 | 0.70 |
| 16k-32k | 4 | 253.5 | 542.0 | 19.7 | 26.9 | 10.29 | 4.29 |
| 32k-48k | 6 | 241.1 | 464.1 | 21.5 | 27.0 | 8.93 | 3.75 |
| 48k-64k | 30 | 194.3 | 318.6 | 23.1 | 27.1 | 0.71 | 0.70 | <!-- claim:conv65k-q2_0 -->

*Flash-Next IQ3_XXS (the higher-quality option), 49 turns to 65,371 tokens.* Whole run: moe-cache 214.66 against 44.15 tok/s for new prompt tokens, decode 22.04 against 16.43 tok/s.

| Depth band | Turns per run | Prompt tok/s, stock | Prompt tok/s, moe-cache | Decode tok/s, stock | Decode tok/s, moe-cache | First token s, stock | First token s, moe-cache |
|---|---|---|---|---|---|---|---|
| 0k-16k | 6 | 40.3 | 225.0 | 15.7 | 21.7 | 1.78 | 1.06 |
| 16k-32k | 4 | 52.5 | 232.7 | 12.4 | 20.1 | 46.07 | 9.68 |
| 32k-48k | 6 | 44.0 | 221.9 | 12.8 | 22.1 | 43.68 | 7.20 |
| 48k-64k | 33 | 38.5 | 187.9 | 18.4 | 22.4 | 1.08 | 1.14 | <!-- claim:conv65k-iq3xxs -->

**Q2_0 against IQ3_XXS on one 64k prompt** (both with moe-cache, order Q2_0, IQ3_XXS, IQ3_XXS, Q2_0, llama.cpp b12064): the 61,843-token prompt is read at 445.6 tok/s with Q2_0 against 238.5 with IQ3_XXS (+86.9 %), and the short answer after it decodes at 24.30 against 16.12 tok/s (+50.7 %). <!-- claim:quant-abba-p64k --> The launcher plans a different GPU layer count for each file (8 against 6 on the GPU), so this compares the files as the launcher runs them.

**Quality** (measured 2026-11-05, record pending: not ingested yet, no claim): top-1 agreement with IQ3_S is 73.3 % for Q2_0 and 76.5 % for IQ3_XXS. That is why IQ3_XXS stays listed as the higher-quality option.

**How clean were these runs.** The 64k Q2_0 / IQ3_XXS pair passed the harness's own clean check before every run (windows, RAM, VRAM and CPU). The two conversation runs had a quiet-CPU check before every run, but other programs and the baseline are **not recorded**, and the llama.cpp build is not recorded in those two records (the chain of the same day on the same kit recorded b12064). The text "not clean: owner work running" that older PC chains wrote at the start of every run was fixed text, not a check; where we used to repeat it we now say "cleanliness not recorded".

**Not measured:** v0.3.7 against v0.3.6; stock against moe-cache in a conversation longer than 65k; the conversation on any file other than Q2_0 and IQ3_XXS; Linux. One PC, two runs per arm, one seeded script.

## New for users

| What | In plain words |
|---|---|
| Default model | Flash-Next **Q2_0** is the default Flash-Next file again (`pick`, `launch`, the catalogue, the docs). IQ3_XXS stays listed as the higher-quality option (193/202 tasks, top-1 agreement with IQ3_S 76.5 % against 73.3 %). |
| Conversation benchmark | `tools/conversation_bench.py` and, on Windows, `tools/conversation_bench_pair.ps1` play a real multi-turn session up to a context limit and record every turn (tokens added, depth, prompt and decode speed, cached tokens, first-token time, plugin hits and misses) plus VRAM, RAM, CPU, load time, disk reads and settings per run. How to run it: `docs/CONVERSATION_BENCH.md`. |
| Serving to your home network | `--lan` (server) and `--ui-lan` (control page) serve to other computers on your network. Off by default. The API key is forced on any address that is not this computer, public addresses are refused, and other computers see only an allow-list of page routes (no key, start, stop or benchmark). |
| Two models at once (prototype) | The control page can plan and start two models next to each other (memory is checked against the servers already running). Prototype; not measured. |
| Safer installers | `install.sh` / `install.ps1` refuse a cut-off installer script, never go on after a failed build attestation, pin the signer workflow, and check that the required files are there after the Windows unpack. `update --apply` refuses a downgrade unless you pass `--allow-downgrade`. |
| Your API key | The stored key is sent only to a server this launcher started (not to any listener on a port). The page's runner and the replay tool follow the same rule. |
| A damaged model file | A model file that is cut short or cannot be read now ends with a clear message on Linux (before: a bus error). On Windows the same check is in (a guard for the in-page read error), tested with a simulated error in CI, not with a real cut file on a PC. |
| Starting | `launch` is ready only when its own server process is alive and answers with its own model (aliases are understood); a llama-server on `PATH` whose ggml API cannot be read gives way to the kit's. |
| The numbers we publish | The README now shows the whole benchmark card as tables (generated from the card page by `tools/card_markdown.py`, checked in CI), every number says its date, context and prompt length, and the results file is split in two: `docs/RESULTS.md` (Windows desktop, newest first) and `docs/RESULTS_OLD_LAPTOP.md`. A release tag now fails on number problems unless the owner signs a waiver that names the claims and an expiry date (`tools/claims_gate.py`). |
| Patch 0070 (decode buffer), prototype, off | `MOE_CACHE_DECODE_RELOC=1` with llama.cpp patch 0070 can shrink the compute buffer during decode. A run on the PC on 2026-11-05 moved nothing, so it is not a claim; it stays off. |

## Known issues (said plainly)

| Issue | Detail |
|---|---|
| `--ram` is a sizing hint | On Windows the memory limit is set only for the plugin, only with `--ram` and only with `MOE_CACHE_WIN_VIEW=0`; the default file-view cache, the on-demand listener and stock llama-server get none, so the server can use more RAM than `--ram` says. On Linux the launcher sets no limit either; only a cgroup you set yourself is a hard limit. No guard is being built. |
| CUDA errors still abort | The GPU-error catch (`MOE_CACHE_GPU_CATCH`, on by default) works for Vulkan only. On NVIDIA / CUDA a GPU error still ends the server (llama.cpp calls `abort()`). The catch has not yet met a real Vulkan failure either. |
| Desktop app not included | The desktop app (the `desktop/` folder) is not part of this release; use the browser control page (`python3 bin/moe_cache.py ui`). |
| Windows read-error guard | Tested with a simulated in-page error only; a real cut-short model file on Windows was not tried. |
| No v0.3.7 measurement | Every speed number is a v0.3.6 kit; the plugin and launcher changed since. A paired run on v0.3.7 is open. |
| Number checks | Many published numbers are older (labelled with their date and RESULTS section, no record) and the claims behind the newer ones are stale for 0.3.7 until a v0.3.7 run is recorded; the release waiver (owner decision) covers them for at most 30 days. |
| One PC | All Windows numbers are one desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB). NVIDIA, Intel graphics and Windows 10 are untested; Linux numbers are older (the laptop file). |

## Upgrading

`moe-cache update --apply` (checks the installer's attestation when `gh` is installed), or download the files from the release page. The previous version is kept.
