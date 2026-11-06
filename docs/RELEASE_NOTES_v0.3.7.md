# moe-cache v0.3.7 (2026-11-05)

**What this release is.** A default-model change (Flash-Next **Q2_0** again), the first multi-turn conversation benchmark, home-network serving that is off by default and key-protected, safer installers and key handling, and a lot of tidying of the numbers we publish. **Every speed number below was measured on the v0.3.7 candidate kit f928d1c** (see "Measured").

## Measured

**What these numbers are.** Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), the **v0.3.7 candidate kit f928d1c** (plugin v0.3.7) with the official llama.cpp **b12064** (51fe42c5d), timers off, launcher defaults, context 65,536, cold file cache before every run, order stock, moe-cache, moe-cache, stock, two runs per arm, 2026-11-05/06. Records: `benchmarks/runs/2026-11-05_pc_b21c0e`, `..._d1b3b3`, `..._11ff0f`; claims `v037-conv-q2_0`, `v037-conv-iq3xxs`, `v037-fits-qwen36`; board `benchmarks/BOARD.md`; the full tables, peaks and the comparison with the v0.3.6 kit are in `docs/RESULTS.md`, Result 80.

**A real 65k conversation, stock llama.cpp against moe-cache.** A seeded conversation (code pastes, logs, follow-ups; `tools/conversation_bench.py`, seed 42, 128 tokens per answer) is sent turn by turn with the whole history, prompt cache on, temperature 0, until the context is full. The tables show the mean of the two runs per arm. Prompt tok/s is new tokens read divided by prompt time (turns with at least 64 new tokens); decode is over turns with at least 16 generated tokens; the band is the depth of the prompt before the answer.

*Flash-Next Q2_0 (the default), 46 turns to 65,379 tokens.* Whole run: moe-cache reads new prompt tokens at 430.01 against 194.5 tok/s (2.2x) and decodes 26.98 against 22.46 tok/s (+20 %); the whole conversation takes 388 s instead of 613 s.

| Depth band | Turns per run | Prompt tok/s, stock | Prompt tok/s, moe-cache | Decode tok/s, stock | Decode tok/s, moe-cache | First token s, stock | First token s, moe-cache |
|---|---|---|---|---|---|---|---|
| 0k-16k | 6 | 66.4 | 393.2 | 22.4 | 26.8 | 1.18 | 0.75 |
| 16k-32k | 4 | 246.7 | 551.0 | 20.6 | 26.9 | 11.24 | 4.30 |
| 32k-48k | 6 | 231.3 | 470.9 | 21.8 | 26.8 | 9.58 | 3.71 |
| 48k-64k | 30 | 201.8 | 321.9 | 22.9 | 27.1 | 0.75 | 0.71 | <!-- claim:v037-conv-q2_0 -->

*Flash-Next IQ3_XXS (the higher-quality option), 49 turns to 65,371 tokens.* Whole run: moe-cache 188.24 against 44.81 tok/s for new prompt tokens (4.2x), decode 21.73 against 16.0 tok/s (+36 %); the conversation takes 663 s instead of 1,852 s.

| Depth band | Turns per run | Prompt tok/s, stock | Prompt tok/s, moe-cache | Decode tok/s, stock | Decode tok/s, moe-cache | First token s, stock | First token s, moe-cache |
|---|---|---|---|---|---|---|---|
| 0k-16k | 6 | 39.7 | 234.1 | 15.4 | 21.6 | 1.23 | 1.02 |
| 16k-32k | 4 | 55.5 | 245.7 | 12.0 | 20.2 | 46.45 | 9.40 |
| 32k-48k | 6 | 44.8 | 186.7 | 12.6 | 21.8 | 42.94 | 9.18 |
| 48k-64k | 33 | 38.0 | 145.4 | 17.9 | 22.0 | 1.04 | 1.17 | <!-- claim:v037-conv-iq3xxs -->

**Q2_0 against IQ3_XXS on one 64k prompt** (both with moe-cache, order Q2_0, IQ3_XXS, IQ3_XXS, Q2_0; chain log, **no record, no claim yet**): the 61,843-token prompt is read at 418.1 tok/s with Q2_0 against 204.7 with IQ3_XXS (+104.3 %), and the 128 tokens after it decode at 24.97 against 12.96 tok/s (+92.6 %); the plugin's decode hit rate is 98.8 % against 92.2 %. The launcher plans a different GPU layer count per file. (On the v0.3.6 kit the day before the same test gave +86.9 % and +50.7 %, claim `quant-abba-p64k`.)

**Identity at a 34k prompt on Q2_0** (plugin then stock; chain log, no record): the answers are the same text; plugin prompt 560.4 against stock 260.2 tok/s, decode after the prompt 26.47 against 24.04 tok/s. One pair.

**A model that fits in RAM** (Qwen3.6-35B, plugin forced; the launcher runs stock for it by itself): in a 65k conversation moe-cache read prompts 1.29x as fast and decoded 68.67 against 65.06 tok/s (+6 %) with a process RAM peak of about 11.9 against 22.4 GB. One model on one machine; older laptop runs of models that fit disagree, so this is not a general claim and the default (stock for a model that fits) is unchanged. Claim `v037-fits-qwen36`.

**Quality** (measured 2026-11-05, record pending: no record, no claim): top-1 agreement with IQ3_S is 73.3 % for Q2_0 and 76.5 % for IQ3_XXS. That is why IQ3_XXS stays listed as the higher-quality option.

**How clean were these runs.** The conversation records do not store the clean check. The overnight chain's own check (windows, RAM, VRAM and CPU before every run) found no window, RAM or VRAM disturbance; its CPU counter came back empty in two checks (the chain's item 10 and the first run of the quant pair, printed as "clean: NO"), so those two are not proven clean; paired runs agreed within 1 %. The text "not clean: owner work running" that PC chains before 2026-11-05 wrote at the start of every run was fixed text, not a check; where it was used we say "cleanliness not recorded".

**Not measured:** a conversation longer than 65k; any file other than Q2_0 and IQ3_XXS in a conversation; Linux; v0.3.7 against v0.3.6 in a paired run. Unpaired, against the v0.3.6 kit of the day before: Q2_0 with moe-cache is unchanged within a few percent, **IQ3_XXS with moe-cache read new prompt tokens 12 % slower** (and 14 % slower on the 64k prompt); not explained (RESULTS 80). One PC, two runs per arm, one seeded script.

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
| IQ3_XXS prompt speed | With moe-cache, IQ3_XXS read new prompt tokens slower on the candidate than on the v0.3.6 kit the day before (12 % in the conversation, 14 % on the 64k prompt; unpaired, different days); Q2_0 was unchanged. Not explained yet (RESULTS 80). |
| Number checks | The numbers on the README card have current claims (`bench-record lint` is clean without a waiver). Older numbers (8k chats, long prompts, agent sessions, other models) moved to `docs/benchmark_card_older.html` with their dates; their claims are stale for v0.3.7. Q2_0 against IQ3_XXS on a 64k prompt, the identity run and the top-1 quality numbers have no benchmark record yet. |
| One PC | All Windows numbers are one desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB). NVIDIA, Intel graphics and Windows 10 are untested; Linux numbers are older (the laptop file). |

## Upgrading

`moe-cache update --apply` (checks the installer's attestation when `gh` is installed), or download the files from the release page. The previous version is kept.
