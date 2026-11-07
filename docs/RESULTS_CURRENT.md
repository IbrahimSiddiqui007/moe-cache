# Current results (v0.3.7, Windows desktop)

Moved out of the README on 2026-11-06 so the README stays short. The block between the `card` markers is generated from [benchmark_card.html](benchmark_card.html) (`python3 tools/card_markdown.py --write`; the quick tests check it). Older results: [results-archive/RESULTS.md](results-archive/RESULTS.md) (Windows desktop, every measurement, newest first) and [results-archive/RESULTS_OLD_LAPTOP.md](results-archive/RESULTS_OLD_LAPTOP.md) (Linux laptop).

<!-- card:begin -->

**Benchmark card · v0.3.7 · measured 2026-11-05/06 · Windows desktop**

moe-cache is a llama.cpp plugin that keeps the experts a model uses most in RAM and reads the rest from disk. On this 32 GB Windows PC, in a real 65,000-token conversation (code pastes, logs and follow-ups, sent turn by turn with the whole history), it reads new prompt tokens faster and decodes faster than stock llama-server on the models that are bigger than the RAM. Older measurements (8k chats, long prompts, agent sessions, other models) are on [the older card page](results-archive/benchmark_card_older.html).

| Machine | |
|---|---|
| CPU | Ryzen 9 7950X (16 cores) |
| GPU | RX 9070 XT 16 GB |
| RAM | 32 GB DDR5 |
| OS | Windows 11 |
| llama.cpp builds | v0.3.7 kit f928d1c (the code released as v0.3.7; v0.3.7.1 only changes the installers and the banner), official llama.cpp b12064 (51fe42c5d) Vulkan |

| Headline | What it is |
|---|---|
| 4.2× | prompt reading in a real 65k conversation on Flash-Next IQ3_XXS, the default (188.2 vs 44.8 tok/s); decode 21.73 vs 16.00 tok/s (+36 %); 663 s instead of 1,852 s; 49 turns to 65,371 tokens <!-- claim:v037-conv-iq3xxs --> |
| 2.2× | the same on Flash-Next Q2_0, the faster option (430.0 vs 194.5 tok/s for stock); decode 26.98 vs 22.46 tok/s (+20 %); the whole conversation takes 388 s instead of 613 s. v0.3.7 candidate kit, PC, 46 turns to 65,379 tokens, A B B A, two runs per arm <!-- claim:v037-conv-q2_0 --> |
| Same text | Flash-Next Q2_0 at a 34k prompt: the plugin's answer was the same text as stock's (v0.3.7 candidate; from the chain log, no record yet) |
| Not here | Q2_0 against IQ3_XXS on one 64k prompt (chain log only, no record yet) and the quality comparison are in the release notes; the 8k chats, long prompts, agent sessions and other models are on the older card page |

**Cleanliness.** The conversation records do not store the clean check. The overnight chain's own check (windows, RAM, VRAM and CPU before every run) found no window, RAM or VRAM disturbance; its CPU counter came back empty in two checks (the chain's item 10 and the first run of the quant pair), so those two are not proven clean; paired runs agreed within 1 %. PC chains before 2026-11-05 wrote “not clean: owner work running” as fixed text without checking, so older numbers say “cleanliness not recorded”.

### A real 65k conversation, stock against moe-cache, per depth band

Windows desktop, v0.3.7 kit f928d1c, official llama.cpp b12064, timers off, launcher defaults. A seeded conversation (code pastes, logs, follow-ups; `tools/conversation_bench.py`, seed 42, 128 tokens per answer) is sent turn by turn with the whole history, prompt cache on, temperature 0, until the 65,536-token context is full. Order stock, moe-cache, moe-cache, stock; the table shows the mean of the two runs per arm; cold file cache before every run. Prompt tok/s = new tokens read divided by prompt time, over turns with at least 64 new tokens; decode over turns with at least 16 generated tokens; the band is the depth of the prompt before the answer. The 48k–64k band holds most turns, mostly small follow-ups.

**Flash-Next IQ3_XXS (the default), 49 turns to 65,371 tokens.** Whole run: moe-cache 188.24 against 44.81 tok/s for new prompt tokens, decode 21.73 against 16.0 tok/s. <!-- claim:v037-conv-iq3xxs -->

| Depth band | turns per run | stock prompt tok/s | moe-cache prompt tok/s | prompt ratio | stock decode tok/s | moe-cache decode tok/s | decode ratio | first token s, stock | first token s, moe-cache | plugin decode hit rate % |
|---|---|---|---|---|---|---|---|---|---|---|
| 0k-16k | 6 | 39.7 | **234.1** | 5.90× | 15.4 | **21.6** | 1.41× | 1.23 | 1.02 | 97.2 <!-- claim:v037-conv-iq3xxs --> |
| 16k-32k | 4 | 55.5 | **245.7** | 4.42× | 12.0 | **20.2** | 1.68× | 46.45 | 9.40 | 95.9 <!-- claim:v037-conv-iq3xxs --> |
| 32k-48k | 6 | 44.8 | **186.7** | 4.17× | 12.6 | **21.8** | 1.74× | 42.94 | 9.18 | 97.9 <!-- claim:v037-conv-iq3xxs --> |
| 48k-64k | 33 | 38.0 | **145.4** | 3.82× | 17.9 | **22.0** | 1.23× | 1.04 | 1.17 | 97.6 <!-- claim:v037-conv-iq3xxs --> |

**Flash-Next Q2_0 (the faster option), 46 turns to 65,379 tokens.** Whole run: moe-cache reads new prompt tokens at 430.01 against 194.5 tok/s and decodes 26.98 against 22.46 tok/s. <!-- claim:v037-conv-q2_0 -->

| Depth band | turns per run | stock prompt tok/s | moe-cache prompt tok/s | prompt ratio | stock decode tok/s | moe-cache decode tok/s | decode ratio | first token s, stock | first token s, moe-cache | plugin decode hit rate % |
|---|---|---|---|---|---|---|---|---|---|---|
| 0k-16k | 6 | 66.4 | **393.2** | 5.92× | 22.4 | **26.8** | 1.19× | 1.18 | 0.75 | 98.8 <!-- claim:v037-conv-q2_0 --> |
| 16k-32k | 4 | 246.7 | **551.0** | 2.23× | 20.6 | **26.9** | 1.31× | 11.24 | 4.30 | 98.8 <!-- claim:v037-conv-q2_0 --> |
| 32k-48k | 6 | 231.3 | **470.9** | 2.04× | 21.8 | **26.8** | 1.23× | 9.58 | 3.71 | 99.2 <!-- claim:v037-conv-q2_0 --> |
| 48k-64k | 30 | 201.8 | **321.9** | 1.60× | 22.9 | **27.1** | 1.18× | 0.75 | 0.71 | 99.5 <!-- claim:v037-conv-q2_0 --> |

- The stock IQ3_XXS conversation took 1,852 s against 663 s with moe-cache, for the same 49 turns. <!-- claim:v037-conv-iq3xxs -->
- With moe-cache the Q2_0 conversation decodes at 26.8–27.1 tok/s in every depth band; stock decodes at 20.6–22.9 tok/s. <!-- claim:v037-conv-q2_0 -->
- The first answer after a big paste is the visible gap: at 16k–32k depth the median first-token time is 4.30 s with moe-cache against 11.24 s stock on Q2_0, and 9.40 s against 46.45 s on IQ3_XXS. <!-- claim:v037-conv-q2_0 --><!-- claim:v037-conv-iq3xxs -->

### Decode speed in a 65k conversation, tokens per second

| Model (file size) | Context, prompt | stock | moe-cache | change |
|---|---|---|---|---|
| **Flash-Next IQ3_XXS** (the default)<br><sub>75.8 GB, drive D:, two runs each, ABBA</sub> | 65,536, a 49-turn conversation | 16.0 | 21.73 | +36 % <!-- claim:v037-conv-iq3xxs --> |
| Flash-Next Q2_0 (the faster option)<br><sub>66.4 GB, drive D:, two runs each, ABBA</sub> | 65,536, a 46-turn conversation | 22.46 | 26.98 | +20 % <!-- claim:v037-conv-q2_0 --> |

Each number is the mean of two cold runs in the order stock, moe-cache, moe-cache, stock (ABBA), whole conversation. Both files are bigger than the free RAM once the GPU and Windows have taken their share. Stock = the same llama-server command the launcher builds for the plugin, without the plugin.

Test details for each row (the card draws the decode table as bars; same numbers):

| Model | Size, drive | stock tok/s | moe-cache tok/s | with MTP tok/s | Test details |
|---|---|---|---|---|---|
| Flash-Next IQ3_XXS, 65k conversation | 75.8 GB, drive D: | 16.00 | 21.73 | – | 42 expert layers on the CPU, launcher defaults, both arms<br>ctx 65536 · 49 turns to 65,371 tokens · ABBA, 2 runs each · cold file cache<br>moe-cache peak: VRAM 14585 MiB · process RAM 19990 MiB (stock 26204 MiB)<br>v0.3.7 kit f928d1c · llama.cpp b12064 (51fe42c5d) · record 2026-11-05_pc_d1b3b3 · claim v037-conv-iq3xxs |
| Flash-Next Q2_0, 65k conversation | 66.4 GB, drive D: | 22.46 | 26.98 | – | 40 expert layers on the CPU, launcher defaults, both arms<br>ctx 65536 · 46 turns to 65,379 tokens · ABBA, 2 runs each · cold file cache<br>moe-cache peak: VRAM 13429 MiB · process RAM 19373 MiB (stock 25955 MiB)<br>v0.3.7 kit f928d1c · llama.cpp b12064 (51fe42c5d) · record 2026-11-05_pc_b21c0e · claim v037-conv-q2_0 |

### Where it does not help

- **Models that fit in RAM.** The launcher runs plain llama.cpp for them by itself. Forced through moe-cache, Qwen3.6-35B in a 65k conversation was somewhat faster than stock (one model, one PC, see RESULTS 80), while earlier laptop runs of models that fit showed stock as fast or faster; no steady gain, so the default stays.
- **Short answers at shallow depth.** The gain in a conversation comes from reading prompts and from keeping the hot experts in RAM; a short chat on a model that already fits in RAM does not gain.
- **CPU kernels.** The numbers here were measured with llama.cpp's own CPU kernels on both sides; the gain comes from which experts are in RAM and how they are read, not from faster math.
- **Cold start.** From a cold file cache the plugin reports ready a few seconds later than stock while it loads its hot experts; the first answer wins it back (RESULTS 61).
- **Small samples.** Two runs per arm, one PC, one seeded script, one v0.3.7 kit. Cleanliness: see the note under the headline numbers.

### Output quality

The plugin does not change the math: on Flash-Next Q2_0 at a 34k prompt the plugin's answer was the same text as stock's (v0.3.7 candidate, chain log only, no record yet). Earlier checks on other models and kits are on the [older card page](results-archive/benchmark_card_older.html). A different GPU layer split, or experts computed on the GPU instead of the CPU, can change a token while the meaning stays; [docs/EXACTNESS.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/main/docs/EXACTNESS.md) lists exactly what is checked.

### Method

- **Cold every run.** Before each server start, `tools\win_flush_cache.ps1` reads 38.5 GB of other files so Windows' file cache no longer holds the model. Stock benefits most from a warm cache, so this is the fair setting for both.
- **ABBA order.** Stock, moe-cache, moe-cache, stock, so drift (heat, background work) hits both sides equally.
- **Same conversation.** A seeded script of user turns (small follow-ups, code pastes, logs, a few big pastes), the whole history re-sent every turn, greedy decoding (temperature 0), a fixed answer length, until the context is full. The model's own answers become part of the history.
- **Same layout.** Stock gets the same GPU expert layers, context, flash attention and batch size that the launcher picks for the plugin.
- **Text identity** is checked separately (stock against plugin, byte for byte, in the identity runs), not in the conversation.

### Commands

**The whole A B B A on a Windows PC (one model per call; trial first with -Ctx 8192 -MaxTurns 4)**

```
powershell -NoProfile -ExecutionPolicy Bypass -File tools\conversation_bench_pair.ps1 -OutDir C:\dl\conv -Model D:\models\model.gguf -Alias flash -Ram 24 -FlushOther "D:\other-models" -Ctx 65536
```

**moe-cache (launcher defaults; it prints the llama-server command it runs)**

```
python bin\moe_cache.py server C:\models\model.gguf --llama-server C:\llama-vk\llama-server.exe
```

**Stock llama-server (N = the plugin's CPU expert layers, e.g. 40 for Flash-Next Q2_0)**

```
llama-server.exe -m C:\models\model.gguf -ngl 99 --n-cpu-moe N -c 65536 -fa on --load-mode mmap -ub 2048
```

**Flush the file cache before each run (reads other big files, writes nothing)**

```
powershell -ExecutionPolicy Bypass -File tools\win_flush_cache.ps1 -Target C:\models\model.gguf -Other "D:\other-models"
```

<!-- card:end -->


On Linux (older laptop runs, v0.2.x) moe-cache was clearly faster than stock on models bigger than the RAM: [docs/results-archive/RESULTS_OLD_LAPTOP.md](results-archive/RESULTS_OLD_LAPTOP.md).

**What to expect by memory budget** (decode tok/s, `--ram` = the memory the cache is sized to (a sizing hint, **not a RAM cap**: on Windows the default path sets no memory limit, so the server can use more than the number you give; on Linux the launcher sets none either; see [docs/WINDOWS.md](WINDOWS.md)); only numbers we measured, one cold run each, "not measured" where we have none):

| `--ram` | Windows desktop, Flash-Next IQ2_XS (68 GB), 16 GB GPU | Linux laptop, Qwen3-Next-80B (48.5 GB), about 2.6 GB GPU use |
|---|---|---|
| 24 GB | **21.81** (RESULTS §60) | **10.6**, stock 2.3 ([RESULTS_OLD_LAPTOP §6](results-archive/RESULTS_OLD_LAPTOP.md); v0.1 build) |
| 20 GB | **21.00** (RESULTS §60) | **9.3**, stock 1.3 ([RESULTS_OLD_LAPTOP §6](results-archive/RESULTS_OLD_LAPTOP.md); v0.1 build) |
| 16 GB | **20.03**, 92 % of the 24 GB speed (RESULTS §60, after the 2026-10-27 job-limit fix) | not measured |
| 12 GB, 8 GB, 32 GB or more | not measured | not measured |

The two columns are different machines and models, so compare down a column, not across. A model that fits in RAM is not in this table: plain llama.cpp is as fast or faster there.

Everything, with methods and caveats: [docs/results-archive/RESULTS.md](results-archive/RESULTS.md) (Windows desktop), [docs/results-archive/RESULTS_OLD_LAPTOP.md](results-archive/RESULTS_OLD_LAPTOP.md) (Linux laptop), the [benchmark card page](benchmark_card.html). Measure your own machine: the control page's **Share my numbers** button, or `python3 bin/moe_cache.py bench-report MODEL.gguf` ([docs/SHARE_YOUR_NUMBERS.md](SHARE_YOUR_NUMBERS.md)).
