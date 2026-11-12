# Current results (v0.3.8, Windows desktop)

Moved out of the README on 2026-11-06 so the README stays short. The block between the `card` markers is generated from [benchmark_card.html](benchmark_card.html) (`python3 tools/card_markdown.py --write`; the quick tests check it). Older results: [results-archive/RESULTS.md](results-archive/RESULTS.md) (Windows desktop, every measurement, newest first) and [results-archive/RESULTS_OLD_LAPTOP.md](results-archive/RESULTS_OLD_LAPTOP.md) (Linux laptop).

## What changed since v0.3.7

The headline numbers below are the 2026-11-08 night chain, measured with the fixed conversation driver (real answers, 256 tokens, sizes by the server's token count) on a PC whose RAM has run at 6000 MT/s since 2026-11-07; the v0.3.7 conversation numbers (records of 2026-11-05) came from the old driver, whose decode figures rest on short answers, and are withdrawn. The long-prompt mode (`-ub 4096`, q8_0 KV cache) is now the default from 32k context; the comparison against the previous default is in the card's last table (claim `v038-longprompt-default`). Every number in the card has a claim in [benchmarks/claims/](../benchmarks/claims/) and a record in [benchmarks/runs/](../benchmarks/runs/).

<!-- card:begin -->

**Benchmark card · v0.3.8 · measured 2026-11-08 · Windows desktop**

moe-cache is a llama.cpp plugin that keeps the experts a model uses most in RAM and reads the rest from disk. On this 32 GB Windows PC, in a real 65,000-token conversation (code pastes, logs and follow-ups, sent turn by turn with the whole history), it reads new prompt tokens and decodes faster than stock llama-server on the models that are bigger than the RAM (a model that fits in RAM is about the same). Older measurements (8k chats, long prompts, agent sessions, other models) are on [the older card page](results-archive/benchmark_card_older.html).

| Machine | |
|---|---|
| CPU | Ryzen 9 7950X (16 cores) |
| GPU | RX 9070 XT 16 GB |
| RAM | 32 GB DDR5 at 6000 MT/s |
| OS | Windows 11 |
| llama.cpp builds | v0.3.8 head kit (the v0.3.7.1 code with the Windows scheduling tuning off), official llama.cpp b12064 (51fe42c5d) Vulkan |

| Headline | What it is |
|---|---|
| 3.5× | prompt reading in a real 65k conversation on Flash-Next IQ3_XXS, the default model (188.6 vs 54.4 tok/s, 3.2–4.3× by depth); decode 19.1 vs 14.25 tok/s (1.34×, 1.25–1.63× by depth); two runs per arm, 42 turns to 65,280 tokens <!-- claim:v038-conv-iq3xxs --> |
| 1.9× | the same on Flash-Next Q2_0, the faster option (434.8 vs 227.6 tok/s, 5.0× falling to 1.3× by depth); decode 26.9 vs 25.4 tok/s (1.18–1.26×, and equal to stock at 48k–64k depth, 0.98×) <!-- claim:v038-conv-q2_0 --> |
| +77 % | prompt reading from the long-prompt mode, now the default (`-ub 4096`, q8_0 KV cache): 331.8 vs 187.4 tok/s on IQ3_XXS at 65k; decode 19.35 vs 19.01 tok/s (+1.8 %). Answers are not byte-identical to stock with it; `--no-long-prompt` or `--exact` turn it off <!-- claim:v038-longprompt-default --> |
| About the same | for a model that fits in RAM (Qwen3.6-35B, plugin forced on): 1.06× prompt, 1.07× decode. By default the launcher runs such a model through the plugin when some expert layers stay on the CPU, and stock when all experts fit the card (`--stock` opts out; the laptop with an 8 GB card gained 10 to 29 % decode, Without a GPU the launcher stays on stock llama.cpp (checked with a dry run on the laptop with the GPU hidden)) <!-- claim:v038-fits-qwen36 --> |

**Cleanliness and RAM speed.** Every run started after the chain's clean check (no other window, no RAM over 0.5 GB, no VRAM over 0.3 GiB, CPU idle); the records keep CPU busy, free RAM and VRAM before the start. The PC's RAM has run at 6000 MT/s (EXPO) since 2026-11-07; records made before that ran at 4800 MT/s and are not comparable with these. The Windows scheduling tuning was off in all runs (it is removed in v0.3.8). The v0.3.7 conversation numbers were taken out: their decode figures came from the old benchmark script with short answers.

### A real 65k conversation, stock against moe-cache, per depth band

Windows desktop, head kit (the v0.3.7.1 code, scheduling tuning off), official llama.cpp b12064, launcher settings of that kit (`-ub 2048`, f16 KV cache: what `--no-long-prompt` gives in v0.3.8). A seeded conversation (code pastes, logs, follow-ups; `tools/conversation_bench.py`, seed 42, real answers: thinking off, up to 256 tokens, an answer hint on every question) is sent turn by turn with the whole history, prompt cache on, temperature 0, until the 65,536-token context is full. Order stock, moe-cache, moe-cache, stock; the table shows the mean of the two runs per arm; cold file cache before every run. Prompt tok/s = new tokens read divided by prompt time, over turns with at least 64 new tokens; decode over turns with at least 16 generated tokens; the band is the depth of the prompt before the answer. The 48k–64k band holds most turns, mostly small follow-ups.

**Flash-Next IQ3_XXS (the default model), 42 turns to 65,280 tokens.** Whole run: moe-cache 188.6 against 54.4 tok/s for new prompt tokens, decode 19.1 against 14.25 tok/s. <!-- claim:v038-conv-iq3xxs -->

| Depth band | turns per run | stock prompt tok/s | moe-cache prompt tok/s | prompt ratio | stock decode tok/s | moe-cache decode tok/s | decode ratio | first token s, stock | first token s, moe-cache | plugin decode hit rate % |
|---|---|---|---|---|---|---|---|---|---|---|
| 0k-16k | 6 | 49.8 | **212.8** | 4.27× | 16.5 | **22.0** | 1.33× | 3.40 | 1.15 | 95.9 <!-- claim:v038-conv-iq3xxs --> |
| 16k-32k | 4 | 66.0 | **215.0** | 3.26× | 13.1 | **19.4** | 1.48× | 39.38 | 10.27 | 92.7 <!-- claim:v038-conv-iq3xxs --> |
| 32k-48k | 6 | 54.9 | **201.4** | 3.67× | 12.6 | **20.5** | 1.63× | 34.67 | 8.00 | 94.2 <!-- claim:v038-conv-iq3xxs --> |
| 48k-64k | 26 | 44.0 | **142.8** | 3.24× | 14.5 | **18.2** | 1.25× | 2.38 | 1.50 | 91.8 <!-- claim:v038-conv-iq3xxs --> |

**Flash-Next Q2_0 (the faster option).** Whole run: moe-cache reads new prompt tokens at 434.8 against 227.6 tok/s and decodes 26.9 against 25.4 tok/s. At 48k–64k depth the decode speed equals stock (0.98×). <!-- claim:v038-conv-q2_0 -->

| Depth band | turns per run | stock prompt tok/s | moe-cache prompt tok/s | prompt ratio | stock decode tok/s | moe-cache decode tok/s | decode ratio | first token s, stock | first token s, moe-cache | plugin decode hit rate % |
|---|---|---|---|---|---|---|---|---|---|---|
| 0k-16k | 6 | 89.0 | **445.0** | 5.00× | 25.0 | **29.5** | 1.18× | 1.89 | 0.68 | 98.6 <!-- claim:v038-conv-q2_0 --> |
| 16k-32k | 4 | 281.3 | **600.0** | 2.13× | 24.7 | **30.0** | 1.22× | 8.81 | 3.94 | 98.9 <!-- claim:v038-conv-q2_0 --> |
| 32k-48k | 6 | 280.8 | **495.6** | 1.77× | 23.8 | **30.1** | 1.26× | 7.29 | 3.33 | 99.3 <!-- claim:v038-conv-q2_0 --> |
| 48k-64k | 24 | 202.2 | **259.3** | 1.28× | 26.0 | **25.4** | 0.98× | 0.99 | 0.74 | 98.4 <!-- claim:v038-conv-q2_0 --> |

**Qwen3.6-35B, a model that fits in RAM (UD-IQ3_XXS file, plugin forced on).** Whole run: 2200 against 2072 tok/s prompt, decode 96.2 against 90.2 tok/s: about the same or slightly faster. The deeper bands have too few long answers for a decode figure. <!-- claim:v038-fits-qwen36 -->

| Depth band | turns per run | stock prompt tok/s | moe-cache prompt tok/s | prompt ratio | stock decode tok/s | moe-cache decode tok/s | decode ratio | first token s, stock | first token s, moe-cache | plugin decode hit rate % |
|---|---|---|---|---|---|---|---|---|---|---|
| 0k-16k | 6 | 1840.7 | **2381.1** | 1.29× | 91.6 | **97.0** | 1.06× | 0.32 | 0.19 | 100.0 <!-- claim:v038-fits-qwen36 --> |
| 16k-32k | 4 | 2544.6 | **2688.2** | 1.06× | 86.5 | **94.2** | 1.09× | 0.92 | 0.86 | 100.0 <!-- claim:v038-fits-qwen36 --> |
| 32k-48k | 6 | 2148.3 | **2239.1** | 1.04× | - | - | - | 0.79 | 0.77 | 100.0 <!-- claim:v038-fits-qwen36 --> |
| 48k-64k | 27 | 1678.1 | **1747.2** | 1.04× | - | - | - | 0.34 | 0.23 | 100.0 <!-- claim:v038-fits-qwen36 --> |

**Long-prompt mode, the default (IQ3_XXS, 65k, previous default A against long-prompt C, order A C C A, one run each).** The server logs show what ran: A `-ub 2048`, f16 KV, 42 expert layers on the CPU; C `-ub 4096`, q8_0 KV, 45 expert layers on the CPU.

| Arm | new prompt tokens, tok/s | decode, tok/s |
|---|---|---|
| A, previous default (two runs: 186.6 / 188.3 prompt, 18.97 / 19.04 decode) | 187.4 | 19.01 <!-- claim:v038-longprompt-default --> |
| C, long-prompt mode (two runs: 331.4 / 332.3 prompt, 19.30 / 19.39 decode) | **331.8** | **19.35** <!-- claim:v038-longprompt-default --> |
| C over A | +77.0 % | +1.8 % <!-- claim:v038-longprompt-default --> |

- With the long-prompt mode the q8_0 KV cache changes the numbers a little, so answers are not byte-identical to stock: KL divergence 0.014–0.015 against f16, the same top token in 94–95 % of positions, and the meaning was kept in 10 of 10 checks at 55k (2026-11-07). `--exact` or `--no-long-prompt` go back to f16.
- The first answer after a big paste is the visible gap: at 16k–32k depth the median first-token time is 10.27 s with moe-cache against 39.38 s stock on IQ3_XXS, and 3.94 s against 8.81 s on Q2_0. <!-- claim:v038-conv-iq3xxs --><!-- claim:v038-conv-q2_0 -->

### Decode speed in a 65k conversation, tokens per second

| Model (file size) | Context, prompt | stock | moe-cache | change |
|---|---|---|---|---|
| **Flash-Next IQ3_XXS** (the default)<br><sub>75.8 GB, drive D:, two runs each, ABBA</sub> | 65,536, a 42-turn conversation | 14.25 | 19.1 | +34 % <!-- claim:v038-conv-iq3xxs --> |
| Flash-Next Q2_0 (the faster option)<br><sub>66.4 GB, drive D:, two runs each, ABBA</sub> | 65,536, a conversation to 65k | 25.42 | 26.9 | +6 % <!-- claim:v038-conv-q2_0 --> |

Each number is the mean of two cold runs in the order stock, moe-cache, moe-cache, stock (ABBA), whole conversation; Q2_0 decode equals stock at 48k–64k depth (0.98×) <!-- claim:v038-conv-q2_0 -->. Both files are bigger than the free RAM once the GPU and Windows have taken their share. Stock = the same llama-server command the launcher builds for the plugin, without the plugin.

Test details for each row (the card draws the decode table as bars; same numbers):

| Model | Size, drive | stock tok/s | moe-cache tok/s | with MTP tok/s | Test details |
|---|---|---|---|---|---|
| Flash-Next IQ3_XXS, 65k conversation | 75.8 GB, drive D: | 14.25 | 19.10 | – | 42 expert layers on the CPU, -ub 2048, both arms<br>ctx 65536 · 42 turns to 65,280 tokens · ABBA, 2 runs each · cold file cache · RAM at 6000 MT/s<br>moe-cache peak: VRAM 14286 MiB · process RAM 20015 MiB (stock 26507 MiB)<br>v0.3.8 head kit · llama.cpp b12064 (51fe42c5d) · record 2026-11-08_pc_e22bae · claim v038-conv-iq3xxs |
| Flash-Next Q2_0, 65k conversation | 66.4 GB, drive D: | 25.42 | 26.91 | – | 40 expert layers on the CPU, -ub 2048, both arms<br>ctx 65536 · conversation to 65k tokens · ABBA, 2 runs each · cold file cache · RAM at 6000 MT/s<br>moe-cache peak: VRAM 13275 MiB · process RAM 20180 MiB (stock 26580 MiB)<br>v0.3.8 head kit · llama.cpp b12064 (51fe42c5d) · record 2026-11-08_pc_08de30 · claim v038-conv-q2_0 |

### Where it does not help

- **Models that fit in RAM.** If all experts fit on the card the launcher runs plain llama.cpp; if some expert layers stay on the CPU it runs the plugin by default since v0.3.8 (`--stock` opts out; laptop with an 8 GB card: decode +10 to +29 %, prompt about 10 % lower; Without a GPU the launcher stays on stock llama.cpp (checked with a dry run on the laptop with the GPU hidden)). On the 16 GB PC, forced through moe-cache, Qwen3.6-35B in a 65k conversation was about the same as stock: 1.06× prompt and 1.07× decode (one model, one PC, 2026-11-08) <!-- claim:v038-fits-qwen36 -->, while earlier laptop runs of models that fit showed stock as fast or faster; no steady gain, so the default stays.
- **Short answers at shallow depth.** The gain in a conversation comes from reading prompts and from keeping the hot experts in RAM; a short chat on a model that already fits in RAM does not gain.
- **CPU kernels.** The numbers here were measured with llama.cpp's own CPU kernels on both sides; the gain comes from which experts are in RAM and how they are read, not from faster math.
- **Cold start.** From a cold file cache the plugin reports ready a few seconds later than stock while it loads its hot experts; the first answer wins it back (RESULTS 61).
- **Q2_0 decode at 48k–64k.** Equal to stock (0.98×); the gain there is in the prompt (1.28×). <!-- claim:v038-conv-q2_0 -->
- **Small samples.** Two runs per arm (one per arm for the long-prompt comparison), one PC, one seeded script, one head kit. RAM speed matters: these runs are at 6000 MT/s. Cleanliness: see the note under the headline numbers.

### Output quality

The plugin does not change the math: with the same settings its answers are the same as stock's (the identity tests of the older card page). The default long-prompt mode is the exception: its q8_0 KV cache means answers are close but not byte-identical to stock (KL 0.014–0.015, same top token in 94–95 % of positions, meaning kept in 10 of 10 checks at 55k); `--exact` gives stock-identical answers. Earlier checks on other models and kits are on the [older card page](results-archive/benchmark_card_older.html). A different GPU layer split, or experts computed on the GPU instead of the CPU, can change a token while the meaning stays; [docs/EXACTNESS.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/main/docs/EXACTNESS.md) lists exactly what is checked.

### Method

- **Cold every run.** Before each server start, `tools\win_flush_cache.ps1` reads 38.5 GB of other files so Windows' file cache no longer holds the model. Stock benefits most from a warm cache, so this is the fair setting for both.
- **ABBA order.** Stock, moe-cache, moe-cache, stock, so drift (heat, background work) hits both sides equally.
- **Same conversation.** A seeded script of user turns (small follow-ups, code pastes, logs, a few big pastes), the whole history re-sent every turn, greedy decoding (temperature 0), real answers (up to 256 tokens), until the context is full. The model's own answers become part of the history.
- **Same layout.** Stock gets the same GPU expert layers, context, flash attention and batch size that the launcher picks for the plugin.
- **Text identity** is checked separately (stock against plugin, byte for byte, in the identity runs, with `--exact`), not in the conversation.

### Commands

**The whole A B B A on a Windows PC (one model per call; trial first with -Ctx 8192 -MaxTurns 4)**

```
powershell -NoProfile -ExecutionPolicy Bypass -File tools\conversation_bench_pair.ps1 -OutDir C:\bench\conv -Model D:\models\model.gguf -Alias flash -Ram 24 -FlushOther "D:\other-models" -Ctx 65536
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

The two columns are different machines and models, so compare down a column, not across. A model that fits in RAM is not in this table (see "Where it does not help").

Everything, with methods and caveats: [docs/results-archive/RESULTS.md](results-archive/RESULTS.md) (Windows desktop), [docs/results-archive/RESULTS_OLD_LAPTOP.md](results-archive/RESULTS_OLD_LAPTOP.md) (Linux laptop), the [benchmark card page](benchmark_card.html). Measure your own machine: the control page's **Share my numbers** button, or `python3 bin/moe_cache.py bench-report MODEL.gguf` ([docs/SHARE_YOUR_NUMBERS.md](SHARE_YOUR_NUMBERS.md)).
