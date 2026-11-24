# Current results (v0.3.9)

> **No speed figures in v0.3.9.** The bundled llama.cpp build moved to b12651 (ggml 1.0) and the speed measurements are redone on it. Speed numbers for this release follow in v0.3.9.1. The v0.3.8 page (earlier llama.cpp build, not v0.3.9 numbers) is kept in [results-archive/RESULTS_CURRENT_v0.3.8.md](results-archive/RESULTS_CURRENT_v0.3.8.md).

The block between the `card` markers is generated from [benchmark_card.html](benchmark_card.html) (`python3 tools/card_markdown.py --write`; the quick tests check it). Older results: [results-archive/RESULTS.md](results-archive/RESULTS.md).

<!-- card:begin -->

**Benchmark card · v0.3.9 · no speed figures in this release**

moe-cache is a llama.cpp plugin that keeps the experts a model uses most in RAM and reads the rest from disk. This page describes how its speed is measured; the figures for v0.3.9 are not published yet.

| Machine | |
|---|---|
| CPU | Ryzen 9 7950X (16 cores) |
| GPU | RX 9070 XT 16 GB |
| RAM | 32 GB DDR5 at 6000 MT/s |
| OS | Windows 11 |
| llama.cpp builds | bundled llama.cpp b12651 (ggml 1.0) |

**No speed figures in v0.3.9.** The speed measurements are redone on the bundled llama.cpp b12651 build; the figures of the earlier release were taken on an earlier build and are not repeated here. Speed numbers for this release follow in v0.3.9.1. The earlier card is kept in [results-archive/benchmark_card_v0.3.8.html](results-archive/benchmark_card_v0.3.8.html) (v0.3.8, earlier build, not v0.3.9 numbers).

### Where it does not help

- **Models that fit in RAM.** If all experts fit on the card the launcher runs plain llama.cpp; if some expert layers stay on the CPU it runs the plugin by default since v0.3.8 (`--stock` opts out; measured on one laptop with an 8 GB card: not faster than stock, no figure quoted; an earlier figure was not reproduced); Without a GPU the launcher stays on stock llama.cpp (checked with a dry run on the laptop with the GPU hidden)). Forced through moe-cache on the 16 GB PC, a model that fits in RAM was about the same as stock in our earlier test (no figure quoted in this release), and earlier laptop runs showed stock as fast or faster; no steady gain, so the default stays.
- **Short answers at shallow depth.** The gain in a conversation comes from reading prompts and from keeping the hot experts in RAM; a short chat on a model that already fits in RAM does not gain.
- **CPU kernels.** Any gain comes from which experts are in RAM and how they are read, not from faster math: llama.cpp's own CPU kernels run on both sides.
- **Small samples.** One or two runs per arm on one PC; speed depends on RAM speed, SSD and GPU.

### Output quality

The plugin does not change the math: with the same settings its answers are the same as stock\'s (the identity tests of the older card page). The default long-prompt mode is the exception: its q8_0 KV cache means answers are close but not byte-identical to stock; quality figures are in [EXACTNESS](EXACTNESS.md). `--exact` turns it off.

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

Everything older, with methods and caveats: [docs/results-archive/RESULTS.md](results-archive/RESULTS.md) (Windows desktop), [docs/results-archive/RESULTS_OLD_LAPTOP.md](results-archive/RESULTS_OLD_LAPTOP.md) (Linux laptop; not v0.3.9 numbers), the [benchmark card page](benchmark_card.html). Measure your own machine: the control page's **Share my numbers** button, or `python3 bin/moe_cache.py bench-report MODEL.gguf` ([docs/SHARE_YOUR_NUMBERS.md](SHARE_YOUR_NUMBERS.md)).
