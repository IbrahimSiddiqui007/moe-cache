<h1 align="center">moe-cache</h1>

<p align="center"><b>Run Mixture-of-Experts models that are bigger than your RAM, without the speed cliff.</b><br>
A plugin for an unmodified <code>llama-server</code>. No fork, no patch: the Windows kit and the Linux ggml 0.26 tarball bundle the official llama.cpp b12064 build as released upstream. A small optional patch set for it exists in <code>patches/llama.cpp</code>; it is off by default and not part of the kit.</p>

<p align="center">
<img alt="status" src="https://img.shields.io/badge/status-v0.3.7-blue">
<img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
<img alt="platform" src="https://img.shields.io/badge/platform-Linux%20%2B%20Windows%20(experimental)-lightgrey">
<img alt="llama.cpp" src="https://img.shields.io/badge/llama.cpp-tested%20b12064%20(Windows)%2C%2023b0202a1%20%2F%207fe450e19%20(Linux)-green">
<a href="https://ko-fi.com/ibrumm"><img alt="Support on Ko-fi" src="https://img.shields.io/badge/Ko--fi-support%20this%20project-ff5e5b?logo=ko-fi&logoColor=white"></a>
</p>

<p align="center">Speeds, test settings and every caveat are in the <a href="#results">benchmark card below</a> (Windows desktop, 32 GB RAM + 16 GB graphics card, stock llama.cpp against moe-cache). Full history: <a href="docs/RESULTS.md">docs/RESULTS.md</a>; the older Linux laptop results: <a href="docs/RESULTS_OLD_LAPTOP.md">docs/RESULTS_OLD_LAPTOP.md</a>.</p>

## What it is

**moe-cache lets a normal PC run big Mixture-of-Experts AI models** (files of 60-120 GB) that are bigger than its RAM, with the **same answers as plain llama.cpp** in the checks we ran (greedy decoding, the same GPU/CPU layer split, same llama.cpp build, context below 128k; the exact list is in [EXACTNESS](docs/EXACTNESS.md)). A different GPU layer split, or experts computed on the GPU instead of the CPU, can change a token (slightly different floating-point sums) while the meaning stays; from 128k context a q8_0 KV cache is used (`--exact` keeps f16), and MTP and the `--fast` modes can change answers. Chat in your browser, or connect coding tools (OpenCode, Hermes, Claude Code). Windows 11 and Linux. Free (MIT), no account.

MoE models (Qwen3, GLM, gpt-oss, DeepSeek) use only a few "experts" per token. When the file is bigger than your RAM, plain llama.cpp keeps reading the whole file from the disk and slows to a crawl; moe-cache keeps the experts the model actually uses in RAM and reads only the rest. If your model already fits in RAM, plain llama.cpp is as fast or faster, and moe-cache steps aside on its own ([when to use something else](docs/FAQ.md#when-to-use-something-else)).

| You need | |
|---|---|
| Memory | 32 GB RAM for the big models (16 GB works for smaller ones such as Qwen3.6-35B); a graphics card is optional but recommended |
| Disk | an SSD (NVMe is best) with the model size plus about 10 GB free |
| Software | Python 3.8 or newer (3.11 or newer for `connect codex --write`, which checks the merged TOML file). llama.cpp is included in the Windows kit and the Linux `-ggml0.26` download (v0.3.3 and newer) |
| Time | about 10 minutes of setup, plus the model download |

**Privacy:** no telemetry, no account; the plugin opens no network connections. The few times the launcher or installer uses the internet (downloads you ask for, the update check) are listed in [docs/SECURITY.md](docs/SECURITY.md#privacy-what-moe-cache-sends-and-stores).

## Get started

### Quickest: one line, then `moe-cache launch`

| | Install (checks the download against the release's SHA256 file; no admin or root; undo with `-Uninstall` / `--uninstall`) |
|---|---|
| Windows PowerShell | `irm https://github.com/IbrahimSiddiqui007/moe-cache/releases/latest/download/install.ps1 \| iex` (needs Python 3.8+; 3.11+ only for `connect codex --write`; it says how to get it if missing) |
| Linux | `curl -fsSL https://github.com/IbrahimSiddiqui007/moe-cache/releases/latest/download/install.sh -o install.sh && sh install.sh` |

Then open a new terminal and run `moe-cache launch`: it checks your PC, suggests models that fit, prints the download command (it downloads only after you say yes), starts the server and offers to connect your coding agents. What it asks, step by step: [docs/INSTALL.md](docs/INSTALL.md#terminal-guide-moe-cache-launch). The manual ways follow.

### Windows 11

| Step | What to do |
|---|---|
| 1. Python | Python 3.8 or newer from python.org (tick "Add python.exe to PATH") |
| 2. The kit | Download `moe-cache-vX-windows-x64.zip` (the newest) from [Releases](https://github.com/IbrahimSiddiqui007/moe-cache/releases). Right-click it > Properties > tick **Unblock** > OK, then unzip it, e.g. to `C:\moe-cache`. The kit is not code-signed: if Windows says "protected your PC", click More info > Run anyway |
| 3. A model | In the kit folder: `python bin\moe_cache.py pick` names the model that suits your PC and prints its download command (needs `pip install -U huggingface_hub` once). Or skip the command: the control page (step 4) has **Settings > Download a model**, which downloads from the catalogue with a progress bar, continues a cut download and checks the sha256 at the end |
| 4. Run | Double-click `moe-cache.bat`: the control page opens; pick the model, press Start. The first start takes a minute or two |
| 5. Agents | Tick Agent mode, then the page's Connect buttons (Hermes, OpenCode) |

If something is off, `python bin\moe_cache.py check D:\models\model.gguf` says what to fix. Every install option: [docs/INSTALL.md](docs/INSTALL.md#windows-quick-start-experimental); Windows details: [docs/WINDOWS.md](docs/WINDOWS.md).

**Using the PC while the model runs?** Tick **Leave room for me** on the control page (or `--leave-room`): the model leaves more RAM, GPU memory and CPU to your own programs and gets a little slower. A CPU with its own graphics: plug the monitor into the motherboard. The GPU also draws your desktop, and the screen flickers or the browser stutters while a long prompt runs? Reserve GPU memory for it with `--gpu-margin GiB` (for example `--gpu-margin 3`): the launcher plans fewer expert layers on the GPU and leaves that much free. Without the option it already keeps more free when other programs hold GPU memory at start.

### Linux

| Step | What to do |
|---|---|
| 1. Download | From [Releases](https://github.com/IbrahimSiddiqui007/moe-cache/releases): `moe-cache-vX-ggml0.26-linux-x86_64.tar.gz` (llama.cpp b12064 included, Vulkan + CPU), or `moe-cache-vX-linux-x86_64.tar.gz` for an older llama.cpp of your own (ggml 0.24 / 0.25). `tar xzf` it and `cd` into the folder |
| 2. A model | `python3 bin/moe_cache.py pick` |
| 3. Check | `bin/moe-cache-check /path/to/model.gguf --run` ("cache runs: yes" means it works on your machine) |
| 4. Run | `bin/moe-cache-server /path/to/model.gguf --open` (chat page and an OpenAI-compatible API at http://localhost:8080), or `bin/moe-cache-server --ui` for the control page |

Which file on the Releases page:

| Your PC | File | Contains |
|---|---|---|
| Windows | `moe-cache-vX-windows-x64.zip` | the kit: plugin, launcher, llama.cpp b12064 (Vulkan) |
| Linux, no llama.cpp of your own, or ggml 0.26 or newer | `moe-cache-vX-ggml0.26-linux-x86_64.tar.gz` | plugin, launcher, llama.cpp b12064 (Vulkan + CPU) |
| Linux, your own older llama.cpp (ggml 0.24 / 0.25) | `moe-cache-vX-linux-x86_64.tar.gz` | plugin and launcher only |

Your own llama.cpp (CUDA, ROCm), `install.sh`, building from source, the speed options (`--fast`, `--hybrid`) and proving the output yourself: [docs/INSTALL.md](docs/INSTALL.md).

### Coding agents

Start the server in agent mode (or let on-demand start it when an agent asks), then point the agent at it. One command writes the agent's config; the old config is kept as `.bak`.

```bash
python3 bin/moe_cache.py on-demand /path/to/model.gguf --install-autostart   # loads the model on the first request, unloads it when idle
# or by hand: bin/moe-cache-server /path/to/model.gguf --agent                # context 65536, tool calls, API key, warm sessions
```

#### Hermes Agent

| Step | What to do |
|---|---|
| Install | Hermes Agent CLI, or Hermes Desktop on Windows (hermes-agent.nousresearch.com). Windows: add `%LOCALAPPDATA%\hermes\bin` to PATH so `connect` finds `hermes` |
| Connect | `python3 bin/moe_cache.py connect hermes --write` (or the control page's **Connect Hermes** button): provider `custom`, URL, model name, context 65536 and the API key go into Hermes' `config.yaml`. Hermes Desktop reads the same file |
| Use | `hermes chat` (CLI), `hermes dashboard` (web UI), or Hermes Desktop: Settings > Model shows the moe-cache model |
| Measured | Windows desktop, Flash-Next IQ2_XS (68 GB, bigger than RAM), a recorded Hermes bug-fix session: about half the time with moe-cache as with stock llama.cpp, first token about three times sooner (2026-10-27, kit of that day, no benchmark record; numbers: [RESULTS section 60](docs/RESULTS.md) and the [older card page](docs/benchmark_card_older.html)) |

#### OpenCode

| Step | What to do |
|---|---|
| Install | OpenCode CLI, or OpenCode Desktop (opencode.ai/download) |
| Connect | `python3 bin/moe_cache.py connect opencode --write` (or **Connect OpenCode** on the control page): adds a `moe-cache` provider to `opencode.json` (or `opencode.jsonc`) and selects it; OpenCode Desktop reads the same folder |
| Use | `opencode`, `opencode web`, or OpenCode Desktop; pick `moe-cache/<model>` in the model list |
| Measured | same PC and model, a recorded OpenCode bug-fix session: less than half the time with moe-cache as with stock llama.cpp, first token about three times sooner (same day and kit, no benchmark record; numbers: [RESULTS section 60](docs/RESULTS.md) and the [older card page](docs/benchmark_card_older.html)) |

#### Other agents

`connect` also writes the config for **Claude Code, Codex CLI, Pi, OpenClaw, Aider and Qwen Code**: `python3 bin/moe_cache.py connect NAME --write` (without `--write` it only prints the settings; the first backup of your config is kept as `.bak`). Checked against the real programs with a small model on 2026-11-03: Codex CLI, Pi, OpenClaw, Aider, OpenCode and Claude Code completed a turn (Claude Code with its settings exported in the shell); Qwen Code read its config but the 1.7B test model returned an empty answer, so it is not proven yet; Hermes was checked on 2026-10-29. Details: [docs/AGENTS.md](docs/AGENTS.md).

**Check it works:** `python3 bin/moe_cache.py connect NAME --check` sends one prompt and one tool call the way that agent does and prints pass/fail per step with the likely fix (wrong key, model name, window too big, tool calls off).

Keep each agent's own context limit below the server's, and set things up before a session: details, desktop apps and on-demand in [docs/AGENTS.md](docs/AGENTS.md).

### Which model?

`pick` decides from your RAM, graphics card, free disk and use (agents, chat, long documents). On a 32 GB RAM + 16 GB GPU PC: Qwen3.8-Flash-Next Q2_0 (66 GB, the default since v0.3.7) for general and agent use: in a real 65k conversation moe-cache reads new prompt tokens 2.2× as fast as stock and decodes 26.98 against 22.46 tok/s <!-- claim:v037-conv-q2_0 -->; Q2_0 is also faster than IQ3_XXS at long context (chain log of 2026-11-06, no record yet), the IQ3_XXS file (76 GB) is the higher-quality option (top-1 agreement with IQ3_S 76.5 % against 73.3 % for Q2_0; measured 2026-11-05, record pending), Nemotron 3.5 Bolt 74B (22 GB, fits in RAM) for 128k contexts. `python3 bin/moe_cache.py models` lists every catalogue model with fits / tight / no for this PC, whether it is already downloaded and its measured speed (or "not measured"). All files and the rules: [docs/MODELS.md](docs/MODELS.md).

## Results

The card below is the Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan): stock llama.cpp against moe-cache with the same command line. It is generated from [docs/benchmark_card.html](docs/benchmark_card.html) (`python3 tools/card_markdown.py --write`), so it has the same numbers; the same page, drawn as a chart, is the image [docs/benchmark_card.png](docs/benchmark_card.png). A `claim:` marker is the id of the claim behind a number ([benchmarks/BOARD.md](benchmarks/BOARD.md)); a number without one has no claim yet. **Which build:** every number on this card was measured with the v0.3.7 candidate kit (f928d1c) and the official llama.cpp b12064 on 2026-11-05/06. Older measurements (8k chats, long prompts, agent sessions, other models) are on the [older card page](docs/benchmark_card_older.html); their claims are stale for v0.3.7. How clean the runs were: see the note under the headline numbers.

<!-- card:begin -->

**Benchmark card · v0.3.7 · measured 2026-11-05/06 · Windows desktop**

moe-cache is a llama.cpp plugin that keeps the experts a model uses most in RAM and reads the rest from disk. On this 32 GB Windows PC, in a real 65,000-token conversation (code pastes, logs and follow-ups, sent turn by turn with the whole history), it reads new prompt tokens faster and decodes faster than stock llama-server on the models that are bigger than the RAM. Older measurements (8k chats, long prompts, agent sessions, other models) are on [the older card page](benchmark_card_older.html).

| Machine | |
|---|---|
| CPU | Ryzen 9 7950X (16 cores) |
| GPU | RX 9070 XT 16 GB |
| RAM | 32 GB DDR5 |
| OS | Windows 11 |
| llama.cpp builds | v0.3.7 candidate kit f928d1c, official llama.cpp b12064 (51fe42c5d) Vulkan |

| Headline | What it is |
|---|---|
| 2.2× | prompt reading in a real 65k conversation on Flash-Next Q2_0, the default (430.0 vs 194.5 tok/s for stock); decode 26.98 vs 22.46 tok/s (+20 %); the whole conversation takes 388 s instead of 613 s. v0.3.7 candidate kit, PC, 46 turns to 65,379 tokens, A B B A, two runs per arm <!-- claim:v037-conv-q2_0 --> |
| 4.2× | the same on Flash-Next IQ3_XXS, the higher-quality option (188.2 vs 44.8 tok/s); decode 21.73 vs 16.00 tok/s (+36 %); 663 s instead of 1,852 s; 49 turns to 65,371 tokens <!-- claim:v037-conv-iq3xxs --> |
| Same text | Flash-Next Q2_0 at a 34k prompt: the plugin's answer was the same text as stock's (v0.3.7 candidate; from the chain log, no record yet) |
| Not here | Q2_0 against IQ3_XXS on one 64k prompt (chain log only, no record yet) and the quality comparison are in the release notes; the 8k chats, long prompts, agent sessions and other models are on the older card page |

**Cleanliness.** The conversation records do not store the clean check. The overnight chain's own check (windows, RAM, VRAM and CPU before every run) found no window, RAM or VRAM disturbance; its CPU counter came back empty in two checks (the chain's item 10 and the first run of the quant pair), so those two are not proven clean; paired runs agreed within 1 %. PC chains before 2026-11-05 wrote “not clean: owner work running” as fixed text without checking, so older numbers say “cleanliness not recorded”.

### A real 65k conversation, stock against moe-cache, per depth band

Windows desktop, v0.3.7 candidate kit f928d1c, official llama.cpp b12064, timers off, launcher defaults. A seeded conversation (code pastes, logs, follow-ups; `tools/conversation_bench.py`, seed 42, 128 tokens per answer) is sent turn by turn with the whole history, prompt cache on, temperature 0, until the 65,536-token context is full. Order stock, moe-cache, moe-cache, stock; the table shows the mean of the two runs per arm; cold file cache before every run. Prompt tok/s = new tokens read divided by prompt time, over turns with at least 64 new tokens; decode over turns with at least 16 generated tokens; the band is the depth of the prompt before the answer. The 48k–64k band holds most turns, mostly small follow-ups.

**Flash-Next Q2_0 (the default), 46 turns to 65,379 tokens.** Whole run: moe-cache reads new prompt tokens at 430.01 against 194.5 tok/s and decodes 26.98 against 22.46 tok/s. <!-- claim:v037-conv-q2_0 -->

| Depth band | turns per run | stock prompt tok/s | moe-cache prompt tok/s | prompt ratio | stock decode tok/s | moe-cache decode tok/s | decode ratio | first token s, stock | first token s, moe-cache | plugin decode hit rate % |
|---|---|---|---|---|---|---|---|---|---|---|
| 0k-16k | 6 | 66.4 | **393.2** | 5.92× | 22.4 | **26.8** | 1.19× | 1.18 | 0.75 | 98.8 <!-- claim:v037-conv-q2_0 --> |
| 16k-32k | 4 | 246.7 | **551.0** | 2.23× | 20.6 | **26.9** | 1.31× | 11.24 | 4.30 | 98.8 <!-- claim:v037-conv-q2_0 --> |
| 32k-48k | 6 | 231.3 | **470.9** | 2.04× | 21.8 | **26.8** | 1.23× | 9.58 | 3.71 | 99.2 <!-- claim:v037-conv-q2_0 --> |
| 48k-64k | 30 | 201.8 | **321.9** | 1.60× | 22.9 | **27.1** | 1.18× | 0.75 | 0.71 | 99.5 <!-- claim:v037-conv-q2_0 --> |

**Flash-Next IQ3_XXS (the higher-quality option), 49 turns to 65,371 tokens.** Whole run: moe-cache 188.24 against 44.81 tok/s for new prompt tokens, decode 21.73 against 16.0 tok/s. <!-- claim:v037-conv-iq3xxs -->

| Depth band | turns per run | stock prompt tok/s | moe-cache prompt tok/s | prompt ratio | stock decode tok/s | moe-cache decode tok/s | decode ratio | first token s, stock | first token s, moe-cache | plugin decode hit rate % |
|---|---|---|---|---|---|---|---|---|---|---|
| 0k-16k | 6 | 39.7 | **234.1** | 5.90× | 15.4 | **21.6** | 1.41× | 1.23 | 1.02 | 97.2 <!-- claim:v037-conv-iq3xxs --> |
| 16k-32k | 4 | 55.5 | **245.7** | 4.42× | 12.0 | **20.2** | 1.68× | 46.45 | 9.40 | 95.9 <!-- claim:v037-conv-iq3xxs --> |
| 32k-48k | 6 | 44.8 | **186.7** | 4.17× | 12.6 | **21.8** | 1.74× | 42.94 | 9.18 | 97.9 <!-- claim:v037-conv-iq3xxs --> |
| 48k-64k | 33 | 38.0 | **145.4** | 3.82× | 17.9 | **22.0** | 1.23× | 1.04 | 1.17 | 97.6 <!-- claim:v037-conv-iq3xxs --> |

- With moe-cache the Q2_0 conversation decodes at 26.8–27.1 tok/s in every depth band; stock decodes at 20.6–22.9 tok/s. <!-- claim:v037-conv-q2_0 -->
- The first answer after a big paste is the visible gap: at 16k–32k depth the median first-token time is 4.30 s with moe-cache against 11.24 s stock on Q2_0, and 9.40 s against 46.45 s on IQ3_XXS. <!-- claim:v037-conv-q2_0 --><!-- claim:v037-conv-iq3xxs -->
- The stock IQ3_XXS conversation took 1,852 s against 663 s with moe-cache, for the same 49 turns. <!-- claim:v037-conv-iq3xxs -->

### Decode speed in a 65k conversation, tokens per second

| Model (file size) | Context, prompt | stock | moe-cache | change |
|---|---|---|---|---|
| **Flash-Next Q2_0** (the default)<br><sub>66.4 GB, drive D:, two runs each, ABBA</sub> | 65,536, a 46-turn conversation | 22.46 | 26.98 | +20 % <!-- claim:v037-conv-q2_0 --> |
| Flash-Next IQ3_XXS (the higher-quality option)<br><sub>75.8 GB, drive D:, two runs each, ABBA</sub> | 65,536, a 49-turn conversation | 16.0 | 21.73 | +36 % <!-- claim:v037-conv-iq3xxs --> |

Each number is the mean of two cold runs in the order stock, moe-cache, moe-cache, stock (ABBA), whole conversation. Both files are bigger than the free RAM once the GPU and Windows have taken their share. Stock = the same llama-server command the launcher builds for the plugin, without the plugin.

Test details for each row (the card draws the decode table as bars; same numbers):

| Model | Size, drive | stock tok/s | moe-cache tok/s | with MTP tok/s | Test details |
|---|---|---|---|---|---|
| Flash-Next Q2_0, 65k conversation | 66.4 GB, drive D: | 22.46 | 26.98 | – | 40 expert layers on the CPU, launcher defaults, both arms<br>ctx 65536 · 46 turns to 65,379 tokens · ABBA, 2 runs each · cold file cache<br>moe-cache peak: VRAM 13429 MiB · process RAM 19373 MiB (stock 25955 MiB)<br>v0.3.7 candidate kit f928d1c · llama.cpp b12064 (51fe42c5d) · record 2026-11-05_pc_b21c0e · claim v037-conv-q2_0 |
| Flash-Next IQ3_XXS, 65k conversation | 75.8 GB, drive D: | 16.00 | 21.73 | – | 42 expert layers on the CPU, launcher defaults, both arms<br>ctx 65536 · 49 turns to 65,371 tokens · ABBA, 2 runs each · cold file cache<br>moe-cache peak: VRAM 14585 MiB · process RAM 19990 MiB (stock 26204 MiB)<br>v0.3.7 candidate kit f928d1c · llama.cpp b12064 (51fe42c5d) · record 2026-11-05_pc_d1b3b3 · claim v037-conv-iq3xxs |

### Where it does not help

- **Models that fit in RAM.** The launcher runs plain llama.cpp for them by itself. Forced through moe-cache, Qwen3.6-35B in a 65k conversation was somewhat faster than stock (one model, one PC, see RESULTS 80), while earlier laptop runs of models that fit showed stock as fast or faster; no steady gain, so the default stays.
- **Short answers at shallow depth.** The gain in a conversation comes from reading prompts and from keeping the hot experts in RAM; a short chat on a model that already fits in RAM does not gain.
- **CPU kernels.** The numbers here were measured with llama.cpp's own CPU kernels on both sides; the gain comes from which experts are in RAM and how they are read, not from faster math.
- **Cold start.** From a cold file cache the plugin reports ready a few seconds later than stock while it loads its hot experts; the first answer wins it back (RESULTS 61).
- **Small samples.** Two runs per arm, one PC, one seeded script, a v0.3.7 candidate kit. Cleanliness: see the note under the headline numbers.

### Output quality

The plugin does not change the math: on Flash-Next Q2_0 at a 34k prompt the plugin's answer was the same text as stock's (v0.3.7 candidate, chain log only, no record yet). Earlier checks on other models and kits are on the [older card page](benchmark_card_older.html). A different GPU layer split, or experts computed on the GPU instead of the CPU, can change a token while the meaning stays; [docs/EXACTNESS.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/main/docs/EXACTNESS.md) lists exactly what is checked.

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

On Linux (older laptop runs, v0.2.x) moe-cache was clearly faster than stock on models bigger than the RAM: [docs/RESULTS_OLD_LAPTOP.md](docs/RESULTS_OLD_LAPTOP.md).

**What to expect by memory budget** (decode tok/s, `--ram` = the memory the cache is sized to (a sizing hint, **not a RAM cap**: on Windows the default path sets no memory limit, so the server can use more than the number you give; on Linux the launcher sets none either; see [docs/WINDOWS.md](docs/WINDOWS.md)); only numbers we measured, one cold run each, "not measured" where we have none):

| `--ram` | Windows desktop, Flash-Next IQ2_XS (68 GB), 16 GB GPU | Linux laptop, Qwen3-Next-80B (48.5 GB), about 2.6 GB GPU use |
|---|---|---|
| 24 GB | **21.81** (RESULTS §60) | **10.6**, stock 2.3 ([RESULTS_OLD_LAPTOP §6](docs/RESULTS_OLD_LAPTOP.md); v0.1 build) |
| 20 GB | **21.00** (RESULTS §60) | **9.3**, stock 1.3 ([RESULTS_OLD_LAPTOP §6](docs/RESULTS_OLD_LAPTOP.md); v0.1 build) |
| 16 GB | **20.03**, 92 % of the 24 GB speed (RESULTS §60, after the 2026-10-27 job-limit fix) | not measured |
| 12 GB, 8 GB, 32 GB or more | not measured | not measured |

The two columns are different machines and models, so compare down a column, not across. A model that fits in RAM is not in this table: plain llama.cpp is as fast or faster there.

Everything, with methods and caveats: [docs/RESULTS.md](docs/RESULTS.md) (Windows desktop), [docs/RESULTS_OLD_LAPTOP.md](docs/RESULTS_OLD_LAPTOP.md) (Linux laptop), the [benchmark card page](docs/benchmark_card.html). Measure your own machine: the control page's **Share my numbers** button, or `python3 bin/moe_cache.py bench-report MODEL.gguf` ([docs/SHARE_YOUR_NUMBERS.md](docs/SHARE_YOUR_NUMBERS.md)).


## Known issues in v0.3.7 (honest list)

| Issue | Detail |
|---|---|
| Not byte-identical at 128k+ by default | at `--ctx 131072` and above the KV cache is q8_0 (`--fast=kv`, faster prompts at 256k on a 16 GB card, from the bigger GPU plan the smaller KV allows): it passed the quality gate at 4k context (KL 0.0079, top-1 96.7 %, tasks = stock), but not the same bytes as stock. `--exact` keeps llama.cpp's f16 KV |
| Windows zip is unsigned | Windows shows "unknown publisher" / SmartScreen: Unblock the zip, More info > Run anyway (see Get started). Check the SHA256 or `gh attestation verify` |
| Windows tested on one desktop | Ryzen 9 7950X + RX 9070 XT 16 GB + 32 GB, Windows 11, Vulkan, plus a GitHub Windows runner (CPU). NVIDIA, Intel GPUs and Windows 10 are untested; HIP loads but is slower than Vulkan |
| Models measured | Qwen3.8-Flash-Next (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, Coder IQ1_M), Qwen3.5-122B IQ2_XXS, GLM-5.3-Flash, Qwen3.6-35B, Nemotron 3.5 Bolt 74B (agent speed; plugin forced slower than stock in one test), Granite 1B (CI). Other MoE models should work but are not measured. Long context (64k-256k) is measured on Flash-Next and a few more, single runs |
| Linux | Tested on Intel + NVIDIA, CPU-only, Vulkan, and a user's RX 9070 XT. `install.sh` has been tried on Arch (`--no-apt`) and, for this release, from a clean clone on that machine; a fresh Ubuntu/Debian run with apt is not tested. macOS: no |
| `--ram` is a sizing hint | On Windows the memory limit is set only for the plugin, only with `--ram`, and only with `MOE_CACHE_WIN_VIEW=0`; the default file-view cache, the on-demand listener and stock llama-server get none (a job limit counts GPU memory and broke Vulkan). The server can use more RAM than `--ram` says. On Linux the launcher sets no limit either; only a cgroup you set yourself is a hard limit. No guard is being built |
| A failed model read still aborts | a model file that is cut short or cannot be read stops the server. On Linux it now prints a clear message; on Windows a guard for the in-page read error is in, tested with a simulated error in CI but not with a real cut-short file on a PC (CACHE-685) |
| Installers need `gh` or an explicit yes | `install.sh`, `install.ps1` and `update --apply` check the build attestation with the GitHub CLI (`gh`, logged in). Without it they stop and say what was and was not checked; `--yes-unverified` (or `MOE_CACHE_ALLOW_UNVERIFIED=1`) goes on with only the SHA256 check, which catches a damaged download but not a tampered release (RELEASE-686 fixed: no silent skip) |
| GPU errors | with Vulkan a GPU error inside an expert block is caught and the block runs on the CPU (`MOE_CACHE_GPU_CATCH`, not yet seen with a real failure); on NVIDIA / CUDA a GPU error still ends the server |
| Context length of the speed numbers | the numbers on the card are real 65k conversations on Flash-Next Q2_0 and IQ3_XXS, two runs per arm, one PC, a v0.3.7 candidate kit (the clean check's CPU counter was empty in two checks, see the note on the card); Q2_0 against IQ3_XXS on one 64k prompt and the identity check are chain-log numbers without records yet (release notes); nothing is measured beyond 65k on v0.3.7 |
| Not tested for this release | `winget install` and `scoop install` (manifests exist but are not submitted); Hermes Desktop and OpenCode Desktop on Windows (click-through still open); a full model run from the final zip |
| Decode after long prompts | on the v0.3.5 kit decode after a 34k-64k prompt was slower than on v0.3.4 (the bigger prompt chunk leaves 2 fewer expert layers on the GPU, RESULTS 79); patch 0070 is a prototype for giving that memory back, off by default, and a PC run on 2026-11-05 moved nothing |
| IQ3_XXS prompt speed on v0.3.7 | with moe-cache the IQ3_XXS conversation read new prompt tokens slower on the v0.3.7 candidate than on the v0.3.6 kit the day before (different days, not paired; Q2_0 was unchanged within a few percent): numbers in RESULTS 80; not explained yet |
| Speed numbers | One PC, cold runs, mostly single series; yours depends on RAM, SSD and GPU. The Linux table is older (v0.2.3.5) |


## Documentation

| Page | What is in it |
|---|---|
| [Install in detail](docs/INSTALL.md) | every install option, the speed options, the control page, proving the output |
| [Set up with a coding agent](docs/AI_SETUP.md) | numbered steps with exact commands for an AI agent that installs moe-cache for you (it asks before downloading a model or editing an agent config) |
| [Windows](docs/WINDOWS.md) | the Windows kit, page file, known limits |
| [Models](docs/MODELS.md) | which model and which file for your machine |
| [Coding agents](docs/AGENTS.md) | Hermes, OpenCode, Claude Code, on-demand |
| [Control page](docs/GUI.md), [try it step by step](docs/CONTROL_PAGE_CHECK.md) | the browser app |
| [How it works](docs/HOW_IT_WORKS.md) | the cache, the plugin, what it changes in llama.cpp (nothing) |
| [Exactness](docs/EXACTNESS.md) | how "same tokens as stock" is checked |
| [Conversation benchmark](docs/CONVERSATION_BENCH.md) | a long multi-turn chat that fills the context, stock against moe-cache, per depth band |
| [Results](docs/RESULTS.md) (Windows desktop), [older laptop results](docs/RESULTS_OLD_LAPTOP.md) and the [benchmark card](docs/benchmark_card.html) | every measurement, newest first |
| [Hardware and limits](docs/PLATFORMS.md) | what is tested where, known limitations |
| [Configuration](docs/CONFIGURATION.md) | environment variables |
| [FAQ](docs/FAQ.md) | questions, and how moe-cache compares with other projects |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | wrong-looking answers, slow runs, start problems |
| [Coding standards](docs/CODING_STANDARDS.md) | rules for anyone building or testing moe-cache |
| [Compatibility](docs/COMPATIBILITY.md), [Roadmap](docs/ROADMAP.md), [Security and privacy](docs/SECURITY.md) | |

## Support

moe-cache is built and tested in my spare time on my own machines. If it saves you a RAM upgrade, you can support it on [Ko-fi](https://ko-fi.com/ibrumm).

## Acknowledgements

- [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml (MIT, Copyright The ggml authors): moe-cache is a plugin for their backend interface, and it runs their CPU kernels unchanged.
- Credit for ideas on tiering experts across VRAM, RAM and SSD: [Strata](https://github.com/Niko1221/Strata) (MIT, by Niko1221) and [Project Maya](https://github.com/mw00/project-maya) (MIT, by mw00).

## License

- moe-cache is MIT-licensed, see [LICENSE](LICENSE).
- `third_party/ggml/` contains interface headers copied from ggml / llama.cpp (MIT, Copyright The ggml authors). Their license text is kept in `third_party/ggml/LICENSE`.
- moe-cache does not include any model weights. Models you run with it keep their own licenses.
- Product and model names (Qwen, gpt-oss, DeepSeek, Granite, Hunyuan, KAT-Coder and others) belong to their owners and are used only to say what was tested.
