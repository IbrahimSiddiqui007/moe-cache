<h1 align="center">moe-cache</h1>

<p align="center"><b>Run Mixture-of-Experts models that are bigger than your RAM, without the speed cliff.</b><br>
A plugin for an unmodified <code>llama-server</code>. No fork, no patch.</p>

<p align="center">
<img alt="status" src="https://img.shields.io/badge/status-v0.3.6%20beta-orange">
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
| Measured | Windows desktop, Flash-Next IQ2_XS (68 GB, bigger than RAM), a recorded Hermes bug-fix session: **81.6 s with moe-cache vs 163.9 s stock (2.0x)**, first token 21.8 vs 66.2 s ([RESULTS section 60](docs/RESULTS.md)) |

#### OpenCode

| Step | What to do |
|---|---|
| Install | OpenCode CLI, or OpenCode Desktop (opencode.ai/download) |
| Connect | `python3 bin/moe_cache.py connect opencode --write` (or **Connect OpenCode** on the control page): adds a `moe-cache` provider to `opencode.json` (or `opencode.jsonc`) and selects it; OpenCode Desktop reads the same folder |
| Use | `opencode`, `opencode web`, or OpenCode Desktop; pick `moe-cache/<model>` in the model list |
| Measured | same PC and model, a recorded OpenCode bug-fix session: **73.4 s with moe-cache vs 174.8 s stock (2.4x)**, first token 38.6 vs 116.8 s ([RESULTS section 60](docs/RESULTS.md)) |

#### Other agents

`connect` also writes the config for **Claude Code, Codex CLI, Pi, OpenClaw, Aider and Qwen Code**: `python3 bin/moe_cache.py connect NAME --write` (without `--write` it only prints the settings; the first backup of your config is kept as `.bak`). Checked against the real programs with a small model on 2026-11-03: Codex CLI, Pi, OpenClaw, Aider, OpenCode and Claude Code completed a turn (Claude Code with its settings exported in the shell); Qwen Code read its config but the 1.7B test model returned an empty answer, so it is not proven yet; Hermes was checked on 2026-10-29. Details: [docs/AGENTS.md](docs/AGENTS.md).

**Check it works:** `python3 bin/moe_cache.py connect NAME --check` sends one prompt and one tool call the way that agent does and prints pass/fail per step with the likely fix (wrong key, model name, window too big, tool calls off).

Keep each agent's own context limit below the server's, and set things up before a session: details, desktop apps and on-demand in [docs/AGENTS.md](docs/AGENTS.md).

### Which model?

`pick` decides from your RAM, graphics card, free disk and use (agents, chat, long documents). On a 32 GB RAM + 16 GB GPU PC: Qwen3.8-Flash-Next IQ3_XXS (76 GB) for general and agent use, the Q2_0 file (66 GB) for speed, Nemotron 3.5 Bolt 74B (22 GB, fits in RAM) for 128k contexts. `python3 bin/moe_cache.py models` lists every catalogue model with fits / tight / no for this PC, whether it is already downloaded and its measured speed (or "not measured"). All files and the rules: [docs/MODELS.md](docs/MODELS.md).

## Results

The card below is the Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan): stock llama.cpp against moe-cache with the same command line. It is generated from [docs/benchmark_card.html](docs/benchmark_card.html) (`python3 tools/card_markdown.py --write`), so it has the same numbers; the same page, drawn as a chart, is the image [docs/benchmark_card.png](docs/benchmark_card.png). A `claim:` marker is the id of the claim behind a number ([benchmarks/BOARD.md](benchmarks/BOARD.md)); a number without one has no claim yet. **Which build:** speeds were measured with the v0.3.5 kit and llama.cpp b11966 (2026-11-04 rows) or llama.cpp 620d2225c (2026-10-27/28 rows). The v0.3.6 kit bundles b12064 and its speed has not been re-measured, so the claims for the 2026-11-04 rows show as "stale" on the board until a run on v0.3.6 is recorded. Every 2026-11-04 run is marked "not clean" (the owner's programs were running).

<!-- card:begin -->

**Benchmark card · measured 2026-10-27 to 2026-11-04 · Windows desktop**

moe-cache is a llama.cpp plugin that keeps the experts a model uses most in RAM and reads the rest from disk. On this 32 GB Windows PC it decoded faster than stock llama-server on every model bigger than the RAM that was tested (short chats at 8k context, table below; long-context decode is measured against stock only for one file, 2026-10-28). The 256-token answers checked were byte-identical to stock (without MTP); a different GPU layer split can change a token while the meaning stays.

| Machine | |
|---|---|
| CPU | Ryzen 9 7950X (16 cores) |
| GPU | RX 9070 XT 16 GB |
| RAM | 32 GB DDR5 |
| OS | Windows 11 |
| llama.cpp builds | llama.cpp Vulkan: 620d2225c (ggml 0.27) for the Q2_0 and IQ2_XS rows of 2026-10-27/28, official b11966 for the 2026-11-04 rows and the 2026-11-03 table |

| Headline | What it is |
|---|---|
| +42 / +38 / +38 % | prompt reading speed at 34k / 64k / 128k tokens, v0.3.5 against v0.3.4 (Flash-Next Q2_0, clean run 2026-11-03, official llama.cpp b11966); decode after those prompts is 6–10 % lower <!-- claim:v035-vs-v034-34k --><!-- claim:v035-vs-v034-64k --><!-- claim:v035-vs-v034-128k --> |
| 1.8× | prompt reading at 64k context, moe-cache against stock: a 63.5k-token prompt in about 289 against 525 s, decode +23 % at that depth (Flash-Next Q2_0; one cold run each, 2026-10-28, Steam running, so not clean) |
| +26 % | prompt reading at the 256k setting with `-ub 2048` against `-ub 1024` (a 222k-token prompt; v0.3.5, 2026-11-03) <!-- claim:v035-ub2048-256k --> |
| 2.0–2.4× | faster coding-agent sessions (OpenCode, Hermes) on Flash-Next IQ2_XS at 65k agent context (2026-10-27, two cold runs each) |
| 8k chat | +42–53 % decode on three Qwen files and 2.1× on GLM-5.3 (2.7× with MTP), at 8k context with a 13-token prompt (2026-11-04, v0.3.5 kit, every run not clean) <!-- claim:chain1103-iq3xxs --><!-- claim:chain1103-iq3s --><!-- claim:chain1103-122b --><!-- claim:chain1103-glm --> |
| Pending | a clean stock against moe-cache run at long context (34k–256k) on the 2026-11-04 files and on v0.3.6 (TEST-684); the long-context stock numbers here are from 2026-10-28 |

### Decode speed, short chat (8k context, 13-token prompt), tokens per second

| Model (file size) | Context, prompt | GPU expert layers | stock | moe-cache | change |
|---|---|---|---|---|---|
| **Flash-Next Q2_0** (the faster file)<br><sub>66.4 GB, drive D:, two runs each, not interleaved; stock = launcher `--stock` defaults</sub> | 8192, short prompt (length not recorded) | 12 (moe-cache) | 18.41 | 22.16 | +20 % <!-- claim:card-decode-flashnext-q2_0 --> |
| Qwen3-Next “Flash-Next” IQ2_XS<br><sub>68 GB, drive C:</sub> | 8192, short prompt (length not recorded) | 12 | 14.89 | 21.94 | +47 % <!-- claim:card-decode-flashnext-iq2xs --> |
| Flash-Next IQ3_XXS<br><sub>76 GB, drive D:, 2026-11-04, v0.3.5 kit, two runs each (replaces 15.14 vs 19.86 of 2026-10-27)</sub> | 8192, 13 tokens | 8 | 14.96 | 22.87 | +53 % <!-- claim:chain1103-iq3xxs --> |
| Flash-Next IQ3_S<br><sub>83.6 GB, drive D:, 2026-11-04 (replaces 12.92 vs 17.28 of 2026-10-27)</sub> | 8192, 13 tokens | 6 | 13.79 | 19.62 | +42 % <!-- claim:chain1103-iq3s --> |
| Qwen3.5-122B UD-IQ2_XXS<br><sub>36.6 GB, drive D:, 2026-11-04 (replaces 15.95 vs 19.73 of 2026-10-28)</sub> | 8192, 13 tokens | 10 | 16.10 | 23.31 | +45 % <!-- claim:chain1103-122b --> |
| GLM-5.3-Flash UD-IQ2_XXS<br><sub>102 GB, drive D:, two runs each, ABBA, 2026-11-04 (replaces the single pair 2.21 vs 4.44 of 2026-10-27)</sub> | 8192, 13 tokens | 0 | 2.98 | 6.27 | 2.1× <!-- claim:chain1103-glm --> |
| GLM-5.3-Flash, moe-cache with `--mtp`<br><sub>two runs; stock without MTP is the other side (replaces the single run 6.92)</sub> | 8192, 13 tokens | 0 | 2.98 | 8.17 | 2.7× <!-- claim:chain1103-glm --> |
| Nemotron 3.5 Bolt 74B-A9B<br><sub>22.0 GB, drive C:, **fits in RAM**: the launcher runs stock for it; moe-cache forced here to compare (two runs each, 2026-11-04; the single runs of 2026-10-28 gave −18 %)</sub> | 8192, 13 tokens | – | 29.99 | 31.68 | +6 % <!-- claim:chain1103-bolt --> |
| Qwen3.6-35B-A3B Q4_K_M<br><sub>22 GB, drive C:, **fits in RAM**: the launcher runs stock for it; moe-cache forced (two runs each, 2026-11-04; 2026-10-27 gave −1 %)</sub> | 8192, 13 tokens | – | 46.59 | 51.80 | +11 % <!-- claim:chain1103-qwen36 --> |
| Qwen3.6-35B, moe-cache with MTP<br><sub>against stock without MTP; stock with MTP was not measured on 2026-11-04 (68.6 on 2026-10-27)</sub> | 8192, 13 tokens | – | 46.59 | 72.69 | +56 % <!-- claim:chain1103-qwen36 --> |

Each pair is two cold runs in the order stock, moe-cache, moe-cache, stock (ABBA), and the table shows the means. Every model here is bigger than the free RAM once the GPU and Windows have taken their share, except the two marked “fits in RAM”. **Dates and builds:** the rows for IQ3_XXS, IQ3_S, Qwen3.5-122B, GLM-5.3 and the two models that fit were re-measured on 2026-11-04 (overnight chain, the released v0.3.5 kit with the official llama.cpp b11966, launcher defaults, context 8192, a 13-token request, two runs each; every run was “not clean”: owner work was running on the PC). The rows for Q2_0 and IQ2_XS are from 2026-10-27/28 (llama.cpp 620d2225c, RESULTS 58–65) and were not re-run; later changes (128-row executor chunks, +3.9 % on Q2_0; async reads, +12 %, RESULTS 67) are in the 2026-11-04 rows but not in those two. The chart and its detail lines use the same numbers. These rows describe the v0.3.5 kit; the current pin is b12064, so a paired re-run is still open. Q2_0: stock decodes it much faster than IQ2_XS (18.4 vs 14.9), so the plugin’s decode gain there is +20 % (code request +13 %); its prompt gain is 2.33× (3k-token prompt 120.8 vs 281 tok/s), which is what agent sessions wait on (RESULTS 63).

Test details for each row (the card draws the decode table as bars; same numbers):

| Model | Size, drive | stock tok/s | moe-cache tok/s | with MTP tok/s | Test details |
|---|---|---|---|---|---|
| Flash-Next Q2_0 | 66.4 GB, drive D: | 18.41 | 22.16 | – | GPU expert layers: 12 (stock: --stock defaults, count not recorded)<br>ctx 8192 · short prompt · 2 runs each, not interleaved<br>VRAM not recorded · RAM: cache limit about 23 GiB of 32 GB<br>launcher 1d2d5a8, DLL 681f3bc · llama.cpp 620d2225c · 10-28 · §63 |
| Flash-Next IQ2_XS | 68 GB, drive C: | 14.89 | 21.94 | – | GPU expert layers: 12 (stock and moe-cache)<br>ctx 8192 · short prompt · ABBA, 2 runs each<br>VRAM not recorded · RAM limit not recorded (32 GB)<br>launcher 65facd1, DLL 681f3bc · llama.cpp 620d2225c · 10-27 · §59 |
| Flash-Next IQ3_XXS | 76 GB, drive D: | 14.96 | 22.87 | – | GPU expert layers: 8 (moe-cache; stock = launcher defaults, count not recorded)<br>ctx 8192 · 13-token water request · ABBA, 2 runs each · every run 'not clean' (owner work running)<br>moe-cache peak: VRAM 12732 MiB · process RAM 22838 MiB<br>v0.3.5 kit · llama.cpp b11966 (ff0d396e3) · 11-04 · run 2026-11-04_pc_cfa905 · claim chain1103-iq3xxs |
| Flash-Next IQ3_S | 83.6 GB, drive D: | 13.79 | 19.62 | – | GPU expert layers: 6 (moe-cache; stock = launcher defaults, count not recorded)<br>ctx 8192 · 13-token water request · ABBA, 2 runs each · every run 'not clean' (owner work running)<br>moe-cache peak: VRAM 12211 MiB · process RAM 23779 MiB<br>v0.3.5 kit · llama.cpp b11966 (ff0d396e3) · 11-04 · run 2026-11-04_pc_42b419 · claim chain1103-iq3s |
| Qwen3.5-122B IQ2_XXS | 36.6 GB, drive D: | 16.10 | 23.31 | – | GPU expert layers: 10 (moe-cache; stock = launcher defaults, count not recorded)<br>ctx 8192 · 13-token water request · ABBA, 2 runs each · every run 'not clean' (owner work running)<br>moe-cache peak: VRAM 12180 MiB · process RAM 21980 MiB<br>v0.3.5 kit · llama.cpp b11966 (ff0d396e3) · 11-04 · run 2026-11-04_pc_15144c · claim chain1103-122b |
| GLM-5.3-Flash IQ2_XXS | 102 GB, drive D: | 2.98 | 6.27 | 8.17 | GPU expert layers: 0 (moe-cache; stock = launcher defaults, count not recorded)<br>ctx 8192 · 13-token water request · ABBA, 2 runs each · every run 'not clean' (owner work running)<br>moe-cache peak: VRAM 7867 MiB · process RAM 23632 MiB<br>v0.3.5 kit · llama.cpp b11966 (ff0d396e3) · 11-04 · run 2026-11-04_pc_bea283 · claim chain1103-glm |
| Nemotron 3.5 Bolt 74B | 22 GB, fits, drive C: | 29.99 | 31.68 | – | GPU expert layers: not recorded (moe-cache; stock = launcher defaults, count not recorded)<br>ctx 8192 · 13-token water request · ABBA, 2 runs each · every run 'not clean' (owner work running)<br>moe-cache peak: VRAM 10299 MiB · process RAM 12514 MiB<br>v0.3.5 kit · llama.cpp b11966 (ff0d396e3) · 11-04 · run 2026-11-04_pc_00ce69 · claim chain1103-bolt |
| Qwen3.6-35B Q4_K_M | 22 GB, fits, drive C: | 46.59 | 51.80 | 72.69 | GPU expert layers: 20 (moe-cache; stock = launcher defaults, count not recorded)<br>ctx 8192 · 13-token water request · ABBA, 2 runs each · every run 'not clean' (owner work running)<br>moe-cache peak: VRAM 12402 MiB · process RAM 12020 MiB<br>v0.3.5 kit · llama.cpp b11966 (ff0d396e3) · 11-04 · run 2026-11-04_pc_92657e · claim chain1103-qwen36 |

### Long context (64k–256k)

Agent mode (`--agent --ctx N`), a real 63.5k-token text fed in chunks, decode of 128 tokens at that depth. **One cold run per point** (not ABBA), Steam running on the PC.

| Model, context setting | prefill tok/s at 4–21k / 53–63k | decode tok/s at 4k | decode tok/s at 63.5k | deeper |
|---|---|---|---|---|
| **Flash-Next Q2_0, moe-cache**, 64k | **266 / 195** | 22.89 | **20.65** | – |
| Flash-Next Q2_0, stock, 64k | 171 / 106 | 21.38 | 16.83 | – |
| Flash-Next Q2_0, moe-cache, 256k setting | 142 / 111 | 18.52 | – | 15.93 at 138k |
| Qwen3.5-122B, moe-cache, 64k | 266 / 217 | 19.69 | 17.26 | – |
| Qwen3.6-35B, stock (fits), 64k | 848 / 739 | 67.21 | 59.09 | – |

- **At 64k, Flash-Next with moe-cache** reads the whole 63.5k prompt 1.8× faster than stock (about 289 vs 525 s) and decodes 23 % faster at that depth (20.65 vs 16.83 tok/s).
- **The 256k setting is slower today:** the planner moves every expert layer to the CPU and uses `-ub 512`, so prefill is about half and decode at 4k is 19 % lower than at the 64k setting. The table above is v0.3.4. In v0.3.5 the planner uses `-ub 2048` at 256k: a 222k-token prompt reads 26 % faster (334.8 / 333.2 vs 263.4 / 265.8 tok/s, RESULTS 79); decode there is not re-measured. <!-- claim:v035-ub2048-256k -->
- Decode falls 10–14 % with depth on every model (attention), with the plugin's hit rate still 95–97 %.
- Saving the agent's conversation at 64k blocks the server for 1.7–2.4 s (5.5 s at 138k). At the 256k setting one decode after a save did not finish (under investigation).
- Not run yet: Qwen3.6 and the 122B at 256k, the 122B stock at 64k. Source: RESULTS 64.

### Long prompts: v0.3.5 vs v0.3.4

Windows desktop, Flash-Next Q2_0, cold file cache each run, order new, old, old, new; the PC was restarted that night and its baseline check passed (clean run, 2026-11-03). Both versions ran on the official llama.cpp b11966 server. Two runs per cell (first / second).

| Context, prompt | v0.3.4 prompt tok/s | v0.3.5 prompt tok/s | change | v0.3.4 decode tok/s | v0.3.5 decode tok/s | change |
|---|---|---|---|---|---|---|
| 34k | 408.6 / 403.1 | **577.0 / 572.6** | +42 % | 28.43 / 28.59 | 27.04 / 26.38 | −6 % <!-- claim:v035-vs-v034-34k --> |
| 64k | 313.8 / 314.1 | **421.3 / 448.3** | +38 % | 27.42 / 27.04 | 25.69 / 23.44 | **−10 %** <!-- claim:v035-vs-v034-64k --> |
| 128k | 304.8 / 305.5 | **420.7 / 421.9** | +38 % | not measured | not measured | – <!-- claim:v035-vs-v034-128k --> |

- Prompts are about 1.4× faster at the three depths.
- **Decode after a long prompt is 6–10 % lower.** The bigger prompt chunk takes GPU memory, so 2 fewer expert layers fit on the GPU (39–40 on the CPU at 64k, against 37–38 for v0.3.4). With 2 more GPU layers at 64k decode was 26.6 / 26.9 against 23.7 / 22.9. <!-- claim:decode-room-64k -->
- Other runs the same night: 256k setting (222k-token prompt) +26 % (334.8 / 333.2 vs 263.4 / 265.8 tok/s); Qwen3.5-122B at 64k +24 %; GLM-5.3-Flash at 30k +62 % with decode −6 %. <!-- claim:v035-ub2048-256k --><!-- claim:guard-122b-64k --><!-- claim:guard-glm-30k -->
- Source: RESULTS 79. Text was not compared in this run. The earlier “experimental” numbers (patched server) overstated this by about 5–7 %.

### Which Flash-Next file

**Decode, 3k/34k prompt speed and tasks were measured 2026-10-28 on v0.3.x (RESULTS 63). The two columns marked “v0.3.5, 2026-11-04” are new: the 34k and 64k prompts and the short answer after them, two passes each on the released v0.3.5 kit with llama.cpp b11966 (every run “not clean”: owner work was running). IQ2_XS and IQ3_XXS at 64k: the overnight run had no decode-after value (after the 64k prompt the model wrote only 2 tokens and ended its answer, so the follow-up generated 0 tokens: model behaviour, not a plugin fault; the prompt speeds above are valid and clean). The decode-after values in those two cells are from a daytime re-run with the fixed harness (128 tokens) that was **not clean** (the owner’s browser held 1.4 GB of VRAM; its prompt speeds were about 40 % lower and are not used); a clean re-run is planned.** Five GSQ-RCO files of Qwen3.8-Flash-Next through moe-cache on the same PC, launcher defaults, cold. Tasks = 202 exact-answer tasks, greedy. A slower file has to earn its place with clearly better quality.

| File | size | decode tok/s | prompt tok/s 3k / 34k <sub>2026-10-28</sub> | prompt tok/s 34k / 64k <sub>v0.3.5, 2026-11-04</sub> | decode after the prompt 34k / 64k <sub>v0.3.5, 2026-11-04</sub> | tasks /202 | verdict |
|---|---|---|---|---|---|---|---|
| **Q2_0** | 66.4 GB | **22.2** | **281 / 287** | 568 / 399<!-- claim:chain1103-q2_0-34k --><!-- claim:chain1103-q2_0-64k --> | 27.4 / 25.5 | 191 | **the faster file** (about 10 % faster than IQ3_XXS). vs IQ2_XS: +1.2 tok/s (+5 %), prompts +11 % / +19 %, 1.6 GB smaller, +4 tasks (not proven) |
| IQ3_XXS | 75.8 GB | 19.9 | – | 349 / 182<!-- claim:chain1103-iq3xxs-34k --><!-- claim:chain1103-iq3xxs-64k --> | 16.1 / 12.3<br><sub>64k: daytime re-run, not clean (browser held 1.4 GB VRAM); clean re-run planned</sub><!-- claim:chain1103-iq3xxs-64k-decode --> | **193** | **recommended default** (3-bit, reads best, popular for coding). vs Q2_0: −2.3 tok/s (−10 %) for +2 tasks (not proven) |
| Coder IQ1_M<br><sub>ISTA GSQ-RCO, 256 of 512 experts</sub> | 58.4 GB | 21.2<br><sub>agent session</sub> | – | not re-run | not re-run | 189 | **coding agents.** OpenCode session 55.5 s vs 73.4 s on IQ2_XS (−24 %, one run); general knowledge not checked beyond the task set |
| IQ2_XS | 68.0 GB | 21.0<br><sub>21.9 in the ABBA above</sub> | 253 / 242 | 504 / 317<!-- claim:chain1103-iq2xs-34k --><!-- claim:chain1103-iq2xs-64k --> | 24.7 / 16.9<br><sub>64k: daytime re-run, not clean (browser held 1.4 GB VRAM); clean re-run planned</sub><!-- claim:chain1103-iq2xs-64k-decode --> | 187 | superseded by Q2_0: slower everywhere, bigger, 4 fewer tasks (not proven) |
| IQ3_S | 83.6 GB | 17.3 | – | 278 / 221<!-- claim:chain1103-iq3s-34k --><!-- claim:chain1103-iq3s-64k --> | 13.4 / 11.9 | 192 | **not worth it.** vs IQ3_XXS: −2.6 tok/s (−13 %), 7.8 GB bigger, −1 task |

**No quality difference between any two files is proven**: on the tasks that changed, every paired sign test has p > 0.05 (e.g. IQ3_XXS vs IQ2_XS 10 / 4 tasks, p 0.18). Q2_0 vs IQ2_XS was a mirrored pair on the same day; in its decode run Q2_0 sat on the faster drive, so part of the +5 % may be the drive (the 34k-prompt pair had both on D:). IQ3_XXS and IQ3_S decode are the ABBA numbers from the table above. Stock vs moe-cache on Q2_0 is in the decode table above. Source: RESULTS 63.

### Other models

A newer model that fits in RAM + VRAM, measured on the same PC. One cold run per point. It needs llama.cpp b11966 or newer (tokenizer fix).

| Model | size | runs as | decode tok/s | long prompt tok/s | tasks /202 | OpenCode session |
|---|---|---|---|---|---|---|
| **Nemotron 3.5 Bolt 74B-A9B**<br><sub>NVIDIA 2-bit QAT file</sub> | 22.0 GB | stock (fits; launcher default) | **27.9**<br><sub>27.8 / 24.9 / 23.2 at 57k / 114k / 228k</sub> | **595** at 57k<br><sub>721 at 114k, 280 at 228k</sub> | 190 | **47.8 s** |
| Nemotron, moe-cache forced | 22.0 GB | plugin | 22.1 at 57k<br><sub>22.8 short</sub> | 329 at 57k | – | 59.2 s |
| Flash-Next Q2_0 (for comparison) | 66.4 GB | moe-cache | 22.2 | 287 at 34k | 191 | –<br><sub>IQ2_XS 73.4 s, Coder 55.5 s</sub> |

Nemotron fits in this PC's memory, so stock llama.cpp is the right choice and the launcher picks it by itself; forcing moe-cache on it cost 18 % decode and 45 % prompt speed in these single runs (2026-10-28) <!-- claim:card-decode-nemotron -->; two runs each on 2026-11-04 (top table, v0.3.5 kit, not clean) gave +6 % decode <!-- claim:chain1103-bolt -->, so there is no steady gain either way. moe-cache is for models bigger than memory. Tasks: 190 vs Flash-Next Q2_0's 191 (not proven, p 1.00); general knowledge beyond the task set not checked. DeepSeek V4-Flash was not tested (desk estimate only). Source: RESULTS 65.

### GLM-5.3-Flash: best setting on this PC

| Setting (moe-cache, model on drive D:, apps closed, cold, ABBA) | chat text tok/s | code tok/s | 3k prompt tok/s |
|---|---|---|---|
| **launcher default** (45 CPU expert layers), no MTP | **5.89** | **2.80** | **67.6** |
| 2 more GPU layers (`--n-cpu-moe 43`) | 6.28 (+6.6 %) | 3.04 (+8.6 %) | 59.1 (−12.6 %) |

- **Best:** the model on the faster drive and the launcher's default layers. The faster drive gave +3.6 % on chat text and +13 % on code (fixed memory, RESULTS 62).
- **43 layers:** better for chat, but one agent turn (3k prompt + 300 tokens out) takes 99 s vs 95 s, so the default stays.
- **Code is slow** (2.8 tok/s): GLM's code experts do not fit in the 14–19 GiB cache; the same request hits no better the second time. Needs more RAM, not a setting.
- This table is from 2026-10-28 and has no stock side. The stock ratio for GLM is the 2026-11-04 pair in the top table (stock 2.98, moe-cache 6.27 tok/s, drive D:) <!-- claim:chain1103-glm -->; the single stock run of 2026-10-27 (2.21 tok/s) was on drive C:.

### Qwen3.6-35B: a model that fits in RAM, with and without MTP

Qwen3.6-35B-A3B Q4_K_M (20 GB) fits in the 32 GB, so the launcher runs stock llama-server for it by itself; moe-cache was forced on here to compare. MTP = the model's own multi-token-prediction head on the GPU (`--spec-type draft-mtp --spec-draft-n-max 2`).

| Server | no MTP, tok/s | MTP, tok/s | MTP gain |
|---|---|---|---|
| stock llama-server | 46.01 / 46.10 | 68.28 / 68.92 | +49 % <!-- claim:card-decode-qwen36 --> |
| moe-cache (forced) | 45.57 / 45.62 | **70.29 / 70.20** | +54 % <!-- claim:card-decode-qwen36 --> |

Measured 2026-10-27 (older kit, llama.cpp 620d2225c): two measurements per cell, cold, PC above (RESULTS 58); the 2026-11-04 rows in the top table (+11 % without MTP, +56 % with MTP against stock without MTP) <!-- claim:chain1103-qwen36 --> are a separate, newer run. About 78 % of the drafted tokens were accepted. Prompt processing is 5–8 % slower with MTP. Without MTP the plugin and stock decode at the same speed, as expected for a model that fits.

### Why MTP is on for some models and off for others

With MTP a small extra head guesses the next two tokens, and the model then checks three tokens in one pass. The answer stays the same (a wrong guess is thrown away). It pays when checking three tokens costs little more than checking one. That is true when the experts the three tokens need are already in RAM or VRAM. It is not true when most of them sit on the CPU or have to be read from disk.

| Model | MTP head | Measured | Launcher |
|---|---|---|---|
| Qwen3.6-35B-A3B<br><sub>fits in RAM</sub> | in the file | decode +49 % stock, **+54 %** moe-cache (table above, 2026-10-27); agent sessions 13–25 % shorter (RESULTS 60) | on in agent mode; `--mtp` otherwise |
| GLM-5.3-Flash<br><sub>102 GB, bigger than RAM</sub> | in the file (`blk.45.nextn`) | moe-cache 4.44 → **6.92 tok/s** (3.1× stock) on 2026-10-27 <!-- claim:card-decode-glm -->, text identical; one cold run on chat text. **Re-measured 2026-11-04** (v0.3.5 kit, two runs each, short water request): stock 2.98, moe-cache 6.27, with `--mtp` **8.17 tok/s** (2.7× stock) <!-- claim:chain1103-glm -->. Later runs (RESULTS 62, D:, fixed memory): chat text **+10–23 %** at 100 % draft acceptance, code **0 %** at 79 %: MTP's draft context takes memory the cache needed. With 2 extra GPU layers it does not fit (load abort). Before the file-view cache it lost (0.76–0.99 vs 2.0–2.2): MTP's GPU memory counts against Windows' commit and shrank the cache to 4.25 GiB | on in agent mode; `--mtp` otherwise |
| Flash-Next (every quant), Qwen3.5-122B<br><sub>bigger than RAM</sub> | none in the files | a separate head file (unsloth Q8_0, 4.14 GB) on Flash-Next IQ2_XS: stock **−15 %**, moe-cache **−38 %**, with 73–74 % of drafts accepted. Checking three tokens at once on CPU-held experts cost more than the drafts saved. Measured before the file-view cache, not re-run | off: no head; separate head files are not supported (dropped) |

The launcher reads the model file: MTP is turned on only when the file has `blk.N.nextn` tensors and llama-server has `draft-mtp`; it prints one line saying why it is on or off. `--no-mtp` turns it off. Sources: RESULTS 58 (Qwen3.6, head file), 60 and 62 (GLM).

### Agent sessions

Recorded OpenCode (5 requests) and Hermes (7 requests) sessions were replayed against each server with the same request bodies, overlap and pauses. Each figure is the mean of two cold runs, order ABBA, with 0 errors in every replay.

| Model | Session | whole task, s stock / moe-cache | first answer, s stock / moe-cache | decode tok/s stock / moe-cache |
|---|---|---|---|---|
| Flash-Next IQ2_XS<br><sub>68 GB, bigger than RAM</sub> | OpenCode | 174.8 / **73.4** <br><sub>2.4× faster</sub> | 116.8 / **38.6** | 21.5 / 21.2 |
|   | Hermes | 163.9 / **81.6** <br><sub>2.0× faster</sub> | 66.2 / **21.8** | 21.1 / 21.3 |
| Flash-Next IQ2_XS, both at once<br><sub>two agents on one server</sub> | OpenCode + Hermes | 335 / **145** <br><sub>2.3× faster</sub> | 177–197 / **54–61** | – |
| Qwen3.6-35B-A3B Q4_K_M, MTP on<br><sub>20 GB, fits in RAM</sub> | OpenCode | 27.8 / 26.4 <br><sub>5 % faster</sub> | 13.5 / 13.4 | 72.4 / 72.7 |
|   | Hermes | 25.3 / 23.0 <br><sub>9 % faster</sub> | 5.2 / 5.1 | 71.5 / 72.2 |

**Idle on-demand server** (`on-demand --idle-release`, default 5 min): after the idle time the plugin gives the cache's RAM back and the server keeps running: 17.39 / 17.55 GiB given back in two rounds, available RAM 6.4 → 24.4 GB. The next answer ran at 18.67 / 18.46 tok/s (pages come back from Windows' standby list, ~2.7 s, no disk reads), the one after at 22.37 / 22.12; a restart instead takes ~40 s (Flash-Next IQ2_XS, RESULTS 61).

On the big model the gain is in reading the prompt: the first answer comes about 3× sooner and every follow-up answer starts in 2–3 s instead of 7–9 s. Decode speed at agent context length is the same for both.

### Where it does not help

- **Models that fit in RAM.** Qwen3.6-35B (20 GB) decoded at the same speed with or without the plugin on 2026-10-27 (46.1 vs 45.6 tok/s without MTP, RESULTS 58); two runs each on 2026-11-04 gave 46.59 stock vs 51.80 moe-cache (+11 %, not clean) <!-- claim:chain1103-qwen36 -->. No steady gain, so the launcher runs stock. Agent sessions are only 5–9 % shorter there, from faster follow-up prompts.
- **Decode at long agent contexts.** On Flash-Next the 2.0–2.4× session gain comes from prompt processing. Decode at 65k-context agent settings is equal (21.5 vs 21.2 tok/s).
- **CPU kernels.** The numbers here were measured with llama.cpp's own CPU kernels (ggml-cpu-zen4) on both sides. The plugin's exact Q2_0 executor kernel, on Windows since 2026-10-28, did not change the PC's decode (it is limited by RAM bandwidth). The gain comes from which experts are in RAM and how they are read, not from faster math.
- **Cold start.** From a cold file cache the plugin reports ready 5.6 s later than stock (28.5 vs 22.9 s) while it loads its hot experts; the first answer wins it back. A warm restart answers 6 s sooner than stock.
- **Small samples.** Two runs per side on every decode pair (GLM with MTP too, 2026-11-04); long-context rows are one cold run per point. Every 2026-11-04 run was “not clean” (owner work was running). Run-to-run spread was under 1 tok/s on the 2026-10-27/28 pairs; it was not checked for the 2026-11-04 pairs.
- **MTP on models without a head.** Flash-Next and the 122B files have no MTP head; a separate head file made both stock and moe-cache slower (see above).

### Output quality

The plugin does not change the math: the 256-token answers were byte-identical to stock on Flash-Next IQ2_XS (1041 characters), IQ3_S (1032), the 122B (1027) and GLM (1148), and GLM with MTP matched stock too. Separately, 202 exact-answer tasks (greedy, no thinking) on the plugin scored 193/202 with Flash-Next IQ3_XXS and 187/202 with IQ2_XS.

### Method

- **Cold every run.** Before each server start, `tools\win_flush_cache.ps1` reads 38.5 GB of other files so Windows' file cache no longer holds the model. Stock benefits most from a warm cache, so this is the fair setting for both.
- **ABBA order.** Stock, moe-cache, moe-cache, stock, so drift (heat, background work) hits both sides equally.
- **Same request.** A 64-token warm-up, then one greedy 256-token request (temperature 0, seed 42, `cache_prompt` off). Decode tok/s is llama-server's own timing.
- **Same layout.** Stock gets the same GPU expert layers, context, flash attention and batch size that the launcher picks for the plugin.
- **Text identical.** The plugin's answer is compared with stock's byte for byte.

### Commands

**moe-cache (launcher defaults; it prints the llama-server command it runs)**

```
python bin\moe_cache.py server C:\models\model.gguf --llama-server C:\llama-vk\llama-server.exe
```

**Stock llama-server (N = the plugin's CPU expert layers, e.g. 36 for Flash-Next IQ2_XS)**

```
llama-server.exe -m C:\models\model.gguf -ngl 99 --n-cpu-moe N -c 8192 -fa on --load-mode mmap -ub 2048
```

**Flush the file cache before each run (reads other big files, writes nothing)**

```
powershell -ExecutionPolicy Bypass -File tools\win_flush_cache.ps1 -Target C:\models\model.gguf -Other "D:\other-models"
```

**The measured request (after a 64-token warm-up)**

```
curl http://127.0.0.1:8080/completion -H "Content-Type: application/json" -d "{\"prompt\":\"Describe the water cycle step by step for a ten-year-old.\",\"n_predict\":256,\"temperature\":0,\"seed\":42,\"cache_prompt\":false}"
```

**Agent sessions (stock and moe-cache, ABBA, cold)**

```
powershell -NoProfile -ExecutionPolicy Bypass -File tools\agent_bench_pair.ps1 -OutDir C:\dl\i43 -Model C:\models\model.gguf -Alias flash -Ram <GB> -FlushOther "D:\other-models"
```

<!-- card:end -->

On Linux (older laptop runs, v0.2.x) the gains were 2.4x to 5.2x: [docs/RESULTS_OLD_LAPTOP.md](docs/RESULTS_OLD_LAPTOP.md).

**What to expect by memory budget** (decode tok/s, `--ram` = the memory the cache is sized to (a sizing hint, **not a RAM cap**: on Windows the default path sets no memory limit, so the server can use more than the number you give; on Linux the launcher sets none either; see [docs/WINDOWS.md](docs/WINDOWS.md)); only numbers we measured, one cold run each, "not measured" where we have none):

| `--ram` | Windows desktop, Flash-Next IQ2_XS (68 GB), 16 GB GPU | Linux laptop, Qwen3-Next-80B (48.5 GB), about 2.6 GB GPU use |
|---|---|---|
| 24 GB | **21.81** (RESULTS §60) | **10.6**, stock 2.3 ([RESULTS_OLD_LAPTOP §6](docs/RESULTS_OLD_LAPTOP.md); v0.1 build) |
| 20 GB | **21.00** (RESULTS §60) | **9.3**, stock 1.3 ([RESULTS_OLD_LAPTOP §6](docs/RESULTS_OLD_LAPTOP.md); v0.1 build) |
| 16 GB | **20.03**, 92 % of the 24 GB speed (RESULTS §60, after the 2026-10-27 job-limit fix) | not measured |
| 12 GB, 8 GB, 32 GB or more | not measured | not measured |

The two columns are different machines and models, so compare down a column, not across. A model that fits in RAM is not in this table: plain llama.cpp is as fast or faster there.

Everything, with methods and caveats: [docs/RESULTS.md](docs/RESULTS.md) (Windows desktop), [docs/RESULTS_OLD_LAPTOP.md](docs/RESULTS_OLD_LAPTOP.md) (Linux laptop), the [benchmark card page](docs/benchmark_card.html). Measure your own machine: the control page's **Share my numbers** button, or `python3 bin/moe_cache.py bench-report MODEL.gguf` ([docs/SHARE_YOUR_NUMBERS.md](docs/SHARE_YOUR_NUMBERS.md)).


## Known issues in v0.3.6 (honest list)

| Issue | Detail |
|---|---|
| Not byte-identical at 128k+ by default | at `--ctx 131072` and above the KV cache is q8_0 (`--fast=kv`, about 2x prompt speed at 256k on a 16 GB card, from the bigger GPU plan the smaller KV allows): it passed the quality gate at 4k context (KL 0.0079, top-1 96.7 %, tasks = stock), but not the same bytes as stock. `--exact` keeps llama.cpp's f16 KV |
| Windows zip is unsigned | Windows shows "unknown publisher" / SmartScreen: Unblock the zip, More info > Run anyway (see Get started). Check the SHA256 or `gh attestation verify` |
| Windows tested on one desktop | Ryzen 9 7950X + RX 9070 XT 16 GB + 32 GB, Windows 11, Vulkan, plus a GitHub Windows runner (CPU). NVIDIA, Intel GPUs and Windows 10 are untested; HIP loads but is slower than Vulkan |
| Models measured | Qwen3.8-Flash-Next (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, Coder IQ1_M), Qwen3.5-122B IQ2_XXS, GLM-5.3-Flash, Qwen3.6-35B, Nemotron 3.5 Bolt 74B (agent speed; plugin forced slower than stock in one test), Granite 1B (CI). Other MoE models should work but are not measured. Long context (64k-256k) is measured on Flash-Next and a few more, single runs |
| Linux | Tested on Intel + NVIDIA, CPU-only, Vulkan, and a user's RX 9070 XT. `install.sh` has been tried on Arch (`--no-apt`) and, for this release, from a clean clone on that machine; a fresh Ubuntu/Debian run with apt is not tested. macOS: no |
| `--ram` is a sizing hint | On Windows the memory limit is set only for the plugin, only with `--ram`, and only with `MOE_CACHE_WIN_VIEW=0`; the default file-view cache, the on-demand listener and stock llama-server get none (a job limit counts GPU memory and broke Vulkan). The server can use more RAM than `--ram` says. On Linux the launcher sets no limit either; only a cgroup you set yourself is a hard limit. No guard is being built |
| Crash on a truncated or unreadable model file | a model file that is cut short or cannot be read can still crash the server instead of giving a clear error (CACHE-685, being fixed) |
| Installer attestation is skipped without `gh` | `install.sh` checks the build attestation only when the GitHub CLI (`gh`) is installed and logged in; without it only the SHA256 file is checked, which catches a damaged download but not a tampered release (RELEASE-686) |
| GPU errors | with Vulkan a GPU error inside an expert block is caught and the block runs on the CPU (`MOE_CACHE_GPU_CATCH`, not yet seen with a real failure); on NVIDIA / CUDA a GPU error still ends the server |
| Context length of the speed numbers | the 2026-11-04 decode numbers are short chats (8k context, 13-token prompt). Long-context (34k-256k) stock-against-moe-cache numbers are from 2026-10-28 (one file, one cold run) and the v0.3.5 against v0.3.4 comparison; a clean long-context stock run on the newest files is pending (TEST-684) |
| Not tested for this release | `winget install` and `scoop install` (manifests exist but are not submitted); Hermes Desktop and OpenCode Desktop on Windows (click-through still open); a full model run from the final zip |
| Decode after long prompts | 6-10 % slower than v0.3.4 after a 34k-64k prompt (the bigger prompt chunk leaves 2 fewer expert layers on the GPU); prompts are about 1.4x faster. A later release aims to give that GPU memory back after the prompt |
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
- Related projects with a similar approach (tiering experts across VRAM, RAM and SSD), built as their own engines: [Strata](https://github.com/Niko1221/Strata) (MIT, by Niko1221) and [Project Maya](https://github.com/mw00/project-maya) (MIT, by mw00; started from Strata's code).

## License

- moe-cache is MIT-licensed, see [LICENSE](LICENSE).
- `third_party/ggml/` contains interface headers copied from ggml / llama.cpp (MIT, Copyright The ggml authors). Their license text is kept in `third_party/ggml/LICENSE`.
- moe-cache does not include any model weights. Models you run with it keep their own licenses.
- Product and model names (Qwen, gpt-oss, DeepSeek, Granite, Hunyuan, KAT-Coder and others) belong to their owners and are used only to say what was tested.
