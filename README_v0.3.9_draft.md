<h1 align="center">moe-cache: run Mixture-of-Experts LLMs bigger than your RAM with llama.cpp</h1>

<p align="center"><b>Run big MoE models (Qwen3, GLM, gpt-oss and more) that do not fit in your RAM on a normal PC with llama.cpp: the experts you use stay in RAM, the rest is read from your SSD.</b><br>
moe-cache is a plugin that runs inside <code>llama-server</code>; it also works on the unmodified official one. No fork: the Windows kit and the Linux `-ggml0.26` tarball (file name kept) bundle our build of llama.cpp b12651 (ggml 1.0) built from the official source plus a small patch set (<code>patches/llama.cpp</code>). Every patch is built in but off until you set its own switch, so by default the build behaves as the official one (checked at temperature 0 in CI). To use the official build instead, point <code>LLAMA_SERVER</code> at it.</p>

<p align="center">
<img alt="status" src="https://img.shields.io/badge/status-v0.3.9-blue">
<img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
<img alt="platform" src="https://img.shields.io/badge/platform-Linux%20%2B%20Windows%20(experimental)-lightgrey">
<img alt="llama.cpp" src="https://img.shields.io/badge/llama.cpp-b12651%20(ggml%201.0)%20bundled%2C%20CI%20tested-green">
<a href="https://ko-fi.com/ibrumm"><img alt="Support on Ko-fi" src="https://img.shields.io/badge/Ko--fi-support%20this%20project-ff5e5b?logo=ko-fi&logoColor=white"></a>
</p>

<p align="center">Test settings and every caveat are in <a href="#results">Results</a> below (Windows desktop, 32 GB RAM + 16 GB graphics card). This release ships without speed figures: no stock-against-moe-cache table and no headline speed claim against plain llama.cpp; Speed numbers for this release follow in v0.3.9.1. Full history: <a href="docs/results-archive/RESULTS.md">docs/results-archive/RESULTS.md</a>; the older Linux laptop results: <a href="docs/results-archive/RESULTS_OLD_LAPTOP.md">docs/results-archive/RESULTS_OLD_LAPTOP.md</a>.</p>

## What it is

**Who it is for:** you have a PC with 32 GB RAM (16 GB for smaller models), perhaps a 16 GB graphics card (AMD, NVIDIA or Intel, Windows 11 or Linux), a fast SSD, and you want to run a large local Mixture-of-Experts model (60-120 GB file) for chat, long documents or a coding agent, and plain llama.cpp is too slow because the model does not fit in RAM. **Who it is not for:** if the whole model fits in RAM, or in your graphics card, plain llama.cpp is fine — stock is as good or better there, and for those the launcher runs stock when all experts fit on the card, and the plugin otherwise (see Known issues; [when to use something else](docs/FAQ.md#when-to-use-something-else)).

**moe-cache lets a normal PC run big Mixture-of-Experts AI models** (files of 60-120 GB) that are bigger than its RAM, with the plugin itself changing no number: with the same settings its answers are the same as plain llama.cpp's in the checks we ran (greedy decoding, the same GPU/CPU layer split, same llama.cpp build; the exact list is in [EXACTNESS](docs/EXACTNESS.md)). **By default, from 32k context on a graphics card with room, answers are not byte-identical to plain llama.cpp's:** the long-prompt mode (below) stores the KV cache as q8_0 (against f16 KV; the meaning is kept in our checks, the quality figures are listed in [EXACTNESS](docs/EXACTNESS.md) and are not repeated here; nothing is measured at 128k or above); `--exact` keeps f16 and the stock-identical answers (also at 128k+), and MTP and the `--fast` modes can change answers too (`--exact` turns MTP off as well). **The default is not token-identical to plain llama.cpp:** it switches on the modes that looked like a win in our runs (table in [CONFIGURATION](docs/CONFIGURATION.md): long-prompt mode from 32k, q8_0 KV from 128k, MTP drafting in agent mode); modes without a measured win (expert skipping, `--approx` tiers) stay off. **`--exact` turns every one of them off and gives plain llama.cpp's output with the same sampling settings** (since v0.3.9 it also turns off the model card's sampling defaults and the chat-template fix, so the answer is what plain llama.cpp gives; `--no-sampling-defaults` turns the launcher's own sampling defaults off on its own: values you pass after `--`, values the model file carries and values a request sends still win). `--fast` and `--approx` still work but are deprecated. A different GPU layer split, or experts computed on the GPU instead of the CPU, can change a token (slightly different floating-point sums) while the meaning stays. Chat in your browser, or connect coding tools (OpenCode, Hermes, Claude Code). Windows 11 and Linux. Free (MIT), no account.

MoE models (Qwen3, GLM, gpt-oss, DeepSeek) use only a few "experts" per token. When the file is bigger than your RAM, plain llama.cpp keeps reading the whole file from the disk and slows to a crawl; moe-cache keeps the experts the model actually uses in RAM and reads only the rest. If your model already fits in RAM and all its experts fit on the graphics card, the launcher runs stock llama.cpp — stock is as good or better there. If it fits in RAM but some layers keep their experts on the CPU, it runs through the plugin by default; there, on the 8 GB laptop, it was not faster than stock for models that fit in RAM (measured on the release code; the result is labelled laptop, and this text quotes no figures for it; an earlier claim that it was faster was retracted when it did not reproduce); `--stock` turns the plugin off, and since v0.3.9 a MoE model that only nearly fits the graphics card also runs stock ([when to use something else](docs/FAQ.md#when-to-use-something-else)).

| You need | |
|---|---|
| Memory | 32 GB RAM for the big models (16 GB works for smaller ones such as Qwen3.6-35B); a graphics card is optional but recommended |
| Disk | an SSD (NVMe is best) with the model size plus about 10 GB free |
| Software | Python 3.8 or newer (3.11 or newer for `connect codex --write`, which checks the merged TOML file). llama.cpp is included in the Windows kit and the Linux `-ggml0.26` download (v0.3.3 and newer) |
| Time | about 10 minutes of setup, plus the model download |

**Privacy:** no telemetry, no account; the plugin opens no network connections. The few times the launcher or installer uses the internet (downloads you ask for, the update check) are listed in [docs/SECURITY.md](docs/SECURITY.md#privacy-what-moe-cache-sends-and-stores).

> **This release carries no speed figures.** The bundled llama.cpp build moved to b12651 (ggml 1.0) and the speed measurements are redone on it; Speed numbers for this release follow in v0.3.9.1. The plugin is CI-tested on b12651 (Windows, Linux identity) only.

**What's new in v0.3.9.** The kits bundle our build of **llama.cpp b12651 (ggml 1.0)** (commit f5e6915c2; every patch built in and off; the identity check compares it with an unpatched build of the same commit that CI builds itself; the Windows kit ships `ggml-moe-cache-api4.dll` next to the api2 and api3 ones; the older b12207 official build stays available with `LLAMA_SERVER`). **The default is still not token-identical to plain llama.cpp's**, and **`--exact` gives plain llama.cpp's answers** — since v0.3.9 it also turns off the model card's sampling defaults and the chat-template fix; on the control page choose **Exact**. **Inkling 2 Mini / Small (looped architecture) can be launched**: the launcher recognises ggml 1.0 and the `inkling2_loop` architecture, places experts by how often they are read (a loop layer is read several times per token), uses a small `-ub 128` micro-batch, and your own `--n-cpu-moe`, `-ot`, `-ub` and `--exact` win; no speed figure is claimed for this model in this release.

For a looped model (Inkling 2) the launcher adds `--no-op-offload` by default: the staging copy of the shared loop-layer experts is 4 GB at Q2 and does not fit an 8 GB card at IQ4_XS. Set `MOE_CACHE_INKLING_OP_OFFLOAD` to any value (for example `=1`) to keep op-offload on.
<!-- LOOP-SPEED-IF-MERGED -->
<!-- only if the loop-speed branch merges: add its sentence here; no speed figure is claimed in this draft -->
<!-- /LOOP-SPEED-IF-MERGED -->

**Dense KV cache default stops at q8_0** (the K q8_0 / V q4_0 and q4_0 / q4_0 steps need `--kv-4bit`; `--exact` keeps f16; the spill message says what `--kv-4bit` costs and gains). **A MoE model that nearly fits the graphics card runs stock llama.cpp** (when the plugin plan would keep more than 1.29 times stock's expert layers on the CPU: a launcher threshold, not a speed claim; stock is as good or better there; `--always-plugin` / `--long-prompt` keep the old way). **Auto-tune changes**: it tunes models that fit the memory the server may use (the "more than 40 % of RAM" skip is gone) and stops after the first load when that load takes over 60 s; a pick must win both runs of a confirm round; settings measured as a loss are never tried again in their scope; MTP and two-fewer-threads exist as tuner candidates only behind `MOE_CACHE_TUNE_MTP=1` / `MOE_CACHE_TUNE_THREADS=1`. **A plugin build for another ggml API is not loaded** into a server it does not fit (another installed llama-server is looked for, else the stock path). **Op-offload batch 512 is set only for a MoE stock run with expert layers on the CPU** (measured on one laptop only; no figure quoted). **A dense model that spills at 128k context or more runs `-np 1 -ub 256` by default on Windows** (no speed figure in this release; off below 128k and on other systems; `MOE_CACHE_DENSE_SPILL_RULE=0` turns it off). **The cache follows the system's available RAM while the server runs** (Windows by default; Linux only with `MOE_CACHE_LINUX_VIEW=0`), and **the plugin's RAM is sized from the RAM available at start, on Windows and Linux** (on Linux this makes the default expert cache smaller — see Known issues). **`--gpu-cache auto` is now the default on Linux with a CUDA llama-server** for a MoE model that fits in RAM and runs stock with experts on the CPU side (one NVIDIA laptop, one model; no speed claim), with one retry without the cache when a chosen cache does not fit. **The launch plan is written as a data file** (`<cache>/plans/port-PORT.json`, `--plan-file`, `--plan-json`) with each setting's value, source and reason, and **one GPU placement is planned from what llama-server really used** (stored per GPU, build and model). **`moe-cache --version` names the commit** (a kit carries it in a `BUILD_COMMIT` file). **An install in its own `-Root` / `--prefix` keeps its own settings, API key, chats and reports** (new installs only). **Stored auto-tune picks and GPU load records survive a copy** (keyed by file content, not modification times; after this update every model is measured once more). `MOE_CACHE_ROUTE=subst` is removed (routing stays exact); `--no-sched-tuning` is removed and refused as an unknown option; `MOE_CACHE_MAX_RAM_GB` is read the same way by the launcher and the plugin (a value with a unit is a typo on both sides). Also new: first-run clarity on Windows, the opt-in `MOE_CACHE_MMAP_SPIKE=1` load mode for dense models, reproducible release archives, stricter installer inputs, QuickEdit switched off while a server runs, new bench and plan-checking tools under `tools/` (no effect on a normal start), and small control-page and bench-driver fixes. Details: [CHANGELOG](CHANGELOG.md), [release notes](docs/releases/RELEASE_NOTES_v0.3.9.md).

## Questions people ask

| Question | Short answer |
|---|---|
| [How do I run a MoE model that is bigger than my RAM?](docs/FAQ.md#how-do-i-run-a-moe-model-that-is-bigger-than-my-ram) | keep the experts you use in RAM and read the rest from the SSD; that is what moe-cache does |
| [Why is llama.cpp slow when the model does not fit in RAM?](docs/FAQ.md#why-is-llamacpp-slow-when-the-model-does-not-fit-in-ram) | the whole file is read again and again from the disk (mmap thrashing) |
| [Does it work with `--n-cpu-moe` and `-ot`?](docs/FAQ.md#does-it-work-with-llamacpps-own-offload-options) | yes, they work together |
| [Which model fits a 32 GB RAM + 16 GB GPU PC?](docs/MODELS.md) | see [Which model?](#which-model) |
| [Does it run on Windows with an AMD graphics card (Vulkan)?](docs/WINDOWS.md) | yes, the Windows kit uses Vulkan; the main test PC has an RX 9070 XT |
| [Can I use it with a coding agent?](docs/AGENTS.md) | OpenCode, Claude Code, Hermes, Codex CLI, Aider and more |
| [Is it free and private?](docs/SECURITY.md#privacy-what-moe-cache-sends-and-stores) | MIT licence, no account, no telemetry |

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

If something is off, `python bin\moe_cache.py check D:\models\model.gguf` says what to fix. The first start of a model may spend up to 180 s auto-tuning (auto-tune now covers models that fit the memory the server may use, not only small ones; `--retune` measures again, `--exact` or your own flags skip it); on Windows the launcher may refuse to start when free memory commit is too low: close big programs or raise the page file as it says ([TROUBLESHOOTING](docs/TROUBLESHOOTING.md), `--no-commit-check` skips the check). Every install option: [docs/INSTALL.md](docs/INSTALL.md#windows-quick-start-experimental); Windows details: [docs/WINDOWS.md](docs/WINDOWS.md).

**Using the PC while the model runs?** Tick **Leave room for me** on the control page (or `--leave-room`): the model leaves more RAM, GPU memory and CPU to your own programs and gets a little slower. A CPU with its own graphics: plug the monitor into the motherboard. The GPU also draws your desktop, and the screen flickers or the browser stutters while a long prompt runs? Reserve GPU memory for it with `--gpu-margin GiB` (for example `--gpu-margin 3`): the launcher plans fewer expert layers on the GPU and leaves that much free. Without the option it already keeps more free when other programs hold GPU memory at start.

**Long prompts (agents, documents): long-prompt mode, on by default.** From 32k context on a graphics card with room, the launcher reads prompts in chunks of 4096 tokens (`-ub 4096`) and stores the KV cache as q8_0. Its speed gain against the previous default (`-ub 2048`, f16 KV) is not quoted in this release. Speed numbers for this release follow in v0.3.9.1. **The answers are not byte-identical to plain llama.cpp's with it** (q8_0 KV cache, against f16; quality figures: [EXACTNESS](docs/EXACTNESS.md)). `--no-long-prompt` (or untick **Long-prompt mode** on the control page) goes back to `-ub 2048` and f16 KV; `--exact` gives plain llama.cpp's answers. It only switches on when your card has room; below 32k context it stays off (`--long-prompt` asks for it).

### Linux

| Step | What to do |
|---|---|
| 1. Download | From [Releases](https://github.com/IbrahimSiddiqui007/moe-cache/releases): `moe-cache-vX-ggml0.26-linux-x86_64.tar.gz` (our llama.cpp b12651 build included, Vulkan + CPU), or `moe-cache-vX-linux-x86_64.tar.gz` for an older llama.cpp of your own (ggml 0.24 / 0.25). `tar xzf` it and `cd` into the folder |
| 2. A model | `python3 bin/moe_cache.py pick` |
| 3. Check | `bin/moe-cache-check /path/to/model.gguf --run` ("cache runs: yes" means it works on your machine) |
| 4. Run | `bin/moe-cache-server /path/to/model.gguf --open` (chat page and an OpenAI-compatible API at http://localhost:8080), or `bin/moe-cache-server --ui` for the control page |

Which file on the Releases page:

| Your PC | File | Contains |
|---|---|---|
| Windows | `moe-cache-vX-windows-x64.zip` | the kit: plugin, launcher, our llama.cpp b12651 build (Vulkan) |
| Linux, no llama.cpp of your own, or ggml 0.26 or newer | `moe-cache-vX-ggml0.26-linux-x86_64.tar.gz` | plugin, launcher, our llama.cpp b12651 build (Vulkan + CPU) |
| Linux, your own older llama.cpp (ggml 0.24 / 0.25) | `moe-cache-vX-linux-x86_64.tar.gz` | plugin and launcher only |

Your own llama.cpp (CUDA, ROCm), `install.sh`, building from source, the speed options (`--fast`, `--hybrid`) and proving the output yourself: [docs/INSTALL.md](docs/INSTALL.md).

### Coding agents

`python3 bin/moe_cache.py connect NAME --write` writes the config of Hermes Agent, OpenCode, Claude Code, Codex CLI, Pi, OpenClaw, Aider or Qwen Code (without `--write` it only prints; the first backup of your config is kept as `.bak`, `--undo` puts it back, `--check` proves the setup with one prompt and one tool call). Start the server in agent mode first, or let on-demand start it when an agent asks:

```bash
python3 bin/moe_cache.py on-demand /path/to/model.gguf --install-autostart   # loads the model on the first request, unloads it when idle
# or by hand: bin/moe-cache-server /path/to/model.gguf --agent                # context 65536, tool calls, API key, warm sessions
```

Step by step for each agent, the desktop apps and the context limit to set: [docs/AGENTS.md](docs/AGENTS.md).

### Which model?

`pick` decides from your RAM, graphics card, free disk and use (agents, chat, long documents). On a 32 GB RAM + 16 GB GPU PC: Qwen3.8-Flash-Next IQ3_XXS (76 GB, the default again since v0.3.8) for general and agent use: the Q2_0 file (66 GB) is smaller, at a little less exactness (neither file was measured against full precision), Nemotron 3.5 Bolt 74B (22 GB, fits in RAM) for 128k contexts. If a model fits in your RAM (and its experts on the graphics card), stock llama.cpp is as good or better — the launcher runs stock for it, and since v0.3.9 also for a MoE model that only nearly fits the GPU (`--always-plugin` keeps the plugin). `python3 bin/moe_cache.py models` lists every catalogue model with fits / tight / no for this PC, whether it is already downloaded and its measured speed (or "not measured"). All files and the rules: [docs/MODELS.md](docs/MODELS.md).

## Results

**No stock-against-moe-cache table in v0.3.9.** No speed figure is shown and no headline speed claim is made for the default model, Q2_0 or Qwen3.6-35B; Speed numbers for this release follow in v0.3.9.1. Older records on the earlier head kit (v0.3.7.1 code, b12064) are kept in [docs/results-archive/](docs/results-archive/RESULTS.md); they are not v0.3.9 numbers. Test PC of earlier records: Windows desktop, Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM at 6000 MT/s (EXPO), Vulkan.

**The long-prompt mode (the default)** has no table in this release. Speed numbers for this release follow in v0.3.9.1. The q8_0 KV cache is not byte-identical to stock (see the top of this page); `--no-long-prompt` and `--exact` turn the mode off.

### Where it does not help

- **Models that fit in RAM.** Stock llama.cpp is as good or better, and the launcher runs stock for them: if all experts fit on the card it runs plain llama.cpp, and since v0.3.9 also when the model only nearly fits the GPU (the plugin plan would keep more than 1.29 times stock's expert layers on the CPU; `--always-plugin` or `--long-prompt` keeps the plugin). If the model fits in RAM but some expert layers stay on the CPU, the launcher runs the plugin by default (`--stock` or `MOE_CACHE_STOCK_WHEN_FITS=1` opts out). Evidence, labelled laptop and quoted here without figures: on one laptop with an 8 GB graphics card the plugin was not faster than stock for a model that fits in RAM (paired 8-turn chat on the release code); an earlier claim that it was faster was retracted when it did not reproduce; no comparison on the 16 GB PC. Without a GPU the launcher stays on stock llama.cpp (checked with a dry run on the laptop with the GPU hidden).
- **Short answers at shallow depth.** The gain in a conversation comes from reading prompts and from keeping the hot experts in RAM; a short chat on a model that already fits in RAM does not gain.
- **CPU kernels.** Any gain comes from which experts are in RAM and how they are read, not from faster math: llama.cpp's own CPU kernels run on both sides.
- **Cold start.** From a cold file cache the plugin reports ready a few seconds later than stock while it loads its hot experts; the first answer wins it back.
- **Small samples.** The measurements behind this project are one or two runs per arm on one PC; speed depends on RAM speed, SSD and GPU.

## Known issues in v0.3.9 (honest list)

| Issue | Detail |
|---|---|
| New in v0.3.9, little measured | the dense KV default stops at q8_0 and the 4-bit steps need `--kv-4bit` (quality checked on one 8 GB laptop, one run each, a forecast, no figure quoted here); a MoE model that nearly fits the GPU runs stock (built, not measured; checked as a plan on a synthetic 35B twin at 64k with `--vram 15`); the Windows dense-spill rule runs `-np 1 -ub 256` at 128k+ (no speed figure in this release; off below 128k and on other systems, not measured there); auto-tune changes (laptop checks only, no PC result); `--gpu-cache auto` on Linux CUDA (one NVIDIA laptop, one model; no speed claim) and its retry (laptop, one machine); the cache following available RAM (partly measured on the laptop, n=1: the `mem:` line and the grow path seen with private memory on Flash-Next, the shrink path not triggered; Windows build only syntax-checked); the plugin's RAM sized from available RAM (built, not measured; not run on the PC yet); the opt-in `MOE_CACHE_MMAP_SPIKE=1` load mode (built, not measured); the GPU placement planned from what llama-server really used (built, not measured on a real load yet); the own-root / own-prefix cache (built and unit-tested; not run on a real Windows PC); the bundled b12651 build has no speed figure in this release |
| Not byte-identical to stock by default (32k+ context) | the long-prompt mode, on by default from 32k context when the card has room, stores the KV cache as q8_0 (and so does the default at `--ctx 131072` and above, unless you pass `--no-long-prompt`): against f16 KV the meaning is kept in our checks (figures: [EXACTNESS](docs/EXACTNESS.md), not repeated here); nothing measured at 128k or above; the bytes differ from stock. `--exact` keeps f16 and stock-identical answers (and since v0.3.9 also turns off the model card's sampling defaults and the chat-template fix); `--no-long-prompt` keeps f16 at any context, also at 128k and above |
| Plugin for models that fit in RAM (default since v0.3.8) | a MoE model that fits in RAM but keeps expert layers on the CPU runs through the plugin (`--stock` or `MOE_CACHE_STOCK_WHEN_FITS=1` opts out). Measured on one laptop with an 8 GB card (labelled laptop; no figures quoted in this text): not faster than stock for a model that fits in RAM; an earlier claim of a decode gain was retracted when it did not reproduce; no PC comparison. Since v0.3.9 a model that nearly fits the GPU runs stock too. Without a GPU the launcher stays on stock llama.cpp (checked with a dry run on the laptop with the GPU hidden). |
| Dense models: `--fit`, FFN in RAM (`-ot`), Intel thread rule | measured on one laptop only (i7-13620H + 8 GB card), one run each; the thread rule applies only when part of the dense model runs on the CPU, and only on Linux hybrid Intel CPUs. `--no-auto-fit`, `--no-ffn-ram` go back to llama.cpp's own split. The KV ladder now stops at q8_0 unless you pass `--kv-4bit`. Other CPUs and cards are untested |
| Long-prompt default sets two more things, and its memory estimate rests on one model | from 32k context it applies `-ub 4096`, the q8_0 KV cache and `MOE_CACHE_GPU_MIN_TOKENS=64`; this is deliberate and part of what the prompt figure above measured (one run per arm on our test PC, forecast). `--no-long-prompt` removes all three. The `-ub 4096` memory estimate rests on one model: the launcher's estimate of the compute buffer at `-ub 4096` comes from Flash-Next IQ3_XXS at 64k. Another model may need the one automatic retry (the launcher falls back by itself) or `--no-long-prompt` |
| New in v0.3.9: browser apps on another local port | with llama.cpp b12651, `--cors-origins localhost` (what the launcher sets for a plain server) means the same origin only: a browser chat app on another local port (e.g. `http://localhost:3000`) is answered 403 until you pass `--cors-origins http://localhost:3000` after `--`. The control page, agents (Hermes, OpenCode, Claude) and `curl` send no Origin and are unaffected. Checked in CI, not by a real browser app |
| New in v0.3.9: Inkling 2 Mini / Small (looped architecture) | can be launched (experts placed by how often they are read, a small `-ub 128` micro-batch; your own `--n-cpu-moe`, `-ot`, `-ub` and `--exact` win; an older llama-server without the support answers `!!!!` and the launcher says so). The correct answer was checked on one 8 GB laptop (CUDA, one short prompt) with an earlier llama.cpp build that carries the support; it has not been checked on the bundled b12651 kit as a release result, and no speed figure is claimed for this model in this release |
| New in v0.3.9: the Linux plugin RAM is sized from available RAM | every Linux plugin start now gets an expert cache about 4 GiB (12.5 % of RAM on big machines) smaller than before, so possibly slower decode on a machine with free RAM; older Linux speed records used the old size. `--ram`, `MOE_CACHE_RAM_GIB`, `--stock` and a Linux cgroup limit are unchanged |
| New in v0.3.9: the cache follows available RAM while the server runs | Windows by default; Linux only with `MOE_CACHE_LINUX_VIEW=0` (on the Linux default file-view cache it is off, and the start-up lines say so). It can only give back the cache's own memory, not the mapped model file or other programs' memory; the 1 GiB, 256 MiB, 250 ms and 500 ms constants are assumptions; the tune-time integration is not unit-tested (only the decision functions are) |
| New in v0.3.9: after this update every model is auto-tuned once more | stored auto-tune picks and GPU load records are now keyed by file content instead of modification times, so a re-unpacked kit or a copied model no longer re-measures by accident — but the first start after this update measures once, up to 180 s |
| New in v0.3.9: `MOE_CACHE_ROUTE=subst` removed | it replaced a picked expert that was not in RAM with a different resident expert's weights and never had a quality measurement; setting it now prints one line and routing stays exact. `MOE_CACHE_ROUTE=skip` (`--fast`, off by default) is unchanged |
| New in v0.3.9: `--no-sched-tuning` removed | the option has done nothing since v0.3.8; a command line that still has it is refused as an unknown option, so delete it from old start scripts |
| New in v0.3.9: `MOE_CACHE_MAX_RAM_GB` is read strictly | the whole value must be a plain decimal number; `20GB`, hex, `inf`, `nan` or full-width digits are a typo on both launcher and plugin side (a warning, and the default cap is used). Check old start scripts that set it with a unit |
| New in v0.3.9: QuickEdit is switched off while a server runs | a stray click in the launcher's console window used to stall the server pipe for seconds. Mouse selection of log lines in conhost stops working meanwhile; `MOE_CACHE_QUICKEDIT=1` leaves it alone (Windows Terminal is not affected) |
| New in v0.3.9: stricter installer inputs | `MOE_CACHE_BASE_URL` may be https, a `file://` folder or `http://` on this machine only; `MOE_CACHE_REF` must look like a release tag; an old `gh` without the `attestation` command is reported as "gh too old" (skipped, not failed); `update --apply --tag X` with a running non-release version needs `--allow-downgrade` |
| Flash-Next at 64k on a 32 GB PC that is already busy | the model's file pages can take the available RAM down to almost nothing and the PC gets slow: our test PC, one run, 2026-11-13 (32 GB, about 18 GB already in use by other programs, Flash-Next at 64k context). The launcher does not check this before the load and only says so in a note once it has started. Close other programs before the start, or use a smaller context (`--ctx 32768`) |
| Windows zip is unsigned | Windows shows "unknown publisher" / SmartScreen: Unblock the zip, More info > Run anyway (see Get started). Check the SHA256 or `gh attestation verify` |
| Windows tested on one desktop | Ryzen 9 7950X + RX 9070 XT 16 GB + 32 GB, Windows 11, Vulkan, plus a GitHub Windows runner (CPU). NVIDIA, Intel GPUs and Windows 10 are untested; HIP loads but is slower than Vulkan |
| Models measured | Qwen3.8-Flash-Next (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, Coder IQ1_M), Qwen3.5-122B IQ2_XXS, GLM-5.3-Flash, Qwen3.6-35B, Nemotron 3.5 Bolt 74B (agent speed; plugin forced slower than stock in one test), Granite 1B (CI), and Inkling 2 Mini (correct answer on one 8 GB laptop with an earlier build; no release result on the bundled kit). Other MoE models should work but are not measured. Long context (64k-256k) is measured on Flash-Next and a few more, single runs |
| Linux | Tested on Intel + NVIDIA, CPU-only, Vulkan, and a user's RX 9070 XT. `install.sh` has been tried on Arch (`--no-apt`) and, in the previous release, from a clean clone on that machine; a fresh Ubuntu/Debian run with apt is not tested. macOS: no. The Linux plugin RAM default changed in v0.3.9 (see above) |
| `--ram` is a sizing hint | On Windows the memory limit is set only for the plugin, only with `--ram`, and only with `MOE_CACHE_WIN_VIEW=0`; the default file-view cache, the on-demand listener and stock llama-server get none (a job limit counts GPU memory and broke Vulkan). The server can use more RAM than `--ram` says. On Linux the launcher sets no limit either; only a cgroup you set yourself is a hard limit. No guard is being built |
| A failed model read still aborts | a model file that is cut short or cannot be read stops the server. On Linux it now prints a clear message; on Windows a guard for the in-page read error is in, tested with a simulated error in CI but not with a real cut-short file on a PC |
| Installers need `gh` or an explicit yes | `install.sh`, `install.ps1` and `update --apply` check the build attestation with the GitHub CLI (`gh`; its own `attestation verify` decides, not a login check). Without a working `gh` they stop and say what was and was not checked; `--yes-unverified` (or `MOE_CACHE_ALLOW_UNVERIFIED=1`) goes on with only the SHA256 check, which catches a damaged download but not a tampered release (a missing check is no longer skipped silently) |
| GPU errors | with Vulkan a GPU error inside an expert block is caught and the block runs on the CPU (`MOE_CACHE_GPU_CATCH`, not yet seen with a real failure); on NVIDIA / CUDA a GPU error still ends the server |
| Context length and RAM speed of the speed numbers | earlier speed records were real 65k conversations on Flash-Next IQ3_XXS and Q2_0 (and Qwen3.6-35B), one PC, RAM at 6000 MT/s; this release quotes none of them; nothing is measured beyond 65k on this kit |
| Not tested for this release | `winget install` and `scoop install` (manifests exist but are not submitted); Hermes Desktop and OpenCode Desktop on Windows (click-through still open); a full model run from the final zip; the v0.3.9 launcher changes listed as "built, not measured" in the first row |
| Decode after long prompts | on the v0.3.5 kit decode after a 34k-64k prompt was slower than on v0.3.4 (the bigger prompt chunk leaves 2 fewer expert layers on the GPU); giving that memory back after the prompt (patch 0070) was tried on the PC on 2026-11-05 and lost, so it is archived (`archive/README.md`); a smaller prompt chunk keeps decode faster at the price of a slower prompt |
| Speed varies between nights | the same test repeated on another night can differ on this PC (RAM speed, background work, Windows state; a restart before a chain removed a large drift in October); compare only runs from the same session |
| Speed numbers | One PC, cold runs, mostly single series; yours depends on RAM, SSD and GPU. No Linux table is shown in this release |

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
| [Current results](docs/RESULTS_CURRENT.md) and the [benchmark card page](docs/benchmark_card.html) | the whole card, per-depth tables, method, commands |
| [Older results](docs/results-archive/RESULTS.md) (Windows desktop), [older laptop results](docs/results-archive/RESULTS_OLD_LAPTOP.md) | every earlier measurement, newest first |
| [Release notes](docs/releases/RELEASE_NOTES_v0.3.9.md), [changelog](CHANGELOG.md) | what changed in each version |
| [Hardware and limits](docs/PLATFORMS.md) | what is tested where, known limitations |
| [Configuration](docs/CONFIGURATION.md) | environment variables |
| [FAQ](docs/FAQ.md) | questions and answers |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | wrong-looking answers, slow runs, start problems |
| [Internal notes](docs/internal/) | design notes, coding standards, release checklist: for people building or testing moe-cache |
| [Compatibility](docs/COMPATIBILITY.md), [Roadmap](docs/ROADMAP.md), [Security and privacy](docs/SECURITY.md) | |

## Support

moe-cache is built and tested in my spare time on my own machines. If it saves you a RAM upgrade, you can support it on [Ko-fi](https://ko-fi.com/ibrumm).

## Acknowledgements

- [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml (MIT, Copyright The ggml authors): moe-cache is a plugin for their backend interface, and it runs their CPU kernels unchanged.
<!-- AT SWAP-IN: copy the "Credit for ideas" bullet from the current README.md here (the pre-commit allows that line in README.md only, so the draft file cannot carry it) -->

## License

- moe-cache is MIT-licensed, see [LICENSE](LICENSE).
- `third_party/ggml/` contains interface headers copied from ggml / llama.cpp (MIT, Copyright The ggml authors). Their license text is kept in `third_party/ggml/LICENSE`.
- moe-cache does not include any model weights. Models you run with it keep their own licenses.
- Product and model names (Qwen, gpt-oss, DeepSeek, Granite, Hunyuan, KAT-Coder and others) belong to their respective holders and are used only to say what was tested.
