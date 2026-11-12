<h1 align="center">moe-cache: run Mixture-of-Experts LLMs bigger than your RAM with llama.cpp</h1>

<p align="center"><b>Run big MoE models (Qwen3, GLM, gpt-oss and more) that do not fit in your RAM on a normal PC with llama.cpp: the experts you use stay in RAM, the rest is read from your SSD.</b><br>
moe-cache is a plugin that runs inside <code>llama-server</code>; it also works on the unmodified official one. No fork: the Windows kit and the Linux ggml 0.26 tarball bundle llama.cpp b12207 built from the official source plus a small patch set (<code>patches/llama.cpp</code>). Every patch is built in but off until you set its own switch, so by default the build behaves as the official one (checked at temperature 0 in CI). To use the official build instead, point <code>LLAMA_SERVER</code> at it.</p>

<p align="center">
<img alt="status" src="https://img.shields.io/badge/status-v0.3.7.1-blue">
<img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
<img alt="platform" src="https://img.shields.io/badge/platform-Linux%20%2B%20Windows%20(experimental)-lightgrey">
<img alt="llama.cpp" src="https://img.shields.io/badge/llama.cpp-tested%20b12207%20(Windows)%2C%2023b0202a1%20%2F%207fe450e19%20(Linux)-green">
<a href="https://ko-fi.com/ibrumm"><img alt="Support on Ko-fi" src="https://img.shields.io/badge/Ko--fi-support%20this%20project-ff5e5b?logo=ko-fi&logoColor=white"></a>
</p>

<p align="center">Speeds, test settings and every caveat are in the <a href="#results">benchmark card below</a> (Windows desktop, 32 GB RAM + 16 GB graphics card, stock llama.cpp against moe-cache). Full history: <a href="docs/results-archive/RESULTS.md">docs/results-archive/RESULTS.md</a>; the older Linux laptop results: <a href="docs/results-archive/RESULTS_OLD_LAPTOP.md">docs/results-archive/RESULTS_OLD_LAPTOP.md</a>.</p>

## What it is

**Who it is for:** you have a PC with 32 GB RAM (16 GB for smaller models), perhaps a 16 GB graphics card (AMD, NVIDIA or Intel, Windows 11 or Linux), a fast SSD, and you want to run a large local Mixture-of-Experts model (60-120 GB file) for chat, long documents or a coding agent, and plain llama.cpp is too slow because the model does not fit in RAM. **Who it is not for:** if the whole model fits in RAM, or in your graphics card, plain llama.cpp is fine ([when to use something else](docs/FAQ.md#when-to-use-something-else)).

**moe-cache lets a normal PC run big Mixture-of-Experts AI models** (files of 60-120 GB) that are bigger than its RAM, with the plugin itself changing no number: with the same settings its answers are the same as plain llama.cpp's in the checks we ran (greedy decoding, the same GPU/CPU layer split, same llama.cpp build; the exact list is in [EXACTNESS](docs/EXACTNESS.md)). **By default, from 32k context on a graphics card with room, answers are not byte-identical to plain llama.cpp's:** the long-prompt mode (below) stores the KV cache as q8_0 (against f16: KL 0.014-0.015, same top token in 94-95 % of positions, meaning kept in 10 of 10 checks at 55k); `--exact` keeps f16 and the stock-identical answers (also at 128k+), and MTP and the `--fast` modes can change answers too. **The default is faster and not token-identical to plain llama.cpp:** it switches on exactly the modes we measured as a win (table in [CONFIGURATION](docs/CONFIGURATION.md): long-prompt mode from 32k, q8_0 KV from 128k, MTP drafting in agent mode); modes without a measured win (expert skipping, `--approx` tiers) stay off. **`--exact` turns every one of them off and gives plain llama.cpp's output.** `--fast` and `--approx` still work but are deprecated. A different GPU layer split, or experts computed on the GPU instead of the CPU, can change a token (slightly different floating-point sums) while the meaning stays. Chat in your browser, or connect coding tools (OpenCode, Hermes, Claude Code). Windows 11 and Linux. Free (MIT), no account.

MoE models (Qwen3, GLM, gpt-oss, DeepSeek) use only a few "experts" per token. When the file is bigger than your RAM, plain llama.cpp keeps reading the whole file from the disk and slows to a crawl; moe-cache keeps the experts the model actually uses in RAM and reads only the rest. If your model already fits in RAM and all its experts fit on the graphics card, the launcher runs stock llama.cpp. If it fits in RAM but some layers keep their experts on the CPU, it runs through the plugin by default; there moe-cache is about the same as plain llama.cpp on the 16 GB PC (1.06x prompt, 1.07x decode in our one test <!-- claim:v038-fits-qwen36 -->) and faster at decode on the 8 GB laptop; `--stock` turns that off ([when to use something else](docs/FAQ.md#when-to-use-something-else)).

| You need | |
|---|---|
| Memory | 32 GB RAM for the big models (16 GB works for smaller ones such as Qwen3.6-35B); a graphics card is optional but recommended |
| Disk | an SSD (NVMe is best) with the model size plus about 10 GB free |
| Software | Python 3.8 or newer (3.11 or newer for `connect codex --write`, which checks the merged TOML file). llama.cpp is included in the Windows kit and the Linux `-ggml0.26` download (v0.3.3 and newer) |
| Time | about 10 minutes of setup, plus the model download |

**Privacy:** no telemetry, no account; the plugin opens no network connections. The few times the launcher or installer uses the internet (downloads you ask for, the update check) are listed in [docs/SECURITY.md](docs/SECURITY.md#privacy-what-moe-cache-sends-and-stores).

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

If something is off, `python bin\moe_cache.py check D:\models\model.gguf` says what to fix. Every install option: [docs/INSTALL.md](docs/INSTALL.md#windows-quick-start-experimental); Windows details: [docs/WINDOWS.md](docs/WINDOWS.md).

**Using the PC while the model runs?** Tick **Leave room for me** on the control page (or `--leave-room`): the model leaves more RAM, GPU memory and CPU to your own programs and gets a little slower. A CPU with its own graphics: plug the monitor into the motherboard. The GPU also draws your desktop, and the screen flickers or the browser stutters while a long prompt runs? Reserve GPU memory for it with `--gpu-margin GiB` (for example `--gpu-margin 3`): the launcher plans fewer expert layers on the GPU and leaves that much free. Without the option it already keeps more free when other programs hold GPU memory at start.

**Long prompts (agents, documents): long-prompt mode, on by default.** From 32k context on a graphics card with room, the launcher reads prompts in chunks of 4096 tokens (`-ub 4096`) and stores the KV cache as q8_0. Flash-Next IQ3_XXS in a 65k conversation, against the previous default (`-ub 2048`, f16 KV): new prompt tokens 187.4 → 331.8 tok/s (+77.0 %), decode 19.01 → 19.35 tok/s (+1.8 %) <!-- claim:v038-longprompt-default --> (server logs: 42 expert layers on the CPU before, 45 now). **The answers are not byte-identical to plain llama.cpp's with it** (q8_0 KV cache: KL 0.014-0.015, same top token in 94-95 % of positions, meaning kept in 10 of 10 checks at 55k). `--no-long-prompt` (or untick **Long-prompt mode** on the control page) goes back to `-ub 2048` and f16 KV; `--exact` gives plain llama.cpp's answers. It only switches on when your card has room; below 32k context it stays off (`--long-prompt` asks for it).

### Linux

| Step | What to do |
|---|---|
| 1. Download | From [Releases](https://github.com/IbrahimSiddiqui007/moe-cache/releases): `moe-cache-vX-ggml0.26-linux-x86_64.tar.gz` (llama.cpp b12207 included, Vulkan + CPU), or `moe-cache-vX-linux-x86_64.tar.gz` for an older llama.cpp of your own (ggml 0.24 / 0.25). `tar xzf` it and `cd` into the folder |
| 2. A model | `python3 bin/moe_cache.py pick` |
| 3. Check | `bin/moe-cache-check /path/to/model.gguf --run` ("cache runs: yes" means it works on your machine) |
| 4. Run | `bin/moe-cache-server /path/to/model.gguf --open` (chat page and an OpenAI-compatible API at http://localhost:8080), or `bin/moe-cache-server --ui` for the control page |

Which file on the Releases page:

| Your PC | File | Contains |
|---|---|---|
| Windows | `moe-cache-vX-windows-x64.zip` | the kit: plugin, launcher, llama.cpp b12207 (Vulkan) |
| Linux, no llama.cpp of your own, or ggml 0.26 or newer | `moe-cache-vX-ggml0.26-linux-x86_64.tar.gz` | plugin, launcher, llama.cpp b12207 (Vulkan + CPU) |
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

`pick` decides from your RAM, graphics card, free disk and use (agents, chat, long documents). On a 32 GB RAM + 16 GB GPU PC: Qwen3.8-Flash-Next IQ3_XXS (76 GB, the default again since v0.3.8) for general and agent use: in a real 65k conversation moe-cache reads new prompt tokens 3.5× as fast as stock (3.2-4.3× by depth) and decodes 1.34× as fast (1.25-1.63× by depth) <!-- claim:v038-conv-iq3xxs -->; the Q2_0 file (66 GB) is the faster option: in the same kind of conversation it reads prompts 1.9× as fast as stock (5.0× falling to 1.3× by depth) and decodes 1.18-1.26× as fast, equal to stock at 48k-64k depth (0.98×) <!-- claim:v038-conv-q2_0 -->, at a little less exactness (top-1 agreement with IQ3_S 73.3 % against 76.5 % for IQ3_XXS; measured 2026-11-05, record pending), Nemotron 3.5 Bolt 74B (22 GB, fits in RAM) for 128k contexts. `python3 bin/moe_cache.py models` lists every catalogue model with fits / tight / no for this PC, whether it is already downloaded and its measured speed (or "not measured"). All files and the rules: [docs/MODELS.md](docs/MODELS.md).

## Results

Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), a real 65k conversation, stock llama.cpp against moe-cache with the same command line. This is the one headline table; the whole benchmark card, the per-depth tables, the method, the commands and the memory-budget table are in [docs/RESULTS_CURRENT.md](docs/RESULTS_CURRENT.md), and every older measurement is in [docs/results-archive/](docs/results-archive/RESULTS.md). **Prompt reading is at stock speed or faster in every row below** (prompt ratios 1.04x to 5.00x, none under 1.0x) <!-- claim:v038-conv-iq3xxs --><!-- claim:v038-conv-q2_0 --><!-- claim:v038-fits-qwen36 -->; decode is at or above stock except one depth band of the Q2_0 row (0.98x).

| Model (file size) | Prompt tok/s by depth band, stock → moe-cache (0-16k / 16-32k / 32-48k / 48-64k) | Prompt ratio | Decode tok/s by depth band, stock → moe-cache | Decode ratio |
|---|---|---|---|---|
| **Flash-Next IQ3_XXS** (the default)<br><sub>75.8 GB, drive D:, two runs each, ABBA</sub> | 49.8→212.8 / 66.0→215.0 / 54.9→201.4 / 44.0→142.8 | 4.27x / 3.26x / 3.67x / 3.24x | 16.5→22.0 / 13.1→19.4 / 12.6→20.5 / 14.5→18.2 | 1.33x / 1.48x / 1.63x / 1.25x <!-- claim:v038-conv-iq3xxs --> |
| Flash-Next Q2_0 (the faster option)<br><sub>66.4 GB, drive D:, two runs each, ABBA</sub> | 89.0→445.0 / 281.3→600.0 / 280.8→495.6 / 202.2→259.3 | 5.00x / 2.13x / 1.77x / 1.28x | 25.0→29.5 / 24.7→30.0 / 23.8→30.1 / 26.0→25.4 | 1.18x / 1.22x / 1.26x / 0.98x <!-- claim:v038-conv-q2_0 --> |
| Qwen3.6-35B, fits in RAM (plugin forced on)<br><sub>UD-IQ3_XXS file, two runs each, ABBA</sub> | 1840.7→2381.1 / 2544.6→2688.2 / 2148.3→2239.1 / 1678.1→1747.2 | 1.29x / 1.06x / 1.04x / 1.04x | 91.6→97.0 / 86.5→94.2 / - / - | 1.06x / 1.09x / - / - <!-- claim:v038-fits-qwen36 --> |

Whole-run means <!-- claim:v038-conv-iq3xxs --><!-- claim:v038-conv-q2_0 -->: Flash-Next IQ3_XXS prompt 54.4 → 188.6 tok/s, decode 14.25 → 19.1 tok/s; Q2_0 prompt 227.6 → 434.8, decode 25.4 → 26.9. Each number is the mean of two cold runs in the order stock, moe-cache, moe-cache, stock (ABBA), whole conversation. **Decode of Q2_0 equals stock at 48k-64k depth (0.98x)** <!-- claim:v038-conv-q2_0 -->; the gain there is in the prompt. Qwen3.6-35B fits in RAM: about the same or slightly faster (1.06x prompt, 1.07x decode) <!-- claim:v038-fits-qwen36 -->; the deeper bands have too few long answers for a decode figure. Stock = the same llama-server command the launcher builds for the plugin, without the plugin. Both Flash-Next files are bigger than the free RAM once the GPU and Windows have taken their share.

**The long-prompt mode (the default) on top of this table**, IQ3_XXS, 65k, previous default against long-prompt (A C C A, one run each, server logs read for the real settings):

| Setting | new prompt tokens, tok/s | decode, tok/s | expert layers on the CPU |
|---|---|---|---|
| previous default (`-ub 2048`, f16 KV) | 187.4 <!-- claim:v038-longprompt-default --> | 19.00 <!-- claim:v038-longprompt-default --> | 42 |
| long-prompt mode (`-ub 4096`, q8_0 KV), now the default | 331.8 <!-- claim:v038-longprompt-default --> | 19.34 <!-- claim:v038-longprompt-default --> | 45 |
| change | +77.0 % <!-- claim:v038-longprompt-default --> | +1.8 % <!-- claim:v038-longprompt-default --> | |

The q8_0 KV cache is not byte-identical to stock (see the top of this page); `--no-long-prompt` and `--exact` turn the mode off.

### Where it does not help

- **Models that fit in RAM.** If all experts fit on the card, the launcher runs plain llama.cpp. If some expert layers stay on the CPU, it runs the plugin (new default in v0.3.8; `--stock` or `MOE_CACHE_STOCK_WHEN_FITS=1` opts out). Evidence: laptop with an 8 GB card, decode +10 to +29 %, prompt about 10 % lower; on the 16 GB PC, Qwen3.6-35B in a 65k conversation was about the same as stock: 1.06x prompt, 1.07x decode (one model, one PC, 2026-11-08) <!-- claim:v038-fits-qwen36 -->. Without a GPU the launcher stays on stock llama.cpp (checked with a dry run on the laptop with the GPU hidden).
- **Short answers at shallow depth.** The gain in a conversation comes from reading prompts and from keeping the hot experts in RAM; a short chat on a model that already fits in RAM does not gain.
- **CPU kernels.** The numbers here were measured with llama.cpp's own CPU kernels on both sides; the gain comes from which experts are in RAM and how they are read, not from faster math.
- **Cold start.** From a cold file cache the plugin reports ready a few seconds later than stock while it loads its hot experts; the first answer wins it back (RESULTS 61).
- **Small samples.** Two runs per arm (one per arm for the long-prompt comparison), one PC, one seeded script, one head kit. The numbers depend on RAM speed (these runs: 6000 MT/s).

## Known issues in v0.3.7.1 (honest list)

| Issue | Detail |
|---|---|
| Not byte-identical to stock by default (32k+ context) | the long-prompt mode, on by default from 32k context when the card has room, stores the KV cache as q8_0 (and so does the default at `--ctx 131072` and above, unless you pass `--no-long-prompt`): against f16 KL is 0.014-0.015 and the top token is the same in 94-95 % of positions (meaning kept in 10 of 10 checks at 55k; at 4k context the quality gate gave KL 0.0079, top-1 96.7 %, tasks = stock), but the bytes differ from stock. `--exact` keeps f16 and stock-identical answers; `--no-long-prompt` keeps f16 at any context, also at 128k and above |
| Plugin for models that fit in RAM (new default) | a MoE model that fits in RAM but keeps expert layers on the CPU now runs through the plugin (`--stock` or `MOE_CACHE_STOCK_WHEN_FITS=1` opts out). Measured on one laptop with an 8 GB card (decode faster, prompt about 10 % lower; numbers in the CHANGELOG); on the 16 GB PC the forced-plugin pair was 1.06x prompt, 1.07x decode (one model) <!-- claim:v038-fits-qwen36 -->. Without a GPU the launcher stays on stock llama.cpp (checked with a dry run on the laptop with the GPU hidden). |
| Dense models: `--fit`, FFN in RAM (`-ot`), Intel thread rule | measured on one laptop only (i7-13620H + 8 GB card), one run each; the thread rule applies only when part of the dense model runs on the CPU, and only on Linux hybrid Intel CPUs. `--no-auto-fit`, `--no-ffn-ram` go back to llama.cpp's own split. Other CPUs and cards are untested |
| Long-prompt default sets two more things | from 32k context it applies `-ub 4096`, the q8_0 KV cache and `MOE_CACHE_GPU_MIN_TOKENS=64`; this is deliberate and part of what the +77 % measured. <!-- claim:v038-longprompt-default --> `--no-long-prompt` removes all three |
| `-ub 4096` memory estimate rests on one model | the launcher's estimate of the compute buffer at `-ub 4096` comes from Flash-Next IQ3_XXS at 64k. Another model may need the one automatic retry (the launcher falls back by itself) or `--no-long-prompt` |
| Desktop app not included | the desktop app is not part of this release (its code is in the repository but it is not built or shipped): use the control page in the browser (`moe-cache-ui`) |
| Windows zip is unsigned | Windows shows "unknown publisher" / SmartScreen: Unblock the zip, More info > Run anyway (see Get started). Check the SHA256 or `gh attestation verify` |
| Windows tested on one desktop | Ryzen 9 7950X + RX 9070 XT 16 GB + 32 GB, Windows 11, Vulkan, plus a GitHub Windows runner (CPU). NVIDIA, Intel GPUs and Windows 10 are untested; HIP loads but is slower than Vulkan |
| Models measured | Qwen3.8-Flash-Next (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, Coder IQ1_M), Qwen3.5-122B IQ2_XXS, GLM-5.3-Flash, Qwen3.6-35B, Nemotron 3.5 Bolt 74B (agent speed; plugin forced slower than stock in one test), Granite 1B (CI). Other MoE models should work but are not measured. Long context (64k-256k) is measured on Flash-Next and a few more, single runs |
| Linux | Tested on Intel + NVIDIA, CPU-only, Vulkan, and a user's RX 9070 XT. `install.sh` has been tried on Arch (`--no-apt`) and, for this release, from a clean clone on that machine; a fresh Ubuntu/Debian run with apt is not tested. macOS: no |
| `--ram` is a sizing hint | On Windows the memory limit is set only for the plugin, only with `--ram`, and only with `MOE_CACHE_WIN_VIEW=0`; the default file-view cache, the on-demand listener and stock llama-server get none (a job limit counts GPU memory and broke Vulkan). The server can use more RAM than `--ram` says. On Linux the launcher sets no limit either; only a cgroup you set yourself is a hard limit. No guard is being built |
| A failed model read still aborts | a model file that is cut short or cannot be read stops the server. On Linux it now prints a clear message; on Windows a guard for the in-page read error is in, tested with a simulated error in CI but not with a real cut-short file on a PC |
| Installers need `gh` or an explicit yes | `install.sh`, `install.ps1` and `update --apply` check the build attestation with the GitHub CLI (`gh`; its own `attestation verify` decides, not a login check). Without a working `gh` they stop and say what was and was not checked; `--yes-unverified` (or `MOE_CACHE_ALLOW_UNVERIFIED=1`) goes on with only the SHA256 check, which catches a damaged download but not a tampered release (a missing check is no longer skipped silently) |
| GPU errors | with Vulkan a GPU error inside an expert block is caught and the block runs on the CPU (`MOE_CACHE_GPU_CATCH`, not yet seen with a real failure); on NVIDIA / CUDA a GPU error still ends the server |
| Context length and RAM speed of the speed numbers | the numbers on the card are real 65k conversations with real answers on Flash-Next IQ3_XXS and Q2_0 (and Qwen3.6-35B), two runs per arm, one PC, a head kit with the scheduling tuning off, RAM at 6000 MT/s (older records ran at 4800 MT/s and are not comparable); the long-prompt comparison has one run per arm and two arms of each; the Q2_0 against IQ3_XXS 64k comparison and the identity check are chain-log numbers without records (release notes); nothing is measured beyond 65k on this kit |
| Not tested for this release | `winget install` and `scoop install` (manifests exist but are not submitted); Hermes Desktop and OpenCode Desktop on Windows (click-through still open); a full model run from the final zip |
| Decode after long prompts | on the v0.3.5 kit decode after a 34k-64k prompt was slower than on v0.3.4 (the bigger prompt chunk leaves 2 fewer expert layers on the GPU, RESULTS 79); giving that memory back after the prompt (patch 0070) was tried on the PC on 2026-11-05 and lost, so it is archived (`archive/README.md`); a smaller prompt chunk keeps decode faster at the price of a slower prompt |
| Speed varies between nights | the same test repeated on another night can differ by several percent on this PC (RAM speed, background work, Windows state; a restart before a chain removed an 18 % drift in October); treat the numbers as good to about 10 % and compare only runs from the same session |
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
| [Current results](docs/RESULTS_CURRENT.md) and the [benchmark card page](docs/benchmark_card.html) | the whole card, per-depth tables, method, commands |
| [Older results](docs/results-archive/RESULTS.md) (Windows desktop), [older laptop results](docs/results-archive/RESULTS_OLD_LAPTOP.md) | every earlier measurement, newest first |
| [Release notes](docs/releases/RELEASE_NOTES_v0.3.7.1.md), [changelog](CHANGELOG.md) | what changed in each version |
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
- Credit for ideas on tiering experts across VRAM, RAM and SSD: [Strata](https://github.com/Niko1221/Strata) (MIT, by Niko1221) and [Project Maya](https://github.com/mw00/project-maya) (MIT, by mw00).

## License

- moe-cache is MIT-licensed, see [LICENSE](LICENSE).
- `third_party/ggml/` contains interface headers copied from ggml / llama.cpp (MIT, Copyright The ggml authors). Their license text is kept in `third_party/ggml/LICENSE`.
- moe-cache does not include any model weights. Models you run with it keep their own licenses.
- Product and model names (Qwen, gpt-oss, DeepSeek, Granite, Hunyuan, KAT-Coder and others) belong to their owners and are used only to say what was tested.
