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

<p align="center">Speeds, test settings and every caveat are in the <a href="#results">benchmark card below</a> (Windows desktop, 32 GB RAM + 16 GB graphics card, stock llama.cpp against moe-cache). Full history: <a href="docs/results-archive/RESULTS.md">docs/results-archive/RESULTS.md</a>; the older Linux laptop results: <a href="docs/results-archive/RESULTS_OLD_LAPTOP.md">docs/results-archive/RESULTS_OLD_LAPTOP.md</a>.</p>

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

`python3 bin/moe_cache.py connect NAME --write` writes the config of Hermes Agent, OpenCode, Claude Code, Codex CLI, Pi, OpenClaw, Aider or Qwen Code (without `--write` it only prints; the first backup of your config is kept as `.bak`, `--undo` puts it back, `--check` proves the setup with one prompt and one tool call). Start the server in agent mode first, or let on-demand start it when an agent asks:

```bash
python3 bin/moe_cache.py on-demand /path/to/model.gguf --install-autostart   # loads the model on the first request, unloads it when idle
# or by hand: bin/moe-cache-server /path/to/model.gguf --agent                # context 65536, tool calls, API key, warm sessions
```

Step by step for each agent, the desktop apps and the context limit to set: [docs/AGENTS.md](docs/AGENTS.md).

### Which model?

`pick` decides from your RAM, graphics card, free disk and use (agents, chat, long documents). On a 32 GB RAM + 16 GB GPU PC: Qwen3.8-Flash-Next Q2_0 (66 GB, the default since v0.3.7) for general and agent use: in a real 65k conversation moe-cache reads new prompt tokens 2.2× as fast as stock and decodes 26.98 against 22.46 tok/s <!-- claim:v037-conv-q2_0 -->; Q2_0 is also faster than IQ3_XXS at long context (chain log of 2026-11-06, no record yet), the IQ3_XXS file (76 GB) is the higher-quality option (top-1 agreement with IQ3_S 76.5 % against 73.3 % for Q2_0; measured 2026-11-05, record pending), Nemotron 3.5 Bolt 74B (22 GB, fits in RAM) for 128k contexts. `python3 bin/moe_cache.py models` lists every catalogue model with fits / tight / no for this PC, whether it is already downloaded and its measured speed (or "not measured"). All files and the rules: [docs/MODELS.md](docs/MODELS.md).

## Results

Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), a real 65k conversation, stock llama.cpp against moe-cache with the same command line. This is the one headline table; the whole benchmark card, the per-depth tables, the method, the commands and the memory-budget table are in [docs/RESULTS_CURRENT.md](docs/RESULTS_CURRENT.md), and every older measurement is in [docs/results-archive/](docs/results-archive/RESULTS.md).

| Model (file size) | Context, prompt | stock | moe-cache | change |
|---|---|---|---|---|
| **Flash-Next Q2_0** (the default)<br><sub>66.4 GB, drive D:, two runs each, ABBA</sub> | 65,536, a 46-turn conversation | 22.46 | 26.98 | +20 % <!-- claim:v037-conv-q2_0 --> |
| Flash-Next IQ3_XXS (the higher-quality option)<br><sub>75.8 GB, drive D:, two runs each, ABBA</sub> | 65,536, a 49-turn conversation | 16.0 | 21.73 | +36 % <!-- claim:v037-conv-iq3xxs --> |

Each number is the mean of two cold runs in the order stock, moe-cache, moe-cache, stock (ABBA), whole conversation. Both files are bigger than the free RAM once the GPU and Windows have taken their share. Stock = the same llama-server command the launcher builds for the plugin, without the plugin.

### Where it does not help

- **Models that fit in RAM.** The launcher runs plain llama.cpp for them by itself. Forced through moe-cache, Qwen3.6-35B in a 65k conversation was somewhat faster than stock (one model, one PC, see RESULTS 80), while earlier laptop runs of models that fit showed stock as fast or faster; no steady gain, so the default stays.
- **Short answers at shallow depth.** The gain in a conversation comes from reading prompts and from keeping the hot experts in RAM; a short chat on a model that already fits in RAM does not gain.
- **CPU kernels.** The numbers here were measured with llama.cpp's own CPU kernels on both sides; the gain comes from which experts are in RAM and how they are read, not from faster math.
- **Cold start.** From a cold file cache the plugin reports ready a few seconds later than stock while it loads its hot experts; the first answer wins it back (RESULTS 61).
- **Small samples.** Two runs per arm, one PC, one seeded script, a v0.3.7 candidate kit. Cleanliness: see the note under the headline numbers.

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
| Installers need `gh` or an explicit yes | `install.sh`, `install.ps1` and `update --apply` check the build attestation with the GitHub CLI (`gh`; its own `attestation verify` decides, not a login check). Without a working `gh` they stop and say what was and was not checked; `--yes-unverified` (or `MOE_CACHE_ALLOW_UNVERIFIED=1`) goes on with only the SHA256 check, which catches a damaged download but not a tampered release (RELEASE-686 fixed: no silent skip) |
| GPU errors | with Vulkan a GPU error inside an expert block is caught and the block runs on the CPU (`MOE_CACHE_GPU_CATCH`, not yet seen with a real failure); on NVIDIA / CUDA a GPU error still ends the server |
| Context length of the speed numbers | the numbers on the card are real 65k conversations on Flash-Next Q2_0 and IQ3_XXS, two runs per arm, one PC, a v0.3.7 candidate kit (the clean check's CPU counter was empty in two checks, see the note on the card); Q2_0 against IQ3_XXS on one 64k prompt and the identity check are chain-log numbers without records yet (release notes); nothing is measured beyond 65k on v0.3.7 |
| Not tested for this release | `winget install` and `scoop install` (manifests exist but are not submitted); Hermes Desktop and OpenCode Desktop on Windows (click-through still open); a full model run from the final zip |
| Decode after long prompts | on the v0.3.5 kit decode after a 34k-64k prompt was slower than on v0.3.4 (the bigger prompt chunk leaves 2 fewer expert layers on the GPU, RESULTS 79); giving that memory back after the prompt (patch 0070) was tried on the PC on 2026-11-05 and lost, so it is archived (`archive/README.md`); a smaller prompt chunk keeps decode faster at the price of a slower prompt |
| IQ3_XXS prompt speed varies | IQ3_XXS prompt speed deep in a 65k conversation varies from night to night on the test PC (145 tok/s at 48k-64k depth in the overnight run <!-- claim:v037-conv-iq3xxs -->, about 30 % more the day before on the v0.3.6 kit). A same-session check of the v0.3.6 and v0.3.7 kits on a 64k prompt gave the same speed, so it is not the plugin; treat IQ3_XXS numbers as good to about 25 %. Q2_0 was unchanged. Details: RESULTS 80 |
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
| [Release notes](docs/releases/RELEASE_NOTES_v0.3.7.md), [changelog](CHANGELOG.md) | what changed in each version |
| [Hardware and limits](docs/PLATFORMS.md) | what is tested where, known limitations |
| [Configuration](docs/CONFIGURATION.md) | environment variables |
| [FAQ](docs/FAQ.md) | questions, and how moe-cache compares with other projects |
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
