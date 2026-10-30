<h1 align="center">moe-cache</h1>

<p align="center"><b>Run Mixture-of-Experts models that are bigger than your RAM, without the speed cliff.</b><br>
A plugin for an unmodified <code>llama-server</code>. No fork, no patch.</p>

<p align="center">
<img alt="status" src="https://img.shields.io/badge/status-v0.3.3%20beta-orange">
<img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
<img alt="platform" src="https://img.shields.io/badge/platform-Linux%20%2B%20Windows%20(experimental)-lightgrey">
<img alt="llama.cpp" src="https://img.shields.io/badge/llama.cpp-tested%20b11669%20%2F%20b11966%20(Windows)%2C%2023b0202a1%20%2F%207fe450e19%20(Linux)-green">
<a href="https://ko-fi.com/ibrumm"><img alt="Support on Ko-fi" src="https://img.shields.io/badge/Ko--fi-support%20this%20project-ff5e5b?logo=ko-fi&logoColor=white"></a>
</p>

<p align="center"><img src="docs/img/results.svg" alt="speed with and without moe-cache" width="860"></p>

## What it is

**moe-cache lets a normal PC run big Mixture-of-Experts AI models** (files of 60-120 GB) that are bigger than its RAM, with the **same answers as plain llama.cpp**: token-identical below 128k context with the same layer split; from 128k context a q8_0 KV cache is used (`--exact` keeps f16), see [EXACTNESS](docs/EXACTNESS.md). Chat in your browser, or connect coding tools (OpenCode, Hermes, Claude Code). Windows 11 and Linux. Free (MIT), no account.

**Example (measured on one Windows 11 desktop, 32 GB RAM + 16 GB graphics card, Vulkan):** Qwen3.8-Flash-Next IQ3_XXS (75.8 GB, the default file) writes **19.9 tokens per second** with moe-cache and 15.1 with plain llama.cpp; the faster Q2_0 file (66.4 GB) 22.2 against 18.4 (RESULTS 59, 63). More machines and models: [Results](#results-measured-clean-runs).

<a href="docs/benchmark_card.html"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/benchmark_card-dark.png"><img src="docs/benchmark_card.png" width="720" alt="Benchmark card: decode tokens per second, stock llama.cpp vs moe-cache, on a Windows PC with 32 GB RAM and a 16 GB graphics card"></picture></a>

<sub>The top of the [benchmark card](docs/benchmark_card.html) (full card: long context, other models, agents). Numbers and methods: [docs/RESULTS.md](docs/RESULTS.md).</sub>

| You need | |
|---|---|
| Memory | 32 GB RAM for the big models (16 GB works for smaller ones such as Qwen3.6-35B); a graphics card is optional but recommended |
| Disk | an SSD (NVMe is best) with the model size plus about 10 GB free; most models here are 20-120 GB |
| Software | Python 3.8 or newer, and llama.cpp's official `llama-server` download |
| Time | about 20 minutes of setup, plus the model download |

If your model already fits in RAM, plain llama.cpp is as fast or faster; moe-cache then steps aside on its own.

### Why it is needed

MoE models such as Qwen3, gpt-oss or DeepSeek only use a few "experts" per token, but stock llama.cpp keeps all expert weights in memory-mapped files.
When they do not fit in RAM, the OS page cache thrashes and speed falls off a cliff.

### When to use something else

| Your situation | Best choice |
|---|---|
| The model **fits in your RAM** | **Stock llama.cpp**, ideally with its GPU expert cache (`--moe-cache-mib N`). That flag is llama.cpp's own GPU expert cache, not moe-cache's: on our Windows Vulkan PC it was slower than without it (RESULTS 59), and its output can differ slightly from stock (not measured by us). On Qwen3.6-35B (21.7 GB) at a 24 GB limit, both with the same 5000 MiB GPU cache: stock 42 tok/s, moe-cache 30-34 (RESULTS section 29). moe-cache adds nothing here but its CPU pinning, which stock gets with `-C MASK --cpu-strict 1`. |
| You want **Qwen3.8-Flash-Next or Qwen3.6-35B as fast as possible** and have **12 GB+ VRAM and 32 GB+ RAM** (more RAM is better) | [Strata](https://github.com/Niko1221/Strata), an engine with its own kernels for those models (its authors report 94 tok/s on an RTX 5070 with 64 GB RAM; see "Strata and Maya today" below). We have not run it ourselves. |
| The model is **bigger than your RAM**, or it is **another MoE architecture**, or you want to stay on **stock llama.cpp with the same tokens** | **moe-cache**: about **1.2-2x faster than plain llama.cpp on Windows and 2-5x on Linux** when the model is bigger than your RAM (Windows since 2026-10-27, RESULTS 59: 1.2x to 2x on the same models, GLM-5.3 3.1x with MTP, coding-agent sessions 2x to 2.4x faster; Linux clean runs since 2026-10-15: 2.4x to 5.2x, the oldest row gpt-oss-120b 12x; tables below). No gain when the model fits in RAM. |

## Get started

**Windows 11?** Go to the [Windows quick start](#windows-quick-start-experimental) below (7 steps, [docs/WINDOWS.md](docs/WINDOWS.md) has the details). **Linux?** Read on.

**You need:** a `llama-server` from llama.cpp's official downloads (the Windows `.zip` and the Linux `llama-bNNNNN-bin-ubuntu-*.tar.gz` files are built with the shared ggml libraries moe-cache needs; we tested the Windows ones, not the Linux ones) or your own build with `-DBUILD_SHARED_LIBS=ON`; Python 3.8 or newer; and a MoE model as a `.gguf` file (step 2 below). A GPU is optional. Tested llama.cpp: b11669 and b11966 (Windows); commits `23b0202a1` (ggml 0.26.0, backend API 3) and `7fe450e19` (ggml 0.25.1, API 2) (Linux), each Linux API needs its own plugin build, see [docs/QUICKSTART.md](docs/QUICKSTART.md). Other versions may work; `check` tells you.
Ollama user? The models in `ollama list` are plain GGUF files: the path is on the line starting with `FROM` in `ollama show NAME --modelfile`.

### 1. Install (pick one)

| | Best if | What you do |
|---|---|---|
| **A. Prebuilt** (2 minutes) | you already have a `llama-server` built with shared ggml libraries (see above) | download the file for your system from [Releases](https://github.com/IbrahimSiddiqui007/moe-cache/releases) (table below), then `tar xzf` it and `cd` into the folder. Releases built by CI can be verified: `gh attestation verify FILE.tar.gz -R IbrahimSiddiqui007/moe-cache` |
| **B. One script** (10 to 20 minutes) | fresh Ubuntu / Debian, nothing installed | `git clone https://github.com/IbrahimSiddiqui007/moe-cache && cd moe-cache && ./install.sh` (add `--vulkan` for AMD / Intel / NVIDIA through Vulkan). It builds the matching llama.cpp and the plugin and prints the next commands. We have not tried it on Ubuntu or Debian yet (only on Arch with `--no-apt`): please report problems, and prefer A with the official llama.cpp download if it fits you. |
| **C. From source** | you want to build it yourself | `git clone ... && cd moe-cache && ./build.sh` (needs g++; finds `libggml-base.so` next to your llama-server, or set `GGML_LIB_DIR`) |

Which file to download (the names show version 0.3.2; take the newest version on the page):

| Your system | File on the Releases page |
|---|---|
| Windows 11 | `moe-cache-v0.3.3-windows-x64.zip` |
| Linux, llama.cpp from commit `23b0202a1` or newer (ggml 0.26 and up, including today's ggml 0.28) | `moe-cache-v0.3.3-ggml0.26-linux-x86_64.tar.gz`: **includes the official llama.cpp b11966 (Vulkan + CPU build) in its `llama/` folder**, so there is nothing else to download |
| Linux, older llama.cpp (ggml 0.24 / 0.25, up to commit `7fe450e19`, e.g. Arch `llama.cpp-cuda`) | `moe-cache-v0.3.3-linux-x86_64.tar.gz` (no llama.cpp inside: it needs your own build of that era) |

Not sure? Run `check` (step 3): it says which file fits.

**Linux, the `-ggml0.26` file:** the launcher, `check`, the picker and the control page find the bundled `llama/llama-server` by themselves (llama.cpp's MIT licence and a note are next to it; the files are unmodified, sha256-checked against the release asset when the tarball is built). A `llama-server` of your own wins when it is on the `PATH` or named in `LLAMA_SERVER`, and among the default folders the newest build wins: use that for a CUDA or ROCm card (the bundled build is the Vulkan + CPU one: AMD, Intel and NVIDIA through Vulkan; it needs `sudo apt install libvulkan1 mesa-vulkan-drivers` on Ubuntu / Debian). `./install.sh` in that folder builds nothing and needs no compiler; `./install.sh --own-llama` builds llama.cpp as before.

**Windows (experimental)**: steps below and in [docs/WINDOWS.md](docs/WINDOWS.md).
**Windows may warn: "unknown publisher".** The kit's DLLs and `.bat` files are not code-signed (a certificate costs money every year; the code is
open source and the zip is built by GitHub Actions from the tagged commit), so Windows treats the downloaded zip as untrusted. Before you
extract it: right-click the zip > **Properties** > tick **Unblock** > **OK**. If SmartScreen still says "Windows protected your PC", click
**More info** > **Run anyway**. To check that your zip is the one CI built, compare its SHA256 with `SHA256SUMS-windows` on the release page:
`Get-FileHash moe-cache-vX-windows-x64.zip` in PowerShell (or `gh attestation verify moe-cache-vX-windows-x64.zip -R IbrahimSiddiqui007/moe-cache`).

### 2. Get a model (once)

Models are single `.gguf` files from Hugging Face, 20-120 GB each. **Not sure which one?** `python3 bin/moe_cache.py pick` (Windows: `python bin\moe_cache.py pick`) reads your RAM, graphics card and free disk, names one model and prints the download command; it never downloads by itself. That command starts with `hf`: install it once with `pip install -U huggingface_hub`. You need free disk space of the file size plus about 10 GB. Keep the file on an SSD, for example in `D:\models`. Any MoE GGUF you already have works too (Ollama's models are plain GGUF files, see above).

### 3. Check your setup (changes nothing)

| System | Command |
|---|---|
| Linux | `bin/moe-cache-check /path/to/model.gguf --llama-server /path/to/llama-server` |
| Windows | `python bin\moe_cache.py check D:\models\model.gguf --llama-server C:\llama\llama-server.exe` |

Every line says OK, WARN or FAIL and what to do about it (it also times a short read test on the model's drive and warns when the drive is too slow). Add `--run` to also start llama-server with the plugin for one short request (about a
minute): "cache runs: yes" means the cache really computed on your machine. **Run this first, on Windows too.** `check` and `doctor` are the same command; it checks the GPU, RAM, the model's drive and llama-server, and ends with a table of your GPU, driver, RAM, disk and llama-server and a list of what to fix.
If the GPU also drives your screen and answers slow down sharply, the card is too full: add `--gpu-margin 2` to the server command (GiB of VRAM left free; default about 0.5).

### 4. Run your model

Not sure which model fits your machine? `python3 bin/moe_cache.py pick` recommends one (see **Which model for my machine?** below); `pick --apply` makes it the default for `moe-cache-server` without a file name.

```bash
bin/moe-cache-server /path/to/model.gguf --ram 12 --llama-server /path/to/llama-server --open
```
`--ram 12` is the memory (GB) it may use; leave it out to use what is free. The launcher plans the GPU/CPU split from your free VRAM, starts `llama-server` with the plugin and opens the chat page (the same address, `http://localhost:8080`, is an OpenAI-compatible API for editors, Open WebUI and scripts).

**The first start is slow.** The server is ready after about 20 seconds (RESULTS 64: 14-19 s); the first answer to a long prompt can take minutes on a model bigger than your RAM (a cold first turn of 7,461 tokens took 186 s on our laptop, [docs/AGENTS.md](docs/AGENTS.md); a 63k-token prompt about 5 minutes on our Windows PC, RESULTS 64). The chat page shows progress. Later turns are much faster because the conversation is kept.

**Speed options that change the answers a little (all off by default; the launcher suggests `--fast` when it would help):**

| Option | What it does | Cost |
|---|---|---|
| `--fast` | about +10 % on models much bigger than your RAM | output no longer identical to stock |
| `--fast=2` | a stronger level for memory far below the model size (Flash-Next at 12 GB: +43 % over `--fast`) | same answers in different words |
| `--fast=kv` | only the q8_0 KV cache (Flash-Next Q2_0 at ctx 262144: prefill 2x, decode +11-13 % at 64k) | KL 0.0079, 190/202 tasks = stock (RESULTS 72); it is the default from 128k context, `--exact` turns it off |
| `--fast=fewer` | `--fast` plus two fewer experts per token on the CPU-held layers | KL 0.034, 192/202 tasks; faster at 4k, **unproven at 64k-256k** |

**Easiest: the control page**: `bin/moe-cache-server --ui` (Windows: double-click `moe-cache.bat` in the kit folder) opens http://127.0.0.1:8070, a three-pane
chat app: left your chats and models (each with the planner's verdict: fits in RAM, needs the cache or fits on the GPU, and why in one sentence),
centre the chat with a speed chip per answer, right live tok/s, cache hit rate, SSD reads, RAM and VRAM, plus a settings drawer (RAM/VRAM budget,
context, mode, MTP). "Quick benchmark" runs a short fixed request and, in one click, the same on stock llama-server for comparison. See [docs/GUI.md](docs/GUI.md).

**Coding agents (Hermes Agent, OpenCode).** They only talk to an OpenAI-compatible address; moe-cache provides it. Easiest, Ollama-style: a small
listener that loads the model on the first request and unloads it when idle (`--install-autostart` starts it at every login):
```bash
python3 bin/moe_cache.py on-demand /path/to/model.gguf --llama-server /path/to/llama-server --install-autostart
python3 bin/moe_cache.py connect hermes --write      # and/or: connect opencode --write (writes URL, model, 65k context, API key); connect claude prints the lines for Claude Code
```
**Agent context limit.** Keep the agent's own context limit below the server's window (65536 by default); the reason and the numbers are in [docs/AGENTS.md](docs/AGENTS.md#keep-the-agents-own-context-limit-below-the-servers).

Several models: `on-demand /path/to/model2.gguf --add` adds one; the request's `"model"` picks it, one is loaded at a time (`connect opencode` lists them all).
Or start a server yourself with `bin/moe-cache-server /path/to/model.gguf --agent` and run the same `connect` line. Measured on a Windows desktop
(RESULTS 60, recorded sessions replayed, cold, stock vs moe-cache): Flash-Next (68 GB on 32 GB RAM) OpenCode session 175 -> 73 s, Hermes 164 -> 82 s,
two agents at once 335 -> 145 s; Qwen3.6-35B (fits in RAM) 5-9 % faster. Details and the Desktop-app steps: [docs/AGENTS.md](docs/AGENTS.md).

### Windows quick start (experimental)

| Step | What to do |
|---|---|
| 1. llama.cpp | **Included in the kit from v0.3.3** (folder `llama\`: the official llama.cpp b11966 Vulkan build, unmodified). With an older kit: download the **Vulkan** zip, the file whose name ends in `-bin-win-vulkan-x64.zip`, from [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases) and unpack it to `C:\llama`. Use it for AMD, Intel and NVIDIA cards; the CUDA zip is untested with moe-cache, skip the HIP and SYCL zips. We tested b11669 and b11966; newer builds should work and `check` (step 5) tells you if not |
| 2. Python | Python 3.8 or newer from python.org (tick "Add python.exe to PATH"). If typing `python` opens the Microsoft Store instead, use `py` in its place |
| 3. The kit | Download `moe-cache-v0.3.3-windows-x64.zip` (take the newest version) from this project's [releases](https://github.com/IbrahimSiddiqui007/moe-cache/releases) and unzip it, e.g. to `C:\moe-cache` |
| 4. A model | Not sure which? `python bin\moe_cache.py pick` names one and prints the download command (see **2. Get a model** above; `pip install -U huggingface_hub` first). Keep the file on an SSD, e.g. `D:\models`, with the file size plus 10 GB free |
| 5. Check | In a command prompt in the kit folder: `python bin\moe_cache.py check D:\models\model.gguf` (finds the kit's `llama\llama-server.exe`, or `C:\llama\llama-server.exe`, by itself; `--version` prints the version). Every line says OK, WARN or FAIL and what to do |
| 6. Run | Double-click `moe-cache.bat`: the control page opens; pick a model and press Start. The first start is slow, see "The first start is slow" above. `moe-cache.bat --shortcut` adds it to the Start menu. Or in a command prompt: `python bin\moe_cache.py server D:\models\model.gguf --llama-server C:\llama\llama-server.exe --open` |
| 7. Agents | Add `--agent`, then `python bin\moe_cache.py connect opencode --write` (or `hermes`); the page also has Connect buttons |

Keep the Windows page file at "system managed" (or 14 GB or more): the cache costs no commit (it keeps experts in read-only views of the model file), but Windows counts the GPU's memory against RAM + page file. Details and known limits: [docs/WINDOWS.md](docs/WINDOWS.md).

### 5. Optional: prove it is correct and measure it on your machine

Linux (from the tarball or a checkout; the Windows kit has `tools\windows_test.ps1` instead, see docs/WINDOWS_TEST.md):

```bash
tests/identity.sh /path/to/model.gguf --llama-server /path/to/llama-server      # PASS = the same tokens as stock llama.cpp
```

More detail, troubleshooting and the Vulkan build: [docs/QUICKSTART.md](docs/QUICKSTART.md).

## Results (measured, clean runs)

Re-measured on 2026-10-15 with the current code where the model is still on the test machine (RESULTS section 44); older rows say which version they come from.

| Situation | Stock llama.cpp | moe-cache | Measured with |
|---|---|---|---|
| **GLM-5.3-Flash (102 GB, 4x the memory limit)**, 24 GB RAM, 8 GB GPU | 0.32 tok/s | **1.75 tok/s** exact, 2.1 with `--fast` | v0.2.3.2 (§26) |
| **Qwen3.8-Flash-Next IQ2_XS (68 GB)**, 24 GB RAM, 8 GB GPU, launcher defaults | 5.2 tok/s | **12.5 tok/s**; 13.1 with `--gpu-cache auto`; 17.6 with `--hybrid` (approximate) | v0.2.3.5 (§44, §28, §31) |
| **Qwen3.5-122B-A10B**, 24 GB RAM, 8 GB GPU, launcher defaults: IQ2_XXS (36.6 GB) / IQ3_XXS (44.7 GB) / IQ4_XS (60.2 GB) | 1.89 / 1.10 / 0.91 tok/s | **7.7 / 5.7 / 3.7 tok/s** (4-5x, same tokens); 9.0 / 6.8 / 4.5 with `--fast` | v0.2.3.5 (§45) |
| **gpt-oss-120b (63 GB)**, 24 GB RAM, 2 expert layers on the GPU, ctx 8k, one run each | 0.37 tok/s | **4.47 tok/s** | v0.1 (§12b, §12d): stock and plugin from different sessions, stock flagged not clean; the clean all-CPU pair of §12 is 0.34 vs 2.71 (8x) |
| Qwen3-Next-80B (48.5 GB), 24 GB RAM, ~2.6 GB GPU use | 2.3 tok/s | **10.6 tok/s** | v0.1 (§6) |
| Same model, 20 GB RAM | 1.3 tok/s | **9.3 tok/s** | v0.1 (§6) |
| Qwen3.6-35B (21.7 GB), 12 GB RAM, 6 expert layers on the GPU | 6.9 tok/s | **22.6 tok/s** | v0.2.3.5 (§44) |
| Qwen3.6-35B, 12 GB RAM, **no GPU at all** | 3.1 tok/s | **9.9 tok/s** | v0.1 (§5) |
| **Model fits in RAM** (Qwen3.6-35B, 24 GB, warm) | **42.9 tok/s** | 36.6 tok/s | v0.2.3.5 (§44): the launcher now runs stock llama.cpp here |

### Prompt processing (prefill)

| Model, memory limit | Stock llama.cpp | moe-cache | Measured with |
|---|---|---|---|
| Qwen3.6-35B (21.7 GB), 12 GB RAM, 2-5k token prompts | 132.8 tok/s (2.4k prompt) | **136.5 tok/s** first prompt, **186 to 206 tok/s** on the next ones | v0.2.3.5 (§46) |
| Qwen3.8-Flash-Next IQ2_XS (68 GB), 24 GB RAM, 2-5k token prompts | 65.9 tok/s (2.4k prompt) | **90 to 103 tok/s** | stock v0.2.3.5 (§46), moe-cache v0.2.3.3 (§38) |
| Qwen3.8-Flash-Next IQ2_XS (68 GB), 24 GB RAM, context 8192, 2-6k token prompts | - | **93 to 135 tok/s** (two runs) | experimental 2026-10-26, with the tuner fix (§52) |

For long prompts llama.cpp already computes the experts on the GPU, so the prefill gain is smaller than the decode gain; it grows with how far the model
exceeds RAM. Bigger prompt chunks on a 12-16 GB card are the next step (BIGGER_MACHINES B13).

"Stock" is the same `llama-server` binary with the same command line (same context, same number of expert layers on the GPU through `--n-cpu-moe`, its default memory-mapped loading), only without the plugin. moe-cache does not replace `--n-cpu-moe`; it manages the experts that `--n-cpu-moe` leaves on the CPU.

Conditions: "RAM" is a memory limit set with a cgroup on a 31 GB machine; before every limited stock run the model file is dropped from the page cache (otherwise
stock can read it from cache that does not count against the limit and looks up to 4.6x faster than it is, RESULTS 44). One run per row unless the section says paired.
When the model fits in the memory llama-server may use, stock llama.cpp is faster (its page cache holds the whole file and stays warm across restarts), so
`moe-cache-server` starts stock llama-server then and says so.

**Tip for Intel 12th gen and newer laptops (performance + efficiency cores):** pin llama.cpp to one thread per performance core, e.g. `-C 0x555 --cpu-strict 1` on a 6-performance-core CPU (this alone gave stock llama.cpp +15 % decode speed on our laptop; `moe-cache-server` does it for you). Details in [docs/RESULTS.md](docs/RESULTS.md), section 11.

Methods, all numbers and every caveat: [docs/RESULTS.md](docs/RESULTS.md). One-page summary of the Windows desktop results: [docs/benchmark_card.html](docs/benchmark_card.html) (its top is the image at the start of this page; `python3 tools/card_image.py` remakes the image after a change to the card).

**Which Flash-Next file?** Default: **IQ3_XXS** (3-bit: reads best, most tasks right, popular for coding); **Q2_0** when you want speed (about 10 % more tokens per second, e.g. for long thinking). Measured with moe-cache on a 16 GB GPU + 32 GB RAM PC (RESULTS 59, 60, 63; Qwen3.8-Flash-Next GSQ-RCO files; tasks = our
202 exact-answer tasks; **no quality difference between any two files is proven**, all paired sign tests p > 0.05):

| File | Size | Decode tok/s | Prompt tok/s (3k / 34k tokens) | Tasks /202 | Use it |
|---|---|---|---|---|---|
| **Q2_0** | 66.4 GB | **22.2**; **24.7-25.7** with the async reads (N15, +12 %, RESULTS 67) | **281 / 287**; long prompts **+8-11 %** (N23 pipelined block) and **+2-5 %** at 34k (N22), RESULTS 69; decode **+0.3-2.6 %** with the fused gate+up job (E3, RESULTS 73, every pair, not clean) | 191 | **the faster file**: about 10 % more tokens per second than IQ3_XXS (e.g. for long thinking), 9 GB smaller; vs IQ2_XS +5 % decode, prompts +11 % / +19 % |
| **IQ3_XXS** | 75.8 GB | 19.9 | - | **193** | **recommended default** (since 0.3.2): 3-bit, the most tasks right, reads best for coding; about 10 % slower decode than Q2_0 |
| ISTA GSQ-RCO Coder IQ1_M | 58.4 GB | 21.2 (agent session) | - | 189 | coding agents: half the experts (256 of 512); OpenCode session 55.5 s vs 73.4 s on IQ2_XS. General knowledge not checked beyond the task set |
| IQ2_XS | 68.0 GB | 21.0 (same day as Q2_0; 21.9 in RESULTS 59) | 253 / 242 | 187 | superseded by Q2_0 (older rows above that say IQ2_XS were measured on IQ2_XS) |
| IQ3_S | 83.6 GB | 17.3 | - | 192 | not worth it: slower and bigger than IQ3_XXS for no gain |

**Which model for my machine?** Run `python3 bin/moe_cache.py pick` (Windows: `python bin\moe_cache.py pick`) or open the control page's Settings > **Which model?**.
It reads your RAM (and the hard cap the server keeps: RAM minus max(4 GiB, 12.5 %)), dedicated VRAM, free disk, SSD or HDD, CPU (AVX2 / AVX-512) and your
llama-server's build, asks what it is for (agents / coding at 64k-256k, plain chat, long documents), and recommends one model, file, context and options.
It never downloads by itself: it prints the command with size and source, and `--apply` (or the page's *Use it*) makes `moe-cache-server`,
`on-demand` and the page start that model by default, so `connect hermes|opencode|claude` points the agent at it. `--gpu-margin GiB` (default 0.5) is taken off the VRAM in its memory estimate, as the launcher does; if the GPU also draws your screen, start with `--gpu-margin 2`. Models Ollama has already downloaded (its manifests in `~/.ollama/models`, `%USERPROFILE%\\.ollama\\models` or `$OLLAMA_MODELS`) count as local: llama-server reads the blobs directly. If the disk is too small for the model that suits your machine, the picker says which model that is and how much disk is missing, offers no download, and prefers a model that is already here. Where we have no number for your class of
machine it says "untested on this class of machine". The rules, from the measurements (RESULTS 44-45, 53, 59-67; the PC is 32 GB RAM + 16 GB GPU):

| Your machine | Agents / coding (64k-256k) | Plain chat | Long documents |
|---|---|---|---|
| **32 GB RAM + 16 GB GPU** (measured) | **Nemotron 3.5 Bolt 74B** at 128k, stock: decode 27.9 (24.9 at 114k), OpenCode session 47.8 s | **Flash-Next IQ3_XXS**: 19.9 tok/s, 193/202 tasks, reads best for coding (Q2_0: 24-25 tok/s, about 10 % faster, 191/202) | **Bolt**: prompts 595 tok/s at 57k, 721 at 114k (Flash-Next Q2_0 170-220 at 128k) |
| 32 GB + 8 GB GPU | Bolt at 64k (untested with 8 GB) | Flash-Next Q2_0 (untested; the IQ2_XS file of the same size gave 12.5 tok/s on 24 GB + 8 GB) | Bolt at 64k (untested) |
| 64 GB + 16 GB GPU | Bolt | Flash-Next IQ3_XXS (untested here) | Bolt |
| 128 GB+ | Bolt | GLM-5.3-Flash (fits; untested) | Bolt |
| 16 GB + 6 GB GPU, or no GPU | Qwen3.6-35B-A3B (22.6 tok/s with moe-cache at 12 GB RAM + 8 GB GPU; 9.9-10.6 tok/s without a GPU; agents on such a machine untested at 64k) | Qwen3.6-35B-A3B | Qwen3.6-35B-A3B |
Bolt needs llama.cpp **b11966 or newer** (the picker checks `llama-server --version` and says so); models bigger than RAM are not suggested when they sit
on a hard disk, and a model is skipped when the disk has less free space than its size. Bolt's general knowledge beyond our 202 tasks is not checked.

**Long context (Windows PC, RESULTS 64; agent mode, one cold run per point, not ABBA).** Flash-Next Q2_0 at a 64k setting: moe-cache reads a 63.5k-token
prompt 1.8x faster than stock (about 289 vs 525 s; prefill 266 / 195 tok/s at 4-21k / 53-63k vs 171 / 106) and decodes 20.7 vs 16.8 tok/s at that depth
(+23 %). Qwen3.5-122B with moe-cache at 64k: prefill 266 / 217, decode 17.3 at 63.5k. Qwen3.6 (fits, stock): 848 / 739, decode 59.1. At the 256k setting the
planner now keeps the long-prompt GPU block its room (RESULTS 67 / 69, paired runs): Flash-Next prefill 108-110 tok/s at 4-34k and decode **21.9 vs 18.5 tok/s** at 4k (+18 %) over the old plan; decode at 138k depth was 15.9 with the old plan (not re-measured).

**Other models (Windows PC, RESULTS 65, one cold run per point).** Nemotron 3.5 Bolt 74B-A9B (NVIDIA's 2-bit QAT file, 22.0 GB) fits in RAM + VRAM,
so the launcher runs stock llama.cpp for it (needs llama.cpp b11966 or newer): decode 27.9 tok/s (23-25 at 114k-228k tokens), prompts 595 tok/s at
57k and 721 at 114k, 190/202 tasks (Flash-Next Q2_0 191, not proven), OpenCode session 47.8 s (the fastest measured). moe-cache forced on it is slower
(decode 22.1, prompt 329 at 57k): the plugin is for models bigger than memory. DeepSeek V4-Flash: not tested.

Hardware for the other measurements above (not this file table): Intel i7-13620H, 31 GB RAM, RTX 4060 8 GB, NVMe SSD (a Ryzen 7 5800H mini PC in RESULTS 50; the Windows 11 desktop in RESULTS 54-55).

## Hardware and OS support

| Platform | Status |
|---|---|
| Linux + Intel CPU + NVIDIA GPU (CUDA) | **Tested** |
| Linux + Vulkan (NVIDIA, and the Intel integrated GPU) | **Tested** on llama.cpp `7fe450e19` only, loaded the official way (`GGML_BACKEND_PATH`; see the feature table below) |
| Linux, no usable GPU (CPU only) | **Tested**, identical to stock, within about 4 % of its speed when the model fits |
| Linux + AMD CPU, AMD integrated GPU (Vulkan) | **Tested** on a Ryzen 7 5800H with Vega 8 (2026-10-16): identical to stock, Qwen3.6 at 12 GB 1.23x stock (no expert layer fits on a 3 GiB iGPU); RESULTS 50 |
| Linux + AMD discrete GPU | **Tested by a user** (GitHub issue #1: RX 9070 XT + Ryzen 7 7700, Linux, Vulkan): identity PASS, Flash-Next 15.9 -> 18.8 tok/s. Not yet run by the developer on Linux |
| Windows | **Experimental, now faster than stock.** Identical to stock in CI on a Windows runner and on a Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB, Vulkan). Since 2026-10-27 the cache keeps experts in read-only views of the model file (Windows charged its private memory, and Vulkan's GPU memory, to commit, which held the cache at 8-11 GiB). Cold, stock vs moe-cache, launcher defaults (RESULTS 59-60): Flash-Next Q2_0 (the recommended file) **22.2 vs 18.4 tok/s**, prompts 2.3x (RESULTS 63), IQ2_XS 21.9 vs 14.9, IQ3_XXS 19.9 vs 15.1, IQ3_S 17.3 vs 12.9, Qwen3.5-122B 19.7 vs 16.0 (re-measured 2026-10-28, RESULTS 61; replaces 16.1 vs 13.9, which ran during a PC-wide slowdown), GLM-5.3 4.4 vs 2.2 (**6.9 with `--mtp`**); text byte-identical. Context checkpoints stay on with AMD driver 32.0.32015.7029 or newer (older ones crash with Flash-Next). See docs/COMPATIBILITY.md |
| macOS | No |

## Benchmark it on your own model and your own use case

**Quick, on Windows or Linux:** `python bin/moe_cache.py bench-report MODEL.gguf` runs stock llama.cpp and moe-cache (stock, moe-cache, moe-cache, stock)
on a model bigger than your RAM and writes a report to paste into a [Benchmark report issue](https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=benchmark_report.yml).
It only measures, and the report holds no prompts, answers or names. Recipe: [docs/SHARE_YOUR_NUMBERS.md](docs/SHARE_YOUR_NUMBERS.md). Reports from other machines are very welcome.

The full benchmark tool (Linux) (web page, standard protocol, many-model matrix, quality harness) now lives in its own repository: [moe-cache-benchmark](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark).
Clone it next to this folder (or set `MOE_CACHE_DIR` for the benchmark tool) and it finds the plugin and the planner by itself.

## How it works

In short, moe-cache manages the CPU-side experts itself:

- keeps the **hot experts in RAM** and reads the rest from your SSD with large, parallel direct reads;
- **sizes its cache from your real memory limit** and stays under it (no out-of-memory kills);
- **learns which experts you use** and warm-starts from that next time (`moe-cache-learn` builds a profile in a minute);
- produces **token-identical output** to stock llama.cpp (checked on every model in [RESULTS](docs/RESULTS.md); the Hunyuan test was short only);
- works with speculative decoding (MTP; it helps when the model or most of it fits: Qwen3.6 at 12 GB 21.7 -> 24.0 tok/s, but it was 17 % slower on Qwen3.8-Flash-Next with an 8 GB card), split GGUF files, MXFP4 / K-quants / IQ4_XS / Q5_0, CPU-only machines and Vulkan GPUs;
- runs long prompts on the GPU like stock llama.cpp does;
- **expert executor** (on by default): while missing experts load from the SSD it already computes the ones in RAM, and on hybrid Intel CPUs it can use the efficiency cores too. Results are bit-identical (every output was compared byte for byte with llama.cpp's own on the test models);
- **faster Q2_0 kernel**: llama.cpp has no SIMD Q2_0 dot product on x86; the plugin supplies one (bit-identical, about 3x faster), which matters for models whose down projections are Q2_0 (for example Qwen3.8-Flash-Next).

The path of one operation:

```
 llama.cpp builds the graph of the model
          |
  ggml scheduler hands each operation to a backend
          |
   +------+------------------+
   |  GPU backend            |  attention, norms, shared layers, the experts you put on the GPU
   |  CPU backend            |  everything else
   |  moe-cache backend  <-----+  the CPU-side expert matrix multiplications:
   +-------------------------+    1. look at which experts the router picked
                                  2. make sure those experts are in RAM (read the missing ones from the SSD, in the background)
                                  3. compute the experts that are ready while the others still load (several threads, optionally efficiency cores)
                                  4. give back the memory of experts nobody needs (LRU)
                                  5. use llama.cpp's own dot-product functions, so every number is identical
```

Each expert keeps its normal address inside one large reserved block of memory; moe-cache decides which experts have real memory behind them.
It plugs into **ggml**, not into llama.cpp's model code, so a model that llama.cpp runs through its expert multiplication should work; the architectures that were actually tested are listed in [COMPATIBILITY](docs/COMPATIBILITY.md) and [RESULTS](docs/RESULTS.md).

## Known limitations

- Long prompts (32+ tokens per batch) run their expert blocks on the GPU, as stock llama.cpp does, so prompt processing is at stock speed when the model fits (DeepSeek-V2-Lite: 166 to 183 tok/s vs 166 to 188 stock) and much faster when memory is short.
- It does not manage the GPU; which layers' experts sit in VRAM is chosen once at start by the planner. A GPU expert tier was evaluated and is not built, see [docs/ENGINE_DECISION.md](docs/ENGINE_DECISION.md).
- **Not every feature works with every llama.cpp build.** How the plugin is loaded depends on how llama.cpp was built (`moe-cache-server` picks it):

  | Feature | `LD_PRELOAD` (llama.cpp that links its backends directly: distro packages, your own CUDA build) | `GGML_BACKEND_PATH` (backends as separate files: official releases, `./install.sh`, Vulkan builds, Windows) |
  |---|---|---|
  | RAM expert cache, SSD streaming, memory limit, warm start | yes | yes |
  | Expert executor (computes ready experts while others load) | yes | yes (since 0.2.3.3; tested on a Vulkan build) |
  | Fast Q2_0 / IQ3_XXS kernels | yes | inside the executor only (Linux, and Windows x86-64 since 2026-10-28; no speed change on the PC, RESULTS 63 addendum) |
  | Working together with llama.cpp's GPU expert cache (`--gpu-cache`) | yes | **no** (refused since cffa6f9: it gave wrong tokens; the launcher uses static GPU expert layers instead) |

  The `moe-cache: active: ...` line in the server log says which parts are on.
- Needs a llama.cpp with shared ggml libraries; tested with commits `23b0202a1` (ggml 0.26.0) and `7fe450e19` (ggml 0.25.1), and release b11966 (ggml 0.28.0; CI row added 2026-10-30). The plugin refuses to load into a ggml with a different backend API than it was built for (it prints why and the model loads normally). Linux; Windows is experimental (identity-tested on a Windows 11 + RX 9070 XT desktop and in CI, see [COMPATIBILITY](docs/COMPATIBILITY.md) and the [Windows quick start](docs/WINDOWS.md)).

## Known issues in v0.3.3 (honest list)

| Issue | Detail |
|---|---|
| Not byte-identical at 128k+ by default | at `--ctx 131072` and above the KV cache is q8_0 (`--fast=kv`, about 2x prompt speed at 256k on a 16 GB card, from the bigger GPU plan the smaller KV allows): it passed the quality gate at 4k context (KL 0.0079, top-1 96.7 %, tasks = stock), but not the same bytes as stock. `--exact` keeps llama.cpp's f16 KV |
| Windows zip is unsigned | Windows shows "unknown publisher" / SmartScreen: Unblock the zip, More info > Run anyway (steps above). Check the SHA256 or `gh attestation verify` |
| Windows tested on one desktop | Ryzen 9 7950X + RX 9070 XT 16 GB + 32 GB, Windows 11, Vulkan, plus a GitHub Windows runner (CPU). NVIDIA, Intel GPUs and Windows 10 are untested; HIP loads but is slower than Vulkan |
| Models measured | Qwen3.8-Flash-Next (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, Coder IQ1_M), Qwen3.5-122B IQ2_XXS, GLM-5.3-Flash, Qwen3.6-35B, Nemotron 3.5 Bolt 74B (agent speed; plugin forced slower than stock in one test), Granite 1B (CI). Other MoE models should work but are not measured. Long context (64k-256k) is measured on Flash-Next and a few more, single runs |
| Linux | Tested on Intel + NVIDIA, CPU-only, Vulkan, and a user's RX 9070 XT. `install.sh` has been tried on Arch (`--no-apt`) and, for this release, from a clean clone on that machine; a fresh Ubuntu/Debian run with apt is not tested. macOS: no |
| Not tested for this release | `winget install` and `scoop install` (manifests exist but are not submitted); Hermes Desktop and OpenCode Desktop on Windows (click-through still open); a full model run from the final zip |
| Speed numbers | One PC, cold runs, mostly single series; yours depends on RAM, SSD and GPU. The Linux table is older (v0.2.3.5) |

## FAQ

**Do I need this if my model fits in RAM?** No. Use stock llama.cpp with `--moe-cache-mib` (see "When to use something else" above). Caveat: `--moe-cache-mib` is llama.cpp's own GPU expert cache; on our Vulkan PC it was slower (RESULTS 59) and its output can differ slightly from stock, so test it on your card. The gain is when memory is short or the model is larger than your RAM. _work in progress though, I'm currently seeing different ways to improve speed and efficiency_

**Do you check answer quality?** Yes. How, in one page: [docs/EXACTNESS.md](docs/EXACTNESS.md). By default moe-cache is exact below 128k context (at `--ctx 131072` and above the KV cache is q8_0 by default, which passed the quality gate at 4k context; `--exact` keeps f16): with the same layer split the tokens are identical to stock llama.cpp (`tests/identity.sh`, CI on Linux and Windows). The approximate modes (`--fast`, `--fast=kv`, `--fast=fewer`, `--hybrid`) must pass a [quality gate](docs/QUALITY_GATE.md) whose limits were fixed before measuring: KL divergence and same-top-token rate against stock, the 202 exact-answer tasks, and a loop count. Limits: `kv` KL at most 0.015, top-1 at least 95 %, tasks within 2 of stock; `fast` 0.040 / 90 % / 4; `fewer` 0.060 / 88 % / 5. Measured on Flash-Next Q2_0: `kv` KL 0.0079, top-1 96.7 %, 190 tasks = stock; `fewer` KL 0.034, 192 vs 190, unproven at 64k-256k. If answers look wrong or loop on your machine, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

**What cache policy does it use?** LRU at expert granularity; the experts of the operation that is running are protected. We replayed 13,464 real decode layer-steps of gpt-oss-120b at the real cache size: LRU 84.2 % hits, 2Q 83.6 %, LFU 84.1-84.2 %, LRU with the hottest experts pinned from a profile 84.3-84.8 %, the offline optimum 92.5 %. No policy beat LRU by more than 0.6 points, so LRU stayed (RESULTS section 12, "Smarter eviction").

**Is a very long prompt still slow?** Yes. Prompt reading runs on the GPU and was 1.8x faster than stock at 63k on our Windows PC, but a 63k-token prompt still took about 290 s (RESULTS 64).

**Why is it not faster when my model already fits in RAM?** Because there is nothing to fix: stock llama.cpp then runs from the page cache with the same kernels. On Qwen3.6-35B at 24 GB, decode needs about 0.5 GB of expert weights per token (8 experts x 32 CPU layers x 1.9 MB), which the memory bus delivers at about 35 GB/s, close to this laptop's measured maximum (40 GB/s), so it is bandwidth-bound for everyone (stock 40, moe-cache 40 tok/s). The speed-ups appear when memory is short (the same model at 12 GB: 10 to 23 tok/s), for models larger than RAM, and for models with slow kernels (Q2_0 and the IQ types, where the executor and the faster kernel help).

**Does it use my GPU?** Yes, through llama.cpp as usual. `moe-cache-server` picks how many layers' experts live on the GPU from your free VRAM (that alone is worth about +60 % over keeping everything on the CPU). moe-cache itself manages the experts that stay on the CPU side.

**Is the output really the same as stock?** Yes by default: `tests/identity.sh` compares the tokens of stock llama-server and llama-server with the plugin (temperature 0, seed 42, same command line) and every model in [RESULTS](docs/RESULTS.md) passed it (Hunyuan only a short test). It runs in CI on Linux and Windows. Only `--fast`, `--fast=kv`, `--fast=fewer`, `--fast=2` and `--hybrid` change the output, and only when you ask for them.

**Why not just let the OS page cache / mmap handle it?** That is what stock llama.cpp does, and it is the slow case in the table: when the model does not fit, missing experts come in through tens of thousands of small page faults per token, and the kernel evicts pages without knowing which experts matter. moe-cache reads the same bytes with a few hundred large parallel direct reads and keeps whole experts (gpt-oss-120b: about 79,000 vs 770 reads per token for the same 325 MB, RESULTS 12b). `--no-mmap` or `--mlock` need the whole model in RAM, so they do not help a model bigger than RAM.

**How does it compare with ktransformers, ik_llama.cpp or PowerInfer?** Not measured. Those are separate engines (forks or rewrites with their own kernels and model support); moe-cache is a plugin for the unmodified upstream llama-server and keeps its exact tokens. Against llama.cpp's own `--n-cpu-moe` / `-ot` offload: moe-cache works together with it (every row above uses the same `--n-cpu-moe` on both sides).

**Does it work with Ollama or LM Studio?** Not directly: both ship their own bundled copy of llama.cpp, and the plugin needs a `llama-server` built with shared ggml libraries from a tested commit. Their model files are plain GGUF, so you can run the same file with `moe-cache-server` (see "Ollama user?" above) and point any OpenAI-compatible client at `http://localhost:8080/v1`.

**On Windows llama-server uses a lot of RAM with GPU layers, should I add `--no-mmap`, `--mlock` (newer llama.cpp: `--load-mode none` / `mlock`) or `--moe-cache-mib`?** No. llama.cpp's Windows loader leaves the model-file pages of the GPU tensors in llama-server's working set (about as much RAM as the GPU layers; RESULTS 56 I37). It is not the GPU driver. The plugin releases those pages at the first graph (the log says "released N GiB of model-file pages from the working set"). `--moe-cache-mib` is llama.cpp's GPU cache, not this plugin's. Keep the page file system-managed: the cache costs no commit, but Vulkan's GPU memory does. Details: [docs/WINDOWS.md](docs/WINDOWS.md), "Windows: where the RAM goes".

**Mac?** No. The plugin uses Linux (and Windows) memory and direct-I/O calls, and on Apple Silicon the GPU and CPU share one memory, which changes the problem.

**Does it wear out my SSD?** It only reads the model file; reads do not wear an SSD. It writes small files: the expert profile, and in `--agent` mode the saved conversations (`--no-warm` turns that off).

**Is there a GUI?** Yes: the control page (`bin/moe-cache-server --ui`, see [docs/GUI.md](docs/GUI.md)), plus llama-server's own chat page (`--open`). Any OpenAI-compatible app such as [Open WebUI](https://github.com/open-webui/open-webui) can use `http://localhost:8080/v1` (we have not tested Open WebUI ourselves).

**Why not an own inference engine like Strata or Maya?** Short answer: they ship their own engine per model family; moe-cache is a small layer for the MoE models stock llama.cpp runs (tested list in COMPATIBILITY). The reasoning and our measurements: [docs/ENGINE_DECISION.md](docs/ENGINE_DECISION.md). A side-by-side with what each project says about itself today follows.

### Strata and Maya today (as of 2026-10-29)

Read from their public release notes and README pages only; we did not clone, build or run either one, so every number in the Strata and Maya columns is **their own, not ours**. Both are MIT-licensed. Both change fast (Strata has had about one release a day), so check their pages.

| | Strata ([Niko1221/Strata](https://github.com/Niko1221/Strata), 0.2.7.1 of 2026-10-27) | Project Maya ([mw00/project-maya](https://github.com/mw00/project-maya)) | moe-cache |
|---|---|---|---|
| What it is | an app, a terminal tool and an engine per hardware family (CUDA, HIP, SYCL, Metal, DGX Spark), with its own kernels and parts of ggml | started from Strata's code, rewritten per model; engine, server and dashboard | a plugin for stock, upstream llama.cpp; the server is llama-server |
| Models | Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, and since 0.2.4 DeepSeek-V4.1-Flash (2.00 bpw and similar) | GLM-5.3-Flash, DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash (their own quantizations) | any MoE model llama.cpp runs; tested list in COMPATIBILITY |
| Bigger than RAM | yes for DeepSeek: the part that does not fit is read from SSD (about 265 GB of disk at 2.00 bpw). Their number: RTX 3060 12 GB + 46 GB RAM, 9-11 tok/s, 155 tok/s prompt at 4K. For the Qwen models we read in October, all experts were pinned in RAM; we have not re-checked | yes: GPU, RAM and NVMe tiers; 32 GB RAM minimum | yes: the RAM cache plus direct SSD reads is the point of the project |
| GPUs | NVIDIA, AMD (HIP), Intel Arc (SYCL), Apple (Metal) | NVIDIA compute capability 7.0+; AMD and Windows are marked experimental | whatever llama.cpp supports: our measurements are Vulkan (AMD) and CUDA (NVIDIA); the plugin itself does not depend on the GPU backend |
| Their reported speeds | RTX 5070 12 GB: Qwen3.8-Flash-Next Q2_0 94 tok/s decode, 2,650 tok/s prompt (64 GB RAM); R9700: Qwen3.6-35B UD-Q4_K_XL 117.8 tok/s (0.1.45) | one RTX 5090 + 128 GB RAM, GLM-5.3: 24-26 tok/s up to 512K context (a user's write-up); no result for 16 GB VRAM + 32 GB RAM is published | see the tables above; PC with a 16 GB card and 32 GB RAM, Flash-Next Q2_0 (66.4 GB file): 22.2 tok/s decode (model table above) |
| Exact output | their "Faster" mode (opt-in on some, default on cards of 16 GB or less for DeepSeek) uses 5 of 6 experts per token; KL not given | not stated | exact by default (output compared byte for byte with stock llama.cpp in CI); the approximate options (`--fast`, `--hybrid`) are opt-in and labelled |
| Long context | their 64K figure is for Strix Halo (749 tok/s prompt) and DGX Spark; AMD 131K / 230K / 500K prompts were read 8.7 / 16 / 79 % faster in 0.1.45 | up to 1M context on GLM-5.3 and MiniMax on big GPUs (H100, RTX PRO 6000); KV cache streams from RAM | 64k-256k with agents on 32 GB RAM: the long-context table above (Windows PC, RESULTS 64) |
| Windows and Linux | both | Linux; Windows experimental | both; Windows is newer and has fewer runs |

Where they are the better choice: you have a card and RAM they list, and you want their model at the highest speed; their installers are polished. Where moe-cache fits: you want to stay on stock llama.cpp, you use a MoE model or a quantization they do not ship, you want output identical to stock, or you have 32 GB of RAM and a model bigger than that.

## Documentation

[Quick start](docs/QUICKSTART.md) | [Coding agents](docs/AGENTS.md) | [Control page](docs/GUI.md) | [Benchmarks](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark) | [Results](docs/RESULTS.md) | [Compatibility](docs/COMPATIBILITY.md) | [Roadmap](docs/ROADMAP.md) | [Engine decision](docs/ENGINE_DECISION.md) | [Security and privacy](docs/SECURITY.md)

## Configuration (environment variables; `moe-cache-server` sets them for you)

| Variable | Meaning |
|---|---|
| `MOE_CACHE_SIZE_GIB` | cache size in GiB, or `auto` (size from the memory limit) |
| `MOE_CACHE_RAM_GIB`, `MOE_CACHE_MARGIN_GIB` | override the detected memory limit; memory to keep free (default 1 GiB) |
| `MOE_CACHE_GGUF` | model file (split models: the first part) |
| `MOE_CACHE_PROFILE` / `MOE_CACHE_WARM` | learn expert usage here and warm-start from it / warm start only |
| `MOE_CACHE_STATS=1` | print hit rate, reads and evictions at exit |
| `MOE_CACHE_EXEC_FUSE=0` | turn off the fused gate+up executor job (on by default since 0.3.2: decode +0.3-2.6 % in every PC pair, same text; RESULTS 73) |
| `MOE_CACHE_IDLE_REBALANCE=0` | turn off the idle cache refill (on since 0.3.3: refills the RAM cache while an agent pauses; every PC pair won) |
| `MOE_CACHE_POLICY=classlru` | frequency-aware cache replacement instead of LRU (opt-in; on the miss-heavy 122B +1.3-2.9 % decode in every pair, on Flash-Next within noise; RESULTS 73) |
| `MOE_CACHE_STATS_SIGNAL=1` | print one `moe-cache-stats {json}` line of the running counters on request: Linux `kill -USR1 <pid>`, Windows set the named event `Local\moe-cache-stats-<pid>` |
| `MOE_CACHE_DISABLE=1` | load nothing: the server behaves exactly like stock |
| `MOE_CACHE_EXEC=0` | switch the expert executor off (default on) |
| `MOE_CACHE_EXEC_ECORES=N` | let the executor use N efficiency cores as well (the launcher sets up to 4 on hybrid Intel CPUs; `--no-ecores` to disable) |
| `MOE_CACHE_EXEC_THREADS=N`, `MOE_CACHE_EXEC_READERS=N` | compute threads (default: what llama.cpp asks for) and background reader threads (default 4) |
| `MOE_CACHE_EXEC_CHECK=1` | recompute every expert op the stock way and compare (slow; for testing) |
| `MOE_CACHE_FAST_KERNELS=0` | use llama.cpp's own Q2_0 kernel instead of the plugin's |
| `MOE_CACHE_IQ3_KERNEL=1` / `=0` | use the plugin's IQ3_XXS gather kernel (bit-identical) on any AVX2 CPU / turn it off; off by default on AMD, where gathers are slow (Zen 4: about 0.6x llama.cpp's own function) |
| `MOE_CACHE_ROUTE=skip` (`moe-cache-server --fast`; `--fast` also sets a q8_0 KV cache: KL 0.0045 alone on the laptop (CUDA), **0.0120 on the PC (Vulkan)**, `--fast` as shipped 0.0145 there; kept in `--fast` by the owner's choice, RESULTS 61 R11) | opt-in approximate routing: skip a low-ranked expert that is not in RAM. About +10-12 % when the model is much bigger than RAM; about 1 top token in 40-50 differs from stock (KL 0.005-0.012; RESULTS section 31). Off by default |
| `MOE_CACHE_HYBRID=MiB\|auto` (`moe-cache-server --hybrid`) | opt-in, experimental: a bank of the most used experts (from the usage profile) in free GPU memory computes those picks while the CPU computes the rest. Flash-Next on 8 GB: 13.1 -> 17.4 tok/s; KL 0.0043 vs exact (about 1 top token in 60 differs). Needs a usage profile (the first session has none). The bank adapts to the session (`MOE_CACHE_HYBRID_ADAPT=N,K`, default 8,16; 0 = keep the profile choice). For models whose experts are slow on the CPU (IQ / Q2_0 quants); with Q4_K-class models such as Qwen3.6 it is slower than `--gpu-cache` (8.7 vs 15.7 tok/s at 12 GB). Off by default |
| `moe-cache-server --fast=kv` | **the default at `--ctx 131072` and above** (`--exact` keeps the f16 KV); otherwise opt-in: only the q8_0 KV cache of `--fast`, every expert computed. Passed the quality gate at 4k context (`docs/QUALITY_GATE.md`, level kv: KL 0.0079, top-1 96.7 %, tasks 190 = stock on Flash-Next Q2_0). At ctx 262144 the freed KV buys more GPU layers and `-ub 1024`: prefill 2x, decode +11-13 % at 64k (RESULTS 72). Off by default |
| `MOE_CACHE_FEWER=N` (`moe-cache-server --fast=fewer` sets 2 and `--fast`) | opt-in approximate: on the CPU-held layers, decode and small batches never compute the N lowest-ranked experts of a token (at least half are kept). Passes the gate (level fewer: KL 0.034, top-1 92.0 %, tasks 192 vs 190); faster at 4k (+2-6 %), **unproven at 64k-256k** (decode pairs lost their values). Off by default |
| `MOE_CACHE_ROUTE_DROP=N`, `MOE_CACHE_ROUTE_KEEP=N` | with `ROUTE=skip`: how many of the lowest-ranked picks may be skipped (default 2; `--fast=2` sets 4, with at least half of the picks always kept) |
| `MOE_CACHE_COALESCE_MIB=N` | merge neighbouring expert reads of a long prompt into reads of up to N MiB (default 16; 0 = off). Flash-Next prefill +14-24 % |
| `MOE_CACHE_SCAN_TOKENS=N` | prompt batches of N+ tokens load experts at the evict-first end, so the cache from before a long prompt survives it (default 512; 0 = off). Flash-Next prefill +27 % |
| starter profiles (`moe-cache-server --no-starter` to skip) | with no usage profile yet, the launcher starts from the shipped one for the model family (`profiles/`): first request 2.1x faster on Flash-Next (RESULTS 53) |
| `MOE_CACHE_GPU_PACK=0` | turn off packing: by default a long-prompt block uploads only the experts its tokens use (exact; 70-token prompts 1.8x faster on Qwen3.6, RESULTS 52) |
| `MOE_CACHE_HOST_RESERVE_MIB` | memory kept free for llama-server's prompt cache (`--cache-ram`); the launcher sets it |
| `MOE_CACHE_GUARD=1` | tests only: reading an evicted expert crashes instead of returning zeros (small models only) |

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
