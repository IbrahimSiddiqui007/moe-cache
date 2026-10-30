# Install, run and measure, in detail

The README has the short path; this page keeps every option. Moved here from the README on 2026-10-30, unchanged.

## Get started

**Windows 11?** Go to the [Windows quick start](#windows-quick-start-experimental) below (7 steps, [docs/WINDOWS.md](WINDOWS.md) has the details). **Linux?** Read on.

**You need:** a `llama-server` from llama.cpp's official downloads (the Windows `.zip` and the Linux `llama-bNNNNN-bin-ubuntu-*.tar.gz` files are built with the shared ggml libraries moe-cache needs; we tested the Windows ones, not the Linux ones) or your own build with `-DBUILD_SHARED_LIBS=ON`; Python 3.8 or newer; and a MoE model as a `.gguf` file (step 2 below). A GPU is optional. Tested llama.cpp: b11669 and b11966 (Windows); commits `23b0202a1` (ggml 0.26.0, backend API 3) and `7fe450e19` (ggml 0.25.1, API 2) (Linux), each Linux API needs its own plugin build, see [docs/QUICKSTART.md](QUICKSTART.md). Other versions may work; `check` tells you.
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

**Windows (experimental)**: steps below and in [docs/WINDOWS.md](WINDOWS.md).
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

**The first start is slow.** The server is ready after about 20 seconds (RESULTS 64: 14-19 s); the first answer to a long prompt can take minutes on a model bigger than your RAM (a cold first turn of 7,461 tokens took 186 s on our laptop, [docs/AGENTS.md](AGENTS.md); a 63k-token prompt about 5 minutes on our Windows PC, RESULTS 64). The chat page shows progress. Later turns are much faster because the conversation is kept.

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
context, mode, MTP). "Quick benchmark" runs a short fixed request and, in one click, the same on stock llama-server for comparison. See [docs/GUI.md](GUI.md).

**Coding agents (Hermes Agent, OpenCode).** They only talk to an OpenAI-compatible address; moe-cache provides it. Easiest, Ollama-style: a small
listener that loads the model on the first request and unloads it when idle (`--install-autostart` starts it at every login):
```bash
python3 bin/moe_cache.py on-demand /path/to/model.gguf --llama-server /path/to/llama-server --install-autostart
python3 bin/moe_cache.py connect hermes --write      # and/or: connect opencode --write (writes URL, model, 65k context, API key); connect claude prints the lines for Claude Code
```
**Agent context limit.** Keep the agent's own context limit below the server's window (65536 by default); the reason and the numbers are in [docs/AGENTS.md](AGENTS.md#keep-the-agents-own-context-limit-below-the-servers).

Several models: `on-demand /path/to/model2.gguf --add` adds one; the request's `"model"` picks it, one is loaded at a time (`connect opencode` lists them all).
Or start a server yourself with `bin/moe-cache-server /path/to/model.gguf --agent` and run the same `connect` line. Measured on a Windows desktop
(RESULTS 60, recorded sessions replayed, cold, stock vs moe-cache): Flash-Next (68 GB on 32 GB RAM) OpenCode session 175 -> 73 s, Hermes 164 -> 82 s,
two agents at once 335 -> 145 s; Qwen3.6-35B (fits in RAM) 5-9 % faster. Details and the Desktop-app steps: [docs/AGENTS.md](AGENTS.md).

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

Keep the Windows page file at "system managed" (or 14 GB or more): the cache costs no commit (it keeps experts in read-only views of the model file), but Windows counts the GPU's memory against RAM + page file. Details and known limits: [docs/WINDOWS.md](WINDOWS.md).

### 5. Optional: prove it is correct and measure it on your machine

Linux (from the tarball or a checkout; the Windows kit has `tools\windows_test.ps1` instead, see docs/WINDOWS_TEST.md):

```bash
tests/identity.sh /path/to/model.gguf --llama-server /path/to/llama-server      # PASS = the same tokens as stock llama.cpp
```

More detail, troubleshooting and the Vulkan build: [docs/QUICKSTART.md](QUICKSTART.md).

## Benchmark it on your own model and your own use case

**Quick, on Windows or Linux:** `python bin/moe_cache.py bench-report MODEL.gguf` runs stock llama.cpp and moe-cache (stock, moe-cache, moe-cache, stock)
on a model bigger than your RAM and writes a report to paste into a [Benchmark report issue](https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=benchmark_report.yml).
It only measures, and the report holds no prompts, answers or names. Recipe: [docs/SHARE_YOUR_NUMBERS.md](SHARE_YOUR_NUMBERS.md). Reports from other machines are very welcome.

The full benchmark tool (Linux) (web page, standard protocol, many-model matrix, quality harness) now lives in its own repository: [moe-cache-benchmark](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark).
Clone it next to this folder (or set `MOE_CACHE_DIR` for the benchmark tool) and it finds the plugin and the planner by itself.
