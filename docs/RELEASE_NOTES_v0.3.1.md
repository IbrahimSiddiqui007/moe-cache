moe-cache v0.3.1 runs MoE models bigger than your RAM through llama.cpp's `llama-server`. The output is identical to stock llama.cpp. **On Windows it is faster than stock** on every model we measured; the **Windows kit** is experimental.

**v0.3.1 fixes the Windows kit of v0.3.0**: the zip left out one launcher file (`moe_cache_pick.py`), so the control page (`moe-cache.bat`), `pick`, and `server` / `on-demand` without a model stopped with "No module named moe_cache_pick". The kit now ships every launcher module, its build fails if one does not import, and Windows CI starts the control page and runs `pick` from the unzipped kit. Also in v0.3.1: llama-server is found the same way by every command (and `on-demand` refuses a missing one at setup instead of failing at the first agent request), clearer messages when llama-server or the matching plugin is missing, settings written safely so the picker and the control page keep each other's changes, the prompt-cache size shown for agents, `connect claude` for Claude Code, and docs fixes (CHANGELOG). The Linux tarballs of v0.3.0 were not affected by the kit problem. Everything below is what v0.3.0 brought.

## Downloads

| File | For |
|---|---|
| `moe-cache-v0.3.1-windows-x64.zip` | Windows 10/11 x64 with the official llama.cpp release (Vulkan, CPU; tested b11669). Unzip, double-click `moe-cache.bat`. Read `README_WINDOWS.txt` first |
| `moe-cache-v0.3.1-linux-x86_64.tar.gz` | Linux x86-64, llama.cpp 7fe450e19 (ggml 0.25) |
| `moe-cache-v0.3.1-ggml0.26-linux-x86_64.tar.gz` | Linux x86-64, llama.cpp 23b0202a1 or newer (ggml 0.26 / 0.27) |

Check a download: `gh attestation verify <file> -R IbrahimSiddiqui007/moe-cache` (`SHA256SUMS` and `SHA256SUMS-windows` only detect corruption).

## Windows: "unknown publisher"

The kit is not code-signed (a certificate costs money every year; the code is open source and the zip is built by GitHub Actions from the tagged commit), so Windows may warn. Before you extract the zip: right-click it > **Properties** > tick **Unblock** > **OK**. If SmartScreen says "Windows protected your PC": **More info** > **Run anyway**. Check that your zip is the one CI built: `Get-FileHash moe-cache-v0.3.1-windows-x64.zip` in PowerShell must equal its line in `SHA256SUMS-windows` (or `gh attestation verify` as above).

## Windows quick start

| Step | What to do |
|---|---|
| 1 | Unpack llama.cpp's `llama-bNNNNN-bin-win-vulkan-x64.zip` to `C:\llama` |
| 2 | Install Python 3.8 or newer (tick "Add python.exe to PATH") |
| 3 | Unzip the kit and double-click `moe-cache.bat`; pick a model on the page and press Start |
| 4 | Coding agents: start with `--agent`, then `python bin\moe_cache.py connect opencode --write` (or `hermes`). Or `python bin\moe_cache.py on-demand MODEL.gguf`: the agents start the model when they need it |

## Speed on Windows

One desktop: Ryzen 9 7950X, 32 GB DDR5, RX 9070 XT 16 GB, Windows 11, llama.cpp Vulkan (ggml 0.27). Every run cold (Windows' file cache emptied first), stock and moe-cache in the order stock, moe-cache, moe-cache, stock, launcher defaults, a 256-token greedy reply. Details: RESULTS 59-63 in [docs/RESULTS.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/main/docs/RESULTS.md).

| Model (file size; all bigger than the 32 GB RAM) | stock llama.cpp | moe-cache | |
|---|---|---|---|
| **Qwen3.8-Flash-Next Q2_0 (66 GB), the recommended file** | 18.4 tok/s; prompts 121 tok/s | **22.2 tok/s; prompts 281 tok/s** | +20 %; prompts 2.33x |
| Qwen3.8-Flash-Next IQ2_XS (68 GB) | 14.9 | **21.9** | +47 % |
| Qwen3.8-Flash-Next IQ3_XXS (76 GB) | 15.1 | **19.9** | +31 % |
| Qwen3.8-Flash-Next IQ3_S (84 GB) | 12.9 | **17.3** | +34 % |
| Qwen3.5-122B UD-IQ2_XXS (37 GB) | 15.9 | **19.7** | +24 % |
| GLM-5.3-Flash UD-IQ2_XXS (102 GB) | 2.2 | **4.4**; **6.9 with `--mtp`** | 2.0x; 3.1x |

The text is byte-identical to stock on every model checked (Flash-Next IQ2_XS and IQ3_S, the 122B, GLM, also with MTP).

**Which Flash-Next file**: **Q2_0** for speed (as fast as IQ2_XS or faster everywhere, prompts 11-19 % faster, 191 of 202 test tasks right), IQ3_XXS when accuracy matters (193/202, ~10 % slower decode), the **Coder IQ1_M** file for coding agents (an OpenCode session 55.5 s instead of 73.4 s). IQ3_S is not worth it. **GLM-5.3-Flash**: put the model on your fastest drive (+4-13 %) and keep the launcher's default layers; `--mtp` helps chat (+10-23 %) but not code.

**Coding agents** (recorded OpenCode and Hermes Agent sessions replayed against the server, Flash-Next IQ2_XS):

| Session | stock | moe-cache | |
|---|---|---|---|
| OpenCode (5 requests) | 175 s | **73 s** | 2.4x |
| Hermes Agent (7 requests) | 164 s | **82 s** | 2.0x |
| both at once on one server | 336 s | **145 s** | 2.3x |
| OpenCode on the Coder IQ1_M file | | **55.5 s** | |

The gain is in the prompts: the first token comes about 3x sooner, and so does the first token of each follow-up turn. A model that fits in RAM (Qwen3.6-35B) gets 5-9 % shorter sessions; `--mtp` raises its decode from 43.5 to 72 tok/s.

These numbers come from one PC. Your speed depends on RAM, the SSD and the GPU. If the model fits in RAM, the launcher runs stock llama.cpp (that is as fast or faster then).

## What is new

| Area | Change |
|---|---|
| Windows speed | Windows counted the cache's private memory, and Vulkan's GPU memory, against RAM + page file, which held the cache at 8-11 GiB and resized it hundreds of times per request. The cache now keeps experts in **read-only views of the model file**, which cost nothing there; a hard working-set minimum keeps them in RAM. Same cache logic as Linux. `MOE_CACHE_WIN_VIEW=0` goes back to the old way |
| Windows memory | No Windows job memory limit with the file view (it only capped llama.cpp's GPU memory: `--ram 16` failed to load, now it runs at 92 % of the 24 GB speed). Another program that needs 8-14 GB while the server runs gets it in 1-4 s, no freeze |
| Idle RAM | `on-demand --idle-release` (default 5 min): while the agents are quiet the plugin gives the cache's RAM back (PC: 6.8 -> 25 GB free) and the server keeps running; the next request pays ~3 s instead of a ~40 s restart |
| Windows fixes | A program with the plugin could hang at exit and could not be killed (seen with llama-perplexity): fixed. Wrong output and a crash at the second request with models bigger than RAM (Qwen3.5-122B) fixed. The plugin releases the model-file pages llama.cpp's Windows loader leaves in memory (about 13 GB with 12 GPU expert layers). No false "plugin disabled itself" warning for a user without the working-set privilege |
| Launcher | **Starts about 6x sooner**: the model's header is read once and fast (a Flash-Next dry run 17 s -> 2-3 s on the PC; every on-demand model start gained the same). on-demand opens its port at once (was 16.5 s; agents that connect early wait instead of being refused). `--gpu-cache` on Vulkan is capped to what the driver allows (it aborted the load). A micro-batch you choose (`--ub N`) now goes through the tuner, so long prompts keep their GPU block: 1.5-2.3x faster prompts at 34k tokens (the default `--ub auto` was not affected). `--ram` is GB everywhere. AMD and Intel GPUs are found on Windows |
| `check` | Measures the model's drive (4 s of random reads past the file cache) and warns below 1.0 GB/s (SATA, hard disk, USB) |
| Agents | `--agent` (65k context, tool calls, warm sessions after a restart), `connect opencode\|hermes --write`. **`on-demand`**: an always-on listener that starts the model at the first request and stops it after 15 idle minutes, with **several models** (`--add`; the request's model picks one, one runs at a time) and start at login |
| Control page | `moe-cache.bat` (Windows) or `moe-cache-server --ui`: a **three-pane** page: chats, agent sessions and models on the left, its own chat in the middle, live speed, cache, RAM/VRAM, warnings and the log on the right. It says why it runs stock or moe-cache, and the Quick benchmark compares both on your PC in one click |
| Speed options | Flash-Next Q2_0 is the recommended file (above). `--mtp` on GLM-5.3-Flash now works with the cache (3.1x stock on chat). `--fast=2` for memory far below the model size; starter usage profiles for Flash-Next, GLM-5.3-Flash, Qwen3.5-122B and Qwen3.6 |
| Executor kernels | The plugin's Q2_0 and IQ3_XXS executor kernels now run on Windows x86-64 too, bit-identical. The Q2_0 one matches llama.cpp's own AVX2 function (#30447) bit for bit (same rule on both systems: at start the plugin checks llama.cpp's Q2_0 function bit for bit and uses the matching kernel, or leaves it alone). No speed change at default settings (bandwidth-bound) |
| Quality | 202 exact-answer tasks (`tools/quality_tasks.json`): Flash-Next IQ3_XXS 193 right, IQ3_S 192, Q2_0 191, Coder IQ1_M 189, IQ2_XS 187 (differences not statistically proven) |
| AMD Vulkan decode | The plugin reads its split's inputs from the GPU asynchronously: decode +12 % (Flash-Next Q2_0, 22.7 -> 25.6 tok/s after a 34k prompt; every ABBA pair, text identical). `MOE_CACHE_ASYNC_IN=0` goes back |
| Long context | 64k-256k: the tuner sizes the KV cache from the model header, and past the measured contexts scales the compute buffer instead of a flat guess; the launcher says when `--fast` would fit more layers. Flash-Next Q2_0: 64k prompt 1.8x stock |
| Memory cap | The whole server stays at or below physical RAM minus max(4 GiB, 12.5 %) on Linux and Windows (`--max-ram GB` changes it); kept by planning, never by stopping a run |
| Linux | The expert cache lives in read-only mappings of the model file too (as on Windows): steady decode +4.7 %, first request +49 %, tokens identical. `MOE_CACHE_LINUX_VIEW=0` goes back |
| Sampling | The launcher passes the model card's sampling settings for Qwen3/3.5/3.6/3.8 and GLM; `--no-sampling-defaults` turns it off |
| Launcher | `--version`; finds `C:\llama\llama-server.exe` (Windows) or install.sh's build (Linux) without `--llama-server` |
| **Model picker** | `moe_cache.py pick` and the control page read your machine (RAM, VRAM, free disk, SSD or not, CPU) and ask what it is for (agents and coding, chat, long documents), then recommend one model, file, context and flags and can use it (`--apply`; `connect` picks it up). It warns but never blocks, its memory estimate includes the chosen context, and it says "untested" where we have no measurement. README: "Which model for my machine?" |
| Agent setup | `connect hermes\|opencode --write` keeps a `.bak` of your own config (also a local Ollama or LM Studio setup) and prints what it replaced; the control page reads back `opencode.jsonc`. Tested end to end on Windows and Linux with Hermes and OpenCode (two-turn coding tasks; follow-up turns reuse the prompt and process only the new tokens) |
| Stock runs | A model that fits in RAM + VRAM (e.g. Nemotron 3.5 Bolt) runs stock llama-server without needing the plugin build, and short requests stay on the CPU when expert layers are there (`GGML_OP_OFFLOAD_MIN_BATCH=512`): a tiny request 1.32 -> 0.6 s, answers identical |
| Fixes from a full code audit | The tuner reserved too little GPU memory for `-ub 2048` at 128k+ contexts (long prompts lost their GPU block); the control page's port scan could send its API key to another local server; the on-demand listener read request bodies before checking the key; Linux idle release did nothing (now 1656 -> 967 MiB); a second Ctrl+C during a prompt could hang the exit; two NVIDIA GPUs were treated as none; the release job's Windows kit build is locked down like the Linux one; about a dozen smaller ones (CHANGELOG) |
| CI | The Windows runner unzips the kit and runs `check --run` and `on-demand` from it, as a user would. Releases are published only after identity and Windows CI passed on the tagged commit |

Full list: [CHANGELOG.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/main/CHANGELOG.md).

## Known limits

| Limit | Note |
|---|---|
| Windows is experimental | measured on one desktop (Ryzen + RX 9070 XT) and tested on a GitHub Windows runner (CPU); NVIDIA and Intel GPUs on Windows are untested |
| Page file | keep it system-managed or 14 GB or more: the cache costs nothing there now, but Vulkan's GPU memory does |
| Working-set privilege | the hard minimum that keeps experts in RAM needs SeIncreaseWorkingSetPrivilege (a normal desktop login has it); without it the cache still runs at full speed, but Windows may trim it under memory pressure |
| Flash-Next on AMD, driver older than 32.0.32015.7029 | context checkpoints are off (the driver crashes); update Adrenalin |
| Replacing llama.cpp's own CPU functions outside the executor, `--gpu-cache` | Linux only (they need `LD_PRELOAD`); the executor's Q2_0 and IQ3_XXS kernels run on Windows too |
| HIP / ROCm on Windows | loads and gives the same answers, but is about 2.4x slower than Vulkan on the RX 9070 XT: use the Vulkan build |
| Nemotron 3.5 Bolt | needs llama.cpp b11966 or newer; the picker says so when your llama-server is older |

## Known issues

| Issue | Note |
|---|---|
| Unsigned Windows zip | see "unknown publisher" above |
| Tested hardware | Windows: one desktop plus a CPU-only GitHub runner. Linux: Intel + NVIDIA, CPU-only, Vulkan; a user's RX 9070 XT. macOS no |
| Models measured | Qwen3.8-Flash-Next (5 files), Qwen3.5-122B, GLM-5.3-Flash, Qwen3.6-35B, Nemotron 3.5 Bolt 74B (agents), Granite 1B (CI). Others untested |
| Not tested | winget / scoop install (manifests are in the repo, not submitted); Hermes / OpenCode Desktop on Windows; fresh Ubuntu `install.sh` with apt; Windows 10 |
| Speed numbers | one PC, cold runs, mostly single series |
