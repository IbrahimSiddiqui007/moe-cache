moe-cache v0.2.3.6 runs MoE models bigger than your RAM through llama.cpp's `llama-server`. The output is identical to stock llama.cpp. This is the first release with a **Windows kit** (experimental).

## Downloads

| File | For |
|---|---|
| `moe-cache-v0.2.3.6-windows-x64.zip` | Windows 10/11 x64 with the official llama.cpp release (Vulkan, CPU; tested b11669). Unzip, double-click `moe-cache.bat`. Read `README_WINDOWS.txt` first |
| `moe-cache-v0.2.3.6-linux-x86_64.tar.gz` | Linux x86-64, llama.cpp 7fe450e19 (ggml 0.25) |
| `moe-cache-v0.2.3.6-ggml0.26-linux-x86_64.tar.gz` | Linux x86-64, llama.cpp 23b0202a1 or newer (ggml 0.26 / 0.27) |

Check a download: `gh attestation verify <file> -R IbrahimSiddiqui007/moe-cache` (`SHA256SUMS` and `SHA256SUMS-windows` only detect corruption).

## Windows quick start

| Step | What to do |
|---|---|
| 1 | Unpack llama.cpp's `llama-bNNNNN-bin-win-vulkan-x64.zip` to `C:\llama` |
| 2 | Install Python 3.8 or newer (tick "Add python.exe to PATH") |
| 3 | Unzip the kit and double-click `moe-cache.bat`; pick a model on the page and press Start |
| 4 | Coding agents: start with `--agent`, then `python bin\moe_cache.py connect opencode --write` (or `hermes`) |

## Speed on Windows (one desktop, 2026-10-27)

| Model | Stock llama.cpp | moe-cache |
|---|---|---|
| Qwen3.8-Flash-Next IQ2_XS (68 GB, bigger than RAM) | decode 15.0-15.2 / 19.2-19.4 tok/s (two prompts), prefill 108-110 tok/s (3.5k-token prompt); 12 GPU expert layers | default (12 GPU expert layers): decode 15.0-15.2 on both prompts, prefill 120. With `MOE_CACHE_WIN_BUFFERED=1` and 13 GPU expert layers (`--n-cpu-moe 35`): decode **18.5 / 19.4-19.6**, prefill **153** |
| Qwen3.6-35B-A3B (20 GB, fits in RAM) | 46.0; 68.6 with MTP | 45.6; 70.3 with MTP (plugin forced; the launcher runs stock for a model that fits) |
| Qwen3.5-122B UD-IQ2_XXS (37 GB) | 12.35-13.06 | wrong output and the crash at the second request are fixed; no speed claim yet (preliminary 12.89, not a clean run) |
| GLM-5.3-Flash UD-IQ2_XXS (102 GB) | 1.2-3.7 (older runs) | 2.02 default, 2.20 buffered (preliminary) |

Setup: Ryzen 9 7950X, 32 GB DDR5, RX 9070 XT 16 GB, Windows 11, llama.cpp 620d2225c Vulkan (ggml 0.27), AMD driver 32.0.32015.7029, 14 GB page file; cold starts, 256-token greedy requests (RESULTS 58). `MOE_CACHE_WIN_BUFFERED=1` (buffered expert reads) is opt-in for now.

These numbers come from one PC. Your speed depends on RAM, the SSD and the GPU. If the model fits in RAM, the launcher runs stock llama.cpp (that is faster then).

## What is new

| Area | Change |
|---|---|
| Windows | The kit (above). Fixed wrong output and a crash at the second request with models bigger than RAM (Qwen3.5-122B). The plugin releases the model-file pages that llama.cpp's Windows loader leaves in memory (about 13 GB with 12 GPU expert layers), so GPU expert layers help again. When Windows runs out of memory commit, the cache shrinks instead of failing |
| Launcher | `--ram` is GB everywhere (it reached the plugin as GiB, 7 % too much). It finds AMD and Intel GPUs on Windows. A llama-server that fails to start is reported as a failure |
| Agents | `--agent` (65k context, tool calls, warm sessions after a restart) and `connect opencode|hermes --write` |
| Control page | `moe-cache.bat` (Windows) or `moe-cache-server --ui`: models with the planner's verdict, start/stop, live speed, cache hit rate, RAM/VRAM, chat, Connect buttons |
| Speed options | `--fast=2` for memory far below the model size; starter usage profiles for Flash-Next, GLM-5.3-Flash, Qwen3.5-122B and Qwen3.6 |
| CI | The Windows runner unzips the kit and runs `check --run` from it, as a user would. Releases are published only after identity and Windows CI passed on the tagged commit |

Full list: [CHANGELOG.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/main/CHANGELOG.md).

## Known limits

| Limit | Note |
|---|---|
| Windows is experimental | tested on one desktop (Ryzen + RX 9070 XT) and a GitHub Windows runner (CPU); the MSI Claw (Intel iGPU) rerun is pending; NVIDIA and Intel GPUs on Windows are untested |
| Page file | a small Windows page file limits the cache: keep it system-managed or 14 GB or more |
| Flash-Next on AMD, driver older than 32.0.32015.7029 | context checkpoints are off (the driver crashes); update Adrenalin |
| Plugin CPU kernels, `--gpu-cache` | Linux only (they need `LD_PRELOAD`) |
| HIP / ROCm on Windows | untested |
