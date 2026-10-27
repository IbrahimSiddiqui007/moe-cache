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

## Speed on Windows (one desktop: Ryzen 9 7950X, 32 GB RAM, RX 9070 XT 16 GB, llama.cpp b11669 Vulkan, 2026-10-27)

@SPEED_TABLE@

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
