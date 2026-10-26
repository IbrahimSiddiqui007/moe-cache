# Trying moe-cache on Windows (experimental: tested, not a supported install)

State: the plugin has a Windows layer (`src/platform.h`) that cross-compiles with mingw-w64 and passes its platform tests under Wine. It has run on real Windows:
the MSI Claw (Vulkan iGPU, rounds 1-2 found bugs) and, on 2026-10-26, a Windows 11 desktop (Ryzen 9 7950X + RX 9070 XT, llama.cpp b11669 Vulkan): identity 4/4 on four
models and the Python launcher (check, server, chat) works; see `COMPATIBILITY.md`. Speed numbers with the plugin on the desktop are preliminary (RESULTS 54). Everything below is how to repeat the
test. If you only need to run models on Windows, use a stock llama.cpp release; the plugin only adds a RAM cache for models bigger than RAM.

## What to run
1. `tools/windows_test.ps1` (PowerShell). It starts an official `llama-server.exe` twice on the same GGUF (stock, then with the plugin DLL through `GGML_BACKEND_PATH`),
   sends 4 prompts at temperature 0 and prints PASS / FAIL, decode speed, and the plugin's `moe-cache:` log lines.
2. Use a model that **fits in RAM** for the first test (an 18 GB Qwen3-Coder 30B-A3B Q4_K_M on a 31 GB machine). Windows has no memory-limit scope in this script:
   a model larger than RAM can freeze the machine.

## How to install
On Linux, before rebooting (the Windows disk is NTFS, the Linux disk is btrfs: Windows cannot read your Linux files):
- Cross-compile (done already for the kit): `GGML_API=3 ./build.sh --windows` and `./build.sh --windows` (API 2); output `build/win/ggml-moe-cache.dll`.
- Mount the Windows partition and copy the kit and one model to it (`lsblk -f` shows which partition is NTFS; if it says the volume is hibernated, boot Windows once
  and shut down with "Shift + Shutdown" or disable Fast Startup).
- In Windows: download an official llama.cpp release zip from https://github.com/ggml-org/llama.cpp/releases (the `win-cuda` build for an NVIDIA GPU, plus its
  `cudart` zip) and unpack it. Pick the plugin DLL by API: **api3** for llama.cpp 23b0202a1 or newer ggml 0.26.x, **api2** for ggml 0.25.x. A newer llama.cpp may
  have a newer backend API; the plugin then prints a clear message and the model loads normally without it.

## How to run
```
powershell -ExecutionPolicy Bypass -File windows_test.ps1 -LlamaDir C:\llama -Model D:\moe-cache-kit\qwen3-coder-30b-q4km.gguf -Dll D:\moe-cache-kit\ggml-moe-cache-api3.dll
```
Send back: the printed result, `%TEMP%\moe-cache-test\plugin.err.txt`, and the llama.cpp release name. Failure modes to expect and report: the DLL fails to load
(missing ggml symbol), the plugin refuses on API mismatch (message names both), the plugin loads but the identical-prompt count is below 4/4 (a bug).
The test turns the cache on at 0.5 GiB (`-CacheGiB`), far below the model, so experts are evicted and read again; it fails if the log has no
`moe-cache: cache mode` line (before 2026-10-15 it ran the plugin in pass-through mode and a PASS proved little).

## Launcher on Windows (T15, tested on the desktop above)
The launcher is Python (`bin/moe_cache.py`, standard library only), so it runs on Windows too; there the plugin is always loaded with `GGML_BACKEND_PATH`:
```
set MOE_CACHE_LIB=D:\moe-cache-kit\ggml-moe-cache-api3.dll
python moe-cache\bin\moe_cache.py server D:\models\model.gguf --llama-server C:\llama\llama-server.exe --ram 16 --open
python moe-cache\bin\moe_cache.py check D:\models\model.gguf --llama-server C:\llama\llama-server.exe
```
Add `--dry-run` to see the command without starting it. The launcher pins threads to the performance cores of a hybrid CPU on Windows too (`win_cores()`;
`check` reports "no hybrid cores: no pinning" on a plain CPU such as the 7950X). It finds an AMD or Intel GPU through the registry/`--list-devices`, not only NVIDIA,
and on an AMD Vulkan GPU it turns llama.cpp's context checkpoints off for Flash-Next (qwen4exp): the AMD Windows driver crashes when one is saved.
`build.sh --windows` writes `build/win/` or `$BUILD_DIR` (`BUILD_DIR=build/win-api3 GGML_API=3 ./build.sh --windows` keeps both APIs side by side); the launcher picks the DLL that matches the server's ggml
(it also finds the kit's `ggml-moe-cache-api2.dll` / `-api3.dll` next to `moe_cache.py`).
