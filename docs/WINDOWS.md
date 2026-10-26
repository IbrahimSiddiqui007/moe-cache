# Windows quick start (experimental)

What follows is the set-up that worked on 2026-10-26 on a Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM): identity 4/4 on four models, launcher
and chat working. Status and caveats per platform: [COMPATIBILITY](COMPATIBILITY.md). If your model fits in RAM, stock llama.cpp is all you need; the plugin
helps with models bigger than RAM.

## 1. What you need

| Piece | What worked | Notes |
|---|---|---|
| llama.cpp | the official release zip **b11669, Vulkan** (`llama-b11669-bin-win-vulkan-x64.zip` from github.com/ggml-org/llama.cpp/releases), unpacked to e.g. `C:\llama` | ggml 0.27.0 = backend API 3. The zip with HIP (ROCm) listed no device on the test PC: it most likely needs AMD's HIP SDK (untested). Use Vulkan |
| The plugin | `ggml-moe-cache-api3.dll` (api3 for ggml 0.26/0.27, api2 for 0.25) | from the kit, or build it on Linux: `GGML_API=3 ./build.sh --windows` (mingw-w64) -> `build/win/ggml-moe-cache.dll` |
| Python 3 | 3.8 or newer, standard library only | run the launcher with `python` |
| This repository (or the kit with `moe_cache.py`) | `C:\moe-cache` | the launcher needs the `planner\`, `ui\` and `tools\` folders: keep the whole repository |
| A GGUF model | any MoE model | Windows has no memory-limit scope in the test script: a model bigger than RAM can slow the machine down, so start with one that fits |

The launcher finds the DLL by itself when it is `ggml-moe-cache-api3.dll` next to `moe_cache.py` or `build\win*\`; otherwise point to it:

```
set MOE_CACHE_LIB=C:\moe-cache-kit\ggml-moe-cache-api3.dll
set LLAMA_SERVER=C:\llama\llama-server.exe
```

## 2. Check, then run

```
python bin\moe_cache.py check D:\models\model.gguf --llama-server C:\llama\llama-server.exe --run
python bin\moe_cache.py server D:\models\model.gguf --llama-server C:\llama\llama-server.exe --ram 24 --open
```

`check --run` starts llama-server with the plugin and a small cache for one short request and reports "cache runs: yes". Two warnings are normal today:
"Windows experimental" and, on a llama.cpp other than b11669, "ggml version". `server` prints the command (add `--dry-run` to only print it); `--open`
opens llama-server's chat page. The model also works with any OpenAI-compatible client at `http://127.0.0.1:8080/v1`.

## 3. The control page (GUI): one click

Double-click **`bin\moe-cache-ui.bat`**. It opens http://127.0.0.1:8070 (this machine only): your models with the planner's verdict and why, start/stop,
live speed, cache hit rate, SSD reads, RAM and VRAM, the chat, and Connect buttons for OpenCode and Hermes. To get it in the Start menu, run once:

```
bin\moe-cache-ui.bat --shortcut
```

What the `.bat` finds by itself (set the variable to override):

| Piece | Looked for, in this order |
|---|---|
| Python | `MOE_CACHE_PYTHON`, the `py` launcher, a `python.exe` that really runs (the one Windows ships without Python only opens the Store) |
| Plugin (`MOE_CACHE_LIB`) | `ggml-moe-cache-api3.dll`, then `ggml-moe-cache.dll`, next to the `.bat` or one folder up (the kit), then `build\win-api3\`, `build\win\` |
| llama-server (`LLAMA_SERVER`) | on the `PATH`, then `C:\llama\llama-server.exe` |

It prints what it found and the page's address (with a per-run token: only you can use the page); the same is in the page header's tooltip. Without the `.bat`: `python bin\moe_cache.py ui --llama-server C:\llama\llama-server.exe`.
Options for both: `--ui-port N`, `--no-browser`. Details: [GUI](GUI.md).

## 4. Coding agents

```
python bin\moe_cache.py server D:\models\model.gguf --llama-server C:\llama\llama-server.exe --agent
python bin\moe_cache.py connect opencode --write        # or: connect hermes --write
```

Details, and the desktop apps: [AGENTS](AGENTS.md). Note for Flash-Next below.

## 5. What the launcher does for you on Windows with an AMD or Intel GPU

The control page shows these too: in the plan before you start ("Windows + AMD: ...", with the reason) and, once started, under "What the launcher decided".

| Automatic | Why |
|---|---|
| Finds the discrete GPU (the planner only knew nvidia-smi and Linux sysfs): reads `llama-server --list-devices` and takes the adapter with the most dedicated VRAM in the registry | otherwise it saw no GPU and kept every expert layer on the CPU; `check` named the built-in iGPU |
| With an AMD GPU and a model **bigger than RAM**: all experts stay on the CPU (`--n-cpu-moe` = all) | on AMD's Windows driver, VRAM in use costs about as much system RAM, so GPU expert layers left the cache ~3 GiB and decode fell to a quarter of stock. Override with `--n-cpu-moe N` |
| Flash-Next (`qwen4exp`) on an AMD Vulkan GPU: `--ctx-checkpoints 0`, with a message | saving a context checkpoint of its recurrent state crashes AMD's Windows Vulkan driver (`amdvlk64.dll` 32.0.23034.4, access violation); stock llama-server crashes the same way, so it is not the plugin. Cost: a conversation cannot jump back to an earlier point (plain continuation still reuses the cache); this matters for agents |
| `--ram N` becomes a job memory limit on llama-server | the Windows counterpart of the Linux memory limit; logged as "Windows job memory limit" |
| Pins threads to the performance cores of a hybrid CPU | none needed on the 7950X (`check` says "no hybrid cores: no pinning") |
| The log line "commit ... failed (Windows error 487), committing page by page" | a handled case, not an error |

## 6. Known limits

- **Speed on the test PC** (RESULTS 55: apps closed, Windows' file cache flushed before every run, stock-plugin-plugin-stock order, all experts on the CPU):

  | Model | Decode: stock vs plugin | Prefill (about 3.5k-token prompt) |
  |---|---|---|
  | Flash-Next IQ2_XS (64 GB) | stock 12.1-16.1, plugin about 12.2 tok/s (about equal) | **2.2x** (120 vs 53-57 tok/s) |
  | Qwen3.5-122B UD-IQ2_XXS (36.6 GB) | stock about 11-14, plugin about 6.5 tok/s (**about half**; cause under diagnosis) | **1.5-1.6x** (128 vs 80-87) |
  | GLM-5.3-Flash UD-IQ2_XXS (102 GB) | plugin lower than stock (the AMD driver's RAM cost squeezes the cache) | about **2.9x**; **numbers to be confirmed** (the clean GLM runs are still going) |

  What it means for you on Windows with an AMD card: the plugin's gain is **prompt processing** (long prompts, coding agents that send 5-20k tokens per turn): the first turn and follow-ups
  after a big tool result come back 1.5-3x sooner, and the output is exact. **Decode is limited by RAM** (the AMD driver takes system RAM for VRAM in use, so the cache is small):
  on models far bigger than RAM expect decode equal to or below stock, not above. If your model fits in RAM, stock llama.cpp is the better choice (the launcher picks it by itself).
  One desktop only; Linux numbers are in the README.
- The plugin's own CPU kernels (Q2_0, IQ3_XXS gather, AVX-512) are compiled for Linux only: they need `LD_PRELOAD` symbol interposition, which Windows does not have. On Zen 4 they would
  not help anyway (bit-exact but 0.60-0.77x of llama.cpp's own kernels), which is why they stay off on AMD.
- `--gpu-cache` is not available (needs `LD_PRELOAD`); the launcher uses static GPU expert layers.
- HIP / ROCm: needs AMD's HIP SDK, untested. Intel GPUs on Windows: untested.
- The bash wrappers (`moe-cache-server` ...) do not run on Windows: use `python bin\moe_cache.py <command>`.
- A GLM-class model (102 GB) crashed the plugin in the long-prompt GPU path on the test PC; a fix is committed and still to be confirmed on the PC.

### Windows + AMD: where the RAM goes, and suggestions that do not help

On AMD's Windows driver, VRAM in use also takes system RAM: about 1.1 GB per GiB, inside the driver and Windows' video memory manager, not in llama.cpp or
the plugin (RESULTS 56, I37: `GGML_VK_MEMORY_LOGGER=1` shows 13.1 GiB of device memory and 169 MiB of host memory, yet 13.5 GB of system RAM gone). The
`GGML_VK_*` switches tried there did not change it. The plugin sees less free RAM, so its cache is smaller and it reads more from the SSD.

| Suggestion often made | What it does here |
|---|---|
| `--no-mmap` | loads the whole model into RAM: impossible for a model bigger than RAM, and it does not touch the driver's copy |
| `--mlock` | pins memory so Windows cannot page it out: less RAM for everything else, the driver's copy stays |
| `--moe-cache-mib N` | this is **llama.cpp's GPU expert cache**, not moe-cache's RAM cache, and it is not available with the plugin on Windows (`--gpu-cache` needs `LD_PRELOAD`). moe-cache's cache size follows `--ram` (or `MOE_CACHE_SIZE_GIB`); forcing it above the free RAM makes Windows page to disk |
| "the plugin's cache shrink causes thrashing" | the other way round: when the rest of llama-server's memory grows, the cache shrinks to stay under the memory limit, so that Windows does not page; the cost is more SSD reads (a lower hit rate), and the log says so ("the expert cache shrank from ...") |
| q8_0 KV cache (`-ctk q8_0 -ctv q8_0`) | saves VRAM and so the driver's RAM copy of it, but only the KV part (with all experts on the CPU, the GPU holds attention and KV); it is approximate (KL 0.0045, RESULTS 53), so it is part of `--fast`, not the default; the saving on this desktop is being measured (RESULTS 56, R11) |
| WSL2 | untested; GPU access from WSL2 still goes through the Windows driver. Native Linux with the RADV Vulkan driver is the tested route for AMD cards (GitHub issue #1, RX 9070 XT) |

Still to try (owner's machine): a newer Adrenalin driver, hardware-accelerated GPU scheduling off, the HIP SDK, Linux/RADV on the same PC (RESULTS 56, I37).

## 7. Fair stock-vs-plugin timings: empty the file cache first

Stock llama-server reads the model through Windows' file cache, so a run after another run of the same model finds it in RAM; the plugin's unbuffered reads never use that cache.
Timing stock after a warm-up and the plugin on a cold start compares two different things (on the test PC the plugin's first session ran at 3-5 tok/s against 14-16 for warm stock).
Before **every** timed run (stock and plugin alike) push the model out of the cache by reading other big files:

```
powershell -ExecutionPolicy Bypass -File tools\win_flush_cache.ps1 -Target D:\models\model.gguf -Other D:\models\other1.gguf,E:\old-models
```

It reads other files (files of 100 MB or more in a folder, searched recursively) for 1.15x the machine's RAM, skips the target, and prints GB read and seconds (about 100 GB in 40 s on the test PC).
No admin rights; it only reads, never writes or deletes. It needs other big files totalling more than your RAM: if they are fewer it says so and exits with 1. For an ABBA series call it between runs.

## 8. Repeat the identity test

`tools\windows_test.ps1`, see [WINDOWS_TEST](WINDOWS_TEST.md).
