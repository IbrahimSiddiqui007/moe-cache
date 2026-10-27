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

Without starting the server by hand each time (as with Ollama): the model starts on the first agent request and stops after 15 idle minutes.

```
python bin\moe_cache.py on-demand D:\models\model.gguf --llama-server C:\llama\llama-server.exe --ram 20
python bin\moe_cache.py connect opencode --write        # or: connect hermes --write  (the model does not start for this)
python bin\moe_cache.py on-demand --install-autostart   # optional: a shortcut in your Startup folder, no admin; --remove-autostart undoes it
```

Hermes Desktop by hand: Settings -> Model -> Main model, provider custom, URL `http://127.0.0.1:8080/v1`, API key from `python bin\moe_cache.py key`.

Details, and the desktop apps: [AGENTS](AGENTS.md#without-starting-a-server-by-hand-on-demand). Note for Flash-Next below.

## 5. What the launcher does for you on Windows with an AMD or Intel GPU

The control page shows these too: in the plan before you start ("Windows + AMD: ...", with the reason) and, once started, under "What the launcher decided".

| Automatic | Why |
|---|---|
| Finds the discrete GPU (the planner only knew nvidia-smi and Linux sysfs): reads `llama-server --list-devices` and takes the adapter with the most dedicated VRAM in the registry | otherwise it saw no GPU and kept every expert layer on the CPU; `check` named the built-in iGPU |
| At the first graph the plugin releases the model-file pages llama.cpp left in llama-server's working set (log: "released N GiB of model-file pages from the working set") and sizes its cache by its private working set | llama.cpp's Windows loader never unmaps the file pages of the GPU tensors, so ~13 GB stayed counted as in use with 12 GPU expert layers and the cache got that much less (RESULTS 56, I37). GPU expert layers are planned as on other GPUs |
| Flash-Next (`qwen4exp`) on an AMD Vulkan GPU with a driver older than 32.0.32015.7029 (or an unknown version): `--ctx-checkpoints 0`, with a message | saving a context checkpoint of its recurrent state crashes AMD's Windows Vulkan driver (`amdvlk64.dll` 32.0.23034.4, access violation); stock llama-server crashes the same way, so it is not the plugin. Cost: a conversation cannot jump back to an earlier point (plain continuation still reuses the cache); this matters for agents. **Driver 32.0.32015.7029 (Adrenalin, 2026-10-27) does not crash** (stock llama-server, checkpoints on, chat answered, no crash in the event log), so from that version on the launcher leaves checkpoints on and says so; the launcher reads the version from the registry (`DriverVersion` of the chosen adapter) |
| `--ram N` becomes a job memory limit on llama-server | the Windows counterpart of the Linux memory limit; logged as "Windows job memory limit" |
| Pins threads to the performance cores of a hybrid CPU | none needed on the 7950X (`check` says "no hybrid cores: no pinning") |
| The log line "commit ... failed (Windows error 487), committing page by page" | a handled case, not an error |

## 6. Known limits

- **Speed on the test PC** (RESULTS 55: apps closed, Windows' file cache flushed before every run, stock-plugin-plugin-stock order, all experts on the CPU):

  | Model | Decode: stock vs plugin | Prefill (about 3.5k-token prompt) |
  |---|---|---|
  | Flash-Next IQ2_XS (64 GB) | stock 12.1-16.1, plugin about 12.2 tok/s (about equal) | **2.2x** (120 vs 53-57 tok/s) |
  | Qwen3.5-122B UD-IQ2_XXS (36.6 GB) | stock about 11-14, plugin about 6.5 tok/s (**about half**; cause under diagnosis) | **1.5-1.6x** (128 vs 80-87) |
  | GLM-5.3-Flash UD-IQ2_XXS (102 GB) | plugin lower than stock in these runs (small cache: before the I37 fix, 6.9 GB page file); plugin 1.52 -> 2.02 tok/s with a 14 GB page file | about **2.9x**; **numbers to be confirmed** (the clean GLM runs are still going) |

  What it means for you on Windows with an AMD card: the plugin's gain is **prompt processing** (long prompts, coding agents that send 5-20k tokens per turn): the first turn and follow-ups
  after a big tool result come back 1.5-3x sooner, and the output is exact. **Decode is limited by RAM** (the cache holds only what fits; a small page file makes it smaller, see below):
  on models far bigger than RAM expect decode equal to or below stock, not above. If your model fits in RAM, stock llama.cpp is the better choice (the launcher picks it by itself).
  One desktop only; Linux numbers are in the README.
- **Which Flash-Next file for a 16 GB GPU + 32 GB RAM PC** (RESULTS 58: Ryzen 9 7950X + RX 9070 XT, Windows 11, ggml 0.27, cold runs, decode = two 256-token prompts, prefill = 3.5k-token prompt; plugin = `MOE_CACHE_WIN_BUFFERED=1` + `--n-cpu-moe 35`, 13 GPU expert layers; stock = the launcher's own split). Both files are listed on purpose; pick by what you value:

  | Quant | File size | Plugin decode (tok/s) | Plugin prefill (tok/s) | Stock decode (tok/s) | Stock prefill (tok/s) | Quality set (20 tasks) |
  |---|---|---|---|---|---|---|
  | IQ2_XS | 68 GB | 18.5 / 19.5 | 153 | 15.1-15.2 / 19.2-19.4 | 108-110 | 17/20, 0 loops in 44 |
  | IQ3_XXS | 76 GB | 11.9-12.0 / 13.2 | 108-109 | 14.7 / 17.5-17.9 | 96-102 | 18/20, 1 loop in 44 |

  How to choose: IQ2_XS is faster with the plugin (decode above stock, prefill 1.4x) and scored 17/20 against 18/20 for IQ3_XXS on a small set, so it is not measurably worse there. Choose IQ3_XXS if you want the larger quant anyway; today the plugin is **slower than stock on IQ3 decode on Windows** (about 12-13 against 15-18 tok/s; prefill is still ahead; being investigated, task T4), so for IQ3 decode alone stock llama.cpp is faster at the moment. An IQ3_S file (83.6 GB) behaves like IQ3_XXS in a first look (plugin decode 8.7-9.8 against stock 10.6-12.7). The quality set is small (20 tasks): it cannot settle "IQ2 answers worse". Details: [RESULTS 58](RESULTS.md#58-windows-desktop-2026-10-27-fixes-confirmed-mtp-flash-next-iq2_xs-vs-iq3_xxs-glm-page-file).
- The plugin's own CPU kernels (Q2_0, IQ3_XXS gather, AVX-512) are compiled for Linux only: they need `LD_PRELOAD` symbol interposition, which Windows does not have. On Zen 4 they would
  not help anyway (bit-exact but 0.60-0.77x of llama.cpp's own kernels), which is why they stay off on AMD.
- `--gpu-cache` is not available (needs `LD_PRELOAD`); the launcher uses static GPU expert layers.
- HIP / ROCm: needs AMD's HIP SDK, untested. Intel GPUs on Windows: untested.
- The bash wrappers (`moe-cache-server` ...) do not run on Windows: use `python bin\moe_cache.py <command>`.
- A GLM-class model (102 GB) crashed the plugin in the long-prompt GPU path on the test PC; a fix is committed and still to be confirmed on the PC.

### Windows: where the RAM goes, and suggestions that do not help

llama.cpp's Windows loader reads the GPU tensors through its mapping of the model file and never unmaps those pages. After loading they stay in
llama-server's working set, and Windows counts them as in use: 13.3 GB with 12 Flash-Next expert layers on the GPU. It looked like AMD's driver taking RAM
for VRAM, but it is not: a newer driver changed nothing, and emptying the working set gave the RAM back with GPU memory unchanged (RESULTS 56, I37). The
plugin releases these pages at the first graph (log: "released N GiB of model-file pages from the working set") and measures itself by its private
working set. Stock llama-server keeps them.

**Page file.** On Windows every allocation needs commit (RAM plus page file), even if it never leaves RAM. With a small page file the cache reaches the
commit limit before RAM is full (log: "close to its commit limit"). On the PC, a 6.9 GB page file with 16 GB committed by other programs cut GLM's cache
from 16.1 to 9.6 GiB; with a 14 GB page file it held 14.9 GiB and decode went from 1.52 to 2.02 tok/s. Let Windows manage the page file, or make it 14 GB or more.

| Suggestion often made | What it does here |
|---|---|
| `--no-mmap` | loads the whole model into RAM: impossible for a model bigger than RAM |
| `--mlock` | pins memory so Windows cannot page it out: less RAM for everything else |
| `--moe-cache-mib N` | this is **llama.cpp's GPU expert cache**, not moe-cache's RAM cache, and it is not available with the plugin on Windows (`--gpu-cache` needs `LD_PRELOAD`). moe-cache's cache size follows `--ram` (or `MOE_CACHE_SIZE_GIB`); forcing it above the free RAM makes Windows page to disk |
| "the plugin's cache shrink causes thrashing" | the other way round: when the rest of llama-server's memory grows, the cache shrinks to stay under the memory limit, so that Windows does not page; the cost is more SSD reads (a lower hit rate), and the log says so ("the expert cache shrank from ...") |
| q8_0 KV cache (`-ctk q8_0 -ctv q8_0`) | saves VRAM (the KV part only), not RAM; it is approximate (KL 0.0045, RESULTS 53), so it is part of `--fast`, not the default; the saving on this desktop is being measured (RESULTS 56, R11) |
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
