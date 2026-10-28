# Windows quick start (experimental)

What follows is the set-up that worked on 2026-10-26 on a Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM): identity 4/4 on four models, launcher
and chat working; speed numbers updated 2026-10-27 with the file-view cache (RESULTS 59-61). Status and caveats per platform: [COMPATIBILITY](COMPATIBILITY.md). If your model fits in RAM, stock llama.cpp is all you need; the plugin
helps with models bigger than RAM.

## 1. What you need

| Piece | What worked | Notes |
|---|---|---|
| llama.cpp | the official release zip **b11669, Vulkan** (`llama-b11669-bin-win-vulkan-x64.zip` from github.com/ggml-org/llama.cpp/releases), unpacked to e.g. `C:\llama` | ggml 0.27.0 = backend API 3. The speed numbers below (RESULTS 58-61) were measured with a Vulkan build of llama.cpp 620d2225c (also ggml 0.27). The zip with HIP (ROCm) listed no device on the test PC: it most likely needs AMD's HIP SDK (untested). Use Vulkan |
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
| At the first graph the plugin releases the model-file pages llama.cpp left in llama-server's working set (log: "released N GiB of model-file pages from the working set"); its own cache then lives in read-only views of the model file (file-view cache, no commit) | llama.cpp's Windows loader never unmaps the file pages of the GPU tensors, so ~13 GB stayed counted as in use with 12 GPU expert layers and the cache got that much less (RESULTS 56, I37). GPU expert layers are planned as on other GPUs |
| Flash-Next (`qwen4exp`) on an AMD Vulkan GPU with a driver older than 32.0.32015.7029 (or an unknown version): `--ctx-checkpoints 0`, with a message | saving a context checkpoint of its recurrent state crashes AMD's Windows Vulkan driver (`amdvlk64.dll` 32.0.23034.4, access violation); stock llama-server crashes the same way, so it is not the plugin. Cost: a conversation cannot jump back to an earlier point (plain continuation still reuses the cache); this matters for agents. **Driver 32.0.32015.7029 (Adrenalin, 2026-10-27) does not crash** (stock llama-server, checkpoints on, chat answered, no crash in the event log), so from that version on the launcher leaves checkpoints on and says so; the launcher reads the version from the registry (`DriverVersion` of the chosen adapter) |
| `--ram N` sizes the cache; a Windows job memory limit on llama-server only with `MOE_CACHE_WIN_VIEW=0` | a job limit caps commit, and Windows commits Vulkan's GPU memory: with the file-view cache (experts use no commit) it only capped llama.cpp's GPU memory and `--ram 16` failed to load. Without the limit `--ram 16` runs at 20.0 tok/s on Flash-Next IQ2_XS, 92 % of `--ram 24` (RESULTS 60) |
| Pins threads to the performance cores of a hybrid CPU | none needed on the 7950X (`check` says "no hybrid cores: no pinning") |
| The log line "commit ... failed (Windows error 487), committing page by page" | a handled case, not an error |

## 6. Known limits

- **Speed on the test PC** (RESULTS 59-60, 2026-10-27: Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5, page file 14 GB, llama.cpp Vulkan ggml 0.27;
  every run cold (file cache flushed), stock-plugin-plugin-stock order, launcher defaults, decode of a 256-token greedy request). These numbers need the
  **file-view cache** (the Windows default since 2026-10-27, `MOE_CACHE_WIN_VIEW=0` turns it off): the cache keeps experts in read-only views of the
  model file, which cost no commit. Before it, Windows charged the cache (and Vulkan's GPU memory) to commit, the cache was held at 8-11 GiB and decode
  was at or below stock (RESULTS 55, 58).

  | Model | GPU expert layers | Decode: stock | Decode: plugin | Change |
  |---|---|---|---|---|
  | Flash-Next Q2_0 (66.4 GB, D:, not interleaved; RESULTS 63) | 12 | 18.41 | **22.16** | +20 % |
  | Flash-Next IQ2_XS (68 GB) | 12 | 14.89 | **21.94** | +47 % |
  | Flash-Next IQ3_XXS (76 GB) | 9 | 15.14 | **19.86** | +31 % |
  | Flash-Next IQ3_S (84 GB) | 7 | 12.92 | **17.28** | +34 % |
  | Qwen3.5-122B UD-IQ2_XXS (34 GB), re-measured 2026-10-28 (RESULTS 61) | 12 | 15.95 | **19.73** | +24 % |
  | GLM-5.3-Flash UD-IQ2_XXS (102 GB), one pair | 3 | 2.21 | **4.44**; **6.92 with `--mtp`** | 2.0x; 3.1x |

  Text byte-identical to stock on Flash-Next IQ2_XS, IQ3_S, the 122B and GLM. Coding-agent sessions (recorded OpenCode / Hermes replays, Flash-Next
  IQ2_XS): 174.8 / 163.9 s stock vs **73.4 / 81.6 s** plugin (2.0-2.4x), first token 3x sooner; two agents at once 335 vs 145 s (RESULTS 60). The
  122B row replaces RESULTS 59's 13.88 vs 16.13 (+16 %): that pair ran during a slowdown of the whole PC (stock was 13 % slower too; RESULTS 61, T2b).
  If your model fits in RAM, stock llama.cpp is the better choice (the launcher picks it by itself). One desktop only; Linux numbers are in the README.
- **Which Flash-Next file for a 16 GB GPU + 32 GB RAM PC** (same PC, plugin with launcher defaults, cold; RESULTS 59, 60, 63). Tasks = 202
  exact-answer tasks; no quality difference between any two files is proven (all paired sign tests p > 0.05):

  | File | Size | Plugin decode (tok/s) | Prompt tok/s, 3k / 34k tokens | Tasks /202 | Use it |
  |---|---|---|---|---|---|
  | **Q2_0** | 66.4 GB | **22.2** | **281 / 287** | 191 | **recommended default**: vs IQ2_XS +5 % decode, prompts +11 % / +19 %, smaller. In the decode pair Q2_0 ran from D: and IQ2_XS from C:, so part of the +5 % may be the drive (the 34k pair had both on D:) |
  | IQ3_XXS | 75.8 GB | 19.9 (stock 15.1) | - | **193** | when accuracy matters: most tasks right, about 10 % slower |
  | ISTA GSQ-RCO Coder IQ1_M | 58.4 GB | 21.2 (in an agent session) | - | 189 | coding agents: half the experts (256 of 512); OpenCode session 55.5 s vs 73.4 s on IQ2_XS, one run. General knowledge not checked beyond the task set |
  | IQ2_XS | 68.0 GB | 21.0 same day as Q2_0 (21.9 in RESULTS 59, stock 14.9) | 253 / 242 | 187 | superseded by Q2_0. The other IQ2_XS numbers on this page were measured on IQ2_XS |
  | IQ3_S | 83.6 GB | 17.3 (stock 12.9) | - | 192 | not worth it: slower and bigger than IQ3_XXS for no gain |

  Stock vs plugin on Q2_0 (D:, cold, two runs each, not interleaved; RESULTS 63): decode 18.41 vs **22.16** tok/s (+20 %; code request 20.01 vs 22.66,
  +13 %), 3k-token prompt 120.8 vs **281** tok/s (2.33x). Stock decodes Q2_0 much faster than IQ2_XS (18.4 vs 14.9), so the decode gain is smaller
  than on IQ2_XS (+47 %); the prompt gain is what agent sessions wait on.
  On Windows the plugin's own Q2_0 kernel is not used (see the next point): Q2_0 runs llama.cpp's own CPU kernel on both sides.
- The plugin's own CPU kernels (Q2_0, IQ3_XXS gather, AVX-512) are compiled for Linux only: they need `LD_PRELOAD` symbol interposition, which Windows does not have. On Zen 4 they would
  not help anyway (bit-exact but 0.60-0.77x of llama.cpp's own kernels), which is why they stay off on AMD.
- `--gpu-cache` is not available (needs `LD_PRELOAD`); the launcher uses static GPU expert layers. A path-loading version exists on a branch, but
  llama.cpp's own GPU expert cache loses on this Vulkan PC (9.7 vs 13.7 tok/s without it) and aborts at load above about 5.9 GiB (RESULTS 59), so it is parked.
- HIP / ROCm: needs AMD's HIP SDK, untested. Intel GPUs on Windows: untested.
- The bash wrappers (`moe-cache-server` ...) do not run on Windows: use `python bin\moe_cache.py <command>`.
- GLM-5.3-Flash (102 GB) runs, long-prompt GPU blocks included (3,020-token prompt at 54.8 tok/s, RESULTS 60); the earlier crash in that path is fixed.
- Run as a normal user (not elevated), Windows may refuse the file-view cache's working-set minimum (error 1314 in the log); the cache still works
  (21.6 tok/s in that test, RESULTS 61). A normal desktop login has the privilege it needs.

### Windows: where the RAM goes, and suggestions that do not help

llama.cpp's Windows loader reads the GPU tensors through its mapping of the model file and never unmaps those pages. After loading they stay in
llama-server's working set, and Windows counts them as in use: 13.3 GB with 12 Flash-Next expert layers on the GPU. It looked like AMD's driver taking RAM
for VRAM, but it is not: a newer driver changed nothing, and emptying the working set gave the RAM back with GPU memory unchanged (RESULTS 56, I37). The
plugin releases these pages at the first graph (log: "released N GiB of model-file pages from the working set"). Stock llama-server keeps them.
The plugin's own cache lives in read-only views of the model file (the file-view cache, default since 2026-10-27): those pages are file-backed, so they
cost no commit, and a hard working-set minimum keeps them in RAM (log: "working-set minimum N GiB"). When another program needs memory Windows finds it
elsewhere: with 8 and 14 GB taken by another program the server kept its speed (22.3 before, 23.4-23.7 during) and nothing froze (RESULTS 59).

**Page file.** On Windows every private allocation needs commit (RAM plus page file), even if it never leaves RAM, and Vulkan's GPU allocations are charged
to commit too (about 24 GB with GLM). The file-view cache needs none, but llama.cpp's own memory does: GLM with `--mtp` left 3.47 GiB of commit free on a
14 GB page file (RESULTS 60). Before the file view (or with `MOE_CACHE_WIN_VIEW=0`) the cache itself was cut by commit: a 6.9 GB page file cut GLM's cache
from 16.1 to 9.6 GiB, a 14 GB one held 14.9 GiB (decode 1.52 -> 2.02 tok/s, RESULTS 58). Let Windows manage the page file, or make it 14 GB or more.

| Suggestion often made | What it does here |
|---|---|
| `--no-mmap` | loads the whole model into RAM: impossible for a model bigger than RAM |
| `--mlock` | pins memory so Windows cannot page it out: less RAM for everything else |
| `--moe-cache-mib N` | this is **llama.cpp's GPU expert cache**, not moe-cache's RAM cache, and it is not available with the plugin on Windows (`--gpu-cache` needs `LD_PRELOAD`). On this Vulkan PC it aborts at load above about 5.9 GiB even without the plugin (each cache bank is one tensor over Vulkan's ~2 GiB buffer limit) and, where it loads, it was slower (RESULTS 59). moe-cache's cache size follows `--ram` (the launcher sets `MOE_CACHE_SIZE_GIB` itself) |
| "the plugin's cache shrink causes thrashing" | the other way round: when the rest of llama-server's memory grows, the cache shrinks to stay under the memory limit, so that Windows does not page; the cost is more SSD reads (a lower hit rate), and the log says so ("the expert cache shrank from ..."). With the file-view cache this no longer happens for commit (1 resize per request instead of 226 on IQ3_XXS, RESULTS 59) |
| q8_0 KV cache (`-ctk q8_0 -ctv q8_0`) | saves VRAM (the KV part only), not RAM; it is approximate (KL 0.0045, RESULTS 53), so it is part of `--fast`, not the default. Measured on this desktop at context 8192 (RESULTS 60, R11): decode 22.01 vs 22.12 tok/s, a 2,950-token prompt 170.6 vs 164.6: no speed difference; it matters only for memory at long agent contexts |
| WSL2 | untested; GPU access from WSL2 still goes through the Windows driver. Native Linux with the RADV Vulkan driver is the tested route for AMD cards (GitHub issue #1, RX 9070 XT) |

Still to try (owner's machine): the HIP SDK, Linux/RADV on the same PC. A newer Adrenalin driver (32.0.32015.7029) was tried: it fixed the Flash-Next
checkpoint crash but not the RAM use, which is the loader's (RESULTS 56, I37).

## 7. Fair stock-vs-plugin timings: empty the file cache first

Stock llama-server reads the model through Windows' file cache, so a run after another run of the same model finds it in RAM. The file-view cache uses the
same file cache (its evicted pages stay there, which is why a warm restart answers 6 s sooner than stock, RESULTS 60). So a run right after another run of
the same model is warm for both. Timing one side warm and the other cold compares two different things (before the file view the plugin's cold first
session ran at 3-5 tok/s against 14-16 for warm stock, RESULTS 54).
Before **every** timed run (stock and plugin alike) push the model out of the cache by reading other big files:

```
powershell -ExecutionPolicy Bypass -File tools\win_flush_cache.ps1 -Target D:\models\model.gguf -Other D:\models\other1.gguf,E:\old-models
```

It reads other files (files of 100 MB or more in a folder, searched recursively) for 1.15x the machine's RAM, skips the target, and prints GB read and seconds (about 100 GB in 40 s on the test PC).
No admin rights; it only reads, never writes or deletes. It needs other big files totalling more than your RAM: if they are fewer it says so and exits with 1. For an ABBA series call it between runs.

## 8. Repeat the identity test

`tools\windows_test.ps1`, see [WINDOWS_TEST](WINDOWS_TEST.md).
