# Hardware, operating systems and limits

## Hardware and OS support

| Platform | Status |
|---|---|
| Linux + Intel CPU + NVIDIA GPU (CUDA) | **Tested** |
| Linux + Vulkan (NVIDIA, and the Intel integrated GPU) | **Tested** on llama.cpp `7fe450e19` only, loaded the official way (`GGML_BACKEND_PATH`; see the feature table below) |
| Linux, no usable GPU (CPU only) | **Tested**, identical to stock, close to its speed when the model fits |
| Linux + AMD CPU, AMD integrated GPU (Vulkan) | **Tested** on a Ryzen 7 5800H with Vega 8 (2026-10-16): identical to stock, Qwen3.6 at 12 GB (speed figures follow in a later release; no expert layer fits on a 3 GiB iGPU); RESULTS 50 |
| Linux + AMD discrete GPU | **Tested by a user** (GitHub issue #1: RX 9070 XT + Ryzen 7 7700, Linux, Vulkan): identity PASS, Flash-Next (speed figures follow in a later release). Not yet run by the developer on Linux |
| Windows | **Experimental.** Identical to stock in CI on a Windows runner and on a Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB, Vulkan). Since 2026-10-27 the cache keeps experts in read-only views of the model file (Windows charged its private memory, and Vulkan's GPU memory, to commit, which held the cache at 8-11 GiB). Cold, stock vs moe-cache, launcher defaults (RESULTS 59-60): tested on Flash-Next Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, Qwen3.5-122B and GLM-5.3 (also with `--mtp`); speed figures follow in a later release; text byte-identical. Context checkpoints stay on with AMD driver 32.0.32015.7029 or newer (older ones crash with Flash-Next). See docs/reference/platforms.md |
| macOS | No |

## Known limitations

- Long prompts (32+ tokens per batch) run their expert blocks on the GPU, as stock llama.cpp does, so prompt processing uses the GPU as stock does; speed figures follow in a later release.
- It does not manage the GPU; which layers' experts sit in VRAM is chosen once at start by the planner. A GPU expert tier was evaluated and is not built, see [docs/internal/ENGINE_DECISION.md](../internal/ENGINE_DECISION.md).
- **Not every feature works with every llama.cpp build.** How the plugin is loaded depends on how llama.cpp was built (`moe-cache-server` picks it):

  | Feature | `LD_PRELOAD` (llama.cpp that links its backends directly: distro packages, your own CUDA build) | `GGML_BACKEND_PATH` (backends as separate files: official releases, `./install.sh`, Vulkan builds, Windows) |
  |---|---|---|
  | RAM expert cache, SSD streaming, memory limit, warm start | yes | yes |
  | Expert executor (computes ready experts while others load) | yes | yes (since 0.2.3.3; tested on a Vulkan build) |
  | Fast Q2_0 / IQ3_XXS kernels | yes | inside the executor only (Linux, and Windows x86-64 since 2026-10-28; no speed change on the PC) |
  | Working together with llama.cpp's GPU expert cache (`--gpu-cache`) | yes | **no** (refused since cffa6f9: it gave wrong tokens; the launcher uses static GPU expert layers instead) |

  The `moe-cache: active: ...` line in the server log says which parts are on.
- **The kits' llama.cpp (v0.3.8):** the Windows kit (`llama\`) and the Linux `-ggml0.26` tarball (`llama/`) bundle our build of llama.cpp b12651 (ggml 1.0, API 4; before v0.3.9: b12207) built by CI from the official source plus `patches/llama.cpp` (0004, 0020, 0030, 0040, 0090, 0091, 0092). Every patch is built in and off (none is default-on; 0040 is a print-only diagnostic), the start-up line says `llama.cpp b12651 + moe-cache patches 0004,0020,0030,0040,0090,0091,0092 built in, all off`, and the plugin does not need any patch: it loads into the official release build too (set `LLAMA_SERVER`). Release-workflow runs started by hand can pick `llama_build: official` to build the kits with the unmodified official release instead.
- Needs a llama.cpp with shared ggml libraries; tested with commits `23b0202a1` (ggml 0.26.0) and `7fe450e19` (ggml 0.25.1), and releases b11966, b12064 (ggml 0.28.0) b12207 (ggml 0.29.0) and b12651 (ggml 1.0.0, API 4: CI only, not yet measured on the test PC; CI rows added 2026-10-30, 2026-11-03 and 2026-11-09). The plugin refuses to load into a ggml with a different backend API than it was built for (it prints why and the model loads normally). Linux; Windows is experimental (identity-tested on a Windows 11 + RX 9070 XT desktop and in CI, see [platforms](platforms.md) and the [Windows quick start](../guides/windows.md)).

## Prototypes: several GPUs and several models at once (all off by default, untested for speed)

These options exist so a user with more than one GPU, or two models, can try them and report. Each prints "PROTOTYPE" when it is on. Without the flag the launch plan is byte-identical to before (`tests/launch_plan_test.py`, every case). The unit tests run on fake devices and fake servers; **(2026-11-03: the only two-device run so far is a smoke test on the Windows PC, RX 9070 XT + iGPU, Granite; the sentence continues as written on 2026-10-30.) Nothing here has been run on a machine with two discrete GPUs, with two models served together through the launcher (the peer rows say what was run by hand on the laptop), or with a real vision model, and no speed gain is claimed.**

| Option | What it does | What is untested |
|---|---|---|
| `moe-cache-server --multi-gpu` | With 2+ discrete GPUs (same backend; an iGPU or a card under 2 GiB is left out), plans the experts across the cards and passes `--device A,B -ts a,b --n-cpu-moe N` (largest card first). `--dry-run` shows the plan. Your own `--n-cpu-moe`, `--vram`, `--device`, `-ts`, `--gpu-cache` or `--hybrid` keep the one-card plan. One card: no change. | Everything beyond the plan: speed, and whether llama.cpp places the layers as planned on real cards. With a plan applied, the plugin is also told `MOE_CACHE_GPU_DEV2=auto` (next row). When `llama-fit-params` is found next to `llama-server`, one line gives llama.cpp's own per-card projection of the plan (MG2; reports only, the plan stands; never run against the real tool on two cards). |
| Plugin switch `MOE_CACHE_GPU_DEV2=auto\|DEVICE` (N3/MG5) | The plugin's long-prompt GPU block normally runs on the first card. With this, a block that does not fit there (llama.cpp's own buffers grow with the context) runs whole on a second card when it fits there, instead of being split into parts or going back to the CPU. `auto`: the next discrete GPU of the same backend (or of your `--device` list); a name such as `Vulkan0` takes that device, an iGPU too. The hybrid bank (`--hybrid`, `--gpu-cache`) stays on the first card. The first card is always tried first. Off by default; `--multi-gpu` sets `auto` when it applies a plan. | Run once on the laptop (RTX 4060 held at 0.3 GiB free by a filler, Intel iGPU as the second card): the blocks ran on the iGPU, the text of a 1283-token prompt (12 tokens) was the same as without the switch. Slower there (the iGPU is slow and that first request read cold experts), so no speed claim. Not run on two discrete cards; the second card's backend, its buffers and read-back are a copy of the first card's code path, not separately tuned. |
| `moe-cache-server --peers`, `on-demand --concurrent --peers`, plugin `MOE_CACHE_PEERS=DIR` + `MOE_CACHE_PEER_CAP_GIB=N` (N2/MM2) | Plugin servers that name the same folder write a small status file once a second (memory held, cache size, busy or quiet) and read the others'. The room a server may use is the shared cap minus what the others hold; a server with no graph for 10 s gives its cache back one GiB a second, down to a floor, while another one is busy and has a full cache; two busy servers drift to their own `--ram` shares. A crashed server's file is ignored after 5 s. Only the size of each cache changes, never what is computed. A stock llama-server (no plugin) is not in the folder: use `--share-ram`. | Run on the laptop with two real plugin servers (35B MoE, CPU only, shared cap 11 GiB, `--ram 6` each): they saw each other, the quiet one dropped from 2.5 to 0.8 GiB while the busy one grew from 6.0 to 7.6 GiB, the files followed. Rules checked on fake peers (`tests/peers_test.cpp`). Not run with the launcher or on-demand end to end, on Windows, on a GPU, or with a speed measurement. |
| `moe-cache-check` | Names the GPUs llama.cpp will use and the iGPU it leaves out (printed only when 2+ devices are listed). | - |
| `on-demand --concurrent` | Keeps all added models running at once, each started with a fixed share of one memory cap (`--ram`), of the free VRAM (`--vram`) and its own CPU cores (`-C`, `-t`). A model whose share holds its file runs as plain stock llama-server. If the shares cannot fit, it falls back to one model at a time. The share is static unless `--peers` is added (row above); a quiet model gives its cache back after 30 s idle. | Interference between the servers (page cache, SSD, PCIe), the right split, Windows commit with two GPU servers. |
| `moe-cache-server --auto-fit` | For a model with no experts, leaves out `-ngl 99` and `--n-cpu-moe` so llama.cpp's `--fit` places the layers by free GPU memory. A MoE model and your own `-ngl` are untouched. | Never run on a real dense model bigger than VRAM. |
| `moe-cache-server --mmproj auto\|FILE` | Passes the vision projector (`mmproj*.gguf` next to the model), adds its size to the RAM decision and takes it, plus 0.5 GB for its compute buffer, off the GPU memory the planner may use. | Never run with a real vision model; the 0.5 GB is an estimate. |
| `moe-cache-server --share-ram` | Counts the RAM other `llama-server` processes hold and shrinks this server's share under the one cap (unless `--ram` is given). | Windows `tasklist` figures (working set) were not compared with the Linux ones. |

If a piece needs llama.cpp itself changed (router-mode memory management, `--models-memory-margin` counting host RAM), it is not done here and nothing in this section needs a patch; see `patches/llama.cpp/README.md` for how such patches are kept.

## Compatibility in detail (honest status)

The plugin is a ggml backend that manages expert weights in CPU-side memory. It does not run any GPU code itself: the GPU part of a model is whatever your
llama.cpp build uses (CUDA, ROCm/HIP, Vulkan, SYCL, ...). That makes it independent of the GPU vendor in principle, but **only what is marked "tested" has been run**.

### Status by platform

| Platform | Status | Notes |
|---|---|---|
| Linux + Intel CPU + NVIDIA GPU (CUDA) | **Tested** | all results in `RESULTS.md` |
| Linux + AMD CPU (x86-64) | **Tested** (Ryzen 7 5800H, Zen 3, 2026-10-16): identity 4/4 with forced eviction, executor byte-checked (0 mismatches); Qwen3.6 at 12 GB (speed figures follow in a later release) | the plugin uses llama.cpp's own CPU kernels, nothing Intel-specific; needs the AVX2 CPU variant like llama.cpp itself |
| Linux, CPU only (no GPU usable by llama.cpp) | **Tested**: identity 4/4; speed figures follow in a later release | stock llama.cpp repacks CPU expert weights when no GPU is present; the plugin does the same per expert (`MOE_CACHE_REPACK=auto`). With a memory limit smaller than the model (Qwen3.6, 12 GB, stock needs `--no-repack`) the plugin can be faster; measured numbers follow in a later release. |
| Linux + Vulkan, Intel integrated GPU | **Tested**: identity 4/4 (Granite), plugin active (3,549 expert loads, 97 % hits) | llama.cpp built from the same commit with `GGML_VULKAN=ON GGML_BACKEND_DL=ON GGML_CPU_ALL_VARIANTS=ON` (how official releases are built); loaded with `GGML_BACKEND_PATH` |
| Linux + Vulkan, NVIDIA GPU | **Tested**: identity 4/4 | same build |
| Linux + Vulkan, AMD integrated GPU | **Tested** (Vega 8, RADV, 3 GiB carve-out, 2026-10-16): identity 4/4 (Granite with eviction, Qwen3.6 at 12 GB) | `install.sh --vulkan`; the user must be in the `render` group (install.sh now warns). `--gpu-cache` is not available with this loading method (it gave wrong tokens before the fix, RESULTS 50 B1); the long-prompt GPU path uses the iGPU since cffa6f9 |
| Linux + Vulkan, AMD discrete GPU | **Tested** (GitHub issue #1, RX 9070 XT + Ryzen 7 7700, Linux): identity PASS; speed figures follow in a later release; also on Windows (next row) | the path AMD cards without ROCm support use |
| Linux + AMD GPU via ROCm/HIP | Expected to work, **untested** on Linux. Windows: the plugin loads into llama.cpp's official b11669 HIP zip on an RX 9070 XT, not identity-checked | HIP was slower than Vulkan on that card even without the plugin; use Vulkan |
| Linux + Intel Arc (discrete) GPU, or SYCL on any Intel GPU | **Untested** (the Intel integrated GPU via Vulkan is tested, row above; Windows Arc 140V iGPU: RESULTS 49) | no Arc card available to the developer |
| Windows 11 + AMD (desktop, 2026-10-26) | **Tested**: Ryzen 9 7950X (Zen 4, AVX-512) + RX 9070 XT, llama.cpp b11669 Vulkan (ggml 0.27.0, API 3): identity **PASS 4/4** on Granite 1B (evicting), Qwen3.6 35B, Qwen3.5-122B IQ2_XXS (bigger than RAM); Flash-Next via the launcher chat OK; executor byte check 0 mismatches (9,069 ops) | Caveats: Flash-Next needs `--ctx-checkpoints 0` on AMD's Windows Vulkan driver older than 32.0.32015.7029 (saving a checkpoint crashes `amdvlk64.dll`; stock llama-server too; the launcher sets it). The plugin's own Q2_0 kernel runs inside the executor on Windows since 2026-10-28 (no speed change on the PC); the IQ3_XXS gather and AVX-512 kernels are off on AMD, and the LD_PRELOAD-interposed versions are Linux-only. The AVX-512 kernels are bit-exact but slower than llama.cpp's on Zen 4, as is the IQ3 gather kernel, so they stay off on AMD. The HIP release zip lists no device and probably needs AMD's HIP SDK (untested). Speed: the October runs of RESULTS 55 were taken before the file-view cache and the I37 fix and are superseded. Speed figures for this PC follow in a later release (see [WINDOWS.md](../guides/windows.md) section 6). The RAM that GPU layers seemed to cost was llama.cpp's model-file pages left in llama-server's working set (I37), not the driver; the plugin releases them, and a small page file limits the cache ([WINDOWS.md](../guides/windows.md)) |
| Windows (any other CPU/GPU) | **Experimental**: identity 4/4 with the cache on and forced eviction on a GitHub Windows runner (official CPU release: b11669 on 2026-10-16, after three Windows-only bugs were fixed: cffa6f9, 87dd800, 150c8fe; the Windows CI row moved to b11966 on 2026-10-30 and to b12064 on 2026-11-03); executor byte-checked, 0 mismatches. On real hardware: MSI Claw rounds 1-2 found the bugs; the 2026-10-26 desktop run (row above) passed; MSI Claw round 3 (Vulkan iGPU) still pending | see "Windows status" below. Pass-through mode ran fine. `--gpu-cache` is not available on Windows (needs LD_PRELOAD) |
| macOS / Apple Silicon | Not supported | uses Linux memory APIs; unified memory changes the problem |

### Repacking (why CPU-only needed extra work)

With a GPU backend present (CUDA, Vulkan, ...) llama.cpp stores CPU-side expert weights in that backend's host buffer, unrepacked. With no GPU backend it stores them in a repacked CPU layout (for example `q4_0_8x8`), which changes the order of the floating-point sums.
moe-cache repacks each expert as it loads it, exactly when stock would (`MOE_CACHE_REPACK=auto`), so both cases give bit-identical results. `MOE_CACHE_REPACK=on` forces repacking even with a GPU (measured: no decode speedup, so it is off by default there).

### Windows status

- The operating-system code is in `src/platform.h` (Linux: `O_DIRECT`, `madvise`, `mmap`, cgroups; Windows: `FILE_FLAG_NO_BUFFERING` overlapped reads, `VirtualAlloc` reserve/commit/decommit, job-object and free-memory limits).
- First run on real Windows (2026-10-15, MSI Claw 8 AI+, official llama.cpp b11669 Vulkan, ggml 0.27.0): the plugin DLL did not load: it imported
  `libwinpthread-1.dll` (not on Windows) and took 4 backend-registry functions from `ggml-base.dll` instead of `ggml.dll`. Both fixed in `build.sh --windows`
  (0b7bbaa); the identity test is to be rerun. The Python launcher works there (check, dry run, real start, chat).
- Round 2 (2026-10-16): the DLL loads and ggml 0.27.0 accepts it; pass-through mode runs, cache mode crashed at the first graph with an access violation
  in the plugin's compute buffer (reserved, never committed: Windows faults on such pages, Linux hands out zero pages). Fixed in cffa6f9; round 3 pending.
- Desktop run (2026-10-26, Ryzen 9 7950X + RX 9070 XT, b11669 Vulkan): identity 4/4 on real hardware with a GPU, see the status row. Launcher fixes it found: AMD/Intel GPU detection (e36344b), the Flash-Next checkpoint crash workaround (7a1b86f), profile saves (b1fc10b).
- Windows CI (`.github/workflows/windows.yml`, 2026-10-16): DLL cross-built as `build.sh --windows`, official b11669 CPU release, `windows_test.ps1`.
  It found a second commit bug (a commit range past the end of a reservation committed nothing; fixed 87dd800) and a third (`--no-repack` was not
  seen by the plugin on Windows; fixed 150c8fe), then passed 4/4 with forced eviction and 4/4 with the executor forced on (6,765 ops byte-checked,
  0 mismatches). `moe-cache-check MODEL --run` passes there too.
- Not available on Windows: `--gpu-cache` / `MOE_CACHE_GPU_CACHE=1` (needs LD_PRELOAD symbol interposition), the file-mapped fallback for a cache smaller
  than one op (loads over the size instead). Core pinning: implemented from the cores' efficiency class (f7721a6); on the 7950X (no hybrid cores) it correctly does nothing. Pinning covers processor group 0 only (up to 64 logical CPUs, like llama.cpp's own `-C`): with more, the launcher does not pin and says so.
- `MOE_CACHE_WIN_BUFFERED` (idea I38): how experts are read into private memory. The default file-view cache already pages through the Windows file cache. Without views (`MOE_CACHE_WIN_VIEW=0`, the expert pack, a failed view map) reads now also go through the file cache by default; `=0` reads unbuffered, `=1` also forces it for a run with views, `=2` adds random access. Linux has no such switch: its private-memory path stays on O_DIRECT, with no measurement either way.
- `./build.sh --windows` cross-compiles `ggml-moe-cache.dll` with mingw-w64 (load it with `GGML_BACKEND_PATH`; official Windows llama.cpp builds load their backends as DLLs).
- `tests/platform_test.cpp` passes natively on Linux and, compiled for Windows, under Wine (reserve/commit/decommit, unbuffered reads at the end of a file, limits, file replacement).
- Identity passes on a Windows CI runner (CPU) and on the AMD desktop with a GPU; speed figures follow in a later release; the Python launcher (`bin/moe_cache.py`) runs there, the bash wrappers do not. Treat it as experimental; `docs/guides/windows.md` and `tools/windows_test.ps1` are the test.

### Model architectures

Tested on llama.cpp commit `7fe450e19` (ggml 0.25.1) unless the table says otherwise. On commit `23b0202a1` (ggml 0.26.0, plugin built with `GGML_API=3`, 2026-10-11) token identity vs stock was re-checked, 4 prompts x 48 tokens, on: Granite 3.x MoE (3B, 1B), OLMoE 7B, Qwen3.6-35B-A3B and Qwen3.8-Flash-Next IQ2_XS (24 GB limit): all identical. Nothing else was re-run on that commit yet. Release b11966 (`ff0d396e3`, ggml 0.28.0) has had its own identity CI row since 2026-10-30 (identity CI on it passed on 2026-11-02; the headers this repo vendors differ from that commit by new declarations only, `tools/check_ggml_headers.py`); the plugin names that build in its tested list (`0.28.0@ff0d396e3`) and still warns on any other 0.28.0 build.

| Architecture (example model) | Status |
|---|---|
| qwen3moe (Qwen3-Coder 30B-A3B), qwen35moe (Qwen3.6 35B-A3B, KAT-Coder), granitemoe (Granite 3.1 MoE), deepseek2 (DeepSeek-V2-Lite), gpt-oss (gpt-oss-20b / 120b), qwen3next (Qwen3-Next-80B) | **Tested**: token-identical to stock |
| hunyuan-moe (Hunyuan-A13B) | short identity test only |
| qwen4exp (Qwen3.8-Flash-Next, IQ2_XS) | **token-identical to stock** on 4 prompts x 24 tokens in a 24 GB cgroup with 3 expert layers on the GPU (short test, 2026-10-10) |
| lfm2moe, granitehybrid, olmoe, smallthinker, bailingmoe2 | supported by llama.cpp at this commit, **not tested** with the plugin (a speed test of olmoe and smallthinker is in progress) |
| Dense models (Ministral 3 3B and others) | nothing to manage: the plugin has no effect (speed figures follow in a later release) |

Q2_0 (Qwen3.8-Flash-Next) runs through a plugin kernel in preload mode (speed figures follow in a later release), checked bit-identical against llama.cpp's (see TECHNIQUES.md).

Quantisation types seen working: Q4_K, Q6_K, Q5_0, Q8_0, MXFP4, IQ4_XS, IQ2_XS and the Q2_0 type of Qwen3.8-Flash-Next. Files with experts below about 4.25 bits are not possible for models whose expert rows are not a multiple of 256 (gpt-oss).

### Clients and agents (OpenAI-compatible API through llama-server)

| Client | Status |
|---|---|
| OpenCode 1.19.4 | **Tested on Linux** (2026-10-26): `--agent` + `connect opencode --write`; tool calls, file edits, warm sessions after a restart (Qwen3.6, Flash-Next). Windows: not yet |
| Hermes Agent v0.22.5 | **Tested on Linux** (2026-10-26): `--agent` + `connect hermes --write`; file tools, warm sessions. Needs a context of at least 64,000 (`--agent` sets 65536). Windows: not yet |
| llama-server's web UI, Open WebUI, editors | work as with stock llama-server (same API) |
