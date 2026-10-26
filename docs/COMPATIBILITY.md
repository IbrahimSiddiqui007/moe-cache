# Compatibility (honest status)

The plugin is a ggml backend that manages expert weights in CPU-side memory. It does not run any GPU code itself: the GPU part of a model is whatever your
llama.cpp build uses (CUDA, ROCm/HIP, Vulkan, SYCL, ...). That makes it independent of the GPU vendor in principle, but **only what is marked "tested" has been run**.

## Status by platform

| Platform | Status | Notes |
|---|---|---|
| Linux + Intel CPU + NVIDIA GPU (CUDA) | **Tested** | all results in `RESULTS.md` |
| Linux + AMD CPU (x86-64) | **Tested** (Ryzen 7 5800H, Zen 3, 2026-10-16): identity 4/4 with forced eviction, executor byte-checked (0 mismatches); Qwen3.6 at 12 GB 1.23x stock | the plugin uses llama.cpp's own CPU kernels, nothing Intel-specific; needs the AVX2 CPU variant like llama.cpp itself (RESULTS 50) |
| Linux, CPU only (no GPU usable by llama.cpp) | **Tested**: identity 4/4 and the same speed as stock | stock llama.cpp repacks CPU expert weights when no GPU is present; the plugin does the same per expert (`MOE_CACHE_REPACK=auto`). With a memory limit smaller than the model: 3.1 -> 9.9 tok/s (Qwen3.6, 12 GB, stock needs `--no-repack`). |
| Linux + Vulkan, Intel integrated GPU | **Tested**: identity 4/4 (Granite), plugin active (3,549 expert loads, 97 % hits) | llama.cpp built from the same commit with `GGML_VULKAN=ON GGML_BACKEND_DL=ON GGML_CPU_ALL_VARIANTS=ON` (how official releases are built); loaded with `GGML_BACKEND_PATH` |
| Linux + Vulkan, NVIDIA GPU | **Tested**: identity 4/4 | same build |
| Linux + Vulkan, AMD integrated GPU | **Tested** (Vega 8, RADV, 3 GiB carve-out, 2026-10-16): identity 4/4 (Granite with eviction, Qwen3.6 at 12 GB) | `install.sh --vulkan`; the user must be in the `render` group (install.sh now warns). `--gpu-cache` is not available with this loading method (it gave wrong tokens before the fix, RESULTS 50 B1); the long-prompt GPU path uses the iGPU since cffa6f9 |
| Linux + Vulkan, AMD discrete GPU | **Tested** (GitHub issue #1, RX 9070 XT + Ryzen 7 7700, Linux): identity PASS, +18 % speed; also on Windows (next row) | the path AMD cards without ROCm support use |
| Linux + AMD GPU via ROCm/HIP | Expected to work, **untested** (Windows HIP zip: see the Windows rows) | |
| Linux + Intel GPU (SYCL or Vulkan) | **Untested** | Intel iGPU available on the development machine for testing |
| Windows 11 + AMD (desktop, 2026-10-26) | **Tested**: Ryzen 9 7950X (Zen 4, AVX-512) + RX 9070 XT, llama.cpp b11669 Vulkan (ggml 0.27.0, API 3): identity **PASS 4/4** on Granite 1B (evicting), Qwen3.6 35B, Qwen3.5-122B IQ2_XXS (bigger than RAM); Flash-Next via the launcher chat OK; executor byte check 0 mismatches (9,069 ops) | Caveats: Flash-Next needs `--ctx-checkpoints 0` on AMD's Windows Vulkan driver (saving a checkpoint crashes `amdvlk64.dll`; stock llama-server too; the launcher sets it). The plugin's own CPU kernels (Q2_0, IQ3_XXS gather, AVX-512) are Linux-only (they need LD_PRELOAD interposition). The AVX-512 kernels are bit-exact but 0.60-0.77x of llama.cpp's on Zen 4, the IQ3 gather kernel 0.58-0.64x, so they stay off on AMD. The HIP release zip lists no device and probably needs AMD's HIP SDK (untested). No speed numbers with the plugin yet |
| Windows (any other CPU/GPU) | **Experimental**: identity 4/4 with the cache on and forced eviction on a GitHub Windows runner (official b11669 CPU release, 2026-10-16, after three Windows-only bugs were fixed: cffa6f9, 87dd800, 150c8fe); executor byte-checked, 0 mismatches. On real hardware: MSI Claw rounds 1-2 found the bugs; the 2026-10-26 desktop run (row above) passed; MSI Claw round 3 (Vulkan iGPU) still pending | see "Windows status" below, RESULTS 49-50. Pass-through mode ran fine. `--gpu-cache` is not available on Windows (needs LD_PRELOAD) |
| macOS / Apple Silicon | Not supported | uses Linux memory APIs; unified memory changes the problem |

## Repacking (why CPU-only needed extra work)

With a GPU backend present (CUDA, Vulkan, ...) llama.cpp stores CPU-side expert weights in that backend's host buffer, unrepacked. With no GPU backend it stores them in a repacked CPU layout (for example `q4_0_8x8`), which changes the order of the floating-point sums.
moe-cache repacks each expert as it loads it, exactly when stock would (`MOE_CACHE_REPACK=auto`), so both cases give bit-identical results. `MOE_CACHE_REPACK=on` forces repacking even with a GPU (measured: no decode speedup, so it is off by default there).

## Windows status

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
- `./build.sh --windows` cross-compiles `ggml-moe-cache.dll` with mingw-w64 (load it with `GGML_BACKEND_PATH`; official Windows llama.cpp builds load their backends as DLLs).
- `tests/platform_test.cpp` passes natively on Linux and, compiled for Windows, under Wine (reserve/commit/decommit, unbuffered reads at the end of a file, limits, file replacement).
- Identity passes on a Windows CI runner (CPU) and on the AMD desktop with a GPU; no speed numbers with the plugin yet; the Python launcher (`bin/moe_cache.py`) runs there, the bash wrappers do not. Treat it as experimental; `docs/WINDOWS_TEST.md` and `tools/windows_test.ps1` are the test.

## Model architectures

Tested on llama.cpp commit `7fe450e19` (ggml 0.25.1) unless the table says otherwise. On commit `23b0202a1` (ggml 0.26.0, plugin built with `GGML_API=3`, 2026-10-11) token identity vs stock was re-checked, 4 prompts x 48 tokens, on: Granite 3.x MoE (3B, 1B), OLMoE 7B, Qwen3.6-35B-A3B and Qwen3.8-Flash-Next IQ2_XS (24 GB limit): all identical. Nothing else was re-run on that commit yet.

| Architecture (example model) | Status |
|---|---|
| qwen3moe (Qwen3-Coder 30B-A3B), qwen35moe (Qwen3.6 35B-A3B, KAT-Coder), granitemoe (Granite 3.1 MoE), deepseek2 (DeepSeek-V2-Lite), gpt-oss (gpt-oss-20b / 120b), qwen3next (Qwen3-Next-80B) | **Tested**: token-identical to stock |
| hunyuan-moe (Hunyuan-A13B) | short identity test only |
| qwen4exp (Qwen3.8-Flash-Next, IQ2_XS) | **token-identical to stock** on 4 prompts x 24 tokens in a 24 GB cgroup with 3 expert layers on the GPU (short test, 2026-10-10) |
| lfm2moe, granitehybrid, olmoe, smallthinker, bailingmoe2 | supported by llama.cpp at this commit, **not tested** with the plugin (a speed test of olmoe and smallthinker is in progress) |
| Dense models (Ministral 3 3B and others) | nothing to manage: the plugin has no effect (a few percent slower than stock in a CPU-only test) |

Q2_0 (Qwen3.8-Flash-Next) runs through a faster plugin kernel in preload mode, checked bit-identical against llama.cpp's (see TECHNIQUES.md).

Quantisation types seen working: Q4_K, Q6_K, Q5_0, Q8_0, MXFP4, IQ4_XS, IQ2_XS and the Q2_0 type of Qwen3.8-Flash-Next. Files with experts below about 4.25 bits are not possible for models whose expert rows are not a multiple of 256 (gpt-oss).

## Clients and agents (OpenAI-compatible API through llama-server)

| Client | Status |
|---|---|
| OpenCode 1.19.4 | **Tested on Linux** (2026-10-26): `--agent` + `connect opencode --write`; tool calls, file edits, warm sessions after a restart (Qwen3.6, Flash-Next). Windows: not yet |
| Hermes Agent v0.22.5 | **Tested on Linux** (2026-10-26): `--agent` + `connect hermes --write`; file tools, warm sessions. Needs a context of at least 64,000 (`--agent` sets 65536). Windows: not yet |
| llama-server's web UI, Open WebUI, editors | work as with stock llama-server (same API) |
