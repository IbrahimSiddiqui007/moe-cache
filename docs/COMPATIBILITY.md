# Compatibility (honest status)

The plugin is a ggml backend that manages expert weights in CPU-side memory. It does not run any GPU code itself: the GPU part of a model is whatever your
llama.cpp build uses (CUDA, ROCm/HIP, Vulkan, SYCL, ...). That makes it independent of the GPU vendor in principle, but **only what is marked "tested" has been run**.

## Status by platform

| Platform | Status | Notes |
|---|---|---|
| Linux + Intel CPU + NVIDIA GPU (CUDA) | **Tested** | all results in `RESULTS.md` |
| Linux + AMD CPU (x86-64) | Expected to work, **untested** | the plugin uses llama.cpp's own CPU kernels, nothing Intel-specific; needs the AVX2 CPU variant like llama.cpp itself |
| Linux, CPU only (no GPU usable by llama.cpp) | **Tested**: identity 4/4 and the same speed as stock | stock llama.cpp repacks CPU expert weights when no GPU is present; the plugin does the same per expert (`MOE_CACHE_REPACK=auto`). With a memory limit smaller than the model: 3.1 -> 9.9 tok/s (Qwen3.6, 12 GB, stock needs `--no-repack`). |
| Linux + Vulkan, Intel integrated GPU | **Tested**: identity 4/4 (Granite), plugin active (3,549 expert loads, 97 % hits) | llama.cpp built from the same commit with `GGML_VULKAN=ON GGML_BACKEND_DL=ON GGML_CPU_ALL_VARIANTS=ON` (how official releases are built); loaded with `GGML_BACKEND_PATH` |
| Linux + Vulkan, NVIDIA GPU | **Tested**: identity 4/4 | same build |
| Linux + Vulkan, AMD GPU | Expected to work (same Vulkan path, same host-buffer behaviour as the tested devices), **not tested on AMD hardware** | the path AMD cards without ROCm support use |
| Linux + AMD GPU via ROCm/HIP | Expected to work, **untested** | |
| Linux + Intel GPU (SYCL or Vulkan) | **Untested** | Intel iGPU available on the development machine for testing |
| Windows (any CPU/GPU) | **Not supported yet** (code written, never run on Windows) | see "Windows status" below |
| macOS / Apple Silicon | Not supported | uses Linux memory APIs; unified memory changes the problem |

## Repacking (why CPU-only needed extra work)

With a GPU backend present (CUDA, Vulkan, ...) llama.cpp stores CPU-side expert weights in that backend's host buffer, unrepacked. With no GPU backend it stores them in a repacked CPU layout (for example `q4_0_8x8`), which changes the order of the floating-point sums.
moe-cache repacks each expert as it loads it, exactly when stock would (`MOE_CACHE_REPACK=auto`), so both cases give bit-identical results. `MOE_CACHE_REPACK=on` forces repacking even with a GPU (measured: no decode speedup, so it is off by default there).

## Windows status

- The operating-system code is in `src/platform.h` (Linux: `O_DIRECT`, `madvise`, `mmap`, cgroups; Windows: `FILE_FLAG_NO_BUFFERING` overlapped reads, `VirtualAlloc` reserve/commit/decommit, job-object and free-memory limits).
- `./build.sh --windows` cross-compiles `ggml-moe-cache.dll` with mingw-w64 (load it with `GGML_BACKEND_PATH`; official Windows llama.cpp builds load their backends as DLLs).
- `tests/platform_test.cpp` passes natively on Linux and, compiled for Windows, under Wine (reserve/commit/decommit, unbuffered reads at the end of a file, limits, file replacement).
- **Never run on real Windows with llama.cpp**: no token-identity test, no speed numbers, and the launcher scripts are still bash. Treat it as a code port waiting for a tester.

## Model architectures (llama.cpp commit 7fe450e19)

| Architecture (example model) | Status |
|---|---|
| qwen3moe (Qwen3-Coder 30B-A3B), qwen35moe (Qwen3.6 35B-A3B, KAT-Coder), granitemoe (Granite 3.1 MoE), deepseek2 (DeepSeek-V2-Lite), gpt-oss (gpt-oss-20b / 120b), qwen3next (Qwen3-Next-80B) | **Tested**: token-identical to stock |
| hunyuan-moe (Hunyuan-A13B) | short identity test only |
| qwen4exp (Qwen3.8-Flash-Next, IQ2_XS) | **token-identical to stock** on 4 prompts x 24 tokens in a 24 GB cgroup with 3 expert layers on the GPU (short test, 2026-10-10) |
| lfm2moe, granitehybrid, olmoe, smallthinker, bailingmoe2 | supported by llama.cpp at this commit, **not tested** with the plugin (a speed test of olmoe and smallthinker is in progress) |
| Dense models (Ministral 3 3B and others) | nothing to manage: the plugin has no effect (a few percent slower than stock in a CPU-only test) |

Quantisation types seen working: Q4_K, Q6_K, Q5_0, Q8_0, MXFP4, IQ4_XS, IQ2_XS and the Q2_0 type of Qwen3.8-Flash-Next. Files with experts below about 4.25 bits are not possible for models whose expert rows are not a multiple of 256 (gpt-oss).
