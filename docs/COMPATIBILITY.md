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
| Windows (any CPU/GPU) | **Not supported yet** | see below |
| macOS / Apple Silicon | Not supported | uses Linux memory APIs; unified memory changes the problem |

## Repacking (why CPU-only needed extra work)

With a GPU backend present (CUDA, Vulkan, ...) llama.cpp stores CPU-side expert weights in that backend's host buffer, unrepacked. With no GPU backend it stores them in a repacked CPU layout (for example `q4_0_8x8`), which changes the order of the floating-point sums.
moe-cache repacks each expert as it loads it, exactly when stock would (`MOE_CACHE_REPACK=auto`), so both cases give bit-identical results. `MOE_CACHE_REPACK=on` forces repacking even with a GPU (measured: no decode speedup, so it is off by default there).

## Why Windows needs work

Linux-specific parts of the plugin: `O_DIRECT` reads, `madvise(DONTNEED)` to give memory back, `mmap` reservations, `/proc` and cgroup files to read memory use and limits.
Windows equivalents exist (`FILE_FLAG_NO_BUFFERING`, `VirtualAlloc`/`DiscardVirtualMemory`, `GetProcessMemoryInfo`, job objects) but are not written or tested.
Official Windows llama.cpp builds load their backends as separate DLLs, so the plugin could be loaded with `GGML_BACKEND_PATH` (no `LD_PRELOAD` needed). That path is already used by the launcher for such builds.
Estimated effort: a few days including testing.

## What a user needs for any platform

1. A llama.cpp whose ggml version matches the headers in `third_party/ggml` (currently 0.25.1, backend API version 2); the plugin warns when it differs.
2. Shared ggml libraries (`libggml-base.so`), not a fully static `llama-server`.
3. A single-file or split GGUF model with MoE expert tensors (any architecture llama.cpp runs through `MUL_MAT_ID`).

## Models tested

Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B, KAT-Coder-35B-A3B, Qwen3-Next-80B-A3B, Hunyuan-A13B, DeepSeek-V2-Lite, gpt-oss-20b, Granite-3.1-MoE-3B (Q4_K_M, MXFP4; IQ4_XS and Q5_0 via a requantized model).
