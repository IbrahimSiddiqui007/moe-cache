# Compatibility (honest status)

The plugin is a ggml backend that manages expert weights in CPU-side memory. It does not run any GPU code itself: the GPU part of a model is whatever your
llama.cpp build uses (CUDA, ROCm/HIP, Vulkan, SYCL, ...). That makes it independent of the GPU vendor in principle, but **only what is marked "tested" has been run**.

## Status by platform

| Platform | Status | Notes |
|---|---|---|
| Linux + Intel CPU + NVIDIA GPU (CUDA) | **Tested** | all results in `RESULTS.md` |
| Linux + AMD CPU (x86-64) | Expected to work, **untested** | the plugin uses llama.cpp's own CPU kernels, nothing Intel-specific; needs the AVX2 CPU variant like llama.cpp itself |
| Linux, CPU only (no GPU usable by llama.cpp) | **Works, with two limits** | stock llama.cpp repacks CPU expert weights into a faster layout when no GPU is present; the plugin keeps the original layout. Measured on DeepSeek-V2-Lite (model fits in RAM): output not bit-identical (3 of 4 prompts identical over 40 tokens, the rest differ by float rounding) and 24 % slower (8.4 stock vs 6.4 tok/s). When the model does not fit in RAM the cache gain should outweigh this, but that combination is not measured. Repack support is on the roadmap. |
| Linux + Vulkan GPU (AMD, Intel, NVIDIA) | **Untested**, build prepared | needs the Vulkan build of llama.cpp; AMD cards without ROCm support use this path |
| Linux + AMD GPU via ROCm/HIP | Expected to work, **untested** | |
| Linux + Intel GPU (SYCL or Vulkan) | **Untested** | Intel iGPU available on the development machine for testing |
| Windows (any CPU/GPU) | **Not supported yet** | see below |
| macOS / Apple Silicon | Not supported | uses Linux memory APIs; unified memory changes the problem |

## Why the plugin is identical to stock with a GPU but not on CPU only

With a GPU backend present (CUDA, Vulkan, ...) llama.cpp stores CPU-side expert weights in that backend's host buffer, unrepacked; the plugin holds the same layout, so the results are bit-identical.
With no GPU backend llama.cpp uses a repacked CPU buffer instead (for example `q4_0_8x8`), which changes the order of the floating-point sums. A machine whose GPU llama.cpp cannot use (for example an unsupported AMD card with a CPU-only build) falls in this case.

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
