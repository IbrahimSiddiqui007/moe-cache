# GPU backends: NVIDIA, AMD (ROCm / HIP), Intel Arc (SYCL / Vulkan), Vulkan

moe-cache talks to llama.cpp only through ggml's generic backend interface, so it is not tied to CUDA. This page says what is **verified** and what is
**expected to work but untested**, and how to build llama.cpp for each GPU so a tester can try it.

| Backend | Status | Notes |
|---|---|---|
| CUDA (NVIDIA) | **tested** (RTX 4060 Laptop, CUDA 13) | plugin, executor, llama.cpp GPU expert cache, identity |
| Vulkan on an Intel iGPU | **tested** earlier (llama.cpp 7fe450e19, loadable backends) | plugin and identity; not re-run on the newer commit |
| Vulkan on AMD / Intel Arc / NVIDIA | untested | expected to behave like the Intel iGPU case |
| ROCm / HIP (AMD) | **loads, not identity-checked** (Windows, official b11669 HIP zip, RX 9070 XT; RESULTS 67) | generic ggml calls only; the llama.cpp GPU expert cache is backend-neutral upstream code. On that card HIP is 2.4x slower than Vulkan even without the plugin, and the launcher reads no GPU memory from a HIP build (it plans every expert layer on the CPU) |
| SYCL (Intel Arc, Intel GPUs) | **untested** | same as above |
| Intel NPU (Core Ultra, Lunar Lake) | **untested** | llama.cpp has an OpenVINO backend (`-DGGML_OPENVINO=ON`, `GGML_OPENVINO_DEVICE=NPU`), labelled a preview. Its formats are FP16, Q8_0, Q4_0/Q4_1/Q4_K (Q5_K/Q6_K converted at load), so IQ2 / Q2_0 models such as Qwen3.8-Flash-Next cannot use it; a Q4_K MoE can. Not tried with this plugin |
| AMD NPU (XDNA, Ryzen AI) | not supported | no ggml backend for it; AMD's own Ryzen AI software is separate |

## Build llama.cpp for your GPU (shared libraries, as moe-cache needs)

Commit `23b0202a1` (ggml 0.26.0) or newer for the GPU expert cache; `7fe450e19` also works without it. Check llama.cpp's own `docs/build.md` if a flag has moved.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 23b0202a1
# NVIDIA
cmake -B build -G Ninja -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
# AMD ROCm / HIP: find your GPU target with `rocminfo | grep gfx` (for example gfx1100, gfx1151 for Strix Halo)
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -B build -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
# Intel Arc / Intel GPUs with oneAPI
source /opt/intel/oneapi/setvars.sh
cmake -B build -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
# any GPU with a Vulkan driver (AMD, Intel, NVIDIA): the easiest first try
cmake -B build -G Ninja -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
./build/bin/llama-server --list-devices        # your GPU must appear here
```

Then, in the moe-cache folder:

```bash
GGML_API=3 GGML_LIB_DIR=/path/to/llama.cpp/build/bin ./build.sh      # llama.cpp 23b0202a1 or newer (use ./build.sh alone for 7fe450e19)
tools/friend_test.sh --llama-server /path/to/llama.cpp/build/bin/llama-server --model /path/to/small-moe.gguf
```

`tools/friend_test.sh` writes one report file (hardware names, results, no prompts or paths); see `docs/FRIEND_TEST.md`.

## What to expect on a unified-memory machine (Ryzen AI Max, Apple-like layouts)

When the whole model fits in the shared memory the plugin has nothing to cache and steps aside; the useful settings are then llama.cpp's own
(`-ngl 99`, a larger `-ub`, flash attention). moe-cache matters when the model is bigger than the memory you can give it.
