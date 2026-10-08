# Quick start (Linux)

You need: Linux, `llama-server` from a **shared-library** llama.cpp build (a distro package or an official release), `g++`, `python3`, and a MoE model in GGUF format.
The plugin was built and tested against **llama.cpp commit 7fe450e19 (ggml 0.25.1)**. Other versions may work; the plugin prints a warning if the version differs.
Currently tested on: Linux, Intel CPU, NVIDIA GPU. Everything else is in `COMPATIBILITY.md`.

## 0. If your distro's llama.cpp is not usable: build the matching llama.cpp with Vulkan (about 20 minutes, once)

Works on any Linux GPU that has a Vulkan driver (AMD with the open Mesa/RADV driver, Intel, NVIDIA). This is exactly how we built and tested the Vulkan setup.

```bash
# packages: cmake ninja g++ and the Vulkan development files + glslc + SPIR-V headers
#   Ubuntu/Debian: sudo apt install cmake ninja-build g++ libvulkan-dev glslc spirv-headers mesa-vulkan-drivers
#   Fedora:        sudo dnf install cmake ninja-build gcc-c++ vulkan-devel glslc spirv-headers-devel mesa-vulkan-drivers
#   openSUSE:      sudo zypper install cmake ninja gcc-c++ vulkan-devel shaderc spirv-headers
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 7fe450e19
cmake -B build -G Ninja -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server -j8
./build/bin/llama-server --list-devices        # your GPU should be listed as Vulkan0
```

Then in the moe-cache folder: `GGML_LIB_DIR=/path/to/llama.cpp/build/bin ./build.sh` and use `--llama-server /path/to/llama.cpp/build/bin/llama-server` below.
The launcher detects this kind of build and loads the plugin with `GGML_BACKEND_PATH` automatically. Package names differ between distro versions; the tools you need are cmake, ninja, a C++ compiler, the Vulkan headers/loader, `glslc` and the SPIR-V headers.

## 1. Get and build it (2 minutes)

```bash
git clone https://github.com/IbrahimSiddiqui007/moe-cache && cd moe-cache
./build.sh                       # finds libggml-base.so next to your llama-server; or: GGML_LIB_DIR=/path/to/libs ./build.sh
pip install gguf                 # only needed for the automatic GPU/CPU split
```

## 2. Check that it is correct on YOUR model (1-5 minutes)

```bash
tests/identity.sh /path/to/model.gguf --llama-server /path/to/llama-server
```

It runs the model once without and once with the plugin and compares the generated tokens. You want `PASS` (identical). Do this once per model.

## 3. Run it

```bash
bin/moe-cache-server /path/to/model.gguf --ram 12 --ctx 8192 --llama-server /path/to/llama-server --open
```

- `--ram 12` is how much memory (GB) the server may use. The cache sizes itself to stay under it. Leave it out to use the free RAM.
- The tool asks the planner how many layers' experts can live on your GPU and keeps the rest in the managed cache.
- `--open` opens the web UI in your browser (it is the built-in llama.cpp UI at http://localhost:8080). The same address is an OpenAI-compatible API, so tools like Open WebUI, editors and scripts can connect to it.
- `--dry-run` prints the exact command without starting anything.
- Force the split yourself with `--n-cpu-moe N` (N layers keep their experts on the CPU).

## 4. Make it faster for you (optional)

The first run learns which experts your prompts use and saves a profile in `~/.cache/moe-cache/`. Later runs warm-start from it.
To prepare a profile before you start (or use your own prompts):

```bash
bin/moe-cache-learn /path/to/model.gguf --ram 20 --llama-server /path/to/llama-server [--prompts my_prompts.txt]
```

## What speed to expect

Rule of thumb from our measurements (see `RESULTS.md`): if the model fits in your RAM you will see about the same speed as stock (within a few percent);
if it does not fit, expect 2x to 7x stock. Example: a 48 GB model in a 20 GB memory limit ran at 9 tok/s instead of 1.3.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `libggml-base.so not found` | `GGML_LIB_DIR=/dir/with/libggml-base.so ./build.sh` |
| `plugin disabled` message | `MOE_CACHE_GGUF` must be the model file the server loads; with `moe-cache-server` this is automatic |
| Warning about an untested ggml version | Usually fine. If the server crashes at start, rebuild against matching headers or set `MOE_CACHE_DISABLE=1` to run without the plugin |
| Server killed by the system ("out of memory") | Lower `--ram`, or close other programs; the cache keeps its own memory under `--ram` but the rest of the system matters |
| Slower than stock | Normal when the model fits in RAM and the cache is cold: run `bin/moe-cache-learn` once, or just use it for a while |
