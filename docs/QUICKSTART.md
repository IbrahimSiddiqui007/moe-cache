# Quick start (Linux)

**The short version** (details below):

```bash
tar xzf moe-cache-vX.Y.Z-ggml0.26-linux-x86_64.tar.gz && cd moe-cache-vX.Y.Z-ggml0.26-linux-x86_64   # 1. get it (Releases page; llama.cpp with ggml 0.25: the file without -ggml0.26)
bin/moe-cache-check /path/to/model.gguf --llama-server /path/to/llama-server              # 2. check
bin/moe-cache-server /path/to/model.gguf --ram 12 --llama-server /path/to/llama-server --open   # 3. run
```

No model yet? `python3 bin/moe_cache.py pick` names one for your machine and prints the download command (`pip install -U huggingface_hub` gives you the `hf` command). Models are 20-120 GB: keep the file size plus about 10 GB of disk free.

**Fastest way on a fresh Ubuntu / Debian machine:** `./install.sh` (see the README). The rest of this page is the manual route.

You need: Linux, `python3`, and a MoE model in GGUF format. With the `-ggml0.26` release file (Option A in section 1) that is all: it includes the official llama.cpp b12064 build, so no compiler and no separate `llama-server` are needed. Only if you build the plugin yourself (Option B) or use another llama.cpp do you also need `g++` and a `llama-server` from a **shared-library** llama.cpp build (a distro package or an official release).
The plugin is built and tested against four llama.cpp versions (the table below), and **a plugin build only works with the ggml backend API it was compiled for**:

| llama.cpp | ggml | backend API | build the plugin with |
|---|---|---|---|
| release **b12064** = commit `51fe42c5d` (2026-11-03; the build the Windows kit and the Linux tarball bundle; the vendored headers differ from it by new declarations and three optional hooks at the end of `ggml_backend_buffer_i`, `tools/check_ggml_headers.py`) | 0.28.0 | 3 | `GGML_API=3 ./build.sh` |
| release b11966 = commit `ff0d396e3` (2026-10-28; the previous pin) | 0.28.0 | 3 | `GGML_API=3 ./build.sh` |
| commit `23b0202a1` (2026-10-10; GLM-5.3-Flash and its MTP support are in) | 0.26.0 | 3 | `GGML_API=3 ./build.sh` (add `BUILD_DIR=build-v3` to keep both builds) |
| commit `7fe450e19` (2026-09-23; Arch `llama.cpp-cuda` 0.5.0-1) | 0.25.1 | 2 | `./build.sh` |

Other versions may work; the plugin prints a warning if the version differs, and refuses to load (the model then loads normally) when it knows the API differs.
Currently tested on: Linux, Intel CPU, NVIDIA GPU. Everything else is in `COMPATIBILITY.md`.

## 0. If your distro's llama.cpp is not usable: build the matching llama.cpp with Vulkan (about 20 minutes, once)

Works on any Linux GPU that has a Vulkan driver (AMD with the open Mesa/RADV driver, Intel, NVIDIA). This is exactly how we built and tested the Vulkan setup.

```bash
# packages: cmake ninja g++ and the Vulkan development files + glslc + SPIR-V headers
#   Ubuntu/Debian: sudo apt install cmake ninja-build g++ libvulkan-dev glslc spirv-headers mesa-vulkan-drivers
#   Fedora:        sudo dnf install cmake ninja-build gcc-c++ vulkan-devel glslc spirv-headers-devel mesa-vulkan-drivers
#   openSUSE:      sudo zypper install cmake ninja gcc-c++ vulkan-devel shaderc spirv-headers
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 7fe450e19   # the commit this recipe was tested on (API 2); for the recommended b12064 use `git checkout 51fe42c5d` and build the plugin with GGML_API=3
cmake -B build -G Ninja -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server -j8
./build/bin/llama-server --list-devices        # your GPU should be listed as Vulkan0
```

Then in the moe-cache folder: `GGML_LIB_DIR=/path/to/llama.cpp/build/bin ./build.sh` (add `GGML_API=3` for 51fe42c5d / b12064) and use `--llama-server /path/to/llama.cpp/build/bin/llama-server` below.
The launcher detects this kind of build and loads the plugin with `GGML_BACKEND_PATH` automatically. Package names differ between distro versions; the tools you need are cmake, ninja, a C++ compiler, the Vulkan headers/loader, `glslc` and the SPIR-V headers.

## 1. Get it (2 minutes)

**Option A, prebuilt plugin (no compiler needed; the `-ggml0.26` file also includes the official llama.cpp b12064 Vulkan + CPU build in `llama/`, which the launcher finds by itself; the other file needs your own llama.cpp of the 7fe450e19 era):** download the newest `moe-cache-vX.Y.Z-ggml0.26-linux-x86_64.tar.gz` (or, for your own older llama.cpp, `moe-cache-vX.Y.Z-linux-x86_64.tar.gz`) from the [Releases page](https://github.com/IbrahimSiddiqui007/moe-cache/releases), check it against `SHA256SUMS`, unpack it and `cd` into the folder. Releases built by GitHub Actions (v0.1.3 and later) can also be verified with `gh attestation verify moe-cache-vX.Y.Z-linux-x86_64.tar.gz -R IbrahimSiddiqui007/moe-cache`, which proves the file was built from the tagged commit.
The plugin in it needs glibc 2.34 or newer and a libstdc++ from GCC 12 or newer (Ubuntu 22.04, Debian 12, Fedora 36 and later; RHEL 9 has an older libstdc++; the release workflow checks both) and a llama.cpp built with shared ggml libraries.
Each release has two Linux files; take the one that matches your llama.cpp's ggml: `moe-cache-vX.Y.Z-ggml0.26-linux-x86_64.tar.gz` for ggml 0.26 and newer (llama.cpp 23b0202a1 and newer, including today's ggml 0.28 releases), `moe-cache-vX.Y.Z-linux-x86_64.tar.gz` for ggml 0.24 / 0.25 (up to commit 7fe450e19). `bin/moe-cache-check` says so when the plugin and your llama-server do not match.

**Option B, build it yourself:**

```bash
git clone https://github.com/IbrahimSiddiqui007/moe-cache && cd moe-cache
GGML_API=3 ./build.sh            # llama.cpp b12064 / 23b0202a1 and newer (backend API 3, the recommended pin); finds libggml-base.so next to your llama-server, or: GGML_LIB_DIR=/path/to/libs GGML_API=3 ./build.sh
./build.sh                       # legacy only: llama.cpp 7fe450e19 / ggml 0.25.1 (backend API 2); the default of build.sh
```

Then, for both options:

```bash
pip install gguf                 # optional: the planner has its own header reader and only falls back to it for older setups
bin/moe-cache-check /path/to/model.gguf --llama-server /path/to/llama-server     # checks your setup, changes nothing
```

`moe-cache-check` tells you with OK / WARN / FAIL whether your llama-server can load the plugin, whether a GPU is found, whether the model is a MoE and can be read with direct reads, and how the model size compares with your memory.

## 2. Check that it is correct on YOUR model (1-5 minutes)

```bash
tests/identity.sh /path/to/model.gguf --llama-server /path/to/llama-server
```

It runs the model once without and once with the plugin and compares the generated tokens. You want `PASS` (identical). Do this once per model.

## 3. Run it

```bash
bin/moe-cache-server /path/to/model.gguf --ram 12 --ctx 8192 --llama-server /path/to/llama-server --open
```

- `--ram 12` is the memory (GB) the cache and the GPU/CPU plan are sized for. It is a sizing hint, not a hard limit (the hard cap is `--max-ram`). Leave it out to use the free RAM.
- The tool asks the planner how many layers' experts can live on your GPU and keeps the rest in the managed cache.
- `--open` opens the web UI in your browser (it is the built-in llama.cpp UI at http://localhost:8080). The same address is an OpenAI-compatible API, so tools like Open WebUI, editors and scripts can connect to it.
- `--dry-run` prints the exact command without starting anything.
- Force the split yourself with `--n-cpu-moe N` (N layers keep their experts on the CPU).
- `--cpu-only` runs on the processor alone (llama-server gets `--device none -ngl 0`). The launcher also chooses it by itself when `llama-server --list-devices` answers and lists no GPU; a `--device` of your own (after `--`) wins.

The `--fast` levels (all optional; each one changes the output a little on purpose, so the default is exact):

| Flag | What it does | Measured (from `--help` and RESULTS) |
|---|---|---|
| (none) | exact: same tokens as stock llama.cpp | the default; at `--ctx 131072` and above the KV cache is q8_0 anyway (add `--exact` to keep f16) |
| `--fast=kv` | only the q8_0 KV cache (half the KV memory), every expert is computed | KL 0.005-0.012 against stock |
| `--fast` | also skips a low-ranked expert that is not in RAM instead of reading it from the SSD | about +10-12 % when the model is much bigger than RAM; about 1 top token in 40-50 differs from stock |
| `--fast=fewer` | `--fast`, and the 2 lowest-ranked experts of every token are never computed on the CPU-held layers | opt-in, new |
| `--fast=2` | up to 4 low-ranked picks may be skipped (at least half are kept), for memory much smaller than the model | Flash-Next at 12 GB: 13.7 vs 9.6 tok/s for `--fast`; KL 0.022; no gain at 24 GB |

## 4. Make it faster for you (optional)

The first run learns which experts your prompts use and saves a profile in `~/.cache/moe-cache/`. Later runs warm-start from it.
To prepare a profile before you start (or use your own prompts):

```bash
bin/moe-cache-learn /path/to/model.gguf --ram 20 --llama-server /path/to/llama-server [--prompts my_prompts.txt]
```

## Measure it on your own model

Run `python3 bin/moe_cache.py bench-report MODEL.gguf` (or press **Share my numbers** on the control page): it runs stock llama.cpp and moe-cache on your model and prints both speeds and whether the tokens were identical. A check with a small model (`moe-cache-check MODEL.gguf --run`) only validates the installation; it says nothing about speed on a big model. The older, separate [moe-cache-benchmark](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark) tool does the same under a memory limit.

## What speed to expect

Rule of thumb from our measurements (see `RESULTS.md`): if the model fits in your RAM, use stock llama.cpp (llama.cpp's own GPU expert cache, `--moe-cache-mib`, is optional there; it was slower on our Vulkan PC, RESULTS 59): moe-cache adds nothing there.
If it does not fit, we measured 2x to 12x stock on Linux and 1.2x to 2x decode on the Windows PC ([RESULTS.md](results-archive/RESULTS.md) for Windows, [RESULTS_OLD_LAPTOP.md](results-archive/RESULTS_OLD_LAPTOP.md) for Linux; the README has a short table). Example: a 48 GB model in a 20 GB memory limit ran at 9.3 tok/s instead of 1.3.

## Which llama-server do I use?

| Your situation | Use |
|---|---|
| Arch / CachyOS with the AUR package `llama.cpp-cuda` | `/usr/bin/llama-server` (this is what we test with) |
| Anything else | the official llama.cpp download for your system (`llama-bNNNNN-bin-ubuntu-*.tar.gz`; built with shared libraries; not tested by us on Linux; the plugin prints a version warning if the version differs), or build the matching commit with `./install.sh` or section 0 above |

## Update, uninstall

Update: `python3 bin/moe_cache.py update --check`, then `update --apply` (or download the newer tarball, or `git pull && ./build.sh`). Uninstall: first `python3 bin/moe_cache.py on-demand --remove-autostart` if you set up the on-demand endpoint (its service would keep starting from the deleted folder), then delete the folder. Other files: `~/.cache/moe-cache/` (usage profiles, control-page settings `ui.json`, the API key `api.key`), benchmark results in `~/.local/share/moe-cache-bench/`, and agent configs that `connect --write` changed (their old versions are kept as `.bak` next to them).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `libggml-base.so not found` | `GGML_LIB_DIR=/dir/with/libggml-base.so ./build.sh` |
| `plugin disabled` message | `MOE_CACHE_GGUF` must be the model file the server loads; with `moe-cache-server` this is automatic |
| Warning about an untested ggml version | Usually fine. If the server crashes at start, rebuild against matching headers or set `MOE_CACHE_DISABLE=1` to run without the plugin |
| Server killed by the system ("out of memory") | Lower `--ram`, or close other programs; the cache sizes itself to `--ram` but is not a hard limit, and the rest of the system matters |
| Slower than stock | Normal when the model fits in RAM and the cache is cold: run `bin/moe-cache-learn` once, or just use it for a while |
