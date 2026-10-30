# Hardware, operating systems and limits

Moved here from the README on 2026-10-30, unchanged.

## Hardware and OS support

| Platform | Status |
|---|---|
| Linux + Intel CPU + NVIDIA GPU (CUDA) | **Tested** |
| Linux + Vulkan (NVIDIA, and the Intel integrated GPU) | **Tested** on llama.cpp `7fe450e19` only, loaded the official way (`GGML_BACKEND_PATH`; see the feature table below) |
| Linux, no usable GPU (CPU only) | **Tested**, identical to stock, within about 4 % of its speed when the model fits |
| Linux + AMD CPU, AMD integrated GPU (Vulkan) | **Tested** on a Ryzen 7 5800H with Vega 8 (2026-10-16): identical to stock, Qwen3.6 at 12 GB 1.23x stock (no expert layer fits on a 3 GiB iGPU); RESULTS 50 |
| Linux + AMD discrete GPU | **Tested by a user** (GitHub issue #1: RX 9070 XT + Ryzen 7 7700, Linux, Vulkan): identity PASS, Flash-Next 15.9 -> 18.8 tok/s. Not yet run by the developer on Linux |
| Windows | **Experimental, now faster than stock.** Identical to stock in CI on a Windows runner and on a Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB, Vulkan). Since 2026-10-27 the cache keeps experts in read-only views of the model file (Windows charged its private memory, and Vulkan's GPU memory, to commit, which held the cache at 8-11 GiB). Cold, stock vs moe-cache, launcher defaults (RESULTS 59-60): Flash-Next Q2_0 (the recommended file) **22.2 vs 18.4 tok/s**, prompts 2.3x (RESULTS 63), IQ2_XS 21.9 vs 14.9, IQ3_XXS 19.9 vs 15.1, IQ3_S 17.3 vs 12.9, Qwen3.5-122B 19.7 vs 16.0 (re-measured 2026-10-28, RESULTS 61; replaces 16.1 vs 13.9, which ran during a PC-wide slowdown), GLM-5.3 4.4 vs 2.2 (**6.9 with `--mtp`**); text byte-identical. Context checkpoints stay on with AMD driver 32.0.32015.7029 or newer (older ones crash with Flash-Next). See docs/COMPATIBILITY.md |
| macOS | No |

## Known limitations

- Long prompts (32+ tokens per batch) run their expert blocks on the GPU, as stock llama.cpp does, so prompt processing is at stock speed when the model fits (DeepSeek-V2-Lite: 166 to 183 tok/s vs 166 to 188 stock) and much faster when memory is short.
- It does not manage the GPU; which layers' experts sit in VRAM is chosen once at start by the planner. A GPU expert tier was evaluated and is not built, see [docs/ENGINE_DECISION.md](ENGINE_DECISION.md).
- **Not every feature works with every llama.cpp build.** How the plugin is loaded depends on how llama.cpp was built (`moe-cache-server` picks it):

  | Feature | `LD_PRELOAD` (llama.cpp that links its backends directly: distro packages, your own CUDA build) | `GGML_BACKEND_PATH` (backends as separate files: official releases, `./install.sh`, Vulkan builds, Windows) |
  |---|---|---|
  | RAM expert cache, SSD streaming, memory limit, warm start | yes | yes |
  | Expert executor (computes ready experts while others load) | yes | yes (since 0.2.3.3; tested on a Vulkan build) |
  | Fast Q2_0 / IQ3_XXS kernels | yes | inside the executor only (Linux, and Windows x86-64 since 2026-10-28; no speed change on the PC, RESULTS 63 addendum) |
  | Working together with llama.cpp's GPU expert cache (`--gpu-cache`) | yes | **no** (refused since cffa6f9: it gave wrong tokens; the launcher uses static GPU expert layers instead) |

  The `moe-cache: active: ...` line in the server log says which parts are on.
- Needs a llama.cpp with shared ggml libraries; tested with commits `23b0202a1` (ggml 0.26.0) and `7fe450e19` (ggml 0.25.1), and release b11966 (ggml 0.28.0; CI row added 2026-10-30). The plugin refuses to load into a ggml with a different backend API than it was built for (it prints why and the model loads normally). Linux; Windows is experimental (identity-tested on a Windows 11 + RX 9070 XT desktop and in CI, see [COMPATIBILITY](COMPATIBILITY.md) and the [Windows quick start](WINDOWS.md)).
