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

## Prototypes: several GPUs and several models at once (all off by default, untested for speed)

These options exist so a user with more than one GPU, or two models, can try them and report. Each prints "PROTOTYPE" when it is on. Without the flag the launch plan is byte-identical to before (`tests/launch_plan_test.py`, every case). The unit tests run on fake devices and fake servers; **(2026-11-03: the only two-device run so far is a smoke test on the Windows PC, RX 9070 XT + iGPU, Granite: plugin 207-213 vs stock 199 tok/s; the sentence continues as written on 2026-10-30.) Nothing here has been run on a machine with two discrete GPUs, with two models served together through the launcher (the peer rows say what was run by hand on the laptop), or with a real vision model, and no speed gain is claimed.**

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
