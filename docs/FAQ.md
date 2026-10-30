# FAQ

Moved here from the README on 2026-10-30, unchanged.

### Why it is needed

MoE models such as Qwen3, gpt-oss or DeepSeek only use a few "experts" per token, but stock llama.cpp keeps all expert weights in memory-mapped files.
When they do not fit in RAM, the OS page cache thrashes and speed falls off a cliff.

### When to use something else

| Your situation | Best choice |
|---|---|
| The model **fits in your RAM** | **Stock llama.cpp**, ideally with its GPU expert cache (`--moe-cache-mib N`). That flag is llama.cpp's own GPU expert cache, not moe-cache's: on our Windows Vulkan PC it was slower than without it (RESULTS 59), and its output can differ slightly from stock (not measured by us). On Qwen3.6-35B (21.7 GB) at a 24 GB limit, both with the same 5000 MiB GPU cache: stock 42 tok/s, moe-cache 30-34 (RESULTS section 29). moe-cache adds nothing here but its CPU pinning, which stock gets with `-C MASK --cpu-strict 1`. |
| You want **Qwen3.8-Flash-Next or Qwen3.6-35B as fast as possible** and have **12 GB+ VRAM and 32 GB+ RAM** (more RAM is better) | [Strata](https://github.com/Niko1221/Strata), an engine with its own kernels for those models (its authors report 94 tok/s on an RTX 5070 with 64 GB RAM; see "Strata and Maya today" below). We have not run it ourselves. |
| The model is **bigger than your RAM**, or it is **another MoE architecture**, or you want to stay on **stock llama.cpp with the same tokens** | **moe-cache**: about **1.2-2x faster than plain llama.cpp on Windows and 2-5x on Linux** when the model is bigger than your RAM (Windows since 2026-10-27, RESULTS 59: 1.2x to 2x on the same models, GLM-5.3 3.1x with MTP, coding-agent sessions 2x to 2.4x faster; Linux clean runs since 2026-10-15: 2.4x to 5.2x, the oldest row gpt-oss-120b 12x; tables below). No gain when the model fits in RAM. |


**Do I need this if my model fits in RAM?** No. Use stock llama.cpp with `--moe-cache-mib` (see "When to use something else" above). Caveat: `--moe-cache-mib` is llama.cpp's own GPU expert cache; on our Vulkan PC it was slower (RESULTS 59) and its output can differ slightly from stock, so test it on your card. The gain is when memory is short or the model is larger than your RAM. _work in progress though, I'm currently seeing different ways to improve speed and efficiency_

**Do you check answer quality?** Yes. How, in one page: [docs/EXACTNESS.md](EXACTNESS.md). By default moe-cache is exact below 128k context (at `--ctx 131072` and above the KV cache is q8_0 by default, which passed the quality gate at 4k context; `--exact` keeps f16): with the same layer split the tokens are identical to stock llama.cpp (`tests/identity.sh`, CI on Linux and Windows). The approximate modes (`--fast`, `--fast=kv`, `--fast=fewer`, `--hybrid`) must pass a [quality gate](QUALITY_GATE.md) whose limits were fixed before measuring: KL divergence and same-top-token rate against stock, the 202 exact-answer tasks, and a loop count. Limits: `kv` KL at most 0.015, top-1 at least 95 %, tasks within 2 of stock; `fast` 0.040 / 90 % / 4; `fewer` 0.060 / 88 % / 5. Measured on Flash-Next Q2_0: `kv` KL 0.0079, top-1 96.7 %, 190 tasks = stock; `fewer` KL 0.034, 192 vs 190, unproven at 64k-256k. If answers look wrong or loop on your machine, see [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md).

**What cache policy does it use?** LRU at expert granularity; the experts of the operation that is running are protected. We replayed 13,464 real decode layer-steps of gpt-oss-120b at the real cache size: LRU 84.2 % hits, 2Q 83.6 %, LFU 84.1-84.2 %, LRU with the hottest experts pinned from a profile 84.3-84.8 %, the offline optimum 92.5 %. No policy beat LRU by more than 0.6 points, so LRU stayed (RESULTS section 12, "Smarter eviction").

**Is a very long prompt still slow?** Yes. Prompt reading runs on the GPU and was 1.8x faster than stock at 63k on our Windows PC, but a 63k-token prompt still took about 290 s (RESULTS 64).

**Why is it not faster when my model already fits in RAM?** Because there is nothing to fix: stock llama.cpp then runs from the page cache with the same kernels. On Qwen3.6-35B at 24 GB, decode needs about 0.5 GB of expert weights per token (8 experts x 32 CPU layers x 1.9 MB), which the memory bus delivers at about 35 GB/s, close to this laptop's measured maximum (40 GB/s), so it is bandwidth-bound for everyone (stock 40, moe-cache 40 tok/s). The speed-ups appear when memory is short (the same model at 12 GB: 10 to 23 tok/s), for models larger than RAM, and for models with slow kernels (Q2_0 and the IQ types, where the executor and the faster kernel help).

**Does it use my GPU?** Yes, through llama.cpp as usual. `moe-cache-server` picks how many layers' experts live on the GPU from your free VRAM (that alone is worth about +60 % over keeping everything on the CPU). moe-cache itself manages the experts that stay on the CPU side.

**Is the output really the same as stock?** Yes by default: `tests/identity.sh` compares the tokens of stock llama-server and llama-server with the plugin (temperature 0, seed 42, same command line) and every model in [RESULTS](RESULTS.md) passed it (Hunyuan only a short test). It runs in CI on Linux and Windows. Only `--fast`, `--fast=kv`, `--fast=fewer`, `--fast=2` and `--hybrid` change the output, and only when you ask for them.

**Why not just let the OS page cache / mmap handle it?** That is what stock llama.cpp does, and it is the slow case in the table: when the model does not fit, missing experts come in through tens of thousands of small page faults per token, and the kernel evicts pages without knowing which experts matter. moe-cache reads the same bytes with a few hundred large parallel direct reads and keeps whole experts (gpt-oss-120b: about 79,000 vs 770 reads per token for the same 325 MB, RESULTS 12b). `--no-mmap` or `--mlock` need the whole model in RAM, so they do not help a model bigger than RAM.

**How does it compare with ktransformers, ik_llama.cpp or PowerInfer?** Not measured. Those are separate engines (forks or rewrites with their own kernels and model support); moe-cache is a plugin for the unmodified upstream llama-server and keeps its exact tokens. Against llama.cpp's own `--n-cpu-moe` / `-ot` offload: moe-cache works together with it (every row above uses the same `--n-cpu-moe` on both sides).

**Does it work with Ollama or LM Studio?** Not directly: both ship their own bundled copy of llama.cpp, and the plugin needs a `llama-server` built with shared ggml libraries from a tested commit. Their model files are plain GGUF, so you can run the same file with `moe-cache-server` (see "Ollama user?" above) and point any OpenAI-compatible client at `http://localhost:8080/v1`.

**On Windows llama-server uses a lot of RAM with GPU layers, should I add `--no-mmap`, `--mlock` (newer llama.cpp: `--load-mode none` / `mlock`) or `--moe-cache-mib`?** No. llama.cpp's Windows loader leaves the model-file pages of the GPU tensors in llama-server's working set (about as much RAM as the GPU layers; RESULTS 56 I37). It is not the GPU driver. The plugin releases those pages at the first graph (the log says "released N GiB of model-file pages from the working set"). `--moe-cache-mib` is llama.cpp's GPU cache, not this plugin's. Keep the page file system-managed: the cache costs no commit, but Vulkan's GPU memory does. Details: [docs/WINDOWS.md](WINDOWS.md), "Windows: where the RAM goes".

**Mac?** No. The plugin uses Linux (and Windows) memory and direct-I/O calls, and on Apple Silicon the GPU and CPU share one memory, which changes the problem.

**Does it wear out my SSD?** It only reads the model file; reads do not wear an SSD. It writes small files: the expert profile, and in `--agent` mode the saved conversations (`--no-warm` turns that off).

**Is there a GUI?** Yes: the control page (`bin/moe-cache-server --ui`, see [docs/GUI.md](GUI.md)), plus llama-server's own chat page (`--open`). Any OpenAI-compatible app such as [Open WebUI](https://github.com/open-webui/open-webui) can use `http://localhost:8080/v1` (we have not tested Open WebUI ourselves).

**Why not an own inference engine like Strata or Maya?** Short answer: they ship their own engine per model family; moe-cache is a small layer for the MoE models stock llama.cpp runs (tested list in COMPATIBILITY). The reasoning and our measurements: [docs/ENGINE_DECISION.md](ENGINE_DECISION.md). A side-by-side with what each project says about itself today follows.

### Strata and Maya today (as of 2026-10-29)

Read from their public release notes and README pages only; we did not clone, build or run either one, so every number in the Strata and Maya columns is **their own, not ours**. Both are MIT-licensed. Both change fast (Strata has had about one release a day), so check their pages.

| | Strata ([Niko1221/Strata](https://github.com/Niko1221/Strata), 0.2.7.1 of 2026-10-27) | Project Maya ([mw00/project-maya](https://github.com/mw00/project-maya)) | moe-cache |
|---|---|---|---|
| What it is | an app, a terminal tool and an engine per hardware family (CUDA, HIP, SYCL, Metal, DGX Spark), with its own kernels and parts of ggml | started from Strata's code, rewritten per model; engine, server and dashboard | a plugin for stock, upstream llama.cpp; the server is llama-server |
| Models | Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, and since 0.2.4 DeepSeek-V4.1-Flash (2.00 bpw and similar) | GLM-5.3-Flash, DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash (their own quantizations) | any MoE model llama.cpp runs; tested list in COMPATIBILITY |
| Bigger than RAM | yes for DeepSeek: the part that does not fit is read from SSD (about 265 GB of disk at 2.00 bpw). Their number: RTX 3060 12 GB + 46 GB RAM, 9-11 tok/s, 155 tok/s prompt at 4K. For the Qwen models we read in October, all experts were pinned in RAM; we have not re-checked | yes: GPU, RAM and NVMe tiers; 32 GB RAM minimum | yes: the RAM cache plus direct SSD reads is the point of the project |
| GPUs | NVIDIA, AMD (HIP), Intel Arc (SYCL), Apple (Metal) | NVIDIA compute capability 7.0+; AMD and Windows are marked experimental | whatever llama.cpp supports: our measurements are Vulkan (AMD) and CUDA (NVIDIA); the plugin itself does not depend on the GPU backend |
| Their reported speeds | RTX 5070 12 GB: Qwen3.8-Flash-Next Q2_0 94 tok/s decode, 2,650 tok/s prompt (64 GB RAM); R9700: Qwen3.6-35B UD-Q4_K_XL 117.8 tok/s (0.1.45) | one RTX 5090 + 128 GB RAM, GLM-5.3: 24-26 tok/s up to 512K context (a user's write-up); no result for 16 GB VRAM + 32 GB RAM is published | see the tables above; PC with a 16 GB card and 32 GB RAM, Flash-Next Q2_0 (66.4 GB file): 22.2 tok/s decode (model table above) |
| Exact output | their "Faster" mode (opt-in on some, default on cards of 16 GB or less for DeepSeek) uses 5 of 6 experts per token; KL not given | not stated | exact by default (output compared byte for byte with stock llama.cpp in CI); the approximate options (`--fast`, `--hybrid`) are opt-in and labelled |
| Long context | their 64K figure is for Strix Halo (749 tok/s prompt) and DGX Spark; AMD 131K / 230K / 500K prompts were read 8.7 / 16 / 79 % faster in 0.1.45 | up to 1M context on GLM-5.3 and MiniMax on big GPUs (H100, RTX PRO 6000); KV cache streams from RAM | 64k-256k with agents on 32 GB RAM: the long-context table above (Windows PC, RESULTS 64) |
| Windows and Linux | both | Linux; Windows experimental | both; Windows is newer and has fewer runs |

Where they are the better choice: you have a card and RAM they list, and you want their model at the highest speed; their installers are polished. Where moe-cache fits: you want to stay on stock llama.cpp, you use a MoE model or a quantization they do not ship, you want output identical to stock, or you have 32 GB of RAM and a model bigger than that.

### Strata and Maya today (as of 2026-10-29)

Read from their public release notes and README pages only; we did not clone, build or run either one, so every number in the Strata and Maya columns is **their own, not ours**. Both are MIT-licensed. Both change fast (Strata has had about one release a day), so check their pages.

| | Strata ([Niko1221/Strata](https://github.com/Niko1221/Strata), 0.2.7.1 of 2026-10-27) | Project Maya ([mw00/project-maya](https://github.com/mw00/project-maya)) | moe-cache |
|---|---|---|---|
| What it is | an app, a terminal tool and an engine per hardware family (CUDA, HIP, SYCL, Metal, DGX Spark), with its own kernels and parts of ggml | started from Strata's code, rewritten per model; engine, server and dashboard | a plugin for stock, upstream llama.cpp; the server is llama-server |
| Models | Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, and since 0.2.4 DeepSeek-V4.1-Flash (2.00 bpw and similar) | GLM-5.3-Flash, DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash (their own quantizations) | any MoE model llama.cpp runs; tested list in COMPATIBILITY |
| Bigger than RAM | yes for DeepSeek: the part that does not fit is read from SSD (about 265 GB of disk at 2.00 bpw). Their number: RTX 3060 12 GB + 46 GB RAM, 9-11 tok/s, 155 tok/s prompt at 4K. For the Qwen models we read in October, all experts were pinned in RAM; we have not re-checked | yes: GPU, RAM and NVMe tiers; 32 GB RAM minimum | yes: the RAM cache plus direct SSD reads is the point of the project |
| GPUs | NVIDIA, AMD (HIP), Intel Arc (SYCL), Apple (Metal) | NVIDIA compute capability 7.0+; AMD and Windows are marked experimental | whatever llama.cpp supports: our measurements are Vulkan (AMD) and CUDA (NVIDIA); the plugin itself does not depend on the GPU backend |
| Their reported speeds | RTX 5070 12 GB: Qwen3.8-Flash-Next Q2_0 94 tok/s decode, 2,650 tok/s prompt (64 GB RAM); R9700: Qwen3.6-35B UD-Q4_K_XL 117.8 tok/s (0.1.45) | one RTX 5090 + 128 GB RAM, GLM-5.3: 24-26 tok/s up to 512K context (a user's write-up); no result for 16 GB VRAM + 32 GB RAM is published | see the tables above; PC with a 16 GB card and 32 GB RAM, Flash-Next Q2_0 (66.4 GB file): 22.2 tok/s decode (model table above) |
| Exact output | their "Faster" mode (opt-in on some, default on cards of 16 GB or less for DeepSeek) uses 5 of 6 experts per token; KL not given | not stated | exact by default (output compared byte for byte with stock llama.cpp in CI); the approximate options (`--fast`, `--hybrid`) are opt-in and labelled |
| Long context | their 64K figure is for Strix Halo (749 tok/s prompt) and DGX Spark; AMD 131K / 230K / 500K prompts were read 8.7 / 16 / 79 % faster in 0.1.45 | up to 1M context on GLM-5.3 and MiniMax on big GPUs (H100, RTX PRO 6000); KV cache streams from RAM | 64k-256k with agents on 32 GB RAM: the long-context table above (Windows PC, RESULTS 64) |
| Windows and Linux | both | Linux; Windows experimental | both; Windows is newer and has fewer runs |

Where they are the better choice: you have a card and RAM they list, and you want their model at the highest speed; their installers are polished. Where moe-cache fits: you want to stay on stock llama.cpp, you use a MoE model or a quantization they do not ship, you want output identical to stock, or you have 32 GB of RAM and a model bigger than that.
