# FAQ: running Mixture-of-Experts models bigger than your RAM with llama.cpp

Short answers to the questions people ask most. Setup is in [INSTALL](install.md), models in [MODELS](../reference/models.md), this release's results in [RESULTS_CURRENT](../results/results-current.md) (speed figures follow in a later release; older per-depth records in [results-archive](../results/archive/results-current-v0.3.8.md)).

### How do I run a MoE model that is bigger than my RAM?

Use moe-cache with llama.cpp: the plugin keeps the experts the model actually uses in RAM and reads the rest from your SSD, instead of letting the operating system re-read the whole file. Install, then `moe-cache launch` ([README](../../README.md#get-started)); `moe_cache.py pick` names the model that suits your RAM and graphics card.

### Why is llama.cpp slow when the model does not fit in RAM?

MoE models such as Qwen3, gpt-oss or DeepSeek only use a few "experts" per token, but stock llama.cpp keeps all expert weights in memory-mapped files.
When they do not fit in RAM, the OS page cache thrashes and speed falls off a cliff.

### When to use something else

| Your situation | Best choice |
|---|---|
| The model **fits in your RAM** | **Stock llama.cpp**, ideally with its GPU expert cache (`--moe-cache-mib N`). That flag is llama.cpp's own GPU expert cache, not moe-cache's: on our Windows Vulkan PC it was slower than without it (RESULTS 59), and its output can differ slightly from stock (not measured by us). On Qwen3.6-35B (21.7 GB) at a 24 GB limit, both with the same 5000 MiB GPU cache: stock and moe-cache were compared in RESULTS 29. moe-cache adds nothing here but its CPU pinning, which stock gets with `-C MASK --cpu-strict 1`. When a model fits in RAM, stock llama.cpp is usually the better choice, and the launcher runs it for a dense model, when every expert fits on the graphics card, and since v0.3.9 when the model only nearly fits the GPU (the plugin plan would keep more than 1.29 times stock's expert layers on the CPU; `--always-plugin` keeps the plugin). A MoE model that fits in RAM but keeps some expert layers on the CPU still runs through the plugin by default (since v0.3.8; `--stock` or `MOE_CACHE_STOCK_WHEN_FITS=1` opts out), also with no GPU (the planner then puts every expert layer on the CPU). Evidence, labelled laptop, no figures quoted: on one laptop with an 8 GB card the plugin was not faster than stock for a model that fits in RAM (release code, paired 40k-token chat; an earlier figure was not reproduced); no stock comparison on the 16 GB PC. |
| The model is **bigger than your RAM**, or it is **another MoE architecture**, or you want to stay on **stock llama.cpp with the same tokens** | **moe-cache**: **built for models bigger than your RAM** on Windows and on Linux (measured on Windows since 2026-10-27, RESULTS 59, and on Linux since 2026-10-15; speed numbers follow in a later release; measurements in [RESULTS](../results/archive/results-windows-desktop.md) and [RESULTS_OLD_LAPTOP](../results/archive/results-old-laptop.md)). No gain when the model fits in RAM. |


### Do I need this if my model fits in RAM?

No. Use stock llama.cpp with `--moe-cache-mib` (see "When to use something else" at the top of this page). Caveat: `--moe-cache-mib` is llama.cpp's own GPU expert cache; on our Vulkan PC it was slower (RESULTS 59) and its output can differ slightly from stock, so test it on your card. The gain is when memory is short or the model is larger than your RAM. When a model fits in RAM, stock llama.cpp is usually the better choice, and the launcher runs it for a dense model, when every expert fits on the graphics card, and since v0.3.9 when the model only nearly fits the GPU (the plugin plan would keep more than 1.29 times stock's expert layers on the CPU; `--always-plugin` keeps the plugin). A MoE model that fits in RAM but keeps some expert layers on the CPU still runs through the plugin by default (since v0.3.8; `--stock` or `MOE_CACHE_STOCK_WHEN_FITS=1` opts out), also with no GPU (the planner then puts every expert layer on the CPU). Evidence, labelled laptop, no figures quoted: on one laptop with an 8 GB card the plugin was not faster than stock for a model that fits in RAM (release code, paired 40k-token chat; an earlier figure was not reproduced); no stock comparison on the 16 GB PC.

### Do you check answer quality?

We measure it; we do not promise equal meaning on arbitrary prompts. How, in one page: [docs/reference/exactness.md](../reference/exactness.md). The plugin itself changes no number: with `--exact` (or below 32k context, or when the card has no room for the long-prompt plan) and the same layer split the tokens are identical to stock llama.cpp. By default, from 32k context on a GPU with room, the long-prompt mode (not applied to a model that fits in RAM) stores the KV cache as q8_0 (so do 128k and above), which means answers are **not byte-identical** to stock (KL 0.014-0.015, same top token 94-95 %, meaning kept in 10 of 10 checks at 55k); `--exact` keeps f16 (`tests/identity.sh`, CI on Linux and Windows). The approximate modes (`--fast`, `--fast=kv`, `--fast=fewer`, `--hybrid`) must pass a [quality gate](../reference/quality-gate.md) whose limits were fixed before measuring: KL divergence and same-top-token rate against stock, the 202 exact-answer tasks, and a loop count. Limits: `kv` KL at most 0.015, top-1 at least 95 %, tasks within 2 of stock; `fast` 0.040 / 90 % / 4; `fewer` 0.060 / 88 % / 5. Measured on Flash-Next Q2_0: `kv` KL 0.0079, top-1 96.7 %, 190 tasks = stock; `fewer` KL 0.034, 192 vs 190, unproven at 64k-256k. If answers look wrong or loop on your machine, see [docs/guides/troubleshooting.md](troubleshooting.md).

### What cache policy does it use?

LRU at expert granularity; the experts of the operation that is running are protected. We replayed 13,464 real decode layer-steps of gpt-oss-120b at the real cache size: LRU 84.2 % hits, 2Q 83.6 %, LFU 84.1-84.2 %, LRU with the hottest experts pinned from a profile 84.3-84.8 %, the offline optimum 92.5 %. No policy beat LRU by more than 0.6 points, so LRU stayed ("Smarter eviction").

### Is a very long prompt still slow?

Less than before, but a long prompt is still a wait. Prompt reading runs on the GPU. On our Windows PC (Flash-Next Q2_0) a 64k-token prompt was measured on v0.3.4 and v0.3.5 (agent mode, 2026-10-28) (§78). A 222k-token prompt at 256k context is a longer wait (launcher default `-ub 2048`, 2026-11-03). Speed figures follow in a later release.

### Is there a gain when my model already fits in RAM?

Because there is nothing to fix: stock llama.cpp then runs from the page cache with the same kernels. On Qwen3.6-35B at 24 GB, decode needs about 0.5 GB of expert weights per token (8 experts x 32 CPU layers x 1.9 MB), which the memory bus delivers at about 35 GB/s, close to the measured maximum of our test laptop (40 GB/s; one machine, one run, a forecast), so it is bandwidth-bound for everyone. moe-cache is aimed at memory that is short (the same model at 12 GB), models larger than RAM, and models with slow kernels (Q2_0 and the IQ types); speed numbers follow in a later release.

### Does it use my GPU?

Yes, through llama.cpp as usual. `moe-cache-server` picks how many layers' experts live on the GPU from your free VRAM (that is the plan, instead of keeping everything on the CPU). moe-cache itself manages the experts that stay on the CPU side.

### Is the output really the same as stock?

With `--exact`, yes (and below 32k context, and without a GPU plan with room); by default from 32k context the long-prompt mode makes it close but not byte-identical (q8_0 KV cache, see EXACTNESS.md; `--no-long-prompt` turns it off). The plugin's own check: `tests/identity.sh` compares the tokens of stock llama-server and llama-server with the plugin (temperature 0, seed 42, same command line) and every model in [RESULTS](../results/archive/results-windows-desktop.md) and [RESULTS_OLD_LAPTOP](../results/archive/results-old-laptop.md) passed it (Hunyuan only a short test). It runs in CI on Linux and Windows. `--fast`, `--fast=kv`, `--fast=fewer`, `--fast=2` and `--hybrid` can change the output; `--fast=kv` is also chosen automatically at `--ctx 131072` and above when a GPU is detected (conditions: [CONFIGURATION.md](../reference/configuration.md)), and MTP can change text too ([EXACTNESS.md](../reference/exactness.md)).

### Why not just let the OS page cache / mmap handle it?

That is what stock llama.cpp does, and it is the slow case in our measurements ([RESULTS](../results/archive/results-windows-desktop.md) and [RESULTS_OLD_LAPTOP](../results/archive/results-old-laptop.md)): when the model does not fit, missing experts come in through tens of thousands of small page faults per token, and the kernel evicts pages without knowing which experts matter. moe-cache keeps whole experts resident and loads them with a few hundred large parallel reads (direct reads, or the file-view cache that is the default on Windows and Linux today; the 79,000 vs 770 figure below was measured with direct reads in October) (gpt-oss-120b: about 79,000 vs 770 reads per token for the same 325 MB, RESULTS_OLD_LAPTOP 12b). `--no-mmap` or `--mlock` need the whole model in RAM, so they do not help a model bigger than RAM.

### Does it work with llama.cpp's own offload options?

Yes: moe-cache works together with `--n-cpu-moe` / `-ot` (every comparison row in the results uses the same `--n-cpu-moe` on both sides). It is a plugin for the unmodified upstream `llama-server` and keeps its exact tokens.


### Windows: llama-server uses a lot of RAM with GPU layers, should I add `--no-mmap`, `--mlock` or `--moe-cache-mib`?

(In newer llama.cpp `--no-mmap` and `--mlock` are `--load-mode none` and `mlock`.) No. llama.cpp's Windows loader leaves the model-file pages of the GPU tensors in llama-server's working set (about as much RAM as the GPU layers; RESULTS 56 I37). It is not the GPU driver. The plugin releases those pages at the first graph (the log says "released N GiB of model-file pages from the working set"). `--moe-cache-mib` is llama.cpp's GPU cache, not this plugin's. Keep the page file system-managed: the cache costs no commit, but Vulkan's GPU memory does. Details: [docs/guides/windows.md](windows.md), "Windows: where the RAM goes".

### Does it run on a Mac?

No. The plugin uses Linux (and Windows) memory and direct-I/O calls, and on Apple Silicon the GPU and CPU share one memory, which changes the problem.

### Does it wear out my SSD?

It only reads the model file; reads do not wear an SSD. It writes small files: the expert profile, and in `--agent` mode the saved conversations (`--no-warm` turns that off).

### Is there a GUI?

Yes: the control page (`bin/moe-cache-server --ui`, see [docs/guides/gui.md](gui.md)), plus llama-server's own chat page (`--open`). Any OpenAI-compatible app such as [Open WebUI](https://github.com/open-webui/open-webui) can use `http://localhost:8080/v1` (we have not tested Open WebUI ourselves).

### Why not an own inference engine?

moe-cache is a small layer for the MoE models stock llama.cpp runs (tested list in platforms.md), so a new model works as soon as llama.cpp runs it. The reasoning and our measurements are in the maintainers' private note ENGINE_DECISION.
