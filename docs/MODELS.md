# Which model and which file: Qwen, GLM, gpt-oss and more on 32 GB RAM and a 16 GB graphics card

`python bin/moe_cache.py pick` applies these rules for your machine; `python bin/moe_cache.py models` lists the catalogue below for your PC (fits / tight / no, size, downloaded or not; `--all` adds the ones that do not fit, `--json` for scripts).

## The catalogue

One file, [`models.json`](../models.json), holds every model moe-cache recommends: files with sizes and sha256, Hugging Face repo and revision, licence, uses, the fit thresholds and the measured numbers (each with machine, date and source). `pick`, `launch`, `models`, this table (`tools/gen_models_md.py`) and `tools/check_model_links.py` all read it. **fits** = it runs on your PC and we measured a machine like it; **tight** = it runs but is outside what we measured (smaller PC, longer context, untested use or drive); **no** = will not run (too little memory or VRAM, a hard disk, llama.cpp too old, wrong use). The fit rule is `fit()` in `bin/moe_cache_pick.py`, built from the launcher's memory cap (RAM minus max(4 GiB, 12.5 %)) and the planner's KV size.

<!-- models:begin (generated from models.json by tools/gen_models_md.py: edit the JSON, then run the tool; CI checks it) -->

| Id | Model | Quant | Size (files) | Good for | Needs | Measured | MTP head | License |
|---|---|---|---|---|---|---|---|---|
| `bolt` | Nemotron 3.5 Bolt 74B-A9B (NVIDIA 2-bit QAT) | NVIDIA 2-bit QAT (24GB file) | 22.0 GB (1) | agents, chat, documents | must fit in the server's RAM (stock llama.cpp); llama.cpp b11966+ | speed figures follow in v0.3.9.1 | unknown | openmdw-1.1 |
| `flash-next-q2_0` | Qwen3.8-Flash-Next Q2_0 (GSQ-RCO) | Q2_0 | 66.4 GB (2) | chat, agents, documents | stock if it fits, else moe-cache, server memory 24 GB+, GPU 6 GiB+, SSD | speed figures follow in v0.3.9.1 | none | apache-2.0 |
| `flash-next-iq3_xxs` | Qwen3.8-Flash-Next IQ3_XXS (GSQ-RCO) | IQ3_XXS | 75.8 GB (2) | chat, agents | stock if it fits, else moe-cache, server memory 24 GB+, GPU 6 GiB+, SSD | speed figures follow in v0.3.9.1 | none | apache-2.0 |
| `qwen3.6-35b` | Qwen3.6-35B-A3B Q4_K_M | Q4_K_M (unsloth UD) | 22.1 GB (1) | agents, chat, documents | stock if it fits, else moe-cache | speed figures follow in v0.3.9.1 | separate-file | apache-2.0 |
| `qwen3.5-122b` | Qwen3.5-122B-A10B UD-IQ2_XXS | UD-IQ2_XXS (unsloth) | 36.6 GB (1) | chat | stock if it fits, else moe-cache, server memory 24 GB+, GPU 6 GiB+, SSD | speed figures follow in v0.3.9.1 | unknown | apache-2.0 |
| `glm-5.3` | GLM-5.3-Flash UD-IQ2_XXS | UD-IQ2_XXS (unsloth) | 101.8 GB (4) | chat | must fit in the server's RAM (stock llama.cpp) | speed figures follow in v0.3.9.1 | in-file | mit |
| `gpt-oss-120b` | gpt-oss-120b MXFP4 (OpenAI, ggml-org) | MXFP4 | 63.4 GB (1) | chat | stock if it fits, else moe-cache, server memory 24 GB+, GPU 6 GiB+, SSD | speed figures follow in v0.3.9.1 | unknown | apache-2.0 |
| `deepseek-v41-flash` | DeepSeek-V4.1-Flash-1017 IQ1_M + Engram (GSQ-RCO) | IQ1_M + Engram IQ4_NL | 231.3 GB (6) | chat (not agents, documents) | stock if it fits, else moe-cache, server memory 24 GB+, GPU 12 GiB+, SSD, 240 GB free disk, llama.cpp b11870+ | speed figures follow in v0.3.9.1 | unknown | mit |

Machines the numbers come from: **pc** = Windows PC: Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5, Windows 11, llama.cpp Vulkan; **laptop-24** = Linux laptop: RTX 4060 8 GB, 31 GB RAM limited to 24 GB; **laptop-12** = Linux laptop: RTX 4060 8 GB, memory limited to 12 GB. Every number, with its date and RESULTS section or claim, is in `models.json`.

- `bolt` ([nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF)): Fits in RAM + VRAM on a 32 GB PC, so the launcher runs stock llama-server (the plugin is for models bigger than memory; speed figures follow in v0.3.9.1). General knowledge beyond our 202 tasks is not checked. **Warning:** Needs llama.cpp b11966 or newer (tokenizer fix #30960).
- `flash-next-q2_0` ([ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)): The smaller Flash-Next file, 9 GB less than IQ3_XXS (the default in v0.3.7; IQ3_XXS is the default again since v0.3.8, decided 2026-11-07); speed figures follow in v0.3.9.1. No quality difference between any two Flash-Next files is proven (all p > 0.05). Decode is CPU-bound on the measured PC; the model card's decode gain does not reproduce. **Warning:** Bigger than the RAM the server may use: moe-cache keeps the hot experts in RAM and reads the rest from the drive (SSD needed).
- `flash-next-iq3_xxs` ([ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)): The default Flash-Next file (the default from 0.3.2 to 0.3.6 and again since v0.3.8, decided 2026-11-07): 3-bit, the most tasks right (193/202), reads best for coding, top-1 agreement with IQ3_S 76.5 % against 73.3 % for Q2_0 (measured 2026-11-05, record pending); speed figures follow in v0.3.9.1. **Warning:** Bigger than the RAM the server may use: moe-cache keeps the hot experts in RAM and reads the rest from the drive (SSD needed).
- `qwen3.6-35b` ([unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)): The smallest candidate: the one that still runs agents at 64k on a small machine. MTP (separate head file) is available for agent mode; speed figures follow in v0.3.9.1. The Hugging Face file is now 22.13 GB; the picker and the measurements used a 21.72 GB file (RESULTS 44), so the file changed since (revision above): check the sha256 before comparing numbers.
- `qwen3.5-122b` ([unsloth/Qwen3.5-122B-A10B-GGUF](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF)): 36.6 GB on disk instead of Flash-Next's 66 GB. **Warning:** Bigger than the RAM the server may use (SSD needed).
- `glm-5.3` ([unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)): The biggest candidate; suggested for chat only where it fits in RAM (128 GB+). MTP is available with --mtp; speed figures follow in v0.3.9.1. **Warning:** On a 32 GB PC it is a 102 GB model read from the drive: very slow, 64k contexts did not finish.
- `gpt-oss-120b` ([ggml-org/gpt-oss-120b-GGUF](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF)): General chat and short answers. A reasoning model: if answers come back empty, give it room or lower its effort (--reasoning-budget); sampling temperature 1.0, top-p 1.0. **Warning:** Coding agents at 64k and long prompts are not tested (check that the agent handles reasoning_content).
- `deepseek-v41-flash` ([ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)): Six files: the backbone in five shards (120.7 GB) and the Engram lookup table (110.6 GB), read from the SSD. **Warning:** Chat and short answers only: long prompts are slow (a llama.cpp user reported very slow prompt reading) and saved prompt state is unreliable until llama.cpp #31292. **Warning:** Needs 240 GB free disk and a fast SSD; never run by us (no measured numbers).

<!-- models:end -->

## Agents: Nemotron Bolt or Flash-Next Q2_0, and thinking on or off

Both run on the 32 GB RAM + 16 GB GPU test PC. Bolt (22 GB) fits in RAM and runs through stock llama.cpp; Flash-Next Q2_0 (66 GB) is bigger than the RAM and runs through moe-cache.

| Question | What we measured | How far to trust it |
|---|---|---|
| Short agent requests (202 tasks, about 4 tokens each) | Flash-Next Q2_0 vs Bolt per request: speed figures follow in v0.3.9.1 (Bolt's total included one long stall of the PC's graphics driver) | one run each, llama.cpp b11966, other programs were running on the PC, no benchmark record |
| Why Bolt's prompts needed a threshold change | llama.cpp sends any batch of 32 or more tokens to the GPU, which copies every expert that lives on the CPU over PCIe for each prompt. The launcher sets the threshold to 512 for stock runs that keep experts on the CPU, speed figures follow in v0.3.9.1; answers identical in 60 of 60 prompts | one run, same conditions |
| Thinking on or off for a tiny chat turn | Bolt's own template thinks by default, which adds a reasoning phase to a tiny chat turn; with `-- --reasoning-budget 0` it answers without reasoning (no reasoning at all; speed figures follow in v0.3.9.1) | one run each |
| Longer sessions | speed figures follow in v0.3.9.1 | different days and kits, so no ratio is claimed |

**What to use:** for coding agents, Flash-Next IQ3_XXS (the default) or Q2_0 are the files we measured with agents; Bolt is the choice when you need a 128k context or the model must fit in RAM. For quick turns with Bolt, `-- --reasoning-budget 0` removes the thinking. We do not turn thinking off by default: NVIDIA's model card recommends thinking on for agent work, and an agent that needs to plan may do worse without it (not measured by us).

## Flash-Next files and the rules by machine


**Which Flash-Next file?** Default again since v0.3.8 (decided 2026-11-07, reason: most people use it, and the quality loss of Q2_0 is small but the name puts people off): **IQ3_XXS** (3-bit, reads best, most tasks right; top-1 agreement with IQ3_S 76.5 % against 73.3 % for Q2_0, measured 2026-11-05, record pending). **Q2_0** is the smaller option (9 GB smaller; speed figures follow in v0.3.9.1). The default was IQ3_XXS from 0.3.2 to 0.3.6 and Q2_0 in v0.3.7. Older rows below: measured with moe-cache on a 16 GB GPU + 32 GB RAM PC (RESULTS 59, 60, 63; Qwen3.8-Flash-Next GSQ-RCO files; tasks = our
202 exact-answer tasks; **no quality difference between any two files is proven**, all paired sign tests p > 0.05):

| File | Size | Decode tok/s | Prompt tok/s (3k / 34k tokens) | Tasks /202 | Use it |
|---|---|---|---|---|---|
| **Q2_0** | 66.4 GB | speed figures follow in v0.3.9.1 | speed figures follow in v0.3.9.1 | 191 | **the smaller option** (the default in v0.3.7; IQ3_XXS is the default again since v0.3.8; 9 GB smaller); speed figures follow in v0.3.9.1 |
| **IQ3_XXS** | 75.8 GB | speed figures follow in v0.3.9.1 | - | **193** | **the default** (also from 0.3.2 to 0.3.6; again since v0.3.8): 3-bit, the most tasks right, reads best for coding; top-1 agreement with IQ3_S 76.5 % against 73.3 % for Q2_0 (measured 2026-11-05, record pending); speed figures follow in v0.3.9.1 |
| ISTA GSQ-RCO Coder IQ1_M | 58.4 GB | speed figures follow in v0.3.9.1 | - | 189 | coding agents: half the experts (256 of 512); speed figures follow in v0.3.9.1. General knowledge not checked beyond the task set |
| IQ2_XS | 68.0 GB | speed figures follow in v0.3.9.1 | speed figures follow in v0.3.9.1 | 187 | superseded by Q2_0 (older rows above that say IQ2_XS were measured on IQ2_XS) |
| IQ3_S | 83.6 GB | speed figures follow in v0.3.9.1 | - | 192 | not worth it: bigger than IQ3_XXS for no gain in tasks |

**Which model for my machine?** Run `python3 bin/moe_cache.py pick` (Windows: `python bin\moe_cache.py pick`) or open the control page's Settings > **Which model?**.
It reads your RAM (and the hard cap the server keeps: RAM minus max(4 GiB, 12.5 %)), dedicated VRAM, free disk, SSD or HDD, CPU (AVX2 / AVX-512) and your
llama-server's build, asks what it is for (agents / coding at 64k-256k, plain chat, long documents), and recommends one model, file, context and options.
It never downloads by itself: it prints the command with size and source, and `--apply` (or the page's *Use it*) makes `moe-cache-server`,
`on-demand` and the page start that model by default, so `connect hermes|opencode|claude` points the agent at it. `--gpu-margin GiB` (default 0.5) is taken off the VRAM in its memory estimate, as the launcher does; if the GPU also draws your screen, start with `--gpu-margin 2`. Models Ollama has already downloaded (its manifests in `~/.ollama/models`, `%USERPROFILE%\\.ollama\\models` or `$OLLAMA_MODELS`) count as local: llama-server reads the blobs directly. If the disk is too small for the model that suits your machine, the picker says which model that is and how much disk is missing, offers no download, and prefers a model that is already here. Where we have no number for your class of
machine it says "untested on this class of machine". The rules, from the measurements (the PC is 32 GB RAM + 16 GB GPU):

| Your machine | Agents / coding (64k-256k) | Plain chat | Long documents |
|---|---|---|---|
| **32 GB RAM + 16 GB GPU** (measured) | **Nemotron 3.5 Bolt 74B** at 128k, stock; speed figures follow in v0.3.9.1 | **Flash-Next IQ3_XXS**: 193/202 tasks, reads best for coding (Q2_0: smaller, 191/202; speed figures follow in v0.3.9.1) | **Bolt**: fits in memory, so stock llama.cpp (speed figures follow in v0.3.9.1) |
| 32 GB + 8 GB GPU | Bolt at 64k (untested with 8 GB) | Flash-Next Q2_0 (untested; the IQ2_XS file of the same size ran on 24 GB + 8 GB) | Bolt at 64k (untested) |
| 64 GB + 16 GB GPU | Bolt | Flash-Next IQ3_XXS (untested here) | Bolt |
| 128 GB+ | Bolt | GLM-5.3-Flash (fits; untested) | Bolt |
| 16 GB + 6 GB GPU, or no GPU | Qwen3.6-35B-A3B (runs with moe-cache at 12 GB RAM + 8 GB GPU and without a GPU; agents on such a machine untested at 64k) | Qwen3.6-35B-A3B | Qwen3.6-35B-A3B |
Bolt needs llama.cpp **b11966 or newer** (the picker checks `llama-server --version` and says so); models bigger than RAM are not suggested when they sit
on a hard disk, and a model is skipped when the disk has less free space than its size. Bolt's general knowledge beyond our 202 tasks is not checked.
