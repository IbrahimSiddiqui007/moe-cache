# Which model and which file

Moved here from the README on 2026-10-30. `python bin/moe_cache.py pick` applies these rules for your machine; `python bin/moe_cache.py models` lists the catalogue below for your PC (fits / tight / no, size, downloaded or not, measured speed; `--all` adds the ones that do not fit, `--json` for scripts).

## The catalogue

One file, [`models.json`](../models.json), holds every model moe-cache recommends: files with sizes and sha256, Hugging Face repo and revision, licence, uses, the fit thresholds and the measured numbers (each with machine, date and source). `pick`, `launch`, `models`, this table (`tools/gen_models_md.py`) and `tools/check_model_links.py` all read it. **fits** = it runs on your PC and we measured a machine like it; **tight** = it runs but is outside what we measured (smaller PC, longer context, untested use or drive); **no** = will not run (too little memory or VRAM, a hard disk, llama.cpp too old, wrong use). The fit rule is `fit()` in `bin/moe_cache_pick.py`, built from the launcher's memory cap (RAM minus max(4 GiB, 12.5 %)) and the planner's KV size.

<!-- models:begin (generated from models.json by tools/gen_models_md.py: edit the JSON, then run the tool; CI checks it) -->

| Id | Model | Quant | Size (files) | Good for | Needs | Measured | MTP head | License |
|---|---|---|---|---|---|---|---|---|
| `bolt` | Nemotron 3.5 Bolt 74B-A9B (NVIDIA 2-bit QAT) | NVIDIA 2-bit QAT (24GB file) | 22.0 GB (1) | agents, chat, documents | must fit in the server's RAM (stock llama.cpp); llama.cpp b11966+ | 27.9 tok/s decode at 64k ctx (Windows PC, RESULTS 65, 2026-10-28) | unknown | openmdw-1.1 |
| `flash-next-q2_0` | Qwen3.8-Flash-Next Q2_0 (GSQ-RCO) | Q2_0 | 66.4 GB (2) | chat, agents, documents | stock if it fits, else moe-cache, server memory 24 GB+, GPU 6 GiB+, SSD | 22.16 tok/s decode at 8k ctx (Windows PC, RESULTS 63, 2026-10-28) | none | apache-2.0 |
| `flash-next-iq3_xxs` | Qwen3.8-Flash-Next IQ3_XXS (GSQ-RCO) | IQ3_XXS | 75.8 GB (2) | chat, agents | stock if it fits, else moe-cache, server memory 24 GB+, GPU 6 GiB+, SSD | 19.86 tok/s decode at 8k ctx (Windows PC, RESULTS 59, 2026-10-27) | none | apache-2.0 |
| `qwen3.6-35b` | Qwen3.6-35B-A3B Q4_K_M | Q4_K_M (unsloth UD) | 22.1 GB (1) | agents, chat, documents | stock if it fits, else moe-cache | 59.09 tok/s decode at 64k ctx (Windows PC, RESULTS 64, 2026-10-28) | separate-file | apache-2.0 |
| `qwen3.5-122b` | Qwen3.5-122B-A10B UD-IQ2_XXS | UD-IQ2_XXS (unsloth) | 36.6 GB (1) | chat | stock if it fits, else moe-cache, server memory 24 GB+, GPU 6 GiB+, SSD | 19.69 tok/s decode at 64k ctx (Windows PC, RESULTS 64, 2026-10-28) | unknown | apache-2.0 |
| `glm-5.3` | GLM-5.3-Flash UD-IQ2_XXS | UD-IQ2_XXS (unsloth) | 101.8 GB (4) | chat | must fit in the server's RAM (stock llama.cpp) | 4.44 tok/s decode at 8k ctx (Windows PC, RESULTS 59, 2026-10-27) | in-file | mit |
| `gpt-oss-120b` | gpt-oss-120b MXFP4 (OpenAI, ggml-org) | MXFP4 | 63.4 GB (1) | chat | stock if it fits, else moe-cache, server memory 24 GB+, GPU 6 GiB+, SSD | 2.71 tok/s decode at 8k ctx (Linux laptop, RESULTS 12, 2026-10-09) | unknown | apache-2.0 |
| `deepseek-v41-flash` | DeepSeek-V4.1-Flash-1017 IQ1_M + Engram (GSQ-RCO) | IQ1_M + Engram IQ4_NL | 231.3 GB (6) | chat (not agents, documents) | stock if it fits, else moe-cache, server memory 24 GB+, GPU 12 GiB+, SSD, 240 GB free disk, llama.cpp b11870+ | not measured | unknown | mit |

Machines the numbers come from: **pc** = Windows PC: Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5, Windows 11, llama.cpp Vulkan; **laptop-24** = Linux laptop: RTX 4060 8 GB, 31 GB RAM limited to 24 GB; **laptop-12** = Linux laptop: RTX 4060 8 GB, memory limited to 12 GB. Every number, with its date and RESULTS section or claim, is in `models.json`.

- `bolt` ([nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF)): Fits in RAM + VRAM on a 32 GB PC, so the launcher runs stock llama-server (moe-cache forced on it was 18 % slower, RESULTS 65). General knowledge beyond our 202 tasks is not checked. **Warning:** Needs llama.cpp b11966 or newer (tokenizer fix #30960).
- `flash-next-q2_0` ([ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)): The faster Flash-Next file: about 10 % more tokens per second than IQ3_XXS, 9 GB smaller. No quality difference between any two Flash-Next files is proven (all p > 0.05). Decode is CPU-bound on the measured PC; the model card's +33 % does not reproduce. **Warning:** Bigger than the RAM the server may use: moe-cache keeps the hot experts in RAM and reads the rest from the drive (SSD needed).
- `flash-next-iq3_xxs` ([ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)): The default Flash-Next file on a 16 GB GPU (owner 2026-10-30): 3-bit, the most tasks right (193/202), reads best for coding; about 10 % slower decode than Q2_0. **Warning:** Bigger than the RAM the server may use: moe-cache keeps the hot experts in RAM and reads the rest from the drive (SSD needed).
- `qwen3.6-35b` ([unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)): The smallest candidate: the one that still runs agents at 64k on a small machine. MTP (separate head file) takes decode from 43.5 to 72 tok/s in agent mode on the PC (RESULTS 60). The Hugging Face file is now 22.13 GB; the picker and the measurements used a 21.72 GB file (RESULTS 44), so the file changed since (revision above): check the sha256 before comparing numbers.
- `qwen3.5-122b` ([unsloth/Qwen3.5-122B-A10B-GGUF](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF)): 36.6 GB on disk instead of Flash-Next's 66 GB. **Warning:** Bigger than the RAM the server may use (SSD needed).
- `glm-5.3` ([unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)): The biggest candidate; suggested for chat only where it fits in RAM (128 GB+). With --mtp 3.1x stock on the PC (RESULTS 60); on the laptop MTP lost 25 % (RESULTS_OLD_LAPTOP 40). **Warning:** On a 32 GB PC it is a 102 GB model read from the drive: 4-7 tok/s, 64k contexts did not finish.
- `gpt-oss-120b` ([ggml-org/gpt-oss-120b-GGUF](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF)): General chat and short answers. A reasoning model: if answers come back empty, give it room or lower its effort (--reasoning-budget); sampling temperature 1.0, top-p 1.0. **Warning:** Coding agents at 64k and long prompts are not tested (check that the agent handles reasoning_content).
- `deepseek-v41-flash` ([ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)): Six files: the backbone in five shards (120.7 GB) and the Engram lookup table (110.6 GB), read from the SSD. **Warning:** Chat and short answers only: long prompts are slow (a llama.cpp user measured about 31 tok/s prompt) and saved prompt state is unreliable until llama.cpp #31292. **Warning:** Needs 240 GB free disk and a fast SSD; never run by us (no measured numbers).

<!-- models:end -->

## Flash-Next files and the rules by machine


**Which Flash-Next file?** Default: **IQ3_XXS** (3-bit: reads best, most tasks right, popular for coding); **Q2_0** when you want speed (about 10 % more tokens per second, e.g. for long thinking). Measured with moe-cache on a 16 GB GPU + 32 GB RAM PC (RESULTS 59, 60, 63; Qwen3.8-Flash-Next GSQ-RCO files; tasks = our
202 exact-answer tasks; **no quality difference between any two files is proven**, all paired sign tests p > 0.05):

| File | Size | Decode tok/s | Prompt tok/s (3k / 34k tokens) | Tasks /202 | Use it |
|---|---|---|---|---|---|
| **Q2_0** | 66.4 GB | **22.2**; **24.7-25.7** with the async reads (N15, +12 %, RESULTS 67) | **281 / 287** (October kits; with the v0.3.5 defaults a 34k prompt reads at 620-640 tok/s, RESULTS 78); long prompts **+8-11 %** (N23 pipelined block) and **+2-5 %** at 34k (N22), RESULTS 69; decode **+0.3-2.6 %** with the fused gate+up job (E3, RESULTS 73, every pair, not clean) | 191 | **the faster file**: about 10 % more tokens per second than IQ3_XXS (e.g. for long thinking), 9 GB smaller; vs IQ2_XS +5 % decode, prompts +11 % / +19 % |
| **IQ3_XXS** | 75.8 GB | 19.9 | - | **193** | **recommended default** (since 0.3.2): 3-bit, the most tasks right, reads best for coding; about 10 % slower decode than Q2_0 |
| ISTA GSQ-RCO Coder IQ1_M | 58.4 GB | 21.2 (agent session) | - | 189 | coding agents: half the experts (256 of 512); OpenCode session 55.5 s vs 73.4 s on IQ2_XS. General knowledge not checked beyond the task set |
| IQ2_XS | 68.0 GB | 21.0 (same day as Q2_0; 21.9 in RESULTS 59) | 253 / 242 | 187 | superseded by Q2_0 (older rows above that say IQ2_XS were measured on IQ2_XS) |
| IQ3_S | 83.6 GB | 17.3 | - | 192 | not worth it: slower and bigger than IQ3_XXS for no gain |

**Which model for my machine?** Run `python3 bin/moe_cache.py pick` (Windows: `python bin\moe_cache.py pick`) or open the control page's Settings > **Which model?**.
It reads your RAM (and the hard cap the server keeps: RAM minus max(4 GiB, 12.5 %)), dedicated VRAM, free disk, SSD or HDD, CPU (AVX2 / AVX-512) and your
llama-server's build, asks what it is for (agents / coding at 64k-256k, plain chat, long documents), and recommends one model, file, context and options.
It never downloads by itself: it prints the command with size and source, and `--apply` (or the page's *Use it*) makes `moe-cache-server`,
`on-demand` and the page start that model by default, so `connect hermes|opencode|claude` points the agent at it. `--gpu-margin GiB` (default 0.5) is taken off the VRAM in its memory estimate, as the launcher does; if the GPU also draws your screen, start with `--gpu-margin 2`. Models Ollama has already downloaded (its manifests in `~/.ollama/models`, `%USERPROFILE%\\.ollama\\models` or `$OLLAMA_MODELS`) count as local: llama-server reads the blobs directly. If the disk is too small for the model that suits your machine, the picker says which model that is and how much disk is missing, offers no download, and prefers a model that is already here. Where we have no number for your class of
machine it says "untested on this class of machine". The rules, from the measurements (RESULTS 44-45, 53, 59-67; the PC is 32 GB RAM + 16 GB GPU):

| Your machine | Agents / coding (64k-256k) | Plain chat | Long documents |
|---|---|---|---|
| **32 GB RAM + 16 GB GPU** (measured) | **Nemotron 3.5 Bolt 74B** at 128k, stock: decode 27.9 (24.9 at 114k), OpenCode session 47.8 s | **Flash-Next IQ3_XXS**: 19.9 tok/s, 193/202 tasks, reads best for coding (Q2_0: 24-25 tok/s, about 10 % faster, 191/202) | **Bolt**: prompts 595 tok/s at 57k, 721 at 114k (Flash-Next Q2_0 170-220 at 128k) |
| 32 GB + 8 GB GPU | Bolt at 64k (untested with 8 GB) | Flash-Next Q2_0 (untested; the IQ2_XS file of the same size gave 12.5 tok/s on 24 GB + 8 GB) | Bolt at 64k (untested) |
| 64 GB + 16 GB GPU | Bolt | Flash-Next IQ3_XXS (untested here) | Bolt |
| 128 GB+ | Bolt | GLM-5.3-Flash (fits; untested) | Bolt |
| 16 GB + 6 GB GPU, or no GPU | Qwen3.6-35B-A3B (22.6 tok/s with moe-cache at 12 GB RAM + 8 GB GPU; 9.9-10.6 tok/s without a GPU; agents on such a machine untested at 64k) | Qwen3.6-35B-A3B | Qwen3.6-35B-A3B |
Bolt needs llama.cpp **b11966 or newer** (the picker checks `llama-server --version` and says so); models bigger than RAM are not suggested when they sit
on a hard disk, and a model is skipped when the disk has less free space than its size. Bolt's general knowledge beyond our 202 tasks is not checked.
