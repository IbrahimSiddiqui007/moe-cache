# Which model and which file

Moved here from the README on 2026-10-30, unchanged. `python bin/moe_cache.py pick` applies these rules for your machine.

**Which Flash-Next file?** Default: **IQ3_XXS** (3-bit: reads best, most tasks right, popular for coding); **Q2_0** when you want speed (about 10 % more tokens per second, e.g. for long thinking). Measured with moe-cache on a 16 GB GPU + 32 GB RAM PC (RESULTS 59, 60, 63; Qwen3.8-Flash-Next GSQ-RCO files; tasks = our
202 exact-answer tasks; **no quality difference between any two files is proven**, all paired sign tests p > 0.05):

| File | Size | Decode tok/s | Prompt tok/s (3k / 34k tokens) | Tasks /202 | Use it |
|---|---|---|---|---|---|
| **Q2_0** | 66.4 GB | **22.2**; **24.7-25.7** with the async reads (N15, +12 %, RESULTS 67) | **281 / 287**; long prompts **+8-11 %** (N23 pipelined block) and **+2-5 %** at 34k (N22), RESULTS 69; decode **+0.3-2.6 %** with the fused gate+up job (E3, RESULTS 73, every pair, not clean) | 191 | **the faster file**: about 10 % more tokens per second than IQ3_XXS (e.g. for long thinking), 9 GB smaller; vs IQ2_XS +5 % decode, prompts +11 % / +19 % |
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
