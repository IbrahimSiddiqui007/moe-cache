---
id: "quant-abba-p64k"
runs: ["2026-11-05_pc_d7229f"]
cells: [{"run": "2026-11-05_pc_d7229f", "arm": "Q2_0", "metric": "prompt_tps", "vs": "IQ3_XXS"}, {"run": "2026-11-05_pc_d7229f", "arm": "Q2_0", "metric": "decode_tps", "vs": "IQ3_XXS"}, {"run": "2026-11-05_pc_d7229f", "arm": "IQ3_XXS", "metric": "prompt_tps"}, {"run": "2026-11-05_pc_d7229f", "arm": "IQ3_XXS", "metric": "decode_tps"}]
valid_moe_cache: ["0.3.6", "0.3.6"]
valid_llama_builds: [12064, 12064]
withdrawn: ""
---
# On a 64k prompt (61,843 tokens) Flash-Next Q2_0 reads the prompt 86.9 % faster than IQ3_XXS (445.6 vs 238.5 tok/s) and the short answer after it decodes 50.7 % faster (24.30 vs 16.12 tok/s)

Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), PC, 2026-11-05, kit head5 with plugin v0.3.6 and the official llama.cpp b12064 (ggml 0.28.0, 51fe42c5d), launcher defaults, context 65,536, -ub 2048. Two quants as the two arms, order Q2_0, IQ3_XXS, IQ3_XXS, Q2_0 (A B B A), one 64k-token prompt per run, then a short answer; moe-cache on both arms. Q2_0 ran with 40 expert layers on the CPU and 8 on the GPU, IQ3_XXS with 42 and 6 (the launcher's plan per file). Means of two runs per arm: prompt Q2_0 445.55 (438.5, 452.6), IQ3_XXS 238.45 (237.9, 239.0) tok/s; decode after the prompt Q2_0 24.30 (23.71, 24.89), IQ3_XXS 16.12 (16.69, 15.55) tok/s. Cleanliness: the harness ran its own clean check before every run (windows, RAM over 0.5 GB, VRAM over 0.3 GiB, CPU busy over 10 s) and all four runs came out clean. The two arms use different files and different GPU layer counts, so this compares the files as the launcher runs them, not one setting.
