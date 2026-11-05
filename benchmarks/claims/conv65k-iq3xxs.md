---
id: "conv65k-iq3xxs"
runs: ["2026-11-05_pc_952d95"]
cells: [{"run": "2026-11-05_pc_952d95", "arm": "stock", "metric": "prompt_tps"}, {"run": "2026-11-05_pc_952d95", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-11-05_pc_952d95", "arm": "moe-cache", "metric": "prompt_tps", "vs": "stock"}, {"run": "2026-11-05_pc_952d95", "arm": "moe-cache", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["0.3.6", "0.3.6"]
valid_llama_builds: [12064, 12064]
withdrawn: ""
---
# Flash-Next IQ3_XXS in a 65k conversation (49 turns to 65,371 tokens): moe-cache reads new prompt tokens 4.86x as fast as stock (214.7 vs 44.2 tok/s) and decodes 22.04 against 16.43 tok/s (+34 %)

Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), PC, 2026-11-05. A seeded multi-turn conversation (tools/conversation_bench.py, seed 42, 128 tokens per answer, history re-sent every turn, prompt cache on, temperature 0). Order stock, moe-cache, moe-cache, stock, two runs per arm, cold file cache before each run, a quiet-CPU gate before each run; other programs and baseline check not recorded, so cleanliness is not recorded. Launcher defaults, -ub 2048, plugin kit head5 (v0.3.6) with MOE_CACHE_LIB ggml-moe-cache-api3.dll; the record's plugin_file field names an API 2 DLL found by a file search, the server ran the API 3 DLL (settings.environment). The llama.cpp build is not recorded in this record (the same day's chain on the same kit recorded b12064). The IQ3_XXS file filled the 65,536 context in 49 turns (65,371 tokens), 42 expert layers on the CPU; the quiet-CPU gate read 1 %, 0 %, 5 %, 2 % (progress.log). Per depth band (0-16k / 16-32k / 32-48k / 48-64k), means of two runs per arm, prompt tok/s stock vs moe-cache: 40.3 vs 225.0, 52.5 vs 232.7, 44.0 vs 221.9, 38.5 vs 187.9; decode tok/s stock vs moe-cache: 15.7 vs 21.7, 12.4 vs 20.1, 12.8 vs 22.1, 18.4 vs 22.4; median time to first token stock vs moe-cache: 1.78 vs 1.06 s, 46.07 vs 9.68 s, 43.68 vs 7.20 s, 1.08 vs 1.14 s; plugin decode hit rate 97.2, 96.1, 98.2, 98.0 %. Prompt tok/s = new tokens read divided by prompt time over turns with at least 64 new tokens; decode over turns with at least 16 generated tokens.
