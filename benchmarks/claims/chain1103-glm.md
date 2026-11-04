---
id: "chain1103-glm"
runs: ["2026-11-04_pc_bea283"]
cells: [{"run": "2026-11-04_pc_bea283", "arm": "stock", "metric": "prompt_tps"}, {"run": "2026-11-04_pc_bea283", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-11-04_pc_bea283", "arm": "plugin", "metric": "prompt_tps", "vs": "stock"}, {"run": "2026-11-04_pc_bea283", "arm": "plugin", "metric": "decode_tps", "vs": "stock"}, {"run": "2026-11-04_pc_bea283", "arm": "mtp", "metric": "prompt_tps", "vs": "stock"}, {"run": "2026-11-04_pc_bea283", "arm": "mtp", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# GLM-5.3-Flash UD-IQ2_XXS decodes 6.27 tok/s with moe-cache against 2.98 stock (+110 %); the 13-token prompt reads 2.32x as fast; with --mtp 8.17 (2.74x stock)

Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), the released v0.3.5 kit with the official llama.cpp b11966, launcher defaults, context 8192, a 13-token water request, order stock, plugin, plugin, stock (ABBA), two runs per arm, 2026-11-04 overnight chain. Every run was 'not clean' (owner work was running). Means of the two runs. Decode: stock 2.98, moe-cache 6.27 tok/s. Prompt (13 tokens, not a long-prompt figure): stock 1.25, moe-cache 2.90 tok/s. With --mtp: decode 8.17 tok/s.
