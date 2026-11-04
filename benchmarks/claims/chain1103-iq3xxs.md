---
id: "chain1103-iq3xxs"
runs: ["2026-11-04_pc_cfa905"]
cells: [{"run": "2026-11-04_pc_cfa905", "arm": "stock", "metric": "prompt_tps"}, {"run": "2026-11-04_pc_cfa905", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-11-04_pc_cfa905", "arm": "plugin", "metric": "prompt_tps", "vs": "stock"}, {"run": "2026-11-04_pc_cfa905", "arm": "plugin", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# Flash-Next IQ3_XXS decodes 22.87 tok/s with moe-cache against 14.96 stock (+53 %); the 13-token prompt reads 2.08x as fast

Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), the released v0.3.5 kit with the official llama.cpp b11966, launcher defaults, context 8192, a 13-token water request, order stock, plugin, plugin, stock (ABBA), two runs per arm, 2026-11-04 overnight chain. Every run was 'not clean' (owner work was running). Means of the two runs. Decode: stock 14.96, moe-cache 22.87 tok/s. Prompt (13 tokens, not a long-prompt figure): stock 12.05, moe-cache 25.05 tok/s.
