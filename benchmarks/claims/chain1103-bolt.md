---
id: "chain1103-bolt"
runs: ["2026-11-04_pc_00ce69"]
cells: [{"run": "2026-11-04_pc_00ce69", "arm": "stock", "metric": "prompt_tps"}, {"run": "2026-11-04_pc_00ce69", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-11-04_pc_00ce69", "arm": "plugin", "metric": "prompt_tps", "vs": "stock"}, {"run": "2026-11-04_pc_00ce69", "arm": "plugin", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# Nemotron 3.5 Bolt 74B-A9B (fits in RAM, plugin forced) decodes 31.68 tok/s with moe-cache against 29.99 stock (+6 %); the 13-token prompt reads 1.29x as fast

Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), the released v0.3.5 kit with the official llama.cpp b11966, launcher defaults, context 8192, a 13-token water request, order stock, plugin, plugin, stock (ABBA), two runs per arm, 2026-11-04 overnight chain. Every run was 'not clean' (owner work was running). Means of the two runs. Decode: stock 29.99, moe-cache 31.68 tok/s. Prompt (13 tokens, not a long-prompt figure): stock 40.75, moe-cache 52.60 tok/s.
