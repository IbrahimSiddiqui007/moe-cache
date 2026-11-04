---
id: "chain1103-qwen36"
runs: ["2026-11-04_pc_92657e"]
cells: [{"run": "2026-11-04_pc_92657e", "arm": "stock", "metric": "prompt_tps"}, {"run": "2026-11-04_pc_92657e", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-11-04_pc_92657e", "arm": "plugin", "metric": "prompt_tps", "vs": "stock"}, {"run": "2026-11-04_pc_92657e", "arm": "plugin", "metric": "decode_tps", "vs": "stock"}, {"run": "2026-11-04_pc_92657e", "arm": "mtp", "metric": "prompt_tps", "vs": "stock"}, {"run": "2026-11-04_pc_92657e", "arm": "mtp", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# Qwen3.6-35B-A3B Q4_K_M (fits in RAM, plugin forced) decodes 51.80 tok/s with moe-cache against 46.59 stock (+11 %); the 13-token prompt reads 1.23x as fast; with --mtp 72.69 (1.56x stock)

Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), the released v0.3.5 kit with the official llama.cpp b11966, launcher defaults, context 8192, a 13-token water request, order stock, plugin, plugin, stock (ABBA), two runs per arm, 2026-11-04 overnight chain. Every run was 'not clean' (owner work was running). Means of the two runs. Decode: stock 46.59, moe-cache 51.80 tok/s. Prompt (13 tokens, not a long-prompt figure): stock 89.85, moe-cache 110.45 tok/s. With --mtp: decode 72.69 tok/s.
