---
id: "chain1103-iq3s"
runs: ["2026-11-04_pc_42b419"]
cells: [{"run": "2026-11-04_pc_42b419", "arm": "stock", "metric": "prompt_tps"}, {"run": "2026-11-04_pc_42b419", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-11-04_pc_42b419", "arm": "plugin", "metric": "prompt_tps", "vs": "stock"}, {"run": "2026-11-04_pc_42b419", "arm": "plugin", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# Flash-Next IQ3_S decodes 19.62 tok/s with moe-cache against 13.79 stock (+42 %); the 13-token prompt reads 2.36x as fast

Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), the released v0.3.5 kit with the official llama.cpp b11966, launcher defaults, context 8192, a 13-token water request, order stock, plugin, plugin, stock (ABBA), two runs per arm, 2026-11-04 overnight chain. Every run was 'not clean' (owner work was running). Means of the two runs. Decode: stock 13.79, moe-cache 19.62 tok/s. Prompt (13 tokens, not a long-prompt figure): stock 9.40, moe-cache 22.20 tok/s.
