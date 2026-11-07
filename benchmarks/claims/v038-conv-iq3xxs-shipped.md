---
id: "v038-conv-iq3xxs-shipped"
runs: ["2026-11-07_pc_87fefb", "2026-11-07_pc_d38756"]
cells: [{"run": "2026-11-07_pc_87fefb", "arm": "moe-cache", "metric": "prompt_tps"}, {"run": "2026-11-07_pc_d38756", "arm": "moe-cache", "metric": "prompt_tps"}, {"run": "2026-11-07_pc_87fefb", "arm": "moe-cache", "metric": "decode_tps"}, {"run": "2026-11-07_pc_d38756", "arm": "moe-cache", "metric": "decode_tps"}]
valid_moe_cache: ["0.3.7", "0.3.8"]
valid_llama_builds: [12064, 12064]
withdrawn: ""
---
# Head kit (v0.3.8 candidate): Flash-Next IQ3_XXS in a 65k conversation with real answers, shipped setting: moe-cache reads new prompt tokens at 223.1 tok/s and decodes 18.3 tok/s

Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), head kit 3342707 with the official llama.cpp b12064, plugin forced on, the shipped setting (--ub 2048, f16 KV cache), context 65,536, cold file cache, 2026-11-07. Conversation driver 9a0a981: thinking off, 256 tokens, an answer hint on every question, sizes by the server's token count. Two runs, 42 turns to 65,280 tokens each, median answer 85 tokens: new prompt tokens 223.1 and 223.2 tok/s, decode 18.33 and 18.23 tok/s. There is no stock arm in these runs, so no ratio against stock is claimed; the stock numbers of v0.3.7 were measured with the old driver (short answers from turn 11).
