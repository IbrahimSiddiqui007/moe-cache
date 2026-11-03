---
id: "card-decode-qwen36"
runs: ["2026-10-27_pc_1e5bc6"]
cells: [{"run": "2026-10-27_pc_1e5bc6", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-10-27_pc_1e5bc6", "arm": "moe-cache (forced)", "metric": "decode_tps", "vs": "stock"}, {"run": "2026-10-27_pc_1e5bc6", "arm": "stock + MTP", "metric": "decode_tps", "vs": "stock"}, {"run": "2026-10-27_pc_1e5bc6", "arm": "moe-cache (forced) + MTP", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["not recorded", "not recorded"]
valid_llama_builds: ["not recorded", "not recorded"]
withdrawn: ""
---
# Qwen3.6-35B-A3B (fits in RAM) decodes at the same speed with or without the plugin (46.06 stock, 45.60 forced); with MTP 70.25 against 68.60 stock

Windows PC, launcher defaults, cold file cache, llama.cpp 620d2225c (build number not recorded), moe-cache version not recorded. Only the decode speed of the water-cycle request was written down per run; prompt speed, VRAM, RAM and CPU were not recorded. Two cold runs each; about 78 % of MTP drafts accepted. Source: docs/RESULTS.md section 58; the benchmark card.
