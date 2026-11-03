---
id: "card-decode-122b-first"
runs: ["2026-10-27_pc_5a4f0a"]
cells: [{"run": "2026-10-27_pc_5a4f0a", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-10-27_pc_5a4f0a", "arm": "moe-cache", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["not recorded", "not recorded"]
valid_llama_builds: ["not recorded", "not recorded"]
withdrawn: "ran during a slowdown of the whole PC (the same CPU work took 22 % longer, stock 13 % slower); corrected by card-decode-122b (RESULTS 61)"
---
# Qwen3.5-122B IQ2_XXS decodes 16.13 tok/s with moe-cache against 13.88 stock (+16 %)

Windows PC, launcher defaults, cold file cache, llama.cpp 620d2225c (build number not recorded), moe-cache version not recorded. Only the decode speed of the water-cycle request was written down per run; prompt speed, VRAM, RAM and CPU were not recorded. Source: docs/RESULTS.md section 59.
