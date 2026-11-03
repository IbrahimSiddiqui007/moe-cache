---
id: "card-decode-122b"
runs: ["2026-10-28_pc_519336"]
cells: [{"run": "2026-10-28_pc_519336", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-10-28_pc_519336", "arm": "moe-cache", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["not recorded", "not recorded"]
valid_llama_builds: ["not recorded", "not recorded"]
withdrawn: ""
---
# Qwen3.5-122B IQ2_XXS decodes 19.73 tok/s with moe-cache against 15.95 stock (+24 %)

Windows PC, launcher defaults, cold file cache, llama.cpp 620d2225c (build number not recorded), moe-cache version not recorded. Only the decode speed of the water-cycle request was written down per run; prompt speed, VRAM, RAM and CPU were not recorded. Cold ABBA, drive D:, 12 GPU expert layers; re-measured on 2026-10-28. This replaces the withdrawn claim card-decode-122b-first. Source: docs/RESULTS.md section 61; the benchmark card.
