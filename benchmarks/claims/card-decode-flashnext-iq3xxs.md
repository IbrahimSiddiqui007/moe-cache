---
id: "card-decode-flashnext-iq3xxs"
runs: ["2026-10-27_pc_8152c4"]
cells: [{"run": "2026-10-27_pc_8152c4", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-10-27_pc_8152c4", "arm": "moe-cache", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["not recorded", "not recorded"]
valid_llama_builds: ["not recorded", "not recorded"]
withdrawn: ""
---
# Flash-Next IQ3_XXS decodes 19.86 tok/s with moe-cache against 15.14 stock (+31 %)

Windows PC, launcher defaults, cold file cache, llama.cpp 620d2225c (build number not recorded), moe-cache version not recorded. Only the decode speed of the water-cycle request was written down per run; prompt speed, VRAM, RAM and CPU were not recorded. Cold ABBA, drive D:, 9 GPU expert layers. Source: docs/RESULTS.md section 59; the benchmark card.
