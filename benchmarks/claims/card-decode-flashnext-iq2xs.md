---
id: "card-decode-flashnext-iq2xs"
runs: ["2026-10-27_pc_01160a"]
cells: [{"run": "2026-10-27_pc_01160a", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-10-27_pc_01160a", "arm": "moe-cache", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["not recorded", "not recorded"]
valid_llama_builds: ["not recorded", "not recorded"]
withdrawn: ""
---
# Flash-Next IQ2_XS decodes 21.94 tok/s with moe-cache against 14.89 stock (+47 %)

Windows PC, launcher defaults, cold file cache, llama.cpp 620d2225c (build number not recorded), moe-cache version not recorded. Only the decode speed of the water-cycle request was written down per run; prompt speed, VRAM, RAM and CPU were not recorded. Cold ABBA, drive C:, 12 GPU expert layers. Source: docs/results-archive/RESULTS.md section 59; the benchmark card.
