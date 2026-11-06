---
id: "card-decode-flashnext-iq3s"
runs: ["2026-10-27_pc_1338b9"]
cells: [{"run": "2026-10-27_pc_1338b9", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-10-27_pc_1338b9", "arm": "moe-cache", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["not recorded", "not recorded"]
valid_llama_builds: ["not recorded", "not recorded"]
withdrawn: ""
---
# Flash-Next IQ3_S decodes 17.28 tok/s with moe-cache against 12.92 stock (+34 %)

Windows PC, launcher defaults, cold file cache, llama.cpp 620d2225c (build number not recorded), moe-cache version not recorded. Only the decode speed of the water-cycle request was written down per run; prompt speed, VRAM, RAM and CPU were not recorded. Cold ABBA, drive C:, 7 GPU expert layers. Source: docs/results-archive/RESULTS.md section 59; the benchmark card.
