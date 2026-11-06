---
id: "card-decode-flashnext-q2_0"
runs: ["2026-10-28_pc_26f9c6"]
cells: [{"run": "2026-10-28_pc_26f9c6", "arm": "stock", "metric": "prompt_tps"}, {"run": "2026-10-28_pc_26f9c6", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-10-28_pc_26f9c6", "arm": "moe-cache", "metric": "prompt_tps", "vs": "stock"}, {"run": "2026-10-28_pc_26f9c6", "arm": "moe-cache", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["not recorded", "not recorded"]
valid_llama_builds: ["not recorded", "not recorded"]
withdrawn: ""
---
# Flash-Next Q2_0 decodes 22.16 tok/s with moe-cache against 18.41 stock (+20 %); a 3k-token prompt reads 2.33x faster

Windows PC, launcher defaults, cold file cache, llama.cpp 620d2225c (build number not recorded), moe-cache version not recorded. Only the decode speed of the water-cycle request was written down per run; prompt speed, VRAM, RAM and CPU were not recorded. Two runs each, not interleaved. Source: docs/results-archive/RESULTS.md section 63; the benchmark card's decode table.
