---
id: "card-decode-glm"
runs: ["2026-10-27_pc_7e482e"]
cells: [{"run": "2026-10-27_pc_7e482e", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-10-27_pc_7e482e", "arm": "moe-cache", "metric": "decode_tps", "vs": "stock"}, {"run": "2026-10-27_pc_7e482e", "arm": "moe-cache --mtp", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["not recorded", "not recorded"]
valid_llama_builds: ["not recorded", "not recorded"]
withdrawn: ""
---
# GLM-5.3-Flash decodes 4.44 tok/s with moe-cache against 2.21 stock (2.0x); with --mtp 6.92 (3.1x)

Windows PC, launcher defaults, cold file cache, llama.cpp 620d2225c (build number not recorded), moe-cache version not recorded. Only the decode speed of the water-cycle request was written down per run; prompt speed, VRAM, RAM and CPU were not recorded. One cold run per arm on drive C:, 3 GPU expert layers. Later moe-cache runs gave 5.36-5.89 tok/s but stock was never re-run, so those ratios are not a fair pair (RESULTS 61-62). Source: docs/results-archive/RESULTS.md sections 59-60 (the card cites 61 for the MTP row).
