---
id: "decode-room-64k"
runs: ["2026-11-03_pc_1573c3"]
cells: [{"run": "2026-11-03_pc_1573c3", "arm": "-ub 2048, 40 CPU layers", "metric": "decode_tps"}, {"run": "2026-11-03_pc_1573c3", "arm": "-ub 512, 38 CPU layers", "metric": "decode_tps", "vs": "-ub 2048, 40 CPU layers"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# At 64k, 2 more expert layers on the GPU (-ub 512, 38 CPU layers) decode 15 % faster than -ub 2048 with 40

Check run behind the decode loss in v0.3.5: Flash-Next Q2_0, Windows PC, official b11966 server, cold, two runs per arm. Prompt speed was not written down. Source: docs/results-archive/RESULTS.md section 79.
