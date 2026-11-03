---
id: "card-decode-nemotron"
runs: ["2026-10-28_pc_c724d9"]
cells: [{"run": "2026-10-28_pc_c724d9", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-10-28_pc_c724d9", "arm": "moe-cache (forced)", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["not recorded", "not recorded"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# Nemotron 3.5 Bolt 74B-A9B (fits in RAM) decodes 22.82 tok/s with moe-cache forced against 27.91 stock (-18 %)

Windows PC, cold, llama.cpp b11966, one run each; the file (22.0 GB) fits in RAM so the launcher runs stock by itself and the plugin was forced only to compare. Only decode speed was written down per run. Source: docs/RESULTS.md section 65; the benchmark card.
