---
id: "guard-122b-64k"
runs: ["2026-11-03_pc_7a512d"]
cells: [{"run": "2026-11-03_pc_7a512d", "arm": "default", "metric": "prompt_tps", "vs": "-ub 1024"}, {"run": "2026-11-03_pc_7a512d", "arm": "-ub 1024", "metric": "prompt_tps"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# Qwen3.5-122B IQ2_XXS at 64k: the new default reads prompts 24 % faster than -ub 1024

Guard run: no prompt regression on the 122B. Windows PC, official b11966 server, cold, order A B B A, two runs per arm. Decode was not measured (the default arm's answer ended after 2 tokens). Source: docs/results-archive/RESULTS.md section 79.
