---
id: "guard-glm-30k"
runs: ["2026-11-03_pc_c9fbdd"]
cells: [{"run": "2026-11-03_pc_c9fbdd", "arm": "default", "metric": "prompt_tps", "vs": "-ub 1024"}, {"run": "2026-11-03_pc_c9fbdd", "arm": "default", "metric": "decode_tps", "vs": "-ub 1024"}, {"run": "2026-11-03_pc_c9fbdd", "arm": "-ub 1024", "metric": "prompt_tps"}, {"run": "2026-11-03_pc_c9fbdd", "arm": "-ub 1024", "metric": "decode_tps"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# GLM-5.3-Flash at 30k: the new default reads prompts 62 % faster than -ub 1024; decode 6 % lower

Guard run on the Windows PC, official b11966 server, cold, order A B B A, two runs per arm. Decode was written down once per arm, so its spread is not known. Source: docs/RESULTS.md section 79.
