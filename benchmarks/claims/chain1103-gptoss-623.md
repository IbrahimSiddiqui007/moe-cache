---
id: "chain1103-gptoss-623"
runs: ["2026-11-04_pc_76a45c"]
cells: [{"run": "2026-11-04_pc_76a45c", "arm": "cc4fe9e (yesterday's code)", "metric": "prompt_tps"}, {"run": "2026-11-04_pc_76a45c", "arm": "cc4fe9e (yesterday's code)", "metric": "decode_tps"}, {"run": "2026-11-04_pc_76a45c", "arm": "f8899ca (today's code)", "metric": "prompt_tps", "vs": "cc4fe9e (yesterday's code)"}, {"run": "2026-11-04_pc_76a45c", "arm": "f8899ca (today's code)", "metric": "decode_tps", "vs": "cc4fe9e (yesterday's code)"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [12064, 12064]
withdrawn: ""
---
# gpt-oss-120b reads a 30k-token prompt at 114.6 tok/s with today's code against 73.2 with yesterday's (+57 %)

gpt-oss-120b MXFP4 on the Windows desktop, two experimental kits (cc4fe9e = yesterday's code, f8899ca = today's code, which has the CACHE-623 fix) with the official llama.cpp b12064, context 65536, -ub 1024, a 30,374-token prompt, order A B B A, two runs per arm, 2026-11-04 overnight chain; every run 'not clean' (owner work was running). Prompt speed: yesterday's 73.1 and 73.2, today's 114.6 and 114.7 tok/s. Decode after the prompt: 8.8 (8.81 and 8.74) against 9.1 (9.12 and 9.20) tok/s. Not the released v0.3.5 kit.
