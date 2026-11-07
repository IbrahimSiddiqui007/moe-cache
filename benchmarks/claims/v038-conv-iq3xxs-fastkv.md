---
id: "v038-conv-iq3xxs-fastkv"
runs: ["2026-11-07_pc_3e0cea", "2026-11-07_pc_796a07"]
cells: [{"run": "2026-11-07_pc_796a07", "arm": "moe-cache", "metric": "prompt_tps"}, {"run": "2026-11-07_pc_3e0cea", "arm": "moe-cache", "metric": "prompt_tps"}, {"run": "2026-11-07_pc_796a07", "arm": "moe-cache", "metric": "decode_tps"}, {"run": "2026-11-07_pc_3e0cea", "arm": "moe-cache", "metric": "decode_tps"}]
valid_moe_cache: ["0.3.7", "0.3.8"]
valid_llama_builds: [12064, 12064]
withdrawn: ""
---
# Head kit (v0.3.8 candidate): Flash-Next IQ3_XXS in a 65k conversation with real answers, --ub 4096 with the q8_0 KV cache: moe-cache reads new prompt tokens at 301.2 tok/s and decodes 16.5 tok/s

Same machine, kit, driver and day as v038-conv-iq3xxs-shipped, with --ub 4096 --fast=kv (q8_0 KV cache, not byte-identical to stock). Two runs, 43 turns to 65,221 tokens each, median answer 92 tokens: new prompt tokens 301.2 and 301.1 tok/s, decode 16.47 and 16.58 tok/s. Against the shipped setting: prompt +35.0 %, decode -9.6 %. The q8_0 KV meaning check passed on 2026-11-07 (night chain report); this setting is opt-in.
