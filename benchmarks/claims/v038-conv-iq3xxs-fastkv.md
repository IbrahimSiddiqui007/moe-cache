---
id: "v038-conv-iq3xxs-fastkv"
runs: ["2026-11-07_pc_719b3b", "2026-11-07_pc_53db60"]
cells: [{"run": "2026-11-07_pc_53db60", "arm": "moe-cache", "metric": "prompt_tps"}, {"run": "2026-11-07_pc_719b3b", "arm": "moe-cache", "metric": "prompt_tps"}, {"run": "2026-11-07_pc_53db60", "arm": "moe-cache", "metric": "decode_tps"}, {"run": "2026-11-07_pc_719b3b", "arm": "moe-cache", "metric": "decode_tps"}]
valid_moe_cache: ["0.3.7", "0.3.8"]
valid_llama_builds: [12064, 12064]
withdrawn: "2026-11-08: measured with all 48 expert layers on the CPU, not the long-prompt plan as shipped; replaced by v038-longprompt-default (45 expert layers on the CPU, a3dac52)."
---
# Head kit (v0.3.8 candidate): Flash-Next IQ3_XXS in a 65k conversation with real answers, --ub 4096 with the q8_0 KV cache: moe-cache reads new prompt tokens at 301.2 tok/s and decodes 16.5 tok/s

Same machine, kit, driver and day as v038-conv-iq3xxs-shipped, with --ub 4096 --fast=kv (q8_0 KV cache, not byte-identical to stock). Two runs, 43 turns to 65,221 tokens each, median answer 92 tokens: new prompt tokens 301.2 and 301.1 tok/s, decode 16.47 and 16.58 tok/s. Against the shipped setting: prompt +35.0 %, decode -9.6 %. The q8_0 KV meaning check passed on 2026-11-07 (night chain report); this setting is opt-in. Correction of 2026-11-07: the first two records of these runs (`_796a07`, `_3e0cea`) show the default plan in their settings and command line (42 expert layers on the CPU, f16 KV) because the driver recorded the launcher command without the extra options. This claim now points at the corrected records `_53db60` (run C2) and `_719b3b` (run C3), which have the same numbers and settings that follow the launcher's start-up lines and the plugin log (the originals stay in runs/, append-only). What ran: `--ub 4096 --fast=kv`, q8_0 KV cache, **all 48 expert layers on the CPU side** (the launcher's buffer estimate for `-ub 4096` left no GPU room). The numbers are for that plan, not for the long-prompt mode as built in v0.3.8 (which keeps a few layers on the GPU and is being measured again).
