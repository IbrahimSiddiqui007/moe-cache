---
id: "chain1103-iq2xs-64k"
runs: ["2026-11-04_pc_73e2b1"]
cells: [{"run": "2026-11-04_pc_73e2b1", "arm": "pass1", "metric": "prompt_tps"}, {"run": "2026-11-04_pc_73e2b1", "arm": "pass2", "metric": "prompt_tps"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# Flash-Next IQ2_XS reads a 64k-token prompt at 317 tok/s

Flash-Next IQ2_XS on the Windows desktop, the released v0.3.5 kit with the official llama.cpp b11966, launcher defaults, context 65536, a 64k-token prompt, two passes (no stock arm), 2026-11-04 overnight chain; every run 'not clean' (owner work was running). Prompt speed: 326.2 and 307.6 tok/s. The decode speed after the prompt is not recorded: the harness stored 0.0 in both passes, which is not a speed (the same text hash and 5 re-read tokens in all four runs of this file at 64k and of IQ3_XXS at 64k; the harness' server log is not in the data).
