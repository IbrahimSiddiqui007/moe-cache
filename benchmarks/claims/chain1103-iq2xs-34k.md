---
id: "chain1103-iq2xs-34k"
runs: ["2026-11-04_pc_af1241"]
cells: [{"run": "2026-11-04_pc_af1241", "arm": "pass1", "metric": "prompt_tps"}, {"run": "2026-11-04_pc_af1241", "arm": "pass1", "metric": "decode_tps"}, {"run": "2026-11-04_pc_af1241", "arm": "pass2", "metric": "prompt_tps"}, {"run": "2026-11-04_pc_af1241", "arm": "pass2", "metric": "decode_tps"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# Flash-Next IQ2_XS reads a 34k-token prompt at 504 tok/s; the answer after it decodes at 24.7 tok/s

Flash-Next IQ2_XS on the Windows desktop, the released v0.3.5 kit with the official llama.cpp b11966, launcher defaults, context 40960, a 34k-token prompt, two passes (no stock arm), 2026-11-04 overnight chain; every run 'not clean' (owner work was running). Prompt speed: 530.4 and 477.0 tok/s. Decode of the short answer after the prompt: 24.68 and 24.77 tok/s.
