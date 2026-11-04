---
id: "chain1103-iq3s-64k"
runs: ["2026-11-04_pc_575654"]
cells: [{"run": "2026-11-04_pc_575654", "arm": "pass1", "metric": "prompt_tps"}, {"run": "2026-11-04_pc_575654", "arm": "pass1", "metric": "decode_tps"}, {"run": "2026-11-04_pc_575654", "arm": "pass2", "metric": "prompt_tps"}, {"run": "2026-11-04_pc_575654", "arm": "pass2", "metric": "decode_tps"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# Flash-Next IQ3_S reads a 64k-token prompt at 221 tok/s; the answer after it decodes at 11.9 tok/s

Flash-Next IQ3_S on the Windows desktop, the released v0.3.5 kit with the official llama.cpp b11966, launcher defaults, context 65536, a 64k-token prompt, two passes (no stock arm), 2026-11-04 overnight chain; every run 'not clean' (owner work was running). Prompt speed: 221.6 and 219.7 tok/s. Decode of the short answer after the prompt: 12.77 and 11.04 tok/s.
