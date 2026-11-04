---
id: "chain1103-iq3xxs-64k"
runs: ["2026-11-04_pc_bdd9ca"]
cells: [{"run": "2026-11-04_pc_bdd9ca", "arm": "pass1", "metric": "prompt_tps"}, {"run": "2026-11-04_pc_bdd9ca", "arm": "pass2", "metric": "prompt_tps"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# Flash-Next IQ3_XXS reads a 64k-token prompt at 182 tok/s

Flash-Next IQ3_XXS on the Windows desktop, the released v0.3.5 kit with the official llama.cpp b11966, launcher defaults, context 65536, a 64k-token prompt, two passes (no stock arm), 2026-11-04 overnight chain; every run 'not clean' (owner work was running). Prompt speed: 181.3 and 183.3 tok/s. The decode speed after the prompt is not recorded. After the 61,843-token prompt the model wrote only 2 tokens and ended its answer (end-of-sequence), so the follow-up request that measures decode after the prompt generated 0 tokens; this is the model's behaviour on that prompt, not a plugin fault, and the prompt speed is valid. The harness will ignore the end-of-sequence token on that request in the next chain.
