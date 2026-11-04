---
id: "chain1103-iq2xs-64k-decode"
runs: ["2026-11-04_pc_755df6"]
cells: [{"run": "2026-11-04_pc_755df6", "arm": "pass1", "metric": "decode_tps"}, {"run": "2026-11-04_pc_755df6", "arm": "pass2", "metric": "decode_tps"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: [11966, 11966]
withdrawn: ""
---
# Flash-Next IQ2_XS decodes the short answer after a 64k-token prompt at 16.63 and 17.18 tok/s (daytime re-run, not clean)

Flash-Next IQ2_XS on the Windows desktop, the released v0.3.5 kit with the official llama.cpp b11966, launcher defaults, context 65536, a 61,843-token prompt, two passes, 2026-11-04 daytime re-run with the fixed harness (the decode-after request now ignores end-of-answer and generates 128 tokens). NOT CLEAN: the owner's browser held 1.4 GB of VRAM, so the prompt speeds of this run are about 40 % below the clean overnight runs and are not claimed here; use the overnight claim for prompt speed. Decode after the prompt: 16.63 and 17.18 tok/s. A clean re-run is planned. The overnight run had no decode value because the model ended its answer after 2 tokens (see the overnight claim).
