---
id: "v038-fits-qwen36"
runs: ["2026-11-08_pc_0f2bf9"]
cells: [{"run": "2026-11-08_pc_0f2bf9", "arm": "stock", "metric": "prompt_tps"}, {"run": "2026-11-08_pc_0f2bf9", "arm": "stock", "metric": "decode_tps"}, {"run": "2026-11-08_pc_0f2bf9", "arm": "moe-cache", "metric": "prompt_tps", "vs": "stock"}, {"run": "2026-11-08_pc_0f2bf9", "arm": "moe-cache", "metric": "decode_tps", "vs": "stock"}]
valid_moe_cache: ["0.3.7.1", "0.3.8"]
valid_llama_builds: [12064, 12064]
withdrawn: ""
---
# v0.3.8 head kit: Qwen3.6-35B (a model that fits in RAM, plugin forced on) in a 65k conversation with real answers: moe-cache 1.06x prompt (2200 vs 2072 tok/s) and 1.07x decode (96.2 vs 90.2 tok/s): about the same or slightly faster

Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5 at 6000 MT/s since 2026-11-07; 4800 MT/s records are not comparable), head kit khead (3342707) with llama.cpp b12064, model Qwen3.6-35B-A3B UD-IQ3_XXS (small quant that fits in RAM, 7 expert layers on the CPU), plugin forced on (the launcher itself runs stock llama.cpp for such a model), context 65,536, driver 9a0a981, seed 42, order stock, moe-cache, moe-cache, stock, 2 runs per arm, 2026-11-08. Per depth band, prompt tok/s stock vs moe-cache: 1840.7 vs 2381.1, 2544.6 vs 2688.2, 2148.3 vs 2239.1, 1678.1 vs 1747.2 (1.29x, 1.06x, 1.04x, 1.04x); decode tok/s: 91.6 vs 97.0, 86.5 vs 94.2, not enough answer tokens, not enough answer tokens (1.06x, 1.09x, -, -); the deeper bands have too few long answers for a decode figure. One model on one machine: the result is within what we would call noise for decode, the launcher default (stock for a model that fits) is unchanged.
