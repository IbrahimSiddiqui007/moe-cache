---
id: "v039-dense27-128k-spill-rule"
runs: ["2026-11-25_pc_4927e5"]
cells: [{"run": "2026-11-25_pc_4927e5", "arm": "rule-on", "metric": "prompt_tps", "vs": "rule-off"}, {"run": "2026-11-25_pc_4927e5", "arm": "rule-on", "metric": "decode_tps", "vs": "rule-off"}, {"run": "2026-11-25_pc_4927e5", "arm": "rule-off", "metric": "prompt_tps"}, {"run": "2026-11-25_pc_4927e5", "arm": "rule-off", "metric": "decode_tps"}]
valid_moe_cache: ["0.3.9", "0.3.9.1"]
valid_llama_builds: [12651, 12651]
withdrawn: ""
---
# 27B dense model (UD-IQ4_XS) at 128k that spills out of GPU memory, rule ON (-ub 256, one request at a time, our default) against OFF: prompt +41.7 % (374.2 vs 264.1 tok/s), decode +23.4 % (9.51 vs 7.70 tok/s), forecast n=2 per arm

Qwen3.8-27B UD-IQ4_XS, dense, spills out of GPU memory at 128k context; the launcher's spill rule ON (`-ub 256`, one request at a time: the default on Windows from 128k) against OFF (`MOE_CACHE_DENSE_SPILL_RULE=0`: `-ub 512`, default slots). Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5 at 6000 MT/s), 128k 6-turn conversation, KV q8_0/q8_0, kit 2e823d2 with llama.cpp b12651, apps closed (harness clean check before each run), order A B B A, 2 runs per arm, 2026-11-25. Prompt, rule on: 374.4 and 373.9 tok/s (mean 374.15); rule off: 264.4 and 263.8 (mean 264.1): +41.7 %. Decode, rule on: 9.51 and 9.50 (mean 9.505); rule off: 7.71 and 7.69 (mean 7.70): +23.4 %. The same item ran earlier the same day (04:14) with rule on 370.5 / 373.2 pp, 9.46 / 9.50 dec and rule off 263.4 / 263.3, 7.77 / 7.65 (same direction, not in the claim). The two arms differ in `-ub` and `-np`: that is the rule, so this compares the rule as a whole, not one flag. Forecast: n=2 per arm, one machine, one model, one context; it was measured with the flags set through the environment switch of the shipped kit and it does not apply below 128k or on other systems. Source: chain log handover/sessions/chain28_2026-11-25/files/records/dense27_128k_spill_rule_abba.log (the per-run record files were not on the laptop: prompt is to 1 decimal, decode to 2, fields the log does not hold say not recorded). -np is not in the log: told by the coordinator.
