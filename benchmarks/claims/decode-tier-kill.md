---
id: "decode-tier-kill"
runs: ["2026-11-03_pc_93ede2"]
cells: [{"run": "2026-11-03_pc_93ede2", "arm": "TIER", "metric": "prompt_tps", "vs": "BASE"}, {"run": "2026-11-03_pc_93ede2", "arm": "TIER", "metric": "decode_tps", "vs": "BASE"}, {"run": "2026-11-03_pc_93ede2", "arm": "BASE", "metric": "prompt_tps"}, {"run": "2026-11-03_pc_93ede2", "arm": "BASE", "metric": "decode_tps"}]
valid_moe_cache: ["0.3.5", "0.3.5"]
valid_llama_builds: ["not recorded", "not recorded"]
withdrawn: ""
---
# Decode tier (2 layers, 38 and 39) did not speed decode up and cut prompt speed by about 40 %: killed

Kill check of the prototype MOE_CACHE_DECODE_TIER on the Windows PC (design: docs/DECODE_TIER_DESIGN.md). Check mode passed (0 groups above 0.05). The verdict rule was: keep at +6 % decode with prompts within 3 %, kill under +3 %. Context, prompt length, kit and time of day were not recorded; the BASE prompt pair differs by 20 % and was not explained. Source: overnight run notes of 2026-11-03.
