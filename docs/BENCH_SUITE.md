# Benchmark suite: test models against each other

`moe_cache.py bench-suite` measures a model with the same tasks and settings as another model and puts the results side by side, one card per model. The records stay on your computer (private) until you publish one; nothing is uploaded.

| What it measures | How | In the record |
|---|---|---|
| Speed | decode and prompt tok/s by depth (a real multi-turn conversation in `standard` and `night`, one probe in `quick`), TTFT | `speed` |
| Memory | server RAM and GPU memory peak during the conversation | `memory` |
| Quality | mean KL and same-top-token against the same model with an f16 KV cache (llama-perplexity, real-text windows) | `quality` |
| Tasks | the 202 short tasks, thinking off and on | `tasks`, `tasks_on` |
| Hard reasoning | 27 long-reasoning tasks without an answer cap: pass, thinking tokens (median, p90, max), loops, cap hits | `hard` |

## Run it

```
python3 bin/moe_cache.py bench-suite run MODEL.gguf [MODEL2.gguf ...] --profile quick|standard|night [--kv-flags "-ctk q8_0 -ctv q4_0"]
python3 bin/moe_cache.py bench-suite run --url http://127.0.0.1:8080 --name my-model      (a server that is already running)
python3 bin/moe_cache.py bench-suite compare RECORD.json RECORD2.json ...                  (table; the first record is the reference)
python3 bin/moe_cache.py bench-suite card RECORD.json RECORD2.json --out cards.html        (one self-contained HTML page)
python3 bin/moe_cache.py bench-suite validate|show|new|publish ...
```

`quick` takes about 15 minutes on a small model, `standard` about 2 hours, `night` several hours (the hard set has no answer cap). Each model is started through the launcher, one at a time, so the plan is the one you would get. Records are written to `benchmarks/suites/` in a checkout (git-ignored) or to `~/.local/share/moe-cache/suites/` (Windows: `%LOCALAPPDATA%\moe-cache\suites\`).

## In the control page

**⚖ Compare** lists your records, lets you tick two or more (the first ticked is the reference), shows one card per model with the change against the reference (green better, red worse, grey within 2 %), imports a record file, starts a suite (stop the page's own model first: a second model would not fit in memory) and publishes a record (a flag on this computer; nothing is shown publicly until the records are reviewed).

## Fair comparisons

Every section carries a protocol hash (task file, windows file, request settings, answer cap, KV type, thinking mode). A section measured with another protocol is shown as `-` and a note says which one differs. The 202-task and hard sets use temperature 0 and the chat template's own thinking default. A model-against-model comparison rests on the task pass rates; KL is always against a reference (by default the same model with f16 KV), so it measures what a quantisation or KV change costs, not which model is smarter.

Schema: [suite_record.schema.json](suite_record.schema.json). The hard set: `tools/hard_tasks.json` (`tools/hard_tasks_make.py` rebuilds it).
