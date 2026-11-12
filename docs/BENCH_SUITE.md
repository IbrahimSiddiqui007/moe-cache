# Benchmark suite: test models against each other

`moe_cache.py bench-suite` measures a model with the same tasks as another model and puts the results side by side, one card per model. The records stay on your computer (private) until you publish one; nothing is uploaded.

| What it measures | How | In the record |
|---|---|---|
| Speed | decode and prompt tok/s by depth (a real multi-turn conversation in `standard` and `night`, one probe in `quick`), TTFT. One run per number. | `speed` |
| Memory | server RAM and GPU memory peak while the conversation runs (needs a server the suite started; a running server given with `--url` has no memory section) | `memory` |
| KV cache cost | mean KL and same-top-token of the KV cache the server really ran with, against the same model with an f16 KV cache (llama-perplexity, 8 chat-style windows of 8192 tokens, the second half of each window is scored). Left out when the served KV cache is f16 (nothing to measure). It is what the KV type costs, not how good the model is. | `quality` |
| Tasks | the 202 short tasks (20 spread over the file in `quick`), thinking off and on; scored on the text after the think block only | `tasks`, `tasks_on` |
| Hard reasoning | 27 long-reasoning tasks without an answer cap: pass, thinking tokens (median, p90, max), loops, cap hits, errors | `hard` |

## Run it

```
python3 bin/moe_cache.py bench-suite run MODEL.gguf [MODEL2.gguf ...] --profile quick|standard|night [--launch-args "-- -ctk q8_0 -ctv q8_0"] [--windows MY_CHATS.txt] [--fresh]
python3 bin/moe_cache.py bench-suite run --url http://127.0.0.1:8080 --name my-model      (a server that is already running)
python3 bin/moe_cache.py bench-suite compare RECORD.json RECORD2.json ...                  (table; the first record is the reference)
python3 bin/moe_cache.py bench-suite card RECORD.json RECORD2.json --out cards.html        (one self-contained HTML page)
python3 bin/moe_cache.py bench-suite validate|show|new|publish ...
```

Each model is started through the launcher on its own free port, one at a time, with the context the profile needs (12k `quick`, 32k `standard`, 65k `night`; `--launch-args "--ctx N"` replaces it). What the server really ran (context, slots, KV types, build, plan flags, command line) is read back and written to the record's `served` block, next to the machine and the kit commit. A model file other than the one asked for on that port is refused.

Time: measured on a laptop with a 1B-class model, `quick` took under a minute. Big models take much longer (decode speed decides): `standard` is an hour or more, `night` many hours (the hard set has no answer cap; a task that runs over 3600 s counts as a cap hit and the run goes on). Each finished section is saved at once to a `.partial` file next to the records; if a run fails or is cancelled, a record marked *partial* with what was measured is written, and running the same command again resumes after the last finished section (`--fresh` starts over). The hard set resumes per task.

Records are written to `benchmarks/suites/` in a checkout (git-ignored) or to `~/.local/share/moe-cache/suites/` (Windows: `%LOCALAPPDATA%\moe-cache\suites\`).

## In the control page

**⚖ Compare** lists your records, lets you tick two or more (the first ticked is the reference), shows one card per model with the change against the reference, imports a record file, starts a suite (stop the page's own model first: a second model would not fit in memory; starting a model by hand is refused while a suite runs) and publishes a record (a flag on this computer; nothing is shown publicly until the records are reviewed). Cancel stops the whole run, the launcher and the llama-server; finished sections are kept.

## Reading the numbers

- Every speed, memory and thinking-length number is **one run**; the cards say "1 run". Green or red appears only for a change of 10 % or more. Run it again before you believe a small difference.
- Pass counts show the number of tasks (`n=`) and are marked better or worse only when their 95 % intervals (Wilson) do not overlap: 20 against 24 of 27 is not a difference. A loop or cap-hit count 0 against 1 is not either.
- The KL rows between two *different* models carry no mark: each is the cost of its own KV cache against its own f16 run.
- The `loops` count uses a strict rule (a 32-token window seen four times); the old 12 x 3 rule flagged long legitimate reasoning, and its false-positive rate on the 27B is not measured yet. `loops_in_passed` in the record counts loops in tasks that passed anyway.
- An answer is accepted only after the last `</think>`; a text cut inside the think block is a *cut*, not a pass, and counts as a cap hit. A model whose template has no think tag shows 0 thinking tokens.

## Fair comparisons

Every section carries a protocol hash (task file, windows file, request settings, answer cap, thinking mode; the hard set and the speed sections also the served context). Besides that, the records carry the server settings. `compare` does not compare speed and memory across a known difference in machine, kit, llama.cpp build, context, slots or KV types, and the hard set not across another context, slots, KV types or build (with no cap the context is the cap); it shows `-` and a line says which setting differs. Tasks and KV cost are compared with a warning. A setting that one record does not carry is reported as "not recorded".

Two models measured in two separate runs were measured at different times (laptop temperature, other programs, file cache); run both on a quiet machine and treat small speed differences with care.

Schema: [suite_record.schema.json](suite_record.schema.json). The hard set: `tools/hard_tasks.json` (`tools/hard_tasks_make.py` rebuilds it; `--verify` re-derives every answer from its own prompt text for all eight categories). It has some well-known puzzles (SEND+MORE=MONEY, the Collatz start 837799) that a model may know by heart: they test memory as much as reasoning.
