# Conversation benchmark: a long chat that fills the context

One 32k prompt, or a chat at 8k context, does not show what a user sees in a long coding or agent session. There the model reads a history that grows turn
after turn: most of it comes back from the server's prompt cache, but every new paste (a file, a log, a diff) still has to be read, and every answer is
decoded at a deeper and deeper position. `tools/conversation_bench.py` plays such a session against a running `llama-server` and records what happened turn
by turn. It is a measuring tool: nothing in this page is a measured result. Numbers go into this file only after a run has been recorded.

## What one run does

- A fixed script of user turns, built from a seed (default 42): small follow-up questions, code pastes, tool-output-like logs and a few big pastes. The same
  seed gives the same conversation every time, so stock and moe-cache read exactly the same text. `--script FILE` plays your own JSON list of messages
  instead.
- Every turn is sent with the whole history so far (system message, all earlier turns and the model's earlier answers), `cache_prompt` on, temperature 0, a
  fixed `--max-tokens` per answer (default 256; 128 with `--legacy`), thinking off, and every user message ends with "Please answer in at least three full sentences." (see "Visible answers" below). The answer is appended and the next turn follows, until the next turn would not fit into the context (`--ctx`,
  default 65536; it is cut to the server's own `n_ctx` when `/props` reports one). The tool never sends a request it expects the server to refuse: before each turn it asks the server's `/tokenize` for the real size of the new user message (a characters-per-token guess only when the server has no `/tokenize`), adds the real depth of the last answer, a margin for the chat template and `--max-tokens`, and stops with `context full` when that would not fit. If the server still refuses a later turn as too big for the context (HTTP 400 "exceeds the available context size"), the run ends as `context full` too, not as an error; a refusal of the very first turn stays an error (the context is too small).
- It stops with a reason: `context full`, `max turns reached`, `script ended`, or `error: ...` (the server's answer, for example an HTTP 400).

Per turn it records: the turn number and kind, tokens added since the last answer, the depth before and after the answer, new tokens read and tokens reused
from the cache, prompt and decode tok/s (the server's own `timings`), time to first token, and, when it can see them, the plugin's `hits_dec` and `misses_dec`
for that turn. A value the server did not give is `null` in the run file and "not recorded" in the record.

Per depth band (a quarter of the context each, 0-16k, 16-32k, 32-48k and 48-64k at 65536): the number of turns that start in the band, prompt tok/s
(new tokens divided by prompt time, over turns that read at least 64 new tokens), decode tok/s (over turns that generated at least 16 tokens), the share of the
prompt that came from the cache, the median time to first token, and the plugin's decode hit rate.

## What every run records besides the turns

The owner's full-metrics rule: a run is only useful with its conditions. Each run in the record carries: peak VRAM, peak process RAM (working set and private
bytes), the lowest free system RAM and CPU averages and peaks (sampled during the run by the same sampler `bench-report` uses, on llama-server's process
tree: `--monitor-pid`), the load time (`--load-s`, from the server start to its first ready answer), disk reads (the plugin's own `read_bytes` counter when
its counters are reachable, `/proc/<pid>/io` on Linux for a stock server; on Windows the raw `PhysicalDisk` read counter of the drive holding the model, taken before and after the run: it sees memory-mapped page-ins but also other programs on that drive, and `runs[].disk_read_source` says so; the server's own `ReadTransferCount` is kept next to it in `disk_read_process_gb`, but it does not count mapped page-ins: a 1 GiB map touched on the PC showed 0.000 GB there and 1.074 GB on the drive counter), and the run's own
settings: the llama-server command line (`--command-line TEXT` or `--command-line @FILE`, with an `env A=1 ` prefix for environment variables), so ctx, `-ub`,
`--n-cpu-moe`, flash attention and KV types are stored, and the server logs (`--server-log`, repeat it for stderr and stdout: the plugin prints to stdout) for the thread count and the layer count; without a log line the layer count is the model's own GGUF `block_count`, so the GPU expert layers are worked out the same
way as in other records. A KV type that is not on the command line is the llama.cpp default f16, and the record says so. `settings.fact_sources` names where each filled value came from. The plugin's counters add per run `gpu_blocks`, `gpu_up_saved_gb` (upload bytes a held GPU copy made unnecessary) and `ssd_wait_s` (time the compute side waited for an expert read still in flight); they are sums over the turns. The two arms can start llama-server with different layer plans, so settings are per run; the record's own settings block is taken from
the first run and a note says so. `runs[].vram_source` says whether VRAM is the server's own share or the whole card. A value that could not be read stays
"not recorded". The stock arm has no expert cache: its hit rate is written "n/a (no cache)", not left empty.

## Commands

Linux, a server that is already running (the plugin run needs `MOE_CACHE_STATS_SIGNAL=1` and its stderr in a file if you want hits and misses):

```bash
python3 tools/conversation_bench.py run --url http://127.0.0.1:8080 --arm moe-cache --ctx 65536 --out run1.json \
    --stats-pid $(pgrep -n llama-server) --stats-log server.err.log
```

Both servers up at once (a model small enough for two copies): stock, moe-cache, moe-cache, stock, then the record and the tables:

```bash
python3 tools/conversation_bench.py ab --stock-url http://127.0.0.1:8081 --plugin-url http://127.0.0.1:8082 --ctx 65536 --out-dir conv1
```

Run files from separate runs, in the order they were run, into one bench record (`docs/bench_record.schema.json`) and the tables:

```bash
python3 tools/conversation_bench.py assemble 1-stock.json 2-moe-cache.json 3-moe-cache.json 4-stock.json --out record.json \
    --model /models/model.gguf --llama-server /path/to/llama-server --command-line "llama-server -m ... -c 65536"
python3 tools/conversation_bench.py table record.json --turns
python3 bin/moe_cache.py bench-record ingest record.json --machine pc
```

**Windows PowerShell, the whole A B B A on a PC with one model** (the PC cannot hold two servers, so the driver starts and stops one per run, flushes the
file cache before each, waits for a quiet CPU, runs the conversation and assembles the record). The first line is a short trial, the second the real run with
the Flash-Next IQ3_XXS file:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\conversation_bench_pair.ps1 -OutDir C:\dl\conv-trial -Model D:\models\Qwen3.8-Flash-Next-IQ3_XXS.gguf -Alias flash -Ram 24 -FlushOther "D:\other-models" -Ctx 8192 -MaxTurns 4
powershell -NoProfile -ExecutionPolicy Bypass -File tools\conversation_bench_pair.ps1 -OutDir C:\dl\conv -Model D:\models\Qwen3.8-Flash-Next-IQ3_XXS.gguf -Alias flash -Ram 24 -FlushOther "D:\other-models" -Ctx 65536
```

`-FlushOther` names other model files (never the one under test) that together are bigger than the RAM; `-Order "stock,mc,mc,stock"` is the default.
Outputs in `-OutDir`: one `N-kind.json` per run (every turn), `record.json`, `tables.md`, the server logs and `progress.log`. The driver was written from
`tools\agent_bench_pair.ps1` and has been **checked only with a fake server and a Python syntax check, not on a PC**: run the trial first.

## What the tables mean, and what they cannot say

- The side-by-side band table gives the mean over the runs of each arm. Two runs per arm is the same small sample as the other paired runs; look at the
  per-run numbers in the record before quoting a ratio.
- Prompt tok/s in a band is the speed of reading the NEW tokens of the turns that start there. A turn that mostly reuses the cache reads few new tokens, so
  it is left out below 64 new tokens; the cache share is reported next to it.
- Decode speed in a band is the speed of the answer at that depth. Answers that end early (the model writes an end-of-text token) can fall under 16 tokens
  and then do not count; `--ignore-eos` makes every answer exactly `--max-tokens` long, and thinking is off by default (`chat_template_kwargs {"enable_thinking": false}`; `--think` leaves it to the server).
- The token estimate before a turn is characters divided by 3, which is cautious: the last turns can stop a little short of the limit.
- Texts are not compared between stock and moe-cache here (`identical_to_baseline` is "not checked"). The conversation is the same, the answers are the
  model's own, so a different answer in turn 5 changes the history after it; use the identity tests for text.
- Hits and misses come from the plugin's own counters (`MOE_CACHE_STATS_SIGNAL=1`: SIGUSR1 on Linux, the named event `Local\moe-cache-stats-<pid>` on
  Windows). If the tool cannot reach them the columns stay empty.
- A run on a machine that is busy is not a clean run: the record carries the conditions (`--baseline`, `--other-programs`, file cache state) like every other
  record, and `docs/BENCHMARK_TEMPLATE.md` rules apply.

## Test

`tests/conversation_bench_test.py` runs the tool against a fake llama-server (no model): history growth, the stop at the limit, timing parsing, the band sums,
A B B A order, the plugin counters per turn and the record format. It is in `tools/run_quick_tests.sh`.

## Visible answers (changed 2026-11-07)

The first version asked for 128 tokens with the model's thinking on. The thinking block used the answer length, so many answers were empty, and from about turn 11 the model copied its own earlier one- or two-token replies: both arms (f16 and q8_0 KV, stock and moe-cache) answered with 1-2 tokens deep in the conversation. Prompt reading was not affected (the user turns and their size are the same), but the decode speed of the deeper bands was measured over a handful of tokens and the conversation could not be used to judge meaning. The driver now defaults to thinking off, 256 tokens per answer and the answer hint on every user message; `--legacy` (`-Legacy` in `conversation_bench_pair.ps1`) restores the old settings so a run stays comparable with the v0.3.7 records, `--think`, `--no-answer-hint` and `--max-tokens` change one setting each. Every run carries `thinking`, `answer_hint` and `answer_stats` (median answer tokens, short and empty turns); when the median is under `--min-median-answer` (24 tokens) or more than a quarter of the turns have no visible answer, the log and the run say WARNING. A record made with the new defaults says "thinking off, answer hint on" in its sampler line.

Records made before this change (n_predict 128, thinking on): the v0.3.7 conversation records of 2026-11-05 and the night-chain conversations of 2026-11-06/07 (`v037-conv-q2_0`, `v037-conv-iq3xxs`, TEST-684, TEST-754). Their prompt numbers stand; their decode numbers per depth band rest on short answers from turn 11 on and are not to be quoted as "decode at 60k" without that caveat. Re-measure with the new defaults before a release quotes decode by band.

### Finding the cause of short answers

Every turn of a run now keeps `answer_head` (the first 400 characters of the visible answer), `finish_reason` and `reasoning_chars`, and `answer_stats` also says where the first short answer came in (`first_short_turn`), how likely a short answer is after a short one and after a long one (`p_short_after_short`, `p_short_after_long`: a large gap means the model is copying its own history), the median answer by kind of user turn and the finish reasons. Switches that change one suspect at a time: `--system neutral` (the system message without "Answer briefly."; any other text is used as given), `--hint TEXT` (another sentence instead of the default one), `--temperature F` (default 0, greedy), `--empty-marker TEXT` (what goes into the history for an answer with no visible text; the default is "(no visible answer)", '' stores an empty answer), `--no-keep-answers` (do not save the answer text). The day-chain item `70_conv_cause` runs five of these combinations on one model (`plans/v0.3.8/PLAN.md`, item 9b).
