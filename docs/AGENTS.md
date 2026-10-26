# Coding agents (Hermes Agent, OpenCode) on moe-cache

Both agents talk to any OpenAI-compatible server, and llama-server is one. `--agent` sets the launcher up for them, and `connect` writes their config.

```bash
bin/moe-cache-server MODEL.gguf --agent            # context 65536, tool calls, stable model name, warm sessions
python3 bin/moe_cache.py connect opencode --write  # or: connect hermes --write   (add --url http://127.0.0.1:PORT for another port)
```

| What `--agent` does | Why |
|---|---|
| context 65536 (unless `--ctx`) | agents send 5-20k tokens of tools and instructions before the conversation starts; Hermes Agent refuses less than 64,000 |
| `--jinja` | tool calls use the model's own chat template (already the default in current llama.cpp) |
| `--alias NAME` (the file name) | the name agents see in `/v1/models`; pass `-- --alias other` to choose it |
| at most 8 context checkpoints per slot, 1 GiB kept free for them and for restored conversations | models with recurrent layers (Qwen3-Next, Qwen3.5/3.6) keep a copy of that state per checkpoint in RAM (about 60 MiB each on Qwen3.6-35B), outside the expert cache |
| warm sessions (`--no-warm` turns them off) | every idle conversation of 2k+ tokens is saved to disk (with its checkpoints) 20 s after it changes, one file per slot; the two longest are restored at the next start, so each agent finds its own |
| prints the base URL and both config blocks once the server is ready | |
| `--api-key KEY` | required when other machines can reach the server |

## What makes agents fast or slow

An agent's first turn is the expensive one. llama-server keeps each conversation in its own slot (4 slots sharing the context), so later turns only
process the new tokens. It also keeps a checkpoint at the start of the last user message, which is where the agent's fixed preamble ends, so a *new*
session on the same server reuses the preamble too. Measured on the laptop (RTX 4060 8 GB, Qwen3.6-35B-A3B Q4_K_M under a 16 GB memory limit, plugin
with a 9 GiB cache, OpenCode 1.19.4, task "fix the bug in calc.py"):

| Request | New prompt tokens | Prompt time |
|---|---|---|
| session 1, turn 1 (OpenCode's tools and instructions) | 7,438 | 45.6 s (163 tok/s) |
| session 1, turns 2-3 | 79 / 20 | 3.4 s / 0.8 s |
| session 2 (same server), turn 1 | 29 | 0.7 s |
| after a server restart, warm session restored (0.2 s), turn 1 | 1,028 | 8.5 s |
| Hermes Agent (file tools only), turn 1 / turn 2 | 3,687 / 76 | 23.0 s / 2.9 s |
| whole session: OpenCode 1 / OpenCode 2 / after restart / Hermes | | 78 s / 30 s / 39 s / 55 s |

With Flash-Next (64 GB, 23 GB limit) the cold first turn is 7,461 tokens in 186 s; after a restart with the warm session the whole OpenCode
session takes 112 s instead of 242 s. Details: [RESULTS.md sections 51-52](RESULTS.md). Saved conversations live in `~/.cache/moe-cache/slots/agent/`, one file per
model, context and slot.

OpenCode's title request runs in parallel in its own slot and does not disturb the main conversation.

## Measuring it yourself: replay a recorded agent session (tools/agent_bench.py)

Decode tok/s says little about agents; what they wait for is the time to the first token of each turn and the whole task. `tools/agent_bench.py`
(Python standard library only, Linux and Windows) records an agent session through a small proxy and replays it against any server, with the same
request bodies, the same overlap of parallel requests and the same pauses between turns. Two recordings come with it (`tests/agent_sessions/`: OpenCode
and Hermes Agent fixing a bug in a small two-file project, 5 and 7 requests, first prompts of 7,455 and 3,717 tokens).

```bash
python3 tools/agent_bench.py replay tests/agent_sessions/opencode_stats.json --url http://127.0.0.1:8080 --json oc.json   # per turn: TTFT, prompt tokens processed / reused, decode tok/s
python3 tools/agent_bench.py table oc.json other.json                                                                       # results side by side
python3 tools/agent_bench.py record --upstream http://127.0.0.1:8080 --port 8199 --out mine.json   # then point the agent at http://127.0.0.1:8199/v1; Ctrl+C ends
```

The replay generates greedily and stops at the recorded number of tokens, so stock llama-server and moe-cache (exact) produce the same tokens and the
same prompt reuse. Follow-up turns re-send the agent's original assistant message, which the server processes again if the model now writes something
else (a few hundred tokens per turn). Recording replaces the home folder, user and host names. Start the server fresh for a cold first turn (with
`--agent`, also give it an empty `--slot-save-path` after `--`, or it restores the last warm session). Measured on the laptop, stock vs moe-cache:
[RESULTS.md section 54](RESULTS.md).

## Agent notes

| Agent | Notes |
|---|---|
| OpenCode | `opencode run "..."` waits on an open stdin: give it `< /dev/null` in scripts. Config: `~/.config/opencode/opencode.json` or `opencode.json` in the project; `connect opencode --write --config PATH` writes either (the old file is kept as `.bak`) |
| Hermes Agent | config `~/.hermes/config.yaml`, model block `provider: custom`, `base_url`, `default`, `context_length`, `api_key`; `connect hermes --write` sets them through `hermes config set` |
