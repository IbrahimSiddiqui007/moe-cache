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
| at most 8 context checkpoints per slot, 512 MiB kept free for them | models with recurrent layers (Qwen3-Next, Qwen3.5/3.6) keep a copy of that state per checkpoint in RAM (about 60 MiB each on Qwen3.6-35B), outside the expert cache |
| warm sessions (`--no-warm` turns them off) | the longest idle conversation is saved to disk (with its checkpoints) 20 s after it changes and restored into slot 0 at the next start |
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
| whole session 1 / session 2 | | 78 s / 30 s |

OpenCode's title request runs in parallel in its own slot and does not disturb the main conversation.

## Agent notes

| Agent | Notes |
|---|---|
| OpenCode | `opencode run "..."` waits on an open stdin: give it `< /dev/null` in scripts. Config: `~/.config/opencode/opencode.json` or `opencode.json` in the project; `connect opencode --write --config PATH` writes either (the old file is kept as `.bak`) |
| Hermes Agent | config `~/.hermes/config.yaml`, model block `provider: custom`, `base_url`, `default`, `context_length`, `api_key`; `connect hermes --write` sets them through `hermes config set` |
