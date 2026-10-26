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

## Using the agents with a GUI

Both agents come with a graphical front end. The steps are the same everywhere: start the server in agent mode, write the agent's config, open the GUI.

```bash
bin/moe-cache-server MODEL.gguf --agent                # Windows: python bin\moe_cache.py server MODEL.gguf --agent --llama-server C:\llama\llama-server.exe
bin/moe-cache-server connect hermes --write            # and/or: connect opencode --write   (Windows: python bin\moe_cache.py connect ...)
```

| | Linux | Windows (signed desktop apps) |
|---|---|---|
| Hermes Agent | `hermes dashboard`: web UI at http://127.0.0.1:9119 (`--port`, `--host`, `--no-open`, `--stop`, `--status`). `hermes desktop` builds and starts the native (Electron) app | Hermes Desktop: `Hermes-Setup.exe` from hermes-agent.nousresearch.com, a Tauri setup app that has to be run by hand. Installs to `%LOCALAPPDATA%\hermes`, the CLI is `%LOCALAPPDATA%\hermes\bin\hermes.exe` |
| OpenCode | `opencode web`: starts the server and opens the browser UI (`--port`, `--hostname`, default 127.0.0.1) | OpenCode Desktop: opencode.ai/download, the `windows-x64` NSIS installer, per-user install in `%LOCALAPPDATA%\Programs\opencodedesktop`; needs Edge WebView2 |

The Linux commands were checked against `hermes --help` and `opencode --help` (2026-10-26); they were not started against a moe-cache server for this page.

**How each is pointed at moe-cache** (read from `connect()` in `bin/moe_cache.py`):

| Agent | What `connect ... --write` does | Windows | Do the GUIs read it? |
|---|---|---|---|
| Hermes | runs `hermes config set` for `model.provider=custom`, `model.base_url`, `model.default`, `model.context_length`, `model.api_key`. It looks for `hermes` on `PATH` and fails if it is not there | The Hermes Desktop installer puts the CLI in `%LOCALAPPDATA%\hermes\bin`: add that directory to `PATH` (or run the command from it) so `connect` finds it. Which file `hermes config set` writes there: **not yet verified** | The dashboard and `hermes desktop` are front ends of the same Hermes install, so they should use the same config. **Not yet verified**, nor that Hermes Desktop uses the same one |
| OpenCode | merges a `moe-cache` provider into `opencode.json` and sets `model` to `moe-cache/NAME`; the old file is kept as `.bak`. Default file: `$XDG_CONFIG_HOME/opencode/opencode.json`, else `~/.config/opencode/opencode.json`; `--config PATH` writes another one | `%USERPROFILE%\.config\opencode\opencode.json` (the launcher does not use `%APPDATA%`). Whether OpenCode Desktop reads that file: **not yet verified**; if it does not, pass `--config` with the file it uses, or paste the block that `connect opencode` (without `--write`) prints into the app's provider settings | `opencode web` reads the same global config as the terminal UI (**not yet verified** here). The Desktop app: see the Windows column |

If a GUI does not show the model, add the provider by hand in its settings: type OpenAI-compatible, base URL `http://127.0.0.1:8080/v1` (your `--port`), model name = the file name, context 65536, API key `none` (or your `--api-key`). `connect` without `--write` prints these values. Keep the server on `127.0.0.1` unless you set `--api-key`.

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

## Agent notes

| Agent | Notes |
|---|---|
| OpenCode | `opencode run "..."` waits on an open stdin: give it `< /dev/null` in scripts. Config: `~/.config/opencode/opencode.json` or `opencode.json` in the project; `connect opencode --write --config PATH` writes either (the old file is kept as `.bak`) |
| Hermes Agent | config `~/.hermes/config.yaml`, model block `provider: custom`, `base_url`, `default`, `context_length`, `api_key`; `connect hermes --write` sets them through `hermes config set` |
