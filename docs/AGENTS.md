# Coding agents (Hermes Agent, OpenCode) on moe-cache

Both agents talk to any OpenAI-compatible server, and llama-server is one. `--agent` sets the launcher up for them, and `connect` writes their config.

```bash
bin/moe-cache-server MODEL.gguf --agent            # context 65536, tool calls, stable model name, warm sessions
python3 bin/moe_cache.py connect opencode --write  # or: connect hermes --write   (add --url http://127.0.0.1:PORT for another port)
```

**Claude Code** (it speaks Anthropic's API; llama-server answers `/v1/messages` and `/v1/messages/count_tokens`, checked on llama.cpp b11966 on the Windows PC with a small model, 2026-10-29: 200 with `x-api-key` and with a Bearer token): `python3 bin/moe_cache.py connect claude` prints the `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL` lines for bash and PowerShell and writes nothing; `connect claude --write` merges them into the `env` of **this folder's** `.claude/settings.local.json` only (first backup kept as `.bak`, file private), so Claude Code started elsewhere is not redirected. Claude Code's own system prompt is long: use `--agent` (64k) or more. Untested: how well a given local model follows Claude Code's tool use.

| What `--agent` does | Why |
|---|---|
| a log line about the prompt cache | `moe-cache: prompt cache: up to 2048 MiB of host RAM keeps finished conversations for reuse (--cache-ram); ... slots, 8 checkpoints per slot`. Two agents on one server reuse a shared prompt only while this cache holds it; below 16 GB of server memory it is off (the RAM is worth more to the expert cache, RESULTS 44). The control page lists the line under "What the launcher decided". Change it with `-- --cache-ram MiB` |
| context 65536 (unless `--ctx`) | agents send 5-20k tokens of tools and instructions before the conversation starts; Hermes Agent refuses less than 64,000 |
| `--jinja` | tool calls use the model's own chat template (already the default in current llama.cpp) |
| `--alias NAME` (the file name) | the name agents see in `/v1/models`; pass `-- --alias other` to choose it |
| at most 8 context checkpoints per slot, 1 GiB kept free for them and for restored conversations | models with recurrent layers (Qwen3-Next, Qwen3.5/3.6) keep a copy of that state per checkpoint in RAM (about 60 MiB each on Qwen3.6-35B), outside the expert cache |
| warm sessions (`--no-warm` turns them off) | every idle conversation of 2k+ tokens is saved to disk (with its checkpoints) once it has been idle for 20 s + 20 s per GB of the file; a conversation that grew by less than a quarter (and less than 4,096 tokens) since its last save waits 3 min + 30 s per GB (at most 10 min), because a save rewrites the whole file and llama-server answers nothing while it writes; one file per slot; the two longest are restored at the next start, so each agent finds its own |
| prints the base URL and both config blocks once the server is ready | |
| an API key (made for you) | `--agent` always sets one: a stable random key kept in `~/.cache/moe-cache/api.key` (mode 0600; Windows: your profile), created on first use and given to llama-server through the environment, never on its command line. Why: agent mode saves conversations to disk and llama-server answers any web page's request on 127.0.0.1 by default, so a page in your browser could erase or read them. `python3 bin/moe_cache.py key` prints it; `connect hermes\|opencode --write` finds it by itself (it uses the stored key when the server answers 401) and writes it into the agent's config. Your own `--api-key KEY` (or `LLAMA_API_KEY`) replaces it |
| (Windows + AMD GPU with a driver older than 32.0.32015.7029 + Flash-Next only) `--ctx-checkpoints 0` instead of 8 | saving a checkpoint crashes AMD's older Windows Vulkan driver (RESULTS 54; 32.0.32015.7029 does not); the cost is that edited history is processed again instead of resuming from a checkpoint; Qwen3.6 keeps its checkpoints. Not yet measured with an agent on Windows |

## Which model

| Use | Recommended file | Measured (16 GB GPU + 32 GB RAM PC, plugin; RESULTS 60, 63) |
|---|---|---|
| general agent use (Hermes, OpenCode) | Qwen3.8-Flash-Next GSQ-RCO **IQ3_XXS** (75.8 GB), the default since 0.3.2 | decode 19.9 tok/s, 193/202 tasks, reads best for coding (RESULTS 63) |
| the faster file | Qwen3.8-Flash-Next GSQ-RCO **Q2_0** (66.4 GB) | decode 22.2 tok/s (about 10 % more than IQ3_XXS), prompts 281 tok/s at 3k and 287 at 34k tokens (vs IQ2_XS: +5 % decode, +11 % / +19 % prompts), 191/202 tasks |
| coding agents | ISTA GSQ-RCO **Coder IQ1_M** (58.4 GB, 256 of 512 experts) | OpenCode session 55.5 s vs 73.4 s on IQ2_XS (one run), 189/202 tasks; general knowledge not checked beyond the task set |

No quality difference between these files is proven on the 202-task set (all p > 0.05). The agent-session numbers further down that say
IQ2_XS were measured on IQ2_XS (superseded by Q2_0). If a smaller model is enough: Qwen3.6-35B (20 GB, fits in 32 GB RAM) decodes at about 72 tok/s with MTP in agent mode on the same PC (RESULTS 60).

## Without starting a server by hand (on-demand)

Hermes Agent and OpenCode only know a URL; they cannot start a server. `on-demand` gives them an endpoint that is always there and starts the model
when an agent asks for it, the way Ollama does:

```bash
python3 bin/moe_cache.py on-demand MODEL.gguf --ram 20      # remembers the model and its options, then listens on 127.0.0.1:8080
python3 bin/moe_cache.py connect opencode --write            # and/or: connect hermes --write  (the model does not start for this)
python3 bin/moe_cache.py on-demand --install-autostart        # optional: start the listener at every login
```

| What happens | Details |
|---|---|
| the listener | a small standard-library process (about 35 MB of RAM) on the agent port (`--port`, default 8080). It checks the API key (the stable key of `moe_cache.py key`) and allows cross-origin requests only from local pages, as agent mode does. `/health`, `/v1/models` and `/props` are answered without starting the model, so agents and `connect` can look at the endpoint for free |
| first request | starts `moe_cache.py server MODEL --agent` with the remembered options on a private port, waits until it is ready (the request waits too; up to `--start-timeout`, 900 s) and passes everything through, streaming (SSE) included |
| idle release (N3) | after `--idle-release` minutes (default 5, `0` = off) without a request the plugin gives the expert cache's RAM back and the server keeps running: on Windows the experts' file pages move to the standby list (PC, Flash-Next IQ2_XS: available RAM 6.8 -> 25 GB, the next request 4.4 s instead of 3.9 s, against 34 s for a restart); on Linux the experts are evicted and read again when used. The next request sets everything up again by itself |
| idle stop | after `--idle` minutes (default 15) without a request the server is stopped and its RAM is free; a long streamed answer counts as busy. The next request starts it again (warm sessions bring the conversations back) |
| the choice | `on-demand MODEL [launcher options] [-- llama-server args]` checks the options with a dry run and keeps them in `~/.cache/moe-cache/on-demand.json` (also `LLAMA_SERVER` and `MOE_CACHE_LIB` from your shell, since autostart has none); `on-demand` alone uses it. Any `server` option works except `--port`/`--agent` (on-demand sets them) |
| log | one line per start and stop on the listener's output (journal on Linux); the launcher's and llama-server's output go to `~/.cache/moe-cache/on-demand.log` |
| `--status`, `--stop` | what is remembered, whether the listener and the model run; stop the listener (and the model) |
| a crash | a model server that dies is noticed; the next request starts it again |
| two models (Ollama-style) | `on-demand MODEL2.gguf [options] --add` adds another model; the first stays the default. `/v1/models` lists all of them, and the request's `"model"` picks one: one model runs at a time, so a request for the other waits until the running one has finished its requests, stops it and starts the one asked for (Granite on the laptop: about 2 s; a big model takes its load time, its pages are often still in the file cache). A name it does not know goes to the running model, else the default (llama-server ignores the field too). `--remove NAME` forgets an added model; `on-demand MODEL` alone replaces the whole choice. `connect opencode --write` puts every listed model into OpenCode's model picker |

**Start at login** (`--install-autostart`, undo with `--remove-autostart`, which also stops the listener):

| | What it installs | Notes |
|---|---|---|
| Linux | a systemd user unit `~/.config/systemd/user/moe-cache-on-demand.service`, enabled and started (`systemctl --user enable --now`) | log: `journalctl --user -u moe-cache-on-demand`; no root. It runs while you are logged in |
| Windows | a shortcut `moe-cache on-demand.lnk` in your Startup folder (`shell:startup`) that runs `pythonw.exe bin\moe_cache.py on-demand`, started right away too | no admin rights, no console window; the model server also runs without a window |

Run `on-demand MODEL ...` (or `--port` / `--idle` / `--idle-release`) again to change the choice; restart the listener to use it (`--stop`, then
`systemctl --user start moe-cache-on-demand` or log in again; without autostart, run `on-demand`).

**Pointing the agents at it.** `connect hermes|opencode --write` uses the on-demand port when on-demand has a model (no `--url` needed). OpenCode CLI and
OpenCode Desktop read the same `opencode.json`. In **Hermes Desktop** you can also set it by hand: Settings -> Model -> Main model, provider **custom**
(OpenAI-compatible), URL `http://127.0.0.1:8080/v1`, API key from `python3 bin/moe_cache.py key`, model name as `connect hermes` prints it (the file name
without `.gguf`), context 65536. The Hermes CLI and Desktop share one config, so `connect hermes --write` covers both.

The control page's Start button starts a server on its own port; do not give it the on-demand port while the listener runs.

**Checked on the Linux laptop (2026-10-27)** with the real Hermes Agent 0.22.5 and OpenCode 1.19.4 (isolated config folders, API key on) against the listener: first request starts the model (Granite 1B: 2 s; Qwen3.6-35B under a 12 GB memory limit: 14-17 s), SSE streaming passes through, OpenCode ran `bash` and `read` tool calls on Qwen3.6, idle stop after the set minutes, the next request (Hermes) started it again, no key / wrong key / raw key without `Bearer` give 401, `x-api-key` and `Authorization: Bearer KEY` work, another site's origin gets no CORS header. One bug found and fixed: Hermes asks `/api/v1/models`, `/api/tags`, `/v1/props` and `/version` once to tell LM Studio, Ollama and llama.cpp apart; these started the model on a probe. The idle listener now answers them with llama-server's own 404. Not tried: the Desktop apps (no Windows here).

## PC checklist: Hermes Desktop and OpenCode Desktop with on-demand (Windows)

Use PowerShell. `LS` is your llama-server folder, `MODEL` your .gguf. Each step says what you should see; if not, stop and note it.

| # | Do | You should see |
|---|---|---|
| 1 | `python bin\moe_cache.py on-demand MODEL --llama-server LS\llama-server.exe --ram 20` | "remembered ... " and a line with port 8080 and idle stop 15 min. If it says "refused these options", fix the path it names |
| 2 | `python bin\moe_cache.py on-demand --install-autostart` | "started it now too"; `on-demand --status` says listener running, model idle. Task Manager: no llama-server |
| 3 | `python bin\moe_cache.py key` | your API key (keep this window for steps 5 and 8) |
| 4 | Add `%LOCALAPPDATA%\hermes\bin` to PATH if `hermes` is not found, then `python bin\moe_cache.py connect hermes --write` and `python bin\moe_cache.py connect opencode --write` | two lines "now uses ... at http://127.0.0.1:8080/v1"; llama-server still not running |
| 5 | **Hermes Desktop**: open it, Settings -> Model -> Main model. Check provider `custom`, URL `http://127.0.0.1:8080/v1`, model = the file name without `.gguf`, context 65536. If empty, type them in and paste the key from step 3 | the values from `connect` are shown. Opening Settings must not start llama-server (note it if it does) |
| 6 | Hermes Desktop: new chat, send "Use your terminal to list the files in this folder." | the answer arrives after the load time (first request waits); llama-server appears in Task Manager without a console window; the reply streams |
| 7 | Send a second message | answers quickly (the conversation is reused, only new tokens are processed) |
| 8 | **OpenCode Desktop**: open it, open a project folder, check the model picker lists `moe-cache/NAME`. If not: Settings -> Providers -> add OpenAI-compatible, base URL `http://127.0.0.1:8080/v1`, key from step 3, model NAME. Note which file got the config: `%USERPROFILE%\.config\opencode\opencode.jsonc` if it exists, else `opencode.json` | `moe-cache/NAME` is selected |
| 9 | OpenCode Desktop: send "Run ls and read a file, tell me what it says." | a bash and a read tool call run, then the answer |
| 10 | Close both apps, wait the idle time (15 min; use `--idle 2` in step 1 for a quick test) | llama-server gone from Task Manager, RAM back; `%USERPROFILE%\.cache\moe-cache\on-demand-listener.log` has a "stopped" line |
| 11 | Open one app again, send a message | the model starts again, the old conversation is picked up (warm session) |
| 12 | Sign out and in, send a message | the listener is back at login (Startup shortcut), the model starts again |
| 13 | `python bin\moe_cache.py on-demand --remove-autostart` | shortcut gone, listener and model stopped |

If a step fails, send: `python bin\moe_cache.py on-demand --status`, the last 30 lines of `on-demand-listener.log` and `on-demand.log` (both in `%USERPROFILE%\.cache\moe-cache\`), and the text of the app's error.

## T8: Hermes Desktop and OpenCode Desktop through the control page (Windows, owner click-through)

Both desktop apps are already installed on the test PC (Hermes Desktop; OpenCode Desktop 2.1.4 in `%LOCALAPPDATA%\Programs\@opencodedesktop`). This path needs no on-demand listener: the control page starts the server and writes both configs. Use a small model first (Qwen3.6, 20 GB). Each step says what you should see; if not, stop and note it.

| # | Do | You should see |
|---|---|---|
| 1 | Double-click `bin\moe-cache-ui.bat` (the console prints `plugin: ...dll`, `llama-server: ...`, and the page opens) | the control page; the header shows RAM and the GPU |
| 2 | Pick the model, turn **Agent mode** on, context 65536, press **Start** | status goes to ready; Why list has no warning in red; the key line says "the launcher's agent key" |
| 3 | Agents tab: **Connect OpenCode**, then **Connect Hermes** | each says "written": OpenCode with the file path (`%USERPROFILE%\.config\opencode\opencode.json`, or `opencode.jsonc` if OpenCode Desktop made one), Hermes "now uses NAME at http://127.0.0.1:PORT/v1 (context 65536)" and a line `before: provider ..., model ...; old config kept as ...config.yaml.bak`. Your own Hermes setup (OpenRouter etc.) is in `%LOCALAPPDATA%\hermes\config.yaml.bak`: copy it back over `config.yaml` to undo |
| 4 | **Hermes Desktop**: open it, Settings -> Model -> Main model | provider `custom`, URL `http://127.0.0.1:PORT/v1`, model = the file name without `.gguf`, context 65536. If it asks at first run, choose "Local / custom endpoint" and type those values (key: **Copy key** on the page) |
| 5 | Hermes Desktop: new chat: "Use your terminal to list the files in this folder." | the answer streams, a terminal tool call runs; the page's Status tab shows prompt tokens and decode tok/s; first turn is the slow one |
| 6 | Hermes Desktop: a second message | much quicker: only the new tokens are processed |
| 7 | **OpenCode Desktop**: open it, open a project folder, open the model picker | `moe-cache/NAME` is listed and selected. If it is not: Settings -> Providers -> add OpenAI-compatible, base URL `http://127.0.0.1:PORT/v1`, key from **Copy key**, model NAME |
| 8 | OpenCode Desktop: "Run ls and read a file, tell me what it says." | a bash and a read tool call, then the answer |
| 9 | Control page, Agents tab | the two conversations appear as saved sessions after they have been idle for a while (warm sessions) |
| 10 | Press **Stop** on the page, then **Start** again, send one more message in either app | the old conversation is picked up (restored at start) |
| 11 | Close the apps; the page's **Stop** ends the server | llama-server is gone from Task Manager |

Harmless: after each Hermes chat the Log tab shows one `unauthorized: Invalid API Key` (HTTP 401) line, because Hermes asks `GET /v1/models` once without the key (seen on the Linux laptop; Hermes' `model_metadata_llamacpp.py`). The chat itself works.

If a step fails, send the page's Log tab (or `ui.log` of the console) and the text of the app's error.

## Using the agents with a GUI

Both agents come with a graphical front end. The steps are the same everywhere: start the server in agent mode, write the agent's config, open the GUI.

```bash
bin/moe-cache-server MODEL.gguf --agent                # Windows: python bin\moe_cache.py server MODEL.gguf --agent --llama-server C:\llama\llama-server.exe
python3 bin/moe_cache.py connect hermes --write        # and/or: connect opencode --write   (Windows: python bin\moe_cache.py connect ...)
```

| | Linux | Windows (signed desktop apps) |
|---|---|---|
| Hermes Agent | `hermes dashboard`: web UI at http://127.0.0.1:9119 (`--port`, `--host`, `--no-open`, `--stop`, `--status`). `hermes desktop` builds and starts the native (Electron) app | Hermes Desktop: `Hermes-Setup.exe` from hermes-agent.nousresearch.com, a Tauri setup app that has to be run by hand. Installs to `%LOCALAPPDATA%\hermes`, the CLI is `%LOCALAPPDATA%\hermes\bin\hermes.exe` |
| OpenCode | `opencode web`: starts the server and opens the browser UI (`--port`, `--hostname`, default 127.0.0.1) | OpenCode Desktop: opencode.ai/download, the `windows-x64` NSIS installer, per-user install in `%LOCALAPPDATA%\Programs\opencodedesktop`; needs Edge WebView2 |

The Linux commands were checked against `hermes --help` and `opencode --help` (2026-10-26); they were not started against a moe-cache server for this page.

**Checked on the Linux laptop with a running agent-mode server (2026-10-29, Hermes Agent 0.22.5, OpenCode 1.19.4, Qwen3.6-35B-A3B, 64k):** `connect hermes --write` and `connect opencode --write` wrote correct configs (OpenCode: into the existing `opencode.jsonc`, old file kept as `.bak`). Both CLIs fixed a bug in a small git repo over two turns (run tests, edit, commit): `hermes chat -Q -t terminal,file,todo -q "..."` (then `-c` for the next turn) and `opencode run --auto "..." < /dev/null` (then `-c`). Flash-Next IQ3_XXS at 64k also fixed it with Hermes (373 s, cold first turn 5.8k tokens in 167 s). Hermes sends one `GET /v1/models` without the API key after each chat; the server logs "unauthorized: Invalid API Key" for it. That comes from Hermes and is harmless. The web UIs (`hermes dashboard`, `opencode web`) were not started.

**How each is pointed at moe-cache** (read from `connect()` in `bin/moe_cache.py`):

| Agent | What `connect ... --write` does | Windows | Do the GUIs read it? |
|---|---|---|---|
| Hermes | runs `hermes config set` for `model.provider=custom`, `model.base_url`, `model.default`, `model.context_length`, `model.api_key`. It looks for `hermes` on `PATH` and fails if it is not there. Before the first change it copies `config.yaml` to `config.yaml.bak` and prints the model you had (a config that already points at a local server is not copied over the backup) | The Hermes Desktop installer puts the CLI in `%LOCALAPPDATA%\hermes\bin`: add that directory to `PATH` (or run the command from it) so `connect` finds it. `hermes config set` writes the dotted `model.*` keys (the key too) to `%LOCALAPPDATA%\hermes\config.yaml`: on Windows Hermes' home is `%LOCALAPPDATA%\hermes` (`HERMES_HOME` overrides it), not `%USERPROFILE%\.hermes` (hermes_constants.py; read 2026-10-27) | Yes (source and docs, 2026-10-27; not yet tried on the PC): Hermes Desktop (apps/desktop, Electron) runs the same Python backend on the same `HERMES_HOME`, so it reads that `config.yaml`. Its model picker in the chat box is remembered per device and does not change the default; the default is Settings → Model → Main model, which is the same `config.yaml` entry |
| OpenCode | merges a `moe-cache` provider into `opencode.json` and sets `model` to `moe-cache/NAME`; the old file is kept as `.bak`. Default file: `$XDG_CONFIG_HOME/opencode/opencode.json`, else `~/.config/opencode/opencode.json`; `--config PATH` writes another one | `%USERPROFILE%\.config\opencode\opencode.json` (the launcher does not use `%APPDATA%`). OpenCode Desktop reads it too: its server runs inside the app on the same global folder (it only moves its state folder). OpenCode merges `config.json`, `opencode.json`, then `opencode.jsonc` (later wins) and creates `opencode.jsonc` on first run, so `connect` writes into `opencode.jsonc` when that is plain JSON, and warns when a `.jsonc` with comments sets its own `model`. `OPENCODE_CONFIG` (a file merged after these) is reported too | Yes for `opencode web` and the Desktop app (same server code, same global folder; source read 2026-10-27, not yet tried on the PC) |

If a GUI does not show the model, add the provider by hand in its settings: type OpenAI-compatible, base URL `http://127.0.0.1:8080/v1` (your `--port`), model name = the file name, context 65536, API key `none` (or your `--api-key`). `connect` without `--write` prints these values. Keep the server on `127.0.0.1`. In agent mode the key is in the config that `connect --write` wrote (or print it with `python3 bin/moe_cache.py key`); llama-server's own web page asks for it once.

## Keep the agent's own context limit below the server's

`connect --write` (and the control page's Connect buttons) write the server's context as the agent's window: 65536 by default, or your `--ctx`. When the agent believes it has the whole window, it keeps adding tool results until llama-server refuses the request ("context too large"), and the reply itself also has to fit. Leave 10-25 % of headroom, so the agent's own summarising or compaction starts first:

| Agent | What to do | Example for the default 65536 |
|---|---|---|
| OpenCode | lower `limit.context` (and set `limit.output`) of the `moe-cache` model in `opencode.json` | `"limit": {"context": 56000, "output": 8192}` |
| Hermes | it needs `model.context_length` of at least 64,000, so keep 65536 there and give the server more room | start the server with `--ctx 81920` (Flash-Next KV is about 27.5 KB per token: about 0.4 GiB more than at 65536), run `connect hermes --write` (it copies the server's 81920), then `hermes config set model.context_length 65536` |

A bigger server window costs KV memory and some speed at depth. Not measured by us: the advice follows from how the limits work and from Strata's agent chat, which reports the peak prompt in a 64K-window test falling from 83 % to 61 % when it summarises early (their number).

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

### Keep the reuse working

The reuse above only works while the start of the prompt stays byte for byte the same. What breaks it, and what to do:

| Breaks the reuse | Do instead |
|---|---|
| adding or removing a tool or an MCP server in the middle of a session (the tool list sits at the start of the prompt) | set up tools before the session; a change means one slow turn while the whole prompt is read again |
| more conversations than slots (agent + title request + sub-agents + a second agent) | `--agent` gives 4 slots; with two agents and sub-agents raise them (`-- -np 6`) or run one agent at a time |
| an editor's autocomplete on the same server (many tiny requests) | give autocomplete its own small model or server |
| the agent's window larger than the server's | keep the agent's limit below the server's (section above); `connect opencode` writes it, and for windows of 64k+ a 30-minute reply timeout (OpenCode's default is 5 minutes, a 63k first turn took 289 s) |

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
else (a few hundred tokens per turn). Recording replaces the home folder, user and host names. Agent-mode servers want the per-user API key: the replay sends `LLAMA_API_KEY`, or the launcher's stored key when the server asks for one (`--api-key` to give another). Start the server fresh for a cold first turn (with
`--agent`, also give it an empty `--slot-save-path` after `--`, or it restores the last warm session). Measured on the laptop, stock vs moe-cache:
[RESULTS.md section 57](RESULTS.md).

## Agent notes

| Agent | Notes |
|---|---|
| OpenCode | `opencode run "..."` waits on an open stdin: give it `< /dev/null` in scripts. Config: `~/.config/opencode/opencode.json` or `opencode.json` in the project; `connect opencode --write --config PATH` writes either (the old file is kept as `.bak`) |
| Hermes Agent | config `~/.hermes/config.yaml` (Windows: `%LOCALAPPDATA%\hermes\config.yaml`), model block `provider: custom`, `base_url`, `default`, `context_length`, `api_key`; `connect hermes --write` sets them through `hermes config set` |
