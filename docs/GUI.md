# Control page (`moe-cache-server --ui`)

A local web page that lists your models, says which mode each one runs in and why, starts and stops the server, and shows live numbers.
It is served by the launcher itself: Python's standard library only, no installs, the same on Linux and Windows.

```bash
bin/moe-cache-server --ui
```

On Windows: double-click `bin\moe-cache-ui.bat` (it finds Python, the plugin DLL and llama-server; `--shortcut` adds it to the Start menu; see
[WINDOWS](WINDOWS.md)), or `python bin\moe_cache.py ui`. The page opens at **http://127.0.0.1:8070**.

| Option | Meaning |
|---|---|
| `--ui-port N` | another port for the page (default 8070) |
| `--llama-server PATH` | the llama-server to start (default: `$LLAMA_SERVER`, else `llama-server` on the PATH) |
| `--no-browser` | do not open the browser |

Ctrl-C stops the page and any server it started.

**Safety.** The page can start programs, so it answers only its own user and its own page:

| Protection | Against |
|---|---|
| listens on 127.0.0.1 only | other machines |
| a random token per run: the address the launcher prints and opens ends in `?t=…`; the page moves it into an HttpOnly, SameSite=Strict cookie and every `/api` call needs it (scripts: header `X-MoeCache-Token`) | other users of a shared machine, who can reach 127.0.0.1 too |
| `Host` must be 127.0.0.1 / localhost, a foreign `Origin` and `Sec-Fetch-Site` other than same-origin are refused, posts must be JSON (at most 64 KiB) | web sites in your browser (cross-site requests, DNS rebinding) |
| `X-Frame-Options: DENY` and `Content-Security-Policy` with `frame-ancestors 'none'` | clickjacking (another site framing the page) |
| only listed models can be planned or started; numbers are range-checked; the API key goes to the launcher in `LLAMA_API_KEY`, never on a command line, and is masked in what the page shows | misuse of the API |

llama-server itself (port 8080): in **Agent mode** it gets the launcher's stable per-user API key (the owner's decision C,
handover/DECISION_LLAMA_SERVER_KEY_20261026.md); the page asks the launcher for it (`moe_cache.py key`), uses it for its own calls, writes it into
the agent's config with **Connect**, and shows **API key in use: ••••last 4** with a **Copy key** button (the key itself is fetched only for the
copy; never in an address, the log or the status). llama-server's own chat page asks for the key once per browser: paste it there. A key typed in
Settings wins over the launcher's. Plain chat servers have no key unless you type one.

## The four panels

| Panel | Shows | Controls |
|---|---|---|
| **1 Model** | every `.gguf` in the model folders (`~/models`, on Windows also `C:\models`, plus folders you add; and Ollama's model store when present), with size, quant (from the file name, else from the GGUF header) and the planner's verdict: **fits in RAM: stock llama-server**, **does not fit: moe-cache exact**, **all experts fit on the GPU**; the selected model lists the reasons (on Windows with an AMD GPU also the launcher's AMD rules: all experts on the CPU for a model bigger than RAM, `--ctx-checkpoints 0` for Flash-Next) and, for the models measured before, the measured speeds on the test laptop (stock and moe-cache, with the RESULTS section) | pick a model, add a folder, rescan |
| **2 Settings** | RAM budget (`--ram`, empty = available RAM), VRAM the planner may use (`--vram`, detected), context 8k / 32k / 64k, mode **Exact** (default, identical output) / **Fast** (`--fast`) / **Hybrid** (`--hybrid`, experimental), **Agent mode** (`--agent`), port, API key | |
| **3 Status** | decode tok/s (live while a reply is generated, else the last request), prefill tok/s of the last request, cache hit rate (last 2 s), SSD reads MB/s, expert cache used / budget (or llama-server's RAM when it runs stock), VRAM used, the mode the launcher chose, the launcher command; 2-minute charts of tok/s and hit rate; **What the launcher decided** (its own lines before it started llama-server: stock or plugin, GPU layers, micro-batch, pinning, and on Windows the GPU it found, the job memory limit and the AMD rules, highlighted); **warnings** from the log (e.g. the expert cache shrank because llama-server grew, the plugin disabled itself, why llama-server stopped), each once with a count | **Start**, **Stop**, copy the command, show the full llama-server command (`--dry-run`), **Quick benchmark** |
| **4 Chat / Agents / Log** | **Chat**: llama-server's own chat page in a frame; **Agents**: base URL, model name and context; per agent **Preview** (what would be written) and **Connect OpenCode / Hermes** (`moe_cache.py connect … --write`), which then shows the file and the block now in it (the API key is passed in the environment and never shown); and **Agent sessions** (below); **Log**: the last 500 lines, plugin lines highlighted | |

## Agent sessions

In Agent mode the launcher saves every long conversation when it is idle (one file per slot in `~/.cache/moe-cache/slots/agent/`, named
`<model>-c<context>[-<KV types>]-s<slot>.bin`, its context checkpoints in the same file) and restores the two largest of the same model and
settings at the next start. The Agents tab lists them all: model, context, KV type, slot, tokens (read from the file's header), size, when it was
saved, which agent it belongs to (guessed from the start of the conversation while its model runs: OpenCode, Hermes, ...), and whether this server
restored it at start; plus the server's live slots (through `/slots`, with the API key). **Move to Trash** sends a saved conversation to the
Trash / Recycle Bin (never a hard delete; the running server may save it again from memory). **Don't restore saved agent sessions** in Settings
starts the server with `--no-warm` (then nothing is restored or saved).

## Quick benchmark

For the running server: one warm-up request, two replies of 256 tokens (`ignore_eos`, so a model that finishes early is still timed over 256 tokens), and one prompt of about 3.5k tokens
(greedy, fixed seed, no prompt reuse; the method of the speed runs in RESULTS 54-55). It shows decode tok/s (both runs and the mean), prefill tok/s
and, with the plugin, the cache hit rate during the two replies, and appends one row to `~/.cache/moe-cache/bench.csv` (Windows:
`%USERPROFILE%\.cache\moe-cache\bench.csv`) with the model, mode, RAM budget, GPU and the launcher command. To compare stock and moe-cache on your
machine: run it once with a RAM budget the model fits in (the launcher runs stock) and once with a smaller one or `--fast` (the plugin), then read
both rows in the table under the button.

**Cold start (fair).** A model read once stays in the operating system's file cache, which makes stock llama-server look faster on the next start
than on a first one (the plugin reads experts unbuffered and gains less from it). Tick **Cold start** and the benchmark first stops the server,
pushes the model out of the file cache, starts it again with the same settings (the time until it is ready is recorded) and then measures; the
CSV row says `cold` or `warm` and how the flush went.

| System | How | Notes |
|---|---|---|
| Linux | `posix_fadvise(DONTNEED)` on the model's files, no root needed; it reports the cached share before and after (mincore) | refuses while another program maps the model (it would slow that program down); a file on tmpfs cannot be dropped and is reported as such |
| Windows | `tools\win_flush_cache.ps1`: reads other big files (more than the RAM in total) so the cache drops the model; reads only, writes nothing | pick the folders with the other files (default: your model folders; the model itself is skipped); too little data is reported as a partial flush |

For a fair stock number on Linux also run it under a memory limit when the model is bigger than the RAM, as the README's "Conditions" say.

## Where the verdict comes from

The same rules and tools as `moe-cache-server` itself, so the page never disagrees with what Start does:

- **fits in RAM** when the model plus 1.5 GB fits in the memory the server may use (the launcher's rule; stock llama-server is faster then, RESULTS 44);
- how many layers keep their experts on the GPU, the cache size and the micro-batch come from `planner/plan.py --json` and `planner/gguf_plan.py tune()`;
- the reasons list both, plus the tuner's own lines and a `--fast` hint when the model is more than twice the memory.

## Where the live numbers come from

The page starts `moe_cache.py server` with `MOE_CACHE_STATS_SIGNAL=1` and `--metrics`, and every 2 seconds asks the plugin for one stats snapshot
(Linux: SIGUSR1, sent only when llama-server has the plugin's handler installed; Windows: the named event `Local\moe-cache-stats-<pid>`).
Speed comes from llama-server's `/slots` (tokens decoded so far) and its timing lines. No change to the plugin or to llama.cpp.

VRAM: `nvidia-smi` for NVIDIA cards, sysfs for AMD on Linux. On Windows with an AMD or Intel GPU the page uses the launcher's own way: the GPU
llama.cpp will use from `llama-server --list-devices` (discrete before integrated) and its dedicated VRAM from the registry, asked every 10 seconds
while the page is open. An integrated GPU is marked as sharing RAM and shows free memory only.

## Tests

`tests/ui_test.py` needs no model (fake models, mocked planner, a fake server). `tests/ui_live.py MODEL.gguf --llama-server PATH` drives the real
page with a real model: start with a RAM budget just under the stock rule (so the plugin runs), one request, the live hit rate, stop. CI runs both
on Granite, on Linux and on Windows.

## Not yet

Model downloads, several servers at once, remote access, a native window. See
`handover/GUI1_DESIGN_DRAFT_20261026.md` and the ideas ledger (GUI1, I21, I33).
