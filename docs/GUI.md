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

llama-server itself (port 8080) has no API key unless you set one: any program of any user on this machine can use it while it runs.

## The four panels

| Panel | Shows | Controls |
|---|---|---|
| **1 Model** | every `.gguf` in the model folders (`~/models`, on Windows also `C:\models`, plus folders you add; and Ollama's model store when present), with size, quant (from the file name, else from the GGUF header) and the planner's verdict: **fits in RAM: stock llama-server**, **does not fit: moe-cache exact**, **all experts fit on the GPU**; the selected model lists the reasons (on Windows with an AMD GPU also the launcher's AMD rules: all experts on the CPU for a model bigger than RAM, `--ctx-checkpoints 0` for Flash-Next) and, for the models measured before, the measured speeds on the test laptop (stock and moe-cache, with the RESULTS section) | pick a model, add a folder, rescan |
| **2 Settings** | RAM budget (`--ram`, empty = available RAM), VRAM the planner may use (`--vram`, detected), context 8k / 32k / 64k, mode **Exact** (default, identical output) / **Fast** (`--fast`) / **Hybrid** (`--hybrid`, experimental), **Agent mode** (`--agent`), port, API key | |
| **3 Status** | decode tok/s (live while a reply is generated, else the last request), prefill tok/s of the last request, cache hit rate (last 2 s), SSD reads MB/s, expert cache used / budget (or llama-server's RAM when it runs stock), VRAM used, the mode the launcher chose, the launcher command; 2-minute charts of tok/s and hit rate; **What the launcher decided** (its own lines before it started llama-server: stock or plugin, GPU layers, micro-batch, pinning, and on Windows the GPU it found, the job memory limit and the AMD rules, highlighted); **warnings** from the log (e.g. the expert cache shrank because llama-server grew, the plugin disabled itself, why llama-server stopped), each once with a count | **Start**, **Stop**, copy the command, show the full llama-server command (`--dry-run`), **Quick benchmark** |
| **4 Chat / Agents / Log** | **Chat**: llama-server's own chat page in a frame; **Agents**: base URL, model name and context; per agent **Preview** (what would be written) and **Connect OpenCode / Hermes** (`moe_cache.py connect … --write`), which then shows the file and the block now in it (the API key is passed in the environment and never shown); in Agent mode also the **warm sessions**: conversations the launcher saved or restored in this run and the saved files it restores at the next start; **Log**: the last 500 lines, plugin lines highlighted | |

## Quick benchmark

For the running server: one warm-up request, two replies of 256 tokens (`ignore_eos`, so a model that finishes early is still timed over 256 tokens), and one prompt of about 3.5k tokens
(greedy, fixed seed, no prompt reuse; the method of the speed runs in RESULTS 54-55). It shows decode tok/s (both runs and the mean), prefill tok/s
and, with the plugin, the cache hit rate during the two replies, and appends one row to `~/.cache/moe-cache/bench.csv` (Windows:
`%USERPROFILE%\.cache\moe-cache\bench.csv`) with the model, mode, RAM budget, GPU and the launcher command. To compare stock and moe-cache on your
machine: run it once with a RAM budget the model fits in (the launcher runs stock) and once with a smaller one or `--fast` (the plugin), then read
both rows in the table under the button. A model bigger than the memory runs stock very slowly: on Linux give the stock run a memory limit and
start from a cold file cache, as the README's "Conditions" say, or the stock number is too good.

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
