# Control page (`moe-cache-server --ui`)

A local web page that lists your models, says which mode each one runs in and why, starts and stops the server, and shows live numbers.
It is served by the launcher itself: Python's standard library only, no installs, the same on Linux and Windows.

```bash
bin/moe-cache-server --ui
```

On Windows: double-click `bin\moe-cache-ui.cmd` (or `python bin\moe_cache.py ui`). The page opens at **http://127.0.0.1:8070**.

| Option | Meaning |
|---|---|
| `--ui-port N` | another port for the page (default 8070) |
| `--llama-server PATH` | the llama-server to start (default: `$LLAMA_SERVER`, else `llama-server` on the PATH) |
| `--no-browser` | do not open the browser |

Ctrl-C stops the page and any server it started. The page listens on 127.0.0.1 only and refuses requests from other web sites (foreign
`Host` or `Origin`, non-JSON posts), because it can start programs.

## The four panels

| Panel | Shows | Controls |
|---|---|---|
| **1 Model** | every `.gguf` in the model folders (`~/models`, on Windows also `C:\models`, plus folders you add; and Ollama's model store when present), with size, quant (from the file name, else from the GGUF header) and the planner's verdict: **fits in RAM: stock llama-server**, **does not fit: moe-cache exact**, **all experts fit on the GPU**; the selected model lists the reasons and, for the models measured before, the measured speeds on the test laptop (stock and moe-cache, with the RESULTS section) | pick a model, add a folder, rescan |
| **2 Settings** | RAM budget (`--ram`, empty = available RAM), VRAM the planner may use (`--vram`, detected), context 8k / 32k / 64k, mode **Exact** (default, identical output) / **Fast** (`--fast`) / **Hybrid** (`--hybrid`, experimental), **Agent mode** (`--agent`), port, API key | |
| **3 Status** | decode tok/s (live while a reply is generated, else the last request), prefill tok/s of the last request, cache hit rate (last 2 s), SSD reads MB/s, expert cache used / budget (or llama-server's RAM when it runs stock), VRAM used, the mode the launcher chose, the launcher command; 2-minute charts of tok/s and hit rate; **warnings** from the log (e.g. the expert cache shrank because llama-server grew, the plugin disabled itself, why llama-server stopped), each once with a count | **Start**, **Stop**, copy the command, show the full llama-server command (`--dry-run`) |
| **4 Chat / Agents / Log** | **Chat**: llama-server's own chat page in a frame; **Agents**: base URL, model name and context; per agent **Preview** (what would be written) and **Connect OpenCode / Hermes** (`moe_cache.py connect … --write`), which then shows the file and the block now in it (the API key is passed in the environment and never shown); in Agent mode also the **warm sessions**: conversations the launcher saved or restored in this run and the saved files it restores at the next start; **Log**: the last 500 lines, plugin lines highlighted | |

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
