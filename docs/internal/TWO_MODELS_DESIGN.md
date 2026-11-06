# Two models at once: design (MULTIMODEL-010 prototype)

Status 2026-11-04: **prototype on the branch `multimodel-010`, not merged, not run with two real models.** Built from what already exists
(`bin/moe_cache_budget.py`, `bin/moe_cache_concurrent.py`, `--peers`, `--share-ram`, the page's "Servers on this machine" card). Nothing here changes a
launch that does not ask for it. Background and the measurements behind each row: `handover/parts/multi_model_gpu_20261030.md` (MM1, MM2, MM5, MM7).

## The rule

One machine has one memory cap (physical RAM minus max(4 GiB, 12.5 %): 28 GiB on a 32 GiB PC). Two servers must stay under **one** cap together, not each
under its own. The page makes the split once, before either server starts, and the plugin servers then trade the split at run time.

## Design table

| Part | What it does | Where | Status |
|---|---|---|---|
| Memory split | Every model first gets its floor (a dense model: its whole file; a MoE model: non-expert weights + KV + 3 GiB for the prompt cache and agent margin, plus a minimum expert cache). The rest is shared by weight, nobody above its own file + 1 GiB. A model whose share holds its whole file runs as plain stock llama-server. | `moe_cache_budget.split_ram` | built earlier, unit-tested on fake models |
| When it does not fit | If the floors alone are above the cap there is **no split**: the page says so in words and starts nothing (swap one at a time with `on-demand` is the answer then). | `moe_cache_concurrent.group_plan` returns `ok: false` | built; the page shows it (this branch) |
| VRAM split | The free VRAM minus one shared reserve, shared by weight (static). | `moe_cache_budget.split_vram` | built earlier |
| CPU cores | One thread per physical core, cut into disjoint slices by weight (`-C mask --cpu-strict 1 -t N`) so the two servers do not time-slice the same cores. Fewer cores than models: no pinning. | `moe_cache_concurrent.core_slices` | built earlier |
| Ports | Each server its own port; the page picks 8080, 8081 (up to 3 servers: the main one and two more), refuses a port in use. One API key logic as today (the key typed on the page goes to the server it was typed for). | page server | built (GUI-013) and used here |
| Starting | **Both from idle** ("Start both with the split"): the page asks the planner, then starts each server through the launcher with `--ram`, `--vram`, `--peers` and the core slice. **One already running**: the new one starts with `--share-ram` (counts what the running one holds) and the page says the running one keeps its own plan. | page server (`/api/servers/plan`, `/api/servers/group`) | built (this branch) |
| What the page shows | The card "Servers on this machine" lists every server of the page (port, model, state, speed, images) with Chat here and Stop; the split panel shows, per model: RAM share, stock or plugin, VRAM share, cores, and the reasons; chat goes to the server picked. | `ui/index.html` | built (this branch) |
| What the plugin shares (`--peers`) | Each plugin server writes a small status file once a second to `~/.cache/moe-cache/peers` (memory held, cache size, busy or quiet) and reads the others'. The room a server may use is the shared cap minus what the others hold; a quiet server (no graph for 10 s) gives its cache back at 1 GiB/s down to a floor while a busy one has a full cache. Only cache sizes change, never what is computed. A stock server is not in the folder (that is why the split sets its share at start). | plugin `MOE_CACHE_PEERS`, launcher `--peers` | built in the plugin, run once on the laptop by hand (quiet 2.5 to 0.8 GiB, busy 6.0 to 7.6 GiB); not through the launcher |
| Stopping | Stop on each row (the main server has its own Stop); closing the page stops all of them. A stopped server's peer file is ignored after 5 s. | page server | built (GUI-013) |

## Where the code is

`bin/moe_cache_ui.py`: `group_plan_view` (the split for 2-3 models), `/api/servers/plan`, `/api/servers/group` (starts them one after the other), `start_extra` (a second server next to a running one gets `--share-ram`); `ui/index.html`: the panel "Two models at once (prototype)" in the card "Servers on this machine". Tests: `tests/servers_test.py` (fake machine and fake launcher, 50 checks).

## Rules decided from the code (2026-11-04, tested with fake servers in `tests/servers_test.py`)

| Question | Rule |
|---|---|
| The second model does not fit | Next to a running server the page subtracts what the running servers hold (their resident memory) from the one cap and checks the new model's minimum against the rest. If it does not fit: refused with the numbers and "stop one of them first, or pick a smaller model". A typed `--ram` skips the check (the user decided). A group start whose minimums do not fit under the cap starts nothing. |
| The same model twice | Refused (group, "Start another server", and a main restart): the warm-conversation files are named after the model, so two servers on one model would overwrite each other's saved conversations. |
| One stops or restarts | Stop on any row stops only that server; the others keep running and answering. Restarting the main server while others run counts what they hold (`--share-ram`) and may not use their model. Closing the page stops all. |
| Half-started group | The servers are started one after the other; if a later one cannot start, the earlier ones keep running and the page shows why (`group_error`). |
| Ports | Default 8080, 8081; a port used by another server of the page or by anything else is refused before anything starts. At most 3 servers (main + 2). |
| Agents | "Connect OpenCode / Hermes" takes an optional `port`: it connects the agent to that server of the page (default: the main one). The saved-conversation list in Agents still shows the main server only. |
| Memory trading | Only between plugin servers started with `--peers` (a group start does that). A server started by hand next to the page's servers is counted at its start by `--share-ram`, not at run time. |

## What is not decided or not done

| Item | Why |
|---|---|
| Resizing a running server | A running server keeps its `--ram`; only `--peers` moves memory at run time. Starting a second one next to a running one is therefore a weaker plan than starting both together. |
| Two big models | Two models whose floors do not fit are refused, not swapped by the page. `on-demand` swaps. |
| Speed | No claim. Interference through the page cache, the SSD and PCIe, and Windows commit with two GPU servers are the things the PC test must look at. |
| Router mode (MM4) | Not used: it does not manage RAM. |

## PC test (about 15 minutes, in an agreed window, not run)

1. Page: pick the two small models, press "Show the split", read the numbers against the 28 GiB cap.
2. "Start both with the split": both reach "ready"; send one short chat to each (Chat here).
3. Watch RAM in Task Manager while one chats and the other is quiet: the quiet one should shrink (peers), total stays under the cap.
4. Stop both from the page; nothing is left running.
