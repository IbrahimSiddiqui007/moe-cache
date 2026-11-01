# Try the control page (Windows and Linux)

About 30 minutes with a small model. Use a model that fits in RAM for steps 1-9 (for example Qwen3.6-35B, 20 GB) and a bigger one for step 10. Nothing here changes your system; the page listens on this machine only. Details of each part: [GUI.md](GUI.md).

| # | Do | You should see |
|---|---|---|
| 1 | Start the page. Windows: double-click `moe-cache.bat` in the kit folder. Linux: `bin/moe-cache-server --ui` | A browser tab opens at `http://127.0.0.1:8070/?t=...`; the terminal prints the same address |
| 2 | Look at the header and hover over it | The llama.cpp build and plugin it found. If llama.cpp is missing, a box "llama.cpp not found" opens with the path example for your system (Windows: `C:\llama\llama-server.exe`; Linux: `.../build/bin/llama-server`) |
| 3 | Type a wrong path in that box and press **Check and remember** | A red message under the box; nothing is saved. A right path closes the box |
| 4 | With no model in any folder, look at the main area | The text "No model found yet. Open Settings > Which model? ... or put a .gguf file in a model folder and press Rescan" |
| 5 | Open **Settings > Which model?**, choose a use (agents, chat, long documents) | One model, its context, the measured numbers (or "untested on this class of machine"), the download command, and warnings in red. A model that does not fit the disk or memory is **warned about, never refused**; the memory line includes the context you chose |
| 6 | Download the model with the printed command, then press **Rescan** | The file appears in the list with the planner's verdict and why |
| 7 | Select it and press **Start** (top right) | The status goes from loading to ready; the first start takes a minute or two (a long first answer on a model bigger than RAM can take minutes: [INSTALL.md](INSTALL.md)) |
| 8 | Send a chat message | An answer streams in; speed, MTP and cache hits show under it |
| 9 | Press **Stop**, close the tab, press Ctrl-C in the terminal | The model process is gone (Task Manager / `ps`) |
| 10 | Repeat 7-8 with a model bigger than your RAM | The live panel shows cache hit rate, SSD reads, RAM and VRAM; answers match what stock llama.cpp gives |
| 11 | Agent mode, Connect buttons, saved sessions | Follow the checklists in [AGENTS.md](AGENTS.md) (PC checklist and T8) |
| 12 | Windows only: run `moe-cache.bat --shortcut`, start "moe-cache" from the Start menu | The page opens the same way |

Report what differed from the right column, with a screenshot, the terminal text and `python bin/moe_cache.py check MODEL.gguf` output (Linux: `bin/moe-cache-check`).
