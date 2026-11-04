# Windows click-through for the desktop app (about 45 minutes)

For the owner, on the Windows PC. Nothing here needs a terminal. The build comes from the **desktop** workflow: GitHub, Actions, desktop, the newest green
run, artifact `moe-cache-desktop-Windows` (a zip with a `-setup.exe` and an `.msi`). Use the `-setup.exe`.

Write down anything that differs from "You should see". If a step fails, send the page's **Log** tab (or a photo of the window) and the step number.

| # | Do | You should see | Min |
|---|---|---|---|
| 1 | Download the artifact zip, unzip it, run the `-setup.exe` | Windows SmartScreen says "Windows protected your PC": click **More info**, then **Run anyway** (expected: the installer is not signed). It installs for you only, no admin prompt | 5 |
| 2 | Start moe-cache from the Start menu | A window with a short "Starting moe-cache…" text, then the control page within about 10 seconds. No black console window. A tray icon appears. Python does not need to be installed on this PC (best checked on a PC without Python) | 3 |
| 3 | Look at the page's header and the model list on the left | The header shows RAM and the GPU. Models in your usual folders are listed. llama.cpp is found by itself (the app carries it: no "llama.cpp not found" box) | 3 |
| 4 | Pick a small model (Qwen3.6 20 GB or smaller) and press **Start** | The status goes to ready. Task Manager shows `llama-server.exe` without a window | 8 |
| 5 | Send two chat messages | The answer streams word by word. The second answer starts quickly | 5 |
| 6 | Click a link on the page (for example "Official llama.cpp releases" in Settings) | The link opens in your normal browser, not inside the app window | 2 |
| 7 | Close the window with **X**, then click the tray icon | The window hides, the model keeps running, the tray icon shows the window again | 3 |
| 8 | Start moe-cache from the Start menu a second time | The first window comes to the front; no second window or second tray icon | 2 |
| 9 | Tray menu: **Start at login** (tick), sign out and in, then untick it again | After signing in only the tray icon appears (no window). Unticked: nothing starts at login | 4 |
| 10 | Move and resize the window, quit from the tray, start the app again | The window opens where and how big you left it | 2 |
| 11 | Tray menu: **Check for updates…** | A box says updates are not set up in this build yet (the update key is not added): that is correct for now | 1 |
| 12 | In Task Manager, end the `python.exe` of moe-cache while the model runs | The window shows "moe-cache's control page stopped". Tray menu **Restart control page** brings the page back. Note whether `llama-server.exe` is still running (it should be gone) | 3 |
| 13 | End `moe-cache.exe` itself in Task Manager (Details tab) while a model runs | Within a few seconds `python.exe` and `llama-server.exe` are gone too. Note if either stays | 3 |
| 14 | Tray menu: **Quit** | Window and tray icon gone; Task Manager shows no moe-cache, python or llama-server | 2 |
| 15 | Uninstall from Settings, Apps | Removed; your models and chats (in your user folder) are untouched | 3 |
| 16 | Send the notes (what differed, step numbers, the Log tab for failures) | | 2 |

Steps 12 and 13 test the safety nets (the model must not be left running when the app dies); they matter most.
