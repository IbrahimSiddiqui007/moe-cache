# moe-cache desktop app (scaffold)

A Tauri 2 shell around the moe-cache control page. It starts `bin/moe_cache.py ui --ui-port 0 --no-browser` as a child process, reads the page's
address from its output, and shows the page in its own window, with a tray icon (Open / Start model / Stop model / Quit) and one instance only.
Design, process model, packaging, updater, risks and plan: [docs/DESKTOP_APP_DESIGN.md](../docs/DESKTOP_APP_DESIGN.md).

Status (2026-11-03): runs on Linux (window shows the control page, chat streams), CI builds Windows NSIS (30.5 MB, with the kit: plugin DLLs and
llama.cpp Vulkan) + MSI and Linux AppImage + deb. Tray: Open, Start model, Stop model, Restart control page, Start at login (off by default), Quit.
Not yet run on Windows by hand (owner click-through, day 3).

Tests without a model: `python3 desktop/tests/shell_contract_test.py` (the page started as the shell starts it, a fake model, chat stream, stop) and
`cd desktop/src-tauri && nice -n 19 cargo test`.

## Run it in development (Linux)

Needs Rust, Node 18+, and the Tauri libraries (Debian/Ubuntu: `libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev patchelf`;
Arch: `webkit2gtk-4.1 gtk3 libayatana-appindicator librsvg patchelf`). Nothing is installed system-wide by these steps.

```bash
cd desktop
npm install --no-audit --no-fund
nice -n 19 npx tauri dev
```

In development the shell finds the kit by walking up from its executable to the repository (`bin/moe_cache.py` + `ui/index.html`), so no staging is
needed. Environment variables: `MOE_CACHE_HOME` (kit folder), `MOE_CACHE_PYTHON` (interpreter).

Compile check only (no window, lighter): `cd src-tauri && nice -n 19 cargo check`.

## Build installers

Linux (AppImage + deb): `desktop/scripts/build-linux.sh`. Windows and Linux in CI: the **desktop** workflow runs on pull requests that touch `desktop/` and by hand (Actions → desktop → Run
workflow, once the file is on the default branch). Python 3.12.8's embeddable zip is bundled with a pinned SHA-256 (inputs override it, `none` skips it). The plugin and llama.cpp are not staged yet (milestone M1b/M2 in
the design); the app uses the ones the page finds, as the browser page does.

## Layout

| Path | What |
|---|---|
| `app_entry.py` | what the shell runs: the control page + stable address line + stdin-EOF watch (the model stops if the shell dies) |
| `src-tauri/src/main.rs` | the whole shell |
| `src-tauri/tauri.conf.json`, `tauri.windows.conf.json` | window, bundle targets, resources (Windows adds the bundled Python) |
| `src-tauri/capabilities/default.json` | the control page gets no Tauri commands |
| `web/index.html` | splash and error page |
| `CLICK_THROUGH.md` | the Windows click-through for the owner (about 45 minutes) |
| `tests/` | `shell_contract_test.py`, no model |
| `scripts/` | `stage.sh` (kit files into the bundle), `build-linux.sh`, `fetch-python-win.ps1` |
| `icons/` | the app icon: the owner's curled dragon holding a red cache block, pixel art. `make_app_icon.py` builds `src-tauri/icons/` from the maps in `icons/concepts/D/` (64x64 for 64 px and up, 32x32 for 32-48 px, hand-made 16x16 for 16-24 px; nearest-neighbour only). `icons/concepts/` keeps the other concepts (A-E, draft 1) for history |
| `src-tauri/icons/` | the generated files (`icon.icns` came from `npx tauri icon` on a 1024 px nearest-neighbour render; macOS is not in version 1) |

Control-page changes the app needs are listed in section 8 of the design; they are not made here.
