# moe-cache desktop app (scaffold)

A Tauri 2 shell around the moe-cache control page. It starts `bin/moe_cache.py ui --ui-port 0 --no-browser` as a child process, reads the page's
address from its output, and shows the page in its own window, with a tray icon (Open / Start model / Stop model / Quit) and one instance only.
Design, process model, packaging, updater, risks and plan: [docs/DESKTOP_APP_DESIGN.md](../docs/DESKTOP_APP_DESIGN.md).

Status: skeleton. `cargo check` passes on Linux (2026-11-03); it has not been run in a window or built on Windows yet (first Windows build: the
`desktop` workflow).

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

Linux (AppImage + deb): `desktop/scripts/build-linux.sh`. Windows and Linux in CI: run the **desktop** workflow by hand (Actions → desktop → Run
workflow); give the SHA-256 of the python.org embeddable zip to bundle Python. The plugin and llama.cpp are not staged yet (milestone M1b/M2 in
the design); the app uses the ones the page finds, as the browser page does.

## Layout

| Path | What |
|---|---|
| `src-tauri/src/main.rs` | the whole shell |
| `src-tauri/tauri.conf.json`, `tauri.windows.conf.json` | window, bundle targets, resources (Windows adds the bundled Python) |
| `src-tauri/capabilities/default.json` | the control page gets no Tauri commands |
| `web/index.html` | splash and error page |
| `scripts/` | `stage.sh` (kit files into the bundle), `build-linux.sh`, `fetch-python-win.ps1` |
| `src-tauri/icons/` | placeholder icon; replace with `npx tauri icon art.png` |

Control-page changes the app needs are listed in section 8 of the design; they are not made here.
