# Desktop app: design (2026-11-03)

Status: **design + scaffold** (`desktop/`). Owner decision 2026-11-03: build it, after release v0.3.6. Nothing here ships yet.

## 1. What the app is

An installer-based Windows (and Linux) app for people who find a terminal or a browser page awkward. Install, click the icon, a window opens with
the **existing control page** in it. No Python to install, no browser tab, no terminal.

It does what the control page does, nothing else of its own:

| Job | How (reuses) |
|---|---|
| Start / stop the server | the page's Start / Stop (`Runner` in `bin/moe_cache_ui.py`); the tray has Start model / Stop model |
| Pick a model, download one | the page's model list and "picker"; downloads: `launch` / `pick` / `models.json` logic. A **download card in the page** is the one new UI feature (GUI-008), shared with the browser page |
| Connect agents | the page's Agents card (`connect`, `connect --check`) |
| Show status | the page's Live card; the tray tooltip later shows the state |
| Tray icon | Open / Start model / Stop model / Quit. Closing the window hides to the tray; Quit ends everything |
| Start at login | optional, **off by default**; a checkbox in the page's Settings (needs the page to ask the shell, see section 8) |
| Single instance | a second launch only shows the first window (`tauri-plugin-single-instance`) |
| Not in version 1 | macOS, several models at once, remote access, an own chat window, auto-update without asking |

**One UI, two hosts.** `ui/index.html` is served by `bin/moe_cache_ui.py` and shown either in the browser (today) or in the app's window. The shell has
no UI of its own except a splash/error page. A fix to the page lands in both.

## 2. Summary of the decisions

| Topic | Decision |
|---|---|
| Shell | Tauri 2 (Rust, system webview: WebView2 on Windows, WebKitGTK on Linux) |
| UI | the control page, unchanged in spirit; the window navigates to `http://127.0.0.1:PORT/` |
| Backend | `python bin/moe_cache.py ui --ui-port 0 --no-browser` as a child ("sidecar") process |
| Python | Windows: python.org embeddable zip (about 11 MB) inside the app. Linux: the system `python3` (every supported distro has it); AppImage does not bundle Python in v1 |
| llama.cpp and plugin | the kit's official llama.cpp build and the plugin DLL/so inside the app's resources, as the Windows kit does today |
| Signing | no code-signing certificate; README explains SmartScreen. Update files are signed with our own Tauri key (that is not a certificate) |
| Updates | Tauri updater against GitHub Releases (`latest.json` + signed bundles), asks before installing |
| Installers | Windows NSIS (per-user, no admin) and MSI; Linux AppImage and deb |
| Releases | built by `desktop.yml` (manual dispatch now); attached to the normal release later |

## 3. How the app hosts the control page

```
 Tauri process (Rust)                        Python child                      llama-server
 ┌──────────────────────┐  spawn, env, pipes ┌───────────────────────┐ spawn   ┌──────────────┐
 │ tray, single-instance│───────────────────▶│ moe_cache.py ui       │────────▶│ + plugin     │
 │ webview ─────────────┼── http://127.0.0.1:PORT/?t=TOKEN ─▶ ui/index.html    │         └──────────────┘
 └──────────────────────┘◀── "control page at URL" (stdout) ──┘
```

1. The shell starts the child with `--ui-port 0` (the OS picks a free port: no clash with an open browser page on 8070 and no "port in use").
2. It reads the line `moe-cache: control page at http://127.0.0.1:PORT/?t=TOKEN …` from the child's output (today's format; section 8 asks for a
   stable machine-readable form) with a 60 s timeout. On a timeout or an early exit the splash page shows the last 15 output lines.
3. It navigates the window to that address. The page server answers with a redirect that moves the token into a cookie and drops it from the address.
4. The webview is a remote `127.0.0.1` page: the Tauri capability file gives it **no Tauri commands**, so a script in the page (or in a chat answer
   rendered in it) cannot call into Rust. The shell talks to the page only through its HTTP API with the `X-MoeCache-Token` header (stop on Quit,
   tray Start / Stop).

**Token handling.** Today the token travels in the address (`?t=`) once, then lives in an HttpOnly SameSite cookie. The address exists in the child's
stdout pipe, the shell's memory and the webview's first request; it never reaches a command line (so not `/proc/*/cmdline`). The control page is
**moving the token from the cookie to a request header right now (another session)**. The app must follow that change; what it needs is in section 8
(item 1): the page must still be able to start from `?t=` in the address, keep the token in JavaScript memory / `sessionStorage` and send the header
on every call. The shell side does not change: it still navigates to the printed address and uses the header for its own calls.

Alternative considered and not taken: the shell proxies all requests and injects the header itself (a custom protocol `moecache://`). It would hide
the token from the page entirely, but costs a streaming proxy for the chat (SSE) and breaks relative-URL assumptions. Revisit only if the page
cannot do the bootstrap above.

**Navigation checks the server makes.** `Handler.allowed()` wants `Host: 127.0.0.1:PORT`, no foreign `Origin`, and `Sec-Fetch-Site` same-origin or
none (or the cross-site launcher link). A webview-initiated first navigation sends `none` or `cross-site` depending on the engine; the `?t=` link rule
covers cross-site. **To verify on WebView2 and WebKitGTK in milestone M1**; if one engine sends something else, the fix is a one-line allowance in
`allowed()` keyed on the app flag (section 8, item 6).

## 4. Process model

| Event | Windows | Linux |
|---|---|---|
| App start | shell spawns bundled `python.exe` (CREATE_NO_WINDOW), cwd = the app's `moe-cache` resource folder | spawns `python3` (or `MOE_CACHE_PYTHON`) |
| Python finds llama-server | `LLAMA_SERVER` unset → the bundled `llama-server.exe` next to the kit (the page's search already includes the kit folder); user override stays | same with the bundled binary; falls back to the existing search |
| Plugin | `MOE_CACHE_LIB` found next to the kit by the existing search | same |
| Quit (tray) | shell: `POST /api/stop`, polls `/api/status` until not starting / ready / stopping (max 20 s), then kills the Python child | same; |
| Window close | hides to the tray; model keeps running (a long download or a chat is not lost) | same |
| Shell killed or crashes | **a hard kill leaves Python and llama-server running** unless guarded: Windows: the child is put in a Job Object with KILL_ON_JOB_CLOSE (M2); Linux: `PR_SET_PDEATHSIG` in the child spawn, plus the stdin-EOF watch below | |
| Python child dies | the shell notices (reader threads end), the window shows "the control page stopped", offers Restart (M2); the model it started is stopped by the page's own exit handler or found again via the "Other servers" list | same |
| llama-server dies | handled by the page already (state "ended", Log card) | same |
| Stdin watch | the shell keeps the child's stdin open; the page treats EOF as "the app is gone" → `runner.stop()` and exit (control-page change, section 8 item 3). This covers the case where nothing else can run | same |
| Second launch | single-instance plugin: focuses the first window | same |

The shell spawns `python -u bin/moe_cache.py ui …` without `-I`: the launcher imports sibling files from its own folder, which `-I` would remove from
the path. The embeddable zip needs its `python3NN._pth` to list `..\moe-cache\bin` and `..\moe-cache\planner`; `scripts/fetch-python-win.ps1` writes that.

## 5. Packaging

| Target | Format | Notes |
|---|---|---|
| Windows | NSIS `-setup.exe` (per user, no admin) and `.msi` | WebView2: the evergreen runtime ships with Windows 10 1803+/11; the NSIS installer uses Tauri's WebView2 bootstrapper as fallback. Unsigned → SmartScreen "More info → Run anyway", explained in the README. Windows Smart App Control can block unsigned exes outright (competitor issue #735 in strata_compare); the app inherits that limit, the browser page + `install.ps1` stays the fallback |
| Linux | AppImage and `.deb` | needs `webkit2gtk-4.1`; the AppImage bundles the Tauri libs but not Python |

Resources inside the app: `moe-cache/` (bin, ui, planner, models.json, the plugin), `python/` (Windows), and llama.cpp (the same official b12064
build the kit uses; GPU backends as in the kit: Vulkan for Windows).

**Size budget** (setup file; to be measured by the first CI run):

| Part | Windows | Linux |
|---|---|---|
| Tauri shell (release, LTO, stripped) | 4-6 MB | 4-6 MB |
| Python embeddable | 11 MB | 0 |
| moe-cache files | under 1 MB | under 1 MB |
| llama.cpp Vulkan build (compressed) | 40-70 MB | 25-45 MB (kit tarball) |
| **Total download** | **about 60-90 MB** (ledger estimate 60-100) | **about 30-50 MB** (AppImage 15-25 MB without llama.cpp) |

Budget rule: the app without llama.cpp stays under 25 MB. If the total exceeds 100 MB on Windows, llama.cpp is downloaded on first run instead
(from the same release, SHA-256 checked, with the owner's OK per file as for models).

## 6. Updater

Tauri's updater plugin reads `latest.json` from the GitHub Release of the app and installs a signed bundle after the user says yes.

- A **minisign key pair made for update files** (`tauri signer generate`). The public key sits in `tauri.conf.json`; the private key is a GitHub
  Actions secret (`TAURI_SIGNING_PRIVATE_KEY`). This is not a code-signing certificate and does not remove SmartScreen warnings.
- Update checks: only when the user presses "Check for updates" in the page's About card or enables "check at start" (privacy statement in
  docs/SECURITY.md lists every network use; this adds one row).
- Releases must be normal (latest) releases so the static `…/releases/latest/download/latest.json` URL works (already the rule in the CHANGELOG).
- The kit updates (the plugin, llama.cpp) travel in the same app bundle: one version, one update. The app version = the moe-cache release version.
- Not in the scaffold (M4). The scaffold has no updater plugin so no key is needed for the first build.

## 7. Risks

| Risk | Impact | Answer |
|---|---|---|
| Unsigned installer: SmartScreen, Smart App Control, antivirus false positives (a bundled Python + Rust exe is a common flag) | friends hesitate or are blocked | README explains; publish SHA256SUMS; keep `install.ps1` and the `.bat` as the other way; submit false positives to Microsoft/vendors; decision is the owner's |
| WebKitGTK old or missing on Linux (and Wayland quirks) | blank window or no app | `.deb` declares the dependency; fall back to the browser page (`--ui`) is documented; Linux is secondary |
| Token handling changes in the control page (cookie → header) | blank page / 403 in the app | section 8 item 1; M1 verifies on both engines; tests in `tests/ui_test.py` for the `?t=` bootstrap |
| Orphaned llama-server after a shell crash | memory held, port busy | Job Object / PDEATHSIG / stdin EOF (section 4) |
| Webview engine differences (WebView2 vs WebKitGTK): the page was only checked in desktop browsers | layout or SSE streaming bugs | M1 checks the chat stream, file attach, download links on both; CI smoke test starts the app headless on Linux (xvfb) |
| Bundled llama.cpp ages | stale, a security fix is missed | one app version per kit release; the updater delivers it; same pin as the kit (`b12064`) |
| Two UIs diverge | double work | the shell has no UI; every feature goes into `ui/index.html` first |
| Antivirus kills or sandboxes the Python child | the splash error page | the error page shows the output; docs/TROUBLESHOOTING gets an entry |
| Scope creep (own chat window, many models) | schedule | listed under "Not in version 1" |
| Maintenance: Tauri/Rust upgrades, a second build system | ongoing cost | pin Tauri 2.x, Dependabot monthly, builds only on dispatch / tag; no Rust beyond about 300 lines |

## 8. What the control page needs to change (do not make these in the audit-fix branches)

Listed by priority. Items 1-4 are needed for a usable app; the rest can follow.

| # | Change in `bin/moe_cache_ui.py` / `ui/` | Why |
|---|---|---|
| 1 | **Token bootstrap without a cookie**: the page reads `?t=` from its address once, keeps the token in memory/`sessionStorage`, removes it from the address (`history.replaceState`), sends `X-MoeCache-Token` on every request. Still accept the cookie for older callers until removed | the other session is moving the token to a header; the app opens the page by address and must keep working. Apply the same rule to the SSE chat (`fetch` with headers, not `EventSource`) |
| 2 | **Stable start line**: with `MOE_CACHE_APP=1` (or `--app`) print one JSON line `{"moe_cache_ui":{"url":"http://127.0.0.1:PORT/?t=…","port":PORT,"pid":N}}` to stdout and flush; the human line stays for the terminal | the shell parses a sentence today ("control page at "); a wording change breaks the app silently |
| 3 | **Parent-gone watch**: with `MOE_CACHE_APP=1` read stdin in a thread; on EOF call `runner.stop()` and exit. Also handle SIGTERM/SIGBREAK the same as Ctrl-C | the shell dying must not leave llama-server running; Windows has no PDEATHSIG |
| 4 | **Do not open a browser** when started by the app (the shell passes `--no-browser`; make `MOE_CACHE_APP=1` imply it) and skip the `open_page` temp file | defence in depth |
| 5 | **`POST /api/quit`**: stop the model and exit the process after replying; lets the shell do a clean Quit with one call instead of stop + poll + kill | simpler and safer Quit |
| 6 | **Navigation allowance**: if M1 shows a webview sends a `Sec-Fetch-Site` the `allowed()` check rejects, accept it when started by the app and the Host and token are right | engine differences |
| 7 | **App-aware UI bits** (`/api/system` returns `{"app": true}`): hide "Open llama-server's own chat page in a new tab" or route it to the system browser; link clicks (`target=_blank`, Hugging Face, GitHub issue links) must leave the app: the shell opens `http(s)` links outside `127.0.0.1` with the system browser (shell side, M2) | a model page inside the app window would have no way back |
| 8 | **Start at login switch**: a Settings checkbox that calls `POST /api/app/autostart` → needs a shell command. Because the page has no Tauri IPC (by design), the shell exposes it as a tiny local endpoint **or** the page writes `ui.json` `autostart` and the shell reads it at its next start | keep the webview free of IPC |
| 9 | **Model downloads in the page (GUI-008)**: card with progress, disk check, SHA-256, resume, owner OK per file, using `models.json` | the app's "pick and download a model" job; useful in the browser page too. This is the largest item (about 2 days) |
| 10 | **Chat/session storage already file-based** (comment at the Store); keep it that way: the app picks a new port each run, so browser storage (localStorage per port) would lose state. Check that no new feature uses `localStorage` for data that matters | per-port storage |
| 11 | **About card**: app version, kit version, "Check for updates" (M4) | updater |
| 12 | **File dialogs**: "Add folder" needs a path typed today; in the app it can ask the shell for a folder picker (needs item 8's channel) | friendliness for non-terminal users |
| 13 | **Windows `moe-cache-ui.bat` and `--shortcut`** stay; the installer's Start-menu entry replaces `--shortcut` for app users | none |

## 9. Effort plan (days, one person)

Estimates assume the audit-fix sessions are merged and v0.3.6 is out. "Day" = a working day including CI round-trips (Windows CI is the slow part).

| Day | Milestone | Done when |
|---|---|---|
| 0 (done) | M0 scaffold | this design; `desktop/` compiles on Linux (`cargo check`); CI workflow exists |
| 1 | **M1a Linux runs** | `tauri dev` on Linux opens the window on the control page; tray Start/Stop/Quit work; the token bootstrap works with section 8 items 1, 2 (small page patches) |
| 2 | **M1b Windows CI build** | `desktop.yml` builds an NSIS + MSI + AppImage + deb; Python embeddable staged; installer size measured against section 5 |
| 3 | **M1c Windows run** | install on a Windows machine (friend or the PC), app opens, Start a small model with the bundled llama-server; webview differences found and fixed (item 6) |
| 4 | **M2 process safety** | Job Object / PDEATHSIG, stdin-EOF watch (item 3), `/api/quit` (item 5), Python-child-died screen, external links (item 7), crash tests |
| 5 | **M2b polish** | icons and name; first-run text; window state (size, position); start-at-login (item 8); single-instance checked on Windows |
| 6-7 | **M3 model downloads in the page** (item 9) | pick a model, press Download, progress, checksum, resume; usable by the browser page too |
| 8 | **M4 updater** | key pair, `latest.json` in the release, "Check for updates" (item 11), update from version N to N+1 tested |
| 9 | **M5 release wiring** | `release.yml` attaches app files and SHA256SUMS; README / WINDOWS.md install section; SmartScreen text; privacy table row; friend test (docs/FRIEND_TEST.md) |
| 10 | buffer | fixes from the friend test |

A **first usable app** (a friend installs it, starts a model, chats, quits cleanly) is the end of **day 4**, assuming items 1-5 of section 8 are in.
Everything after is polish, downloads and updates. Total about 9-10 days, in line with the ledger's 8-10.

## 10. Scaffold in `desktop/`

| File | Role |
|---|---|
| `src-tauri/src/main.rs` (about 280 lines) | spawn the page, read its address, navigate, tray, single instance, clean Quit through the page's API |
| `src-tauri/tauri.conf.json`, `Cargo.toml`, `capabilities/default.json` | window, bundle targets, resources mapping; no IPC for the remote page |
| `web/index.html` | splash and error page (the only UI of the shell) |
| `scripts/stage.sh`, `build-linux.sh`, `fetch-python-win.ps1` | stage kit files; Linux build; Windows Python staging |
| `.github/workflows/desktop.yml` | manual build, Windows + Linux |

How to build and run: `desktop/README.md`.
