# PC script and Actions cache (contributor notes)

## When to use what

| Situation | Use | Why |
|---|---|---|
| Work branch, before pushing | `tools/pc_ci.sh <branch-or-sha>` (add `--model` for the Granite steps) | 4 min without the model, 8.5 min with it; nothing is pushed; finds Windows problems before a GitHub round |
| Merge into `experimental` | GitHub CI (quick, windows, identity) on the branch | The merge rule needs green runs on the branch |
| Release commit / tag | GitHub CI on the exact commit (`tools/release_gate.sh`) | The gate reads GitHub runs of the exact tag commit; the PC script is never evidence for it |

The PC is not a GitHub runner (the repository is public: no self-hosted runner, ever).

## What the PC script runs

| Part | Where | Same as GitHub? |
|---|---|---|
| Plugin DLL, test programs, kit (mingw cross-build) | laptop (the PC has no compiler) | `windows.yml` job `dll`, but the laptop compiler is GCC 16 / UCRT, GitHub's is GCC 13 / msvcrt.dll. Both pass the same import check. The DLL bytes differ |
| Native Windows tests, launcher and python tests, one-click start | PC (`tools/pc_ci_windows.ps1`) | `windows.yml` job `windows` (PowerShell 5.1 instead of pwsh 7, Python 3.13 instead of 3.12) |
| Granite steps (`--model`): kit check, bench-report, real start, control page, on-demand, 4 identity runs | PC | Same model and llama.cpp b12207 as the workflow (sha256-checked, cached in `C:\dl\pc_ci\cache`); max 15 min, a watchdog stops the PC's llama-server after that |
| `installer` job, `--shortcut`, `--install-autostart`, crash-dump registry key | not run | They change the real PATH, Start menu, Startup folder or HKLM of the owner's PC: GitHub only |
| `identity.yml` (Linux, builds llama.cpp at 4 pins) | not run | Linux only; the PC has no compiler and no WSL distro. Runs on GitHub; the Windows identity steps above are the PC's counterpart |

The script refuses while `C:\dl\afternoon\runner.pid` is alive, any `llama-server` runs, or another pc_ci run is active. It writes only below `C:\dl\pc_ci`; work folders older than the newest 3 are moved to `D:\to_delete\pc_ci` (never deleted). Logs: `~/.cache/pc_ci/<time>_<sha>/` (summary table printed at the end, exit 0 only if nothing failed).

## Actions cache

* `windows.yml` and `identity.yml` cache nothing: their builds are short (dll job 2 min; llama.cpp at a pin 4 of 40 min in identity).
* GitHub scopes a cache to the branch that saved it. A run reads its own branch, the default branch (`main`) and, for a pull request, the base branch. The same key saved from three branches is three entries; that is how 8 caches / 8.5 GB came about.
* Policy now: work branches only restore (`restore-keys` falls back to the newest entry of the same OS); `main` and `experimental` save one entry per OS and delete their older ones. A push to a `claude/desktop-*` branch can only read `main`'s cache (not `experimental`'s), so it is warm only after `main` has run the workflow once.
* After merging and deleting a work branch: `tools/cache_cleanup.sh` lists caches of closed PRs and deleted branches; `--delete` removes them.
