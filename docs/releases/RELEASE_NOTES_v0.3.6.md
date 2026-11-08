# moe-cache v0.3.6 (2026-11-04)

**This is the last release that supports ggml backend API 2** (llama.cpp up to commit `7fe450e19`, ggml 0.25.1). From v0.3.7 on, only backend API 3 (ggml 0.26 and newer) is supported. If your llama.cpp is the old one, stay on v0.3.6 or move to a newer llama.cpp.

Easier to install and keep up to date: a one-line install, `moe-cache launch` as a guided start, `update --apply`, five more coding agents, a model catalogue, and a bundled official llama.cpp b12064. Decode speed on the big Qwen files that do not fit in RAM was re-measured on the v0.3.5 kit: **+42 % to +53 %** over stock llama.cpp (table below).

## Measured

**What these numbers are.** Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB RAM, Vulkan), overnight run of 2026-11-04. They were measured on the **released v0.3.5 kit with the official llama.cpp b11966**. **v0.3.6 bundles b12064 and has not been re-measured against it**: a paired run is still open, so treat these as the v0.3.5 numbers. The cleanliness of the runs is not recorded (the PC harness writes "not clean: owner work running" as fixed text at the start of every run, without checking, so that text is no evidence either way). Order stock, moe-cache, moe-cache, stock (ABBA), two runs per arm, a 13-token request; the table shows the means. Records: `benchmarks/runs/2026-11-04_pc_*`, claims `chain1103-*`, board `benchmarks/BOARD.md`.

| Model | Decode, stock tok/s | Decode, moe-cache tok/s | Change |
|---|---|---|---|
| Flash-Next IQ3_XXS (75.8 GB) | 14.96 | 22.87 | **+53 %** |
| Flash-Next IQ3_S (83.6 GB) | 13.79 | 19.62 | **+42 %** |
| Qwen3.5-122B UD-IQ2_XXS (36.6 GB) | 16.10 | 23.31 | **+45 %** |
| GLM-5.3-Flash UD-IQ2_XXS (102 GB) | 2.98 | 6.27 | **2.1x** |
| GLM-5.3-Flash with `--mtp` | 2.98 | 8.17 | 2.7x |
| Nemotron 3.5 Bolt 74B (22 GB, fits in RAM; plugin forced) | 29.99 | 31.68 | +6 % |
| Qwen3.6-35B (22 GB, fits in RAM; plugin forced) | 46.59 | 51.80 | +11 % |
| Qwen3.6-35B with MTP (against stock without MTP) | 46.59 | 72.69 | +56 % |

Models that fit in RAM: the launcher still runs plain llama.cpp for them; the two forced rows are for comparison (earlier single runs gave -18 % and -1 %, so there is no steady gain and the default stays). Flash-Next Q2_0 (+20 %) and IQ2_XS (+47 %) were not in this run; their numbers are from 2026-10-27/28.

**Long prompts, Flash-Next files (plugin only, two passes, context 65536):**

| File | Prompt tok/s, 34k | Prompt tok/s, 64k | Decode after the prompt, 34k | Decode after the prompt, 64k |
|---|---|---|---|---|
| Q2_0 | 568 | 399 | 27.4 | 25.5 |
| IQ2_XS | 504 | 317 | 24.7 | 16.9 (*) |
| IQ3_XXS | 349 | 182 | 16.1 | 12.3 (*) |
| IQ3_S | 278 | 221 | 13.4 | 11.9 |

(*) From a daytime re-run, **not clean** (the owner's browser held 1.4 GB of VRAM): the overnight run had no value here because the model ended its answer after 2 tokens, and the daytime re-run with the fixed harness generated 128 tokens. A clean re-run is planned. The prompt speeds in the table are from the overnight run, whose cleanliness is also not recorded.

**gpt-oss-120b, today's code against yesterday's** (two experimental kits, official llama.cpp b12064, 30,374-token prompt, ABBA, two runs each, cleanliness not recorded): prompt **73 -> 115 tok/s** (73.1 / 73.2 against 114.6 / 114.7; the fix keeps resident experts from being moved to the most-recently-used end during a long prompt). Decode after the prompt 8.8 -> 9.2 tok/s. Record `2026-11-04_pc_76a45c`, claim `chain1103-gptoss-623`. This compares two builds of moe-cache, not moe-cache against stock.

**Not measured:** v0.3.6 against v0.3.5 on b12064; the "about +5-7 % on long prompts" that the CHANGELOG expects from b12064 is from runs with our former patch and is not a claim here.

## New for users

| What | In plain words |
|---|---|
| One-line install | `install.sh` (Linux) and `install.ps1` (Windows PowerShell) come with each release. They download the release, check it against the release's SHA256 list (a mismatch is refused), unpack under your home folder (no admin, no root), add a `moe-cache` command, and say how to undo it (`--uninstall` / `-Uninstall`). If anything fails midway, the previous install is put back. |
| `moe-cache launch` | A guided start: checks the PC, shows 2-3 models that fit (size, speed only if we measured it), prints the download command and runs it only after your yes, starts the server, and offers to connect the coding agents it finds. |
| `moe_cache.py update --apply` | Installs the newest stable release with the release's own installer. It checks the installer's build attestation when `gh` is available, the installer checks the package, the previous version is kept (Linux: `versions/`; Windows: `kit.prev`), and the new version is checked without a model; if that check fails it goes back by itself. Only for a copy the installer put there. `--dry-run` shows the plan. |
| More coding agents | `connect` now also writes the config for Codex CLI, Pi, OpenClaw, Aider and Qwen Code (besides Hermes, OpenCode and Claude Code). The first backup of your file is kept and never replaced; a file in an unexpected format is refused. `connect NAME --check` sends one prompt and one tool call the way that agent does and says which step fails. |
| Models | One catalogue (`models.json`) feeds `pick`, `launch` and `docs/MODELS.md`. `moe_cache.py models` lists every model as fits / tight / no for this PC, with the measured speed and its source where there is one. |
| Control page | Download a model from the catalogue (progress bar, a cut download continues, sha256 checked at the end). "Start on first request" works. The agents card reads the real OpenCode and Hermes configs back. Each chat shows its expert-cache hit rate. The page no longer uses a cookie. |
| Always-on server | `moe-cache-server MODEL --print-systemd` prints a systemd user unit with your paths, options and memory limits; `docs/moe-cache-server.service` is a template. (Linux. The unit has not been run as a long-lived service by us.) |
| CPU-only mode | `--cpu-only`, and automatically when `llama-server --list-devices` lists no GPU: llama-server gets `--device none -ngl 0`. |
| Bundled llama.cpp | The Windows kit and the Linux tarball now include the official llama.cpp **b12064** (ggml 0.28.0) instead of b11966. It contains the upstream fix that was our patch 0022 and the batched tensor copy that was patch 0027, so both patches moved to `archive/`. |
| Slow-path explanations | When a long-prompt block stays on the CPU, a GPU backend cannot compute some expert weights, or a GPU allocation fails, the launcher says it once in a plain sentence. |
| Benchmarks | Every run is saved as a record; every number we publish is a claim tied to its runs and goes stale by itself when the code or the llama.cpp pin moves past it (`benchmarks/BOARD.md`). |
| Docs | Privacy statement (no telemetry, no account, the plugin opens no network connections; a table of every time the launcher or installers use the internet). New: `docs/AI_SETUP.md` (set up through a coding agent), `docs/RELEASE_CHECKLIST.md`, known-problem lines in `docs/TROUBLESHOOTING.md`, a README table of what to expect by memory budget (measured numbers only), and all 160+ `MOE_CACHE_*` settings in `docs/CONFIGURATION.md`. |

## Fixes you may have hit

| Area | Fix |
|---|---|
| Long prompts | A GPU error during a long prompt (for example another program takes the graphics memory) no longer ends the server: that block runs on the CPU with the same text. `MOE_CACHE_GPU_CATCH=0` turns this off. |
| Settings | Environment switches read the same way everywhere: empty, `0`, `off`, `false`, `no` is off; `1`, `on`, `true`, `yes` is on. Before, `MOE_CACHE_HYBRID=off` could switch a feature on. Values like `512M`, `inf` or `nan` are refused with a message. |
| Disks | A file system that accepts O_DIRECT but fails the read (encrypted, FUSE, virtiofs) switches to buffered reads after the first failure instead of costing about 60 ms per expert. A failing expert read is retried 3 times before the server stops with the disk error. |
| Connecting agents | A symlinked config stays a link; no preview prints your API key; Claude Code's backup is kept under `~/.cache/moe-cache/backups`, not next to a file that could be committed; the key no longer goes on a `hermes config set` command line; your `LLAMA_API_KEY` is no longer sent to a host that is not on this machine. |
| Proxies | Requests to the server on your own machine no longer go through a system or corporate proxy (before, health checks failed and the API key reached the proxy). |
| Control page and on-demand | An `--api-key` given to on-demand is kept in a private file, not in `on-demand.json` or on a command line. The page's chat waits up to an hour for a silent server and says so. A damaged `ui.json` is kept as `ui.json.bad`. |
| Windows | `moe-cache.exe` (the winget command) passes redirects and pipes on; `powershell` and `taskkill` are run from System32; the winget package no longer installs a second Python. |
| Downloads | `launch --download` checks the free disk first (download size + 2 GB) and resumes a cut download from its `.part` file. A downloaded model is checked against the sha256 in the catalogue. |
| Launcher | A bare `--fast` no longer reads the model path as its level; model-card sampling defaults notice `--temp=0.2` in the `=` form; the launcher starts faster (dry-run launch 0.47 s to 0.19 s on the laptop, median of 7; first launch after a llama.cpp update pays 0.3 s once). |
| Usage profiles | A usage profile of another model is no longer overwritten. |
| Others | About 40 more small fixes from the audits are in CHANGELOG.md (hardening of opt-in features, planner edge cases, Windows output with non-ASCII names, and more). |

## Built but not yet tested on real hardware (said plainly)

| What | State |
|---|---|
| `MOE_CACHE_DECODE_RELOC` (decode after long prompts; **off by default**, needs the llama.cpp patch 0070 that is not in the bundled build) | Design and tests on synthetic data; **not yet run on the PC**. The ceiling of +15 % at 64k is from the design, not a release claim. |
| Shared-memory (integrated) GPUs | The copy of the experts for a long-prompt block is now capped and counted against the cache. **Not measured on an integrated GPU.** |
| Intel SYCL | The launcher sets `ZES_ENABLE_SYSMAN=1` for SYCL devices. **Not run on an Intel SYCL machine.** |
| `connect` for Qwen Code | The config is written and checked by tests; a full turn with the real program is **not proven yet**. The other agents (Codex CLI, Pi, OpenClaw, Aider, OpenCode, Claude Code) completed a turn with a small model. |
| `update --apply` | Tested with a fake release on Linux, on a real Windows PC and in CI. **Never run against a real published release yet** (this release is the first it can update to). |
| Always-on systemd unit | Written from the code; **not run as a long-lived service**. |
| Stitched GGUF (`tools/stitch_gguf.py`) | Tested on small synthetic files only; a stitched file is not exact and needs the quality gate. |
| Desktop app (`desktop/`) | A design and a buildable skeleton; **not released**, the browser page is unchanged. |
| Hip/ROCm hint in `check` | Names missing libraries; **not run on an AMD ROCm machine**. |
| llama.cpp b12064 | Identity and Windows runs pass in CI; a paired speed run against b11966 has **not** been done (see "Measured"). |

## Known limits carried over
- Decode after a long prompt was 6-10 % lower in v0.3.5 than in v0.3.4 (a bigger prompt chunk takes GPU memory); v0.3.6 has not been measured on this.
- The benchmark card's Flash-Next Q2_0 and IQ2_XS rows are from 2026-10-27/28 and were not re-run in the overnight run.
- Windows is experimental: one desktop, AMD GPU, Vulkan.
- Not tested: macOS.

## Upgrading
- Installed with `install.sh` / `install.ps1`: `moe-cache update --apply` (or run the installer again).
- Kit or tarball: download the new one from the Releases page; verify with `gh attestation verify FILE -R IbrahimSiddiqui007/moe-cache`.
- On backend API 2 (ggml 0.25.1): stay on the `moe-cache-vX-linux-x86_64.tar.gz` file without `-ggml0.26`. **v0.3.6 is the last release for it.**

## Files
`moe-cache-v0.3.6-ggml0.26-linux-x86_64.tar.gz` (with llama.cpp b12064, Vulkan + CPU), `moe-cache-v0.3.6-linux-x86_64.tar.gz` (plugin only, backend API 2), `moe-cache-v0.3.6-windows-x64.zip`, `SHA256SUMS`, `SHA256SUMS-windows`, `install.sh`, `install.ps1`.
