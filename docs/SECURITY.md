# Security and privacy notes

moe-cache is a plugin that runs inside `llama-server` plus a few helper scripts. This page says what it touches and what it keeps.

## Privacy: what moe-cache sends and stores

**No telemetry. No account. The plugin opens no network sockets.** (Checked: nothing in `src/` mentions a socket, a connect call, curl, an address lookup or a URL.) The launcher and the tools never send your prompts, answers, model names, paths or hardware details anywhere. The only things that listen are the local ones in the next section, on `127.0.0.1` unless you pass a different `--host`.

moe-cache reaches the internet only in the places below, and only when you run that command:

| When | What is contacted | What is sent | How to avoid it |
|---|---|---|---|
| `moe_cache.py update --check` | `api.github.com`: the release lists of IbrahimSiddiqui007/moe-cache and ggml-org/llama.cpp (two GET requests) | the request itself (your IP address, the header `User-Agent: moe-cache/VERSION`). Nothing is downloaded or installed | do not run it. No other command checks for updates |
| `pick` or `launch`, when you say yes to the model download (`y` at the question, or `launch --download`) | `huggingface.co`, through the `hf` tool (`pip install -U huggingface_hub`), which does the transfer | what `hf` sends: the repository and file names of the model. moe-cache itself makes no request. Without your yes, both only *print* the command (`pick` also prints it when it has no terminal) | answer no and run the printed command yourself, or download the `.gguf` another way and use `--model FILE` |
| One-line install, Windows: `install.ps1` | `github.com`: the release zip and `SHA256SUMS-windows` (served from GitHub's download hosts). `api.github.com` as well, only when the script is run from a checkout without `-Version`, to find the newest release | the requests (your IP address; `User-Agent: moe-cache-install` on the API call). Nothing about your machine | download the zip and the sums file by hand; `-DryRun` shows the plan without a request for the zip. Nothing is sent after the install |
| One-line install, Linux: `install.sh` | `github.com`: the release tarball and `SHA256SUMS`, with `curl` or `wget` | the request (your IP address). Nothing about your machine | download the tarball yourself and run `install.sh` from inside it; `--dry-run` prints the plan |
| `install.sh --build`, or `install.sh` run from a checkout (builds llama.cpp) | `github.com`: git clone of moe-cache at the tag and a fetch of one llama.cpp commit (ggml-org/llama.cpp); PyPI for the python `gguf` package (pip); your distribution's `apt` for the build tools (asks for sudo; `--no-apt` skips) | what git, pip and apt send in the ordinary way | the release install needs none of this |
| Links in the control page ("Official llama.cpp releases", the issue link after `bench-report`) | nothing until you click; your browser then opens `github.com` | what your browser sends | do not click. `bench-report` writes a local `.md` file and contacts nothing: you paste it into an issue yourself |
| `tools/check_model_links.py` (maintainers) | `huggingface.co/api`: file lists of the catalogue's repositories (metadata only) | the repository names | not part of an install; do not run it |
| CI and release scripts (`.github/workflows`, `tools/bundle_llama_linux.sh`, `tools/make_windows_kit.sh`) | GitHub, Hugging Face (a tiny test model), the Vulkan SDK site | CI only; none of it runs on your PC | not part of an install |

Everything else is local: the control page, the on-demand listener and the bench and agent tools talk to `127.0.0.1` only. `connect NAME --write` puts the local server address and your API key into the agent's own config file and sends nothing; what that agent then sends to others is the agent's business, not moe-cache's.

What is stored on your PC, and where, is in [What is written to disk](#what-is-written-to-disk) below. Removing `~/.cache/moe-cache/` (on Windows the same path under your user profile) removes it.

## What runs and what listens
| Component | Network | Notes |
|---|---|---|
| Plugin (`libggml-moe-cache.so`) | none | no sockets, no HTTP; reads your model file and a few small files named below |
| `moe-cache-server` | the `llama-server` it starts listens on `127.0.0.1` by default (a user `--host` can expose it) | a non-loopback `--host` without `--api-key` prints a warning. Agent mode (`--agent`) and `--project` always use an API key (stable random key in `~/.cache/moe-cache/api.key`, mode 0600, passed by environment, `moe_cache.py key` prints it) because they keep conversations on disk and llama-server lets any web page call a local server by default (CORS echo, no Host check). Every server the launcher starts also gets `LLAMA_ARG_CORS_ORIGINS=localhost`, so normal websites cannot read answers; this does not stop DNS rebinding, only a key does. A user-given `--api-key` is moved from the command line to the environment (not visible in the process list) |
| Control page (`moe-cache-server --ui`, `moe-cache.bat`) | `127.0.0.1:8070` only (hard-wired, `--ui-port` changes the port) | every call needs a per-run token in the page's address (`?t=...`); the page starts and stops servers, reads the API key for its Copy button and writes the files listed below |
| On-demand listener (`moe_cache.py on-demand`) | `127.0.0.1` on the agent port (default 8080) | checks the API key; answers `/health`, `/v1/models` and `/props` without one; starts the model server on the first request |
| `moe-cache-bench` (older external GUI, now in [moe-cache-benchmark](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark)) | `127.0.0.1:8765` only | every API call needs a per-run token that is embedded in the page the server serves; Host, Origin and Sec-Fetch-Site are checked; run settings are validated (program name, library name, `MOE_CACHE_*` variables only, no file-writing or network flags); the servers it starts use a key file |
| `install.sh` | git and pip over HTTPS | llama.cpp is pinned to a commit hash, the python `gguf` package to a version |

Releases from v0.1.3 on are built by GitHub Actions from the tagged commit and carry a build attestation: `gh attestation verify <tarball> -R IbrahimSiddiqui007/moe-cache --signer-workflow IbrahimSiddiqui007/moe-cache/.github/workflows/release.yml` (the installers and `update --apply` call it with that signer pinned). `--yes-unverified` only covers a check that could not run (gh missing, not logged in, GitHub unreachable); a verification that ran and FAILED always stops the install. Only tag builds are attested (from v0.3.0 on); to be strict, also check the tag: `gh attestation verify <tarball> -R IbrahimSiddiqui007/moe-cache --source-ref refs/tags/vX.Y.Z`.

## What is written to disk
| File | Where | Contains | Mode |
|---|---|---|---|
| Usage profile `*.usage` | `~/.cache/moe-cache/` | counts of which experts fired (a fingerprint of what you run), a model fingerprint; no text, no paths | 0600 |
| Routing trace (only with `MOE_CACHE_TRACE_IDS`) | the path you give | expert ids per decode step | 0600 |
| Benchmark results, logs, token file (the older external benchmark GUI) | `~/.local/share/moe-cache-bench/` | generated text, token ids, your custom prompts, local paths | 0700 / 0600 |
| API key | `~/.cache/moe-cache/api.key` | the stable random key for agent mode and `--project`; the control page can show it | 0600 |
| Control-page settings and benchmark table | `~/.cache/moe-cache/ui.json`, `bench.csv` (+ `bench.csv.bak` after a format upgrade) | model folders, the options you chose, an API key if you typed one on the page, measured speeds | user default |
| Control-page chats | `~/.cache/moe-cache/ui-chats/` | your chat texts as JSON | user default |
| Saved agent conversations | `~/.cache/moe-cache/slots/agent/` | llama-server slot files (the KV state of your conversations, i.e. your prompts in model form); `--no-warm` turns saving off | user default |
| On-demand choice and logs | `~/.cache/moe-cache/on-demand.json`, `on-demand.log`, `on-demand.pid`, `on-demand-listener.log` | the remembered model and options; server output | user default |
| Agent configuration written by `connect <agent> --write` | the agent's own config file (table below) | the local server address and **the API key** | user default |

Where each agent's file and its first backup live (checked in `bin/moe_cache.py`, `connect`). The backup is made only when the file already existed, and the first one is never replaced, so it holds your own setup and whatever keys you had:

| Agent | File that is written | First backup of your previous file |
|---|---|---|
| Claude Code | `./.claude/settings.local.json` of the current folder | `~/.cache/moe-cache/backups/<12 hex of the file's path>-settings.local.json.bak` (outside the project on purpose, so a blanket `git add` cannot commit it) |
| OpenCode | `~/.config/opencode/opencode.json` (`$XDG_CONFIG_HOME` honoured) | `opencode.json.bak` next to it |
| Hermes | the file `hermes config path` prints (changed through `hermes config set`) | `config.yaml.bak` next to it; if that exists, `config.yaml.bak.YYYYmmdd-HHMMSS`. No new backup when the file is only our own earlier setting |
| Codex CLI | `~/.codex/config.toml` (`$CODEX_HOME` honoured) | `config.toml.bak` next to it |
| Pi | `~/.pi/agent/models.json` and `settings.json` (`$PI_CODING_AGENT_DIR` honoured) | `models.json.bak` and `settings.json.bak` next to them |
| OpenClaw | `~/.openclaw/openclaw.json` | `openclaw.json.bak` next to it |
| Qwen Code | `~/.qwen/settings.json` | `settings.json.bak` next to it |
| Aider | `~/.aider.moe-cache.conf.yml` and `.aider.moe-cache.model.metadata.json` (its own files; your `.aider.conf.yml` is not touched) | `.bak` next to each, only if an older copy of our file exists |

With a symlinked config the `.bak` sits beside the real file the link points to.

Delete a row's files to remove that data; the cache folder (`~/.cache/moe-cache/`, or `$XDG_CACHE_HOME/moe-cache`) can be removed whole. Backups of your home folder will contain these files too. On Windows the cache folder is the same relative path under your user profile.

The shareable report and the default JSON export contain none of the prompts, outputs, token ids or absolute paths; the full versions (`?full=1`) do, so look before you share them.

## Known limits
- The plugin trusts the GGUF header of the model you load (llama.cpp does too). A hostile model file is a risk for llama.cpp itself, not only for this plugin.
- Most testing is on Linux x86-64 with an NVIDIA GPU (CUDA and Vulkan) and CPU-only builds; Windows 11 with an AMD GPU is tested on one desktop (identity and speed, [COMPATIBILITY.md](COMPATIBILITY.md)) and in CI.
- On a machine shared with other users, files in `/tmp` and other users' processes are outside what this project protects.

## Reporting
Open a GitHub issue without exploit details, or use GitHub's private vulnerability reporting for this repository.
