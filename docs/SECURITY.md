# Security and privacy notes

moe-cache is a plugin that runs inside `llama-server` plus a few helper scripts. This page says what it touches and what it keeps.

## What runs and what listens
| Component | Network | Notes |
|---|---|---|
| Plugin (`libggml-moe-cache.so`) | none | no sockets, no HTTP; reads your model file and a few small files named below |
| `moe-cache-server` | the `llama-server` it starts listens on `127.0.0.1` only | a non-loopback `--host` without `--api-key` prints a warning. Agent mode (`--agent`) and `--project` always use an API key (stable random key in `~/.cache/moe-cache/api.key`, mode 0600, passed by environment, `moe_cache.py key` prints it) because they keep conversations on disk and llama-server lets any web page call a local server by default (CORS echo, no Host check). Every server the launcher starts also gets `LLAMA_ARG_CORS_ORIGINS=localhost`, so normal websites cannot read answers; this does not stop DNS rebinding, only a key does. A user-given `--api-key` is moved from the command line to the environment (not visible in the process list) |
| `moe-cache-bench` (GUI, now in [moe-cache-benchmark](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark)) | `127.0.0.1:8765` only | every API call needs a per-run token that is embedded in the page the server serves; Host, Origin and Sec-Fetch-Site are checked; run settings are validated (program name, library name, `MOE_CACHE_*` variables only, no file-writing or network flags); the servers it starts use a key file |
| `install.sh` | git and pip over HTTPS | llama.cpp is pinned to a commit hash, the python `gguf` package to a version |

Releases from v0.1.3 on are built by GitHub Actions from the tagged commit and carry a build attestation: `gh attestation verify <tarball> -R IbrahimSiddiqui007/moe-cache`. Only tag builds are attested (from v0.0.0.TBD on); to be strict, also check the tag: `gh attestation verify <tarball> -R IbrahimSiddiqui007/moe-cache --source-ref refs/tags/vX.Y.Z`.

## What is written to disk
| File | Where | Contains | Mode |
|---|---|---|---|
| Usage profile `*.usage` | `~/.cache/moe-cache/` | counts of which experts fired (a fingerprint of what you run), a model fingerprint; no text, no paths | 0600 |
| Routing trace (only with `MOE_CACHE_TRACE_IDS`) | the path you give | expert ids per decode step | 0600 |
| Benchmark results, logs, token file | `~/.local/share/moe-cache-bench/` | generated text, token ids, your custom prompts, local paths | 0700 / 0600 |

The shareable report and the default JSON export contain none of the prompts, outputs, token ids or absolute paths; the full versions (`?full=1`) do, so look before you share them.

## Known limits
- The plugin trusts the GGUF header of the model you load (llama.cpp does too). A hostile model file is a risk for llama.cpp itself, not only for this plugin.
- Only Linux on x86-64 with an NVIDIA GPU (CUDA and Vulkan) and CPU-only builds are tested; Windows code exists but never ran on real Windows.
- On a machine shared with other users, files in `/tmp` and other users' processes are outside what this project protects.

## Reporting
Open a GitHub issue without exploit details, or use GitHub's private vulnerability reporting for this repository.
