# Set up moe-cache with a coding agent

For an AI coding agent (and the person who asked it) that has to install and run moe-cache on this PC. Plain steps, exact commands, what to expect. Nothing here needs administrator rights or root. Background: [INSTALL.md](INSTALL.md), [QUICKSTART.md](QUICKSTART.md), [privacy](SECURITY.md#privacy-what-moe-cache-sends-and-stores).

## Rules for the agent

1. **Never download a model without asking the user.** Tell them the name, the size and the folder, then wait for a yes. The installer in step 1 also downloads (the release: the plugin and the official llama.cpp): say so before you run it.
2. **Never change the user's agent configuration without asking** (`connect NAME --write` edits a file such as `~/.config/opencode/opencode.json` or `.claude/settings.local.json` and writes the API key into it). Without `--write` the command only prints what it would do. The old file is kept as `.bak`.
3. **Report the output of step 2 (`check`) to the user as it is.** Do not summarise a `[FAIL]` away. If a step fails, show the message and stop; do not try other commands on your own.
4. Use only the commands on this page. Output marked "human-readable" is plain text: read it, do not parse it. Only `pick --json` prints JSON.
5. Do not make up speeds. The picker says "not measured" when it has no number for this kind of PC; pass that on.

`moe-cache` is the command the installer creates. In a source checkout use `python3 bin/moe_cache.py` (Linux) or `python bin\moe_cache.py` (Windows) in its place; the arguments are the same.

## Steps

### 1. Install

Windows (PowerShell; needs Python 3.8 or newer, it never installs Python itself):

```powershell
irm https://github.com/IbrahimSiddiqui007/moe-cache/releases/latest/download/install.ps1 | iex
```

Linux x86-64 (downloads, then runs, so the script can be read in between):

```bash
curl -fsSL https://github.com/IbrahimSiddiqui007/moe-cache/releases/latest/download/install.sh -o install.sh && sh install.sh
```

Expect: a line `moe-cache X.Y.Z ...` naming the folder it installs into, a SHA-256 check of the download (it refuses and installs nothing on a mismatch), and at the end where the `moe-cache` command went. Windows: **open a new terminal** afterwards, the PATH entry is read at start. If `releases/latest` answers 404 the release is still marked pre-release: use the versioned address from [INSTALL.md](INSTALL.md#one-line-to-install-one-command-to-start).

Check: `moe-cache --version` prints `moe-cache X.Y.Z`.

### 2. Check this PC

```bash
moe-cache check
```

Expect human-readable text: one line per test starting with `[ OK ]`, `[WARN]` or `[FAIL]` (system, cpu, llama-server, plugin, memory, swap, ...), then "this machine at a glance" and a last line `summary: N failed, M warnings`. **Show this output to the user.** Continue only with 0 failed. `check MODEL.gguf` also checks a model file; `check --run` additionally starts a tiny server for about a minute (ask first).

### 3. Choose a model

```bash
moe-cache pick --use agents --json
```

(`--use chat` for plain chat, `--use docs` for long documents.) Expect one JSON object with `machine` (RAM, GPU, VRAM, free disk, which known models are already on the PC under `local`) and `recommendations` (a list: the first is the suggestion; each has `id`, `name`, `size` in bytes, `ctx`, `args`, `measured`, `why`, `warnings`, `alternatives`, and `download` with a `command` when the model is not on the PC yet). Without `--json` the same is printed as text, and `pick --apply` remembers the choice so that `server` and `on-demand` need no file name.

Tell the user the suggestion, its size, what `measured` says, and the `warnings`. If a model is already on the PC (`local`), use that file and skip step 4.

### 4. Download (only after the user said yes)

Run exactly the command `pick` printed (field `download.command`). It looks like this, and needs the `hf` tool once (`pip install -U huggingface_hub`):

```bash
hf download REPO --include "FILE.gguf" --local-dir "FOLDER"
```

Expect `hf`'s progress bars, then the file in `FOLDER`. Large models take a long time; do not start the server before it has finished. The user can also download by any other means and give you the `.gguf` path.

### 5. Start the server

Choose one. Both open the server on `127.0.0.1:8080` with an API key (`moe-cache key` prints it).

Stays in the foreground, so an agent must run it as a background process or in a second terminal; Ctrl+C stops it:

```bash
moe-cache server MODEL.gguf --agent
```

Or on-demand, a small listener that loads the model on the first request and lets go of the memory when idle (also stays in the foreground; `--install-autostart` makes it start at login instead):

```bash
moe-cache on-demand MODEL.gguf
```

Expect, for `server`: the plan (RAM, GPU layers, context) and, once the model is loaded, the base URL `http://127.0.0.1:8080/v1` with config snippets for the agents; loading a large model takes minutes. For `on-demand`: a line saying it listens on `http://127.0.0.1:8080/v1` and that the model starts on the first request. `moe-cache on-demand --status` (from another terminal) says whether it runs. Add `--dry-run` to `server` to see the command without starting it.

A third way for a person, not for an agent: `moe-cache launch` asks the five questions of INSTALL.md. For scripts: `moe-cache launch --yes --model MODEL.gguf --agent opencode` (never downloads without `--download`, never edits a config without `--agent`).

### 6. Connect the coding agent (ask first)

Agents: `hermes`, `opencode`, `claude`, `codex`, `pi`, `openclaw`, `aider`, `qwen` (`moe-cache connect --help` lists the current ones). Look first, change nothing:

```bash
moe-cache connect opencode
```

Expect the settings it would write (server address, model name, key). After the user agrees:

```bash
moe-cache connect opencode --write
```

Expect a line naming the file it changed; the previous file is kept next to it as `.bak`. `claude` writes only `.claude/settings.local.json` of the current folder. With a server on another port: `--url http://127.0.0.1:PORT`. Start the agent afterwards; it uses the local model.

### 7. Check that it works

The server must answer before the agent can use it. Linux:

```bash
curl -s http://127.0.0.1:8080/health
curl -s -H "Authorization: Bearer $(moe-cache key)" http://127.0.0.1:8080/v1/models
```

Windows (PowerShell):

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/v1/models -Headers @{ Authorization = "Bearer $(moe-cache key)" }
```

Expect `/health` to answer with a short status (once the model is loaded; an on-demand listener answers it before that) and `/v1/models` to list the model's name. A failed connection means the server is not (yet) up: look at its terminal, or `moe-cache on-demand --status`. Then ask the agent a small task; the first answer after an on-demand start waits for the model to load.

### 8. Stop

* `server` in the foreground: Ctrl+C in its terminal.
* On-demand listener (and its model): `moe-cache on-demand --stop`. At-login start, if the user had set it: `moe-cache on-demand --remove-autostart`.
* An agent connection: restore the `.bak` file `connect --write` named, or delete the `moe-cache` entry from that config.

### 9. Uninstall

Removes exactly what the installer added; models and `~/.cache/moe-cache` (settings, API key, saved conversations) stay unless the user also deletes that folder.

Windows:

```powershell
& ([scriptblock]::Create((irm https://github.com/IbrahimSiddiqui007/moe-cache/releases/latest/download/install.ps1))) -Uninstall
```

Linux:

```bash
sh install.sh --uninstall
```

(`install.sh` is the file from step 1; fetch it again if it was removed.)
