#!/usr/bin/env python3
"""moe_cache.py launch: a guided start in plain words. It checks this PC, shows two or three models that fit it (from `pick`), prints the exact
download command (and runs it only after an explicit yes), starts the server or the on-demand listener, and offers `connect` for the coding
agents found on the PC. Every step reuses what exists (pick, check's facts, server, on-demand, connect); this file only asks and decides.

  moe-cache launch                         asks; Enter takes the suggestion at each question
  moe-cache launch --yes --model PATH      no questions: start that model (chat); never downloads
  moe-cache launch --yes --agent opencode  agent mode with the recommended model if it is on this PC, then connect OpenCode

Standard library only. The decisions are plain functions of the answers and of `Deps` (the machine facts, pick's recommendation, running things),
so tests/launch_test.py runs the whole walk-through on a fake machine."""
import argparse, os, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_cache as mc  # noqa: E402
import moe_cache_pick as pk  # noqa: E402

AGENTS = ("hermes", "opencode", "claude")
USE_ASK = {"1": "chat", "2": "agents", "3": "docs"}
USE_WORDS = {"chat": "plain chat", "agents": "a coding agent (64k context)", "docs": "long documents"}
GIB = 2 ** 30


class Deps:
    """Everything the walk-through touches outside itself. Tests replace these."""
    def __init__(self):
        self.ask_fn = None           # (question, default) -> answer, lower-cased and stripped; None: input()
        self.say = mc.say

    def ask(self, q, default=""):
        if self.ask_fn:
            return self.ask_fn(q, default)
        try:
            return input(q).strip().lower() or default
        except EOFError:
            return default

    def llama(self):
        return mc.find_llama_server()

    def machine(self, models_dirs, llama_path):
        return pk.machine_specs(models_dirs, llama_path)

    def recommend(self, spec, use, model=None):
        return pk.recommend(spec, use, model)

    def find_local(self, models_dirs):
        return pk.find_local(pk.model_folders(models_dirs))

    def which(self, name):
        return shutil.which(name)

    def download(self, rec):
        """Runs the download of recommendation `rec` (hf, then the link step if the model has one). Returns 0 or an error text."""
        d = rec["download"]
        m = pk.MODELS[rec["id"]]
        os.makedirs(d["folder"], exist_ok=True)
        rc = subprocess.call(["hf", "download", m["repo"], "--include", *pk.includes(m), "--local-dir", d["folder"]])
        if rc:
            return f"the download failed (exit {rc}); nothing was changed"
        if m.get("link"):
            shard, target = (os.path.join(d["folder"], x) for x in m["link"])
            try:
                if not os.path.exists(shard):
                    os.link(target, shard)
            except OSError as e:
                return f"the download is done, but the link failed ({e}); run: {d['steps'][1]}"
        return 0

    def start(self, cmd):
        return subprocess.Popen(cmd)

    def up(self, url, on_demand):
        """Is the thing at `url` answering? The server: its model is loaded (/v1/models); the on-demand listener: it accepts connections."""
        if on_demand:
            import socket
            from urllib.parse import urlparse
            u = urlparse(url)
            try:
                socket.create_connection((u.hostname, u.port), timeout=2).close()
                return True
            except OSError:
                return False
        try:
            return bool(mc.server_info(url, mc.key_for(url, None))[0])
        except Exception:
            return False

    def connect(self, argv):
        return mc.connect(argv)

    def sleep(self, s):
        time.sleep(s)


def gbs(b):
    return "?" if b is None else f"{b / 1e9:.1f} GB"


def command_word():
    """How to write moe-cache in the commands we print: the installed command when there is one, else this interpreter and file."""
    if shutil.which("moe-cache"):
        return "moe-cache"
    return f'"{sys.executable}" "{os.path.join(mc.HERE, "bin", "moe_cache.py")}"'


def this_pc_lines(spec, ls):
    """Step 1, in plain words."""
    out = [f"RAM:           {gbs(spec['ram_total'])} (the server may use up to {gbs(spec['ram_cap'])}; the rest stays for Windows/Linux and your programs)"]
    if spec.get("gpu"):
        out.append(f"graphics card: {spec['gpu']}" + (f", {spec['vram'] / GIB:.1f} GiB of its own memory" if spec.get("vram") else ""))
    else:
        out.append("graphics card: none with its own memory found (everything runs on the CPU; slower)")
    out.append(f"free disk:     {gbs(spec['disk_free'])} at {spec['disk_path']} ({spec['disk_kind'] or 'type unknown'})")
    path = ls.get("path")
    if path:
        own = os.path.abspath(path).startswith(os.path.abspath(mc.HERE) + os.sep) if os.path.isabs(path) else False
        out.append(f"llama-server:  {path} ({'the one that came with moe-cache' if own else 'found by ' + str(ls.get('source'))}"
                   + (f", build b{spec['llama_build']}" if spec.get("llama_build") else "") + ")")
    else:
        out.append("llama-server:  NOT FOUND")
    return out


def speed_text(r):
    """Expected speed only when it was measured on a PC like this one; otherwise say so."""
    if r.get("untested") or not r.get("key"):
        return "speed: not measured on a PC like this one"
    nums = ", ".join(f"{label} {v}" for label, v in r["key"] if label in ("decode", "prompt"))
    return f"speed: {nums} (measured by the maintainer on a similar PC; yours will differ)" if nums else "speed: not measured on a PC like this one"


def short(t, n=180):
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " ..."


def candidate_lines(i, r, first, why=None):
    out = [f"  {i}) {r['name']}  ({gbs(r['size'])})" + ("   <- suggested" if first else "")]
    if r["local"]:
        out.append(f"     on this PC: yes ({r['local']})")
    elif not r["fits_disk"]:
        out.append(f"     on this PC: no, and the disk is {gbs(r['disk_missing'])} too small for the download")
    else:
        out.append(f"     on this PC: no (download {gbs(r['download']['size'])} from {r['download']['source']})")
    out.append("     " + speed_text(r))
    out.append(f"     runs as: {r['engine']}, context {r['ctx']}")
    if why or r.get("why"):
        out.append(f"     why: {short(why or r['why'][0])}")
    if r.get("warnings"):
        out.append(f"     note: {short(r['warnings'][0])}")
    return out


def candidates(deps, spec, use, models_dirs):
    """The suggestion plus up to two alternatives, each a full recommendation for the same use."""
    first = deps.recommend(spec, use)
    recs, whys = [first], [None]
    for alt in first["alternatives"]:
        if len(recs) >= 3:
            break
        try:
            recs.append(deps.recommend(spec, use, alt["id"]))
            whys.append(alt["why"])
        except Exception:
            continue
    return recs, whys


def parse(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py launch", allow_abbrev=False,
                                 description="guided start: check this PC, pick a model, (download it if you say yes), start the server, connect your coding agent")
    ap.add_argument("--yes", "-y", action="store_true", help="no questions: take the suggestion at each step. Never downloads (add --download), never edits an agent's config (add --agent)")
    ap.add_argument("--model", help="a .gguf file, or a candidate name from `pick --help` (default: the suggestion for this PC)")
    ap.add_argument("--use", choices=list(pk.USES), help="chat, agents (coding agents, 64k+ context) or docs (long documents)")
    ap.add_argument("--agent", nargs="?", const="all", metavar="NAME",
                    help="start in agent mode and connect this agent (hermes, opencode or claude) after the server is up; without a name: every agent found on this PC")
    ap.add_argument("--download", action="store_true", help="the explicit yes to downloading the chosen model when it is not on this PC")
    ap.add_argument("--start", choices=["server", "on-demand", "no"], help="server: run it in this window; on-demand: a listener that loads the model on the first request; no: only print the commands")
    ap.add_argument("--port", type=int, default=None, help="the port (default 8080)")
    ap.add_argument("--models-dir", action="append", default=[], help="another folder with .gguf files (repeatable)")
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for the model to load (default 900)")
    ap.add_argument("--dry-run", action="store_true", help="show the steps and the commands; start and download nothing")
    a = ap.parse_args(argv)
    if a.agent and a.agent != "all" and a.agent not in AGENTS:
        ap.error(f"--agent: one of {', '.join(AGENTS)} (or no name for every agent found)")
    return a


def launch(argv, deps=None):
    a = parse(argv)
    deps = deps or Deps()
    say = deps.say
    interactive = not a.yes
    ask = (lambda q, d="": deps.ask(q, d)) if interactive else (lambda q, d="": d)
    say(f"moe-cache {mc.VERSION} guided start. Nothing is downloaded or changed without your yes; Ctrl+C leaves at any time.\n")

    # 1/5 this PC
    say("1/5  This PC")
    ls = deps.llama()
    spec = deps.machine(a.models_dir, ls.get("path"))
    for line in this_pc_lines(spec, ls):
        say("     " + line)
    if not ls.get("path"):
        say("\n     No llama-server was found. The moe-cache download for Windows and the ggml0.26 Linux download carry one (in the llama folder);\n"
            "     if you installed with the moe-cache installer, run it again. Or point to your own build:  set LLAMA_SERVER=C:\\path\\llama-server.exe  (Linux: export LLAMA_SERVER=...).\n"
            "     `moe-cache doctor` explains what it looked at.")
        return 1

    # what for
    use = "agents" if a.agent else a.use
    if not use:
        if interactive:
            say("\n     What will you use it for?  1) plain chat in the browser   2) a coding agent (Hermes, OpenCode, Claude Code)   3) long documents")
            use = USE_ASK.get(ask("     choose 1-3 [1]: ", "1"), "chat")
        else:
            use = "chat"
    say(f"     -> {USE_WORDS[use]}")

    # 2/5 models
    say("\n2/5  Models for this PC")
    model_path, rec = None, None
    if a.model and (os.path.isfile(a.model) or a.model.lower().endswith(".gguf")):
        if not os.path.isfile(a.model):
            say(f"     {a.model} does not exist.")
            return 1
        model_path = os.path.abspath(a.model)
        say(f"     using your file: {model_path}")
    else:
        if a.model:
            if a.model not in pk.MODELS:
                say(f"     --model: a .gguf file or one of: {', '.join(pk.MODELS)}")
                return 1
            recs, whys = [deps.recommend(spec, use, a.model)], [None]
        else:
            recs, whys = candidates(deps, spec, use, a.models_dir)
        for i, r in enumerate(recs, 1):
            for line in candidate_lines(i, r, i == 1 and not a.model, whys[i - 1]):
                say(line)
        n = 1
        if not interactive and not a.model and not recs[0]["local"]:
            here = next((i for i, r in enumerate(recs, 1) if r["local"]), None)
            if here:
                n = here
                say(f"     (no questions: taking {recs[n - 1]['name']}, which is already on this PC; the suggestion would need a download)")
        if len(recs) > 1 and interactive:
            c = ask(f"     which one? [1-{len(recs)}, Enter = 1]: ", "1")
            n = int(c) if c.isdigit() and 1 <= int(c) <= len(recs) else 1
        rec = recs[n - 1]
        say(f"     -> {rec['name']}")
        model_path = rec["local"]
        if rec.get("warnings") and interactive and len(rec["warnings"]) > 1:
            for w in rec["warnings"][1:]:
                say(f"     note: {w}")

    # 3/5 download
    if not model_path:
        say("\n3/5  Getting the model")
        if not rec["fits_disk"]:
            say(f"     The disk has {gbs(rec['disk_missing'])} too little room for {rec['name']}. Free some space or choose another model (run this again, or use --models-dir).")
            return 1
        d = rec["download"]
        say(f"     {rec['name']} is not on this PC. This is the download ({gbs(d['size'])}, from {d['source']}, into {d['folder']}):")
        for step in d["steps"]:
            say("       " + step)
        if d.get("note"):
            say("       (" + d["note"] + ")")
        if not deps.which("hf"):
            say("     The `hf` tool is not installed. Install it once with:   pip install -U huggingface_hub\n"
                "     then run the command above (or run `moe-cache launch` again and answer yes).")
            return 2
        yes = a.download or (interactive and ask(f"     Download {gbs(d['size'])} now? [y/N]: ", "n") in ("y", "yes"))
        if not yes or a.dry_run:
            say("     Not downloaded." + ("" if a.dry_run else " Run the command above yourself, or `moe-cache launch` again and answer y" + ("; scripts: add --download." if a.yes else ".")))
            return 2
        err = deps.download(rec)
        if err:
            say("     " + err)
            return 1
        found = deps.find_local(a.models_dir)
        model_path = found.get(rec["id"])
        if not model_path:
            say(f"     The download finished but the file was not found in {d['folder']}; use --model PATH.")
            return 1
        say(f"     downloaded: {model_path}")
    else:
        say("\n3/5  Getting the model: not needed, it is on this PC")

    # 4/5 start
    ctx = (rec or {}).get("ctx") or (65536 if use == "agents" else 32768)
    sargs = ["--ctx", str(ctx)] + (["--agent"] if use == "agents" else [])
    port = a.port or 8080
    start = a.start
    if not start:
        if not interactive:
            say("\n4/5  Starting it")
        if interactive:
            say("\n4/5  Starting it\n     1) start the server now, in this window (Ctrl+C stops it)\n"
                "     2) on-demand: a small listener that loads the model on the first request and lets go of the memory when idle\n"
                "     3) not now; just show me the commands")
            start = {"1": "server", "2": "on-demand", "3": "no"}.get(ask("     choose 1-3 [1]: ", "1"), "server")
        else:
            start = "server"
    else:
        say("\n4/5  Starting it")
    script = os.path.join(mc.HERE, "bin", "moe_cache.py")
    base = [sys.executable, script]
    server_cmd = base + ["server", model_path] + sargs + (["--port", str(port)] if port != 8080 else [])
    od_cmd = base + ["on-demand", model_path, "--port", str(port)]
    cw = command_word()
    shown = {"server": f'{cw} server "{model_path}" ' + " ".join(sargs) + (f" --port {port}" if port != 8080 else ""),
             "on-demand": f'{cw} on-demand "{model_path}" --port {port}'}
    url = f"http://127.0.0.1:{port}"
    if start == "no" or a.dry_run:
        say("     the commands (nothing was started):")
        say("       " + shown["server" if start != "on-demand" else "on-demand"])
        if use == "agents":
            say(f"       {cw} connect opencode --url {url} --write      (or hermes, claude; after the server is up)")
        return 0
    cmd = server_cmd if start == "server" else od_cmd
    say("     starting: " + shown[start])
    child = deps.start(cmd)
    say("     loading the model; the first start can take a few minutes (it reads the file from the drive)...")
    waited, last = 0, 0
    while not deps.up(url, start == "on-demand"):
        if child.poll() is not None:
            say(f"     it stopped with exit code {child.returncode}. `{cw} doctor \"{model_path}\"` says what is wrong.")
            return 1
        if waited >= a.wait:
            say(f"     still not answering after {a.wait} s; leaving it running. Check {url} later, or `{cw} doctor`.")
            break
        deps.sleep(2)
        waited += 2
        if waited - last >= 30:
            last = waited
            say(f"     ... still loading ({waited} s)")
    else:
        say(f"     ready: {url}  (chat page in your browser: {url})")

    # 5/5 agents
    say("\n5/5  Coding agents")
    found = [x for x in AGENTS if deps.which(x)]
    if not found:
        say("     none of hermes, opencode, claude found on this PC" + (f" ({a.agent} is not installed; skipped)" if a.agent not in (None, "all") else "")
            + f" (install one, then: {cw} connect NAME --write).")
    elif use != "agents":
        say(f"     found: {', '.join(found)}. Agents need agent mode: run  {cw} launch --agent  to start that way.")
    else:
        say(f"     found: {', '.join(found)}")
        want = []
        for x in found:
            if a.agent and a.agent != "all":
                if x == a.agent:
                    want.append(x)
                continue
            if a.agent == "all" or (interactive and ask(f"     connect {x} to this server (edits its config; the old one is kept as .bak)? [y/N]: ", "n") in ("y", "yes")):
                want.append(x)
        if a.agent and a.agent != "all" and a.agent not in found:
            say(f"     {a.agent} is not installed on this PC; skipped.")
        for x in want:
            try:
                deps.connect([x, "--url", url, "--write"])
            except SystemExit as e:
                say(f"     could not connect {x}: {e}")
        if not want:
            say(f"     not connected. Later:  {cw} connect NAME --write")

    say(f"\nRunning at {url}. Ctrl+C stops it." if child.poll() is None else "")
    try:
        return child.wait() or 0
    except KeyboardInterrupt:
        child.terminate()
        try:
            child.wait(timeout=20)
        except Exception:
            child.kill()
        say("stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(launch(sys.argv[1:]))
