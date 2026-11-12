#!/usr/bin/env python3
"""TEST-1142: `moe_cache.py bench-suite run`: measure one or more models with the same suite and write one private suite record each (bin/moe_cache_suite.py), then print the comparison.
  moe_cache.py bench-suite run MODEL.gguf [MODEL2.gguf ...] [--profile quick|standard|night] [--out DIR] [--launch-args "--ctx 70000 ..."] [--port 8140] [--windows FILE] [--fresh]
  moe_cache.py bench-suite run --url http://127.0.0.1:8080 --name NAME [--quant Q] [--profile ...]      (measure a server that is already running; no launch, no memory, no quality section)
  moe_cache.py bench-suite compare R1.json R2.json ... | validate | show | new | publish      (see bin/moe_cache_suite.py)
Sections by profile: quick = speed probe at 8k + 20 tasks (thinking off); standard = real 32k conversation (speed + memory) + 202 tasks (off and on) + KL of the SERVED KV cache against f16 KV (8 windows);
night = standard with a 65k conversation + the hard set (no cap, 1 h per task). The model is started through the launcher (the plan a user gets) on its own free port, one model at a time; what the server
really ran (context, slots, KV types, build, command line) is read back from /props and the process and recorded in the record's `served` block; two records are only compared on what they share
(bin/moe_cache_suite.py compare). Every finished section is saved at once (a `.partial` file next to the records), so a failure or a cancel keeps the hours already measured and the next run resumes.
Records go to benchmarks/suites/ (git-ignored: private until `publish`)."""
import argparse, hashlib, json, os, re, shlex, signal, socket, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, HERE)
import moe_cache_suite as S      # noqa: E402
from moe_cache_wintool import win_tool      # noqa: E402     (Windows helper programs by their System32 path)

# ctx = the context the launcher is asked for (a conversation fills the context, so it must be at least the depth; the hard set has no cap, the context is its only limit)
PROFILES = {"quick": {"speed": "probe", "depth": 8192, "ctx": 12288, "tasks_n": 20, "think": ["off"], "kl_chunks": 0, "hard": False},
            "standard": {"speed": "conv", "depth": 32768, "ctx": 32768, "tasks_n": 202, "think": ["off", "on"], "kl_chunks": 8, "hard": False},
            "night": {"speed": "conv", "depth": 65536, "ctx": 65536, "tasks_n": 202, "think": ["off", "on"], "kl_chunks": 8, "hard": True}}
HARD_CAP = 0
HARD_TASK_SECONDS = 3600
KL_CTX = 8192                    # llama-perplexity window; it scores the second half of each window
PLAN_FLAGS_WITH_VALUE = ("-ngl", "--n-gpu-layers", "-ncmoe", "--n-cpu-moe", "-ot", "--override-tensor", "-t", "--threads", "-ts", "--tensor-split", "-sm", "--split-mode", "-mg", "--main-gpu", "-dev", "--device")
PLAN_FLAGS_SOLO = ("--cpu-moe", "-cmoe", "--no-mmap", "--mlock")


def http(url, path, body=None, timeout=3600):
    req = urllib.request.Request(url.rstrip("/") + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ---- inputs --------------------------------------------------------------------------------------------------------------------------------------
def _filler_files():
    fs = ["CHANGELOG.md", "README.md"] + sorted(os.path.join("docs", f) for f in os.listdir(os.path.join(ROOT, "docs")) if f.endswith(".md"))
    return [f for f in fs if os.path.isfile(os.path.join(ROOT, f))]


def filler_text():
    """Real text that ships with the repo (the CHANGELOG, the README and the docs), in a fixed order: the same prompt for every model on the same checkout."""
    return "\n\n".join(open(os.path.join(ROOT, p), encoding="utf-8").read() for p in _filler_files())


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def chat_windows_text(target_chars=420_000):
    """Chat-STYLE text for the KL windows, deterministic: user turns made by tools/conversation_bench.py (code to review, tool output, follow-ups) answered with prose from the repo's docs, written as
    'User: ... / Assistant: ...' turns. It is not a real chat (real chats are private); --windows FILE uses your own file instead, and its hash is part of the quality protocol."""
    sys.path.insert(0, TOOLS); import conversation_bench as CB
    prose = [p.strip() for p in filler_text().split("\n\n") if 200 < len(p.strip()) < 1500 and not p.lstrip().startswith(("|", "```", "#", "-", "*"))]
    out, n, i = [], 0, 0
    while n < target_chars and prose:
        u = CB.make_turn(7, i)["text"]; a = " ".join(prose[(i * 3 + j) % len(prose)] for j in range(2)); t = f"User: {u}\n\nAssistant: {a}\n\n"; out.append(t); n += len(t); i += 1
    return "".join(out)


# ---- what the server really ran ----------------------------------------------------------------------------------------------------------------
def proc_cmdline(pid):
    """The argv of a process (Linux /proc; Windows through PowerShell), or None."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return [x for x in f.read().decode("utf-8", "replace").split("\0") if x]
    except OSError:
        pass
    if os.name == "nt":
        try:
            out = subprocess.run([win_tool("powershell"), "-NoProfile", "-Command", f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"], capture_output=True, text=True, timeout=30).stdout.strip()
            return [x.strip('"') for x in shlex.split(out, posix=False)] if out else None
        except Exception:
            return None
    return None


def find_server(root_pid):
    """(pid, argv) of the llama-server in the process tree of root_pid (the launcher), or (None, None)."""
    import moe_cache_record as rr
    for pid in rr.probe().tree(root_pid):
        argv = proc_cmdline(pid)
        if argv and os.path.basename(argv[0]).lower().replace(".exe", "") == "llama-server":
            return pid, argv
    return None, None


def _flag_value(argv, *names):
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "=") and n.startswith("--"):
                return a.split("=", 1)[1]
    return None


def served_facts(url, root_pid=None):
    """What the server runs, read from /props and (when we started it) from its command line: ctx, slots, build, model file, KV types ('f16' when the command line names none), plan flags."""
    props = http(url, "/props", timeout=30); dgs = props.get("default_generation_settings") or {}
    f = {"ctx": dgs.get("n_ctx") or props.get("n_ctx"), "np": props.get("total_slots"), "build": props.get("build_info"), "model_path": props.get("model_path")}
    pid, argv = find_server(root_pid) if root_pid else (None, None)
    if argv:
        f.update(server_pid=pid, cmdline=" ".join(argv)[:3000], ctk=_flag_value(argv, "-ctk", "--cache-type-k") or "f16", ctv=_flag_value(argv, "-ctv", "--cache-type-v") or "f16",
                 ub=_flag_value(argv, "-ub", "--ubatch-size"), ngl=_flag_value(argv, "-ngl", "--n-gpu-layers"), n_cpu_moe=_flag_value(argv, "-ncmoe", "--n-cpu-moe"), llama_server=argv[0])
        f["plan_flags"] = plan_flags(argv)
    return f


def plan_flags(argv):
    """The flags of the served plan that decide where the layers and experts sit, in the form llama-perplexity takes (no KV types: the caller sets those)."""
    out = []
    for i, a in enumerate(argv):
        base = a.split("=", 1)[0]
        if base in PLAN_FLAGS_WITH_VALUE:
            out += [a] if "=" in a else ([a, argv[i + 1]] if i + 1 < len(argv) else [])
        elif a in PLAN_FLAGS_SOLO:
            out.append(a)
    return out


def machine_facts(llama_server=None):
    import moe_cache_record as rr, shutil
    m = {}
    try:
        pr = rr.probe(); m.update(pr.hardware(None))
        ls = llama_server or shutil.which("llama-server")
        if ls:
            r = subprocess.run([ls, "--list-devices"], capture_output=True, text=True, timeout=60); m.update(pr.gpu(r.stdout + r.stderr))
    except Exception as ex:
        m["facts_error"] = f"{type(ex).__name__}: {ex}"[:200]
    m = {k: v for k, v in m.items() if v not in (None, "not recorded")}
    m.setdefault("os", sys.platform)
    return m


def software_facts(served):
    import moe_cache_record as rr
    sw = {"kit": None, "llama_build": served.get("build"), "python": sys.version.split()[0]}
    try:
        sw["kit"] = rr.git_commit(ROOT)
        m = re.search(r'^VERSION\s*=\s*"([^"]+)"', open(os.path.join(HERE, "moe_cache.py"), encoding="utf-8").read(), re.M); sw["moe_cache_version"] = m.group(1) if m else None
    except Exception:
        pass
    return {k: v for k, v in sw.items() if v}


# ---- the sections --------------------------------------------------------------------------------------------------------------------------------
def run_speed_probe(url, depth, gen=128, ctx=None):
    """One request with a prompt of about `depth` tokens (cache off): prompt and decode tok/s. A probe, not a conversation: the record says so in its protocol. The depth is the tokenizer's own count of
    the prompt actually sent."""
    txt = filler_text(); n = len(http(url, "/tokenize", {"content": txt[:200000]})["tokens"]); cpt = 200000 / max(1, n)
    body = txt[:int(depth * cpt)]; real = len(http(url, "/tokenize", {"content": body})["tokens"])
    r = http(url, "/completion", {"prompt": body, "n_predict": gen, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False}); t = r.get("timings", {})
    r1 = http(url, "/completion", {"prompt": txt[:int(1000 * cpt)], "n_predict": 1, "temperature": 0, "cache_prompt": False}); t1 = r1.get("timings", {})
    ttft = (t1.get("prompt_ms") or 0) + (t1.get("predicted_ms") or 0)
    return {"protocol": S.protocol_hash(kind="speed-probe", depth=depth, gen=gen, filler=text_hash(txt)), "bands": [{"depth": depth, "tokens": real, "prompt_tps": round(t.get("prompt_per_second", 0), 1), "decode_tps": round(t.get("predicted_per_second", 0), 2)}],
            "ttft_ms": round(ttft), "decode_b1": round(t.get("predicted_per_second", 0), 2), "runs": 1}


def sections_from_conversation(run, ctx):
    """speed + memory sections from a tools/conversation_bench.py run dict (a real multi-turn conversation filled to ctx; memory from its `monitor` block, which exists when --monitor-pid was given)."""
    bands = [{"depth": b["hi"], "prompt_tps": b.get("prompt_tps"), "decode_tps": b.get("decode_tps")} for b in run.get("bands", []) if b.get("decode_tps") is not None]
    sp = {"protocol": S.protocol_hash(kind="speed-conv", ctx=ctx, seed=run.get("seed")), "bands": bands, "final_depth": run.get("final_depth"), "stop_reason": run.get("stop_reason"), "runs": 1}
    t = [b.get("ttft_median_s") for b in run.get("bands", []) if b.get("ttft_median_s") is not None]
    if t:
        sp["ttft_ms"] = round(t[0] * 1000)
    mem = None; mon = run.get("monitor") or {}
    if mon.get("process_ram_peak_mib") or mon.get("vram_used_peak_mib"):
        mem = {"protocol": S.protocol_hash(kind="memory", ctx=ctx), "runs": 1}
        if mon.get("process_ram_peak_mib"):
            mem["rss_gb"] = round(mon["process_ram_peak_mib"] / 1024, 2)
        if mon.get("vram_used_peak_mib"):
            mem["vram_gb"] = round(mon["vram_used_peak_mib"] / 1024, 2)
        if mon.get("vram_source"):
            mem["vram_source"] = str(mon["vram_source"])[:120]
    return sp, mem


def _py(script, *args, timeout=None):
    r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, script), *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed: {(r.stdout + r.stderr)[-400:]}")
    return r.stdout


def task_subset(tasks, n):
    """n tasks spread evenly over the whole (category-grouped) file; all of them when n >= len."""
    return list(tasks) if n >= len(tasks) else [tasks[int(i * len(tasks) / n)] for i in range(n)]


def run_tasks(url, think, n, tmp):
    tf = os.path.join(TOOLS, "quality_tasks.json"); sub = task_subset(json.load(open(tf, encoding="utf-8")), n)
    sf = os.path.join(tmp, f"tasks_{think}.json"); json.dump(sub, open(sf, "w"))
    out = os.path.join(tmp, f"tasks_{think}_run.json"); cap = 2048 if think == "on" else 64
    _py("quality_yardstick.py", "run", "--url", url, "--out", out, "--label", f"suite-{think}", "--tasks", sf, "--tasks-only", "--task-n-predict", str(cap), *(["--no-think"] if think == "off" else []))
    sec = S.section_tasks(json.load(open(out)), think, S.file_hash(sf), cap)
    sec["protocol"] = S.protocol_hash(kind="tasks", tasks=S.file_hash(sf), think=think, cap=cap, temp=0, n=len(sub))
    return sec


def run_hard(url, cap, workers, tmp, ctx):
    out = os.path.join(tmp, "hard_run.json")
    _py("hard_tasks_run.py", "run", "--url", url, "--out", out, "--label", "suite", "--cap", str(cap), "--workers", str(workers), "--resume", "--task-seconds", str(HARD_TASK_SECONDS))
    sec = S.section_hard(json.load(open(out)), S.file_hash(os.path.join(TOOLS, "hard_tasks.json")), ctx)
    sec["task_seconds"] = HARD_TASK_SECONDS; sec["workers"] = workers
    return sec


def run_quality(model, served, chunks, tmp, perplexity="llama-perplexity", windows=None):
    """KL and same-top-token of the SERVED KV cache (served['ctk'] / ['ctv'], the plan flags the server ran with) against the same model with f16 KV, on chat-style windows. Run after the server has stopped
    (the model is not held twice). Returns None when the served KV cache is f16 (identical to the reference by definition: nothing to measure)."""
    ctk, ctv = served.get("ctk"), served.get("ctv")
    if not ctk or not ctv:
        raise RuntimeError("the served KV types are not known (the server's command line could not be read)")
    if ctk == "f16" and ctv == "f16":
        return None
    wpath = windows or os.path.join(tmp, "windows.txt")
    if not windows:
        open(wpath, "w", encoding="utf-8").write(chat_windows_text())
    base = os.path.join(tmp, "base.kld"); common = ["-m", model, "-f", wpath, "-c", str(KL_CTX), "-fa", "on", "--chunks", str(chunks), *(served.get("plan_flags") or [])]
    if "-ngl" not in common and "--n-gpu-layers" not in common:
        common += ["-ngl", "99"]
    subprocess.run([perplexity, *common, "-ctk", "f16", "-ctv", "f16", "--save-all-logits", base], capture_output=True, check=True)
    r = subprocess.run([perplexity, *common, "-ctk", ctk, "-ctv", ctv, "--kl-divergence-base", base, "--kl-divergence"], capture_output=True, text=True, check=True)
    kv = f"{ctk}/{ctv}"
    sec = S.section_quality(r.stdout + r.stderr, f"same model, f16 KV (served KV {kv})", S.file_hash(wpath), chunks, chunks * KL_CTX // 2 - chunks, kv)
    sec.update(window_tokens=chunks * KL_CTX, served_kv=kv, windows_kind="file" if windows else "chat-style (generated)", flags=" ".join(served.get("plan_flags") or []))
    return sec


# ---- the server ----------------------------------------------------------------------------------------------------------------------------------
def free_port(preferred):
    """The preferred port when nothing listens on it, else a free one (a leftover server of an earlier run must never be mistaken for ours)."""
    for p in (preferred, 0):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p)); port = s.getsockname()[1]; s.close(); return port
        except OSError:
            s.close()
    raise RuntimeError("no free port")


def kill_tree(pid, grace=20):
    """Stop a process and everything it started (the launcher AND its llama-server), children first; the rest is killed after `grace` seconds."""
    import moe_cache_record as rr
    if os.name == "nt":
        subprocess.run([win_tool("taskkill"), "/PID", str(pid), "/T", "/F"], capture_output=True); return
    try:
        pids = rr.probe().tree(pid)
    except Exception:
        pids = [pid]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in reversed(pids):
            try:
                os.kill(p, sig)
            except OSError:
                pass
        end = time.time() + (grace if sig == signal.SIGTERM else 5)
        while time.time() < end and any(os.path.exists(f"/proc/{p}") for p in pids):
            time.sleep(0.5)
        if not any(os.path.exists(f"/proc/{p}") for p in pids):
            return


def launch(model, port, extra, ctx=None):
    """Start the model through the launcher (the plan a user gets) on `port` and wait for /health; returns (process, url). The answering server must be our own child serving this model file: a leftover
    or foreign server on the port is refused. On failure the error carries the launcher's last lines."""
    args = list(extra)
    if ctx and not any(a in ("--ctx", "-c", "--ctx-size") for a in args):
        i = args.index("--") if "--" in args else len(args)          # the launcher's own options come before a `--`; what follows goes to llama-server as it is
        args[i:i] = ["--ctx", str(ctx)]
    cmd = [sys.executable, os.path.join(HERE, "moe_cache.py"), "server", model, "--port", str(port)] + args
    log = tempfile.TemporaryFile("w+"); kw = {"start_new_session": True} if os.name != "nt" else {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, **kw); url = f"http://127.0.0.1:{port}"
    for _ in range(900):
        if p.poll() is not None:
            log.seek(0); raise RuntimeError("the server exited while starting: " + " | ".join(log.read().strip().splitlines()[-4:])[-500:])
        try:
            if urllib.request.urlopen(url + "/health", timeout=3).status == 200:
                props = http(url, "/props", timeout=30); got = os.path.basename((props.get("model_path") or "").replace("\\", "/")).lower(); want = os.path.basename(model.replace("\\", "/")).lower()
                if got != want:
                    kill_tree(p.pid); raise RuntimeError(f"the server on port {port} serves {got or '?'}, not {want}: refusing to measure it")
                return p, url
        except RuntimeError:
            raise
        except Exception:
            time.sleep(3)
    kill_tree(p.pid); raise RuntimeError("the server did not become ready")


# ---- partial records (resume) -----------------------------------------------------------------------------------------------------------------
class Partial:
    """Every finished section of a model's run, saved at once to <out>/.<name>_<profile>.partial (JSON) so that a failure or a cancel keeps it and the next run of the same model, profile and
    settings resumes after the last finished section. The key is the model file (size, mtime), the profile and the settings that change a result."""
    def __init__(self, out, name, profile, key_parts):
        self.path = os.path.join(out, "." + re.sub(r"[^A-Za-z0-9._-]+", "-", name) + "_" + profile + ".partial")
        self.key = hashlib.sha256(json.dumps(key_parts, sort_keys=True, default=str).encode()).hexdigest()[:16]
        self.data = {"key": self.key, "sections": {}, "served": {}, "notes": [], "record": ""}

    def load(self):
        try:
            d = json.load(open(self.path, encoding="utf-8"))
            if d.get("key") == self.key:
                self.data = d; return True
        except (OSError, ValueError):
            pass
        return False

    def save(self):
        tmp = self.path + ".tmp"; json.dump(self.data, open(tmp, "w", encoding="utf-8")); os.replace(tmp, self.path)

    def remove(self):
        try:
            os.remove(self.path)
        except OSError:
            pass


def measure_server(url, name, prof, work, part, workers=1, monitor_pid=None, served=None, progress=print):
    """The sections that need the running server; each one is skipped when the partial file already has it and saved as soon as it is done."""
    P = PROFILES[prof]; sec = part.data["sections"]; ctx = (served or {}).get("ctx")

    def done(key, value):
        if value is not None:
            sec[key] = value; part.save()
    if "speed" not in sec:
        progress(f"[{name}] speed ({P['speed']})")
        if P["speed"] == "probe":
            done("speed", run_speed_probe(url, P["depth"], ctx=ctx))
        else:
            out = os.path.join(work, "conv.json")
            _py("conversation_bench.py", "run", "--url", url, "--arm", "suite", "--ctx", str(P["depth"]), "--out", out, *(["--monitor-pid", str(monitor_pid)] if monitor_pid else []))
            run = json.load(open(out)); run = run["runs"][0] if "runs" in run else run
            sp, mem = sections_from_conversation(run, P["depth"])
            if sp.get("final_depth") and sp["final_depth"] < 0.9 * P["depth"]:
                part.data["notes"].append(f"the conversation stopped at {sp['final_depth']} tokens, short of the {P['depth']} asked ({sp.get('stop_reason')})")
            if mem is None:
                part.data["notes"].append("memory section empty: the resource sampler gave no numbers" if monitor_pid else "no memory section: no server process to sample (a running server was measured)")
            done("speed", sp); done("memory", mem)
    for think in P["think"]:
        key = "tasks" if think == "off" else "tasks_on"
        if key not in sec:
            progress(f"[{name}] tasks, thinking {think}"); done(key, run_tasks(url, think, P["tasks_n"], work))
    if P["hard"] and "hard" not in sec:
        progress(f"[{name}] hard set (no cap, {HARD_TASK_SECONDS} s per task)"); done("hard", run_hard(url, HARD_CAP, workers, work, ctx))


def measure(url, name, quant, prof, model_path=None, tmp=None, workers=1, perplexity="llama-perplexity", progress=print, served=None, monitor_pid=None, windows=None):
    """All sections of the profile against a running server, in memory (no partial files; the quality section runs here too, so a caller that keeps the server up holds the model twice: cmd_run
    stops the server first). Returns the suite record."""
    served = served if served is not None else served_facts(url); part = Partial(tmp, name, prof, {"t": "measure"}); part.data["served"] = served
    measure_server(url, name, prof, tmp, part, workers, monitor_pid, served, progress)
    return finish(part, name, quant, prof, model_path, served, tmp, perplexity, progress, windows)


def finish(part, name, quant, prof, model_path, served, tmp, perplexity, progress=print, windows=None, partial=False):
    P = PROFILES[prof]; sec = part.data["sections"]; notes = list(part.data["notes"])
    if P["kl_chunks"] and "quality" not in sec:
        if partial:
            notes.append("quality section not run: the run stopped before it")
        elif not model_path:
            notes.append("quality section skipped: no model file given (a running server was measured)")
        elif not served.get("ctk"):
            notes.append("quality section skipped: the served KV types are not known")
        else:
            try:
                progress(f"[{name}] KL of the served KV cache ({served['ctk']}/{served['ctv']}) against f16 KV")
                q = run_quality(model_path, served, P["kl_chunks"], tmp, perplexity, windows)
                if q is None:
                    notes.append("quality section left out: the served KV cache is f16, identical to the reference")
                else:
                    sec["quality"] = q; part.save()
            except (subprocess.CalledProcessError, RuntimeError, ValueError, OSError) as ex:
                notes.append(f"quality section failed: {str(ex)[:200]}")
    model = {"name": name, "quant": quant, "file": os.path.basename(model_path or "")}
    if model_path and os.path.isfile(model_path):
        model["size_gb"] = round(os.path.getsize(model_path) / 1e9, 2)
    return S.new_record(model, machine=machine_facts(served.get("llama_server")), software=software_facts(served), served={k: v for k, v in served.items() if k not in ("plan_flags", "model_path")},
                        settings={"profile": prof, "plan_flags": served.get("plan_flags") or [], "hard_task_seconds": HARD_TASK_SECONDS if P["hard"] else None}, sections=sec, profile=prof, notes=notes, partial=partial)


def _write(out, rec):
    path = os.path.join(out, rec["id"] + ".json"); S._write_json(path, rec); return path


def cmd_run(a):
    out = a.out or S.default_dir(); os.makedirs(out, exist_ok=True); recs = []; failed = 0
    targets = [(None, a.name or "server", a.quant, a.url)] if a.url else [(m, os.path.splitext(os.path.basename(m))[0], a.quant, None) for m in a.models]
    P = PROFILES[a.profile]
    for model, name, quant, url in targets:
        proc = None; part = None; served = {}
        try:
            st = os.stat(model) if model else None
            part = Partial(out, name, a.profile, {"model": os.path.basename(model or url), "size": st.st_size if st else None, "mtime": int(st.st_mtime) if st else None, "launch": a.launch_args, "windows": a.windows,
                                                  "workers": a.workers, "ctx": P["ctx"], "hard": HARD_TASK_SECONDS})
            if not a.fresh and part.load():
                print(f"[{name}] resuming: already measured: {', '.join(part.data['sections']) or 'nothing'}")
            work = os.path.join(out, "." + os.path.basename(part.path)[1:-8] + ".work"); os.makedirs(work, exist_ok=True)
            mon = None
            if url is None:
                port = free_port(a.port)
                if port != a.port:
                    print(f"port {a.port} is in use: using {port}")
                proc, url = launch(model, port, shlex.split(a.launch_args), P["ctx"]); served = served_facts(url, proc.pid); mon = served.get("server_pid")
            else:
                served = served_facts(url)
            old = part.data.get("served") or {}
            if old and part.data["sections"] and any(old.get(k) != served.get(k) for k in ("ctx", "ctk", "ctv", "np")):
                print(f"[{name}] the server now runs with other settings than the saved partial results: starting over"); part.data = {"key": part.key, "sections": {}, "served": {}, "notes": [], "record": ""}
            part.data["served"] = {k: v for k, v in served.items() if k != "plan_flags"}; part.save()
            try:
                measure_server(url, name, a.profile, work, part, a.workers, mon, served)
            finally:
                if proc is not None:
                    kill_tree(proc.pid); proc = None            # the model leaves memory before llama-perplexity loads it
            rec = finish(part, name, quant, a.profile, model, served, work, a.perplexity, windows=a.windows)
            pr = S.validate_record(rec)
            if pr:
                print("invalid record:", pr, file=sys.stderr); failed += 1; continue
            if part.data.get("record") and os.path.exists(part.data["record"]):
                os.remove(part.data["record"])                    # an earlier partial record of this run is replaced by the complete one
            path = _write(out, rec); recs.append(rec); print("wrote", path); part.remove()
            import shutil; shutil.rmtree(work, ignore_errors=True)
        except BaseException as ex:
            if proc is not None:
                kill_tree(proc.pid)
            failed += 1; print(f"[{name}] stopped: {type(ex).__name__}: {str(ex)[:300]}", file=sys.stderr)
            if part is not None and part.data["sections"]:
                try:                                              # keep what was measured: a partial record (marked, with the reason) and the .partial file for the resume
                    rec = finish(part, name, quant, a.profile, None, served or part.data.get("served") or {}, out, a.perplexity, partial=True); rec["notes"].append(f"PARTIAL: stopped before the end ({type(ex).__name__}: {str(ex)[:150]})")
                    if not S.validate_record(rec):
                        if part.data.get("record") and os.path.exists(part.data["record"]):
                            os.remove(part.data["record"])
                        part.data["record"] = _write(out, rec); part.save(); print("wrote partial record", part.data["record"], "(run the same command again to resume)")
                except Exception as ex2:
                    print("partial record not written:", ex2, file=sys.stderr)
            if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                raise
    if len(recs) > 1:
        print(S.compare_text(S.compare(recs)))
    return 1 if failed else 0


def _term(signum, frame):
    raise SystemExit(143)


def main(argv):
    if argv[:1] == ["card"]:
        import moe_cache_suite_card as C
        return C.main(argv[1:])
    if argv[:1] and argv[0] != "run":
        return S.main(argv)
    ap = argparse.ArgumentParser(prog="bench-suite run"); ap.add_argument("run"); ap.add_argument("models", nargs="*"); ap.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    ap.add_argument("--url", default=""); ap.add_argument("--name", default=""); ap.add_argument("--quant", default=""); ap.add_argument("--out", default=""); ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--launch-args", default="", help="extra launcher arguments, e.g. '--ctx 70000 --kv q8_0' (a --ctx here replaces the profile's)"); ap.add_argument("--windows", default="", help="your own text file for the KL windows (default: generated chat-style text)")
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--perplexity", default="llama-perplexity"); ap.add_argument("--fresh", action="store_true", help="ignore a saved partial run of this model and profile")
    a = ap.parse_args(argv)
    if not a.url and not a.models:
        ap.error("give model files or --url")
    try:
        signal.signal(signal.SIGTERM, _term)            # a cancel from the control page ends the run through the cleanup above (the server is stopped, the partial results are kept)
    except (ValueError, OSError):
        pass
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
