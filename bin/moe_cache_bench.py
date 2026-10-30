#!/usr/bin/env python3
"""moe_cache.py bench-report MODEL.gguf: one command for friends on Linux and Windows (Reddit C11, R2-1). Runs stock llama-server and moe-cache in the
order stock, moe-cache, moe-cache, stock (per pair), with the launcher's own defaults, and writes one report to paste into a GitHub issue
(moe-cache-report-YYYYMMDD-HHMM.md + .json). It only measures: nothing is downloaded, uploaded or changed, and the report holds no prompts, answers,
user name, host name or folder names. Standard library only; the same code on both systems (no systemd-run: the cache size is the launcher's --ram).
docs/SHARE_YOUR_NUMBERS.md is the recipe."""
import argparse, contextlib, datetime, io, json, os, platform, re, socket, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moe_cache as mc  # noqa: E402

SCHEMA = 1
DECODE_PROMPT = "Write a short story about a lighthouse keeper who finds a message in a bottle."
PARA = ("The history of computing spans centuries, from mechanical calculators and punched cards to vacuum tubes, transistors, integrated "
        "circuits and the microprocessors in every phone. ")          # about 37 tokens with the usual tokenizers
# device names in `llama-server --list-devices` ("  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15890 MiB free)") -> backend
BACKENDS = (("CUDA", "CUDA"), ("ROCm", "ROCm-HIP"), ("HIP", "ROCm-HIP"), ("Vulkan", "Vulkan"), ("SYCL", "SYCL"), ("Metal", "Metal"), ("MTL", "Metal"), ("MUSA", "MUSA"), ("CANN", "CANN"), ("RPC", "RPC (remote)"), ("OpenCL", "OpenCL"))


def backend_of(list_devices):
    """The GPU backend llama-server uses, from its --list-devices text; 'CPU only' when it lists no device, 'UNKNOWN - please state it' when unreadable."""
    if not list_devices or "Available devices" not in list_devices:
        return "UNKNOWN - please state it"
    found = []
    listing = list_devices.split("Available devices", 1)[1]       # only the device lines (the log above it has "name0:" lookalikes)
    for line in listing.splitlines():
        m = re.match(r"\s*([A-Za-z]+)\d*\s*:", line)
        if m and m.group(1).upper() != "CPU":
            name = next((n for key, n in BACKENDS if m.group(1).upper().startswith(key.upper())), m.group(1))    # an unknown backend is named as printed
            if name not in found:
                found.append(name)
    return " + ".join(found) if found else "CPU only"


def scrubber():
    """A function that removes what identifies the person: home folder, user name, host name (applied to every line of the report)."""
    home = os.path.expanduser("~")
    words = [w for w in {os.environ.get("USERNAME", ""), os.environ.get("USER", ""), socket.gethostname()} if len(w) >= 3]
    def scrub(text):
        text = str(text)
        for h in {home, home.replace("\\", "/")}:
            if len(h) > 3:
                text = re.sub(re.escape(h), "~", text, flags=re.I)
        for w in words:
            text = re.sub(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", "<name>", text, flags=re.I)
        # any other folder path down to its last part: drive-letter paths may have spaces in folder names (D:\\Private Stuff\\x.gguf),
        # / and ~/ paths may not (so "a / b" text stays); URLs (://) are left alone
        text = re.sub(r"(?<![\w:/])[A-Za-z]:[\\/](?:[^\n|`'\"<>()\\/:]+[\\/])+([^\s|`'\"<>()\\/,;]*)", r"\1", text)
        text = re.sub(r"(?<![\w:/])(?:~[\\/]|/)(?:[^\s|`'\"<>()\\/]+[\\/])+([^\s|`'\"<>()\\/]*)", r"\1", text)
        return text
    return scrub


def scrub_all(x, scrub):
    """scrub() on every string inside a dict / list (before json.dumps, which would double the backslashes of Windows paths)."""
    if isinstance(x, dict):
        return {k: scrub_all(v, scrub) for k, v in x.items()}
    if isinstance(x, list):
        return [scrub_all(v, scrub) for v in x]
    return scrub(x) if isinstance(x, str) else x


def doctor_facts(model, ls, ram):
    """Runs `check` (the same code as `doctor`: no second copy of the checks) silently; (failed, facts {key: value}, fix lines)."""
    buf = io.StringIO()
    argv = [model, "--llama-server", ls] + (["--ram", str(ram)] if ram else [])
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = mc.check(argv)
    facts, fixes, part = {}, [], None
    for line in buf.getvalue().splitlines():
        if line.startswith("this machine at a glance"):
            part = "facts"; continue
        if line.startswith("what to fix"):
            part = "fix"; continue
        if line.startswith("summary:"):
            part = None; continue
        m = re.match(r"\s{2}(\S[\w-]*)\s{2,}(.*)$", line)
        if part == "facts" and m:
            facts[m.group(1)] = m.group(2).strip()
        elif part == "fix" and line.strip():
            fixes.append(line.strip())
        m = re.match(r"\s*\[\s*(?:OK|WARN)\s*\]\s+drive speed\s+(.*)$", line)
        if m:
            facts["drive speed"] = m.group(1).strip()
    return rc != 0, facts, fixes


def launcher_plan(model, ls, ram, ctx, extra):
    """The launcher's own decision for this model (a dry run): ('plugin' or 'stock', the llama-server command line it would run)."""
    cmd = [sys.executable, os.path.join(HERE, "moe_cache.py"), "server", model, "--llama-server", ls, "--ctx", str(ctx), "--plan-json"]
    cmd += (["--ram", str(ram)] if ram else []) + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    try:        # XA-2: the launcher's plan as data (engine and command line), not text parsed from a dry run
        plan = json.loads(r.stdout)
        return plan["engine"], plan["command_line"]
    except Exception:
        pass
    cmd[cmd.index("--plan-json")] = "--dry-run"       # a launcher from before --plan-json
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    out = r.stdout + r.stderr
    line = next((l for l in out.splitlines() if "llama-server" in l and " -m " in l), "")
    return ("plugin" if "MOE_CACHE_" in line else "stock"), line


def plan_summary(line):
    """The settings worth reporting from the launcher's command line: experts on the CPU, GPU layers, context, micro-batch, KV type."""
    keep = []
    for flag in ("--n-cpu-moe", "-ngl", "-c", "-ub", "-ctk", "-ctv", "-fa", "--cache-ram"):
        m = re.search(r"(?:^|\s)" + re.escape(flag) + r"\s+(\S+)", line)
        if m:
            keep.append(f"{flag} {m.group(1)}")
    m = re.search(r"MOE_CACHE_SIZE_GIB=(\S+)", line)
    if m:
        keep.append(f"cache {m.group(1)} GiB")
    return ", ".join(keep) or "launcher defaults"


def run_one(model, mode, a, port):
    """One server start through `moe_cache.py server` (stock: --stock, moe-cache: --always-plugin), one long prompt and one greedy answer.
    {'mode', 'decode', 'prefill', 'prompt_tokens', 'ttft_s', 'text', 'plugin_active', 'hit_rate', 'gb_read', 'load_s', 'error'}."""
    import moe_cache_ui as ui
    cmd = [sys.executable, os.path.join(HERE, "moe_cache.py"), "server", model, "--llama-server", a.llama_server, "--port", str(port),
           "--ctx", str(a.ctx), "--no-warm", "--stock" if mode == "stock" else "--always-plugin"] + (["--ram", str(a.ram)] if a.ram else []) + list(a.extra)
    env = dict(os.environ, MOE_CACHE_STATS_SIGNAL="1")
    kw = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    if mc.WINDOWS:
        kw["creationflags"] = 0x00000200                  # CREATE_NEW_PROCESS_GROUP
    t0 = time.time()
    p = subprocess.Popen(cmd, **kw)
    lines = []
    threading.Thread(target=lambda: [lines.append(l.decode(errors="replace").rstrip()) for l in iter(p.stdout.readline, b"")], daemon=True).start()
    res = {"mode": mode, "error": ""}
    try:
        while not mc.healthy(port):
            if p.poll() is not None:
                res["error"] = "the server stopped while loading: " + next((l for l in reversed(lines) if l.strip()), "no output")
                return res
            if time.time() - t0 > a.load_timeout:
                res["error"] = f"the server was not ready after {a.load_timeout} s"
                return res
            time.sleep(1)
        res["load_s"] = round(time.time() - t0, 1)
        url = f"http://127.0.0.1:{port}/completion"
        long_prompt = "Summarise the following text in five bullet points.\n" + PARA * max(1, a.prompt_tokens // 37)
        st, body = mc.http(url, {"prompt": long_prompt, "n_predict": 1, "temperature": 0, "cache_prompt": False}, timeout=3600)
        tm = json.loads(body).get("timings", {}) if st == 200 else {}
        res["prefill"], res["prompt_tokens"] = tm.get("prompt_per_second"), tm.get("prompt_n")
        t1 = time.time()
        st, body = mc.http(url, {"prompt": DECODE_PROMPT, "n_predict": a.tokens, "temperature": 0, "seed": 42, "cache_prompt": False}, timeout=3600)
        if st != 200:
            res["error"] = f"the answer request failed (HTTP {st})"
            return res
        j = json.loads(body); tm = j.get("timings", {})
        res["decode"], res["text"] = tm.get("predicted_per_second"), j.get("content", "")
        res["ttft_s"] = round(tm.get("prompt_ms", 0) / 1000, 2) if tm.get("prompt_ms") is not None else round(time.time() - t1, 2)
        if getattr(a, "scenario", "") == "agent":
            res.update(agent_turns(url, a))
        res["plugin_active"] = any("moe-cache: cache mode" in l for l in lines)
        if mode == "moe-cache":
            pid = ui.child_pid(p.pid)
            if pid and ui.ask_snapshot(pid):
                for _ in range(20):
                    snap = [l for l in lines if "moe-cache-stats " in l]
                    if snap:
                        s = json.loads(snap[-1].split("moe-cache-stats ", 1)[1])
                        h, m = s.get("hits", 0), s.get("misses", 0)
                        res["hit_rate"] = round(100 * h / (h + m), 1) if h + m else None
                        res["gb_read"] = round(s.get("read_bytes", 0) / 1e9, 2)
                        break
                    time.sleep(0.25)
        return res
    except (ValueError, OSError) as e:
        res["error"] = f"{type(e).__name__}: {e}"
        return res
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        for _ in range(60):          # Windows: llama-server ends a moment after the launcher; the next run needs the port
            if not mc.port_busy(port):
                break
            time.sleep(1)


def server_info(ls):
    """(`llama-server --list-devices` text, its version line)."""
    try:
        lst = subprocess.run([ls, "--list-devices"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        ver = subprocess.run([ls, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        return lst.stdout + lst.stderr, next((l.strip() for l in (ver.stdout + ver.stderr).splitlines() if "version" in l), "unknown")
    except (OSError, subprocess.TimeoutExpired):
        return "", "unknown"


AGENT_TURNS = ["Read the material above. List the three most important functions it describes and what each one does.",
               "Now write a short Python function that would test the first of them, with a docstring.",
               "Shorten that function to the essential lines and explain the change in one sentence."]


def agent_turns(url, a):
    """--scenario agent: one conversation of len(AGENT_TURNS) turns on a long first message (about --prompt-tokens), each later turn adds the answer and a
    short request (cache_prompt: only the new tokens are processed, as with OpenCode / Hermes), with --gap seconds between turns (a tool running).
    Returns {'agent_ttft_s': [...], 'agent_decode': [...]}: time to the first token and decode speed per turn."""
    msgs = [{"role": "system", "content": "Material:\n" + PARA * max(1, a.prompt_tokens // 37)}]
    ttft, dec = [], []
    for i, q in enumerate(AGENT_TURNS):
        if i:
            time.sleep(a.gap)
        msgs.append({"role": "user", "content": q})
        st, body = mc.http(url.replace("/completion", "/v1/chat/completions"),
                           {"messages": msgs, "max_tokens": a.tokens, "temperature": 0, "seed": 42, "cache_prompt": True}, timeout=3600)
        if st != 200:
            break
        j = json.loads(body); tm = j.get("timings", {})
        ttft.append(round(tm.get("prompt_ms", 0) / 1000, 2)); dec.append(round(tm.get("predicted_per_second") or 0, 2))
        msgs.append({"role": "assistant", "content": (j.get("choices") or [{}])[0].get("message", {}).get("content", "")})
    return {"agent_ttft_s": ttft, "agent_decode": dec}


def order(pairs):
    """stock, moe-cache, moe-cache, stock for each pair (both sides see the same warm-up and drift)."""
    return [m for _ in range(pairs) for m in ("stock", "moe-cache", "moe-cache", "stock")]


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 2) if xs else None


def summarize(runs, exact):
    """Means per side, ratios and the identity verdict (decode texts of stock vs moe-cache, byte for byte)."""
    side = {m: [r for r in runs if r["mode"] == m and not r.get("error")] for m in ("stock", "moe-cache")}
    out = {}
    for m, rs in side.items():
        out[m] = {k: mean([r.get(k) for r in rs]) for k in ("decode", "prefill", "ttft_s", "hit_rate", "gb_read", "load_s")}
        out[m]["decode_runs"] = [round(r["decode"], 2) for r in rs if isinstance(r.get("decode"), (int, float))]
        turns = [r["agent_ttft_s"] for r in rs if r.get("agent_ttft_s")]
        out[m]["agent_ttft_s"] = [mean([t_[i] for t_ in turns if len(t_) > i]) for i in range(max((len(t_) for t_ in turns), default=0))]
    texts = {m: {r.get("text", "") for r in rs} for m, rs in side.items()}
    if not exact:
        out["identity"] = "n/a (approximate mode)"
    elif not texts["stock"] or not texts["moe-cache"]:
        out["identity"] = "not checked (a side has no successful run)"
    else:
        out["identity"] = "IDENTICAL" if texts["stock"] == texts["moe-cache"] and len(texts["stock"]) == 1 else "DIFFERENT"
    ratio = lambda k: round(out["moe-cache"][k] / out["stock"][k], 2) if out["moe-cache"][k] and out["stock"][k] else None
    out["decode_x"], out["prefill_x"] = ratio("decode"), ratio("prefill")
    out["plugin_active"] = any(r.get("plugin_active") for r in side["moe-cache"])
    return out


def report_md(d):
    """The report as GitHub markdown, sections in a fixed order (tests/bench_report_test.py checks them)."""
    s, f = d["summary"], d["facts"]
    v = lambda x, unit="": "-" if x is None else f"{x}{unit}"
    lines = [f"## moe-cache benchmark report",
             "",
             f"moe-cache {d['version']} · report schema {SCHEMA} · {d['date']} UTC · **{d['start']}** · mode {d['mode']}",
             "",
             f"**Result:** moe-cache {v(s['moe-cache']['decode'])} tok/s vs stock {v(s['stock']['decode'])} tok/s decode ({v(s['decode_x'], 'x')}), "
             f"prompt {v(s['moe-cache']['prefill'])} vs {v(s['stock']['prefill'])} tok/s, tokens {s['identity']}",
             "",
             "| Machine | |", "|---|---|",
             f"| OS | {d['os']} |",
             f"| CPU | {f.get('cpu', '-')} |",
             f"| RAM | {f.get('ram', '-')} |",
             f"| GPU | {f.get('gpu', '-')} |",
             f"| **Backend** | **{d['backend']}** |",
             f"| llama.cpp | {d['llama']} |",
             f"| plugin | {f.get('moe-cache', '-')} |",
             f"| Drive | {f.get('drive speed', f.get('disk', '-'))} |",
             "",
             "| Model | |", "|---|---|",
             f"| File | {d['model']['file']} |",
             f"| Size | {d['model']['size_gb']} GB ({d['model']['ratio']}x the RAM) |",
             f"| Quant | {d['model']['quant']} |",
             f"| Launcher plan | {d['plan']} |",
             f"| Stock plan | {d.get('stock_plan', '-')} |",
             f"| Fits in RAM | {d['fits']} |",
             "",
             "| | stock | moe-cache | ratio |", "|---|---|---|---|",
             f"| decode tok/s (mean; runs) | {v(s['stock']['decode'])}; {s['stock']['decode_runs']} | {v(s['moe-cache']['decode'])}; {s['moe-cache']['decode_runs']} | {v(s['decode_x'], 'x')} |",
             f"| prompt tok/s ({d['prompt_tokens']} tokens) | {v(s['stock']['prefill'])} | {v(s['moe-cache']['prefill'])} | {v(s['prefill_x'], 'x')} |",
             f"| time to first token, s | {v(s['stock']['ttft_s'])} | {v(s['moe-cache']['ttft_s'])} | |",
             f"| server start, s | {v(s['stock']['load_s'])} | {v(s['moe-cache']['load_s'])} | |",
             *([f"| agent turns: time to first token, s (turn 1 / 2 / 3, {d.get('gap', 0)} s gaps) | {' / '.join(map(str, s['stock'].get('agent_ttft_s') or ['-']))} | "
                f"{' / '.join(map(str, s['moe-cache'].get('agent_ttft_s') or ['-']))} | |"] if d.get("scenario") == "agent" else []),
             f"| cache hit rate | | {v(s['moe-cache']['hit_rate'], ' %')} | |",
             f"| read from the drive | | {v(s['moe-cache']['gb_read'], ' GB')} | |",
             "",
             f"Checks: tokens {s['identity']}; plugin active {'yes' if s['plugin_active'] else 'NO'}; order {' / '.join(d['order'])}",
             ]
    if d["notes"]:
        lines += ["", "Notes:"] + [f"- {n}" for n in d["notes"]]
    lines += ["", "_No prompts, answers, user name, host name or folder names are in this report. Nothing was uploaded._"]
    return "\n".join(lines) + "\n"


def bench_report(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py bench-report", description="measure stock llama-server and moe-cache on this machine and write a report "
                                 "to paste into a GitHub issue (docs/SHARE_YOUR_NUMBERS.md). Measures only; nothing is uploaded.")
    ap.add_argument("model")
    ap.add_argument("--llama-server", default=mc.default_llama_server())
    ap.add_argument("--ram", help="GB the cache may use (as `server`); default: what is free")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--tokens", type=int, default=128, help="tokens in the timed answer")
    ap.add_argument("--pairs", type=int, default=1, help="each pair runs stock, moe-cache, moe-cache, stock")
    ap.add_argument("--out", default=".", help="folder for the report files")
    ap.add_argument("--flush-from", default="", help="Windows: folder(s) with other big files (comma-separated), read so the file cache drops the model before each run")
    ap.add_argument("--warm", action="store_true", help="skip the cold start (the report says WARM)")
    ap.add_argument("--force-plugin", action="store_true", help="also measure when the model fits in RAM (the launcher would run stock)")
    ap.add_argument("--fast", nargs="?", const="1", choices=["1", "2", "kv", "fewer"], help="measure an approximate mode (identity is then n/a)")
    ap.add_argument("--scenario", choices=["chat", "agent"], default="chat", help="agent: also a 3-turn conversation with --gap seconds between turns (coding agents)")
    ap.add_argument("--gap", type=float, default=5.0, help="seconds between agent turns (a tool running)")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--load-timeout", type=int, default=1800)
    a = ap.parse_args(argv)
    a.extra = ["--fast=" + a.fast] if a.fast else []
    mc.say("moe-cache bench-report: about 5-20 minutes; leave the machine alone (close the browser and games: they hold GPU memory)")
    if not os.path.isfile(a.model):
        mc.say(f"bench-report: model not found: {a.model}", err=True); return 2
    if mc.port_busy(a.port):
        mc.say(f"bench-report: port {a.port} is in use (--port N)", err=True); return 2
    failed, facts, fixes = doctor_facts(a.model, a.llama_server, a.ram)
    if failed:
        mc.say("bench-report: the doctor found a problem; fix it first:", err=True)
        for l in fixes:
            if l.startswith("[FAIL]"):
                mc.say("  " + l, err=True)
        return 2
    plan_mode, plan_line = launcher_plan(a.model, a.llama_server, a.ram, a.ctx, a.extra)
    stock_line = launcher_plan(a.model, a.llama_server, a.ram, a.ctx, ["--stock"])[1]
    fits = plan_mode == "stock"
    if fits and not a.force_plugin:
        mc.say("bench-report: this model fits in your RAM, so moe-cache would not be used (the launcher runs stock llama.cpp). "
               "Pick a bigger model (`moe_cache.py pick`) or pass --force-plugin.", err=True)
        return 3
    import moe_cache_ui as ui
    notes, start = [], "WARM" if a.warm else "COLD"
    other = [x for x in a.flush_from.split(",") if x]
    runs = []
    for i, mode in enumerate(order(a.pairs)):
        if not a.warm:
            ok_, why = ui.flush_model_cache(a.model, other)
            if not ok_:
                if start == "COLD":
                    notes.append(f"cold start failed ({why}): stock and moe-cache are not comparable with the README rows (WARM)")
                start = "WARM"
        mc.say(f"bench-report: run {i + 1}/{len(order(a.pairs))}: {mode} ...")
        r = {"mode": mode, "error": f"port {a.port} still in use by the previous server"} if mc.port_busy(a.port) else run_one(a.model, mode, a, a.port)
        if r.get("error"):
            notes.append(f"run {i + 1} ({mode}): {r['error']}")
            mc.say(f"bench-report: run {i + 1} failed: {r['error']}", err=True)
        else:
            mc.say(f"bench-report: run {i + 1}: decode {r.get('decode') or 0:.2f} tok/s, prompt {r.get('prefill') or 0:.1f} tok/s")
        runs.append(r)
    devices, llama = server_info(a.llama_server)
    size = sum(os.path.getsize(p) for p in mc.model_parts(a.model)) / 1e9
    ram_total = mc.win_memory()[0] / 1e9 if mc.WINDOWS and mc.win_memory()[0] else mc.meminfo_total_gb()
    if fits:
        notes.append("the model fits in RAM: moe-cache was forced (--force-plugin); the launcher would run stock")
    if a.ram:
        notes.append(f"cache limit by --ram {a.ram} GB, not an OS memory limit")
    s_ = summarize(runs, exact=not a.fast)
    if s_["identity"] == "DIFFERENT":
        notes.append("tokens differ: stock and moe-cache ran with the launcher's own plans (stock plan above), and a layer computed on the GPU instead of "
                     "the CPU can change the last digits and then a token; the same-split identity proof is the CI identity suite. Please report it anyway")
    d = {"schema": SCHEMA, "version": mc.VERSION, "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"), "start": start,
         "mode": f"fast={a.fast}" if a.fast else "exact", "os": f"{platform.system()} {platform.release()} ({platform.version()})", "facts": facts,
         "backend": backend_of(devices), "llama": llama,
         "model": {"file": os.path.basename(a.model), "size_gb": round(size, 1), "ratio": round(size / ram_total, 2) if ram_total else "-",
                   "quant": ui.header_quant(a.model) or "-"},
         "plan": plan_summary(plan_line), "scenario": a.scenario, "gap": a.gap, "fits": "yes (forced)" if fits else "no", "prompt_tokens": a.prompt_tokens, "order": order(a.pairs),
         "summary": s_, "notes": notes, "stock_plan": plan_summary(stock_line),
         "runs": [{k: v for k, v in r.items() if k != "text"} for r in runs]}
    scrub = scrubber()
    md = scrub(report_md(d))
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, "moe-cache-report-" + datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    open(base + ".md", "w", encoding="utf-8").write(md)
    open(base + ".json", "w", encoding="utf-8").write(json.dumps(scrub_all(d, scrub), indent=1, ensure_ascii=False))
    mc.say("")
    mc.say(md)
    mc.say(f"bench-report: written {base}.md and .json. Paste the .md into a new issue: https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=benchmark_report.yml")
    mc.say("bench-report: no prompts, answers, user name, host name or folder names are in the report; nothing was uploaded.")
    return 0
