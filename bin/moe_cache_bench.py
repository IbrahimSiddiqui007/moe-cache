#!/usr/bin/env python3
"""moe_cache.py bench-report MODEL.gguf: one command for friends on Linux and Windows (Reddit C11, R2-1). Runs stock llama-server and moe-cache in the
order stock, moe-cache, moe-cache, stock (per pair), with the launcher's own defaults, and writes one report to paste into a GitHub issue
(moe-cache-report-YYYYMMDD-HHMM.md + .json). It only measures: nothing is downloaded, uploaded or changed, and the report holds no prompts, answers,
user name, host name or folder names. Standard library only; the same code on both systems (no systemd-run: the cache size is the launcher's --ram).
docs/SHARE_YOUR_NUMBERS.md is the recipe."""
import argparse, contextlib, datetime, io, json, os, platform, re, shutil, socket, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(os.path.dirname(HERE), "tools"))      # tools/conversation_bench.py (--chat); both kits have tools/ next to bin/
import moe_cache as mc  # noqa: E402
import moe_cache_record as rec  # noqa: E402
from moe_cache_wintool import win_tool  # noqa: E402  (AUD-034: Windows tools by their System32 path)

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
        # Windows extended prefixes (\\?\C:\x, \\?\UNC\NAS\share\x) become the plain form, then UNC paths (\\NAS\share\x.gguf) go down to the last part too (AUD-032)
        text = re.sub(r"\\\\[?.]\\UNC\\", lambda m: "\\\\", text, flags=re.I)
        text = re.sub(r"\\\\[?.]\\", "", text)
        text = re.sub(r"(?<![\w\\])\\\\(?:[^\n|`'\"<>()\\/:]+\\)+([^\s|`'\"<>()\\/,;]*)", r"\1", text)
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


def model_section(model, size_gb, ram_total_gb):
    """The report's model part, from the one model identity (bin/moe_cache_model.py): file, size, quant, and what the header says (architecture, experts, the
    experts' quant, a short hash of the header and tensor table). No path."""
    import moe_cache_model as mm
    d = {"file": os.path.basename(model), "size_gb": round(size_gb, 1), "ratio": round(size_gb / ram_total_gb, 2) if ram_total_gb else "-", "quant": "-"}
    try:
        ident = mm.identity(model)
        d["quant"] = ident.file_quant or ident.expert_quant or "-"
        d["identity"] = ident.summary()
        d["header_fp"] = ident.header_fp
    except Exception:
        d["quant"] = mm.file_quant(model) or "-"
    return d


def _launch(model, mode, a, port, tee=None, extra=()):
    """The server of one run, through `moe_cache.py server` (stock: --stock, moe-cache: --always-plugin). (process, lines, sampler): lines fill from its output (also
    appended to the file `tee` when given: the plugin's counters are read from it), the sampler watches its process tree."""
    cmd = [sys.executable, os.path.join(HERE, "moe_cache.py"), "server", model, "--llama-server", a.llama_server, "--port", str(port),
           "--ctx", str(a.ctx), "--no-warm", "--stock" if mode == "stock" else "--always-plugin"] + (["--ram", str(a.ram)] if a.ram else []) + list(extra) + list(a.extra)
    env = dict(os.environ, MOE_CACHE_STATS_SIGNAL="1")
    kw = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    if mc.WINDOWS:
        kw["creationflags"] = 0x00000200                  # CREATE_NEW_PROCESS_GROUP
    elif sys.platform.startswith("linux"):
        kw["preexec_fn"] = mc._die_with_parent_linux      # "Cancel" on the control page SIGTERMs this process alone: the server must not stay behind
    p = subprocess.Popen(cmd, **kw)
    lines = []
    def pump():
        f = open(tee, "ab") if tee else None
        try:
            for raw in iter(p.stdout.readline, b""):
                lines.append(raw.decode(errors="replace").rstrip())
                if f:
                    f.write(raw); f.flush()
        finally:
            if f:
                f.close()
    threading.Thread(target=pump, daemon=True).start()
    smp = rec.Sampler(p.pid, log_lines=lines)         # VRAM, process RAM, free RAM, CPU % while this run goes on (the launcher's tree: llama-server is its child)
    smp.start()
    return p, lines, smp


def _wait_ready(p, lines, port, t0, timeout):
    """None when the server answers /health, else the reason it did not."""
    while not mc.healthy(port):
        if p.poll() is not None:
            return "the server stopped while loading: " + next((l for l in reversed(lines) if l.strip()), "no output")
        if time.time() - t0 > timeout:
            return f"the server was not ready after {timeout} s"
        time.sleep(1)
    return None


def _stop_server(p, port):
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


def _log_facts(lines):
    return {"lines": [l for l in lines if re.search(r"n_layer\s*=|offloaded \d+/\d+|buffer size\s*=", l)][:60]}


def run_one(model, mode, a, port):
    """One server start through `moe_cache.py server` (stock: --stock, moe-cache: --always-plugin), one long prompt and one greedy answer.
    {'mode', 'decode', 'prefill', 'prompt_tokens', 'ttft_s', 'text', 'plugin_active', 'hit_rate', 'gb_read', 'load_s', 'error'}."""
    import moe_cache_ui as ui
    t0 = time.time()
    p, lines, smp = _launch(model, mode, a, port)
    res = {"mode": mode, "error": ""}
    try:
        res["error"] = _wait_ready(p, lines, port, t0, a.load_timeout) or ""
        if res["error"]:
            return res
        res["load_s"] = round(time.time() - t0, 1)
        t_req = time.time()
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
        res["wall_s"] = round(time.time() - t_req, 1)
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
        res["monitor"] = smp.stop()
        res["log_facts"] = _log_facts(lines)
        _stop_server(p, port)


def chat_ctx(ram_gb):
    """Context of the long chat for this machine (report section 7): 16k up to 20 GB of RAM, 32k up to 28 GB, else the published 65k."""
    return 16384 if ram_gb and ram_gb <= 20 else 32768 if ram_gb and ram_gb <= 28 else 65536


def run_chat_arm(model, mode, a, port, workdir, key, plan_line):
    """One long-chat run (the published method, tools/conversation_bench.py): server through the launcher with --agent, the conversation up to the context,
    resource peaks, the plugin's counters per turn. {'mode', 'error', 'chat': the run dict, 'decode', 'prefill', 'ttft_s', 'load_s', 'plugin_active', 'hit_rate'}."""
    import conversation_bench as cb
    import moe_cache_ui as ui
    log = os.path.join(workdir, f"server-{mode}-{int(time.time())}.log")
    t0 = time.time()
    p, lines, smp = _launch(model, mode, a, port, tee=log, extra=["--agent"])
    res = {"mode": mode, "error": ""}
    try:
        res["error"] = _wait_ready(p, lines, port, t0, a.load_timeout) or ""
        if res["error"]:
            return res
        load_s = round(time.time() - t0, 1)
        pid = ui.child_pid(p.pid) or p.pid
        ns = cb.default_args(ctx=a.ctx, api_key=key, stats_pid=pid if mode != "stock" else None, stats_log=log if mode != "stock" else None, max_turns=a.max_turns or None,
                             monitor_pid=pid, server_cmd=plan_line, server_log=[log], load_s=load_s)
        run = cb._one(ns, f"http://127.0.0.1:{port}", "stock" if mode == "stock" else "moe-cache", log=lambda m: mc.say("bench-report:   " + m))
        o = cb.overall(run)
        res.update(chat=run, decode=o["decode_tps"], prefill=o["prompt_tps"], ttft_s=o["ttft_s"], load_s=load_s, hit_rate=o["hit_rate_dec_pct"], wall_s=run.get("wall_s"),
                   plugin_active=any("moe-cache: cache mode" in l for l in lines))
        if run["stop_reason"].startswith("error"):
            res["error"] = run["stop_reason"]
        return res
    except (ValueError, OSError) as e:
        res["error"] = f"{type(e).__name__}: {e}"
        return res
    finally:
        res["monitor"] = smp.stop()
        res["log_facts"] = _log_facts(lines)
        _stop_server(p, port)


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


NOISE_FLOOR_PCT = 10        # the project rule: a difference under 10 % is never a claim


def spread_pct(xs):
    """(max - min) / mean of one arm's runs, in percent; None with fewer than two runs."""
    xs = [x for x in xs if isinstance(x, (int, float)) and x > 0]
    return round(100 * (max(xs) - min(xs)) / (sum(xs) / len(xs)), 1) if len(xs) >= 2 else None


def noise_verdict(out):
    """The noise label (rule: under 10 % is not a claim). Per metric the ratio is compared with the larger of 10 % and the spread of either arm's own runs;
    {'noise_pct', 'decode': 'faster|slower|within noise|n/a', 'prefill': ..., 'sentence', 'claimable'}."""
    spreads = [spread_pct(out[m].get(k + "_runs")) for m in ("stock", "moe-cache") for k in ("decode", "prefill")]
    noise = max([NOISE_FLOOR_PCT] + [x for x in spreads if x is not None])
    v = {"noise_pct": noise}
    for k in ("decode", "prefill"):
        x = out.get(k + "_x")
        v[k] = "n/a" if x is None else "within noise" if abs(x - 1) * 100 < noise else "faster" if x > 1 else "slower"
    word = {"prefill": "prompt reading", "decode": "answers"}
    part = [f"{out[k + '_x']}x on {word[k]}" for k in ("prefill", "decode") if v[k] in ("faster", "slower")]
    if all(v[k] == "n/a" for k in ("decode", "prefill")):
        v["sentence"], v["claimable"] = "No ratio: one side has no successful run. Nothing to claim.", False
    elif part:
        v["claimable"] = True
        slow = [k for k in ("prefill", "decode") if v[k] == "slower"]
        v["sentence"] = ("moe-cache vs stock: " + ", ".join(part) + f"; the difference is larger than the noise (+-{noise:g} %)"
                         + ("; but it is SLOWER on " + " and ".join(word[k] for k in slow) if slow else "") + ".")
    else:
        v["claimable"] = False
        v["sentence"] = f"Within noise (+-{noise:g} %): do not claim a difference."
    return v


def _on_battery():
    """True when the computer runs on battery, False on mains, None when unknown (desktops, no sensor)."""
    if mc.WINDOWS:
        try:
            import ctypes
            class SPS(ctypes.Structure):
                _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("pct", ctypes.c_ubyte), ("sf", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
            st = SPS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)) and st.ac in (0, 1):
                return st.ac == 0
        except Exception:
            pass
        return None
    try:
        sup = [d for d in os.listdir("/sys/class/power_supply") if open(f"/sys/class/power_supply/{d}/type").read().strip() == "Battery"]
        if not sup:
            return None
        ac = [d for d in os.listdir("/sys/class/power_supply") if open(f"/sys/class/power_supply/{d}/type").read().strip() == "Mains"]
        if ac:
            return not any(open(f"/sys/class/power_supply/{d}/online").read().strip() == "1" for d in ac)
        return open(f"/sys/class/power_supply/{sup[0]}/status").read().strip() == "Discharging"
    except OSError:
        return None


def _big_programs(limit_gb=1.0):
    """Names and GB of other programs holding over limit_gb of RAM (llama-server and python excluded); [] when none or unknown."""
    out = []
    try:
        if mc.WINDOWS:
            r = subprocess.run([win_tool("tasklist"), "/FO", "CSV", "/NH"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
            for row in r.stdout.splitlines():
                f = [x.strip('"') for x in row.split('","')]
                if len(f) >= 5:
                    kb = int(re.sub(r"\D", "", f[4]) or 0)
                    out.append((f[0], kb / 1048576))
        else:
            for pid in os.listdir("/proc"):
                if pid.isdigit():
                    try:
                        rss = int(open(f"/proc/{pid}/statm").read().split()[1]) * 4096 / 1e9
                        if rss > limit_gb:
                            out.append((open(f"/proc/{pid}/comm").read().strip(), rss))
                    except (OSError, ValueError, IndexError):
                        pass
    except (OSError, subprocess.SubprocessError):
        return []
    mine = ("llama-server", "python", "python3", "moe_cache", "moe-cache")
    best = {}
    for n, g in out:
        if g > limit_gb and not n.lower().startswith(mine):
            best[n] = best.get(n, 0) + g
    return sorted(((n, round(g, 1)) for n, g in best.items()), key=lambda x: -x[1])[:4]


def estimate(size_gb, pairs, chat_ctx_tokens=0):
    """(minutes, GB read at most) for a report: 4 runs per pair, each a cold start (the model is read again) plus about 75 s of prompt and answer; a long chat adds
    about 10 minutes per 65k tokens of conversation (one moe-cache run at 65k took 388 s on the PC, stock is slower)."""
    runs = 4 * max(1, pairs)
    return round(runs * (75 + 4 * size_gb + 600 * chat_ctx_tokens / 65536) / 60), round(runs * size_gb)


def readiness(model, pairs=2, ram=None, chat=False):
    """The ready-to-measure panel: [{id, level 'ok'|'warn'|'bad'|'info', text, fix}] for the model about to be measured (the checks that make a report unusable:
    the model fits in RAM, little free disk, battery, other big programs, power plan, temperature). Reads only; the doctor's own checks stay in `check`."""
    pr = rec.probe()
    rows = []
    add = lambda id_, level, text, fix="": rows.append({"id": id_, "level": level, "text": text, "fix": fix})
    size = sum(os.path.getsize(p) for p in mc.model_parts(model)) / 1e9
    tot = (mc.win_memory()[0] / 1e9) if mc.WINDOWS and mc.win_memory()[0] else mc.meminfo_total_gb()
    if tot:
        if size >= tot * 0.9:
            add("size", "ok", f"The model ({size:.0f} GB) is bigger than the usable RAM ({tot:.0f} GB): moe-cache is meant for this.")
        else:
            add("size", "bad", f"The model ({size:.0f} GB) fits in {tot:.0f} GB of RAM, so the launcher runs stock llama.cpp and there is nothing to compare.",
                "Pick a bigger model, or tick \"Measure anyway\" for a stress case.")
    try:
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(model))).free / 1e9
        add("disk", "ok" if free >= 10 else "warn", f"{free:.0f} GB free on the model's drive.", "" if free >= 10 else "Free at least 10 GB: the system and the saved sessions write there.")
    except OSError:
        add("disk", "info", "Free disk space not readable.")
    bat = _on_battery()
    add("power", "info" if bat is None else "warn" if bat else "ok", "Plugged in (or no battery)." if not bat else "Running on battery.",
        "Plug in the charger: on battery the CPU slows down and the numbers are not comparable." if bat else "")
    big = _big_programs()
    add("programs", "warn" if big else "ok", ("Other big programs: " + ", ".join(f"{n} {g} GB" for n, g in big) + ".") if big else "No other program holds over 1 GB of RAM.",
        "Close them (browser tabs and games also hold GPU memory)." if big else "")
    try:
        plan = pr.power_plan() if hasattr(pr, "power_plan") else rec.NR
    except Exception:
        plan = rec.NR
    if plan and plan != rec.NR:
        slow = bool(re.search(r"power-saver|power saver|Power saver", str(plan), re.I))
        add("plan", "warn" if slow else "ok", f"Power plan: {plan}.", "Switch to Balanced or Performance." if slow else "")
    else:
        add("plan", "info", "Power plan not recorded on this machine.")
    t = (pr.temperatures() or {}).get("cpu_c")
    if t is None:
        add("temp", "info", "CPU temperature not readable: the report says \"not recorded\", and a run cannot be called clean.")
    else:
        add("temp", "warn" if t >= 70 else "ok", f"CPU temperature {t:.0f} C.", "Let the machine cool down first (a hot start throttles the first runs)." if t >= 70 else "")
    ctx = chat_ctx(tot) if chat else 0
    mins, gb = estimate(size, pairs, ctx)
    add("estimate", "info", f"About {mins} minutes and up to {gb} GB read from the drive for {pairs} pair(s) (stock, moe-cache, moe-cache, stock; cold start"
        + (f"; long chat to {ctx // 1024}k tokens" if chat else "") + "). Rough; leave the machine alone.")
    return {"model": os.path.basename(model), "size_gb": round(size, 1), "ready": not any(r["level"] == "bad" for r in rows), "pairs": pairs, "minutes": mins, "gb_read": gb,
            "chat_ctx": ctx, "checks": rows}


def summarize(runs, exact):
    """Means per side, ratios and the identity verdict (decode texts of stock vs moe-cache, byte for byte)."""
    side = {m: [r for r in runs if r["mode"] == m and not r.get("error")] for m in ("stock", "moe-cache")}
    out = {}
    for m, rs in side.items():
        out[m] = {k: mean([r.get(k) for r in rs]) for k in ("decode", "prefill", "ttft_s", "hit_rate", "gb_read", "load_s")}
        out[m]["decode_runs"] = [round(r["decode"], 2) for r in rs if isinstance(r.get("decode"), (int, float))]
        out[m]["prefill_runs"] = [round(r["prefill"], 1) for r in rs if isinstance(r.get("prefill"), (int, float))]
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
    out["noise"] = noise_verdict(out)
    return out


def rec_for_issue(record, chat):
    """The record without its notes; a long chat's per-turn lists are left out too (they stay in the .record.json file: an issue holds 65,000 characters, a 65k conversation has ~150 turns)."""
    out = {k: v for k, v in record.items() if k != "notes"}
    if chat:
        out["runs"] = [{k: v for k, v in r.items() if k != "turns"} for r in record.get("runs", [])]
    return out


RECORD_HOLE = "@@RECORD-JSON@@"


def scrubbed_report(d, scrub):
    """The report text, scrubbed. The record's JSON is scrubbed value by value BEFORE it is written (scrubbing the dumped text would turn a Windows path's
    escaped backslashes into invalid JSON, and an issue's record could then not be read back)."""
    if not d.get("record"):
        return scrub(report_md(d))
    block = json.dumps(scrub_all(rec_for_issue(d["record"], bool(d.get("chat"))), scrub), indent=1, ensure_ascii=False)
    return scrub(report_md(d, record_json=RECORD_HOLE)).replace(RECORD_HOLE, block)


def report_md(d, record_json=None):
    """The report as GitHub markdown, sections in a fixed order (tests/bench_report_test.py checks them)."""
    s, f = d["summary"], d["facts"]
    v = lambda x, unit="": "-" if x is None else f"{x}{unit}"
    lines = [f"## moe-cache benchmark report",
             "",
             f"moe-cache {d['version']} · report schema {SCHEMA} · {d['date']} UTC · **{d['start']}** · mode {d['mode']}",
             "",
             *([f"**Long chat** (the published method: one real multi-turn conversation filled to a {d['chat']['ctx']}-token context, thinking off, 256-token answers; "
                f"prompt and decode tok/s are means over the whole conversation)", ""] if d.get("chat") else []),
             f"**Result:** moe-cache {v(s['moe-cache']['decode'])} tok/s vs stock {v(s['stock']['decode'])} tok/s decode ({v(s['decode_x'], 'x')}), "
             f"prompt {v(s['moe-cache']['prefill'])} vs {v(s['stock']['prefill'])} tok/s, tokens {s['identity']}",
             "",
             f"**Verdict:** {s['noise']['sentence']}",
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
             f"| Identity | {d['model'].get('identity', '-')} |",
             f"| Launcher plan | {d['plan']} |",
             f"| Stock plan | {d.get('stock_plan', '-')} |",
             f"| Fits in RAM | {d['fits']} |",
             "",
             *(["| | stock | moe-cache | ratio |", "|---|---|---|---|",
                f"| decode tok/s (mean over the conversation; per run) | {v(s['stock']['decode'])}; {s['stock']['decode_runs']} | {v(s['moe-cache']['decode'])}; {s['moe-cache']['decode_runs']} | {v(s['decode_x'], 'x')} |",
                f"| prompt tok/s (new tokens read; per run) | {v(s['stock']['prefill'])}; {s['stock']['prefill_runs']} | {v(s['moe-cache']['prefill'])}; {s['moe-cache']['prefill_runs']} | {v(s['prefill_x'], 'x')} |",
                f"| time to first token, median s | {v(s['stock']['ttft_s'])} | {v(s['moe-cache']['ttft_s'])} | |",
                f"| server start, s | {v(s['stock']['load_s'])} | {v(s['moe-cache']['load_s'])} | |",
                f"| cache hit rate (decode) | | {v(s['moe-cache']['hit_rate'], ' %')} | |",
                "", d["chat"]["bands"]] if d.get("chat") else [
                "| | stock | moe-cache | ratio |", "|---|---|---|---|",
                f"| decode tok/s (mean; runs) | {v(s['stock']['decode'])}; {s['stock']['decode_runs']} | {v(s['moe-cache']['decode'])}; {s['moe-cache']['decode_runs']} | {v(s['decode_x'], 'x')} |",
                f"| prompt tok/s ({d['prompt_tokens']} tokens) | {v(s['stock']['prefill'])} | {v(s['moe-cache']['prefill'])} | {v(s['prefill_x'], 'x')} |",
                f"| time to first token, s | {v(s['stock']['ttft_s'])} | {v(s['moe-cache']['ttft_s'])} | |",
                f"| server start, s | {v(s['stock']['load_s'])} | {v(s['moe-cache']['load_s'])} | |",
                *([f"| agent turns: time to first token, s (turn 1 / 2 / 3, {d.get('gap', 0)} s gaps) | {' / '.join(map(str, s['stock'].get('agent_ttft_s') or ['-']))} | "
                   f"{' / '.join(map(str, s['moe-cache'].get('agent_ttft_s') or ['-']))} | |"] if d.get("scenario") == "agent" else []),
                f"| cache hit rate | | {v(s['moe-cache']['hit_rate'], ' %')} | |",
                f"| read from the drive | | {v(s['moe-cache']['gb_read'], ' GB')} | |",
                ""]),
             f"Checks: tokens {s['identity']}; plugin active {'yes' if s['plugin_active'] else 'NO'}; order {' / '.join(d['order'])}",
             ]
    if d["notes"]:
        lines += ["", "Notes:"] + [f"- {n}" for n in d["notes"]]
    if d.get("record"):         # the benchmark record (docs/BENCHMARK_TEMPLATE.md): the card's detail lines, then the whole record for the issue form
        lines += ["", "Details (benchmark card format):", ""] + ["- " + l for l in rec.details_lines(d["record"])]
        gaps = rec.not_recorded_fields(d["record"])
        if gaps:
            lines += ["", f"Not recorded on this machine ({len(gaps)} fields): " + ", ".join(g.replace("$.", "") for g in gaps[:12]) + (" ..." if len(gaps) > 12 else "")]
        lines += ["", "<details><summary>Record (JSON)</summary>", "", "```json", record_json if record_json is not None else json.dumps(rec_for_issue(d["record"], bool(d.get("chat"))), indent=1, ensure_ascii=False), "```", "", "</details>"]
    lines += ["", "_No prompts, answers, user name, host name or folder names are in this report. Nothing was uploaded._"]
    return "\n".join(lines) + "\n"


def build_record(a, argv, d, runs, devices, llama, plan_line, stock_line, conditions):
    """The benchmark record for this bench-report (docs/BENCHMARK_TEMPLATE.md): machine, software, model, settings, conditions, protocol, per-run metrics."""
    env = rec.collect_environment(a.model, None, listing=devices, version_line=llama, backend=d["backend"], identity={"quant": None if d["model"].get("quant") in (None, "-") else d["model"]["quant"]}, model_url=a.model_url)
    env["software"]["moe_cache_version"] = mc.VERSION
    moe_run = next((r for r in runs if r["mode"] == "moe-cache" and r.get("log_facts")), None) or next((r for r in runs if r.get("log_facts")), {})
    st = rec.settings_from(plan_line, env={}, log_lines=(moe_run.get("log_facts") or {}).get("lines"),
                           extra={"stock_command_line": stock_line, "fast_flag": "yes" if a.fast else "no"})
    entries = [rec.run_entry(r["mode"], i + 1, r, r.get("monitor")) for i, r in enumerate(runs)]
    long_prompt = "Summarise the following text in five bullet points.\n" + PARA * max(1, a.prompt_tokens // 37)
    ntok = [r["prompt_tokens"] for r in runs if isinstance(r.get("prompt_tokens"), int)]
    proto = {"arms": ["stock", "moe-cache"], "order": "ABBA", "runs_per_arm": 2 * a.pairs, "prompt_set": "bench-report/chat-v1",
             "prompt_sha256": rec.sha256_text(long_prompt + "\n" + DECODE_PROMPT)[:16], "prompt_tokens": ntok[0] if ntok else a.prompt_tokens,
             "n_predict": a.tokens, "temperature": 0.0, "seed": 42, "sampler": "greedy (temperature 0, seed 42)"}
    cmd = "python bin/moe_cache.py bench-report " + os.path.basename(a.model) + " " + " ".join(x for x in argv if x != a.model and not x.endswith(".gguf"))
    return rec.assemble(env, st, conditions, proto, entries, notes=d["notes"],
                        reproduce={"commands": [cmd.strip()], "kit": "https://github.com/IbrahimSiddiqui007/moe-cache/releases", "model_url": a.model_url or rec.NR, "prompt_file": "built into bin/moe_cache_bench.py (PARA, DECODE_PROMPT)"})


def build_chat_record(a, d, chat_runs, devices, llama, cond):
    """The long chat's record: tools/conversation_bench.py build_record on the runs this report played (ABBA), the cold / warm stamp of this report."""
    import types
    import conversation_bench as cb
    ns = types.SimpleNamespace(model=a.model, llama_server=a.llama_server, backend=d["backend"], model_url=a.model_url, command_line=None, server_cmd=None,
                               cache="warm" if a.warm else "cold", baseline=a.baseline, other_programs=a.other_programs)
    r = cb.build_record(chat_runs, ns)
    r["software"]["moe_cache_version"] = mc.VERSION
    r["notes"] = list(r.get("notes", [])) + d["notes"]
    return r


def bench_report(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py bench-report", description="measure stock llama-server and moe-cache on this machine and write a report "
                                 "to paste into a GitHub issue (docs/SHARE_YOUR_NUMBERS.md). Measures only; nothing is uploaded.")
    ap.add_argument("model")
    ap.add_argument("--llama-server", default=mc.default_llama_server())
    ap.add_argument("--ram", help="GB the cache may use (as `server`); default: what is free")
    ap.add_argument("--ctx", type=int, default=None, help="context of the servers (default 8192; with --chat: from the RAM, 16k to 65k)")
    ap.add_argument("--chat", action="store_true", help="long chat (the published method): each run plays one real multi-turn conversation filled to --ctx, tools/conversation_bench.py")
    ap.add_argument("--max-turns", type=int, default=0, help="--chat: stop each conversation after this many turns (0 = fill the context; for a trial)")
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
    ap.add_argument("--baseline", default="", help="the stock-speed check you ran before (e.g. 'PASS 49.4 tok/s'); goes into the record")
    ap.add_argument("--other-programs", default="", help="what else was running (e.g. 'nothing'); goes into the record")
    ap.add_argument("--model-url", default="", help="where the model file comes from (Hugging Face page); goes into the record")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--load-timeout", type=int, default=1800)
    a = ap.parse_args(argv)
    a.extra = ["--fast=" + a.fast] if a.fast else []
    if a.chat and a.scenario == "agent":
        mc.say("bench-report: --chat and --scenario agent are two different tests: use one", err=True); return 2
    if a.chat and a.fast:
        mc.say("bench-report: --chat measures the exact mode; leave --fast out", err=True); return 2
    ram_gb = mc.win_memory()[0] / 1e9 if mc.WINDOWS and mc.win_memory()[0] else mc.meminfo_total_gb()
    a.ctx = a.ctx or (chat_ctx(ram_gb) if a.chat else 8192)
    mc.say("moe-cache bench-report: about " + ("30-120 minutes for a long chat (4 conversations per pair)" if a.chat else "5-20 minutes") + "; leave the machine alone (close the browser and games: they hold GPU memory)")
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
    chat_extra = ["--agent"] if a.chat else []
    plan_mode, plan_line = launcher_plan(a.model, a.llama_server, a.ram, a.ctx, a.extra + chat_extra)
    stock_line = launcher_plan(a.model, a.llama_server, a.ram, a.ctx, ["--stock"] + chat_extra)[1]
    fits = plan_mode == "stock"
    if fits and not a.force_plugin:
        mc.say("bench-report: this model fits in your RAM, so moe-cache would not be used (the launcher runs stock llama.cpp). "
               "Pick a bigger model (`moe_cache.py pick`) or pass --force-plugin.", err=True)
        return 3
    import moe_cache_ui as ui
    notes, start = [], "WARM" if a.warm else "COLD"
    other = [x for x in a.flush_from.split(",") if x]
    runs = []
    workdir = tempfile.mkdtemp(prefix="moe-cache-chat-") if a.chat else None
    key = mc.stored_key(create=True) if a.chat else ""
    cond = rec.conditions_now(cold=None if a.warm else True if start == "COLD" else False, baseline=a.baseline, other_programs=a.other_programs)      # one second of CPU sampling; uptime, free RAM, VRAM in use, temperatures
    for i, mode in enumerate(order(a.pairs)):
        if not a.warm:
            ok_, why = ui.flush_model_cache(a.model, other)
            if not ok_:
                if start == "COLD":
                    notes.append(f"cold start failed ({why}): stock and moe-cache are not comparable with the README rows (WARM)")
                start = "WARM"
        mc.say(f"bench-report: run {i + 1}/{len(order(a.pairs))}: {mode} ...")
        if mc.port_busy(a.port):
            r = {"mode": mode, "error": f"port {a.port} still in use by the previous server"}
        elif a.chat:
            r = run_chat_arm(a.model, mode, a, a.port, workdir, key, plan_line if mode != "stock" else stock_line)
        else:
            r = run_one(a.model, mode, a, a.port)
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
    s_ = summarize(runs, exact=not a.fast and not a.chat)
    if a.chat:
        s_["identity"] = "not checked (long chat)"
    if s_["identity"] == "DIFFERENT":
        notes.append("tokens differ: stock and moe-cache ran with the launcher's own plans (stock plan above), and a layer computed on the GPU instead of "
                     "the CPU can change the last digits and then a token; the same-split identity proof is the CI identity suite. Please report it anyway")
    d = {"schema": SCHEMA, "version": mc.VERSION, "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"), "start": start,
         "mode": f"fast={a.fast}" if a.fast else "exact, long chat" if a.chat else "exact", "os": f"{platform.system()} {platform.release()} ({platform.version()})", "facts": facts,
         "backend": backend_of(devices), "llama": llama,
         "model": model_section(a.model, size, ram_total),
         "plan": plan_summary(plan_line), "scenario": a.scenario, "gap": a.gap, "fits": "yes (forced)" if fits else "no", "prompt_tokens": a.prompt_tokens, "order": order(a.pairs),
         "summary": s_, "notes": notes, "stock_plan": plan_summary(stock_line),
         "runs": [{k: v for k, v in r.items() if k not in ("text", "log_facts", "chat")} for r in runs]}
    chat_runs = [r["chat"] for r in runs if r.get("chat")]
    if a.chat and chat_runs:
        import conversation_bench as cb
        d["chat"] = {"ctx": a.ctx, "bands": cb.band_table(chat_runs)}
        for r in chat_runs:
            for w in (r.get("answer_stats") or {}).get("warnings", []):
                notes.append(f"{r['arm']}: {w}")
    try:
        cond["file_cache"] = "cold" if start == "COLD" else "warm"
        d["record"] = build_chat_record(a, d, chat_runs, devices, llama, cond) if a.chat and len(chat_runs) == len(runs) else build_record(a, argv, d, runs, devices, llama, plan_line, stock_line, cond) if not a.chat else None
        if a.chat and d["record"] is None:
            notes.append("no benchmark record: not every conversation finished")
    except Exception as e:          # the record is an addition: a report without it is still a report
        notes.append(f"benchmark record not built ({type(e).__name__}: {e})")
        d["notes"] = notes
    scrub = scrubber()
    md = scrubbed_report(d, scrub)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, "moe-cache-report-" + datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    open(base + ".md", "w", encoding="utf-8").write(md)
    open(base + ".json", "w", encoding="utf-8").write(json.dumps(scrub_all(d, scrub), indent=1, ensure_ascii=False))
    if a.chat and d.get("record"):         # the whole record with every turn: what `bench-record ingest` takes
        open(base + ".record.json", "w", encoding="utf-8").write(json.dumps(scrub_all(d["record"], scrub), indent=1, ensure_ascii=False))
    mc.say("")
    mc.say(md)
    mc.say(f"bench-report: written {base}.md and .json. Paste the .md into a new issue: https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=benchmark_report.yml")
    mc.say("bench-report: no prompts, answers, user name, host name or folder names are in the report; nothing was uploaded.")
    return 0
