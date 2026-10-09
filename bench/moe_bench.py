#!/usr/bin/env python3
"""moe-cache-bench: run speed / identity / custom-prompt / perplexity benchmarks for a model with stock llama.cpp and with moe-cache,
and look at the results as graphs in a local web page. Python standard library only.

    bin/moe-cache-bench              opens http://127.0.0.1:8765
    bin/moe-cache-bench --port 9000 --no-browser

Every server run happens in its own memory-limited scope (systemd-run, swap off) so a model bigger than RAM can never take the desktop down.
Results are saved in ~/.local/share/moe-cache-bench/results (override with MOE_BENCH_DATA).
"""
import argparse, csv, glob, html, io, json, os, re, shlex, shutil, signal, socket, subprocess, sys, threading, time, urllib.error, urllib.request, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = Path(os.environ.get("MOE_BENCH_DATA", str(Path.home() / ".local/share/moe-cache-bench")))
RESULTS = DATA / "results"
CONFIG = DATA / "config.json"
STOCK_PROMPTS = [
    "Write a Python function that merges overlapping intervals and explain its time complexity.",
    "Explain how a hash table handles collisions, with examples.",
    "Write a SQL query that finds the top 3 customers by revenue per month, and explain which indexes would help.",
    "Explain the difference between processes and threads in Linux and show a small C example of each.",
]
C_STOCK, C_PLUG = "#9aa0a6", "#2f81f7"


# ---------------------------------------------------------------- system information
def sh(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def parse_cpulist(t):
    r = []
    for part in t.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            r += range(int(a), int(b) + 1)
        elif part.strip():
            r.append(int(part))
    return r


def pcore_mask():
    """Hybrid Intel CPU: mask with one thread per performance core (None otherwise)."""
    f = Path("/sys/devices/cpu_core/cpus")
    if not f.exists():
        return None
    try:
        prim = [c for c in parse_cpulist(f.read_text())
                if min(parse_cpulist(Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read_text())) == c]
        return hex(sum(1 << c for c in prim))
    except Exception:
        return None


def mem_gb():
    d = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":")
        d[k] = int(v.split()[0]) / 1048576
    return d.get("MemTotal", 0), d.get("MemAvailable", 0)


def temp_c():
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            name = Path(hw, "name").read_text().strip()
        except Exception:
            continue
        if name in ("coretemp", "k10temp", "zenpower"):
            best = None
            for lab in glob.glob(hw + "/temp*_label"):
                if Path(lab).read_text().strip() in ("Package id 0", "Tctl", "Tdie"):
                    best = lab.replace("_label", "_input")
            best = best or hw + "/temp1_input"
            try:
                return int(Path(best).read_text()) / 1000
            except Exception:
                pass
    for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            if "pkg" in Path(z.replace("/temp", "/type")).read_text():
                return int(Path(z).read_text()) / 1000
        except Exception:
            pass
    return None


def bg_load():
    """CPU percent used by everything except the benchmark itself."""
    tot = 0.0
    for l in sh(["ps", "-eo", "pcpu,comm"]).splitlines()[1:]:
        try:
            p, c = l.strip().split(None, 1)
        except ValueError:
            continue
        if c in ("llama-server", "llama-perplexity", "ps", "systemd-run") or c.startswith("python"):
            continue
        tot += float(p)
    return tot


def system_info(llama_server=None):
    cpu = ""
    for l in open("/proc/cpuinfo"):
        if l.startswith("model name"):
            cpu = l.split(":", 1)[1].strip()
            break
    tot, av = mem_gb()
    devs = []
    if llama_server and os.access(llama_server, os.X_OK):
        for l in sh([llama_server, "--list-devices"], 40).splitlines():
            m = re.match(r"\s+(\w+\d*): (.+?) \((\d+) MiB", l)
            if m:
                devs.append(f"{m.group(1)}: {m.group(2)} ({int(m.group(3)) // 1024} GB)")
    return {"cpu": cpu, "threads": os.cpu_count(), "hybrid": pcore_mask() is not None, "ram_gb": round(tot, 1), "ram_free_gb": round(av, 1),
            "temp_c": temp_c(), "devices": devs, "systemd_run": bool(shutil.which("systemd-run")), "kernel": os.uname().release}


def find_llama_server():
    for c in [os.environ.get("LLAMA_SERVER", ""), shutil.which("llama-server") or "", str(Path.home() / "moe-cache-setup/llama.cpp/build/bin/llama-server")]:
        if c and os.access(c, os.X_OK):
            return c
    return ""


def default_config():
    tot, _ = mem_gb()
    lib = str(REPO / "build/libggml-moe-cache.so")
    return {"llama_server": find_llama_server(), "plugin_lib": lib if os.path.exists(lib) else "", "model": "", "modes": ["stock", "plugin"],
            "limits": [max(4, int(tot * 0.8))], "ctx": 8192, "n_cpu_moe": "auto", "pin": True, "extra_args": "",
            "speed": True, "speed_tokens": 100, "speed_prompts": 4, "long": True, "custom": "", "custom_tokens": 100,
            "ppl_file": "", "ppl_chunks": 8, "both_orders": False, "gate_temp": 58, "gate_wait_s": 300}


def model_size_gb(path):
    try:
        m = re.match(r"(.*)-(\d+)-of-(\d+)\.gguf$", path)
        if m:
            return sum(os.path.getsize(f) for f in glob.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf")) / 1e9
        return os.path.getsize(path) / 1e9
    except Exception:
        return 0.0


# ---------------------------------------------------------------- running servers
def load_env(llama_server, lib):
    ld = sh(["ldd", llama_server])
    if "libggml-cpu" in ld:
        return {"LD_PRELOAD": lib, "MOE_CACHE_PRELOAD": "1"}, "preload"
    return {"GGML_BACKEND_PATH": lib}, "path"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def planner_ncmoe(cfg, limit):
    py = os.environ.get("MOE_CACHE_PYTHON", sys.executable)
    cmd = [py, str(REPO / "planner/plan.py"), cfg["model"], "--ctx", str(cfg["ctx"]), "--json"]
    if limit:
        cmd += ["--ram-gib", str(limit)]
    try:
        return int(json.loads(sh(cmd, 120))["n_cpu_moe"])
    except Exception:
        return 999


def stop_proc(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(40)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def complete(port, prompt, n):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "seed": 42, "cache_prompt": False, "return_tokens": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    t = r.get("timings", {})
    return {"gen_tps": t.get("predicted_per_second", 0.0), "prompt_tps": t.get("prompt_per_second", 0.0), "prompt_n": t.get("prompt_n", 0),
            "gen_n": t.get("predicted_n", 0), "ttft_ms": t.get("prompt_ms", 0.0), "wall_s": time.time() - t0, "tokens": r.get("tokens", []),
            "text": r.get("content", "")[:600]}


def long_prompt(n_words=1500):
    words = "alpha river stone engine memory window cable forest signal garden planet silver copper orbit lantern harbor meadow pencil rocket violet".split()
    import random
    rnd = random.Random(7)
    return "Summarize the following text in two sentences.\n\n" + " ".join(rnd.choice(words) for _ in range(n_words))


def parse_custom(text):
    """Blocks separated by a line with only '---'. A line 'EXPECT: word' inside a block is a required (case-insensitive) substring of the answer."""
    out = []
    for blk in re.split(r"(?m)^\s*---\s*$", text or ""):
        lines = [l for l in blk.strip().splitlines()]
        exp = [l.split(":", 1)[1].strip() for l in lines if l.upper().startswith("EXPECT:")]
        prompt = "\n".join(l for l in lines if not l.upper().startswith("EXPECT:")).strip()
        if prompt:
            out.append({"prompt": prompt, "expect": exp})
    return out


class Job:
    def __init__(self, jid, cfg):
        self.id, self.cfg = jid, cfg
        self.state, self.done, self.total = "queued", 0, 1
        self.log, self.result_id, self.error = [], None, ""
        self.cancel, self.proc = False, None

    def say(self, s):
        self.log.append(f"{time.strftime('%H:%M:%S')} {s}")
        self.log = self.log[-400:]

    def public(self):
        return {"id": self.id, "state": self.state, "done": self.done, "total": self.total, "log": self.log[-40:], "result_id": self.result_id, "error": self.error}


JOBS = {}


def gate(job, cfg):
    """Wait until the CPU is cool and the machine is quiet (the clean-run rule). Returns (start_temp, bg, clean)."""
    limit_t, deadline = float(cfg["gate_temp"]), time.time() + float(cfg["gate_wait_s"])
    t = temp_c()
    while t is not None and t > limit_t and time.time() < deadline and not job.cancel:
        job.say(f"waiting for the CPU to cool: {t:.0f} C (target {limit_t:.0f} C)")
        time.sleep(10)
        t = temp_c()
    bg = bg_load()
    clean = (t is None or t <= limit_t + 4) and bg < 15
    return t, bg, clean


def run_one(job, cfg, mode, limit, rep, tag):
    """Start one server (stock or plugin) under a memory limit, run the selected suites against it, stop it."""
    model, ls = cfg["model"], cfg["llama_server"]
    st = {"mode": mode, "limit_gb": limit, "repeat": rep, "requests": [], "suites": []}
    t, bg, clean = gate(job, cfg)
    st.update(start_temp=t, bg_cpu=round(bg, 1), clean=clean)
    ncm = cfg["n_cpu_moe"]
    ncm = planner_ncmoe(cfg, limit) if ncm == "auto" else (999 if ncm == "all" else int(ncm))
    st["n_cpu_moe"] = ncm
    port = free_port()
    prefix = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={limit}G", "-p", "MemorySwapMax=0"] if limit else []
    envs = {}
    if mode == "plugin":
        e, method = load_env(ls, cfg["plugin_lib"])
        envs.update(e)
        envs.update(MOE_CACHE_SIZE_GIB="auto", MOE_CACHE_GGUF=model, MOE_CACHE_STATS="1")
        st["load_method"] = method
    args = [ls, "-m", model, "-ngl", "99", "--n-cpu-moe", str(ncm), "-c", str(cfg["ctx"]), "-fa", "on", "--load-mode", "mmap", "--no-warmup",
            "--port", str(port), "--seed", "42", "-np", "1"]
    if cfg.get("pin"):
        m = pcore_mask()
        if m:
            args += ["-C", m, "--cpu-strict", "1"]
    args += shlex.split(cfg.get("extra_args") or "")
    cmd = prefix + (["env"] + [f"{k}={v}" for k, v in envs.items()] if envs else []) + args
    logp = DATA / f"server_{job.id}_{tag}.log"
    job.say(f"starting {mode} server (limit {limit} GB, experts on CPU layers: {ncm})")
    with open(logp, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        job.proc = p
        try:
            t0 = time.time()
            while True:
                if job.cancel or p.poll() is not None:
                    raise RuntimeError("server stopped before it was ready" if not job.cancel else "cancelled")
                try:
                    if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).status == 200:
                        break
                except Exception:
                    pass
                if time.time() - t0 > 1800:
                    raise RuntimeError("server did not become ready in 30 minutes")
                time.sleep(2)
            job.say("server ready")
            reqs = []
            if cfg["speed"]:
                for i, pr in enumerate(STOCK_PROMPTS[:max(1, min(4, int(cfg["speed_prompts"])))]):
                    if job.cancel:
                        raise RuntimeError("cancelled")
                    r = complete(port, pr, int(cfg["speed_tokens"]))
                    r.update(suite="speed", name=f"prompt {i + 1}", index=i + 1)
                    reqs.append(r)
                    job.say(f"  speed prompt {i + 1}: {r['gen_tps']:.2f} tok/s generation, {r['prompt_tps']:.1f} tok/s prompt")
                st["suites"].append("speed")
            if cfg["long"]:
                if job.cancel:
                    raise RuntimeError("cancelled")
                r = complete(port, long_prompt(), 16)
                r.update(suite="long", name="long prompt", index=1)
                reqs.append(r)
                job.say(f"  long prompt ({r['prompt_n']} tokens): {r['prompt_tps']:.1f} tok/s prompt processing")
                st["suites"].append("long")
            cust = parse_custom(cfg.get("custom", ""))
            for i, c in enumerate(cust):
                if job.cancel:
                    raise RuntimeError("cancelled")
                r = complete(port, c["prompt"], int(cfg["custom_tokens"]))
                ok = all(x.lower() in r["text"].lower() or x.lower() in r.get("text", "").lower() for x in c["expect"]) if c["expect"] else None
                r.update(suite="custom", name=f"custom {i + 1}", index=i + 1, expect=c["expect"], passed=ok, prompt=c["prompt"][:200])
                reqs.append(r)
                job.say(f"  custom prompt {i + 1}: {r['gen_tps']:.2f} tok/s" + ("" if ok is None else (" PASS" if ok else " FAIL")))
            if cust:
                st["suites"].append("custom")
            st["requests"] = reqs
        finally:
            stop_proc(p)
            job.proc = None
    try:
        txt = Path(logp).read_text(errors="ignore")
        m = re.findall(r"hits=(\d+) misses=(\d+) \(hit ([\d.]+)%\) read=([\d.]+) MB evictions=(\d+)", txt)
        if m:
            h, mi, hp, rd, ev = m[-1]
            st["plugin_stats"] = {"hit_pct": float(hp), "read_gb": float(rd) / 1000, "evictions": int(ev)}
    except Exception:
        pass
    return st


def run_perplexity(job, cfg, runs):
    """Stock records reference logits, the plugin run compares against them (KL divergence), both report perplexity."""
    ppl = str(Path(cfg["llama_server"]).with_name("llama-perplexity"))
    if not os.access(ppl, os.X_OK) or not os.path.exists(cfg["ppl_file"]):
        job.say("perplexity skipped: llama-perplexity or the text file was not found")
        return None
    limit = cfg["limits"][0]
    out = {"file": cfg["ppl_file"], "chunks": cfg["ppl_chunks"]}
    kld = DATA / f"ref_{job.id}.kld"
    for mode in [m for m in ("stock", "plugin") if m in cfg["modes"]]:
        job.say(f"perplexity, {mode} (can take a long time)")
        args = [ppl, "-m", cfg["model"], "-f", cfg["ppl_file"], "-c", "512", "--chunks", str(cfg["ppl_chunks"]), "-ngl", "99", "--n-cpu-moe", "999"]
        args += ["--kl-divergence-base", str(kld)] + (["--kl-divergence"] if mode == "plugin" and kld.exists() else [])
        envs = {}
        if mode == "plugin":
            e, _ = load_env(cfg["llama_server"], cfg["plugin_lib"])
            envs.update(e)
            envs.update(MOE_CACHE_SIZE_GIB="auto", MOE_CACHE_GGUF=cfg["model"])
        prefix = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={limit}G", "-p", "MemorySwapMax=0"] if limit else []
        cmd = prefix + (["env"] + [f"{k}={v}" for k, v in envs.items()] if envs else []) + args
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        job.proc = p
        txt = p.communicate()[0]
        job.proc = None
        m = re.search(r"Final estimate: PPL = ([\d.]+)", txt)
        out[mode] = {"ppl": float(m.group(1)) if m else None}
        k = re.search(r"Mean\s+KLD:\s+([\d.eE+-]+)", txt)
        s = re.search(r"Same top p:\s+([\d.]+)", txt)
        if k:
            out[mode]["mean_kld"] = float(k.group(1))
        if s:
            out[mode]["same_top_pct"] = float(s.group(1))
    try:
        kld.unlink()
    except Exception:
        pass
    return out


def summarize(runs):
    """runs -> numbers used by the charts and tables."""
    sm = {"speed": {}, "long": {}, "identity": {}, "custom": {}, "plugin": {}, "clean": {}}
    keyf = lambda r: f"{r['limit_gb']}|{r['mode']}"
    for r in runs:
        k = keyf(r)
        sp = [q for q in r["requests"] if q["suite"] == "speed"]
        if sp:
            g = [q["gen_tps"] for q in sp]
            e = sm["speed"].setdefault(k, {"first": [], "steady": [], "per_req": [[] for _ in g]})
            e["first"].append(g[0])
            if len(g) > 1:
                e["steady"].append(sum(g[1:]) / (len(g) - 1))
            for i, x in enumerate(g):
                e["per_req"][i].append(x)
        for q in r["requests"]:
            if q["suite"] == "long":
                sm["long"].setdefault(k, []).append(q["prompt_tps"])
            if q["suite"] == "custom" and q.get("passed") is not None:
                c = sm["custom"].setdefault(k, [0, 0])
                c[1] += 1
                c[0] += 1 if q["passed"] else 0
        if "plugin_stats" in r:
            sm["plugin"][k] = r["plugin_stats"]
        sm["clean"].setdefault(k, []).append(bool(r.get("clean")))
    mean = lambda v: sum(v) / len(v) if v else None
    for k, e in sm["speed"].items():
        e["first"], e["steady"] = mean(e["first"]), mean(e["steady"])
        e["per_req"] = [mean(v) for v in e["per_req"]]
    sm["long"] = {k: mean(v) for k, v in sm["long"].items()}
    for lim in sorted({r["limit_gb"] for r in runs}):
        a = next((r for r in runs if r["limit_gb"] == lim and r["mode"] == "stock"), None)
        b = next((r for r in runs if r["limit_gb"] == lim and r["mode"] == "plugin"), None)
        if a and b:
            ta = [q["tokens"] for q in a["requests"] if q["suite"] in ("speed", "custom")]
            tb = [q["tokens"] for q in b["requests"] if q["suite"] in ("speed", "custom")]
            n = min(len(ta), len(tb))
            sm["identity"][str(lim)] = [sum(1 for i in range(n) if ta[i] == tb[i]), n]
    return sm


def run_job(job):
    cfg = job.cfg
    job.state = "running"
    try:
        modes = [m for m in ("stock", "plugin") if m in cfg["modes"]]
        limits = cfg["limits"] or [None]
        plan = []
        for rep in range(2 if cfg["both_orders"] else 1):
            for lim in limits:
                for m in (modes if rep == 0 else modes[::-1]):
                    plan.append((m, lim, rep))
        job.total = len(plan) + (1 if cfg["ppl_file"] else 0)
        runs = []
        for i, (m, lim, rep) in enumerate(plan):
            if job.cancel:
                break
            job.say(f"run {i + 1}/{len(plan)}: {m}, limit {lim} GB, repeat {rep + 1}")
            try:
                runs.append(run_one(job, cfg, m, lim, rep, f"{i}"))
            except Exception as e:
                job.say(f"run failed: {e}")
                if job.cancel:
                    break
                runs.append({"mode": m, "limit_gb": lim, "repeat": rep, "requests": [], "suites": [], "error": str(e), "clean": False})
            job.done = i + 1
        ppl = run_perplexity(job, cfg, runs) if (cfg["ppl_file"] and not job.cancel) else None
        job.done = job.total
        rid = time.strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"[^A-Za-z0-9._-]", "_", Path(cfg["model"]).stem)[:40]
        RESULTS.mkdir(parents=True, exist_ok=True)
        res = {"id": rid, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg, "system": system_info(cfg["llama_server"]),
               "model_gb": round(model_size_gb(cfg["model"]), 2), "runs": runs, "summary": summarize([r for r in runs if r["requests"]]), "perplexity": ppl}
        (RESULTS / f"{rid}.json").write_text(json.dumps(res))
        job.result_id = rid
        job.state = "cancelled" if job.cancel else "done"
    except Exception as e:
        job.state, job.error = "failed", str(e)
        job.say(f"failed: {e}")


def start_job(cfg):
    errs = []
    if not (cfg["llama_server"] and os.access(cfg["llama_server"], os.X_OK)):
        errs.append("llama-server path is not an executable file")
    if not os.path.isfile(cfg["model"]):
        errs.append("model file not found")
    if "plugin" in cfg["modes"] and not os.path.isfile(cfg["plugin_lib"]):
        errs.append("plugin library not found (build it with ./build.sh)")
    if not cfg["modes"]:
        errs.append("pick stock, plugin or both")
    tot, av = mem_gb()
    size = model_size_gb(cfg["model"]) if os.path.isfile(cfg["model"]) else 0
    if not cfg["limits"] and size > 0.5 * tot:
        errs.append("this model is large for your RAM: set a memory limit (GB) so a runaway server cannot take the desktop down")
    if cfg["limits"] and not shutil.which("systemd-run"):
        errs.append("systemd-run not found: memory limits need systemd (use a model that fits in RAM and clear the limits)")
    if errs:
        return None, errs
    jid = time.strftime("%H%M%S") + f"{len(JOBS):02d}"
    job = Job(jid, cfg)
    JOBS[jid] = job
    DATA.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg))
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return job, []


# ---------------------------------------------------------------- charts (SVG, theme aware)
def esc(s):
    return html.escape(str(s))


def nice_max(v):
    if v <= 0:
        return 1.0
    import math
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if v <= m * e:
            return m * e
    return 10 * e


def svg_wrap(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(title)}" '
            f'font-family="system-ui,Helvetica,Arial,sans-serif"><style>.t{{fill:#1f2328}}.m{{fill:#656d76}}.g{{stroke:#d0d7de}}@media(prefers-color-scheme:dark){{.t{{fill:#e6edf3}}.m{{fill:#8b949e}}.g{{stroke:#30363d}}}}.t{{fill:var(--fg,#1f2328)}}.m{{fill:var(--muted,#656d76)}}.g{{stroke:var(--grid,#d0d7de)}}</style>{body}</svg>')


def bar_chart(title, cats, series, unit="tok/s"):
    """cats: group labels; series: [(name, colour, [value|None per cat])]"""
    W, H, L, R, T, B = 860, 350, 64, 20, 60, 66
    vals = [v for _, _, vs in series for v in vs if v]
    ymax = nice_max(max(vals) * 1.15 if vals else 1)
    pw, ph = W - L - R, H - T - B
    o = [f'<text x="{L}" y="26" class="t" font-size="17" font-weight="600">{esc(title)}</text>']
    lx = L
    for name, col, _ in series:
        o.append(f'<rect x="{lx}" y="36" width="12" height="12" rx="2" fill="{col}"/><text x="{lx + 17}" y="47" class="m" font-size="12">{esc(name)}</text>')
        lx += 24 + 8 * len(name)
    for i in range(6):
        y = T + ph - ph * i / 5
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" class="g" stroke-width="1"/><text x="{L - 8}" y="{y + 4:.1f}" class="m" font-size="11" text-anchor="end">{ymax * i / 5:g}</text>')
    o.append(f'<text x="14" y="{T + ph / 2}" class="m" font-size="11" transform="rotate(-90 14 {T + ph / 2})" text-anchor="middle">{esc(unit)}</text>')
    gw = pw / max(1, len(cats))
    bw = min(70, gw * 0.7 / max(1, len(series)))
    for gi, c in enumerate(cats):
        gx = L + gw * gi + gw / 2 - bw * len(series) / 2
        for si, (_, col, vs) in enumerate(series):
            v = vs[gi]
            if v is None:
                continue
            h = ph * v / ymax
            x = gx + si * bw
            o.append(f'<rect x="{x + 2:.1f}" y="{T + ph - h:.1f}" width="{bw - 4:.1f}" height="{h:.1f}" rx="3" fill="{col}"/>'
                     f'<text x="{x + bw / 2:.1f}" y="{T + ph - h - 6:.1f}" class="t" font-size="12" text-anchor="middle" font-weight="600">{v:.2f}</text>')
        if len(series) == 2 and series[0][2][gi] and series[1][2][gi]:
            o.append(f'<text x="{L + gw * gi + gw / 2:.1f}" y="{H - 10}" fill="{C_PLUG}" font-size="14" font-weight="700" text-anchor="middle">moe-cache {series[1][2][gi] / series[0][2][gi]:.1f}x</text>')
        o.append(f'<text x="{L + gw * gi + gw / 2:.1f}" y="{H - 34}" class="t" font-size="12" text-anchor="middle">{esc(c)}</text>')
    return svg_wrap(W, H, "".join(o), title)


def line_chart(title, xs, series, unit="tok/s"):
    W, H, L, R, T, B = 860, 320, 64, 20, 56, 44
    vals = [v for _, _, vs, _ in series for v in vs if v]
    ymax = nice_max(max(vals) * 1.15 if vals else 1)
    pw, ph = W - L - R, H - T - B
    o = [f'<text x="{L}" y="26" class="t" font-size="17" font-weight="600">{esc(title)}</text>']
    lx = L
    for name, col, _, dash in series:
        o.append(f'<line x1="{lx}" y1="42" x2="{lx + 16}" y2="42" stroke="{col}" stroke-width="3" {dash}/><text x="{lx + 21}" y="47" class="m" font-size="12">{esc(name)}</text>')
        lx += 36 + 7 * len(name)
    for i in range(6):
        y = T + ph - ph * i / 5
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" class="g" stroke-width="1"/><text x="{L - 8}" y="{y + 4:.1f}" class="m" font-size="11" text-anchor="end">{ymax * i / 5:g}</text>')
    px = lambda i: L + (pw * i / max(1, len(xs) - 1) if len(xs) > 1 else pw / 2)
    for i, x in enumerate(xs):
        o.append(f'<text x="{px(i):.1f}" y="{H - 18}" class="m" font-size="12" text-anchor="middle">{esc(x)}</text>')
    for _, col, vs, dash in series:
        pts = [(px(i), T + ph - ph * v / ymax) for i, v in enumerate(vs) if v is not None]
        if len(pts) > 1:
            o.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.5" {dash} points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
        for x, y in pts:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{col}"/>')
    return svg_wrap(W, H, "".join(o), title)


def charts(res):
    sm, out = res["summary"], {}
    lims = sorted({k.split("|")[0] for k in list(sm["speed"]) + list(sm["long"])}, key=lambda x: float(x) if x != "None" else 0)
    lab = lambda l: f"{l} GB limit" if l != "None" else "no limit"
    def ser(src, getter):
        s = []
        for mode, col in (("stock", C_STOCK), ("plugin", C_PLUG)):
            vs = [getter(src.get(f"{l}|{mode}")) for l in lims]
            if any(v is not None for v in vs):
                s.append(("stock llama.cpp" if mode == "stock" else "with moe-cache", col, vs))
        return s
    if sm["speed"]:
        out["speed"] = bar_chart("Generation speed, steady state (requests 2 to end)", [lab(l) for l in lims], ser(sm["speed"], lambda e: e and e["steady"]))
        out["first"] = bar_chart("First request (cold start)", [lab(l) for l in lims], ser(sm["speed"], lambda e: e and e["first"]))
        n = max(len(e["per_req"]) for e in sm["speed"].values())
        series = []
        for k, e in sorted(sm["speed"].items()):
            l, mode = k.split("|")
            series.append((f"{'stock' if mode == 'stock' else 'moe-cache'}, {lab(l)}", C_STOCK if mode == "stock" else C_PLUG, e["per_req"], 'stroke-dasharray="6 4"' if mode == "stock" else ""))
        out["per_request"] = line_chart("Generation speed per request", [f"#{i + 1}" for i in range(n)], series)
    if sm["long"]:
        out["long"] = bar_chart("Prompt processing speed (long prompt)", [lab(l) for l in lims], ser(sm["long"], lambda e: e))
    return out


def tables_html(res):
    sm = res["summary"]
    o = []
    s = res["system"]
    o.append(f'<p class="muted">{esc(s.get("cpu", ""))} · {s.get("ram_gb", "?")} GB RAM · {esc(", ".join(s.get("devices", [])) or "no GPU backend")} · model {res["model_gb"]} GB · {esc(res["created"])}</p>')
    rows = []
    for r in res["runs"]:
        sp = [q["gen_tps"] for q in r["requests"] if q["suite"] == "speed"]
        rows.append(f'<tr><td>{r["limit_gb"] or "-"}</td><td>{r["mode"]}</td><td>{r.get("repeat", 0) + 1}</td>'
                    f'<td>{", ".join(f"{x:.2f}" for x in sp) or esc(r.get("error", "-"))}</td>'
                    f'<td>{"yes" if r.get("clean") else "<b>no</b>"} ({(r.get("start_temp") or 0):.0f} C)</td></tr>')
    o.append('<h3>Runs</h3><table><tr><th>limit GB</th><th>mode</th><th>repeat</th><th>generation tok/s per request</th><th>clean start</th></tr>' + "".join(rows) + "</table>")
    if sm["identity"]:
        o.append("<h3>Identity (same tokens, stock vs moe-cache)</h3><table><tr><th>limit GB</th><th>identical prompts</th></tr>" +
                 "".join(f"<tr><td>{k}</td><td>{a}/{b} {'PASS' if a == b else '<b>DIFFERENT</b>'}</td></tr>" for k, (a, b) in sm["identity"].items()) + "</table>")
    if sm["custom"]:
        o.append("<h3>Custom prompts with expected answers</h3><table><tr><th>run</th><th>passed</th></tr>" +
                 "".join(f"<tr><td>{esc(k.replace('|', ' GB, '))}</td><td>{a}/{b}</td></tr>" for k, (a, b) in sm["custom"].items()) + "</table>")
    if sm["plugin"]:
        o.append("<h3>moe-cache cache statistics</h3><table><tr><th>run</th><th>hit rate</th><th>read from SSD</th><th>evictions</th></tr>" +
                 "".join(f"<tr><td>{esc(k.replace('|', ' GB, '))}</td><td>{v['hit_pct']:.1f}%</td><td>{v['read_gb']:.1f} GB</td><td>{v['evictions']}</td></tr>" for k, v in sm["plugin"].items()) + "</table>")
    if res.get("perplexity"):
        p = res["perplexity"]
        o.append("<h3>Perplexity / KL divergence</h3><table><tr><th>mode</th><th>perplexity</th><th>mean KLD vs stock</th><th>same top token %</th></tr>" +
                 "".join(f"<tr><td>{m}</td><td>{p[m].get('ppl')}</td><td>{p[m].get('mean_kld', '-')}</td><td>{p[m].get('same_top_pct', '-')}</td></tr>" for m in ("stock", "plugin") if m in p) + "</table>")
    cust = [(r, q) for r in res["runs"] for q in r["requests"] if q["suite"] == "custom"]
    if cust:
        o.append("<h3>Custom prompt answers</h3>" + "".join(
            f'<details><summary>{r["mode"]} {r["limit_gb"] or ""} GB · {esc(q["name"])} · {q["gen_tps"]:.2f} tok/s'
            f'{"" if q.get("passed") is None else (" · PASS" if q["passed"] else " · FAIL")}</summary><p class="muted">{esc(q.get("prompt", ""))}</p><pre>{esc(q["text"])}</pre></details>'
            for r, q in cust))
    return "".join(o)


STYLE = ":root{--bg:#fff;--fg:#1f2328;--muted:#656d76;--grid:#d0d7de;--card:#f6f8fa;--line:#d0d7de}@media(prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#8b949e;--grid:#30363d;--card:#161b22;--line:#30363d}}body{background:var(--bg);color:var(--fg);font-family:system-ui,Helvetica,Arial,sans-serif;max-width:920px;margin:0 auto;padding:16px}table{border-collapse:collapse;width:100%;margin:8px 0}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}.muted{color:var(--muted)}pre{white-space:pre-wrap;background:var(--card);padding:8px;border-radius:6px}"


def report_html(res):
    ch = charts(res)
    return (f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>moe-cache benchmark report</title><style>{STYLE}</style>'
            f'<h1>moe-cache benchmark: {esc(Path(res["config"]["model"]).name)}</h1>' + tables_html(res) + "".join(f"<div>{v}</div>" for v in ch.values()))


def csv_export(res):
    b = io.StringIO()
    w = csv.writer(b)
    w.writerow(["limit_gb", "mode", "repeat", "suite", "name", "gen_tok_s", "prompt_tok_s", "prompt_tokens", "gen_tokens", "ttft_ms", "clean_start", "start_temp_c"])
    for r in res["runs"]:
        for q in r["requests"]:
            w.writerow([r["limit_gb"], r["mode"], r.get("repeat", 0) + 1, q["suite"], q["name"], round(q["gen_tps"], 3), round(q["prompt_tps"], 2), q["prompt_n"], q["gen_n"], round(q["ttft_ms"], 1), r.get("clean"), r.get("start_temp")])
    return b.getvalue()


# ---------------------------------------------------------------- http
def list_results():
    out = []
    for f in sorted(RESULTS.glob("*.json"), reverse=True):
        try:
            d = json.loads(f.read_text())
            sm = d["summary"]["speed"]
            best = {m: max([e["steady"] for k, e in sm.items() if k.endswith("|" + m) and e["steady"]] or [0]) for m in ("stock", "plugin")}
            out.append({"id": d["id"], "created": d["created"], "model": Path(d["config"]["model"]).name, "model_gb": d["model_gb"], "stock": best["stock"], "plugin": best["plugin"]})
        except Exception:
            continue
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def result(self, rid):
        f = RESULTS / (re.sub(r"[^A-Za-z0-9._-]", "", rid) + ".json")
        return json.loads(f.read_text()) if f.exists() else None

    def do_GET(self):
        p = self.path.split("?")[0]
        try:
            if p == "/":
                return self.send(200, (HERE / "ui.html").read_bytes(), "text/html")
            if p == "/api/config":
                cfg = default_config()
                if CONFIG.exists():
                    cfg.update(json.loads(CONFIG.read_text()))
                return self.send(200, cfg)
            if p == "/api/system":
                q = self.path.split("ls=", 1)[1] if "ls=" in self.path else find_llama_server()
                return self.send(200, system_info(urllib.request.unquote(q)))
            if p == "/api/modelinfo":
                path = urllib.request.unquote(self.path.split("path=", 1)[1]) if "path=" in self.path else ""
                return self.send(200, {"gb": round(model_size_gb(path), 2), "exists": os.path.isfile(path)})
            if p == "/api/results":
                return self.send(200, list_results())
            m = re.match(r"/api/job/(\w+)$", p)
            if m:
                j = JOBS.get(m.group(1))
                return self.send(200 if j else 404, j.public() if j else {"error": "unknown job"})
            m = re.match(r"/api/result/([\w.-]+)$", p)
            if m:
                r = self.result(m.group(1))
                if not r:
                    return self.send(404, {"error": "unknown result"})
                return self.send(200, {"result": r, "charts": charts(r), "tables": tables_html(r)})
            m = re.match(r"/report/([\w.-]+)\.html$", p)
            if m and self.result(m.group(1)):
                return self.send(200, report_html(self.result(m.group(1))), "text/html")
            m = re.match(r"/export/([\w.-]+)\.csv$", p)
            if m and self.result(m.group(1)):
                return self.send(200, csv_export(self.result(m.group(1))), "text/csv")
            m = re.match(r"/export/([\w.-]+)\.json$", p)
            if m and self.result(m.group(1)):
                return self.send(200, json.dumps(self.result(m.group(1)), indent=1))
            self.send(404, {"error": "not found"})
        except Exception as e:
            self.send(500, {"error": str(e)})

    def do_POST(self):
        p = self.path
        try:
            if p == "/api/run":
                cfg = default_config()
                cfg.update(self.body())
                cfg["limits"] = [float(x) if float(x) != int(float(x)) else int(float(x)) for x in cfg.get("limits", [])]
                job, errs = start_job(cfg)
                return self.send(400 if errs else 200, {"errors": errs} if errs else {"id": job.id})
            m = re.match(r"/api/job/(\w+)/cancel$", p)
            if m and m.group(1) in JOBS:
                j = JOBS[m.group(1)]
                j.cancel = True
                if j.proc and j.proc.poll() is None:
                    j.proc.terminate()
                return self.send(200, {"ok": True})
            m = re.match(r"/api/result/([\w.-]+)/delete$", p)
            if m:
                f = RESULTS / (re.sub(r"[^A-Za-z0-9._-]", "", m.group(1)) + ".json")
                f.unlink(missing_ok=True)
                return self.send(200, {"ok": True})
            self.send(404, {"error": "not found"})
        except Exception as e:
            self.send(500, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser(description="moe-cache benchmark GUI")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{a.port}"
    print(f"moe-cache-bench is running at {url}  (Ctrl+C to stop; results are kept in {RESULTS})")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        for j in JOBS.values():
            j.cancel = True
            if j.proc and j.proc.poll() is None:
                j.proc.terminate()


if __name__ == "__main__":
    main()
