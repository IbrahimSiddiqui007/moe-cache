#!/usr/bin/env python3
"""moe-cache-bench: run speed / identity / custom-prompt / perplexity benchmarks for a model with stock llama.cpp and with moe-cache,
and look at the results as graphs in a local web page. Python standard library only.

    bin/moe-cache-bench              opens http://127.0.0.1:8765
    bin/moe-cache-bench --port 9000 --no-browser

Every server run happens in its own memory-limited scope (systemd-run, swap off) so a model bigger than RAM can never take the desktop down.
Results are saved in ~/.local/share/moe-cache-bench/results (override with MOE_BENCH_DATA).
"""
import argparse, csv, glob, hmac, html, io, json, os, re, secrets, shlex, shutil, signal, socket, subprocess, sys, threading, time, urllib.error, urllib.request, webbrowser
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


def _cpu_times():
    d = {}
    for p in glob.glob("/proc/[0-9]*/stat"):
        try:
            x = open(p).read()
            f = x[x.rindex(")") + 2:].split()
            d[p] = (x[x.index("(") + 1:x.rindex(")")], int(f[11]) + int(f[12]))
        except Exception:
            pass
    return d


def bg_load(window=1.5):
    """CPU used right now (percent of one core) by everything except the benchmark itself, measured over a short window."""
    a = _cpu_times()
    time.sleep(window)
    b = _cpu_times()
    clk = os.sysconf("SC_CLK_TCK")
    tot = 0.0
    for p, (c, t) in b.items():
        if p in a and c not in ("llama-server", "llama-perplexity", "systemd-run") and not c.startswith("python"):
            tot += (t - a[p][1]) / clk / window * 100
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
            "limits": [max(4, int(tot * 0.8))], "ctx": 8192, "n_cpu_moe": "auto", "pin": True, "extra_args": "", "plugin_env": "",
            "speed": True, "speed_tokens": 100, "sweep": "100,1000,4000", "sweep_tokens": 64, "speed_prompts": 4, "long": True, "custom": "", "custom_tokens": 100,
            "ppl_file": "", "ppl_chunks": 8, "both_orders": False, "gate_temp": 58, "bg_max": 25, "gate_wait_s": 300}


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
    """GPU placement: as many whole layers of experts on the GPU as fit in the free VRAM (reads only the GGUF header, no extra packages)."""
    try:
        sys.path.insert(0, str(HERE))
        from gguf_plan import fit_n_cpu_moe
        n, lay, free = fit_n_cpu_moe(cfg["model"], int(cfg["ctx"]))
        return int(n)
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
    hdr = {"Content-Type": "application/json"}
    if port in API_KEYS:
        hdr["Authorization"] = "Bearer " + API_KEYS[port]
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, hdr)
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    t = r.get("timings", {})
    return {"gen_tps": t.get("predicted_per_second", 0.0), "prompt_tps": t.get("prompt_per_second", 0.0), "prompt_n": t.get("prompt_n", 0),
            "gen_n": t.get("predicted_n", 0), "ttft_ms": t.get("prompt_ms", 0.0), "wall_s": time.time() - t0, "tokens": r.get("tokens", []),
            "text": r.get("content", "")[:600]}


def long_prompt(n_words=1500):
    return words_prompt(n_words)


def words_prompt(n_words):
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
    clean = (t is None or t <= limit_t + 4) and bg < float(cfg.get("bg_max", 25))
    return t, bg, clean


# ---------------------------------------------------------------- per-request measurements
CLK = os.sysconf("SC_CLK_TCK")


def find_pid(port):
    needle = f"--port\0{port}\0".encode()
    for d in glob.glob("/proc/[0-9]*"):
        try:
            if needle in open(d + "/cmdline", "rb").read() and Path(d + "/comm").read_text().startswith("llama-server"):
                return int(d.rsplit("/", 1)[1])
        except Exception:
            pass
    return None


def gpu_mem_mb():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return float(sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], 5).split()[0])
    except Exception:
        return None


def proc_sample(pid):
    d = {"temp": temp_c(), "gpu_mb": gpu_mem_mb()}
    try:
        io = dict(l.split(": ") for l in open(f"/proc/{pid}/io").read().splitlines())
        d["read_bytes"] = int(io["read_bytes"])
        st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        d["majflt"], d["cpu_s"] = int(st[9]), (int(st[11]) + int(st[12])) / CLK
        for l in open(f"/proc/{pid}/status"):
            if l.startswith(("RssAnon", "RssFile")):
                k, v = l.split(":")
                d[k.lower()] = int(v.split()[0]) / 1048576
        base = "/sys/fs/cgroup" + open(f"/proc/{pid}/cgroup").read().strip().split("::")[-1]
        for f in ("memory.current", "memory.peak"):
            try:
                d["cg_" + f.split(".")[1]] = int(open(f"{base}/{f}").read()) / 1e9
            except Exception:
                pass
    except Exception:
        pass
    return d


def plugin_snapshot(pid, logp, seen):
    """Ask the plugin for its cumulative counters (SIGUSR1) and read the answer from the server log."""
    try:
        os.kill(pid, signal.SIGUSR1)
    except Exception:
        return None, seen
    for _ in range(50):
        time.sleep(0.1)
        lines = [l for l in open(logp, errors="ignore") if l.startswith("moe-cache-stats ")]
        if len(lines) > seen:
            try:
                return json.loads(lines[-1].split(" ", 1)[1]), len(lines)
            except Exception:
                return None, len(lines)
    return None, seen


def derive(r, s0, s1, c0, c1):
    """Per-request metrics from the before/after samples."""
    n = max(1, r["gen_n"])
    m = {"gen_tps": r["gen_tps"], "prompt_tps": r["prompt_tps"], "ttft_ms": r["ttft_ms"], "e2e_s": r["wall_s"]}
    if "read_bytes" in s0 and "read_bytes" in s1:
        m["io_mb_tok"] = (s1["read_bytes"] - s0["read_bytes"]) / 1e6 / n
        m["majflt_tok"] = (s1["majflt"] - s0["majflt"]) / n
        m["cpu_ms_tok"] = (s1["cpu_s"] - s0["cpu_s"]) * 1000 / n
    for k in ("temp", "gpu_mb", "rssanon", "rssfile", "cg_current", "cg_peak"):
        if s1.get(k) is not None:
            m[k] = s1[k]
    if c0 and c1:
        d = lambda k: c1[k] - c0[k]
        hd, md = d("hits_dec"), d("misses_dec")
        h, mi = d("hits"), d("misses")
        m["hit_dec_pct"] = 100.0 * hd / (hd + md) if hd + md else None
        m["hit_pct"] = 100.0 * h / (h + mi) if h + mi else None
        m["cache_read_mb_tok"] = d("read_bytes") / 1e6 / n
        m["evictions"] = d("evictions")
        m["evictions_tok"] = d("evictions") / n
        m["evict_age"] = d("evict_age_sum") / d("evictions") if d("evictions") else None
        m["misses_tok"] = mi / n
        wall = r["wall_s"] * 1e9
        reads, comp, prep = d("ns_read"), d("ns_cpu"), d("ns_prep")
        m["ms_tok_reads"] = reads / 1e6 / n
        m["ms_tok_compute"] = comp / 1e6 / n
        m["ms_tok_bookkeeping"] = max(0.0, prep - reads) / 1e6 / n
        m["ms_tok_other"] = max(0.0, wall - d("ns_total")) / 1e6 / n
        m["resident_gb"] = c1["resident_bytes"] / 1e9
        m["budget_gb"] = c1["budget"] / 1e9
        m["layer_hits"] = [a - b for a, b in zip(c1["layer_hits"], c0["layer_hits"] + [0] * len(c1["layer_hits"]))]
        m["layer_misses"] = [a - b for a, b in zip(c1["layer_misses"], c0["layer_misses"] + [0] * len(c1["layer_misses"]))]
    return m


def run_one(job, cfg, mode, limit, rep, tag):
    """Start one server (stock or plugin) under a memory limit, run the selected suites against it, stop it."""
    model, ls = cfg["model"], cfg["llama_server"]
    st = {"mode": mode, "limit_gb": limit, "repeat": rep, "requests": [], "suites": []}
    t, bg, clean = gate(job, cfg)
    st.update(start_temp=t, bg_cpu=round(bg, 1), clean=clean)
    ncm = cfg["n_cpu_moe"]
    ncm = planner_ncmoe(cfg, limit) if ncm in ("auto", "fit") else (999 if ncm == "all" else int(ncm))
    st["n_cpu_moe"] = ncm
    port = free_port()
    prefix = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={limit}G", "-p", "MemorySwapMax=0"] if limit else []
    envs = {}
    if mode == "plugin":
        e, method = load_env(ls, cfg["plugin_lib"])
        envs.update(e)
        envs.update(MOE_CACHE_SIZE_GIB="auto", MOE_CACHE_GGUF=model, MOE_CACHE_STATS="1", MOE_CACHE_STATS_SIGNAL="1")
        for kv in shlex.split(cfg.get("plugin_env") or ""):
            if "=" in kv:
                envs[kv.split("=", 1)[0]] = kv.split("=", 1)[1]
        st["load_method"] = method
    key = secrets.token_hex(16)
    keyf = DATA / f"key_{job.id}_{tag}"
    keyf.write_text(key)
    keyf.chmod(0o600)
    API_KEYS[port] = key       # key file, not --api-key: command lines are visible to other users
    args = [ls, "-m", model, "-ngl", "99", "--n-cpu-moe", str(ncm), "-c", str(cfg["ctx"]), "-fa", "on", "--load-mode", "mmap", "--no-warmup",
            "--port", str(port), "--seed", "42", "-np", "1", "--api-key-file", str(keyf)]
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
            pid = find_pid(port)
            seen = 0
            if mode == "plugin" and pid:
                _, seen = plugin_snapshot(pid, logp, 0)      # baseline (loading counts are not part of any request)
            reqs = []

            def ask(prompt, n, **tags):
                nonlocal seen
                if job.cancel:
                    raise RuntimeError("cancelled")
                s0 = proc_sample(pid) if pid else {}
                c0 = None
                if mode == "plugin" and pid:
                    c0, seen = plugin_snapshot(pid, logp, seen)
                r = complete(port, prompt, n)
                s1 = proc_sample(pid) if pid else {}
                c1 = None
                if mode == "plugin" and pid:
                    c1, seen = plugin_snapshot(pid, logp, seen)
                r.update(tags)
                r["m"] = derive(r, s0, s1, c0, c1)
                reqs.append(r)
                return r

            if cfg["speed"]:
                for i, pr in enumerate(STOCK_PROMPTS[:max(1, min(4, int(cfg["speed_prompts"])))]):
                    r = ask(pr, int(cfg["speed_tokens"]), suite="speed", name=f"prompt {i + 1}", index=i + 1)
                    job.say(f"  speed prompt {i + 1}: {r['gen_tps']:.2f} tok/s generation, {r['prompt_tps']:.1f} tok/s prompt")
                st["suites"].append("speed")
            if cfg["long"]:
                r = ask(long_prompt(), 16, suite="long", name="long prompt", index=1)
                job.say(f"  long prompt ({r['prompt_n']} tokens): {r['prompt_tps']:.1f} tok/s prompt processing")
                st["suites"].append("long")
            lens = [int(x) for x in re.findall(r"\d+", str(cfg.get("sweep", "")))]
            for n_in in lens:
                if n_in + 200 > int(cfg["ctx"]):
                    job.say(f"  latency sweep: {n_in} tokens skipped (context is {cfg['ctx']})")
                    continue
                r = ask(words_prompt(max(10, int(n_in / 1.46))), int(cfg.get("sweep_tokens", 64)), suite="sweep", name=f"{n_in} in", index=n_in, nominal=n_in)
                job.say(f"  sweep {n_in} tokens in: first token after {r['ttft_ms'] / 1000:.2f} s, then {r['gen_tps']:.2f} tok/s")
            if lens:
                st["suites"].append("sweep")
            cust = parse_custom(cfg.get("custom", ""))
            for i, c in enumerate(cust):
                r = ask(c["prompt"], int(cfg["custom_tokens"]), suite="custom", name=f"custom {i + 1}", index=i + 1)
                ok = all(x.lower() in r["text"].lower() for x in c["expect"]) if c["expect"] else None
                r.update(expect=c["expect"], passed=ok, prompt=c["prompt"][:200])
                job.say(f"  custom prompt {i + 1}: {r['gen_tps']:.2f} tok/s" + ("" if ok is None else (" PASS" if ok else " FAIL")))
            if cust:
                st["suites"].append("custom")
            st["requests"] = reqs
        finally:
            stop_proc(p)
            job.proc = None
            keyf.unlink(missing_ok=True)
            API_KEYS.pop(port, None)
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
    """runs -> the numbers used by the charts and tables."""
    sm = {"speed": {}, "long": {}, "sweep": {}, "identity": {}, "custom": {}, "deep": {}, "layers": {}, "clean": {}}
    mean = lambda v: sum(v) / len(v) if v else None
    for r in runs:
        k = f"{r['limit_gb']}|{r['mode']}"
        sm["clean"].setdefault(k, []).append(bool(r.get("clean")))
        sp = [q for q in r["requests"] if q["suite"] == "speed"]
        if sp:
            e = sm["speed"].setdefault(k, {"first": [], "steady": [], "per_req": [[] for _ in sp], "all": []})
            g = [q["gen_tps"] for q in sp]
            e["first"].append(g[0])
            e["steady"].append(sum(g[1:]) / (len(g) - 1) if len(g) > 1 else g[0])
            e["all"] += g[1:] if len(g) > 1 else g
            for i, x in enumerate(g):
                e["per_req"][i].append(x)
            dp = sm["deep"].setdefault(k, {"per_req": {}, "reqs": 0})
            dp["reqs"] = max(dp["reqs"], len(sp))
            for i, q in enumerate(sp):
                for mk, mv in q.get("m", {}).items():
                    if mk.startswith("layer_") or mv is None:
                        continue
                    dp["per_req"].setdefault(mk, [[] for _ in range(len(sp))])[i].append(mv)
        for q in r["requests"]:
            if q["suite"] == "long":
                sm["long"].setdefault(k, []).append(q["prompt_tps"])
            if q["suite"] == "sweep":
                sw = sm["sweep"].setdefault(k, {})
                sw.setdefault(q["nominal"], []).append((q["prompt_n"], q["ttft_ms"], q["gen_tps"], q["wall_s"]))
            if q["suite"] == "custom" and q.get("passed") is not None:
                c = sm["custom"].setdefault(k, [0, 0])
                c[1] += 1
                c[0] += 1 if q["passed"] else 0
            lh = q.get("m", {}).get("layer_hits")
            if lh:
                L = sm["layers"].setdefault(k, {"hits": [], "misses": []})
                for key, arr in (("hits", lh), ("misses", q["m"]["layer_misses"])):
                    L[key] += [0] * (len(arr) - len(L[key]))
                    for i, v in enumerate(arr):
                        L[key][i] += v
    for k, e in sm["speed"].items():
        lo, hi = (min(e["all"]), max(e["all"])) if e["all"] else (None, None)
        e.update(first=mean(e["first"]), steady=mean(e["steady"]), per_req=[mean(v) for v in e["per_req"]], min=lo, max=hi, n=len(e["all"]))
        e.pop("all")
    for k, dp in sm["deep"].items():
        per = {mk: [mean(v) for v in lst] for mk, lst in dp["per_req"].items()}
        steady = {}
        for mk, lst in per.items():
            vals = [x for x in lst[1:] if x is not None] or [x for x in lst if x is not None]
            steady[mk] = mean(vals) if mk not in ("cg_peak", "gpu_mb", "temp") else (max(vals) if vals else None)
        dp["per_req"], dp["steady"] = per, steady
    sm["long"] = {k: mean(v) for k, v in sm["long"].items()}
    for k, sw in sm["sweep"].items():
        sm["sweep"][k] = {str(n): {"prompt_n": mean([x[0] for x in v]), "ttft_ms": mean([x[1] for x in v]), "out_tps": mean([x[2] for x in v]), "e2e_s": mean([x[3] for x in v])} for n, v in sorted(sw.items())}
    for lim in sorted({r["limit_gb"] for r in runs}, key=lambda x: x or 0):
        a = next((r for r in runs if r["limit_gb"] == lim and r["mode"] == "stock" and r["requests"]), None)
        b = next((r for r in runs if r["limit_gb"] == lim and r["mode"] == "plugin" and r["requests"]), None)
        if a and b:
            ta = [q["tokens"] for q in a["requests"] if q["suite"] in ("speed", "custom", "sweep")]
            tb = [q["tokens"] for q in b["requests"] if q["suite"] in ("speed", "custom", "sweep")]
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
    errs = validate_cfg(cfg)
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


def esc(s):
    return html.escape(str(s))


def tables_html(res):
    sm = res["summary"]
    o = []
    s = res["system"]
    o.append(f'<p class="muted">{esc(s.get("cpu", ""))} · {s.get("ram_gb", "?")} GB RAM · {esc(", ".join(s.get("devices", [])) or "no GPU backend")} · model {res["model_gb"]} GB · {esc(res["created"])}</p>')
    # the one table that tells the story: every run, every key number
    f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
    rows = []
    for k in sorted(sm["speed"], key=lambda x: (float(x.split("|")[0]) if x.split("|")[0] != "None" else 0, x.split("|")[1] != "stock")):
        lim, mode = k.split("|")
        e, st = sm["speed"][k], sm["deep"].get(k, {}).get("steady", {})
        rows.append(f'<tr><td>{lim}</td><td>{"stock" if mode == "stock" else "moe-cache"}</td><td><b>{f(e["steady"])}</b></td><td>{f(e["min"])} to {f(e["max"])}</td><td>{f(e["first"])}</td>'
                    f'<td>{f(st.get("io_mb_tok"), 1)}</td><td>{f(st.get("majflt_tok"), 1)}</td><td>{f(st.get("cpu_ms_tok"), 1)}</td>'
                    f'<td>{f(st.get("hit_dec_pct"), 1)}</td><td>{f(st.get("evictions_tok"), 2)}</td><td>{f(st.get("evict_age"), 0)}</td>'
                    f'<td>{f(st.get("cg_peak"), 1)}</td><td>{f(st.get("temp"), 0)}</td></tr>')
    o.append('<h3>Summary (steady state = requests 2 and later)</h3><div style="overflow-x:auto"><table><tr><th>limit GB</th><th>mode</th><th>gen tok/s</th><th>range</th><th>1st req</th>'
             '<th>SSD MB / token</th><th>page faults / token</th><th>CPU ms / token</th><th>cache hit % (decode)</th><th>evictions / token</th><th>evicted expert idle (groups)</th><th>peak memory GB</th><th>max temp C</th></tr>' + "".join(rows) + "</table></div>")
    rows = []
    for r in res["runs"]:
        sp = [q["gen_tps"] for q in r["requests"] if q["suite"] == "speed"]
        rows.append(f'<tr><td>{r["limit_gb"] or "-"}</td><td>{r["mode"]}</td><td>{r.get("repeat", 0) + 1}</td><td>{", ".join(f"{x:.2f}" for x in sp) or esc(r.get("error", "-"))}</td>'
                    f'<td>{"yes" if r.get("clean") else "<b>no</b>"} ({(r.get("start_temp") or 0):.0f} C)</td></tr>')
    o.append('<h3>Runs</h3><table><tr><th>limit GB</th><th>mode</th><th>repeat</th><th>generation tok/s per request</th><th>clean start</th></tr>' + "".join(rows) + "</table>")
    if sm["identity"]:
        o.append("<h3>Identity (same tokens, stock vs moe-cache)</h3><table><tr><th>limit GB</th><th>identical answers</th></tr>" +
                 "".join(f"<tr><td>{k}</td><td>{a}/{b} {'PASS' if a == b else '<b>DIFFERENT</b>'}</td></tr>" for k, (a, b) in sm["identity"].items()) + "</table>")
    if sm["custom"]:
        o.append("<h3>Custom prompts with expected answers</h3><table><tr><th>run</th><th>passed</th></tr>" +
                 "".join(f"<tr><td>{esc(k.replace('|', ' GB, '))}</td><td>{a}/{b}</td></tr>" for k, (a, b) in sm["custom"].items()) + "</table>")
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


STYLE = ":root{--bg:#fff;--fg:#1f2328;--muted:#656d76;--grid:#d0d7de;--card:#f6f8fa;--line:#d0d7de}@media(prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#8b949e;--grid:#30363d;--card:#161b22;--line:#30363d}}body{background:var(--bg);color:var(--fg);font-family:system-ui,Helvetica,Arial,sans-serif;max-width:960px;margin:0 auto;padding:16px}table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:5px 7px;text-align:left}.muted{color:var(--muted)}pre{white-space:pre-wrap;background:var(--card);padding:8px;border-radius:6px}.chart{margin:10px 0;border:1px solid var(--line);border-radius:8px;padding:6px}"


def report_html(res):
    """One self-contained file: tables, the interactive charts (same library as the GUI) and the data."""
    lib = (HERE / "charts.js").read_text()
    data = json.dumps(res).replace("</", "<\\/")
    return (f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>moe-cache benchmark report</title><style>{STYLE}</style>'
            f'<h1>moe-cache benchmark: {esc(Path(res["config"]["model"]).name)}</h1>{tables_html(res)}<div id="charts"></div>'
            f'<script>{lib}</script><script>const RESULT={data};MoeCharts.render(document.getElementById("charts"),RESULT);</script>')


_PATH = re.compile(r"/(?:[^\s/\"']+/)+([^\s/\"']+)")


def share_safe(res):
    """Copy of a result for sharing: no prompts, outputs, token ids, kernel string or absolute paths (paths shrink to file names).
    The full result stays in the local store; the full export needs ?full=1."""
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return _PATH.sub(r"\1", o) if isinstance(o, str) else o
    r = clean(res)
    r["config"]["custom"] = ""
    r["config"]["ppl_file"] = ""
    r["system"].pop("kernel", None)
    for run in r["runs"]:
        for q in run["requests"]:
            q["text"] = ""
            q["tokens"] = []
            if "prompt" in q:
                q["prompt"] = "(hidden in the shareable report)"
    return r


def csv_export(res):
    b = io.StringIO()
    w = csv.writer(b)
    w.writerow(["limit_gb", "mode", "repeat", "suite", "name", "gen_tok_s", "prompt_tok_s", "prompt_tokens", "gen_tokens", "ttft_ms", "e2e_s", "ssd_mb_per_token", "page_faults_per_token", "cpu_ms_per_token", "cache_hit_pct_decode", "evictions", "cache_read_mb_per_token", "clean_start", "start_temp_c"])
    for r in res["runs"]:
        for q in r["requests"]:
            w.writerow([r["limit_gb"], r["mode"], r.get("repeat", 0) + 1, q["suite"], q["name"], round(q["gen_tps"], 3), round(q["prompt_tps"], 2), q["prompt_n"], q["gen_n"], round(q["ttft_ms"], 1), round(q["wall_s"], 2)] + [q.get("m", {}).get(x) for x in ("io_mb_tok", "majflt_tok", "cpu_ms_tok", "hit_dec_pct", "evictions", "cache_read_mb_tok")] + [r.get("clean"), r.get("start_temp")])
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


# ---------------------------------------------------------------- security
# The GUI starts programs and reads your results, so a web page you visit must not be able to drive it:
#  - a random token (embedded in the page this server serves; never readable cross-site) is required on every API call
#  - Host must be 127.0.0.1 / localhost with our port (blocks DNS rebinding); Origin and Sec-Fetch-Site must be ours
#  - the run configuration is validated (program name, library name, MOE_CACHE_* variables only, no file-writing or network flags)
TOKEN = secrets.token_urlsafe(24)
TOKEN_FILE = DATA / "token"          # 0600; read by bin/moe-cache-bench-watch and bench/standard.py
PORT = 0
API_KEYS = {}                         # llama-server port -> key (passed by key file, never on the command line)
DENIED_FLAGS = {"--log-file", "--slot-save-path", "--host", "--api-key", "--api-key-file", "--path", "--ssl-key-file", "--ssl-cert-file",
                "--webui-config-file", "--models-dir", "--models-preset"}


def is_gguf(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def server_name_ok(path):
    return bool(path) and os.path.basename(path).startswith("llama-server") and os.access(path, os.X_OK) and os.path.isfile(path)


def validate_cfg(cfg):
    errs = []
    if not server_name_ok(cfg.get("llama_server", "")):
        errs.append("llama-server path must be an executable file named llama-server*")
    if not is_gguf(cfg.get("model", "")):
        errs.append("model must be a GGUF file")
    if "plugin" in cfg.get("modes", []):
        n = os.path.basename(cfg.get("plugin_lib", ""))
        if not (re.fullmatch(r"(lib)?ggml-moe-cache[\w.-]*\.(so|dll)", n) and os.path.isfile(cfg["plugin_lib"])):
            errs.append("plugin library must be the built ggml-moe-cache library (build it with ./build.sh)")
    try:
        toks = shlex.split(cfg.get("plugin_env") or "")
    except ValueError:
        toks = []
        errs.append("plugin environment: unbalanced quotes")
    for kv in toks:
        if not re.fullmatch(r"MOE_CACHE_[A-Z0-9_]+=[^\s]*", kv):
            errs.append(f"plugin environment: only MOE_CACHE_* variables are allowed (got '{kv.split('=')[0][:40]}')")
    try:
        xs = shlex.split(cfg.get("extra_args") or "")
    except ValueError:
        xs = []
        errs.append("extra arguments: unbalanced quotes")
    for a in xs:
        if a.split("=", 1)[0] in DENIED_FLAGS:
            errs.append(f"extra arguments: {a.split('=')[0]} is not allowed here")
    return errs


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                         "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def gate(self, need_token=True, post=False):
        """True when the request may proceed; otherwise the 4xx answer was already sent."""
        host = (self.headers.get("Host") or "").lower()
        if host not in (f"127.0.0.1:{PORT}", f"localhost:{PORT}"):
            self.send(403, {"error": "bad Host header"}); return False
        site = self.headers.get("Sec-Fetch-Site")
        if site not in (None, "same-origin", "none"):
            self.send(403, {"error": "cross-site request refused"}); return False
        origin = self.headers.get("Origin")
        if origin is not None and origin.lower() not in (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"):
            self.send(403, {"error": "foreign Origin refused"}); return False
        if post and not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            self.send(415, {"error": "Content-Type must be application/json"}); return False
        if need_token:
            q = self.path.partition("?")[2].split("&")
            tok = self.headers.get("X-Bench-Token") or next((x[2:] for x in q if x.startswith("t=")), "")
            if not hmac.compare_digest(tok.encode(), TOKEN.encode()):
                self.send(401, {"error": "missing or wrong token"}); return False
        return True

    def full(self):
        """?full=1 asks for the unfiltered result (prompts, outputs, local paths): only for your own use."""
        return "full=1" in self.path.partition("?")[2].split("&")

    def result(self, rid):
        f = RESULTS / (re.sub(r"[^A-Za-z0-9._-]", "", rid) + ".json")
        return json.loads(f.read_text()) if f.exists() else None

    def do_GET(self):
        p = self.path.split("?")[0]
        try:
            if not self.gate(need_token=p not in ("/", "/charts.js")):
                return
            if p == "/":
                page = (HERE / "ui.html").read_text().replace("__BENCH_TOKEN__", TOKEN)
                return self.send(200, page, "text/html")
            if p == "/charts.js":
                return self.send(200, (HERE / "charts.js").read_bytes(), "application/javascript")
            if p == "/api/config":
                cfg = default_config()
                if CONFIG.exists():
                    cfg.update(json.loads(CONFIG.read_text()))
                return self.send(200, cfg)
            if p == "/api/system":
                q = urllib.request.unquote(self.path.split("ls=", 1)[1].split("&")[0]) if "ls=" in self.path else find_llama_server()
                return self.send(200, system_info(q if server_name_ok(q) else None))
            if p == "/api/modelinfo":
                path = urllib.request.unquote(self.path.split("path=", 1)[1].split("&")[0]) if "path=" in self.path else ""
                ok = is_gguf(path)
                return self.send(200, {"gb": round(model_size_gb(path), 2) if ok else 0, "exists": ok})
            if p == "/api/jobs":
                return self.send(200, [j.public() for j in JOBS.values() if j.state in ("queued", "running")])
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
                return self.send(200, {"result": r, "tables": tables_html(r)})
            m = re.match(r"/report/([\w.-]+)\.html$", p)
            if m and self.result(m.group(1)):
                r = self.result(m.group(1))
                return self.send(200, report_html(r if self.full() else share_safe(r)), "text/html")
            m = re.match(r"/export/([\w.-]+)\.csv$", p)
            if m and self.result(m.group(1)):
                return self.send(200, csv_export(self.result(m.group(1))), "text/csv")
            m = re.match(r"/export/([\w.-]+)\.json$", p)
            if m and self.result(m.group(1)):
                r = self.result(m.group(1))
                return self.send(200, json.dumps(r if self.full() else share_safe(r), indent=1))
            self.send(404, {"error": "not found"})
        except Exception as e:
            self.send(500, {"error": str(e)})

    def do_POST(self):
        p = self.path.split("?")[0]
        try:
            if not self.gate(post=True):
                return
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


def make_store_private():
    """Results, logs and reports can contain prompts and outputs: keep them readable by this user only."""
    os.umask(0o077)
    DATA.mkdir(parents=True, exist_ok=True)
    for d, _, files in os.walk(DATA):
        try:
            os.chmod(d, 0o700)
            for f in files:
                os.chmod(os.path.join(d, f), 0o600)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description="moe-cache benchmark GUI")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    global PORT
    PORT = a.port
    make_store_private()
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as tf:
        tf.write(TOKEN)
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
