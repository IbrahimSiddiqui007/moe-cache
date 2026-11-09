"""moe-cache monitor (GUI-041): the numbers behind the control page's Monitor view. Python 3.8+, standard library only.

One Monitor per server run (Runner.reset makes a new one). It keeps, in memory only:
  * samples   one row every ~2 s (decode / prompt tok/s, context depth, hit rate, SSD GB/s and wait, executor share, RAM / commit / VRAM)
  * layer cols one column of per-layer hit rates per plugin snapshot (the heat map)
  * requests  the last finished requests (prompt / decode speed, tokens, MTP acceptance)
  * log       the last 20000 lines with time, source (launcher / plugin / llama-server / ui) and level
Everything is derived from what already exists: the plugin's "moe-cache-stats {json}" line (cumulative counters), llama-server's /metrics
(--metrics is always on) and its log lines. Nothing here touches the plugin or llama-server; MOE_CACHE_MONITOR=0 turns the sampling off.
"""
import collections, json, os, re, shlex, threading, time

GIB = 2**30
MAX_SAMPLES = 3600          # about two hours at one row per 2 s
MAX_LOG = 20000
MAX_REQUESTS = 100
MAX_LAYER_COLS = 120

TIMING_RE = re.compile(r"(prompt eval|eval) time =\s*([\d.]+) ms /\s*(\d+) (?:tokens|runs).*?([\d.]+) tokens per second")
CTX_RE = re.compile(r"\bn_tokens = (\d+)")
SLOT_ID_RE = re.compile(r"\bslot\s+\S+:\s+id\s+(\d+)")
SLOT_LRU_RE = re.compile(r"selected slot by LRU")
SLOT_LCP_RE = re.compile(r"selected slot by LCP similarity.*?f_keep = ([\d.]+)")
NPAST_SET_RE = re.compile(r"n_past was set to (\d+)")
NTOK_RE = re.compile(r"task\.n_tokens\(\) = (\d+)")
RELEASE_RE = re.compile(r"stop processing: n_tokens = (\d+)")
NEWPROMPT_RE = re.compile(r"new prompt,.*?task\.n_tokens = (\d+)")
TOTAL_RE = re.compile(r"\btotal time =\s*([\d.]+) ms /\s*(\d+) tokens")
MTP_RE = re.compile(r"draft acceptance rate = ([\d.]+)\s*\(\s*(\d+) accepted /\s*(\d+) generated")
APPROX_RE = re.compile(r"moe-cache: approximate routing \((skip|subst)\): (\d+) low-ranked picks looked at, (\d+) given up, (\d+) replaced")
FEWER_RE = re.compile(r"moe-cache: approximate: fewer experts \(MOE_CACHE_FEWER=(\d+)\): (\d+) of the lowest-ranked picks")
LLAMA_RE = re.compile(r"^(?:[\d.]+ [A-Z] |srv |slot |llama_|load_tensors|print_info|main:|common_|ggml_|sched|clip_|build:|system info|init:|prompt eval|eval time|total time|request:|que |set_|update_slots|Device |[a-z_]+: )")
LEVEL_RE = re.compile(r"^[\d.]+ ([A-Z]) ")
ERR_RE = re.compile(r"(?i)\b(error|failed|exception|fatal|abort|cannot|bad alloc)")
WARN_RE = re.compile(r"(?i)\b(warn|warning|refus|disabled|fallback|stall|slow|low memory|out of memory)\b")


def enabled():
    return os.environ.get("MOE_CACHE_MONITOR", "1").strip().lower() not in ("0", "off", "false", "no")


def parse_prom(text):
    """llama-server /metrics (Prometheus text) -> {name: float} for the llamacpp: series, comments and labels dropped."""
    out = {}
    for line in (text or "").splitlines():
        if not line.startswith("llamacpp:"):
            continue
        name, _, val = line.partition(" ")
        try:
            out[name[len("llamacpp:"):]] = float(val.split()[0])
        except (ValueError, IndexError):
            continue
    return out


def classify(line, starting):
    """(source, level) of a log line. starting: the launcher is still deciding (before its 'web UI and API will be at' line)."""
    if line.startswith("ui:"):
        src = "ui"
    elif line.startswith("moe-cache:") or line.startswith("moe-cache "):
        src = "launcher" if starting else "plugin"
    elif LLAMA_RE.match(line):
        src = "llama"
    else:
        src = "llama" if not starting else "launcher"
    level = "error" if ERR_RE.search(line) and not line.startswith("moe-cache: ") else "warn" if WARN_RE.search(line) else "info"
    if line.startswith("moe-cache: warning") or line.startswith("WARNING"):
        level = "warn"
    m = LEVEL_RE.match(line)            # llama-server's own level letter after its clock: W / E
    if m and m.group(1) == "E" or line.startswith("error:"):
        level = "error"
    elif m and m.group(1) == "W":
        level = "warn"
    return src, level


def plan_from_args(argv):
    """The plan as the owner reads it (-ub, KV type, device, layers on the GPU, experts on the CPU, context, threads) from the llama-server arguments."""
    def val(*names):
        for i, a in enumerate(argv):
            for n in names:
                if a == n and i + 1 < len(argv):
                    return argv[i + 1]
                if a.startswith(n + "=") and n.startswith("--"):
                    return a.split("=", 1)[1]
        return None
    out = {"ub": val("-ub", "--ubatch-size"), "b": val("-b", "--batch-size"), "ctx": val("-c", "--ctx-size"), "ngl": val("-ngl", "--n-gpu-layers", "--gpu-layers"),
           "cpu_moe_layers": val("--n-cpu-moe", "-ncmoe"), "device": val("-dev", "--device"), "threads": val("-t", "--threads"), "parallel": val("-np", "--parallel"),
           "kv_k": val("-ctk", "--cache-type-k"), "kv_v": val("-ctv", "--cache-type-v"), "split_mode": val("-sm", "--split-mode"), "mmproj": val("--mmproj")}
    out["cpu_moe_all"] = "--cpu-moe" in argv or "-cmoe" in argv
    out["flash_attn"] = val("-fa", "--flash-attn")
    return {k: v for k, v in out.items() if v not in (None, False)}


def split_command(cmd):
    try:
        return shlex.split(cmd, posix=os.name != "nt")
    except ValueError:
        return cmd.split()


class Monitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.samples = collections.deque(maxlen=MAX_SAMPLES)
        self.layer_cols = collections.deque(maxlen=MAX_LAYER_COLS)
        self.requests = collections.deque(maxlen=MAX_REQUESTS)
        self.log = collections.deque(maxlen=MAX_LOG)       # (seq, t, source, level, text)
        self.starting = True
        self.snap = None            # (t, counters) of the newest plugin snapshot
        self.rates = {}             # derived between the last two snapshots
        self.layers = {"hits": [], "misses": []}
        self.prom_prev = None       # (t, parsed /metrics)
        self.prom_rates = {}
        self.prom_now = {}
        self.ctx_depth = None
        self.mtp = {"accepted": 0, "generated": 0, "last": None}
        self.approx = {}            # from the plugin's lines at server stop (older plugins)
        self.approx_live = {}       # from the stats snapshot (GUI-042): cumulative counters while the server runs
        self.alt_rate = None
        self.pending_req = {}
        self.slot_tokens = {}       # slot id -> tokens its cache held when its last request ended (a new request keeps f_keep of them)
        self.vram_cache = (0.0, None)
        self.enabled = enabled()
        self.touched = 0.0          # when a page or `top` last asked: hardware readings (a subprocess for nvidia-smi) are taken only while somebody looks
        self.hw = {}

    # ---- input ----
    def add_line(self, seq, t, line):
        src, level = classify(line, self.starting)
        if self.starting and "web UI and API will be at" in line:
            self.starting = False
        with self.lock:
            self.log.append((seq, round(t, 2), src, level, line[:2000]))
            m = CTX_RE.search(line)
            if m:
                self.ctx_depth = int(m.group(1))
            sid = SLOT_ID_RE.search(line)
            sid = sid.group(1) if sid else None
            m = RELEASE_RE.search(line)
            if m and sid is not None:
                self.slot_tokens[sid] = int(m.group(1))
            if SLOT_LRU_RE.search(line):
                self.pending_req = {"reused": 0}                                # an empty slot: the whole prompt is processed
            m = SLOT_LCP_RE.search(line)
            if m and sid is not None:
                self.pending_req = {"reused": int(round(float(m.group(1)) * self.slot_tokens.get(sid, 0)))}      # the part of the old cache that matches
            m = NPAST_SET_RE.search(line)
            if m:
                self.pending_req["reused"] = int(m.group(1))                    # llama-server re-evaluates the last token: the exact figure
            m = NTOK_RE.search(line)
            if m:
                self.pending_req["prompt_total"] = int(m.group(1))
            m = NEWPROMPT_RE.search(line)
            if m:
                self.pending_req["prompt_total"] = int(m.group(1))             # an older llama-server logs the whole prompt length when a request starts
            m = TOTAL_RE.search(line)
            if m and self.requests:
                self.requests[-1]["total_ms"] = float(m.group(1))              # printed after the decode line: the request's whole duration
            m = MTP_RE.search(line)
            if m:
                self.mtp["accepted"] += int(m.group(2)); self.mtp["generated"] += int(m.group(3)); self.mtp["last"] = float(m.group(1))
                self.pending_req["mtp"] = float(m.group(1))
            m = APPROX_RE.search(line)
            if m:
                self.approx = {"mode": m.group(1), "considered": int(m.group(2)), "given_up": int(m.group(3)), "replaced": int(m.group(4)), "t": round(t, 1)}
            m = FEWER_RE.search(line)
            if m:
                self.approx = dict(self.approx, fewer=int(m.group(1)), fewer_dropped=int(m.group(2)), t=round(t, 1))
            m = TIMING_RE.search(line)
            if m:
                kind = "prefill" if m.group(1) == "prompt eval" else "decode"
                self.pending_req[kind + "_ms"] = float(m.group(2)); self.pending_req[kind + "_tokens"] = int(m.group(3)); self.pending_req[kind + "_tps"] = float(m.group(4))
                if kind == "decode":        # llama-server prints "eval time" after "prompt eval time": the request is complete
                    req = dict(self.pending_req, t=round(t, 1), ctx=self.ctx_depth)
                    if "prompt_total" in req and "prefill_tokens" in req and "reused" not in req:
                        req["reused"] = max(0, req["prompt_total"] - req["prefill_tokens"])
                    elif "reused" in req and "prompt_total" not in req and "prefill_tokens" in req:
                        req["prompt_total"] = req["reused"] + req["prefill_tokens"]
                    self.requests.append(req)
                    self.pending_req = {}

    def add_snapshot(self, t, s):
        if not self.enabled:
            return
        with self.lock:
            prev, self.snap = self.snap, (t, s)
            hits, miss = list(s.get("layer_hits") or []), list(s.get("layer_misses") or [])
            if prev:
                t0, p = prev
                dt = max(t - t0, 1e-3)
                dh, dm = s.get("hits", 0) - p.get("hits", 0), s.get("misses", 0) - p.get("misses", 0)
                dhd, dmd = s.get("hits_dec", 0) - p.get("hits_dec", 0), s.get("misses_dec", 0) - p.get("misses_dec", 0)
                dns_total = s.get("ns_total", 0) - p.get("ns_total", 0)
                r = {"hit": dh / (dh + dm) if dh + dm > 0 else None, "hit_dec": dhd / (dhd + dmd) if dhd + dmd > 0 else None,
                     "ssd_gbs": max(0, s.get("read_bytes", 0) - p.get("read_bytes", 0)) / dt / 1e9,
                     "ssd_wait": max(0, s.get("ns_ssd_wait", 0) - p.get("ns_ssd_wait", 0)) / 1e9 / dt,       # seconds waiting for the SSD per second
                     "ssd_waits": max(0, s.get("n_ssd_wait", 0) - p.get("n_ssd_wait", 0)),
                     "cpu": min(1.0, max(0, s.get("ns_cpu", 0) - p.get("ns_cpu", 0)) / dns_total) if dns_total > 0 else None,    # share of the cache's time spent in the CPU expert executor
                     "evict": max(0, s.get("evictions", 0) - p.get("evictions", 0)) / dt,
                     "pf_used": max(0, s.get("pf_used", 0) - p.get("pf_used", 0)), "pf_wasted": max(0, s.get("pf_wasted", 0) - p.get("pf_wasted", 0)),
                     "gpu_up_gbs": max(0, s.get("gpu_up_bytes", 0) - p.get("gpu_up_bytes", 0)) / dt / 1e9}
                a1, a0 = s.get("approx") or {}, p.get("approx") or {}
                if a1:                  # picks the plugin changed per second (given up + replaced + dropped by "fewer experts")
                    ch = lambda a: a.get("skipped", 0) + a.get("substituted", 0) + a.get("fewer_dropped", 0)
                    r["alt"] = max(0, ch(a1) - ch(a0)) / dt
                ph, pm = list(p.get("layer_hits") or []), list(p.get("layer_misses") or [])
                col = []
                for i in range(len(hits)):
                    a = hits[i] - (ph[i] if i < len(ph) else 0)
                    b = (miss[i] if i < len(miss) else 0) - (pm[i] if i < len(pm) else 0)
                    col.append(round(a / (a + b), 3) if a + b > 0 else None)
                if any(c is not None for c in col):
                    self.layer_cols.append({"t": round(t, 1), "rate": col})
                self.rates = r
            self.layers = {"hits": hits, "misses": miss}
            ap = s.get("approx")
            if ap:
                self.approx_live = {"mode": {1: "skip", 2: "subst"}.get(ap.get("route")), "considered": ap.get("considered", 0), "given_up": ap.get("skipped", 0), "replaced": ap.get("substituted", 0),
                                    "fewer": ap.get("fewer", 0), "fewer_considered": ap.get("fewer_considered", 0), "fewer_dropped": ap.get("fewer_dropped", 0), "live": True, "t": round(t, 1)}
            self.snapshot_extra = {k: s.get(k) for k in ("policy", "evictions", "resident_bytes", "budget", "gpu_segments", "fallbacks", "exec_ops", "exec_fused", "pf_issued", "pf_used", "pf_wasted") if k in s}
            if s.get("ex"):
                self.snapshot_extra["threads"] = s["ex"].get("threads")

    def add_metrics(self, t, text):
        if not self.enabled:
            return
        m = parse_prom(text)
        with self.lock:
            self.prom_now = m
            if self.prom_prev:
                t0, p = self.prom_prev
                r = {}
                for key, tok, sec in (("pre", "prompt_tokens_total", "prompt_seconds_total"), ("dec", "tokens_predicted_total", "tokens_predicted_seconds_total")):
                    dtok, dsec = m.get(tok, 0) - p.get(tok, 0), m.get(sec, 0) - p.get(sec, 0)
                    r[key] = dtok / dsec if dtok > 0 and dsec > 0 else None
                self.prom_rates = r
            self.prom_prev = (t, m)

    def watched(self, now=None, within=20.0):
        return (now or time.time()) - self.touched < within

    def tick(self, t, live_tps, rss, ram, commit, vram, busy, hw=None):
        """One sample row. ram / commit / vram: (used, total) bytes or None. hw: moe_cache_hw readings (gpu_*, cpu_*, io_*), merged when given."""
        if not self.enabled:
            return
        with self.lock:
            r, pr = self.rates, self.prom_rates
            ctx = self.prom_now.get("kv_cache_tokens") or self.ctx_depth
            self.samples.append({"t": round(t, 1), "dec": pr.get("dec") if pr.get("dec") is not None else live_tps, "pre": pr.get("pre"), "ctx": ctx, "busy": 1 if busy else 0,
                                 "hit": r.get("hit"), "hit_dec": r.get("hit_dec"), "ssd": r.get("ssd_gbs"), "wait": r.get("ssd_wait"), "cpu": r.get("cpu"),
                                 "evict": r.get("evict"), "up": r.get("gpu_up_gbs"), "alt": r.get("alt"), "rss": rss,
                                 "ram": ram[0] if ram else None, "commit": commit[0] if commit else None, "vram": vram[0] if vram else None})
            if hw:
                self.samples[-1].update({k: (round(v, 2) if isinstance(v, float) else v) for k, v in hw.items()})

    # ---- output ----
    def payload(self, since=0.0, log_seq=0, ctx_max=None, plan=None, cap=None, totals=None, include_log=True, tail=None, modes=None):
        self.touched = time.time()
        with self.lock:
            out = {"now": round(time.time(), 1), "enabled": self.enabled, "t0": round(self.t0, 1),
                   "samples": [s for s in self.samples if s["t"] > since][-MAX_SAMPLES:],
                   "layer_cols": [c for c in self.layer_cols if c["t"] > since],
                   "layer_total": [round(h / (h + m), 3) if h + m else None for h, m in zip(self.layers["hits"], self.layers["misses"])],
                   "requests": list(self.requests)[-30:], "mtp": dict(self.mtp, rate=(self.mtp["accepted"] / self.mtp["generated"]) if self.mtp["generated"] else None),
                   "approx": dict(self.approx_live or self.approx), "ctx": {"depth": self.ctx_depth, "max": ctx_max}, "plan": plan or {}, "cap": cap, "totals": totals or {}, "modes": modes or [],
                   "extra": dict(getattr(self, "snapshot_extra", {})), "metrics": {k: v for k, v in self.prom_now.items() if k in ("requests_processing", "requests_deferred", "kv_cache_usage_ratio", "kv_cache_tokens", "n_busy_slots_per_decode")}}
            if include_log:
                out["log"] = [list(x) for x in self.log if x[0] > log_seq]
                if tail:                    # the mini window and `top` want only the newest few lines
                    out["log"] = out["log"][-int(tail):]
                out["log_seq"] = self.log[-1][0] if self.log else log_seq
            return out


def export_text(lines, fmt="text"):
    """The log lines as a download: plain text ('time source level text') or JSON lines."""
    out = []
    for seq, t, src, level, text in lines:
        if fmt == "jsonl":
            out.append(json.dumps({"seq": seq, "t": t, "source": src, "level": level, "text": text}))
        else:
            out.append(f"{time.strftime('%H:%M:%S', time.localtime(t))} [{src}/{level}] {text}")
    return "\n".join(out) + "\n"
