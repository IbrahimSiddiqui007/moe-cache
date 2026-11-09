#!/usr/bin/env python3
"""moe-cache top (GUI-045): the Monitor as a full-screen terminal app. Python 3.8+, standard library only; Linux, macOS, Windows Terminal / PowerShell
(Unicode and colour where the terminal has them, plain ASCII in the classic Windows console). Read only: it never starts, stops or changes anything.

  python3 bin/moe_cache.py top [--url http://127.0.0.1:8070/?t=...] [--port 8080] [--interval 2] [--ascii] [--theme dark|light|mono] [--once] [--view N]

Where the numbers come from (first that works): --url (a control page's address with its token), the control page running on this machine (`ui` leaves its
port and token in a file only this user can read), --port N (a llama-server on 127.0.0.1:N: its /metrics and /slots; no plugin hit rate and no log, because
this process did not start it), the first llama-server found on ports 8080-8089.

Views (keys 1-7 or left/right): 1 Overview, 2 Serving, 3 Requests, 4 Engine, 5 Hardware, 6 Logs, 7 Connect.
Keys: / search (Logs)  l level  s source  f follow  space freeze  r refresh  t theme  b open in browser  ? help  q quit (the server keeps running).
"""
import argparse, json, os, re, shutil, sys, time, urllib.error, urllib.parse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_cache_monitor as mm  # noqa: E402
import moe_cache_hw as hw  # noqa: E402

GIB = 2**30
UNICODE = {"bar": "█░", "blocks": " ▁▂▃▄▅▆▇█", "dot": "●", "h": "─", "v": "│", "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "mid": "·", "deg": "°", "up": "▲", "heat": "█", "none": "·"}
ASCII = {"bar": "#.", "blocks": " .:-=+*#@", "dot": "*", "h": "-", "v": "|", "tl": "+", "tr": "+", "bl": "+", "br": "+", "mid": "|", "deg": " ", "up": "^", "heat": "#", "none": "."}
THEMES = {   # style name -> SGR codes
    "dark": {"ok": "92", "warn": "93", "bad": "91", "dim": "90", "bold": "1", "accent": "96", "gpu": "95", "sel": "7", "tab": "1;96"},
    "light": {"ok": "32", "warn": "33", "bad": "31", "dim": "2", "bold": "1", "accent": "34", "gpu": "35", "sel": "7", "tab": "1;34"},
    "mono": {"ok": "1", "warn": "1", "bad": "1;4", "dim": "2", "bold": "1", "accent": "", "gpu": "", "sel": "7", "tab": "1;7"},
}
VIEWS = ["Overview", "Serving", "Requests", "Engine", "Hardware", "Logs", "Connect"]
SERIES_MAX = 600
FIXED_TOP = {"hit": 1.0, "hit_dec": 1.0, "cpu": 1.0, "wait": 1.0, "gpu_util": 100.0, "cpu_sys": 100.0}      # percentages are drawn against 100 %, not against the biggest value in view


def top_file():
    """The page's port/token file (bin/moe_cache_ui.py TOP_FILE): the launcher's cache folder."""
    try:
        import moe_cache as mc
        return os.path.join(mc.CACHE_DIR, "ui-top.json")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".cache", "moe-cache", "ui-top.json")


# ---- sources: both give the payload shape of GET /api/monitor ----

class PageSource:
    kind = "page"

    def __init__(self, base, token):
        self.base, self.token, self.since, self.seq = base.rstrip("/"), token, 0.0, 0
        self.label = "control page " + urllib.parse.urlparse(base).netloc
        self.url = self.base

    def get(self, path):
        req = urllib.request.Request(self.base + path, headers={"X-MoeCache-Token": self.token})
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.load(r)

    def poll(self):
        j = self.get(f"/api/monitor?since={self.since}&log={self.seq}")
        if j.get("samples"):
            self.since = j["samples"][-1]["t"]
        self.seq = j.get("log_seq", self.seq)
        return j

    def api_key(self):
        try:
            return self.get("/api/key").get("key")
        except (OSError, ValueError):
            return None

    def open_in_browser(self):
        import moe_cache_ui as ui
        ui.open_page(f"{self.base}/?t={self.token}")
        return "opened the control page in the browser"


class ServerSource:
    """A llama-server this process did not start: /metrics gives the rates, /slots whether it is busy, the host meter its CPU and disk. No plugin counters, no log."""
    kind = "server"

    def __init__(self, port):
        self.port, self.mon, self.model, self.ctx = port, mm.Monitor(), None, None
        self.mon.touched = time.time()
        self.label = f"llama-server :{port} (attached)"
        self.url = f"http://127.0.0.1:{port}"
        self.pid = hw.pid_of_port(port)
        self.meter = hw.HostMeter(self.pid)
        self.gpu, self.gpu_t = None, 0.0

    def get(self, path):
        with urllib.request.urlopen(f"{self.url}{path}", timeout=3) as r:
            return r.read().decode("utf-8", "replace")

    def poll(self):
        busy, state = False, "ready"
        try:
            self.mon.add_metrics(time.time(), self.get("/metrics"))
            slots = json.loads(self.get("/slots"))
            busy = any(s.get("is_processing") for s in slots)
            if self.model is None:
                props = json.loads(self.get("/props"))
                self.model = os.path.basename(str(props.get("model_path") or props.get("model_alias") or "")) or "llama-server"
                self.ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or (slots[0].get("n_ctx") if slots else None)
        except urllib.error.HTTPError as e:
            state = "needs an API key" if e.code in (401, 403) else f"http {e.code}"
        except (OSError, ValueError):
            state = "not reachable"
        if time.time() - self.gpu_t > 4:
            self.gpu, self.gpu_t = hw.gpu_stats(), time.time()
        g, r = self.gpu or {}, dict(self.meter.sample())
        r.update(gpu_util=g.get("util"), gpu_temp=g.get("temp"), gpu_power=g.get("power"), gpu_plimit=g.get("power_limit"), pcie_gen=g.get("pcie_gen"), pcie_w=g.get("pcie_width"))
        mem = system_memory()
        self.mon.tick(time.time(), None, None, mem, None, (g["mem_used"], g["mem_total"]) if g.get("mem_used") is not None else None, busy, hw=r)
        totals = {"state": state, "model": self.model or "", "ram_total": mem[1] if mem else None, "gpu": g.get("name"), "vram_total": g.get("mem_total"),
                  "threads": hw.logical_cpus(), "cores": hw.physical_cores(), "port": self.port}
        p = self.mon.payload(0, 0, ctx_max=self.ctx, totals=totals, include_log=False)
        p["samples"] = p["samples"][-1:]
        return p

    def api_key(self):
        return None

    def open_in_browser(self):
        import webbrowser
        webbrowser.open(self.url)
        return "opened llama-server's own page in the browser"


class RunnerSource:
    """A server this process started through the control page's own Runner (`moe-cache-server --top MODEL`): the same numbers and the whole log, in this process."""
    kind = "runner"

    def __init__(self, runner):
        self.runner, self.since, self.seq = runner, 0.0, 0
        self.label = "started here"
        self.url = runner.status().get("url") or ""

    def poll(self):
        j = self.runner.monitor_payload(self.since, self.seq)
        if j.get("samples"):
            self.since = j["samples"][-1]["t"]
        self.seq = j.get("log_seq", self.seq)
        self.url = self.runner.status().get("url") or self.url
        return j

    def api_key(self):
        return self.runner.key() or None

    def open_in_browser(self):
        raise RuntimeError("start the control page (moe_cache.py ui) to open it; this server is only watched here")


def system_memory():
    """(used, total) bytes of this machine's RAM, or None."""
    try:
        if os.name == "nt":
            import moe_cache as mc
            t, a = mc.win_memory()
            return (t - a, t) if t and a is not None else None
        d = {}
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            d[k] = int(v.split()[0]) * 1024
        return d["MemTotal"] - d["MemAvailable"], d["MemTotal"]
    except Exception:
        return None


def port_up(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
            return r.status in (200, 503)
    except urllib.error.HTTPError as e:
        return e.code in (200, 503)
    except OSError:
        return False


def discover(url=None, port=None):
    """(source, None) for the options given, or (None, why)."""
    if url:
        u = urllib.parse.urlparse(url)
        tok = (urllib.parse.parse_qs(u.query).get("t") or [""])[0]
        if not u.scheme or not tok:
            return None, "--url needs the whole address the control page printed (it ends in ?t=...)"
        return PageSource(f"{u.scheme}://{u.netloc}", tok), None
    if port:
        return (ServerSource(port), None) if port_up(port) else (None, f"nothing answers on 127.0.0.1:{port}")
    try:
        with open(top_file(), encoding="utf-8") as f:
            d = json.load(f)
        src = PageSource(f"http://127.0.0.1:{int(d['port'])}", d["token"])
        src.get("/api/monitor?since=1e18&log=1000000000")
        return src, None
    except (OSError, ValueError, KeyError, urllib.error.URLError):
        pass
    for p in range(8080, 8090):
        if port_up(p):
            return ServerSource(p), None
    return None, "no control page and no llama-server found (start one with `moe_cache.py ui` or `moe-cache-server MODEL`, or pass --url / --port)"


# ---- the screen model ----

class Top:
    def __init__(self, glyphs=UNICODE, theme="dark"):
        self.g, self.theme = glyphs, theme
        self.rows, self.cols, self.log, self.payload = [], [], [], {}
        self.view, self.sel, self.scroll = 0, 0, 0
        self.follow, self.level, self.source, self.query, self.typing = True, "all", "all", "", False
        self.frozen, self.help, self.msg, self.msg_t = False, False, "", 0.0
        self.key_shown, self.api_key = False, None
        self.src_url, self.src_kind, self.note = "", "page", ""

    def feed(self, p):
        self.payload = p
        self.rows = (self.rows + p.get("samples", []))[-SERIES_MAX:]
        self.cols = (self.cols + p.get("layer_cols", []))[-mm.MAX_LAYER_COLS:]
        if p.get("log"):
            self.log = (self.log + [tuple(x) for x in p["log"]])[-5000:]

    def last(self, k):
        for r in reversed(self.rows):
            if isinstance(r.get(k), (int, float)):
                return r[k]
        return None

    def series(self, k, n=60):
        return [r.get(k) for r in self.rows[-n:]]

    def notify(self, text):
        self.msg, self.msg_t = text, time.time()

    def filtered_log(self):
        q, re_ = self.query, None
        if len(q) > 2 and q[0] == "/" and q.endswith("/"):
            try:
                re_ = re.compile(q[1:-1], re.I)
            except re.error:
                re_ = None
        out = []
        for l in self.log:
            if self.source != "all" and l[2] != self.source:
                continue
            if self.level == "warn" and l[3] == "info" or self.level == "error" and l[3] != "error":
                continue
            if q and not (re_.search(l[4]) if re_ else q.lower() in l[4].lower()):
                continue
            out.append(l)
        return out

    def handle(self, key, height=24):
        """A key (see read_key) -> None, 'quit', 'refresh', 'open' or 'key'."""
        if self.typing:
            if key in ("enter", "esc"):
                self.typing = False
                if key == "esc":
                    self.query = ""
            elif key == "backspace":
                self.query = self.query[:-1]
            elif len(key) == 1 and key.isprintable():
                self.query += key
            self.scroll, self.follow = 0, True
            return None
        if self.help:
            self.help = False
            return None
        n = len(VIEWS)
        if key in ("q", "Q", "ctrl-c"):
            return "quit"
        if key in "1234567" and len(key) == 1:
            self.view = int(key) - 1
        elif key == "right" or key == "tab":
            self.view = (self.view + 1) % n
        elif key == "left":
            self.view = (self.view - 1) % n
        elif key == " ":
            self.frozen = not self.frozen
        elif key in ("r", "R"):
            return "refresh"
        elif key in ("t", "T"):
            names = list(THEMES)
            self.theme = names[(names.index(self.theme) + 1) % len(names)]
        elif key in ("b", "B"):
            return "open"
        elif key == "?":
            self.help = True
        elif key == "/" and self.view == 5:
            self.typing, self.query = True, ""
        elif key in ("l", "L") and self.view == 5:
            self.level = {"all": "warn", "warn": "error", "error": "all"}[self.level]; self.scroll, self.follow = 0, True
        elif key in ("s", "S") and self.view == 5:
            order = ["all", "launcher", "plugin", "llama", "ui"]
            self.source = order[(order.index(self.source) + 1) % len(order)]; self.scroll, self.follow = 0, True
        elif key in ("f", "F") and self.view == 5:
            self.follow = not self.follow; self.scroll = 0
        elif key in ("k", "K") and self.view == 6:
            self.key_shown = not self.key_shown
            return "key" if self.key_shown else None
        elif key in ("up", "down", "pgup", "pgdn", "home", "end"):
            step = max(1, height - 8) if key in ("pgup", "pgdn") else 1
            if self.view == 5:                       # log: scroll counts lines up from the newest
                if key in ("up", "pgup"):
                    self.scroll += step; self.follow = False
                elif key in ("down", "pgdn"):
                    self.scroll = max(0, self.scroll - step); self.follow = self.scroll == 0
                elif key == "home":
                    self.scroll, self.follow = 10 ** 6, False
                else:
                    self.scroll, self.follow = 0, True
            elif self.view == 2:                     # requests: the selected row
                reqs = len(self.payload.get("requests") or [])
                self.sel = max(0, min(max(0, reqs - 1), self.sel + (step if key in ("down", "pgdn") else -step if key in ("up", "pgup") else 0)))
                if key == "home": self.sel = 0
                if key == "end": self.sel = max(0, reqs - 1)
        return None


# ---- drawing helpers: a line is a list of (text, style) pieces ----

def plain(line):
    return sum(len(t) for t, _ in line)


def fit(line, width):
    """The pieces cut or padded to exactly `width` visible characters."""
    out, used = [], 0
    for t, s in line:
        t = t[:max(0, width - used)]
        if t:
            out.append((t, s)); used += len(t)
    if used < width:
        out.append((" " * (width - used), ""))
    return out


def join(*blocks, gap=1):
    """Blocks side by side: each block is (lines, width); shorter blocks are padded with blanks."""
    h = max(len(b[0]) for b in blocks)
    out = []
    for i in range(h):
        row = []
        for j, (lines, w) in enumerate(blocks):
            row += fit(lines[i] if i < len(lines) else [], w)
            if j < len(blocks) - 1:
                row.append((" " * gap, ""))
        out.append(row)
    return out


def fnum(v, d=1):
    return "-" if v is None else f"{v:,.{d}f}"


def human_tokens(n):
    return "-" if n is None else (f"{n / 1000:.0f}K" if n >= 10000 else f"{n:,}")


def hbar(frac, width, g, style="accent", ok=True):
    f = 0.0 if frac is None else max(0.0, min(1.0, frac))
    n = int(round(f * width))
    return [(g["bar"][0] * n, style), (g["bar"][1] * (width - n), "dim")]


def spark(values, width, g, style="accent", top=None):
    vals = list(values)[-width:]
    nums = [v for v in vals if v is not None]
    hi = top if top else (max(nums) if nums else 0)
    ramp = g["blocks"]
    s = ""
    for v in vals:
        if v is None:
            s += " "
        elif hi <= 0 or v <= 0:
            s += ramp[1]
        else:
            s += ramp[max(1, min(len(ramp) - 1, int(v / hi * (len(ramp) - 1) + .5)))]
    return [(s.rjust(width), style)]


def box(title, inner, width, g, style="dim"):
    """A frame around `inner` lines (each already `width - 4` wide)."""
    top = [(g["tl"] + g["h"] + " ", style), (title, "bold"), (" " + g["h"] * max(0, width - 5 - len(title)) + g["tr"], style)]
    out = [fit(top, width)]
    for l in inner:
        out.append([(g["v"] + " ", style)] + fit(l, width - 4) + [(" " + g["v"], style)])
    out.append([(g["bl"] + g["h"] * (width - 2) + g["br"], style)])
    return out


def heat(values, width, g):
    """One coloured cell per layer (averaged into `width` cells when there are more layers): green all hits, yellow some misses, red mostly misses, dim = no lookups."""
    n = len(values)
    if n == 0:
        return [(g["none"] * width, "dim")]
    cells = min(n, width)
    out = []
    for c in range(cells):
        chunk = [v for v in values[c * n // cells:(c + 1) * n // cells] if v is not None]
        if not chunk:
            out.append((g["none"], "dim"))
        else:
            v = sum(chunk) / len(chunk)
            out.append((g["heat"], "ok" if v >= .8 else "warn" if v >= .5 else "bad"))
    return out


def kv(rows, width, label_w=None):
    rows = [(k, v) for k, v in rows if v not in (None, "")]
    lw = label_w or (max([len(k) for k, _ in rows] + [4]) + 2)
    return [fit([(f"{k:<{lw}}", "dim"), (str(v), "")], width) for k, v in rows]


def wrap(text, width):
    out, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return out + ([cur] if cur else [])


def hms(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


# ---- chrome ----

def phase(t):
    p, tot = t.payload, t.payload.get("totals") or {}
    st = tot.get("state") or "idle"
    cur = t.rows[-1] if t.rows else {}
    m = p.get("metrics") or {}
    if st in ("exited", "not reachable") or st.startswith("http") or st == "needs an API key":
        return "Error"
    if m.get("requests_deferred"):
        return "Queued"
    if cur.get("busy"):
        return "Generating" if cur.get("dec") is not None else "Reading"
    return "Idle"


def header(t, W, g):
    p, tot = t.payload, t.payload.get("totals") or {}
    st = tot.get("state") or "idle"
    sty = "ok" if st in ("ready", "waiting") else "bad" if st in ("exited", "not reachable") else "warn" if st not in ("idle", "stopped") else "dim"
    ph = phase(t)
    ctx = (p.get("ctx") or {}).get("max")
    model = (tot.get("model") or "no model") + (f" {ctx // 1024}K" if ctx else "")
    right = [(g["dot"] + " ", sty), ({"Reading": "Reading the prompt", "Generating": "Generating a reply", "Idle": "Idle", "Queued": "Queued", "Error": "Error"}[ph] if st in ("ready", "waiting") else st, sty),
             ("   [FROZEN]" if t.frozen else "   [LIVE]", "warn" if t.frozen else "ok")]
    if W < 84:
        right = right[:1] + [(right[1][0] if len(right[1][0]) < 12 else right[1][0][:11], sty), (" [FROZEN]" if t.frozen else " [LIVE]", right[2][1])]
    left = [("MOE-CACHE ", "bold"), (VIEWS[t.view] + "  ", "tab"), (t.src_url + "  ", "dim"), (model, "bold")]
    if plain(left) + plain(right) + 1 > W:                       # narrow screen: drop the address first, then cut the model name
        left = [("MOE-CACHE ", "bold"), (VIEWS[t.view] + "  ", "tab"), (model, "bold")]
    room = W - plain(right) - 1
    left = fit(left, room)
    gap = W - plain(left) - plain(right)
    line1 = fit(left + [(" " * max(1, gap), "")] + right, W)
    tabs, narrow = [], W < 90
    for i, name in enumerate(VIEWS):
        tabs.append((f"[{i + 1} {name}]" if i == t.view else (f"{i + 1} {name}" if narrow else f" {i + 1} {name} "), "tab" if i == t.view else ""))
        tabs.append((" ", ""))
    return [line1, fit(tabs, W), [(g["h"] * W, "dim")]]


def footer(t, W, g):
    if t.typing:
        return [fit([("search: ", "dim"), (t.query + "_", "bold"), ("   Enter keeps it, Esc clears", "dim")], W), []]
    keys = "1-7 views   space freeze   r refresh   t theme   b open in browser   ? help   q quit" if W >= 90 else "1-7 views  space freeze  r refresh  t theme  b browser  ? help  q quit"
    if t.view == 5:
        keys = "/ search   l level   s source   f follow   arrows/PgUp/PgDn scroll   space freeze   b browser   ? help   q quit"
    if t.view == 2:
        keys = "up/down select   1-7 views   space freeze   r refresh   b browser   ? help   q quit"
    msg = t.msg if time.time() - t.msg_t < 4 else ""
    return [fit([(keys, "dim")], W), fit([(msg, "warn")] if msg else [(t.note or ("Read only  " + g["mid"] + "  q leaves the server running"), "dim")], W)]


# ---- views ----

TILE_ORDER = ["speed", "hit", "ssd", "exec", "gpu", "vram", "ram", "cpu", "temp", "power", "pcie", "disk"]


def make_tile(name, t, w, g):
    p, tot = t.payload, t.payload.get("totals") or {}
    cur = t.rows[-1] if t.rows else {}
    iw = w - 4
    S = lambda k, style="accent": spark(t.series(k, iw), iw, g, style, top=FIXED_TOP.get(k))
    na = lambda title, why: box(title, [[("n/a", "dim")], [(why, "dim")], [(" " * iw, "")]], w, g)
    cap = p.get("cap")
    if name == "speed":
        busy = bool(cur.get("busy"))
        big = f"{fnum(cur.get('dec'))} tok/s" if busy and cur.get("dec") is not None else "idle" if not busy else "reading"
        sub = "now" if busy and cur.get("dec") is not None else f"prompt {fnum(cur.get('pre'), 0)} tok/s" if busy else (f"last {fnum((p.get('requests') or [{}])[-1].get('decode_tps'))} tok/s" if p.get("requests") else "")
        return box("Speed", [[(big, "bold")], [(sub, "dim")], S("dec")], w, g)
    if name == "hit":
        hit = t.last("hit")
        if hit is None:
            return na("Cache hit", "no plugin numbers here")
        return box("Cache hit", [[(f"{hit * 100:.1f} %", "bold")], [(f"decode {fnum((t.last('hit_dec') or 0) * 100, 0)} %  run {fnum((tot.get('hits_total') or 0) * 100, 0)} %", "dim")], S("hit", "ok")], w, g)
    if name == "ssd":
        ssd, wait = t.last("ssd"), t.last("wait")
        if ssd is None:
            return na("SSD read", "no plugin numbers here")
        return box("SSD read", [[(f"{ssd:.2f} GB/s", "bold")], [(f"waiting {(wait or 0) * 100:.0f} % of time", "warn" if (wait or 0) > .3 else "dim")], S("ssd", "gpu")], w, g)
    if name == "exec":
        cpu = t.last("cpu")
        if cpu is None:
            return na("CPU executor", "no plugin numbers here")
        thr = (p.get("plan") or {}).get("threads")
        return box("CPU executor", [[(f"{cpu * 100:.0f} %", "bold")], [(f"{thr} threads" if thr else "of cache time", "dim")], S("cpu")], w, g)
    if name == "gpu":
        v = t.last("gpu_util")
        if v is None:
            return na("GPU load", "no reading here")
        return box("GPU load", [[(f"{v:.0f} %", "bold")], [(str(tot.get("gpu") or ""), "dim")], S("gpu_util", "gpu")], w, g)
    if name == "vram":
        vr, vt = t.last("vram"), tot.get("vram_total")
        if vr is None:
            return na("VRAM", "no reading here")
        return box("VRAM", [[(f"{vr / GIB:.1f} / {vt / GIB:.0f} GiB" if vt else f"{vr / GIB:.1f} GiB", "bold")], [((f"{(p.get('extra') or {}).get('gpu_segments')} GPU segments" if (p.get("extra") or {}).get("gpu_segments") else ""), "dim")], S("vram", "gpu")], w, g)
    if name == "ram":
        rss, ram = t.last("rss"), t.last("ram")
        mem = rss if rss is not None else ram
        if mem is None:
            return na("RAM", "no reading here")
        sty = "bad" if cap and mem > cap else "warn" if cap and mem > .92 * cap else "bold"
        return box("RAM", [[(f"{mem / GIB:.1f}" + (f" / {cap / GIB:.0f} GiB" if cap and rss is not None else " GiB"), sty)], [("server, against the cap" if rss is not None else "whole machine", "dim")], S("rss" if rss is not None else "ram", "ok")], w, g)
    if name == "cpu":
        v = t.last("cpu_sys")
        if v is None:
            return na("CPU", "no reading here")
        c, th = tot.get("cores"), tot.get("threads")
        return box("CPU", [[(f"{v:.0f} %", "bold")], [((f"{c} cores {g['mid']} " if c else "") + f"{th} threads" + (f" {g['mid']} server {t.last('cpu_proc'):.1f}" if t.last("cpu_proc") is not None else ""), "dim")], S("cpu_sys")], w, g)
    if name == "temp":
        v = t.last("gpu_temp")
        if v is None:
            return na("GPU temp", "no reading here")
        return box("GPU temp", [[(f"{v:.0f}{g['deg']}C", "warn" if v >= 80 else "bold")], [(str(tot.get("gpu") or ""), "dim")], S("gpu_temp", "warn" if v >= 80 else "accent")], w, g)
    if name == "power":
        v = t.last("gpu_power")
        if v is None:
            return na("Power", "no reading here")
        lim = t.last("gpu_plimit")
        return box("Power", [[(f"{v:.0f} W", "bold")], [(f"of {lim:.0f} W limit" if lim else "", "dim")], S("gpu_power", "gpu")], w, g)
    if name == "pcie":
        gen, wd, up = t.last("pcie_gen"), t.last("pcie_w"), t.last("up")
        if gen is None and up is None:
            return na("PCIe", "no reading here")
        return box("PCIe", [[(f"Gen{gen:.0f} x{wd:.0f}" if gen is not None else "n/a", "bold")], [(f"to GPU {fnum((up or 0) * 1000, 0)} MB/s", "dim")], S("up", "gpu")], w, g)
    if name == "disk":
        r, wr = t.last("io_r"), t.last("io_w")
        if r is None:
            return na("Disk read", "no reading here")
        return box("Disk read", [[(f"{r:,.1f} MB/s", "bold")], [(f"write {wr:,.1f} MB/s" if wr is not None else "", "dim")], S("io_r", "gpu")], w, g)
    return box(name, [[], [], []], w, g)


def tile_grid(t, W, g, rows, names=TILE_ORDER):
    per = 4 if W >= 100 else 3 if W >= 78 else 2 if W >= 52 else 1
    tw = (W - (per - 1)) // per
    out = []
    shown = names[:rows * per]
    for i in range(0, len(shown), per):
        blocks = [(make_tile(n, t, tw, g), tw) for n in shown[i:i + per]]
        out += join(*blocks)
    return out


def requests_table(t, W, g, n, cols_full=False):
    reqs = (t.payload.get("requests") or [])[::-1][:n]
    risky = any(r.get("risk") for r in reqs)                       # an approximate mode is on: how many steps of each answer it touched
    spec = [("TIME", 8, 0), ("STATUS", 7, 0), ("PROMPT", 7, 1), ("REUSED", 7, 1), ("OUTPUT", 7, 1), ("TOK/S", 6, 1), ("DURATION", 9, 1)] + ([("MTP", 5, 1)] if cols_full else []) + ([("AT RISK", 9, 1)] if risky else [])
    for drop in ("MTP", "STATUS", "REUSED", "DURATION"):          # on a narrow screen the least useful columns go first
        if sum(w + 1 for _, w, _ in spec) <= W:
            break
        spec = [c for c in spec if c[0] != drop]
    names = [c[0] for c in spec]

    def line(cells, style=""):
        pieces = []
        for (name, w, right), c in zip(spec, cells):
            pieces += [((("{:>%d}" if right else "{:<%d}") % w).format(str(c)[:w]), style), (" ", "")]
        return fit(pieces, W)

    def pick(row):                                                  # the cells by column name
        return [row[nm] for nm in names]
    out = [line(names, "dim")]
    cur = t.rows[-1] if t.rows else {}
    if cur.get("busy"):
        out.append(line(pick({"TIME": "now", "STATUS": "Running", "PROMPT": "", "REUSED": "", "OUTPUT": "", "TOK/S": fnum(cur.get("dec")), "DURATION": "", "MTP": "", "AT RISK": ""}), "accent"))
    for r in reqs:
        pt = r.get("prompt_total", r.get("prefill_tokens"))
        out.append(line(pick({"TIME": hms(r["t"]), "STATUS": "Done", "PROMPT": f"{pt:,}" if pt is not None else "-", "REUSED": f"{r['reused']:,}" if r.get("reused") is not None else "-",
                              "OUTPUT": f"{r['decode_tokens']:,}" if r.get("decode_tokens") is not None else "-", "TOK/S": fnum(r.get("decode_tps")),
                              "DURATION": f"{r['total_ms'] / 1000:.1f} s" if r.get("total_ms") else "-", "MTP": f"{r['mtp'] * 100:.0f} %" if r.get("mtp") is not None else "-",
                              "AT RISK": (f"{r['risk']['changed_steps']}/{r['risk']['steps']}" + ("" if r["risk"].get("final") else "~")) if r.get("risk") else "-"})))
    return out[:n + 1]


def view_overview(t, W, H, g):
    p, tot = t.payload, t.payload.get("totals") or {}
    out = []
    ph = phase(t)
    ctx = p.get("ctx") or {}
    depth, cmax = ctx.get("depth"), ctx.get("max")
    label = [("Model state  ", "tab")]
    for s in ("Idle", "Reading", "Generating", "Queued", "Error"):
        label.append((f"[{s}]" if s == ph else f" {s} ", ("bad" if s == "Error" else "ok") if s == ph else "dim"))
        label.append((" ", ""))
    right = [(f"{depth:,} / {cmax:,} tokens {g['mid']} {depth * 100 // cmax}%" if depth is not None and cmax else "", "")]
    out.append(fit(label, W - plain(right)) + right)
    out.append(fit(hbar(depth / cmax if depth is not None and cmax else None, W, g, "ok" if ph != "Error" else "bad"), W))
    out.append([])
    bottom = 9                                     # the bars and the requests table under the tiles
    rows = max(1, min(3, (H - len(out) - bottom - 1) // 5))
    out += tile_grid(t, W, g, rows)
    out.append([])
    lw = max(30, min(46, W * 2 // 5))
    rw = W - lw - 2
    cap, resident, budget = p.get("cap"), tot.get("resident"), tot.get("budget")
    rss = t.last("rss")
    left = []
    left.append(fit([("Context fill  ", "tab"), (f"{human_tokens(depth)} / {human_tokens(cmax)}" + (f"  {depth * 100 // cmax}%" if depth is not None and cmax else ""), "")], lw))
    left.append(fit(hbar(depth / cmax if depth is not None and cmax else None, lw, g, "ok"), lw))
    left.append(fit([("Cache budget  ", "tab"), (f"{resident / GIB:.1f} / {budget / GIB:.1f} GiB" if resident is not None and budget else "-", "")], lw))
    left.append(fit(hbar(resident / budget if resident is not None and budget else None, lw, g, "gpu"), lw))
    left.append(fit([("Server RAM  ", "tab"), (f"{rss / GIB:.1f} / {cap / GIB:.1f} GiB cap" if rss is not None and cap else (f"{rss / GIB:.1f} GiB" if rss is not None else "-"), "")], lw))
    left.append(fit(hbar(rss / cap if rss is not None and cap else None, lw, g, "bad" if cap and rss and rss > cap else "warn" if cap and rss and rss > .92 * cap else "ok"), lw))
    cols = t.cols[-1]["rate"] if t.cols else []
    left.append(fit([("Hit rate by layer", "tab")], lw))
    left.append(fit(heat(cols, lw, g), lw))
    reqs = [[("Recent requests", "tab")]] + requests_table(t, rw, g, 6, cols_full=rw >= 66)
    out += join((left, lw), (reqs, rw), gap=2)
    return out


def view_serving(t, W, H, g):
    p, tot, plan = t.payload, t.payload.get("totals") or {}, t.payload.get("plan") or {}
    m = p.get("metrics") or {}
    half = (W - 2) // 2
    a = [[("Server", "tab")]] + kv([("URL", t.src_url), ("model", tot.get("model")), ("state", tot.get("state")), ("uptime", f"{int(tot['uptime'] // 60)} min" if tot.get("uptime") else None),
                                     ("mode", plan.get("mode")), ("agent mode", "on" if plan.get("agent") else None), ("MTP", "on" if plan.get("mtp") else None),
                                     ("requests running", int(m["requests_processing"]) if m.get("requests_processing") is not None else None), ("requests queued", int(m["requests_deferred"]) if m.get("requests_deferred") is not None else None), ("KV cache full", f"{m['kv_cache_usage_ratio'] * 100:.0f} %" if m.get("kv_cache_usage_ratio") is not None else None)], half)
    kvk = plan.get("kv_k") or plan.get("kv_v")
    b = [[("Plan as it runs", "tab")]] + kv([("-ub / -b", f"{plan['ub']} / {plan.get('b', '-')}" if plan.get("ub") else None), ("KV cache type", (plan.get("kv_k") or "f16") + (" / " + plan["kv_v"] if plan.get("kv_v") and plan.get("kv_v") != plan.get("kv_k") else "") if kvk else None),
                                              ("context", plan.get("ctx")), ("GPU layers", plan.get("ngl")), ("experts on the CPU", "all" if plan.get("cpu_moe_all") else plan.get("cpu_moe_layers")),
                                              ("device", plan.get("device") or tot.get("gpu")), ("threads", plan.get("threads")), ("parallel slots", plan.get("parallel")), ("flash attention", plan.get("flash_attn")),
                                              ("approximate tier", plan.get("approx"))], half)
    out = join((a, half), (b, half), gap=2)
    out.append([])
    out.append([("Token modes (what changes answers; bin/moe_cache_approx.py TOKEN_MODES)", "tab")])
    for mo in p.get("modes") or []:
        act = {True: "in use", False: "off", None: "?"}[mo.get("active")]
        out.append(fit([(f"{mo['id']:<14}", "bold"), (f"{mo['status']:<11}", {"won": "ok", "lost": "bad"}.get(mo["status"], "warn")), (f"{act:<7}", "ok" if mo.get("active") else "dim"), (mo["what"], "dim")], W))
    out.append([])
    out.append([("What the launcher decided", "tab")])
    room = H - len(out) - (3 if plan.get("env") else 1)
    for d in plan.get("decisions") or []:
        for i, l in enumerate(wrap(d, W - 4)):
            if room <= 0:
                break
            out.append(fit([("  " if i else g["mid"] + " ", "dim"), (l, "")], W))
            room -= 1
    if plan.get("env"):
        out.append([])
        out.append(fit([("environment ", "dim"), (" ".join(f"{k.replace('MOE_CACHE_', '')}={v}" for k, v in plan["env"].items() if not re.search(r"_(LIB|GGUF|PROFILE)$", k)), "")], W))
    return out


def view_requests(t, W, H, g):
    reqs = (t.payload.get("requests") or [])
    out = requests_table(t, W, g, max(3, H - 14), cols_full=True)
    t.sel = min(t.sel, max(0, len(reqs) - 1))
    shown = reqs[::-1]
    if shown:
        r = shown[min(t.sel, len(shown) - 1)]
        base = 1 + (1 if (t.rows and t.rows[-1].get("busy")) else 0)
        i = base + min(t.sel, len(shown) - 1)
        if i < len(out):
            out[i] = fit([(text, "sel") for text, _ in out[i]], W)
        out.append([])
        out.append([(f"Request at {hms(r['t'])}", "tab")])
        pt, ru = r.get("prompt_total"), r.get("reused")
        out += kv([("prompt", f"{pt:,} tokens" if pt else (f"{r['prefill_tokens']:,} tokens processed" if r.get("prefill_tokens") is not None else None)),
                   ("reused from the prompt cache", f"{ru:,} tokens ({ru * 100 // pt} %)" if ru is not None and pt else None),
                   ("prompt processing", f"{r['prefill_tokens']:,} tokens in {r['prefill_ms'] / 1000:.1f} s = {r['prefill_tps']:.0f} tok/s" if r.get("prefill_ms") else None),
                   ("decoding", f"{r['decode_tokens']:,} tokens in {r['decode_ms'] / 1000:.1f} s = {r['decode_tps']:.1f} tok/s" if r.get("decode_ms") else None),
                   ("whole request", f"{r['total_ms'] / 1000:.1f} s" if r.get("total_ms") else None), ("MTP acceptance", f"{r['mtp'] * 100:.0f} %" if r.get("mtp") is not None else None),
                   ("steps the routing change touched", (f"{r['risk']['changed_steps']:,} of {r['risk']['steps']:,} decode steps, {r['risk']['picks']:,} picks changed" + ("" if r["risk"].get("final") else " (still counting)")) if r.get("risk") else None),
                   ("context after", f"{r['ctx']:,} tokens" if r.get("ctx") else None)], W)
    else:
        out.append([("no finished request yet", "dim")])
    return out


def trend(label, t, key, W, g, fmt, style="accent", scale=1.0):
    lw, vw = 14, 14
    cur = t.last(key)
    sw = max(10, W - lw - vw - 3)
    return fit([(f"{label:<{lw}}", "dim"), (f"{fmt(cur * scale) if cur is not None else '-':>{vw}}  ", "bold")] + spark(t.series(key, sw), sw, g, style, top=FIXED_TOP.get(key)), W)


def view_engine(t, W, H, g):
    p = t.payload
    ex, a, m = p.get("extra") or {}, p.get("approx") or {}, p.get("mtp") or {}
    out = [[("Cache and storage over the last minutes", "tab")]]
    out.append(trend("hit rate", t, "hit", W, g, lambda v: f"{v:.1f} %", "ok", 100))
    out.append(trend("hit, decode", t, "hit_dec", W, g, lambda v: f"{v:.1f} %", "ok", 100))
    out.append(trend("SSD read", t, "ssd", W, g, lambda v: f"{v:.2f} GB/s", "gpu"))
    out.append(trend("SSD wait", t, "wait", W, g, lambda v: f"{v:.0f} % of time", "warn", 100))
    out.append(trend("executor", t, "cpu", W, g, lambda v: f"{v:.0f} %", "accent", 100))
    out.append(trend("evictions", t, "evict", W, g, lambda v: f"{v:,.0f} /s", "dim"))
    out.append(trend("GPU upload", t, "up", W, g, lambda v: f"{v:.2f} GB/s", "gpu"))
    if a:
        out.append(trend("picks changed", t, "alt", W, g, lambda v: f"{v:,.0f} /s", "warn"))
    out.append([])
    out.append([("Hit rate by layer (top = newest; green all hits, yellow some misses, red mostly misses)", "tab")])
    cols = t.cols[::-1][:max(2, min(10, H - 26))]
    layers = len(cols[0]["rate"]) if cols else 0
    if layers:
        lbl = 10
        for c in cols:
            out.append(fit([(hms(c["t"]) + "  ", "dim")] + heat(c["rate"], W - lbl, g), W))
        tot = p.get("layer_total") or []
        out.append(fit([("whole run  ", "dim")] + heat(tot, W - lbl, g), W))
        worst = sorted([(v, i) for i, v in enumerate(tot) if v is not None])[:6]
        if worst:
            out.append(fit([("lowest layers ", "dim"), ("  ".join(f"{i}: {v * 100:.0f} %" for v, i in worst), "bad" if worst[0][0] < .5 else "warn")], W))
    else:
        out.append([("no per-layer numbers yet (the plugin sends them when it manages experts)", "dim")])
    out.append([])
    half = (W - 2) // 2
    left = [[("Counters", "tab")]] + kv([("evictions (run)", ex.get("evictions")), ("prefetch used", ex.get("pf_used")), ("prefetch issued", ex.get("pf_issued")), ("prefetch wasted", ex.get("pf_wasted")),
                                         ("fallback reads", ex.get("fallbacks")), ("GPU segments", ex.get("gpu_segments")), ("cache policy", ex.get("policy")),
                                         ("cache resident", f"{ex['resident_bytes'] / GIB:.1f} GiB" if ex.get("resident_bytes") is not None else None)], half)
    right = [[("Speed-ups that change answers", "tab")]] + kv([("MTP drafted / accepted", f"{m['generated']:,} / {m['accepted']:,} ({m['rate'] * 100:.0f} %)" if m.get("generated") else None),
                                                              ("approx routing", (a.get("mode") or "") + (" (live)" if a.get("live") else " (at last stop)") if a.get("mode") else None),
                                                              ("looked at", f"{a['considered']:,}" if a.get("mode") else None), ("given up", f"{a['given_up']:,}" if a.get("mode") else None),
                                                              ("replaced", f"{a['replaced']:,}" if a.get("mode") else None),
                                                              ("share changed", f"{(a['given_up'] + a['replaced']) * 100 / a['considered']:.1f} % of the picks looked at" if a.get("considered") else None),
                                                              ("fewer experts", f"{a['fewer']} (dropped {a.get('fewer_dropped', 0):,} of {a.get('fewer_considered', 0):,})" if a.get("fewer") else None)], half)
    if not a and not m.get("generated"):
        right += [fit([("no approximate mode and no MTP drafting in use", "dim")], half)]
    out += join((left, half), (right, half), gap=2)
    return out


def view_hardware(t, W, H, g):
    out = tile_grid(t, W, g, max(1, min(3, (H - 14) // 5)))
    tot = t.payload.get("totals") or {}
    out.append([])
    cap = t.payload.get("cap")
    out += kv([("GPU", tot.get("gpu") or "n/a (no reading on this machine: NVIDIA through nvidia-smi, AMD on Linux through sysfs; AMD on Windows has none without admin rights)"),
               ("CPU threads", tot.get("threads")), ("CPU cores", tot.get("cores")), ("RAM total", f"{tot['ram_total'] / GIB:.1f} GiB" if tot.get("ram_total") else None),
               ("memory cap", f"{cap / GIB:.1f} GiB (the launcher's whole-server limit)" if cap else None),
               ("server RAM now", f"{t.last('rss') / GIB:.1f} GiB" if t.last("rss") is not None else None), ("machine RAM used", f"{t.last('ram') / GIB:.1f} GiB" if t.last("ram") is not None else None),
               ("commit used (Windows)", f"{t.last('commit') / GIB:.1f} GiB" if t.last("commit") is not None else None)], W)
    return out


def view_logs(t, W, H, g):
    lines = t.filtered_log()
    room = max(1, H - 6)
    out = [fit([("Log ", "tab"), (f"{len(lines):,} of {len(t.log):,} lines", "dim"), (f"   level: {t.level}   source: {t.source}   search: {t.query or '-'}   {'following' if t.follow else 'scrolled'}", "dim")], W)]
    if not t.log:
        out.append([("no log lines here: the log comes from a server started by the launcher or the control page (attached servers have none)", "dim")])
        return out
    t.scroll = min(t.scroll, max(0, len(lines) - room))
    end = len(lines) - t.scroll
    view = lines[max(0, end - room):end]
    q = t.query
    re_ = None
    if len(q) > 2 and q[0] == "/" and q.endswith("/"):
        try:
            re_ = re.compile(q[1:-1], re.I)
        except re.error:
            pass
    for l in view:
        style = "bad" if l[3] == "error" else "warn" if l[3] == "warn" else ""
        text = re.sub(r"^[\d.]+ [A-Z] ", "", l[4])                 # llama-server's own clock: the time column already says it
        pieces = [(hms(l[1]) + " ", "dim"), (f"{l[2]:<8}", {"plugin": "accent", "launcher": "gpu", "ui": "ok"}.get(l[2], "dim"))]
        m = (re_.search(text) if re_ else (re.search(re.escape(q), text, re.I) if q else None))
        if m and m.end() > m.start():
            pieces += [(text[:m.start()], style), (text[m.start():m.end()], "sel"), (text[m.end():], style)]
        else:
            pieces.append((text, style))
        out.append(fit(pieces, W))
    return out


def view_connect(t, W, H, g):
    tot = t.payload.get("totals") or {}
    port = tot.get("port")
    base = t.src_url if t.src_kind == "server" else (f"http://127.0.0.1:{port}" if port else "")
    key = (t.api_key if t.key_shown and t.api_key else "(hidden: k shows it)" if t.api_key is not None or t.src_kind == "page" else "(not known when attached)")
    out = [[("Connect an app to this server", "tab")], []]
    out += kv([("OpenAI-compatible URL", f"{base}/v1" if base else None), ("llama-server's own page", base or None), ("model", tot.get("model")), ("API key", key)], W)
    out.append([])
    out.append([("Set up an agent (from a terminal; each command only writes that agent's config)", "tab")])
    for name in ("opencode", "hermes", "claude", "codex"):
        out.append(fit([(f"  python3 bin/moe_cache.py connect {name} --url {base}/v1 --write", "")], W))
    out.append([])
    out.append(fit([("b opens ", "dim"), ("the Monitor page in the browser" if t.src_kind == "page" else "llama-server's page in the browser", "")], W))
    return out


VIEW_FUNCS = [view_overview, view_serving, view_requests, view_engine, view_hardware, view_logs, view_connect]


def help_lines(W, g):
    rows = [("1-7, left/right", "switch view"), ("space", "freeze / resume the updates"), ("r", "refresh now"), ("t", "theme: dark, light, mono"), ("b", "open the control page (or llama-server's page) in the browser"),
            ("/", "search the log (text or /regex/)"), ("l / s / f", "log: level, source, follow"), ("up/down/PgUp/PgDn", "log: scroll; requests: select"), ("k", "Connect: show the API key"), ("q", "leave (the server keeps running)")]
    out = [[("Keys (any key closes this)", "tab")], []]
    out += kv(rows, W, 20)
    out.append([])
    out.append(fit([("Read only: nothing here starts, stops or changes a server.", "dim")], W))
    return out


def render(t, W, H, g=None):
    """The screen as H lines of exactly W visible characters (lists of (text, style) pieces)."""
    g = g or t.g
    W, H = max(40, W), max(12, H)
    head, foot = header(t, W, g), footer(t, W, g)
    room = H - len(head) - len(foot)
    body = help_lines(W, g) if t.help else VIEW_FUNCS[t.view](t, W, room, g)
    body = [fit(l, W) for l in body][:room]
    body += [fit([], W)] * (room - len(body))
    return head + body + foot


def sgr(style, theme):
    return THEMES[theme].get(style, "")


def paint(lines, theme, color=True):
    out = []
    for line in lines:
        s = ""
        for text, style in line:
            code = sgr(style, theme) if color and style else ""
            s += f"\x1b[{code}m{text}\x1b[0m" if code else text
        out.append(s)
    return out


# ---- the terminal ----

def enable_vt():
    """Windows 10+: let the console draw ANSI colour and cursor codes; False when it will not."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(h, mode.value | 0x0004)) or bool(mode.value & 0x0004)
    except Exception:
        return False


_WIN_KEYS = {"H": "up", "P": "down", "K": "left", "M": "right", "I": "pgup", "Q": "pgdn", "G": "home", "O": "end"}
_ESC_KEYS = {"[A": "up", "[B": "down", "[C": "right", "[D": "left", "[5~": "pgup", "[6~": "pgdn", "[H": "home", "[F": "end", "[1~": "home", "[4~": "end", "OH": "home", "OF": "end"}


def decode_keys(data):
    """Bytes (POSIX) -> key names; a lone ESC is 'esc'. Tests feed this recorded sequences."""
    keys, i = [], 0
    while i < len(data):
        c = data[i]
        if c == "\x1b":
            for L in (3, 2):
                if data[i + 1:i + 1 + L] in _ESC_KEYS:
                    keys.append(_ESC_KEYS[data[i + 1:i + 1 + L]]); i += 1 + L
                    break
            else:
                keys.append("esc"); i += 1
        else:
            keys.append({"\r": "enter", "\n": "enter", "\x7f": "backspace", "\x08": "backspace", "\x03": "ctrl-c", "\t": "tab"}.get(c, c)); i += 1
    return keys


class Keys:
    """Non-blocking key reads (no echo) for a terminal; nothing when stdin is not one."""
    def __init__(self):
        self.ok = sys.stdin.isatty()
        self.old = None
        if self.ok and os.name != "nt":
            import termios, tty
            self.old = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())

    def get(self, timeout):
        """A list of key names (empty when nothing was pressed within `timeout`)."""
        if not self.ok:
            time.sleep(timeout)
            return []
        if os.name == "nt":
            import msvcrt
            end, keys = time.time() + timeout, []
            while time.time() < end or msvcrt.kbhit():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\x00", "\xe0"):
                        keys.append(_WIN_KEYS.get(msvcrt.getwch(), ""))
                    else:
                        keys += decode_keys(ch)
                else:
                    time.sleep(0.03)
                if keys:
                    break
            return [k for k in keys if k]
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return []
        data = os.read(sys.stdin.fileno(), 64).decode("utf-8", "replace")
        return decode_keys(data)

    def close(self):
        if self.old is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old)


def choose_glyphs(force_ascii):
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    if force_ascii or not enc.startswith("utf"):
        return ASCII
    if os.name == "nt" and not (os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM")):
        return ASCII                    # the classic console's fonts lack the block characters
    return UNICODE


def top(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py top", description="live full-screen monitor of a running moe-cache server (read only)")
    ap.add_argument("--url", help="the control page's address with ?t=TOKEN")
    ap.add_argument("--port", type=int, help="attach to a llama-server on 127.0.0.1:PORT")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between updates (default 2)")
    ap.add_argument("--ascii", action="store_true", help="plain characters only")
    ap.add_argument("--theme", choices=list(THEMES), default="dark")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--once", action="store_true", help="print one screen and exit (for scripts)")
    ap.add_argument("--view", type=int, default=1, choices=range(1, 8), help="the view to start in (with --once: the one printed)")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    a = ap.parse_args(argv)
    src, why = discover(a.url, a.port)
    if not src:
        print("moe-cache top: " + why, file=sys.stderr)
        return 1
    vt = enable_vt()
    color = vt and not a.no_color and not os.environ.get("NO_COLOR") and (sys.stdout.isatty() or not a.once)
    t = Top(choose_glyphs(a.ascii or not vt), a.theme)
    a.color = color
    t.view, t.src_url, t.src_kind = a.view - 1, src.url, src.kind
    size = lambda: (a.width or shutil.get_terminal_size((110, 33)).columns, a.height or shutil.get_terminal_size((110, 33)).lines)
    if a.once:
        t.feed(src.poll())
        if src.kind == "server":
            time.sleep(min(a.interval, 1.0)); t.feed(src.poll())
        w, h = size()
        print("\n".join(paint(render(t, w, h), t.theme, color)))
        return 0
    return loop(src, t, a)


def loop(src, t, a, on_quit=None):
    """The live screen: poll every `a.interval`, draw only the lines that changed, answer keys. Returns 0."""
    size = lambda: (a.width or shutil.get_terminal_size((110, 33)).columns, a.height or shutil.get_terminal_size((110, 33)).lines)
    color = getattr(a, "color", True)
    keys, out = Keys(), sys.stdout
    err, last = "", []
    out.write("\x1b[?1049h\x1b[?25l\x1b[2J")
    try:
        next_poll = 0.0
        while True:
            now = time.time()
            if not t.frozen and now >= next_poll:
                try:
                    t.feed(src.poll())
                    err = ""
                except (OSError, ValueError) as e:
                    err = f"{type(e).__name__}: {e}"
                next_poll = now + a.interval
            w, h = size()
            lines = paint(render(t, w, h), t.theme, color)
            if err:
                lines[0] = f"\x1b[31m{('connection lost: ' + err)[:w]}\x1b[0m" if color else ("connection lost: " + err)[:w]
            if len(last) != len(lines):
                out.write("\x1b[2J"); last = [None] * len(lines)
            buf = ""
            for i, l in enumerate(lines):
                if l != last[i]:
                    buf += f"\x1b[{i + 1};1H{l}\x1b[0m\x1b[K"
            last = lines
            if buf:
                out.write(buf); out.flush()
            stop = False
            for k in keys.get(0.2):
                act = t.handle(k, h)
                if act == "quit":
                    stop = True
                elif act == "refresh":
                    next_poll = 0.0
                elif act == "open":
                    try:
                        t.notify(src.open_in_browser())
                    except Exception as e:
                        t.notify(f"could not open the browser: {e}")
                elif act == "key":
                    t.api_key = src.api_key()
                    if t.api_key is None:
                        t.notify("no API key in use (or this server was not started by the control page)")
            if stop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        keys.close()
        out.write("\x1b[?25h\x1b[?1049l")
        out.flush()
    return 0




def top_start(argv):
    """moe-cache-server --top MODEL [--ram GB] [--vram GiB] [--ctx N] [--port N] [--mode default|exact|fast|hybrid] [--agent] [--mtp on|off] [--stock] [--llama-server PATH]:
    start the model the way the control page's Start does, and watch it here. q stops the server this command started. Anything more exotic: start it with the
    launcher and attach with `moe_cache.py top`."""
    ap = argparse.ArgumentParser(prog="moe-cache-server --top", description="start a model and watch it in the terminal monitor (q stops it)")
    ap.add_argument("model")
    ap.add_argument("--ram", type=float)
    ap.add_argument("--vram", type=float)
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--mode", choices=["default", "exact", "fast", "hybrid"], default="default")
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--mtp", choices=["on", "off"])
    ap.add_argument("--stock", action="store_true")
    ap.add_argument("--llama-server")
    ap.add_argument("--top", action="store_true")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--ascii", action="store_true")
    ap.add_argument("--theme", choices=list(THEMES), default="dark")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--once", action="store_true", help="wait until the server is ready, print one screen, stop it (for scripts)")
    ap.add_argument("--view", type=int, default=1, choices=range(1, 8))
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    a, extra = ap.parse_known_args(argv)
    if extra:
        print("moe-cache-server --top: " + " ".join(extra) + " is not supported here (common options only); start the server with the launcher and run `moe_cache.py top --port N`", file=sys.stderr)
        return 2
    import moe_cache_ui as ui
    import moe_cache as mc
    model = os.path.abspath(a.model)
    try:
        with open(model, "rb") as f:
            if f.read(4) != b"GGUF":
                raise OSError("not a GGUF file")
    except OSError as e:
        print(f"moe-cache-server --top: {a.model}: {e}", file=sys.stderr)
        return 2
    st = {"model": model, "mode": a.mode, "engine": "stock" if a.stock else "auto", "mtp": {"on": True, "off": False}.get(a.mtp), "agent": a.agent, "no_warm": False, "leave_room": False,
          "no_long_prompt": False, "approx": "", "port": a.port, "llama_server": a.llama_server or mc.default_llama_server()}
    if a.stock:
        st["mode"] = "exact"
    for k, lo, hi, conv, v in (("ram", 0.1, 65536, float, a.ram), ("vram", 0.1, 4096, float, a.vram), ("ctx", 512, 1048576, int, a.ctx)):
        try:
            x = ui.number(v, lo, hi, conv, k)
        except ValueError as e:
            print("moe-cache-server --top: " + str(e), file=sys.stderr)
            return 2
        if x is not None:
            st[k] = x
    runner = ui.Runner()
    try:
        runner.start(st)
    except ValueError as e:
        print("moe-cache-server --top: " + str(e), file=sys.stderr)
        return 1
    vt = enable_vt()
    a.color = vt and not a.no_color and not os.environ.get("NO_COLOR") and (sys.stdout.isatty() or not a.once)
    t = Top(choose_glyphs(a.ascii or not vt), a.theme)
    src = RunnerSource(runner)
    t.view, t.src_kind, t.note = a.view - 1, "runner", "q stops the server this command started"
    try:
        if a.once:
            t0 = time.time()
            while runner.state in ("starting",) and time.time() - t0 < 120:
                time.sleep(0.5)
            time.sleep(min(a.interval, 2.0) + 0.5)
            t.src_url = src.url
            t.feed(src.poll())
            w, h = a.width or shutil.get_terminal_size((110, 33)).columns, a.height or shutil.get_terminal_size((110, 33)).lines
            print("\n".join(paint(render(t, w, h), t.theme, a.color)))
            return 0
        t.src_url = src.url
        return loop(src, t, a)
    finally:
        runner.stop()


if __name__ == "__main__":
    sys.exit(top(sys.argv[1:]))
