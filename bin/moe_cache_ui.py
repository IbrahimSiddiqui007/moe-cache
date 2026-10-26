#!/usr/bin/env python3
"""moe-cache control page (GUI1 step 1; Linux and Windows; Python 3.8+, standard library only).

  moe-cache-server --ui [--ui-port 8070] [--llama-server PATH] [--no-browser]
  python bin/moe_cache.py ui [same options]                     (Windows: the same, with python.exe)

Serves one local page at http://127.0.0.1:8070 (this machine only) that lists the GGUF models in the model folders with the planner's verdict
and its reasons (fits in RAM / does not fit / experts on the GPU), starts and stops one server through the launcher (`moe_cache.py server`),
and shows live speed, cache hit rate, SSD reads, RAM and VRAM, llama-server's own chat page and the log.
Live numbers: the plugin's stats snapshot (MOE_CACHE_STATS_SIGNAL: SIGUSR1 on Linux, a named event on Windows), llama-server's /slots and
its timing lines. The plugin and llama.cpp are not changed.
"""
import argparse, collections, ctypes, glob, json, os, re, signal, subprocess, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_cache as mc  # noqa: E402

HERE = mc.HERE
WINDOWS = mc.WINDOWS
PAGE = os.path.join(HERE, "ui", "index.html")
SETTINGS_FILE = os.path.join(mc.CACHE_DIR, "ui.json")
LAUNCHER = [mc.python_cmd(), os.path.join(HERE, "bin", "moe_cache.py"), "server"]     # tests replace it with a fake server
HEADROOM = 1.5e9            # the launcher's rule: stock when model + 1.5 GB fits in the memory the server may use
GIB = 2**30


# ---- model folders ----

def default_folders():
    out = [os.path.join(os.path.expanduser("~"), "models")]
    if WINDOWS:
        out.append("C:\\models")
    return out


def ollama_roots():
    """Ollama's model stores that exist here (its blobs are GGUF files without an extension; the manifests name them)."""
    c = [os.environ.get("OLLAMA_MODELS"), os.path.join(os.path.expanduser("~"), ".ollama", "models"), "/usr/share/ollama/.ollama/models",
         "/var/lib/ollama/.ollama/models"]
    return [p for p in dict.fromkeys(x for x in c if x) if os.path.isdir(os.path.join(p, "manifests"))]


QUANT_RE = re.compile(r"(?i)(?:^|[-_.])((?:UD-)?I?Q\d(?:_[0-9A-Z]+)*|MXFP4(?:_MOE)?|BF16|F16|F32)(?=[-_.]|$)")


def quant_of(name):
    m = QUANT_RE.findall(name)
    return m[-1].upper() if m else ""


def size_of(path):
    try:
        return sum(os.path.getsize(p) for p in mc.model_parts(path))
    except OSError:
        return 0


def scan_folder(folder, depth=4):
    """GGUF models below `folder`: split models once (their first part), projector files (mmproj) left out."""
    out = []
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        return out
    base = folder.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(folder):
        if root.count(os.sep) - base >= depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in sorted(files):
            if not f.lower().endswith(".gguf") or "mmproj" in f.lower():
                continue
            m = re.search(r"-(\d{5})-of-\d{5}\.gguf$", f)
            if m and m.group(1) != "00001":
                continue
            p = os.path.join(root, f)
            out.append({"path": p, "name": re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", f), "size": size_of(p), "quant": quant_of(f), "source": folder})
    return out


def scan_ollama(root):
    out = []
    for man in glob.glob(os.path.join(root, "manifests", "*", "*", "*", "*")):
        try:
            layers = json.load(open(man, encoding="utf-8")).get("layers", [])
        except Exception:
            continue
        for l in layers:
            if l.get("mediaType") == "application/vnd.ollama.image.model":
                p = os.path.join(root, "blobs", l.get("digest", "").replace(":", "-"))
                if os.path.isfile(p):
                    parts = man.replace(os.sep, "/").split("/")
                    name = parts[-2] + ":" + parts[-1]
                    out.append({"path": p, "name": name, "size": os.path.getsize(p), "quant": "", "source": "Ollama (" + root + ")"})
    return out


def find_models(folders):
    seen, out = set(), []
    for f in folders:
        for m in scan_folder(f):
            if m["path"] not in seen:
                seen.add(m["path"]); out.append(m)
    for r in ollama_roots():
        for m in scan_ollama(r):
            if m["path"] not in seen:
                seen.add(m["path"]); out.append(m)
    return sorted(out, key=lambda m: m["name"].lower())


# ---- what this machine has ----

def ram_now():
    """(total, available) bytes."""
    if WINDOWS:
        return mc.win_memory()
    tot = avail = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                tot = int(line.split()[1]) * 1024.0
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024.0
    except OSError:
        pass
    return tot, avail


def vram_now():
    """{name, used, total} bytes of the GPU (NVIDIA: nvidia-smi; AMD on Linux: sysfs); None when neither answers."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            name, used, tot = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            return {"name": name, "used": float(used) * 2**20, "total": float(tot) * 2**20}
    except Exception:
        pass
    best = None
    for d in glob.glob("/sys/class/drm/card[0-9]*/device"):
        try:
            if open(d + "/vendor").read().strip() != "0x1002":
                continue
            tot = int(open(d + "/mem_info_vram_total").read()); used = int(open(d + "/mem_info_vram_used").read())
        except Exception:
            continue
        if tot >= GIB and (best is None or tot > best["total"]):
            best = {"name": "AMD GPU", "used": float(used), "total": float(tot)}
    return best


_GPU = {"done": False, "gpu": None}


def detect_windows_gpu(ls):
    """Windows without nvidia-smi: the launcher's own way (llama-server --list-devices + the registry). Slow (starts llama-server), so done once."""
    try:
        if WINDOWS and not mc.nvidia_reports():
            g = mc.pick_gpu(mc.list_devices(ls))
            ded = mc.win_dedicated_vram().get(g[1]) if g else None
            if g and not (ded is not None and ded < GIB):          # an integrated GPU plans without VRAM, as the launcher does
                _GPU["gpu"] = {"name": g[1], "free": g[2] * 2**20}
    finally:
        _GPU["done"] = True


def system_info():
    tot, avail = ram_now()
    v = vram_now()
    free_vram = (v["total"] - v["used"]) if v else (_GPU["gpu"]["free"] if _GPU["gpu"] else None)
    return {"os": "Windows" if WINDOWS else sys.platform, "ram_total": tot, "ram_avail": avail, "server_may_use": mc.mem_for_server(None),
            "gpu": (v or _GPU["gpu"] or {}).get("name"), "vram_total": v["total"] if v else None, "vram_free": free_vram,
            "gpu_detecting": WINDOWS and not _GPU["done"]}


# ---- the plan and why (I33): the same rules and tools as the launcher ----

def run_planner(path, ctx, ram, vram, kv="f16"):
    """planner/plan.py --json, exactly as the launcher calls it; raises with the planner's own last line on failure."""
    p = [mc.python_cmd(), os.path.join(HERE, "planner", "plan.py"), path, "--ctx", str(ctx), "--kv", kv, "--json"]
    if ram:
        p += ["--ram-gib", str(ram)]
    if vram:
        p += ["--vram-gib", str(vram)]
    r = subprocess.run(p, capture_output=True, text=True, timeout=300)
    try:
        return json.loads(r.stdout)
    except ValueError:
        raise RuntimeError((r.stderr.strip().splitlines() or ["planner failed"])[-1])


def run_tuner(path, ctx, vram, kv_q8):
    sys.path.insert(0, os.path.join(HERE, "planner"))
    from gguf_plan import tune
    return tune(path, ctx, free=float(vram) * GIB if vram else None, want_cache=False, kv_q8=kv_q8)


def gb(x):
    return f"{x / 1e9:.1f} GB"


def plan_model(path, ram=None, vram=None, ctx=8192, mode="exact"):
    """Verdict, mode and the reasons for one model under these settings. ram in GB, vram in GiB (the launcher's units)."""
    size = size_of(path)
    avail = mc.mem_for_server(ram)
    stock_fits = bool(size and avail and size + HEADROOM <= avail)
    out = {"path": path, "size": size, "avail": avail, "reasons": [], "numbers": {}}
    R = out["reasons"]
    if avail:
        R.append(f"model {gb(size)} {'+' if stock_fits else 'plus'} {HEADROOM / 1e9:.1f} GB headroom "
                 f"{'fits in' if stock_fits else 'is more than'} the {gb(avail)} the server may use"
                 + (" (--ram)" if ram else " (available RAM now)"))
    else:
        R.append("free RAM unknown: set a RAM budget")
    try:
        plan = run_planner(path, ctx, ram, vram, kv="q8_0" if mode == "fast" else "f16")
    except Exception as e:
        out.update(kind="stock" if stock_fits else "error", verdict="fits in RAM: stock llama-server" if stock_fits else "cannot plan this model")
        R.append(f"planner: {e}" + (" (not a MoE model: moe-cache has nothing to manage; llama-server runs it stock)" if "MoE" in str(e) else ""))
        return out
    try:
        tuned = run_tuner(path, ctx, vram, mode == "fast")
    except Exception:
        tuned = None
    no_vram = bool(tuned) and any("no GPU memory information" in w for w in tuned.get("why", []))
    ncmoe = tuned["n_cpu_moe"] if tuned and not no_vram else plan["n_cpu_moe"]       # the launcher's choice with --gpu-cache off, --ub auto
    layers = plan["layers"]
    gpu_layers = max(0, layers - min(ncmoe, layers))
    out["numbers"] = {"layers": layers, "gpu_expert_layers": gpu_layers, "n_cpu_moe": ncmoe, "cache_GiB": plan["cache_GiB"],
                      "cpu_experts_GiB": plan["cpu_experts_GiB"], "free_vram_GiB": plan["free_vram_GiB"], "ub": tuned["ub"] if tuned else None,
                      "experts": plan["experts"], "used": plan["used"], "arch": plan["arch"]}
    plugin_mode = mode in ("fast", "hybrid")
    if stock_fits and not plugin_mode:
        out.update(kind="stock", verdict="fits in RAM: stock llama-server (fastest)")
        R.append("when everything fits, stock llama-server is faster: the page cache holds the whole file (Qwen3.6-35B: 42.9 vs 36.6 tok/s, RESULTS 44)")
    elif ncmoe == 0:
        out.update(kind="gpu", verdict="all experts fit on the GPU")
        R.append("every layer of experts fits in VRAM next to the rest of the model: the cache has nothing to manage")
    else:
        name = {"fast": "moe-cache --fast", "hybrid": "moe-cache --hybrid (experimental)"}.get(mode, "moe-cache exact")
        out.update(kind="cache", verdict=("fits in RAM, but " + mode + " needs the plugin: " if stock_fits else "does not fit: ") + name)
        if mode == "fast":
            R.append("--fast skips low-ranked experts that are not in RAM: about +10 %, about 1 token in 40-50 differs from stock")
        elif mode == "hybrid":
            R.append("--hybrid computes the most used experts on the GPU from a bank, the rest on the CPU (output differs from stock)")
        else:
            R.append("exact: output identical to stock llama-server; experts that are not in RAM are read from the SSD")
    if gpu_layers:
        R.append(f"{gpu_layers} of {layers} layers keep their experts on the GPU ({plan['free_vram_GiB']:.1f} GiB free VRAM)")
    elif out["kind"] != "stock" or plan["free_vram_GiB"] == 0:
        R.append("no layer of experts fits on the GPU" if plan["free_vram_GiB"] else "no GPU with its own memory found: all experts on the CPU")
    if out["kind"] == "cache":
        cpu = plan["cpu_experts_GiB"]
        pct = min(100, 100 * plan["cache_GiB"] / cpu) if cpu else 100
        R.append(f"expert cache {plan['cache_GiB']:.1f} GiB holds {pct:.0f} % of the {cpu:.1f} GiB of CPU experts"
                 + ("" if pct >= 100 else "; the rest are read from the SSD when needed"))
        lim = float(ram) * 1e9 if ram else mc.mem_total()
        if mode == "exact" and lim and size > 2 * lim:
            R.append(f"the model is {size / lim:.1f}x the memory: --fast would add about +10 % (changes some tokens; off by default)")
    for w in (tuned or {}).get("why", []):
        if not no_vram:
            R.append("tuner: " + w)
    if plan.get("note"):
        R.append("planner: " + plan["note"])
    return out


def server_args(s):
    """moe_cache.py server arguments for the page's settings."""
    a = [s["model"], "--port", str(int(s.get("port") or 8080)), "--ctx", str(int(s.get("ctx") or 8192))]
    if s.get("ram"):
        a += ["--ram", str(float(s["ram"]))]
    if s.get("vram"):
        a += ["--vram", str(float(s["vram"]))]
    if s.get("mode") == "fast":
        a.append("--fast")
    elif s.get("mode") == "hybrid":
        a.append("--hybrid")
    if s.get("agent"):
        a.append("--agent")
    if s.get("llama_server"):
        a += ["--llama-server", s["llama_server"]]
    if s.get("api_key"):
        a += ["--api-key", s["api_key"]]
    return a + ["--", "--metrics"]


def shown_command(s):
    a = server_args(dict(s, api_key="KEY" if s.get("api_key") else ""))
    return " ".join(('"' + x + '"') if " " in x else x for x in ["moe-cache-server"] + a)


def dry_run(s):
    a = server_args(s)
    a.insert(a.index("--"), "--dry-run")          # a launcher option: after the -- it would go to llama-server
    r = subprocess.run(LAUNCHER + a, capture_output=True, text=True, timeout=300, env=child_env())
    return [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]


# ---- the running server ----

def child_env():
    return dict(os.environ, MOE_CACHE_STATS_SIGNAL="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")


def child_pid(pid):
    """The launcher's child (llama-server); None when there is none yet."""
    if WINDOWS:
        try:
            from ctypes import wintypes
            class PE(ctypes.Structure):
                _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                            ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                            ("szExeFile", ctypes.c_wchar * 260)]
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
            snap = k.CreateToolhelp32Snapshot(2, 0)
            e = PE(); e.dwSize = ctypes.sizeof(PE)
            found = None
            ok = k.Process32FirstW(wintypes.HANDLE(snap), ctypes.byref(e))
            while ok:
                if e.th32ParentProcessID == pid and e.szExeFile.lower().startswith("llama"):
                    found = e.th32ProcessID
                ok = k.Process32NextW(wintypes.HANDLE(snap), ctypes.byref(e))
            k.CloseHandle(wintypes.HANDLE(snap))
            return found
        except Exception:
            return None
    try:
        for t in os.listdir(f"/proc/{pid}/task"):
            kids = open(f"/proc/{pid}/task/{t}/children").read().split()
            if kids:
                return int(kids[0])
    except (OSError, ValueError):
        pass
    return None


def catches_usr1(pid):
    """Linux: True when the process handles SIGUSR1 (the plugin installs the handler; stock llama-server would die of it)."""
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("SigCgt:"):
                return bool(int(line.split()[1], 16) >> (signal.SIGUSR1 - 1) & 1)
    except (OSError, ValueError, AttributeError):
        pass
    return False


def ask_snapshot(pid):
    """Ask the plugin in llama-server `pid` for one 'moe-cache-stats {json}' line; False when it cannot be asked."""
    if WINDOWS:
        try:
            k = ctypes.windll.kernel32
            h = k.OpenEventW(0x0002, False, f"Local\\moe-cache-stats-{pid}")
            if not h:
                return False
            k.SetEvent(h); k.CloseHandle(h)
            return True
        except Exception:
            return False
    if catches_usr1(pid):
        try:
            os.kill(pid, signal.SIGUSR1)
            return True
        except OSError:
            pass
    return False


def proc_rss(pid):
    if WINDOWS:
        try:
            from ctypes import wintypes
            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            k = ctypes.windll.kernel32
            k.OpenProcess.restype = wintypes.HANDLE
            h = k.OpenProcess(0x1000, False, pid)
            if not h:
                return None
            c = PMC(); c.cb = ctypes.sizeof(PMC)
            ok = k.K32GetProcessMemoryInfo(wintypes.HANDLE(h), ctypes.byref(c), c.cb)
            k.CloseHandle(wintypes.HANDLE(h))
            return float(c.WorkingSetSize) if ok else None
        except Exception:
            return None
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024.0
    except OSError:
        pass
    return None


TIMING_RE = re.compile(r"(prompt eval|eval) time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")


class Runner:
    """One server started through the launcher; its log, live numbers and a 2-minute history."""

    def __init__(self):
        self.lock = threading.Lock()
        self.p = None
        self.reset({})

    def reset(self, s):
        self.settings = s
        self.state, self.rc, self.started = "idle", None, None
        self.log = collections.deque(maxlen=500)
        self.seq = 0
        self.mode = ""
        self.last = {}                     # last finished request: prefill / decode tok/s
        self.live_tps = None
        self.snap = None; self.prev_snap = None
        self.cache = {}
        self.hist = collections.deque(maxlen=120)
        self.pid = None
        self.rss = None
        self.profile = None

    # -- control --
    def start(self, s):
        with self.lock:
            if self.p and self.p.poll() is None:
                raise ValueError("a server is already running: stop it first")
            port = int(s.get("port") or 8080)
            if mc.port_busy(port):
                raise ValueError(f"port {port} is already in use (another server?): pick another port")
            self.reset(s)
            # Linux sends the parent-death signal when the THREAD that started the child ends: started from a request thread, the launcher was
            # killed as soon as the reply went out. So a thread that lives as long as the server starts it (and then polls it)
            got = {}
            ready = threading.Event()
            threading.Thread(target=self.life, args=(s, got, ready), daemon=True).start()
            ready.wait(30)
            if "error" in got or "p" not in got:
                self.state = "exited"
                raise ValueError(f"could not start the launcher: {got.get('error', 'timeout')}")

    def life(self, s, got, ready):
        try:
            kw = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "env": child_env()}
            if sys.platform.startswith("linux"):
                kw["preexec_fn"] = mc._die_with_parent_linux
            p = self.p = got["p"] = subprocess.Popen(LAUNCHER + server_args(s), **kw)
        except Exception as e:
            got["error"] = str(e)
            ready.set()
            return
        if WINDOWS:
            try:
                mc._kill_with_parent_windows(p)          # closing the page server also ends the launcher and so llama-server
            except Exception:
                pass
        self.state, self.started = "starting", time.time()
        self.add_line("ui: " + shown_command(s))
        threading.Thread(target=self.read, args=(p,), daemon=True).start()
        ready.set()
        self.poll(p)
        p.wait()

    def stop(self):
        p = self.p
        if not p or p.poll() is not None:
            return
        self.state = "stopping"
        if WINDOWS:
            # a server on Windows is killed, so the plugin's save at exit never runs: ask for the usage profile first (as the launcher does)
            kid = child_pid(p.pid)
            if kid and self.profile:
                mc.win_request_save(kid, self.profile, timeout=30)
            p.terminate()
        else:
            p.send_signal(signal.SIGTERM)      # the launcher passes it on and waits for llama-server
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()

    # -- input --
    def add_line(self, line):
        self.seq += 1
        self.log.append((self.seq, line))

    def read(self, p):
        for raw in iter(p.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line.startswith("moe-cache-stats "):
                try:
                    self.take_snapshot(json.loads(line[len("moe-cache-stats "):]))
                except ValueError:
                    pass
                continue
            self.add_line(line)
            m = TIMING_RE.search(line)
            if m:
                key = "prefill" if m.group(1) == "prompt eval" else "decode"
                self.last[key + "_tps"] = float(m.group(4)); self.last[key + "_tokens"] = int(m.group(3))
            if "running stock llama-server" in line:
                self.mode = "stock llama-server (the model fits in RAM)"
            elif "stay on the CPU and are managed by the cache" in line:
                self.mode = "moe-cache " + {"fast": "--fast", "hybrid": "--hybrid"}.get(self.settings.get("mode"), "exact")
            elif "all experts fit in your GPU memory" in line:
                self.mode = "all experts on the GPU"
            m = re.search(r"profile (\S+\.usage)", line)
            if m:
                self.profile = m.group(1).rstrip(";")
        rc = p.wait()
        if p is self.p:
            self.rc = rc
            self.state = "stopped" if self.state == "stopping" or rc in (0, -signal.SIGTERM, 143, 1 if WINDOWS else 0) else "exited"
            self.add_line(f"ui: the server ended (exit code {rc})")

    def take_snapshot(self, s):
        now = time.time()
        prev, self.prev_snap = self.snap, self.snap
        self.snap = (now, s)
        c = {"resident": s.get("resident_bytes"), "budget": s.get("budget")}
        h, m = s.get("hits", 0), s.get("misses", 0)
        c["hit_rate_total"] = h / (h + m) if h + m else None
        if prev:
            t0, p0 = prev
            dh, dm = h - p0.get("hits", 0), m - p0.get("misses", 0)
            c["hit_rate"] = dh / (dh + dm) if dh + dm > 0 else self.cache.get("hit_rate")
            c["read_mbs"] = max(0, s.get("read_bytes", 0) - p0.get("read_bytes", 0)) / max(now - t0, 1e-3) / 1e6
        self.cache = c

    def poll(self, p):
        api_key = self.settings.get("api_key")
        port = int(self.settings.get("port") or 8080)
        n_prev, t_prev, tick = None, None, 0
        while p.poll() is None:
            tick += 1
            if self.state == "starting" and mc.healthy(port, api_key):
                self.state = "ready"
                if not self.mode:
                    self.mode = "llama-server"
            self.pid = self.pid or child_pid(p.pid)
            self.rss = proc_rss(self.pid) if self.pid else None
            if self.state == "ready":
                st, body = mc.http(f"http://127.0.0.1:{port}/slots", api_key=api_key, timeout=2)
                n = busy = None
                if st == 200:
                    try:
                        slots = json.loads(body)
                        busy = any(x.get("is_processing") for x in slots)
                        n = sum(int((x.get("next_token") or [{}])[0].get("n_decoded", 0) if isinstance(x.get("next_token"), list)
                                    else x.get("n_decoded", 0)) for x in slots)
                    except (ValueError, TypeError, AttributeError):
                        n = None
                now = time.time()
                if busy and n is not None and n_prev is not None and n > n_prev:
                    self.live_tps = (n - n_prev) / (now - t_prev)
                elif not busy:
                    self.live_tps = None
                n_prev, t_prev = n, now
                if tick % 2 == 0 and self.pid:
                    ask_snapshot(self.pid)
            self.hist.append([round(time.time(), 1), self.live_tps, self.cache.get("hit_rate")])
            time.sleep(1)

    def status(self):
        s = self.settings
        port = int(s.get("port") or 8080)
        return {"state": self.state, "rc": self.rc, "model": s.get("model"), "port": port, "url": f"http://127.0.0.1:{port}",
                "uptime": time.time() - self.started if self.started and self.state in ("starting", "ready") else None,
                "mode": self.mode, "command": shown_command(s) if s.get("model") else "", "agent": bool(s.get("agent")), "ctx": s.get("ctx"),
                "live_tps": self.live_tps, "last": self.last, "cache": self.cache, "server_rss": self.rss, "log_seq": self.seq,
                "history": list(self.hist), "system": system_info()}


# ---- the page server ----

class Store:
    def __init__(self, ls):
        self.ls = ls
        try:
            self.saved = json.load(open(SETTINGS_FILE, encoding="utf-8"))
        except Exception:
            self.saved = {}
        self.models, self.models_at = [], 0
        self.plans = {}

    def folders(self):
        return list(dict.fromkeys(default_folders() + self.saved.get("folders", [])))

    def save(self):
        try:
            os.makedirs(mc.CACHE_DIR, exist_ok=True)
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.saved, f, indent=1)
        except OSError:
            pass

    def list_models(self, fresh=False):
        if fresh or time.time() - self.models_at > 30:
            self.models, self.models_at = find_models(self.folders()), time.time()
        return self.models


class Handler(BaseHTTPRequestHandler):
    store = None
    runner = None
    server_version = "moe-cache-ui"

    def log_message(self, *a):
        pass

    def allowed(self):
        """Only pages from this server: the Host must be ours (no DNS rebinding) and a foreign Origin is refused (no cross-site requests)."""
        port = self.server.server_address[1]
        ok = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in ok:
            return False
        o = self.headers.get("Origin")
        return o is None or o in {"http://" + h for h in ok}

    def send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, str)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text", "application/json")) else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return self.send(403, {"error": "forbidden"})
        u = urllib.parse.urlparse(self.path)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self.send(200, open(PAGE, "rb").read(), "text/html")
            if u.path == "/api/system":
                return self.send(200, dict(system_info(), folders=self.store.folders(), last=self.store.saved.get("last", {}),
                                           llama_server=self.store.ls))
            if u.path == "/api/models":
                return self.send(200, {"folders": self.store.folders(), "models": self.store.list_models(q.get("fresh") == "1")})
            if u.path == "/api/plan":
                m = self.known(q.get("model"))
                key = (m, q.get("ram"), q.get("vram"), q.get("ctx"), q.get("mode"))
                hit = self.store.plans.get(key)
                if not hit or time.time() - hit[0] > 60:          # free RAM and VRAM change: a plan is kept for a minute
                    hit = self.store.plans[key] = (time.time(), plan_model(m, q.get("ram") or None, q.get("vram") or None, int(q.get("ctx") or 8192), q.get("mode") or "exact"))
                return self.send(200, hit[1])
            if u.path == "/api/command":
                s = self.settings(q)
                return self.send(200, {"launcher": shown_command(s), "lines": dry_run(s)})
            if u.path == "/api/status":
                return self.send(200, self.runner.status())
            if u.path == "/api/log":
                since = int(q.get("since") or 0)
                return self.send(200, {"lines": [[n, l] for n, l in list(self.runner.log) if n > since]})
        except ValueError as e:
            return self.send(400, {"error": str(e)})
        except Exception as e:
            return self.send(500, {"error": f"{type(e).__name__}: {e}"})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.allowed() or not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self.send(403, {"error": "forbidden"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/api/folders":
                f = os.path.abspath(os.path.expanduser(str(body.get("add") or body.get("remove") or "")))
                lst = self.store.saved.setdefault("folders", [])
                if body.get("add"):
                    if not os.path.isdir(f):
                        raise ValueError(f"no such folder: {f}")
                    if f not in lst:
                        lst.append(f)
                elif f in lst:
                    lst.remove(f)
                self.store.save()
                return self.send(200, {"folders": self.store.folders(), "models": self.store.list_models(True)})
            if self.path == "/api/start":
                s = self.settings(body)
                self.store.saved["last"] = {k: v for k, v in s.items() if k != "api_key"}
                self.store.save()
                self.runner.start(s)
                return self.send(200, self.runner.status())
            if self.path == "/api/stop":
                threading.Thread(target=self.runner.stop, daemon=True).start()
                return self.send(200, {"ok": True})
            if self.path == "/api/connect":
                agent = body.get("agent")
                if agent not in ("opencode", "hermes") or self.runner.state != "ready":
                    raise ValueError("start a server with Agent mode first; agents: opencode, hermes")
                st = self.runner.status()
                cmd = [mc.python_cmd(), os.path.join(HERE, "bin", "moe_cache.py"), "connect", agent, "--url", st["url"], "--write"]
                env = dict(os.environ, LLAMA_API_KEY=self.runner.settings.get("api_key") or "")
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
                return self.send(200, {"rc": r.returncode, "output": (r.stdout + r.stderr).strip()})
        except ValueError as e:
            return self.send(400, {"error": str(e)})
        except Exception as e:
            return self.send(500, {"error": f"{type(e).__name__}: {e}"})
        self.send(404, {"error": "not found"})

    def known(self, path):
        """Only models the page listed can be planned or started."""
        for m in self.store.list_models():
            if m["path"] == path:
                return path
        raise ValueError("unknown model (refresh the list)")

    def settings(self, d):
        s = {"model": self.known(d.get("model")), "mode": d.get("mode") if d.get("mode") in ("exact", "fast", "hybrid") else "exact",
             "agent": str(d.get("agent")).lower() in ("1", "true", "on"), "llama_server": self.store.ls, "api_key": d.get("api_key") or ""}
        for k, conv in (("ram", float), ("vram", float), ("ctx", int), ("port", int)):
            v = d.get(k)
            if v not in (None, ""):
                s[k] = conv(v)
        if not 1 <= s.get("port", 8080) <= 65535 or s.get("ctx", 8192) < 512:
            raise ValueError("bad port or context")
        if s.get("api_key") and not re.fullmatch(r"[\x21-\x7e]{1,200}", s["api_key"]):
            raise ValueError("the API key may only contain visible ASCII characters")
        return s


def make_server(port=8070, ls=None):
    store = Store(ls or os.environ.get("LLAMA_SERVER", "llama-server"))
    h = type("H", (Handler,), {"store": store, "runner": Runner()})
    srv = ThreadingHTTPServer(("127.0.0.1", port), h)
    srv.daemon_threads = True
    return srv


def ui(argv):
    ap = argparse.ArgumentParser(prog="moe-cache-server --ui", description="local control page for moe-cache (this machine only)")
    ap.add_argument("--ui-port", type=int, default=8070)
    ap.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER", "llama-server"))
    ap.add_argument("--no-browser", action="store_true", help="do not open the page in the browser")
    a = ap.parse_args(argv)
    try:
        srv = make_server(a.ui_port, a.llama_server)
    except OSError as e:
        raise SystemExit(f"moe-cache: cannot listen on 127.0.0.1:{a.ui_port} ({e}): --ui-port N to pick another port")
    if WINDOWS:
        threading.Thread(target=detect_windows_gpu, args=(a.llama_server,), daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    mc.say(f"moe-cache: control page at {url} (this machine only; Ctrl-C stops it and any server it started)")
    if not a.no_browser:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.RequestHandlerClass.runner.stop()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(ui(sys.argv[1:]))
