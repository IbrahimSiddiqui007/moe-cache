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
import argparse, collections, csv, ctypes, glob, json, math, secrets, os, re, signal, subprocess, sys, threading, time, urllib.parse
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


# llama.cpp's llama_ftype numbers (general.file_type) -> the usual quant names
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
              14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
              23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
              36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE"}


def header_quant(path):
    """The quant named in the GGUF header (general.file_type), for files whose name says nothing (Ollama's blobs); "" when unknown."""
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        import struct
        from gguf_plan import _read_str, _skip_value, _FMT
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return ""
            f.read(4)
            _, n_kv = struct.unpack("<QQ", f.read(16))
            for _ in range(n_kv):
                key = _read_str(f)
                t = struct.unpack("<I", f.read(4))[0]
                if key == "general.file_type" and t in (4, 5):
                    return FILE_TYPES.get(struct.unpack(_FMT[t], f.read(4))[0], "")
                _skip_value(f, t)
    except Exception:
        pass
    return ""


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
                    out.append({"path": p, "name": name, "size": os.path.getsize(p), "quant": header_quant(p), "source": "Ollama (" + root + ")"})
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


_GPU = {"done": False, "gpu": None, "watched": 0.0}


def windows_gpu_once(ls):
    """Windows without nvidia-smi (AMD, Intel): the GPU the launcher picks (llama-server --list-devices, discrete before integrated) with its
    free memory, and its total dedicated VRAM from the registry. {name, free, total, integrated} or None."""
    g = mc.pick_gpu(mc.list_devices(ls))
    if not g:
        return None
    ded = mc.win_dedicated_vram().get(g[1])
    integrated = ded is not None and ded < GIB       # the launcher plans without VRAM there: Vulkan's "free" is shared system RAM
    return {"name": g[1] + (" (integrated: shares RAM)" if integrated else ""), "free": g[2] * 2**20 if g[2] else None,
            "total": float(ded) if ded and not integrated else None, "integrated": integrated, "device": g[0], "adapter": g[1]}


def watch_windows_gpu(ls, every=10.0):
    """Keeps _GPU current on Windows without nvidia-smi. Asking llama-server --list-devices takes a second or two, so: every 10 s, and only
    while the page is open (a status request in the last 30 s)."""
    if not WINDOWS or mc.nvidia_reports():
        _GPU["done"] = True
        return
    while True:
        try:
            _GPU["gpu"] = windows_gpu_once(ls)
        except Exception:
            pass
        _GPU["done"] = True
        time.sleep(every)
        while time.time() - _GPU["watched"] > 30:
            time.sleep(1)


def system_info():
    tot, avail = ram_now()
    v = vram_now()
    w = _GPU["gpu"] if not v else None
    if v:
        free_vram, total = v["total"] - v["used"], v["total"]
    elif w:
        free_vram, total = w["free"], w["total"]
    else:
        free_vram = total = None
    return {"os": "Windows" if WINDOWS else sys.platform, "ram_total": tot, "ram_avail": avail, "server_may_use": mc.mem_for_server(None),
            "gpu": (v or w or {}).get("name"), "vram_total": total, "vram_free": free_vram, "gpu_integrated": bool(w and w.get("integrated")),
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


# Measured decode speeds from README "Results" / docs/RESULTS.md, all on the test laptop (i7-13620H, 31 GB RAM, RTX 4060 8 GB, NVMe SSD), by model
# family (gguf_plan.model_key) and file size (GB, so each quant has its own row). (RAM GB, setting, stock tok/s, moe-cache tok/s, where)
MEASURED = {
    ("qwen35moe-41x256", 21.7): [(24, "fits in RAM (warm)", 42.9, 36.6, "RESULTS 44: the launcher runs stock here"),
                                 (12, "8 GB GPU", 6.9, 22.6, "RESULTS 44"), (12, "no GPU", 3.1, 9.9, "RESULTS 5")],
    ("qwen4exp-48x512", 68.0): [(24, "8 GB GPU", 5.2, 12.5, "RESULTS 44; 17.6 with --hybrid (approximate), RESULTS 31")],
    ("qwen35moe-48x256", 36.6): [(24, "8 GB GPU", 1.89, 7.7, "RESULTS 45; 9.0 with --fast")],
    ("qwen35moe-48x256", 44.7): [(24, "8 GB GPU", 1.10, 5.7, "RESULTS 45; 6.8 with --fast")],
    ("qwen35moe-48x256", 60.2): [(24, "8 GB GPU", 0.91, 3.7, "RESULTS 45; 4.5 with --fast")],
    ("glm5-next-46x288", 101.8): [(24, "8 GB GPU", 0.32, 1.75, "RESULTS 26; 2.1 with --fast")],
}


def measured(path, size):
    """Lines like 'test laptop, 12 GB RAM, 8 GB GPU: stock 6.9, moe-cache 22.6 tok/s (RESULTS 44)' for a model measured before; [] otherwise."""
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import model_key
        key = model_key(path)
    except Exception:
        return []
    for (k, gbs), rows in MEASURED.items():
        if k == key and abs(size / 1e9 - gbs) <= 0.03 * gbs:
            return [f"{ram} GB RAM, {what}: stock {st} tok/s, moe-cache {mc_} tok/s ({src})" for ram, what, st, mc_, src in rows]
    return []


def gb(x):
    return f"{x / 1e9:.1f} GB"


def windows_amd_rules(gpu, path, size, avail, agent=False, windows=WINDOWS):
    """The launcher's Windows + AMD rules that apply to this model, as (all experts on the CPU?, [reasons in the launcher's words]).
    gpu: {device, adapter} as windows_gpu_once finds it. Same conditions as moe_cache.server(); the checkpoint rule is the launcher's own function."""
    if not (windows and gpu and str(gpu.get("device", "")).startswith("Vulkan") and re.search(r"AMD|Radeon", gpu.get("adapter", ""))):
        return False, []
    reasons, cpu_only = [], False
    if avail and size > avail:
        cpu_only = True
        reasons.append("Windows + AMD: all experts stay on the CPU (on AMD's Windows Vulkan driver VRAM in use also costs system RAM, so GPU expert layers "
                       "starve the cache when the model is bigger than RAM; Flash-Next 3 -> 12-13.5 tok/s; --n-cpu-moe N overrides)")
    import contextlib, io
    buf, was = io.StringIO(), mc.WINDOWS
    try:
        mc.WINDOWS = True                        # the function checks it; here the caller already decided this is Windows
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            args = mc.amd_windows_checkpoint_args((gpu["device"], gpu["adapter"], 0), path, [], agent=agent)
    except Exception:
        args = []
    finally:
        mc.WINDOWS = was
    if args:
        reasons.append("Windows + AMD: " + buf.getvalue().strip().replace("moe-cache: ", "", 1))
    return cpu_only, reasons


def plan_model(path, ram=None, vram=None, ctx=8192, mode="exact", agent=False):
    """Verdict, mode and the reasons for one model under these settings. ram in GB, vram in GiB (the launcher's units)."""
    size = size_of(path)
    avail = mc.mem_for_server(ram)
    gpu = _GPU["gpu"] if WINDOWS else None
    if vram is None and gpu and gpu.get("free") and not gpu.get("integrated"):
        vram = f"{gpu['free'] / GIB:.2f}"          # as the launcher: on Windows without nvidia-smi it passes the free VRAM it found to the planner
    amd_cpu_only, amd_reasons = windows_amd_rules(gpu, path, size, avail, agent, windows=WINDOWS)
    stock_fits = bool(size and avail and size + HEADROOM <= avail)
    out = {"path": path, "size": size, "avail": avail, "reasons": [], "numbers": {}, "measured": measured(path, size)}
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
    if amd_cpu_only:
        ncmoe = layers
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
    R.extend(amd_reasons)
    for w in (tuned or {}).get("why", []):
        if not no_vram and not amd_cpu_only:
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
    return a + ["--", "--metrics"]          # the API key goes to the launcher in LLAMA_API_KEY (child_env), never on a command line


def shown_command(s):
    return (("LLAMA_API_KEY=KEY " if s.get("api_key") else "")
            + " ".join(('"' + x + '"') if " " in x else x for x in ["moe-cache-server"] + server_args(s)))


def dry_run(s):
    a = server_args(s)
    a.insert(a.index("--"), "--dry-run")          # a launcher option: after the -- it would go to llama-server
    r = subprocess.run(LAUNCHER + a, capture_output=True, text=True, timeout=300, env=child_env(s.get("api_key")))
    key = s.get("api_key")
    return [l.replace(key, "KEY") if key else l for l in (r.stdout + r.stderr).splitlines() if l.strip()]


# ---- the running server ----

# Owner's decision C (handover/DECISION_LLAMA_SERVER_KEY_20261026.md): agent servers get a stable per-user random key from the launcher.
# The page asks the launcher for it (stdout: the key alone, exit 0), keeps it in memory, sends it in headers and to `connect` through the
# environment; never in a URL, a log line or the status JSON (the page fetches it from /api/key only for the Copy button).
KEY_CMD = [mc.python_cmd(), os.path.join(HERE, "bin", "moe_cache.py"), "key"]


def launcher_key():
    """The launcher's key for agent servers; None when this launcher has none (before option C) or the command fails."""
    try:
        r = subprocess.run(KEY_CMD, capture_output=True, text=True, timeout=30, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    except Exception:
        return None
    k = r.stdout.strip()
    return k if r.returncode == 0 and re.fullmatch(r"[\x21-\x7e]{8,200}", k) else None


def child_env(api_key=None):
    """The launcher's environment: stats on request, unbuffered UTF-8 output, and the API key (the launcher's --api-key default is LLAMA_API_KEY)."""
    env = dict(os.environ, MOE_CACHE_STATS_SIGNAL="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    env.pop("LLAMA_API_KEY", None)
    if api_key:
        env["LLAMA_API_KEY"] = api_key
    return env


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


# lines the page lifts out of the log: the launcher's and the plugin's warnings (e.g. the cache shrank because llama-server grew), the plugin
# disabling itself, and why llama-server stopped
WARN_RE = re.compile(r"(?i)moe-cache: (warning|.*plugin disabled|.*refus|llama-server (exited|was stopped))|^WARNING|^error: |\binvalid argument\b")
def warm_files():
    """Saved agent conversations the launcher restores at the next start (~/.cache/moe-cache/slots/agent/<model>-c<ctx>-s<slot>.bin)."""
    out = []
    for f in glob.glob(os.path.join(mc.CACHE_DIR, "slots", "agent", "*-s*.bin")):
        if f.endswith(".new.bin"):
            continue
        try:
            out.append({"name": os.path.basename(f), "size": os.path.getsize(f), "t": os.path.getmtime(f)})
        except OSError:
            pass
    return sorted(out, key=lambda x: -x["t"])[:10]


# ---- quick benchmark: the method of windows-kit/pc_speed.ps1 (RESULTS 54/55), for the running server ----
BENCH_CSV = os.path.join(mc.CACHE_DIR, "bench.csv")
BENCH_PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
                 "Explain how a CPU cache works, with an example of a cache miss.",
                 "Describe the water cycle step by step for a ten-year-old."]
BENCH_PARA = ("The history of computing spans centuries, from mechanical calculators and punched cards to vacuum tubes, transistors, integrated "
              "circuits and the microprocessors in every phone. ")
BENCH_LONG = "Summarise the following text in five bullet points.\n" + BENCH_PARA * 95          # about 3.5k tokens
BENCH_FIELDS = ["time", "model", "size_gb", "mode", "ram_gb", "vram_gib", "ctx", "gpu", "os", "decode_1", "decode_2", "decode_mean", "decode_tokens",
                "prefill_tokens", "prefill", "hit_rate", "warmup_s", "command"]


def read_bench_rows(limit=20):
    try:
        with open(BENCH_CSV, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))[-limit:]
    except OSError:
        return []


def append_bench_row(row):
    os.makedirs(os.path.dirname(BENCH_CSV), exist_ok=True)
    new = not os.path.isfile(BENCH_CSV)
    with open(BENCH_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BENCH_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: csv_safe(v) for k, v in row.items()})


def csv_safe(v):
    """A cell a spreadsheet would run as a formula (=, +, -, @, tab, CR first; e.g. from a model file name) gets a leading apostrophe."""
    v = "" if v is None else str(v)
    return "'" + v if v[:1] in ("=", "+", "-", "@", "\t", "\r", "\uff20", "\uff1d", "\uff0b", "\uff0d") else v


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
        self.warnings = []                 # [{"line", "count", "t"}]
        self.warm = []                     # the launcher's "warm session:" lines (agent mode: saved / restored conversations)
        self.decisions = []                # what the launcher decided before starting llama-server (its "moe-cache: ..." lines up to the URL line)
        self.deciding = True
        self.bench = {"state": "idle"}
        self.auto_key = None               # the launcher's key (agent mode, no key typed on the page)

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
            kw = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "env": child_env(s.get("api_key"))}
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
        if s.get("agent") and not s.get("api_key"):
            self.auto_key = launcher_key()
        self.add_line("ui: " + shown_command(s) + (" (API key: the launcher's agent key)" if self.auto_key else ""))
        threading.Thread(target=self.read, args=(p,), daemon=True).start()
        ready.set()
        self.poll(p)
        p.wait()

    def key(self):
        """The key llama-server needs: the one typed on the page, else the launcher's agent key, else None."""
        return self.settings.get("api_key") or self.auto_key

    def key_info(self):
        k = self.key()
        if not k:
            return {"in_use": False}
        return {"in_use": True, "source": "typed on this page" if self.settings.get("api_key") else "the launcher's agent key (stable for your user)",
                "ends": k[-4:]}

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
            k = self.key()
            if k and k in line:
                line = line.replace(k, "KEY")                  # e.g. the launcher's agent snippets print it
            if line.startswith("moe-cache-stats "):
                try:
                    self.take_snapshot(json.loads(line[len("moe-cache-stats "):]))
                except ValueError:
                    pass
                continue
            self.add_line(line)
            if self.deciding and line.startswith("moe-cache: "):
                if "web UI and API will be at" in line:
                    self.deciding = False
                elif not WARN_RE.search(line):
                    self.decisions.append(line[len("moe-cache: "):])
            if "moe-cache: warm session:" in line:
                self.warm = (self.warm + [{"t": time.time(), "line": line.split("warm session:", 1)[1].strip()}])[-8:]
            if WARN_RE.search(line):
                self.add_warning(line)
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

    def add_warning(self, line):
        text = re.sub(r"^[\d.]+ [A-Z] ", "", line).strip()          # llama-server's time and level prefix
        for w in self.warnings:
            if w["line"] == text:
                w["count"] += 1; w["t"] = time.time()
                return
        self.warnings = (self.warnings + [{"line": text, "count": 1, "t": time.time()}])[-10:]

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
        api_key = self.key()
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

    # -- quick benchmark --
    def start_bench(self):
        if self.state != "ready":
            raise ValueError("start a model first")
        if self.bench.get("state") == "running":
            raise ValueError("a benchmark is already running")
        self.bench = {"state": "running", "step": "warm-up", "started": time.time()}
        threading.Thread(target=self.run_bench, args=(self.p,), daemon=True).start()

    def complete(self, prompt, n, ignore_eos=False):
        """One /completion as pc_speed.ps1 asks it: greedy, fixed seed, no prompt reuse, so stock and moe-cache runs compare. The decode runs also
        set ignore_eos: a model that finishes its answer early would otherwise time fewer tokens (Granite in CI stopped at 149 of 256)."""
        port = int(self.settings.get("port") or 8080)
        st, body = mc.http(f"http://127.0.0.1:{port}/completion", dict({"prompt": prompt, "n_predict": n, "temperature": 0, "seed": 42, "cache_prompt": False}, **({"ignore_eos": True} if ignore_eos else {})),
                           api_key=self.key(), timeout=1800)
        if st != 200:
            raise RuntimeError(f"HTTP {st}: {body[:200]}" if st else "the server did not answer")
        return json.loads(body)["timings"]

    def cache_counts(self):
        """(hits, misses) right now: asks the plugin for a fresh snapshot; None in stock mode."""
        if not self.pid or not ask_snapshot(self.pid):
            return None
        before = self.snap
        for _ in range(20):
            time.sleep(0.1)
            if self.snap is not before:
                break
        return (self.snap[1].get("hits", 0), self.snap[1].get("misses", 0)) if self.snap else None

    def run_bench(self, p):
        b = self.bench
        try:
            t0 = time.time()
            self.complete(BENCH_PROMPTS[0], 32)
            warm = time.time() - t0
            c0 = self.cache_counts()
            dec, ntok = [], []
            for i, q in enumerate(BENCH_PROMPTS[1:3]):
                b["step"] = f"decode {i + 1} of 2 (256 tokens)"
                t = self.complete(q, 256, ignore_eos=True)
                dec.append(t["predicted_per_second"]); ntok.append(t.get("predicted_n"))
            c1 = self.cache_counts()
            b["step"] = "prefill (about 3.5k tokens)"
            pf = self.complete(BENCH_LONG, 16)
            hr = None
            if c0 and c1 and (c1[0] - c0[0]) + (c1[1] - c0[1]) > 0:
                hr = (c1[0] - c0[0]) / ((c1[0] - c0[0]) + (c1[1] - c0[1]))
            s, sy = self.settings, system_info()
            row = {"time": time.strftime("%Y-%m-%d %H:%M"), "model": os.path.basename(s.get("model", "")), "size_gb": f"{size_of(s.get('model', '')) / 1e9:.1f}",
                   "mode": self.mode or "?", "ram_gb": s.get("ram") or "auto", "vram_gib": s.get("vram") or "auto", "ctx": s.get("ctx") or 8192,
                   "gpu": sy.get("gpu") or "none", "os": sy.get("os"), "decode_1": f"{dec[0]:.2f}", "decode_2": f"{dec[1]:.2f}",
                   "decode_mean": f"{sum(dec) / len(dec):.2f}", "decode_tokens": "/".join(str(n) for n in ntok), "prefill_tokens": pf.get("prompt_n"),
                   "prefill": f"{pf['prompt_per_second']:.1f}", "hit_rate": f"{hr:.3f}" if hr is not None else "", "warmup_s": f"{warm:.1f}",
                   "command": shown_command(s)}
            try:
                append_bench_row(row)
                b["csv"] = BENCH_CSV
            except OSError as e:
                b["csv_error"] = str(e)
            b.update(state="done", result=row, step="")
            self.add_line(f"ui: quick benchmark: decode {row['decode_1']} / {row['decode_2']} tok/s, prefill {row['prefill_tokens']} tokens at {row['prefill']} tok/s")
        except Exception as e:
            b.update(state="failed", error=str(e) if p.poll() is None else "the server stopped during the benchmark", step="")

    def status(self):
        s = self.settings
        port = int(s.get("port") or 8080)
        return {"state": self.state, "rc": self.rc, "model": s.get("model"), "port": port, "url": f"http://127.0.0.1:{port}",
                "uptime": time.time() - self.started if self.started and self.state in ("starting", "ready") else None,
                "mode": self.mode, "command": shown_command(s) if s.get("model") else "", "agent": bool(s.get("agent")), "ctx": s.get("ctx"),
                "live_tps": self.live_tps, "last": self.last, "cache": self.cache, "server_rss": self.rss, "log_seq": self.seq,
                "history": list(self.hist), "warnings": self.warnings, "decisions": self.decisions, "bench": self.bench, "api_key": self.key_info(), "warm": {"events": self.warm, "files": warm_files() if s.get("agent") else []},
                "system": system_info()}


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


# what the page may do: its own scripts and requests, llama-server's chat page in the frame, never be framed itself (clickjacking)
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
       "frame-src http://127.0.0.1:*; frame-ancestors 'none'; base-uri 'none'; form-action 'none'; object-src 'none'")
MAX_BODY = 64 * 1024


class Handler(BaseHTTPRequestHandler):
    store = None
    runner = None
    token = ""                  # random per run: the URL the launcher opens carries it once, the page keeps it in a cookie
    server_version = "moe-cache-ui"
    timeout = 30                # a client that sends nothing does not hold a thread forever

    def log_message(self, *a):
        pass

    def allowed(self, navigation=False):
        """Only pages from this server: the Host must be ours (no DNS rebinding), a foreign Origin is refused and the browser must say the request
        comes from this page (Sec-Fetch-Site same-origin, or 'none' for opening the page itself from the address bar or the launcher)."""
        port = self.server.server_address[1]
        ok = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in ok:
            return False
        o = self.headers.get("Origin")
        if o is not None and o not in {"http://" + h for h in ok}:
            return False
        site = self.headers.get("Sec-Fetch-Site")
        return site is None or site == "same-origin" or (navigation and site == "none")

    def cookie_name(self):
        return f"moe_cache_ui_{self.server.server_address[1]}"       # cookies are not per port: the name is

    def has_token(self):
        """The per-run token, from the page's cookie or an X-MoeCache-Token header (scripts, tests): other users of a shared machine can reach
        127.0.0.1 too, but not this token (it is printed for and opened by the user who started the page)."""
        t = self.headers.get("X-MoeCache-Token")
        if not t:
            for part in (self.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == self.cookie_name():
                    t = v
        return bool(t) and secrets.compare_digest(t, self.token)

    def end_headers(self):
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

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
        u = urllib.parse.urlparse(self.path)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query).items()}
        if not self.allowed(navigation=u.path in ("/", "/index.html")):
            return self.send(403, {"error": "forbidden"})
        if u.path in ("/", "/index.html"):
            if q.get("t") and secrets.compare_digest(q["t"], self.token):
                # the launcher's link: keep the token in a cookie only this page sends, and drop it from the address bar
                self.send_response(303)
                self.send_header("Set-Cookie", f"{self.cookie_name()}={self.token}; Path=/; HttpOnly; SameSite=Strict")
                self.send_header("Location", "/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if not self.has_token():
                return self.send(403, "<!doctype html><meta charset=utf-8><title>moe-cache</title><p style='font:15px system-ui;margin:2em'>Open the "
                                 "address the launcher printed (it ends in <code>?t=…</code>): this page only answers the user who started it.</p>",
                                 "text/html")
            return self.send(200, open(PAGE, "rb").read(), "text/html")
        if not self.has_token():
            return self.send(403, {"error": "forbidden: no or wrong page token (open the address the launcher printed)"})
        try:
            if u.path == "/api/system":
                return self.send(200, dict(system_info(), folders=self.store.folders(), last=self.store.saved.get("last", {}),
                                           llama_server=self.store.ls, plugin=os.environ.get("MOE_CACHE_LIB") or "found by the launcher (build folder or kit)"))
            if u.path == "/api/models":
                return self.send(200, {"folders": self.store.folders(), "models": self.store.list_models(q.get("fresh") == "1")})
            if u.path == "/api/plan":
                m = self.known(q.get("model"))
                ps = self.settings(dict(q, model=m))             # the same checks as Start (numbers in range, no nan / inf)
                q = {"ram": str(ps["ram"]) if "ram" in ps else "", "vram": str(ps["vram"]) if "vram" in ps else "", "ctx": str(ps.get("ctx", 8192)),
                     "mode": ps["mode"], "agent": "1" if ps["agent"] else ""}
                key = (m, q.get("ram"), q.get("vram"), q.get("ctx"), q.get("mode"), q.get("agent"))
                hit = self.store.plans.get(key)
                if not hit or time.time() - hit[0] > 60:          # free RAM and VRAM change: a plan is kept for a minute
                    hit = self.store.plans[key] = (time.time(), plan_model(m, q.get("ram") or None, q.get("vram") or None, int(q.get("ctx") or 8192), q.get("mode") or "exact",
                                                                             q.get("agent") in ("1", "true")))
                return self.send(200, hit[1])
            if u.path == "/api/status":
                _GPU["watched"] = time.time()
                return self.send(200, self.runner.status())
            if u.path == "/api/bench":
                return self.send(200, {"csv": BENCH_CSV, "rows": read_bench_rows()})
            if u.path == "/api/key":                       # for the Copy button only (token-protected, not cached)
                if self.runner.state not in ("starting", "ready") or not self.runner.key():
                    raise ValueError("no API key in use")
                return self.send(200, {"key": self.runner.key()})
            if u.path == "/api/log":
                since = int(q.get("since") or 0)
                return self.send(200, {"lines": [[n, l] for n, l in list(self.runner.log) if n > since]})
        except ValueError as e:
            return self.send(400, {"error": str(e)})
        except Exception as e:
            return self.send(500, {"error": f"{type(e).__name__}: {e}"})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.allowed() or not (self.headers.get("Content-Type") or "").startswith("application/json") or not self.has_token():
            return self.send(403, {"error": "forbidden"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if not 0 <= n <= MAX_BODY:
            self.close_connection = True
            return self.send(413, {"error": "request too large"})
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("a JSON object is expected")
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
            if self.path == "/api/command":
                s = self.settings(body)
                return self.send(200, {"launcher": shown_command(s), "lines": dry_run(s)})
            if self.path == "/api/bench":
                self.runner.start_bench()
                return self.send(200, self.runner.bench)
            if self.path == "/api/stop":
                threading.Thread(target=self.runner.stop, daemon=True).start()
                return self.send(200, {"ok": True})
            if self.path == "/api/connect":
                agent = body.get("agent")
                if agent not in ("opencode", "hermes"):
                    raise ValueError("agents: opencode, hermes")
                if self.runner.state != "ready":
                    raise ValueError("start a server first (with Agent mode on)")
                return self.send(200, connect_agent(agent, self.runner.status()["url"], self.runner.key() or "", bool(body.get("write"))))
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
        for k, lo, hi, conv, what in (("ram", 0.1, 65536, float, "RAM budget (GB)"), ("vram", 0.1, 4096, float, "VRAM (GiB)"),
                                      ("ctx", 512, 1048576, int, "context"), ("port", 1, 65535, int, "port")):
            x = number(d.get(k), lo, hi, conv, what)
            if x is not None:
                s[k] = x
        if s.get("api_key") and not re.fullmatch(r"[\x21-\x7e]{1,200}", s["api_key"]):
            raise ValueError("the API key may only contain visible ASCII characters")
        return s


def number(v, lo, hi, conv=float, what="value"):
    """A finite number in [lo, hi] (no nan / inf), or None for empty."""
    if v in (None, ""):
        return None
    try:
        x = conv(float(v)) if conv is int else conv(v)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{what}: not a number")
    if not (isinstance(x, (int, float)) and math.isfinite(x) and lo <= x <= hi):
        raise ValueError(f"{what} must be between {lo} and {hi}")
    return x


def connect_agent(agent, url, api_key, write):
    """`moe_cache.py connect AGENT --url URL [--write]` for the running server. The key goes in the environment (LLAMA_API_KEY), never on a
    command line. Returns {rc, output, file, written}: for OpenCode with write, the config file and the provider block now in it."""
    cmd = [mc.python_cmd(), os.path.join(HERE, "bin", "moe_cache.py"), "connect", agent, "--url", url] + (["--write"] if write else [])
    env = dict(os.environ, LLAMA_API_KEY=api_key, PYTHONIOENCODING="utf-8")
    if not api_key:
        env.pop("LLAMA_API_KEY", None)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    out = {"rc": r.returncode, "output": (r.stdout + r.stderr).strip(), "file": None, "written": None}
    m = re.search(r"; config (.+?\.json)", out["output"])
    if write and r.returncode == 0 and m:
        out["file"] = m.group(1)
        try:
            data = json.load(open(m.group(1), encoding="utf-8"))
            block = {"model": data.get("model"), "provider": {"moe-cache": data.get("provider", {}).get("moe-cache")}}
            if api_key and block["provider"]["moe-cache"]:
                block["provider"]["moe-cache"].get("options", {})["apiKey"] = "(your key)"
            out["written"] = json.dumps(block, indent=2)
        except Exception as e:
            out["written"] = f"(could not read it back: {e})"
    if api_key:
        out["output"] = out["output"].replace(api_key, "(your key)")
    return out


def make_server(port=8070, ls=None, token=None):
    store = Store(ls or os.environ.get("LLAMA_SERVER", "llama-server"))
    h = type("H", (Handler,), {"store": store, "runner": Runner(), "token": token or secrets.token_urlsafe(24)})
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
        threading.Thread(target=watch_windows_gpu, args=(a.llama_server,), daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/?t={srv.RequestHandlerClass.token}"
    mc.say(f"moe-cache: control page at {url} (this machine and this user only; Ctrl-C stops it and any server it started)")
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
