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
import argparse, collections, csv, ctypes, glob, http.client, json, math, secrets, os, re, signal, subprocess, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_cache as mc  # noqa: E402
import moe_cache_pick as picker  # noqa: E402

HERE = mc.HERE
WINDOWS = mc.WINDOWS
PAGE = os.path.join(HERE, "ui", "index.html")
SETTINGS_FILE = os.path.join(mc.CACHE_DIR, "ui.json")
LAUNCHER = [mc.python_cmd(), os.path.join(HERE, "bin", "moe_cache.py"), "server"]     # tests replace it with a fake server
HEADROOM = mc.STOCK_HEADROOM   # shown in the reasons; the rule itself is mc.decide()
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
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5)
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
        p += ["--ram-gib", mc.ram_gib(ram)]       # ram is GB, the planner takes GiB
    if vram:
        p += ["--vram-gib", str(vram)]
    r = subprocess.run(p, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    try:
        return json.loads(r.stdout)
    except ValueError:
        raise RuntimeError((r.stderr.strip().splitlines() or ["planner failed"])[-1])


def run_tuner(path, ctx, vram, kv_q8):
    sys.path.insert(0, os.path.join(HERE, "planner"))
    from gguf_plan import tune
    return tune(path, ctx, free=float(vram) * GIB if vram else None, want_cache=False, kv_q8=kv_q8)


_MTP_SERVER = {}


def server_has_mtp(ls):
    """True when this llama-server offers MTP speculative decoding (draft-mtp in its --help), as the launcher checks (moe_cache.mtp_args);
    asked once per llama-server."""
    if ls not in _MTP_SERVER:
        try:
            _MTP_SERVER[ls] = "draft-mtp" in subprocess.run([ls, "--help"], capture_output=True, text=True, timeout=30).stdout
        except Exception:
            _MTP_SERVER[ls] = False
    return _MTP_SERVER[ls]


def mtp_reason(path, want, agent, ls):
    """The launcher's MTP line for this model and setting (its own mtp_decide(); the head from gguf_plan.mtp_layers), without the prefix; None
    when the launcher would say nothing. want: None (auto), True (--mtp), False (--no-mtp)."""
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import mtp_layers
        head, declared = mtp_layers(path)
    except Exception:
        return None
    try:
        on, why = mc.mtp_decide(want, agent, head, declared, bool(head and want is not False and ls and server_has_mtp(ls)), False)
    except SystemExit as e:                     # --mtp the launcher would refuse: say so before Start
        return f"MTP: Start would stop with: {e}"
    return why.replace("moe-cache: ", "", 1) if why else None


# ---- servers started elsewhere (GUI1 step 4 #6, status only): a llama-server on this machine that the page did not start ----
PEEK_PORTS = range(8080, 8090)


def on_demand_port():
    """The port of this user's on-demand listener (from its pid file in the private cache folder) while that process lives, else None."""
    try:
        pid, port = (int(x) for x in open(os.path.join(mc.CACHE_DIR, "on-demand.pid")).read().split()[:2])
    except (OSError, ValueError):
        return None
    return port if alive(pid) else None


def peek_server(port, api_key=None, trusted=False):
    """What a llama-server on 127.0.0.1:port says about itself: {port, up, health, model, ctx, slots, busy, key}. Only this machine (no other host:
    the page must not become a way to probe the network); the key itself is never returned. key: "needed" (the server wants one; the stored key
    or `api_key` was used only when `trusted`), "rejected" (401 even with it), or null. The stored key goes only to a `trusted` port (the page's own
    server, this user's on-demand listener): any local user can listen on a free port of 8080-8089 and answer 401 to collect it (audit 2026-10-29 U1)."""
    url = f"http://127.0.0.1:{int(port)}"
    st = mc.http(url + "/health", timeout=1)[0]          # llama-server answers /health without a key; 503 while it loads the model
    if st not in (200, 503):
        return {"port": port, "up": False}
    if not trusted and not api_key:
        if mc.http(url + "/v1/models", timeout=2)[0] in (401, 403):
            return {"port": port, "up": True, "health": "ready" if st == 200 else "loading", "model": None, "ctx": None, "slots": None, "busy": None,
                    "key": "needed"}
    key = mc.key_for(url, api_key)
    name, ctx = mc.server_info(url, key)
    out = {"port": port, "up": True, "health": "ready" if st == 200 else "loading", "model": name, "ctx": ctx, "slots": None, "busy": None,
           "key": ("rejected" if not name and mc.http(url + "/v1/models", api_key=key, timeout=2)[0] in (401, 403) else "needed") if key else None}
    sst, body = mc.http(url + "/slots", api_key=key, timeout=2)
    if sst == 200:
        try:
            sl = json.loads(body)
            out.update(slots=len(sl), busy=sum(1 for x in sl if x.get("is_processing")))
        except (ValueError, AttributeError, TypeError):
            pass
    return out


def peek_servers(ports=None, own=None):
    """peek_server() for the usual ports at once (closed ports answer at once); `own` = the port of the server this page started."""
    res, ports = {}, list(ports or PEEK_PORTS)
    mine = {own, on_demand_port()} - {None}
    ts = [threading.Thread(target=lambda p=p: res.__setitem__(p, peek_server(p, trusted=p in mine)), daemon=True) for p in ports]
    for t in ts:
        t.start()
    for t in ts:
        t.join(15)
    return [dict(res[p], started_here=p == own) for p in ports if p in res and res[p].get("up")]


# Measured decode speeds from README "Results" / docs/RESULTS.md, by model family (gguf_plan.model_key) and file size (GB, so each quant has its own
# row): (machine, RAM GB, setting, stock tok/s or None, moe-cache tok/s, where). Machines: laptop = i7-13620H, 31 GB RAM, RTX 4060 8 GB, Linux;
# PC = Ryzen 9 7950X, 32 GB RAM, RX 9070 XT 16 GB, Windows (cold, ABBA, launcher defaults, RESULTS 59-60).
LAPTOP, PC = "laptop", "PC"
MEASURED = {
    ("qwen35moe-41x256", 21.7): [(LAPTOP, 24, "fits in RAM (warm)", 42.9, 36.6, "RESULTS 44: the launcher runs stock here"),
                                 (LAPTOP, 12, "8 GB GPU", 6.9, 22.6, "RESULTS 44"), (LAPTOP, 12, "no GPU", 3.1, 9.9, "RESULTS 5"),
                                 (PC, 32, "fits in RAM, agent context", 43.6, 43.1, "RESULTS 60; about 72 with --mtp: the launcher runs stock here")],
    ("qwen4exp-48x512", 68.0): [(LAPTOP, 24, "8 GB GPU", 5.2, 12.5, "RESULTS 44; 17.6 with --hybrid (approximate), RESULTS 31"),
                                (PC, 32, "16 GB GPU", 14.89, 21.94, "RESULTS 59; agent sessions 2.0-2.4x faster, RESULTS 60"),
                                (PC, 16, "16 GB GPU, --ram 16", None, 20.03, "RESULTS 60: 92 % of the 24 GB speed")],
    ("qwen4exp-48x512", 66.4): [(PC, 32, "16 GB GPU", 18.41, 22.16, "RESULTS 63 (Q2_0, the recommended file): prompts 2.33x stock")],
    ("qwen4exp-48x512", 58.4): [(PC, 32, "16 GB GPU, coding agent", None, 21.24, "RESULTS 63 (Coder IQ1_M): agent session 24 % shorter than IQ2_XS")],
    ("qwen4exp-48x512", 76.0): [(PC, 32, "16 GB GPU", 15.14, 19.86, "RESULTS 59")],
    ("qwen4exp-48x512", 83.6): [(PC, 32, "16 GB GPU", 12.92, 17.28, "RESULTS 59")],
    ("qwen35moe-48x256", 36.6): [(LAPTOP, 24, "8 GB GPU", 1.89, 7.7, "RESULTS 45; 9.0 with --fast"),
                                 (PC, 32, "16 GB GPU", 15.95, 19.73, "RESULTS 61 (re-measured; RESULTS 59 had 13.88 / 16.13)")],
    ("qwen35moe-48x256", 44.7): [(LAPTOP, 24, "8 GB GPU", 1.10, 5.7, "RESULTS 45; 6.8 with --fast")],
    ("qwen35moe-48x256", 60.2): [(LAPTOP, 24, "8 GB GPU", 0.91, 3.7, "RESULTS 45; 4.5 with --fast")],
    ("glm5-next-46x288", 101.8): [(LAPTOP, 24, "8 GB GPU", 0.32, 1.75, "RESULTS 26; 2.1 with --fast"),
                                  (PC, 32, "16 GB GPU", 2.21, 4.44, "RESULTS 59; 6.92 with --mtp, same text, RESULTS 60")],
}


def measured(path, size):
    """Lines like 'laptop, 12 GB RAM, 8 GB GPU: stock 6.9, moe-cache 22.6 tok/s (RESULTS 44)' for a model measured before; [] otherwise."""
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import model_key
        key = model_key(path)
    except Exception:
        return []
    # the closest size within 3 % (Flash-Next Q2_0 66.4 GB and IQ2_XS 68.0 GB are both within 3 % of each other)
    near = [(abs(size / 1e9 - gbs), rows) for (k, gbs), rows in MEASURED.items() if k == key and abs(size / 1e9 - gbs) <= 0.03 * gbs]
    if not near:
        return []
    rows = min(near, key=lambda x: x[0])[1]
    return [f"{where}, {ram} GB RAM, {what}: " + (f"stock {st} tok/s, " if st is not None else "") + f"moe-cache {mc_} tok/s ({src})"
            for where, ram, what, st, mc_, src in rows]


def summary(out, mode="exact", engine="auto"):
    """One plain sentence: why this mode (I33). Built from the verdict and the numbers plan_model() already has."""
    kind, size, avail, n = out.get("kind"), out.get("size") or 0, out.get("avail"), out.get("numbers") or {}
    room = f"the {gb(avail)} this server may use" if avail else "the memory this server may use"
    if kind == "error":
        return "The planner could not read this model, so the page cannot say how it would run."
    if kind == "gpu":
        return "Every layer of experts fits in the graphics card's memory, so the cache has nothing to do."
    if kind == "stock" and engine == "stock" and avail and size + HEADROOM > avail:
        return (f"Stock llama-server because you chose Run as Stock: the model ({gb(size)}) is bigger than {room}, "
                "so moe-cache is usually faster here. Use it to compare.")
    if kind == "stock":
        return (f"Stock llama-server: the model ({gb(size)}) fits in {room}, and when everything fits the plain llama-server is fastest. "
                "The plugin only helps models bigger than RAM.")
    if kind == "cache":
        x = f"{size / avail:.1f}x" if avail and size > avail else "more than"
        cpu, cache = n.get("cpu_experts_GiB"), n.get("cache_GiB")
        held = f" keeps {min(100, 100 * cache / cpu):.0f} % of the experts that run on the CPU in RAM and" if cpu and cache is not None else ""
        if avail and size + HEADROOM <= avail:
            return f"moe-cache, as you chose: the model ({gb(size)}) fits in {room}, so stock would likely be faster."
        tail = {"fast": " --fast skips a few rarely used experts that are not in RAM (a little faster, some tokens differ).",
                "hybrid": " --hybrid also runs the most used experts on the GPU (experimental, answers differ)."}.get(mode, " The answers are the same as stock's.")
        return (f"moe-cache: the model ({gb(size)}) is {x} {room}. Stock llama-server would keep re-reading it from the disk; "
                f"the cache{held} reads the rest only when needed." + tail)
    return ""


def measured_here(path, rows=None):
    """The Quick benchmark's stock vs moe-cache pairs for this model file on this computer (bench.csv, newest first, at most 3), as lines."""
    name = os.path.basename(path)
    rows = bench_compare(read_bench_rows(500) if rows is None else rows)
    out = []
    for r in sorted(rows, key=lambda r: r.get("moe_time") or "", reverse=True):
        if r["model"] != name:
            continue
        x = f" ({r['decode_x']:.2f}x)" if r.get("decode_x") else ""
        out.append(f"{r['moe_time'] or '?'}, RAM {r['ram_gb']}, {r['start']} start: stock {r['stock_decode']} tok/s, {r['mode']} {r['moe_decode']} tok/s{x}")
    return out[:3]


def gb(x):
    return f"{x / 1e9:.1f} GB"


def windows_amd_rules(gpu, path, agent=False, windows=WINDOWS):
    """The launcher's Windows + AMD rules that apply to this model, as [reasons in the launcher's words] (today: context checkpoints off for some
    models; the old "all experts on the CPU when the model is bigger than RAM" rule was dropped, PC test 2026-10-27: GPU expert layers are faster).
    gpu: {device, adapter} as windows_gpu_once finds it. Same conditions as moe_cache.server(); the checkpoint rule is the launcher's own function."""
    if not (windows and gpu and str(gpu.get("device", "")).startswith("Vulkan") and re.search(r"AMD|Radeon", gpu.get("adapter", ""))):
        return []
    reasons = []
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
    return reasons


def plan_model(path, ram=None, vram=None, ctx=8192, mode="exact", agent=False, engine="auto", mtp=None, ls=None):
    """Verdict, mode and the reasons for one model under these settings. ram in GB, vram in GiB (the launcher's units).
    engine: auto (the launcher decides), plugin (--always-plugin) or stock (--stock). mtp: None (auto), True, False; ls: llama-server for its MTP check."""
    if engine == "stock":
        mode = "exact"                             # --fast and --hybrid need the plugin
    stock_fits, size, avail = mc.decide(path, ram)     # the launcher's own stock-or-plugin rule: does it fit?
    stock = mc.decide(path, ram, plugin_wanted=engine == "plugin" or mode in ("fast", "hybrid"), stock_wanted=engine == "stock")[0]   # what it will run
    size = size or 0
    gpu = _GPU["gpu"] if WINDOWS else None
    if vram is None and gpu and gpu.get("free") and not gpu.get("integrated"):
        vram = f"{gpu['free'] / GIB:.2f}"          # as the launcher: on Windows without nvidia-smi it passes the free VRAM it found to the planner
    amd_reasons = windows_amd_rules(gpu, path, agent, windows=WINDOWS)
    out = {"path": path, "size": size, "avail": avail, "reasons": [], "numbers": {}, "measured": measured(path, size), "here": measured_here(path)}
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
        out.update(kind="stock" if stock else "error", verdict=("stock llama-server" + ("" if stock_fits else ", as asked (--stock)")) if stock else "cannot plan this model")
        R.append(f"planner: {e}" + (" (not a MoE model: moe-cache has nothing to manage; llama-server runs it stock)" if "MoE" in str(e) else ""))
        out["summary"] = ("Stock llama-server: this is not a MoE model, so moe-cache has nothing to manage." if "MoE" in str(e) and stock
                          else summary(out, mode, engine))
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
    if stock and not stock_fits:
        out.update(kind="stock", verdict="does not fit: stock llama-server, as asked (to compare)")
        R.append("--stock: no plugin; experts that are not in the file cache are read from the SSD by the OS (usually much slower than moe-cache "
                 "when the model is bigger than RAM; run the Quick benchmark in both to see it)")
    elif stock:
        out.update(kind="stock", verdict="fits in RAM: stock llama-server (fastest)")
        R.append("when everything fits, stock llama-server is faster: the page cache holds the whole file (Qwen3.6-35B: 42.9 vs 36.6 tok/s, RESULTS 44)")
    elif ncmoe == 0:
        out.update(kind="gpu", verdict="all experts fit on the GPU")
        R.append("every layer of experts fits in VRAM next to the rest of the model: the cache has nothing to manage")
    else:
        name = {"fast": "moe-cache --fast", "hybrid": "moe-cache --hybrid (experimental)"}.get(mode, "moe-cache exact")
        why = "moe-cache was asked (--always-plugin)" if mode == "exact" else mode + " needs the plugin"
        out.update(kind="cache", verdict=("fits in RAM, but " + why + ": " if stock_fits else "does not fit: ") + name)
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
        if not no_vram:
            R.append("tuner: " + w)
    if plan.get("note"):
        R.append("planner: " + plan["note"])
    m = mtp_reason(path, mtp, agent, ls)
    if m:
        R.append(m)
    out["summary"] = summary(out, mode, engine)
    return out


def server_args(s):
    """moe_cache.py server arguments for the page's settings."""
    a = [s["model"], "--port", str(int(s.get("port") or 8080)), "--ctx", str(int(s.get("ctx") or (65536 if s.get("agent") else 8192)))]
    if s.get("ram"):
        a += ["--ram", str(float(s["ram"]))]
    if s.get("vram"):
        a += ["--vram", str(float(s["vram"]))]
    if s.get("engine") == "stock":
        a.append("--stock")                # to compare: no plugin whatever the size (the mode is exact then)
    elif s.get("mode") == "fast":
        a.append("--fast")
    elif s.get("mode") == "hybrid":
        a.append("--hybrid")
    elif s.get("engine") == "plugin":
        a.append("--always-plugin")
    if s.get("mtp") in (True, False):
        a.append("--mtp" if s["mtp"] else "--no-mtp")
    if s.get("agent"):
        a.append("--agent")
        if s.get("no_warm"):
            a.append("--no-warm")          # neither restores nor saves agent conversations
    if s.get("llama_server"):
        a += ["--llama-server", s["llama_server"]]
    return a + ["--", "--metrics"]          # the API key goes to the launcher in LLAMA_API_KEY (child_env), never on a command line


def shown_command(s):
    return (("LLAMA_API_KEY=KEY " if s.get("api_key") else "")
            + " ".join(('"' + x + '"') if " " in x else x for x in ["moe-cache-server"] + server_args(s)))


def dry_run(s):
    a = server_args(s)
    a.insert(a.index("--"), "--dry-run")          # a launcher option: after the -- it would go to llama-server
    r = subprocess.run(LAUNCHER + a, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, env=child_env(s.get("api_key")))
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
        r = subprocess.run(KEY_CMD, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
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
    # the launcher also runs short helpers first (planner/plan.py, `llama-server --help`): take the child that serves (--port on its command line)
    try:
        for t in os.listdir(f"/proc/{pid}/task"):
            for k in open(f"/proc/{pid}/task/{t}/children").read().split():
                try:
                    if b"\0--port\0" in open(f"/proc/{k}/cmdline", "rb").read():
                        return int(k)
                except OSError:
                    pass
    except (OSError, ValueError):
        pass
    return None


def alive(pid):
    if not pid:
        return False
    if WINDOWS:
        try:
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x1000, False, pid)
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code))
            k.CloseHandle(h)
            return bool(ok) and code.value == 259          # STILL_ACTIVE
        except Exception:
            return False
    return os.path.isdir(f"/proc/{pid}") or _kill0(pid)


def _kill0(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def catches_usr1(pid, sig=None):
    """Linux: True when the process handles SIGUSR1 (or sig; the plugin installs the handler; stock llama-server would die of it)."""
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("SigCgt:"):
                return bool(int(line.split()[1], 16) >> ((sig or signal.SIGUSR1) - 1) & 1)
    except (OSError, ValueError, AttributeError):
        pass
    return False


def ask_idle_release(pid):
    """Ask the plugin in llama-server `pid` to give its cache's RAM back while the server idles (N3); False when it cannot be asked (stock)."""
    if WINDOWS:
        try:
            k = ctypes.windll.kernel32
            h = k.OpenEventW(0x0002, False, f"Local\\moe-cache-idle-{pid}")
            if not h:
                return False
            k.SetEvent(h); k.CloseHandle(h)
            return True
        except Exception:
            return False
    if catches_usr1(pid, signal.SIGUSR2):
        try:
            os.kill(pid, signal.SIGUSR2)
            return True
        except OSError:
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
# llama-server's own warnings that only repeat what the plan already says (a CPU-only machine: "no usable GPU found, --gpu-layers ... ignored")
NOT_WARN_RE = re.compile(r"(?i)no usable GPU found")


def is_warning(line):
    return bool(WARN_RE.search(line)) and not NOT_WARN_RE.search(line)
# ---- agent sessions: the conversations the launcher's warm keeper saved (moe_cache.warm_keeper) ----
# One file per slot: slots/agent/<model alias>-c<ctx>[-<K type>-<V type>]-s<slot>.bin, written by llama-server's slot save (state file: 'ggsq' magic,
# version, token count, tokens, KV state) with the context checkpoints appended to the same file. The launcher restores the two largest files of
# the same model and settings at start; a half-written save is <name>.new.bin.
SESSION_RE = re.compile(r"^(?P<tag>.+)-s(?P<slot>\d+)\.bin$")
TAG_RE = re.compile(r"^(?P<model>.+?)-c(?P<ctx>\d+)(?:-(?P<kv>.+))?$")
AGENT_WORDS = [("opencode", "OpenCode"), ("hermes", "Hermes"), ("claude code", "Claude Code"), ("aider", "Aider"), ("cline", "Cline"),
               ("continue.dev", "Continue"), ("roo code", "Roo Code"), ("goose", "Goose")]


def slot_dir():
    return os.path.join(mc.CACHE_DIR, "slots", "agent")


def session_tag(s):
    """The tag the launcher gives this server's saved conversations (moe_cache.server: alias, context, KV types; --fast stores the KV as q8_0)."""
    alias = re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", os.path.basename(s.get("model") or ""))
    ctx = int(s.get("ctx") or (65536 if s.get("agent") else 8192))
    kv = "-q8_0-q8_0" if s.get("mode") == "fast" else ""
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"{alias}-c{ctx}{kv}").rstrip("-_")


def session_tokens(path, n_ids=0):
    """(token count, the first n_ids token ids) from a slot file's header; (None, []) when it is not one."""
    import struct
    try:
        with open(path, "rb") as f:
            magic, _, n = struct.unpack("<III", f.read(12))
            if magic != 0x67677371:                    # 'ggsq' (LLAMA_FILE_MAGIC_GGSQ)
                return None, []
            data = f.read(4 * min(n, n_ids))
            k = len(data) // 4                      # a truncated file gives what it has
            ids = list(struct.unpack(f"<{k}i", data[:4 * k])) if k else []
            return n, ids
    except (OSError, struct.error):
        return None, []


def guess_agent(text):
    t = (text or "").lower()
    for w, name in AGENT_WORDS:
        if w in t:
            return name
    return None


def warm_files():
    """Saved agent conversations (newest first), with what the file names and headers say."""
    out = []
    for f in glob.glob(os.path.join(slot_dir(), "*-s*.bin")):
        name = os.path.basename(f)
        m = SESSION_RE.match(name)
        if not m or name.endswith(".new.bin"):
            continue
        t = TAG_RE.match(m.group("tag"))
        try:
            st = os.stat(f)
        except OSError:
            continue
        out.append({"name": name, "size": st.st_size, "t": st.st_mtime, "slot": int(m.group("slot")), "tag": m.group("tag"),
                    "model": t.group("model") if t else m.group("tag"), "ctx": int(t.group("ctx")) if t else None,
                    "kv": t.group("kv") if t and t.group("kv") else "f16", "tokens": session_tokens(f)[0]})
    return sorted(out, key=lambda x: -x["t"])


def move_to_trash(path):
    """Move a file to the platform's Trash / Recycle Bin and return where it went; raises when that is not possible (never a hard delete)."""
    path = os.path.abspath(path)
    if WINDOWS:
        from ctypes import wintypes
        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                        ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL), ("hNameMappings", ctypes.c_void_p),
                        ("lpszProgressTitle", wintypes.LPCWSTR)]
        # FO_DELETE with FOF_ALLOWUNDO = Recycle Bin; FOF_WANTNUKEWARNING asks the user before Windows would delete for good (e.g. bin too small)
        buf = ctypes.create_unicode_buffer(path + "\0")           # a list of paths ends with two NULs
        op = SHFILEOPSTRUCTW(None, 3, ctypes.cast(buf, wintypes.LPCWSTR), None, 0x40 | 0x10 | 0x4 | 0x400 | 0x4000, False, None, None)
        rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        if rc != 0 or op.fAnyOperationsAborted or os.path.exists(path):
            raise OSError(f"the Recycle Bin did not take it (code {rc})")
        return "the Recycle Bin"
    gio = shutil_which("gio")
    if gio:
        r = subprocess.run([gio, "trash", path], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if r.returncode == 0 and not os.path.exists(path):
            return "the Trash"
    # the freedesktop.org Trash in the user's home (same file system as ~/.cache in the usual set-up): info file first, then a rename
    if sys.platform == "darwin":
        raise OSError("moving to the Trash is not supported on this system yet")
    import datetime, errno
    trash = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share"), "Trash")
    os.makedirs(os.path.join(trash, "files"), exist_ok=True); os.makedirs(os.path.join(trash, "info"), exist_ok=True)
    base, n = os.path.basename(path), 0
    while True:
        name = base if n == 0 else f"{base}.{n}"
        info = os.path.join(trash, "info", name + ".trashinfo")
        try:
            fd = os.open(info, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            n += 1
    with os.fdopen(fd, "w") as f:
        f.write(f"[Trash Info]\nPath={urllib.parse.quote(path)}\nDeletionDate={datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}\n")
    try:
        os.rename(path, os.path.join(trash, "files", name))
    except OSError as e:
        os.unlink(info)
        if e.errno == errno.EXDEV:
            raise OSError("the Trash is on another drive: move or delete the file yourself") from e
        raise
    return "the Trash"


def shutil_which(prog):
    import shutil
    return shutil.which(prog)


# ---- quick benchmark: the method of windows-kit/pc_speed.ps1 (RESULTS 54/55), for the running server ----
BENCH_CSV = os.path.join(mc.CACHE_DIR, "bench.csv")
BENCH_PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
                 "Explain how a CPU cache works, with an example of a cache miss.",
                 "Describe the water cycle step by step for a ten-year-old."]
BENCH_PARA = ("The history of computing spans centuries, from mechanical calculators and punched cards to vacuum tubes, transistors, integrated "
              "circuits and the microprocessors in every phone. ")
BENCH_LONG = "Summarise the following text in five bullet points.\n" + BENCH_PARA * 95          # about 3.5k tokens
BENCH_FIELDS = ["time", "model", "size_gb", "mode", "ram_gb", "vram_gib", "ctx", "gpu", "os", "decode_1", "decode_2", "decode_mean", "decode_tokens",
                "prefill_tokens", "prefill", "hit_rate", "warmup_s", "command", "start", "load_s", "flush"]


def read_bench_rows(limit=20):
    try:
        with open(BENCH_CSV, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))[-limit:]
    except OSError:
        return []


def bench_compare(rows):
    """Stock and moe-cache runs of the same model side by side: for each (model, RAM, context, warm/cold, GPU, OS) with both kinds, the newest
    stock run and the newest run of each moe-cache mode, and how many times faster moe-cache was (decode and prefill)."""
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    groups = {}
    for r in rows:
        mode = str(r.get("mode", ""))
        kind = "stock" if mode.startswith("stock") else mode if mode.startswith("moe-cache") else None
        if kind and num(r.get("decode_mean")):
            key = tuple(r.get(k) or "" for k in ("model", "ram_gb", "ctx", "start", "gpu", "os")) + (mode.endswith("+ MTP"),)   # MTP runs pair with MTP runs
            groups.setdefault(key, {})[kind] = r                 # rows come oldest first: the newest of each kind wins
    out = []
    for key, g in groups.items():
        st = g.pop("stock", None)
        for mode, r in sorted(g.items()):
            if not st:
                continue
            ratio = lambda k: round(num(r.get(k)) / num(st.get(k)), 2) if num(r.get(k)) and num(st.get(k)) else None
            out.append({"model": key[0], "ram_gb": key[1], "ctx": key[2], "start": key[3] or "warm", "gpu": key[4], "mode": mode,
                        "stock_decode": st.get("decode_mean"), "moe_decode": r.get("decode_mean"), "decode_x": ratio("decode_mean"),
                        "stock_prefill": st.get("prefill"), "moe_prefill": r.get("prefill"), "prefill_x": ratio("prefill"),
                        "hit_rate": r.get("hit_rate"), "stock_time": st.get("time"), "moe_time": r.get("time")})
    return out


def append_bench_row(row):
    os.makedirs(os.path.dirname(BENCH_CSV), exist_ok=True)
    migrate_bench_csv()
    new = not os.path.isfile(BENCH_CSV)
    with open(BENCH_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BENCH_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: csv_safe(v) for k, v in row.items()})


def migrate_bench_csv():
    """An older bench.csv (fewer columns) gets the current header; its rows are kept (old file saved as bench.csv.bak first)."""
    try:
        with open(BENCH_CSV, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            head = next(rd, None)
    except OSError:
        return
    if head is None or head == BENCH_FIELDS:
        return
    import shutil
    shutil.copy2(BENCH_CSV, BENCH_CSV + ".bak")
    with open(BENCH_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(BENCH_CSV + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BENCH_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict({k: "" for k in BENCH_FIELDS}, **{k: v for k, v in r.items() if k in BENCH_FIELDS}, start=r.get("start") or "warm"))
    os.replace(BENCH_CSV + ".tmp", BENCH_CSV)


# ---- cold start for a fair benchmark: the model out of the OS file cache before the server starts again ----
# Stock llama-server reads the model through the file cache (a warm cache from an earlier run makes it look faster than a first start); the
# plugin reads experts unbuffered. Linux: posix_fadvise(DONTNEED) drops a file's clean cached pages without root (pages another process maps
# stay, so it refuses then). Windows: tools/win_flush_cache.ps1 only READS other big files until the cache no longer holds the model.
WIN_FLUSH = os.path.join(HERE, "tools", "win_flush_cache.ps1")


def cold_start_support():
    if WINDOWS:
        ps = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
        ok = os.path.isfile(WIN_FLUSH) and os.path.isfile(ps)
        return {"available": ok, "method": "reads other big files (tools\\win_flush_cache.ps1) so Windows' file cache drops the model; nothing is written",
                "needs_other": True, "why": "" if ok else "tools\\win_flush_cache.ps1 or PowerShell not found"}
    if hasattr(os, "posix_fadvise") and sys.platform.startswith("linux"):
        return {"available": True, "method": "drops the model's pages from the page cache (posix_fadvise, no root needed)", "needs_other": False, "why": ""}
    return {"available": False, "method": "", "needs_other": False, "why": "not available on this system"}


def mapped_by(path):
    """PIDs (of this user's processes) that map this file now; Linux only, [] elsewhere."""
    out, real = [], os.path.realpath(path)
    for d in glob.glob("/proc/[0-9]*/maps"):
        try:
            with open(d, encoding="utf-8", errors="replace") as f:
                if any(line.rstrip("\n").endswith(" " + real) or line.rstrip("\n").endswith("\t" + real) for line in f):
                    out.append(int(d.split("/")[2]))
        except OSError:
            pass
    return out


def cached_fraction(path):
    """Share of the file's pages in the page cache (Linux mincore); None when it cannot be measured."""
    try:
        import mmap
        libc = ctypes.CDLL(None, use_errno=True)
        size = os.path.getsize(path)
        if not size:
            return 0.0
        page = mmap.PAGESIZE
        with open(path, "rb") as f:
            m = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_COPY)       # private: nothing can reach the file
            try:
                addr = ctypes.addressof(ctypes.c_char.from_buffer(m))
                n = (size + page - 1) // page
                vec = (ctypes.c_ubyte * n)()
                if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
                    return None
                return sum(v & 1 for v in vec) / n
            finally:
                del addr
                m.close()
    except Exception:
        return None


def flush_model_cache(model, other):
    """Push the model (all its parts) out of the OS file cache. Returns (ok, one line on what happened)."""
    parts = mc.model_parts(model)
    if WINDOWS:
        ps = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
        folders = [o for o in other if o and os.path.exists(o)]
        if not folders:
            return False, "no folder with other big files to read (set one under Cold start)"
        r = subprocess.run([ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", WIN_FLUSH, "-Target", "|".join(parts), "-Other", "|".join(folders)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
        out = " ".join(l.strip() for l in (r.stdout + r.stderr).splitlines() if l.strip())
        return r.returncode == 0, out or f"win_flush_cache.ps1 exit code {r.returncode}"
    users = sorted({pid for p in parts for pid in mapped_by(p)})
    if users:
        return False, f"another program uses this model file right now (PID {', '.join(map(str, users))}): not flushed, so it is not slowed down"
    before = [cached_fraction(p) for p in parts]
    for p in parts:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    after = [cached_fraction(p) for p in parts]
    def pct(xs):
        xs = [x for x in xs if x is not None]
        return f"{100 * sum(xs) / len(xs):.0f} %" if xs else "?"
    left = [x for x in after if x is not None]
    if left and sum(left) / len(left) > 0.5:      # e.g. a file on tmpfs, which lives in the cache
        return False, f"the file system kept the model in memory (cached before {pct(before)}, after {pct(after)})"
    return True, f"dropped the model from the page cache (cached before {pct(before)}, after {pct(after)})"


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
        self.mtp = False                   # the launcher said "MTP on:"
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
        self.agent_guess = {}              # (file, mtime) -> agent name guessed from the conversation's start

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
        ready.set()                 # the launcher runs: start() may answer now (the key lookup below can take as long as start() waits)
        if s.get("agent") and not s.get("api_key"):
            self.auto_key = launcher_key()
        self.add_line("ui: " + shown_command(s) + (" (API key: the launcher's agent key)" if self.auto_key else ""))
        threading.Thread(target=self.read, args=(p,), daemon=True).start()
        self.poll(p)
        p.wait()

    def sessions(self):
        """The saved agent conversations, which of them this server restored at start, a guess of the agent, and the server's live slots."""
        files = warm_files()
        s = self.settings
        running = self.state == "ready" and s.get("agent")
        tag = session_tag(s) if s.get("model") else None
        restored_slots = {int(m.group(1)) for e in self.warm for m in [re.search(r"restored .* into slot (\d+)", e["line"])] if m}
        failed = {m.group(1) for e in self.warm for m in [re.search(r"could not restore (\S+)", e["line"])] if m}
        port = int(s.get("port") or 8080)
        for f in files:
            f["this_server"] = bool(tag) and f["tag"] == tag
            f["restored"] = f["this_server"] and self.state in ("starting", "ready") and f["slot"] in restored_slots
            f["restore_failed"] = f["name"] in failed
            f["agent"] = None
            if running and f["this_server"]:
                ck = (f["name"], f["t"])
                if ck not in self.agent_guess:
                    _, ids = session_tokens(os.path.join(slot_dir(), f["name"]), 600)
                    st, body = mc.http(f"http://127.0.0.1:{port}/detokenize", {"tokens": ids}, api_key=self.key(), timeout=10) if ids else (0, "")
                    try:
                        self.agent_guess[ck] = guess_agent(json.loads(body).get("content")) if st == 200 else None
                    except ValueError:
                        self.agent_guess[ck] = None
                f["agent"] = self.agent_guess[ck]
        live = None
        if self.state == "ready":
            st, body = mc.http(f"http://127.0.0.1:{port}/slots", api_key=self.key(), timeout=3)
            if st == 200:
                try:
                    live = [{"id": x.get("id"), "tokens": x.get("n_prompt_tokens"), "processing": bool(x.get("is_processing")), "n_ctx": x.get("n_ctx")}
                            for x in json.loads(body)]
                except (ValueError, AttributeError):
                    live = None
            elif st in (401, 403):
                live = "no access (API key)"
        return {"dir": slot_dir(), "files": files, "live": live, "tag": tag, "agent_mode": bool(s.get("agent")), "no_warm": bool(s.get("no_warm")),
                "running": self.state in ("starting", "ready")}

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
        # Windows: llama-server ends a moment after the launcher (when its job closes); a Start or the cold-start bench right after found the
        # port still in use
        port = int(self.settings.get("port") or 8080)
        for _ in range(40):
            if not mc.port_busy(port):
                break
            time.sleep(0.25)

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
            if is_warning(line):
                self.add_warning(line)
            m = TIMING_RE.search(line)
            if m:
                key = "prefill" if m.group(1) == "prompt eval" else "decode"
                self.last[key + "_tps"] = float(m.group(4)); self.last[key + "_tokens"] = int(m.group(3))
            if "--stock: running stock llama-server" in line:
                self.mode = "stock llama-server (--stock, to compare)"
            elif "running stock llama-server" in line:
                self.mode = "stock llama-server (the model fits in RAM)"
            elif "stay on the CPU and are managed by the cache" in line:
                self.mode = "moe-cache " + {"fast": "--fast", "hybrid": "--hybrid"}.get(self.settings.get("mode"), "exact")
            elif "all experts fit in your GPU memory" in line:
                self.mode = "all experts on the GPU"
            if line.startswith("moe-cache: MTP on:"):
                self.mtp = True
            m = re.search(r"profile (\S+\.usage)", line)
            if m:
                self.profile = m.group(1).rstrip(";")
        rc = p.wait()
        if p is self.p:
            self.rc = rc
            self.state = "stopped" if self.state == "stopping" or rc in (0, -signal.SIGTERM, 143) else "exited"
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
            if self.state == "ready" and not alive(self.pid):     # found once it serves, so not one of the launcher's short helpers
                self.pid = child_pid(p.pid)
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
    def start_bench(self, cold=False, other=None, compare=False):
        if self.state != "ready":
            raise ValueError("start a model first")
        if self.bench.get("state") == "running":
            raise ValueError("a benchmark is already running")
        if cold and not cold_start_support()["available"]:
            raise ValueError("cold start: " + cold_start_support()["why"])
        self.bench = {"state": "running", "step": "warm-up", "started": time.time(), "cold": bool(cold), "compare": bool(compare)}
        threading.Thread(target=self.run_bench, args=(self.p, cold, list(other or []), compare), daemon=True).start()

    def restart(self, b, s, flush, other, what=""):
        """Stop the server, (flush=True) push the model out of the file cache, start it again with settings s; returns (seconds to ready, flush
        line or "")."""
        b["step"] = what + "stopping the server"
        self.stop()
        ok, line = True, ""
        if flush:
            b["step"] = what + "pushing the model out of the file cache"
            ok, line = flush_model_cache(s["model"], other)
            b["flush"] = line
        b["step"] = what + "starting the server again"
        t0 = time.time()
        self.start(s)
        self.bench = b                          # start() resets the runner; the benchmark goes on
        while self.state == "starting" and time.time() - t0 < 1800:
            time.sleep(0.5)
        if self.state != "ready":
            raise RuntimeError(f"the server did not come back ({self.state})")
        return time.time() - t0, ("" if ok else "partial: ") + line

    def cold_restart(self, b, other):
        """Stop the server, push the model out of the file cache, start it again with the same settings; returns (seconds to ready, flush line)."""
        return self.restart(b, dict(self.settings), True, other, "cold start: ")

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

    def measure(self, b, what=""):
        """One run of the method (warm-up, 2 x 256 decode, one ~3.5k-token prefill) on the running server; returns the CSV row."""
        b["step"] = what + "warm-up"
        t0 = time.time()
        self.complete(BENCH_PROMPTS[0], 32)
        warm = time.time() - t0
        c0 = self.cache_counts()
        dec, ntok = [], []
        for i, q in enumerate(BENCH_PROMPTS[1:3]):
            b["step"] = what + f"decode {i + 1} of 2 (256 tokens)"
            t = self.complete(q, 256, ignore_eos=True)
            dec.append(t["predicted_per_second"]); ntok.append(t.get("predicted_n"))
        c1 = self.cache_counts()
        b["step"] = what + "prefill (about 3.5k tokens)"
        pf = self.complete(BENCH_LONG, 16)
        hr = None
        if c0 and c1 and (c1[0] - c0[0]) + (c1[1] - c0[1]) > 0:
            hr = (c1[0] - c0[0]) / ((c1[0] - c0[0]) + (c1[1] - c0[1]))
        s, sy = self.settings, system_info()
        return {"time": time.strftime("%Y-%m-%d %H:%M"), "model": os.path.basename(s.get("model", "")), "size_gb": f"{size_of(s.get('model', '')) / 1e9:.1f}",
                "mode": (self.mode or "?") + (" + MTP" if self.mtp else ""), "ram_gb": s.get("ram") or "auto", "vram_gib": s.get("vram") or "auto", "ctx": s.get("ctx") or 8192,
                "gpu": sy.get("gpu") or "none", "os": sy.get("os"), "decode_1": f"{dec[0]:.2f}", "decode_2": f"{dec[1]:.2f}",
                "decode_mean": f"{sum(dec) / len(dec):.2f}", "decode_tokens": "/".join(str(n) for n in ntok), "prefill_tokens": pf.get("prompt_n"),
                "prefill": f"{pf['prompt_per_second']:.1f}", "hit_rate": f"{hr:.3f}" if hr is not None else "", "warmup_s": f"{warm:.1f}",
                "command": shown_command(s), "start": "warm", "load_s": "", "flush": ""}

    def save_row(self, b, row):
        try:
            append_bench_row(row)
            b["csv"] = BENCH_CSV
        except OSError as e:
            b["csv_error"] = str(e)
        self.add_line(f"ui: quick benchmark ({row['mode']}): decode {row['decode_1']} / {row['decode_2']} tok/s, prefill {row['prefill_tokens']} tokens at {row['prefill']} tok/s")

    def run_bench(self, p, cold=False, other=(), compare=False):
        """The benchmark on the running server. cold: stop, flush, start again first. compare (I42): then the same with Run as flipped (stock <->
        moe-cache, cold again if cold), and at the end the server is started again with the settings it had, so the user is back where they were."""
        b = self.bench
        orig = dict(self.settings)
        try:
            rows = []
            kinds = [orig]
            if compare:
                was_stock = (self.mode or "").startswith("stock")
                kinds.append(dict(orig, engine="plugin" if was_stock else "stock"))
            for i, s in enumerate(kinds):
                what = f"run {i + 1} of {len(kinds)}: " if compare else ""
                load_s, flush = None, ""
                if cold:
                    load_s, flush = self.restart(b, s, True, other, what + "cold start: ")
                elif i:
                    load_s, flush = self.restart(b, s, False, other, what)
                p = self.p
                row = self.measure(b, what + ("stock: " if (self.mode or "").startswith("stock") else ""))
                row.update(start="cold" if cold else "warm", load_s=f"{load_s:.0f}" if load_s is not None else "", flush=flush)
                self.save_row(b, row)
                rows.append(row)
                b["results"] = list(rows)
            if compare:
                self.restart(b, orig, False, other, "back to your settings: ")
                p = self.p
            b.update(state="done", result=rows[-1], step="")
        except Exception as e:
            err = str(e) if p is None or p.poll() is None else "the server stopped during the benchmark"
            if compare and self.settings != orig:          # never leave the user on the flipped side
                try:
                    self.restart(b, orig, False, other, "back to your settings: ")
                except Exception:
                    err += "; your settings could not be started again: press Start"
            b.update(state="failed", error=err, step="")

    def status(self):
        s = self.settings
        port = int(s.get("port") or 8080)
        return {"state": self.state, "rc": self.rc, "model": s.get("model"), "port": port, "url": f"http://127.0.0.1:{port}",
                "uptime": time.time() - self.started if self.started and self.state in ("starting", "ready") else None,
                "mode": self.mode + (" + MTP" if self.mode and self.mtp else ""), "command": shown_command(s) if s.get("model") else "", "agent": bool(s.get("agent")), "ctx": s.get("ctx"),
                "live_tps": self.live_tps, "last": self.last, "cache": self.cache, "server_rss": self.rss, "log_seq": self.seq,
                "history": list(self.hist), "warnings": self.warnings, "decisions": self.decisions, "bench": self.bench, "api_key": self.key_info(), "warm": {"events": self.warm},
                "system": system_info()}


# ---- the page's own chat: chats kept as JSON files, answers streamed from llama-server through the page server ----
# The page talks only to its own server (CSP connect-src 'self', the page token); this server adds llama-server's API key, so the key never reaches
# the browser. Chats are files and not browser storage because browser storage is per port, and the app (APP2) picks a free port each run.

CHAT_DIR = os.path.join(mc.CACHE_DIR, "ui-chats")
CHAT_ID = re.compile(r"[a-z0-9-]{1,40}")
MAX_CHAT_BODY = 8 * 1024 * 1024        # a chat with a few attached text files
ON_DEMAND = False                       # hook: True once the launcher can start the model on the first request (claude/on-demand)


def chat_path(cid):
    if not isinstance(cid, str) or not CHAT_ID.fullmatch(cid):
        raise ValueError("bad chat id")
    return os.path.join(CHAT_DIR, cid + ".json")


def list_chats():
    out = []
    for f in glob.glob(os.path.join(CHAT_DIR, "*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                c = json.load(fh)
            out.append({"id": c["id"], "title": c.get("title") or "", "model": c.get("model") or "", "t": c.get("t") or os.path.getmtime(f),
                        "n": len(c.get("messages") or []), "tokens": c.get("tokens")})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(out, key=lambda c: -c["t"])


def save_chat(c):
    path = chat_path(c.get("id"))
    msgs = c.get("messages")
    if not isinstance(msgs, list) or not all(isinstance(m, dict) and m.get("role") in ("system", "user", "assistant") and isinstance(m.get("content"), str) for m in msgs):
        raise ValueError("messages: a list of {role, content}")
    keep = {k: c.get(k) for k in ("id", "title", "model", "system", "tokens", "files")}
    keep.update(messages=msgs, t=time.time())
    os.makedirs(CHAT_DIR, mode=0o700, exist_ok=True)
    if not WINDOWS:
        os.chmod(CHAT_DIR, 0o700)       # a folder made 0755 by an older version: the chats inside become private too
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(keep, fh)
    os.replace(tmp, path)
    return {"id": keep["id"], "t": keep["t"]}


def chat_request(body):
    """The /v1/chat/completions body the page may send: messages and a few sampling settings, nothing else."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")
                                                         and isinstance(m.get("content"), str) for m in msgs):
        raise ValueError("messages: a non-empty list of {role, content}")
    r = {"messages": [{"role": m["role"], "content": m["content"]} for m in msgs], "stream": True, "timings_per_token": False}
    t = number(body.get("temperature"), 0, 5, float, "temperature")
    if t is not None:
        r["temperature"] = t
    n = number(body.get("max_tokens"), 1, 1048576, int, "max_tokens")
    if n is not None:
        r["max_tokens"] = n
    return r


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

    def pick_specs(self, fresh=False):
        """The picker's view of this machine (kept 5 minutes: it asks nvidia-smi, PowerShell and llama-server --version)."""
        if fresh or not getattr(self, "_spec", None) or time.time() - self._spec[0] > 300:
            self._spec = (time.time(), picker.machine_specs(self.folders(), self.ls))
        return self._spec[1]

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


# what the page may do: its own scripts and requests (the chat goes through /api/chat), no frames, never be framed itself (clickjacking)
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; "
       "frame-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'; object-src 'none'")
MAX_BODY = 64 * 1024


def token_eq(given, token):
    """Constant-time token check that also takes odd input (compare_digest raises TypeError on non-ASCII str: a dropped connection)."""
    return bool(given) and bool(token) and secrets.compare_digest(str(given).encode("utf-8", "replace"), str(token).encode("utf-8"))


def open_page(url):
    """Open the page without putting its token on the browser's command line, which other users of a Linux machine can read in /proc:
    the browser gets a private (0600) file that forwards to the address; the file is removed after a minute."""
    import tempfile
    import webbrowser
    fd, path = tempfile.mkstemp(prefix="moe-cache-ui-", suffix=".html")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><meta charset=utf-8><meta http-equiv=refresh content='0;url={url}'><title>moe-cache</title>"
                f"<p style='font:15px system-ui;margin:2em'><a href='{url}'>Open the moe-cache control page</a></p>")
    import pathlib
    webbrowser.open(pathlib.Path(path).as_uri())

    def gone():
        try:
            os.unlink(path)
        except OSError:
            pass
    t = threading.Timer(60, gone)
    t.daemon = True             # Ctrl-C right after the start does not keep the process (and the Windows window) for a minute (audit U3)
    t.start()


class Handler(BaseHTTPRequestHandler):
    store = None
    runner = None
    token = ""                  # random per run: the URL the launcher opens carries it once, the page keeps it in a cookie
    server_version = "moe-cache-ui"
    timeout = 30                # a client that sends nothing does not hold a thread forever

    def log_message(self, *a):
        pass

    def allowed(self, navigation=False, cross_site=False):
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
        return site is None or site == "same-origin" or (navigation and (site == "none" or (cross_site and site == "cross-site")))

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
        return bool(t) and token_eq(t, self.token)

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
        link = u.path in ("/", "/index.html") and token_eq(q.get("t"), self.token)
        # the launcher opens the link from a private file:// page (cross-site), so the token never shows in the browser's command line
        if not self.allowed(navigation=u.path in ("/", "/index.html")) and not (link and self.allowed(navigation=True, cross_site=True)):
            return self.send(403, {"error": "forbidden"})
        if u.path in ("/", "/index.html"):
            if link and self.headers.get("Sec-Fetch-Site") == "cross-site":
                # from the launcher's file:// page: a redirect would stay "cross-site" and the browser would hold back the SameSite=Strict cookie,
                # so set the cookie here and let this page itself move on to / (a same-origin navigation)
                body = b"<!doctype html><meta charset=utf-8><meta http-equiv=refresh content='0;url=/'><title>moe-cache</title>"
                self.send_response(200)
                self.send_header("Set-Cookie", f"{self.cookie_name()}={self.token}; Path=/; HttpOnly; SameSite=Strict")
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if link:
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
                return self.send(200, dict(system_info(), folders=self.store.folders(), cold_start=cold_start_support(), last=self.store.saved.get("last", {}), pick=self.store.saved.get("pick"),
                                           llama_server=self.store.ls, on_demand=ON_DEMAND, plugin=os.environ.get("MOE_CACHE_LIB") or "found by the launcher (build folder or kit)"))
            if u.path == "/api/pick":                      # "Which model?": this machine and one recommendation per use (bin/moe_cache_pick.py)
                spec = self.store.pick_specs(q.get("fresh") == "1")
                uses = [q["use"]] if q.get("use") in picker.USES else list(picker.USES)
                return self.send(200, {"machine": spec, "lines": picker.spec_lines(spec), "uses": picker.USES,
                                       "recommendations": [picker.recommend(spec, x, q.get("model") or None) for x in uses], "pick": self.store.saved.get("pick")})
            if u.path == "/api/models":
                return self.send(200, {"folders": self.store.folders(), "models": self.store.list_models(q.get("fresh") == "1")})
            if u.path == "/api/plan":
                m = self.known(q.get("model"))
                ps = self.settings(dict(q, model=m))             # the same checks as Start (numbers in range, no nan / inf)
                q = {"ram": str(ps["ram"]) if "ram" in ps else "", "vram": str(ps["vram"]) if "vram" in ps else "", "ctx": str(ps.get("ctx", 8192)),
                     "mode": ps["mode"], "agent": "1" if ps["agent"] else "", "engine": ps["engine"], "mtp": ps["mtp"]}
                key = (m, q.get("ram"), q.get("vram"), q.get("ctx"), q.get("mode"), q.get("agent"), q["engine"], q["mtp"])
                hit = self.store.plans.get(key)
                if not hit or time.time() - hit[0] > 60:          # free RAM and VRAM change: a plan is kept for a minute
                    hit = self.store.plans[key] = (time.time(), plan_model(m, q.get("ram") or None, q.get("vram") or None, int(q.get("ctx") or 8192), q.get("mode") or "exact",
                                                                             q.get("agent") in ("1", "true"), q["engine"], q["mtp"], self.store.ls))
                return self.send(200, dict(hit[1], here=measured_here(m)))       # the benchmark lines are fresh even when the plan is cached
            if u.path == "/api/status":
                _GPU["watched"] = time.time()
                return self.send(200, self.runner.status())
            if u.path == "/api/bench":
                rows = read_bench_rows(500)
                return self.send(200, {"csv": BENCH_CSV, "rows": rows[-20:], "compare": bench_compare(rows)})
            if u.path == "/api/sessions":
                return self.send(200, self.runner.sessions())
            if u.path == "/api/servers":                   # llama-servers on this machine (ports 8080-8089), started here or elsewhere
                own = int(self.runner.settings.get("port") or 8080) if self.runner.state in ("starting", "ready") else None
                return self.send(200, {"servers": peek_servers(own=own)})
            if u.path == "/api/peek":                      # one port on 127.0.0.1 (a server started elsewhere, e.g. by a script or the on-demand server)
                port = number(q.get("port"), 1, 65535, int, "port")
                if port is None:
                    raise ValueError("port: a number")
                return self.send(200, peek_server(port, trusted=port == on_demand_port()))
            if u.path == "/api/key":                       # for the Copy button only (token-protected, not cached)
                if self.runner.state not in ("starting", "ready") or not self.runner.key():
                    raise ValueError("no API key in use")
                return self.send(200, {"key": self.runner.key()})
            if u.path == "/api/chats":
                if q.get("id"):
                    with open(chat_path(q["id"]), encoding="utf-8") as fh:
                        return self.send(200, json.load(fh))
                return self.send(200, {"dir": CHAT_DIR, "chats": list_chats()})
            if u.path == "/api/log":
                since = int(q.get("since") or 0)
                return self.send(200, {"lines": [[n, l] for n, l in list(self.runner.log) if n > since]})
        except FileNotFoundError:
            return self.send(404, {"error": "no such chat"})
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
        if not 0 <= n <= (MAX_CHAT_BODY if self.path in ("/api/chat", "/api/chats/save") else MAX_BODY):
            self.close_connection = True
            return self.send(413, {"error": "request too large"})
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("a JSON object is expected")
            if self.path == "/api/chat":
                return self.chat(body)
            if self.path == "/api/chats/save":
                return self.send(200, save_chat(body))
            if self.path == "/api/chats/trash":
                return self.send(200, {"ok": True, "where": move_to_trash(chat_path(body.get("id")))})
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
                if body.get("on_demand") and not ON_DEMAND:
                    raise ValueError("Start on first request is not in this version yet")
                s = self.settings(body)
                self.store.saved["last"] = {k: v for k, v in s.items() if k != "api_key"}
                self.store.save()
                self.runner.start(s)
                return self.send(200, self.runner.status())
            if self.path == "/api/pick/use":               # "Use it": the same settings `moe_cache.py pick --apply` writes (no download here)
                if body.get("use") not in picker.USES:
                    raise ValueError("use: " + ", ".join(picker.USES))
                rec = picker.recommend(self.store.pick_specs(True), body["use"], body.get("model") or None)
                picker.apply(rec, self.store.saved)
                self.store.save()
                self.store.models_at = 0                   # the list shows the file at once
                return self.send(200, {"recommendation": rec, "last": self.store.saved.get("last", {}), "pick": self.store.saved.get("pick")})
            if self.path == "/api/command":
                s = self.settings(body)
                return self.send(200, {"launcher": shown_command(s), "lines": dry_run(s)})
            if self.path == "/api/sessions/trash":
                name = str(body.get("name") or "")
                f = next((x for x in warm_files() if x["name"] == name), None)
                if not f:
                    raise ValueError("no such saved session (refresh)")
                where = move_to_trash(os.path.join(slot_dir(), name))
                self.runner.add_line(f"ui: moved the saved agent session {name} to {where}")
                return self.send(200, {"ok": True, "where": where, "running_note": "the running server keeps it in memory and may save it again"
                                       if self.runner.state == "ready" and f.get("tag") == session_tag(self.runner.settings) else ""})
            if self.path == "/api/bench":
                other = body.get("other") or []
                if not isinstance(other, list) or len(other) > 20 or not all(isinstance(x, str) and len(x) < 1000 for x in other):
                    raise ValueError("other: a list of folders")
                other = [os.path.abspath(os.path.expanduser(x)) for x in other if x.strip()]
                for x in other:
                    if not os.path.exists(x):
                        raise ValueError(f"no such folder or file: {x}")
                self.runner.start_bench(cold=bool(body.get("cold")), other=other or self.store.folders(), compare=bool(body.get("compare")))
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

    def chat(self, body):
        """Stream one answer: llama-server's server-sent events passed through line by line, plus one {"moe": ...} event before [DONE] with this
        answer's expert-cache hits and misses (plugin only). The page stops an answer by closing the request; llama-server then stops too."""
        r = self.runner
        if r.state != "ready":
            raise ValueError("no model is running: start one first")
        req = chat_request(body)
        before = r.cache_counts()
        port = int(r.settings.get("port") or 8080)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
        hdr = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if r.key():
            hdr["Authorization"] = "Bearer " + r.key()
        streaming = False
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps(req), hdr)
            up = conn.getresponse()
            if up.status != 200:
                text = up.read().decode(errors="replace")
                try:
                    text = json.loads(text)["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    pass
                return self.send(502, {"error": f"llama-server answered {up.status}: {str(text)[:500]}"})
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            streaming = True
            while True:
                line = up.readline()
                if not line:
                    break
                if line.strip() == b"data: [DONE]":
                    after = r.cache_counts() if before else None
                    if before and after:
                        h, m = after[0] - before[0], after[1] - before[1]
                        self.wfile.write(b"data: " + json.dumps({"moe": {"hits": h, "misses": m, "hit_rate": h / (h + m) if h + m else None}}).encode() + b"\n\n")
                self.wfile.write(line)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                                  # the page pressed Stop: closing our side makes llama-server stop generating
        except OSError as e:
            if not streaming:
                self.send(502, {"error": f"llama-server did not answer: {e}"})
        finally:
            conn.close()

    def known(self, path):
        """Only models the page listed can be planned or started."""
        for m in self.store.list_models():
            if m["path"] == path:
                return path
        raise ValueError("unknown model (refresh the list)")

    def settings(self, d):
        s = {"model": self.known(d.get("model")), "mode": d.get("mode") if d.get("mode") in ("exact", "fast", "hybrid") else "exact",
             "engine": d.get("engine") if d.get("engine") in ("auto", "plugin", "stock") else "auto",
             "mtp": {"on": True, "off": False}.get(str(d.get("mtp")).lower()),
             "agent": str(d.get("agent")).lower() in ("1", "true", "on"), "no_warm": str(d.get("no_warm")).lower() in ("1", "true", "on"), "llama_server": self.store.ls, "api_key": d.get("api_key") or ""}
        if s["engine"] == "stock":
            s["mode"] = "exact"                   # --fast and --hybrid need the plugin
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
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, env=env)
    out = {"rc": r.returncode, "output": (r.stdout + r.stderr).strip(), "file": None, "written": None}
    m = re.search(r"; config (.+?\.jsonc?)(?: \(old file|$)", out["output"], re.M)       # opencode.jsonc too (OpenCode Desktop creates it)
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
    store = Store(ls or mc.default_llama_server())
    h = type("H", (Handler,), {"store": store, "runner": Runner(), "token": token or secrets.token_urlsafe(24)})
    srv = ThreadingHTTPServer(("127.0.0.1", port), h)
    srv.daemon_threads = True
    return srv


def ui(argv):
    ap = argparse.ArgumentParser(prog="moe-cache-server --ui", description="local control page for moe-cache (this machine only)")
    ap.add_argument("--ui-port", type=int, default=8070)
    ap.add_argument("--llama-server", default=mc.default_llama_server())
    ap.add_argument("--no-browser", action="store_true", help="do not open the page in the browser")
    a = ap.parse_args(argv)
    if not WINDOWS:
        os.umask(0o077)         # chats (whole conversations, attached files), ui.json and bench.csv private to this user, as the launcher's files (audit U2)
    try:
        srv = make_server(a.ui_port, a.llama_server)
    except OSError as e:
        raise SystemExit(f"moe-cache: cannot listen on 127.0.0.1:{a.ui_port} ({e}): --ui-port N to pick another port")
    if WINDOWS:
        threading.Thread(target=watch_windows_gpu, args=(a.llama_server,), daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/?t={srv.RequestHandlerClass.token}"
    mc.say(f"moe-cache: control page at {url} (this machine and this user only; Ctrl-C stops it and any server it started)")
    if not a.no_browser:
        t = threading.Timer(0.5, open_page, args=(url,))
        t.daemon = True
        t.start()
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
