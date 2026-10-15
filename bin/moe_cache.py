#!/usr/bin/env python3
"""moe-cache launcher (Linux and Windows; Python 3.8+, standard library only).

  moe_cache.py server  MODEL.gguf [options] [-- extra llama-server args]   run llama-server with the expert-cache plugin
  moe_cache.py check   [MODEL.gguf] [--llama-server PATH] [--ram GB]       check that this machine, llama-server and the model are ready
  moe_cache.py learn   MODEL.gguf [options]                                build a warm-start profile from a set of prompts
  moe_cache.py project save|restore|list [NAME] [--url URL] [--api-key KEY] save / restore a running server's prompt state

`moe_cache.py server --help` (and so on) lists the options. The bash scripts bin/moe-cache-server, -check, -learn and -project call this file.
"""
import argparse, glob, json, os, re, shutil, signal, socket, subprocess, sys, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = os.name == "nt"
CACHE_DIR = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "moe-cache")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def say(msg, err=False):
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def which(prog):
    return shutil.which(prog) or prog


# ---- how a llama-server loads the plugin ----

def ldd(path):
    if WINDOWS:
        return ""
    try:
        return subprocess.run(["ldd", which(path)], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def load_env(ls, lib, method="auto"):
    """Environment that makes this llama-server load the plugin, and the method used.
    preload: llama-server links its backends directly (distro packages): LD_PRELOAD.  path: backends are separate files (official releases): GGML_BACKEND_PATH."""
    deps = ldd(ls)
    if not WINDOWS and "libggml-base" not in deps:
        say(f"warning: {which(ls)} does not use shared ggml libraries, the plugin cannot be loaded into it (build llama.cpp with BUILD_SHARED_LIBS=ON)", err=True)
    if method == "auto":
        method = "preload" if (not WINDOWS and "libggml-cpu" in deps) else "path"
    if method == "preload":
        if WINDOWS:
            raise SystemExit("--load-method preload does not exist on Windows (use path)")
        return [("LD_PRELOAD", lib), ("MOE_CACHE_PRELOAD", "1")], method
    if method == "path":
        return [("GGML_BACKEND_PATH", lib)], method
    raise SystemExit(f"unknown load method: {method} (use auto, preload or path)")


GGML_API_OF = {(0, 24): "2", (0, 25): "2", (0, 26): "3", (0, 27): "3"}


def ls_api(ls):
    """Backend API of the ggml this llama-server uses, from its libggml-base version (0.24-0.25 -> 2, 0.26-0.27 -> 3); None when unknown."""
    if WINDOWS:
        # no ldd there (the warning never fired on the Claw): ggml-base.dll next to llama-server.exe holds its version as a plain string ("0.27.0")
        try:
            data = open(os.path.join(os.path.dirname(which(ls)), "ggml-base.dll"), "rb").read()
            for m in re.finditer(rb"\x00(\d+)\.(\d+)\.(\d+)\x00", data):
                v = GGML_API_OF.get((int(m.group(1)), int(m.group(2))))
                if v:
                    return v
        except OSError:
            pass
        return None
    for line in ldd(ls).splitlines():
        if "libggml-base" in line and "=>" in line:
            path = line.split("=>")[1].split()[0]
            m = re.search(r"(\d+)\.(\d+)\.(\d+)", os.path.basename(os.path.realpath(path)))
            if m:
                v = (int(m.group(1)), int(m.group(2)))
                return GGML_API_OF.get(v)
    return None


def lib_api(lib):
    """API a plugin build was made for (a plain string inside the library); None for builds from before 0.2.3."""
    try:
        m = re.search(rb"moe-cache-api=(\d+)", open(lib, "rb").read())
        return m.group(1).decode() if m else None
    except OSError:
        return None


def default_lib():
    return os.path.join(HERE, "build", "win", "ggml-moe-cache.dll") if WINDOWS else os.path.join(HERE, "build", "libggml-moe-cache.so")


def pick_lib(ls):
    """The plugin build in build*/ that matches the server's ggml, else the default build."""
    want = ls_api(ls)
    if want:
        if os.path.isfile(default_lib()) and lib_api(default_lib()) == want:
            return default_lib()
        # several builds for this API (old test builds, too): the newest one, which is the most likely to match the code
        found = [c for c in glob.glob(os.path.join(HERE, "build-*", "libggml-moe-cache.so")) if lib_api(c) == want]
        if found:
            return max(found, key=os.path.getmtime)
    return default_lib()


# ---- small helpers ----

def model_parts(model):
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", model):
        return glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", model))
    return [model]


def win_memory():
    """(total, available) bytes on Windows (GlobalMemoryStatusEx); (None, None) elsewhere or on failure."""
    if not WINDOWS:
        return None, None
    try:
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return float(m.ullTotalPhys), float(m.ullAvailPhys)
    except Exception:
        pass
    return None, None


def parse_core_records(buf):
    """Cores from a GetLogicalProcessorInformationEx(RelationProcessorCore) buffer: [(efficiency_class, [logical CPUs])], processor group 0 only.
    Record layout: DWORD Relationship, DWORD Size, then PROCESSOR_RELATIONSHIP: BYTE Flags, BYTE EfficiencyClass, BYTE Reserved[20],
    WORD GroupCount, GROUP_AFFINITY GroupMask[] at offset 32 (KAFFINITY Mask, WORD Group, WORD Reserved[3])."""
    import struct
    cores, off = [], 0
    while off + 48 <= len(buf):
        rel, size = struct.unpack_from("<II", buf, off)
        if size < 48:
            break
        if rel == 0:
            eff = buf[off + 9]
            mask, group = struct.unpack_from("<QH", buf, off + 32)
            if group == 0:
                cores.append((eff, [i for i in range(64) if mask >> i & 1]))
        off += size
    return cores


def hybrid_masks(cores):
    """(performance-core mask with one thread per core, efficiency CPUs) when the cores have more than one efficiency class, else None.
    Windows gives the performance cores the HIGHER efficiency class."""
    classes = {e for e, _ in cores}
    if len(classes) < 2:
        return None
    top = max(classes)
    pmask = sum(1 << cpus[0] for e, cpus in cores if e == top and cpus)
    ecpus = sorted(c for e, cpus in cores if e != top for c in cpus)
    return pmask, ecpus


def win_cores():
    """[(efficiency_class, [logical CPUs])] on Windows, [] elsewhere or on failure (Claw finding 5: Windows had no core pinning)."""
    if not WINDOWS:
        return []
    try:
        import ctypes
        k = ctypes.windll.kernel32
        n = ctypes.c_uint32(0)
        k.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value)
        if not k.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(n)):
            return []
        return parse_core_records(buf.raw[:n.value])
    except Exception:
        return []


def mem_total():
    if WINDOWS:
        return win_memory()[0]
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def mem_for_server(ram_gb):
    """Bytes the server may use: --ram (GB), else the cgroup limit, else available RAM; None when unknown."""
    if ram_gb:
        return float(ram_gb) * 1e9
    if WINDOWS:
        return win_memory()[1]      # found on the MSI Claw: /proc does not exist there, so stock-vs-plugin could not be decided without --ram
    try:
        rel = open("/proc/self/cgroup").read().strip().split(":")[-1]
        for f in (f"/sys/fs/cgroup{rel}/memory.max", "/sys/fs/cgroup/memory.max"):
            if os.access(f, os.R_OK):
                v = open(f).read().strip()
                if v != "max":
                    return float(v)
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) * 1024.0
    except (OSError, ValueError):
        pass
    return None


def parse_cpus(text):
    out = []
    for part in text.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def arg_after(args, *names):
    v = None
    for i, a in enumerate(args):
        if a in names and i + 1 < len(args):
            v = args[i + 1]
    return v


def http(url, body=None, api_key=None, timeout=10):
    """(status, text) of a request; status 0 when the server cannot be reached."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None, headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception:
        return 0, ""


def healthy(port, api_key=None):
    return http(f"http://127.0.0.1:{port}/health", api_key=api_key, timeout=2)[0] == 200


def port_busy(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def python_cmd():
    return os.environ.get("MOE_CACHE_PYTHON") or sys.executable or "python3"


_JOB = None   # Windows job object: kept open for the launcher's lifetime; closing it (the launcher exits or is killed) kills llama-server


def _kill_with_parent_windows(p):
    """Put the child in a job object that kills it when the launcher's handle closes (the launcher exits, is closed or killed).
    Found on the MSI Claw (2026-10-15): without it a killed launcher left llama-server running on its port."""
    global _JOB
    import ctypes
    from ctypes import wintypes
    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]
    class IOC(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]
    class EXT(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IOC), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = EXT()
    info.Basic.LimitFlags = 0x2000           # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = job and k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)) and \
        k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(p._handle)))
    if ok:
        _JOB = job
    else:
        say("moe-cache: warning: could not tie llama-server to the launcher (closing this window may leave it running)", err=True)


def _die_with_parent_linux():
    """preexec_fn: the child gets SIGTERM when the launcher dies, even if the launcher is killed with SIGKILL."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
    except Exception:
        pass


def run_child(cmd, env, port=None, expect_plugin=False):
    """Run llama-server as a child (so helper threads can keep working), pass Ctrl-C / SIGTERM on and return its exit code.
    The child dies with the launcher. Its log is passed through; when the plugin was meant to load and the server becomes ready without a single
    'moe-cache:' line, a warning says it is running stock (found on the MSI Claw, 2026-10-15: the plugin failed to load and nothing said so)."""
    kw = {"env": env, "stderr": subprocess.PIPE}
    if sys.platform.startswith("linux"):
        kw["preexec_fn"] = _die_with_parent_linux
    p = subprocess.Popen(cmd, **kw)
    if WINDOWS:
        try:
            _kill_with_parent_windows(p)
        except Exception as e:
            say(f"moe-cache: warning: could not tie llama-server to the launcher ({e})", err=True)
    seen = {"plugin": False, "disabled": "", "last": "", "ready": False}
    def pump():
        out = sys.stderr.buffer
        for line in iter(p.stderr.readline, b""):
            out.write(line); out.flush()
            if b"moe-cache:" in line:
                seen["plugin"] = True
                seen["last"] = line.decode(errors="replace").strip()
                if b"plugin disabled" in line or b"refus" in line:
                    seen["disabled"] = line.decode(errors="replace").strip()
    threading.Thread(target=pump, daemon=True).start()
    if expect_plugin and port:
        def check():
            for _ in range(900):
                if p.poll() is not None:
                    return
                if healthy(port):
                    seen["ready"] = True
                    time.sleep(1)
                    if seen["disabled"]:
                        say(f"moe-cache: WARNING: the plugin loaded but disabled itself ({seen['disabled']}): llama-server runs stock", err=True)
                    elif not seen["plugin"]:
                        say("moe-cache: WARNING: the plugin did not load: llama-server is running stock llama.cpp. Likely causes: the plugin library "
                            "does not match this llama-server's ggml version (API 2 vs 3), a wrong MOE_CACHE_LIB path, or (preload method) a llama-server "
                            "without shared ggml libraries. Run: moe_cache.py check MODEL --llama-server ...", err=True)
                    return
                time.sleep(1)
        threading.Thread(target=check, daemon=True).start()
    def stop(sig, frame):
        p.terminate()
    if not WINDOWS:
        signal.signal(signal.SIGTERM, stop)
    while True:
        try:
            rc = p.wait()
            break
        except KeyboardInterrupt:      # the child got the same Ctrl-C; wait for it to finish
            continue
    quiet = (rc & 0xFFFFFFFF) in (0xC000013A, 1) if WINDOWS else rc in (-signal.SIGINT, -signal.SIGTERM, 130, 143)   # Ctrl-C / terminate
    if rc not in (0, None) and not quiet:
        say("moe-cache: " + exit_reason(rc, seen["ready"] if (expect_plugin and port) else None) + (f"; last plugin line: {seen['last']}" if seen["last"] else ""), err=True)
    return rc


def exit_reason(rc, ready):
    """One line on why llama-server stopped (the Claw's cache-mode crash ended the launcher without a word: RESULTS 50)."""
    when = "" if ready is None else (" after it was ready" if ready else " before it was ready")
    if WINDOWS:
        code = rc & 0xFFFFFFFF
        names = {0xC0000005: "access violation", 0xC0000409: "stack buffer overrun / fast fail", 0xC000001D: "illegal instruction (CPU lacks an instruction set)",
                 0xC0000135: "a DLL was not found", 0xC0000139: "a DLL entry point was not found", 0xC00000FD: "stack overflow", 0xC0000017: "out of memory"}
        return f"llama-server exited{when} with code 0x{code:08X}" + (f" ({names[code]})" if code in names else "")
    if rc < 0:
        try:
            nm = signal.Signals(-rc).name
        except ValueError:
            nm = f"signal {-rc}"
        hint = " (killed: out of memory? check 'journalctl -k' or the cgroup limit)" if -rc == signal.SIGKILL else ""
        return f"llama-server was stopped{when} by {nm}{hint}"
    return f"llama-server exited{when} with code {rc}"


# ---- server ----

SERVER_HELP = """\
--fast     opt-in approximate routing (MOE_CACHE_ROUTE=skip): a low-ranked expert that is not in RAM is skipped instead of read from the SSD.
           About +10-12 % when the model is much bigger than RAM; about 1 top token in 40-50 differs from stock (KL 0.005-0.012, docs/RESULTS.md section 31)
--hybrid   opt-in, experimental: a bank of your most used experts (from the usage profile) lives in free GPU memory; the GPU computes the picks it holds
           while the CPU computes the rest at the same time (Flash-Next on 8 GB: 13.1 -> 17.4 tok/s). Not identical to stock (GPU float order, like
           --gpu-cache); replaces --gpu-cache and keeps all experts on the CPU side. MOE_CACHE_HYBRID=MIB sets the bank size (default: all free VRAM)
--profile NAME  use (and keep learning) a separate usage profile for one kind of work, e.g. coding, thinking, research; default: one shared profile
--project NAME  its own profile plus a saved prompt state (KV cache) restored at start; save it with: moe_cache.py project save NAME
--gpu-cache off|auto|MIB  llama.cpp's GPU expert cache (--moe-cache-mib) on top of this cache. auto = all free VRAM the tuner sees minus --gpu-margin GiB.
           Not bit-identical to CPU-computed experts (GPU float order), identical to llama.cpp's own cache. Needs llama.cpp 23b0202a1 or newer (API 3 plugin build).
--gpu-margin GiB  VRAM the tuner leaves free next to everything it places (default about 0.47 GiB = 0.5 GB; raise it if the GPU runs out of memory with a large -ub)
--ub auto|N  micro-batch size (prompt speed: 512 -> 2048 was +58 % at a 6k prompt). auto = the largest that fits next to the KV cache of --ctx
--ram GB   memory the server may use (default: the cgroup limit or free RAM). The cache sizes itself to stay below it.
--vram GiB GPU memory to plan with (default: what the GPU reports free)
--n-cpu-moe N  layers whose experts stay on the CPU (default: chosen by the planner from free VRAM)
--no-ecores  do not let the expert executor use efficiency cores (hybrid Intel CPUs: up to 4 are used by default, about +10-15 %)
--always-plugin  use the plugin even when the model fits in the memory the server may use (by default stock llama-server runs then: faster, RESULTS 44)
--no-pin     do not pin llama-server's threads (hybrid Intel CPUs: one thread per performance core by default, about +15 % decode)
MOE_CACHE_PYTHON  python with the gguf package for the planner (default: this python)"""


def server(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py server", description="run llama-server with the expert-cache plugin",
                                 epilog=SERVER_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--project"); ap.add_argument("--profile"); ap.add_argument("--ram"); ap.add_argument("--vram")
    ap.add_argument("--ctx", type=int, default=8192); ap.add_argument("--n-cpu-moe"); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER", "llama-server"))
    ap.add_argument("--load-method", default="auto", choices=["auto", "preload", "path"])
    ap.add_argument("--gpu-cache", default="off"); ap.add_argument("--gpu-margin", type=float); ap.add_argument("--ub", default="auto")
    ap.add_argument("--fast", action="store_true"); ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--no-ecores", action="store_true"); ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--open", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--always-plugin", action="store_true")
    extra = []
    if "--" in argv:
        i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    if not WINDOWS:
        os.umask(0o077)       # the usage profile is private to this user
    ls, model, ctx, port = a.llama_server, a.model, a.ctx, a.port
    gcache = a.gpu_cache
    if a.hybrid and gcache != "off":
        say("moe-cache: --hybrid replaces --gpu-cache: the GPU memory goes to the hybrid bank"); gcache = "off"
    if not os.path.isfile(model):
        raise SystemExit(f"usage: moe-cache-server MODEL.gguf [options]  (model file not found: '{model}')")
    lib = os.environ.get("MOE_CACHE_LIB") or pick_lib(ls)
    want, have = ls_api(ls), lib_api(lib)
    if want and have and want != have:
        say(f"moe-cache: WARNING: {lib} was built for ggml backend API {have} but {ls} uses API {want}: the plugin will refuse to load "
            "(build with GGML_API=3 ./build.sh for llama.cpp 23b0202a1 or newer)", err=True)
    if not os.path.isfile(lib):
        if os.environ.get("MOE_CACHE_LIB"):
            raise SystemExit(f"MOE_CACHE_LIB points to a file that does not exist: {lib}")
        raise SystemExit(f"plugin not built: run ./build.sh first (looked for {lib})")
    ncmoe_user = a.n_cpu_moe
    ncmoe = a.n_cpu_moe
    if ncmoe is None:
        p = [python_cmd(), os.path.join(HERE, "planner", "plan.py"), model, "--ctx", str(ctx), "--json"]
        if a.ram: p += ["--ram-gib", a.ram]
        if a.vram: p += ["--vram-gib", a.vram]
        r = None
        try:
            r = subprocess.run(p, capture_output=True, text=True, timeout=300)
            ncmoe = str(json.loads(r.stdout)["n_cpu_moe"])
        except Exception as e:
            # say why (the Claw's planner failed on /proc/meminfo and this line blamed the gguf package, which was not the cause)
            why = (r.stderr.strip().splitlines() or [""])[-1] if r is not None else str(e)
            say(f"moe-cache: planner failed ({why or e}); keeping all experts on the CPU"); ncmoe = "999"
    host = arg_after(extra, "--host") or ""
    if not (host == "" or host.startswith("127.") or host in ("localhost", "::1")) and "--api-key" not in extra:
        say(f"WARNING: llama-server will listen on '{host}' without --api-key: anyone who can reach this machine can use the model and the web UI. "
            "Add --api-key KEY after the -- or bind to 127.0.0.1.", err=True)
    pname = a.profile or ""
    if pname and not NAME_RE.match(pname):
        raise SystemExit("--profile NAME may only contain letters, digits, _ and -")
    slotdir = None
    if a.project:
        if not NAME_RE.match(a.project):
            raise SystemExit("--project NAME may only contain letters, digits, _ and -")
        pname = pname or a.project
        slotdir = os.path.join(CACHE_DIR, "slots", a.project); os.makedirs(slotdir, exist_ok=True)
        extra += ["--slot-save-path", slotdir]
    profile = os.path.join(CACHE_DIR, os.path.basename(model) + "." + (pname + "." if pname else "") + "usage")
    load, method = load_env(ls, lib, a.load_method)
    # the model fits in the memory the server may use: stock llama-server is faster (its page cache holds the whole file and stays warm across
    # restarts; Qwen3.6-35B at 24 GB: stock 42.9 vs plugin 36.6 warm / 33.4 cold, RESULTS 44). --fast and --hybrid need the plugin, so they keep it.
    try:
        size = sum(os.path.getsize(p) for p in model_parts(model))
    except OSError:
        size = None
    avail = mem_for_server(a.ram)
    stock = bool(size and avail and size + 1.5e9 <= avail and not (a.fast or a.hybrid or a.always_plugin))
    if gcache != "off" and not stock and method == "path":
        # the plugin can feed llama.cpp's GPU cache only when preloaded; loaded as a file it would let the GPU read experts that are not in RAM (T7)
        say("moe-cache: --gpu-cache needs a llama-server that links its backends directly (LD_PRELOAD loading); this one loads them as files "
            "(GGML_BACKEND_PATH): using static GPU expert layers instead", err=True)
        gcache = "off"
    env = load + [("MOE_CACHE_SIZE_GIB", "auto"), ("MOE_CACHE_GGUF", model), ("MOE_CACHE_PROFILE", profile)]
    if a.ram:
        env.append(("MOE_CACHE_RAM_GIB", a.ram))
    if a.fast:
        env.append(("MOE_CACHE_ROUTE", "skip"))
        say("moe-cache: --fast: low-ranked experts that are not in RAM are skipped (about +10 %, output no longer identical to stock)")
    elif not stock:
        # suggest --fast only where it pays: the model is much bigger than the memory it may use
        try:
            size = sum(os.path.getsize(p) for p in model_parts(model))
            lim = float(a.ram) * 1e9 if a.ram else mem_total()
            if lim and size > 2 * lim:
                say(f"moe-cache: the model is {size / lim:.1f}x the memory it may use: --fast skips rarely needed experts for about +10 % (changes some tokens; off by default)")
        except Exception:
            pass
    if not stock and not a.no_ecores and not os.environ.get("MOE_CACHE_EXEC_ECORES") and os.access("/sys/devices/cpu_atom/cpus", os.R_OK):
        try:
            ne = min(len(parse_cpus(open("/sys/devices/cpu_atom/cpus").read())), 4)
        except Exception:
            ne = 0
        if ne > 0:
            env.append(("MOE_CACHE_EXEC_ECORES", str(ne)))
            say(f"moe-cache: expert executor will also use {ne} efficiency cores (--no-ecores to disable)")
    pinargs = []
    hm = hybrid_masks(win_cores()) if WINDOWS else None
    if hm and not any(x in ("-C", "--cpu-mask", "-Cr", "--cpu-range") for x in extra):
        pmask, ecpus = hm
        if not a.no_pin:
            pinargs = ["-C", hex(pmask), "--cpu-strict", "1"]
            env.append(("MOE_CACHE_CPU_MASK", hex(pmask)))
            say(f"moe-cache: hybrid CPU: pinning llama-server threads to the performance cores (-C {hex(pmask)}); --no-pin to disable")
        ne = min(len(ecpus), 4)
        if ne and not stock and not a.no_ecores and not os.environ.get("MOE_CACHE_EXEC_ECORES"):
            env += [("MOE_CACHE_EXEC_ECORES", str(ne)), ("MOE_CACHE_ECORE_MASK", hex(sum(1 << c for c in ecpus)))]
            say(f"moe-cache: expert executor will also use {ne} efficiency cores (--no-ecores to disable)")
    if not a.no_pin and os.access("/sys/devices/cpu_core/cpus", os.R_OK) and not any(x in ("-C", "--cpu-mask", "-Cr", "--cpu-range") for x in extra):
        try:
            cores = parse_cpus(open("/sys/devices/cpu_core/cpus").read())
            prim = [c for c in cores if min(parse_cpus(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read())) == c]
            mask = hex(sum(1 << c for c in prim))
            pinargs = ["-C", mask, "--cpu-strict", "1"]
            say(f"moe-cache: hybrid CPU: pinning llama-server threads to the performance cores (-C {mask}); --no-pin to disable")
        except Exception:
            pass
    # micro-batch size and GPU cache size: one shared tuner (planner/gguf_plan.py: tune), so the launcher and the benchmark tool decide the same way
    tuned = None
    if a.ub == "auto" or gcache == "auto":
        try:
            sys.path.insert(0, os.path.join(HERE, "planner"))
            from gguf_plan import tune
            free = float(a.vram) * 2**30 if a.vram else None          # GiB, the same unit as the planner's --vram-gib
            kw = {"reserve_gb": a.gpu_margin * 2**30 / 1e9} if a.gpu_margin is not None else {}     # default: the tuner's own 0.5 GB
            tuned = tune(model, ctx, free=free, want_cache=gcache != "off", **kw)
        except Exception:
            tuned = None
    ub = a.ub
    if ub == "auto":
        ub = "" if any(x in ("-ub", "--ubatch-size") for x in extra) else (str(tuned["ub"]) if tuned else "")
    if ub and ub != "auto":
        if not ub.isdigit():
            say("moe-cache: --ub must be auto or a number")
        else:
            extra += ["-ub", ub, "-b", str(max(int(ub), 2048))]
            say(f"moe-cache: micro-batch -ub {ub} (prompt processing; --ub N to change)")
    # without a GPU cache, take the static expert layers from the same decision as the micro-batch (the planner alone assumes a small compute
    # buffer and fills the card, so the long-prompt GPU block found no room next to -ub 2048: Flash-Next prefill 18 instead of 50+ tok/s)
    no_vram = bool(tuned) and any("no GPU memory information" in w for w in tuned.get("why", []))   # e.g. an integrated GPU: keep the planner's answer
    if gcache == "off" and ncmoe_user is None and ncmoe != "0" and tuned and not no_vram:
        ncmoe = str(tuned["n_cpu_moe"])
    if gcache != "off" and ncmoe != "0":
        gcmib = gcache
        if gcache == "auto":
            gcmib = str(tuned["gpu_cache_mib"]) if tuned and tuned["gpu_cache_mib"] else ""
            for w in (tuned or {}).get("why", []):
                say("moe-cache: tuner: " + w)
        if not gcmib.isdigit():
            say("moe-cache: --gpu-cache: could not work out a size (planner unavailable?), using the planner's static GPU layers"); gcmib = ""
        if gcmib and int(gcmib) < 512:
            say(f"moe-cache: --gpu-cache: only {gcmib} MiB of VRAM would be left, using the planner's static GPU layers"); gcmib = ""
        if gcmib:
            ncmoe = "999"; extra += ["--moe-cache-mib", gcmib]; env.append(("MOE_CACHE_GPU_CACHE", "1"))
            say(f"moe-cache: GPU expert cache of {gcmib} MiB; " + ("llama.cpp keeps the hottest experts on the GPU" if stock else "all experts are managed by the plugin (RAM) and the hottest ones are computed on the GPU"))
    if a.hybrid and ncmoe != "0":
        ncmoe = "999"; env.append(("MOE_CACHE_HYBRID", os.environ.get("MOE_CACHE_HYBRID", "auto")))
        say("moe-cache: --hybrid (experimental): the most used experts are computed on the GPU from a bank, the rest on the CPU in parallel (output no longer identical to stock)")
        if not os.path.isfile(profile):
            say(f"moe-cache: --hybrid: no usage profile yet ({profile}): the bank is chosen from it, so this first session runs without one")
    # short prompts: in the approximate modes, batches under 64 tokens stay on the CPU (the exact default keeps llama.cpp's own threshold, 32)
    if (a.hybrid or a.fast) and not os.environ.get("MOE_CACHE_GPU_MIN_TOKENS"):
        env.append(("MOE_CACHE_GPU_MIN_TOKENS", "64"))
    # llama-server keeps finished prompts in host RAM (--cache-ram, 8 GiB by default), which competes with the expert cache: capped at 2 GiB here
    # (the plugin keeps the same amount free). Pass --cache-ram yourself to change it.
    cram = arg_after(extra, "--cache-ram", "-cram")
    if cram is None:
        try:
            if "--cache-ram" in subprocess.run([ls, "--help"], capture_output=True, text=True, timeout=30).stdout:
                # below 16 GB the reserve is a large share of the expert cache (Qwen3.6-35B at 12 GB: 16.9 tok/s with 2048 MiB, 19.2 with 512, 22.6 with 0;
                # RESULTS 44). The prompt cache only helps when one server switches between conversations; a single chat reuses its slot either way
                lim = mem_for_server(a.ram)
                cram = "0" if lim and lim < 16e9 else "2048"
                extra += ["--cache-ram", cram]
        except Exception:
            pass
    if cram is not None and re.fullmatch(r"[0-9-]+", cram):
        env.append(("MOE_CACHE_HOST_RESERVE_MIB", cram))     # told to the plugin directly instead of it parsing our command line
    if stock:
        say(f"moe-cache: the model ({size / 1e9:.1f} GB) fits in the memory it may use ({avail / 1e9:.1f} GB): running stock llama-server without the plugin, "
            "which is faster when everything fits (RESULTS 44). --always-plugin to use the plugin anyway.")
        env = [(k, v) for k, v in env if not (k.startswith("MOE_CACHE") or k in ("LD_PRELOAD", "GGML_BACKEND_PATH"))]
        method = "none (stock)"
    cmd = [ls, "-m", model, "-ngl", "99", "--n-cpu-moe", ncmoe, "-c", str(ctx), "-fa", "on", "--load-mode", "mmap", "--no-warmup", "--port", str(port)] + pinargs + extra
    if method == "none (stock)":
        pass
    elif ncmoe == "0":
        say(f"moe-cache: all experts fit in your GPU memory; the cache has nothing to manage for this model/context (profile {profile}; plugin loaded by {method})")
    else:
        say(f"moe-cache: experts of {ncmoe} layers stay on the CPU and are managed by the cache; profile {profile}; plugin loaded by {method}")
    say(f"moe-cache: web UI and API will be at http://127.0.0.1:{port}")
    if a.dry_run:
        if WINDOWS:   # PowerShell form, so it can be pasted there
            say("".join(f'$env:{k}="{v}"; ' for k, v in env) + "& " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        else:
            say(("env " + " ".join(f"{k}={v}" for k, v in env) + " " if env else "") + " ".join(cmd))   # stock mode: nothing to set (T7)
        return 0
    api_key = arg_after(extra, "--api-key")
    if a.project and os.path.isfile(os.path.join(slotdir, a.project + ".bin")):
        def restore():
            for _ in range(900):
                if healthy(port, api_key):
                    st, body = http(f"http://127.0.0.1:{port}/slots/0?action=restore", {"filename": a.project + ".bin"}, api_key, timeout=600)
                    if st == 200:
                        say(f"moe-cache: restored the saved prompt state of project {a.project}", err=True)
                    else:
                        say(f"moe-cache: could not restore project {a.project} (HTTP {st}): {body[:200]}", err=True)
                    return
                time.sleep(1)
        threading.Thread(target=restore, daemon=True).start()
    if a.open:
        def opener():
            for _ in range(600):
                if healthy(port, api_key):
                    import webbrowser
                    webbrowser.open(f"http://127.0.0.1:{port}")
                    return
                time.sleep(1)
        threading.Thread(target=opener, daemon=True).start()
    return run_child(cmd, dict(os.environ, **dict(env)), port=port, expect_plugin=method != "none (stock)")


# ---- learn ----

def learn(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py learn", description="build a warm-start profile by running a set of prompts through the model with the plugin; "
                                 "moe-cache-server loads it automatically (same default path)")
    ap.add_argument("model"); ap.add_argument("--name", default=""); ap.add_argument("--ram", default="20"); ap.add_argument("--n-cpu-moe", default="999")
    ap.add_argument("--tokens", type=int, default=150); ap.add_argument("--prompts", default=os.path.join(HERE, "tools", "learn_prompts.txt"))
    ap.add_argument("--out"); ap.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER", "llama-server")); ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--load-method", default="auto", choices=["auto", "preload", "path"])
    a = ap.parse_args(argv)
    if not WINDOWS:
        os.umask(0o077)
    if not os.path.isfile(a.model):
        raise SystemExit("usage: moe-cache-learn MODEL.gguf [options]")
    lib = os.environ.get("MOE_CACHE_LIB") or default_lib()
    if not os.path.isfile(lib):
        raise SystemExit("plugin not built: run ./build.sh first")
    if a.name and not NAME_RE.match(a.name):
        raise SystemExit("--name may only contain letters, digits, _ and -")
    out = a.out or os.path.join(CACHE_DIR, os.path.basename(a.model) + "." + (a.name + "." if a.name else "") + "usage")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if port_busy(a.port):
        raise SystemExit(f"port {a.port} is already in use (another server?): stop it or pass --port, so the prompts go to the right server")
    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    before = os.path.getmtime(out) if os.path.isfile(out) else None
    import tempfile
    logf = tempfile.NamedTemporaryFile(prefix="moe-cache-learn.", suffix=".log", delete=False)
    say(f"learning from {len(prompts)} prompts, {a.tokens} tokens each; profile -> {out}")
    load, _ = load_env(a.llama_server, lib, a.load_method)
    env = dict(os.environ, **dict(load + [("MOE_CACHE_SIZE_GIB", "auto"), ("MOE_CACHE_RAM_GIB", a.ram), ("MOE_CACHE_GGUF", a.model), ("MOE_CACHE_PROFILE", out)]))
    cmd = [a.llama_server, "-m", a.model, "-ngl", "99", "--n-cpu-moe", a.n_cpu_moe, "-c", "4096", "-fa", "on", "--load-mode", "mmap", "--no-warmup",
           "--port", str(a.port), "--seed", "42"]
    p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            if p.poll() is not None:
                raise SystemExit(f"llama-server stopped before it was ready (exit {p.returncode}); log: {logf.name}")
            if healthy(a.port):
                break
            time.sleep(2)
        else:
            raise SystemExit(f"llama-server did not become ready; log: {logf.name}")
        for i, q in enumerate(prompts, 1):
            if p.poll() is not None:
                raise SystemExit(f"llama-server stopped during prompt {i} (exit {p.returncode}); log: {logf.name}")
            st, body = http(f"http://127.0.0.1:{a.port}/completion", {"prompt": q, "n_predict": a.tokens, "temperature": 0, "seed": 42, "cache_prompt": False}, timeout=3000)
            if st != 200:
                raise SystemExit(f"prompt {i}: HTTP {st}: {body[:200]}; log: {logf.name}")
            say(f"  prompt {i}/{len(prompts)} done")
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
    if not os.path.isfile(out) or (before is not None and os.path.getmtime(out) <= before):
        raise SystemExit(f"no new profile was written to {out} (server log: {logf.name})")
    say(f"done: moe-cache-server will warm-start from {out} ({os.path.getsize(out)} bytes; server log: {logf.name})")
    return 0


# ---- project ----

def project(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py project", description="save or restore the prompt state (KV cache of slot 0) of a server started with --project NAME. "
                                 "The state is only valid for the same model file and the same -c / cache type settings.")
    ap.add_argument("cmd", choices=["save", "restore", "list"]); ap.add_argument("name", nargs="?")
    ap.add_argument("--url", default="http://127.0.0.1:8080"); ap.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"))
    a = ap.parse_args(argv)
    d = os.path.join(CACHE_DIR, "slots")
    if a.cmd == "list":
        files = sorted(glob.glob(os.path.join(d, "*", "*.bin")))
        for f in files:
            say(f"{os.path.getsize(f) / 1e6:10.1f} MB  {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(f)))}  {f}")
        if not files:
            say("no saved projects")
        return 0
    if not a.name or not NAME_RE.match(a.name):
        raise SystemExit("usage: moe-cache-project save|restore NAME [--url URL] [--api-key KEY]")
    st, body = http(f"{a.url}/slots/0?action={a.cmd}", {"filename": a.name + ".bin"}, a.api_key, timeout=600)
    say(body.strip())
    if st != 200:
        raise SystemExit(f"{a.cmd} failed: HTTP {st}" if st else f"{a.cmd} failed: no server at {a.url}")
    return 0


# ---- check ----

def check(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py check", description="check that this machine, your llama-server and your model are ready for moe-cache. Changes nothing.")
    ap.add_argument("model", nargs="?"); ap.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER", "llama-server")); ap.add_argument("--ram")
    a = ap.parse_args(argv)
    n = {"fail": 0, "warn": 0}
    def ok(k, m): say(f"  [ OK ] {k:<22} {m}")
    def warn(k, m): say(f"  [WARN] {k:<22} {m}"); n["warn"] += 1
    def fail(k, m): say(f"  [FAIL] {k:<22} {m}"); n["fail"] += 1
    say("moe-cache check")
    import platform
    if sys.platform.startswith("linux"):
        ok("system", "Linux " + platform.release())
    elif WINDOWS:
        warn("system", f"Windows {platform.release()}: experimental (never identity-tested on real Windows yet)")
    else:
        fail("system", f"{platform.system()}: only Linux (and experimental Windows) are supported")
    if platform.machine().lower() in ("x86_64", "amd64"):
        ok("cpu", "x86-64")
    else:
        warn("cpu", f"{platform.machine()}: only x86-64 was tested")
    if os.access("/sys/devices/cpu_core/cpus", os.R_OK):
        ok("cpu pinning", f"hybrid Intel CPU (performance cores {open('/sys/devices/cpu_core/cpus').read().strip()}): threads are pinned to the performance cores automatically (about +15 %)")
    elif not WINDOWS:
        ok("cpu pinning", "not a hybrid Intel CPU: no pinning needed")
    lsp = shutil.which(a.llama_server)
    if not lsp:
        fail("llama-server", f"not found ('{a.llama_server}'); pass --llama-server PATH")
    else:
        try:
            v = subprocess.run([lsp, "--version"], capture_output=True, text=True, timeout=30)
            m = re.search(r"commit [0-9a-f]+", v.stdout + v.stderr)
        except Exception:
            m = None
        ver = m.group(0) if m else ""
        ok("llama-server", f"{lsp} ({ver or 'version unknown'})")
        if re.search(r"7fe450e|23b0202", ver):
            ok("ggml version", f"matches a tested commit ({ver})")
        else:
            warn("ggml version", "tested with 7fe450e19 (API 2) and 23b0202a1 (API 3); other versions usually load but may not (the plugin warns at start)")
        if not WINDOWS:
            if "libggml-base" in ldd(lsp):
                ok("shared ggml", "libggml-base is a shared library")
            else:
                fail("shared ggml", "this llama-server has ggml built in; the plugin cannot be loaded (use a build with BUILD_SHARED_LIBS=ON)")
    lib = os.environ.get("MOE_CACHE_LIB") or (pick_lib(lsp) if lsp else default_lib())
    if not os.path.isfile(lib):
        if shutil.which("g++"):
            warn("plugin", "not built yet: run ./build.sh (g++ found)")
        else:
            fail("plugin", "not built and no g++ found; use the prebuilt release or install g++")
    else:
        ok("plugin", lib)
        if lsp:
            load, method = load_env(lsp, lib)
            probe = ""
            try:
                out = subprocess.run([lsp, "--list-devices"], capture_output=True, text=True, timeout=60, env=dict(os.environ, **dict(load)))
                probe = out.stdout + out.stderr
            except Exception:
                pass
            listed = "MOE_CACHE" in probe
            if listed:
                ok("plugin loads", f"yes, via {method} (loaded and listed; the cache itself first runs with the first request: use tests/identity.sh "
                   "or tools/windows_test.ps1 to see it compute)")
            else:
                why = [l.strip() for l in probe.splitlines() if "load_backend" in l or "moe-cache:" in l][:3]
                fail("plugin loads", f"llama-server did not list the MOE_CACHE device (method tried: {method})" + ("; llama-server said: " + " | ".join(why) if why else "")
                     + ("; on Windows an empty 'failed to load' reason usually means a missing DLL dependency or a DLL built for another ggml" if WINDOWS else ""))
            try:
                out = subprocess.run([lsp, "--list-devices"], capture_output=True, text=True, timeout=60)
                devs = [l.strip() for l in (out.stdout + out.stderr).splitlines() if re.match(r"^ +(CUDA|Vulkan|ROCm|HIP|SYCL|Metal)[0-9]*:", l)]
            except Exception:
                devs = []
            if devs:
                ok("gpu", devs[0])
            else:
                warn("gpu", "no GPU backend in this llama-server: CPU-only mode (works, slower)")
    memav = cg = None
    if WINDOWS:
        t, av = win_memory()
        memav = av / 1e9 if av else None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                memav = int(line.split()[1]) * 1024 / 1e9
        rel = open("/proc/self/cgroup").read().strip().split(":")[-1]
        for f in (f"/sys/fs/cgroup{rel}/memory.max", "/sys/fs/cgroup/memory.max"):
            if os.access(f, os.R_OK):
                v = open(f).read().strip()
                if v != "max":
                    cg = int(v) / 1e9
                    break
    except OSError:
        pass
    ok("memory", (f"--ram {a.ram} GB; " if a.ram else "") + (f"available {memav:.1f} GB" if memav else "available: unknown") + (f", cgroup limit {cg:.1f} GB" if cg else ""))
    try:
        sw = next(int(l.split()[1]) * 1024 / 1e9 for l in open("/proc/meminfo") if l.startswith("SwapTotal"))
        if sw >= 1:
            warn("swap", f"swap is on ({sw:.1f} GB): if memory runs out the system may swap (very slow) instead of killing the server")
    except (OSError, StopIteration):
        pass
    try:
        import gguf  # noqa: F401
        ok("planner", "python gguf package found (the planner also has its own header reader)")
    except ImportError:
        ok("planner", "built-in GGUF header reader (the python gguf package is not needed)")
    d = CACHE_DIR
    while d and not os.path.exists(d) and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    if os.access(d, os.W_OK):
        ok("profile dir", CACHE_DIR + ("" if os.path.isdir(CACHE_DIR) else " (will be created)"))
    else:
        warn("profile dir", "cache directory not writable: profiles will not be saved")
    if a.model:
        if not os.access(a.model, os.R_OK):
            fail("model", f"cannot read '{a.model}'")
        else:
            size = sum(os.path.getsize(p) for p in model_parts(a.model)) / 1e9
            head = open(a.model, "rb").read(4 << 20)
            if head[:4] == b"GGUF":
                ok("model", f"GGUF, {size:.1f} GB")
            else:
                fail("model", "not a GGUF file")
            if b"expert_count" in head:
                ok("MoE model", "has experts (expert_count found)")
            else:
                warn("MoE model", "no expert_count in the header: this may not be a MoE model, moe-cache will not help")
            if not WINDOWS:
                try:
                    os.close(os.open(a.model, os.O_RDONLY | os.O_DIRECT))
                    ok("direct reads", "O_DIRECT works on this file system")
                except OSError:
                    fail("direct reads", "O_DIRECT is not supported here (tmpfs/overlay/network disk?): put the model on a normal disk")
            lim = float(a.ram) if a.ram else (cg or memav)
            if lim and size > lim:
                ok("fit", f"model ({size:.1f} GB) is larger than the memory limit ({lim:.1f} GB): this is where moe-cache helps most")
            elif lim:
                warn("fit", f"model ({size:.1f} GB) fits in {lim:.1f} GB: you get about stock speed, the gain is when memory is short")
    else:
        say("  (give a model path to also check the model: moe-cache-check MODEL.gguf)")
    say(f"summary: {n['fail']} failed, {n['warn']} warnings")
    return 1 if n["fail"] else 0


def main():
    cmds = {"server": server, "check": check, "learn": learn, "project": project}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        say(__doc__.strip())
        return 0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 1
    return cmds[sys.argv[1]](sys.argv[2:]) or 0


if __name__ == "__main__":
    sys.exit(main())
