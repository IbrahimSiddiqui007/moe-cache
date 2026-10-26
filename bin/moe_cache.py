#!/usr/bin/env python3
"""moe-cache launcher (Linux and Windows; Python 3.8+, standard library only).

  moe_cache.py server  MODEL.gguf [options] [-- extra llama-server args]   run llama-server with the expert-cache plugin
  moe_cache.py check   [MODEL.gguf] [--llama-server PATH] [--ram GB]       check that this machine, llama-server and the model are ready
  moe_cache.py learn   MODEL.gguf [options]                                build a warm-start profile from a set of prompts
  moe_cache.py project save|restore|list [NAME] [--url URL] [--api-key KEY] save / restore a running server's prompt state
  moe_cache.py connect hermes|opencode [--url URL] [--write]               point Hermes Agent or OpenCode at a running server (start it with --agent)

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


def win_dedicated_vram():
    """{adapter name: dedicated VRAM bytes} from the Windows registry (display adapter class); {} elsewhere or on failure."""
    out = {}
    if not WINDOWS:
        return out
    try:
        import winreg
        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as k:
            for i in range(winreg.QueryInfoKey(k)[0]):
                sub = winreg.EnumKey(k, i)
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(k, sub) as d:
                        name = str(winreg.QueryValueEx(d, "DriverDesc")[0]).strip()
                        mem = winreg.QueryValueEx(d, "HardwareInformation.qwMemorySize")[0]
                        out[name] = mem if isinstance(mem, int) else int.from_bytes(bytes(mem), "little")
                except OSError:
                    pass
    except Exception:
        pass
    return out


def pick_gpu(listing):
    """(device, name, free MiB) of the GPU llama.cpp will use, from `llama-server --list-devices` output; None without a GPU.
    llama.cpp skips an integrated GPU when a discrete one exists, but --list-devices lists both (on a Ryzen 7000 desktop the built-in Radeon
    comes first and reports shared RAM as its memory). Windows: the adapter with the most dedicated VRAM in the registry wins; else the most free memory."""
    devs = re.findall(r"^\s+((?:CUDA|Vulkan|ROCm|HIP|SYCL|Metal)\d*): (.+?) \((\d+) MiB, (\d+) MiB free\)", listing, re.M)
    if not devs:
        m = re.search(r"^\s+((?:CUDA|Vulkan|ROCm|HIP|SYCL|Metal)\d*): (.+)$", listing, re.M)     # a device line without memory figures
        return (m.group(1), m.group(2).strip(), 0) if m else None
    ded = win_dedicated_vram()
    d = max(devs, key=lambda x: (ded.get(x[1].strip(), -1), int(x[3])))
    return d[0], d[1].strip(), int(d[3])


def nvidia_reports():
    """True when nvidia-smi exists and reports at least one GPU's free memory."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and out.stdout.strip()[:1].isdigit()
    except Exception:
        return False

def list_devices(ls):
    try:
        out = subprocess.run([ls, "--list-devices"], capture_output=True, text=True, timeout=60)
        return out.stdout + out.stderr
    except Exception:
        return ""

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


def _kill_with_parent_windows(p, mem_limit=None):
    """Put the child in a job object that kills it when the launcher's handle closes (the launcher exits, is closed or killed).
    Found on the MSI Claw (2026-10-15): without it a killed launcher left llama-server running on its port.
    mem_limit (bytes, task WP3): also cap the job's committed memory, the Windows counterpart of the cgroup limit used on Linux. It counts the plugin's
    cache and llama-server's own allocations; it does not count the file cache behind a memory-mapped model, so stock llama-server is never limited
    by it (compare stock and the plugin on Windows with a model bigger than the real RAM instead). Returns the limit read back, or None."""
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
    if mem_limit:
        info.Basic.LimitFlags |= 0x200       # JOB_OBJECT_LIMIT_JOB_MEMORY
        info.JobMemoryLimit = int(mem_limit)
    ok = job and k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)) and \
        k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(p._handle)))
    if ok:
        _JOB = job
    else:
        say("moe-cache: warning: could not tie llama-server to the launcher (closing this window may leave it running)", err=True)
        return None
    if mem_limit:
        back = EXT()
        if k32.QueryInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(back), ctypes.sizeof(back), None) and back.Basic.LimitFlags & 0x200:
            say(f"moe-cache: Windows job memory limit {back.JobMemoryLimit / 1e9:.1f} GB on llama-server (--ram; the model's file cache is not counted)", err=True)
            return back.JobMemoryLimit
        say("moe-cache: warning: the Windows job memory limit could not be set; the cache still sizes itself to --ram", err=True)
    return None


def _die_with_parent_linux():
    """preexec_fn: the child gets SIGTERM when the launcher dies, even if the launcher is killed with SIGKILL."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
    except Exception:
        pass


def run_child(cmd, env, port=None, expect_plugin=False, mem_limit=None):
    """Run llama-server as a child (so helper threads can keep working), pass Ctrl-C / SIGTERM on and return its exit code.
    The child dies with the launcher. Its log is passed through; when the plugin was meant to load and the server becomes ready without a single
    'moe-cache:' line, a warning says it is running stock (found on the MSI Claw, 2026-10-15: the plugin failed to load and nothing said so)."""
    kw = {"env": env, "stderr": subprocess.PIPE}
    if sys.platform.startswith("linux"):
        kw["preexec_fn"] = _die_with_parent_linux
    p = subprocess.Popen(cmd, **kw)
    if WINDOWS:
        try:
            _kill_with_parent_windows(p, mem_limit)
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
--fast     opt-in approximate routing (MOE_CACHE_ROUTE=skip): a low-ranked expert that is not in RAM is skipped instead of read from the SSD;
           also a q8_0 KV cache (half the KV memory; KL 0.0045 on its own, RESULTS 53).
--fast=2   stronger level for memory much smaller than the model: up to 4 low-ranked picks may be skipped (at least half are kept).
           Flash-Next at 12 GB: 13.7 vs 9.6 tok/s for --fast; KL 0.022 (two 2-bit quants of one model differ by 0.243). No gain at 24 GB
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
--ram GB   memory the server may use (default: the cgroup limit or free RAM). The cache sizes itself to stay below it. On Windows it is also set as a
           job memory limit on llama-server when the plugin runs (the counterpart of a Linux cgroup; it does not count the model's file cache)
--vram GiB GPU memory to plan with (default: what the GPU reports free)
--n-cpu-moe N  layers whose experts stay on the CPU (default: chosen by the planner from free VRAM)
--no-ecores  do not let the expert executor use efficiency cores (hybrid Intel CPUs: up to 4 are used by default, about +10-15 %)
--always-plugin  use the plugin even when the model fits in the memory the server may use (by default stock llama-server runs then: faster, RESULTS 44)
--no-pin     do not pin llama-server's threads (hybrid Intel CPUs: one thread per performance core by default, about +15 % decode)
--agent    for coding agents (Hermes Agent, OpenCode, ...): context 65536 unless --ctx is given (Hermes refuses less than 64000), tool calls on (--jinja), a stable model name
           (--alias, the file name) and, once the server is ready, the base URL and config snippets for Hermes and OpenCode. Prompt reuse across
           turns is llama-server's own: each slot keeps its conversation, and models with recurrent layers (Qwen3-Next, Qwen3.5/3.6) roll back to a
           checkpoint made at the last user message and just before the end of each prompt (at most 8 per slot here; 1 GiB kept free for them and for restored conversations).
           moe_cache.py connect hermes|opencode [--write] sets the agent up for a running server
           Warm sessions (ideas I29, I34): every idle conversation of 2k+ tokens (with its checkpoints) is saved to disk 20 s after it changes,
           one file per slot, and the two longest are restored at the next start (each agent finds its own), so the agent's 5-20k-token preamble is not processed again after a restart (OpenCode on Qwen3.6: first turn
           7,438 tokens in 45.6 s cold; a session that finds the preamble cached starts with 29 new tokens in 0.7 s). --no-warm to turn it off
--no-starter  do not start from the shipped usage profile for this model family when you have no profile yet (profiles/<arch>-<layers>x<experts>.usage)
--api-key KEY  require this key on the API (passed to llama-server; also LLAMA_API_KEY); needed when other machines can reach the server
MOE_CACHE_PYTHON  python with the gguf package for the planner (default: this python)"""


def server(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py server", description="run llama-server with the expert-cache plugin",
                                 epilog=SERVER_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--project"); ap.add_argument("--profile"); ap.add_argument("--ram"); ap.add_argument("--vram")
    ap.add_argument("--ctx", type=int); ap.add_argument("--n-cpu-moe"); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER", "llama-server"))
    ap.add_argument("--load-method", default="auto", choices=["auto", "preload", "path"])
    ap.add_argument("--gpu-cache", default="off"); ap.add_argument("--gpu-margin", type=float); ap.add_argument("--ub", default="auto")
    ap.add_argument("--fast", nargs="?", const="1", choices=["1", "2"]); ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--no-ecores", action="store_true"); ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--open", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--always-plugin", action="store_true")
    ap.add_argument("--agent", action="store_true"); ap.add_argument("--no-warm", action="store_true"); ap.add_argument("--no-starter", action="store_true"); ap.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"))
    extra = []
    if "--" in argv:
        i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    if not WINDOWS:
        os.umask(0o077)       # the usage profile is private to this user
    ls, model, port = a.llama_server, a.model, a.port
    ctx = a.ctx or (65536 if a.agent else 8192)     # agents send 5-20k-token preambles (tools, instructions); Hermes needs at least 64000
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
    gpu = pick_gpu(list_devices(ls)) if WINDOWS else None
    if WINDOWS and not a.vram and not nvidia_reports():
        # the planner reads free VRAM from nvidia-smi or Linux sysfs; on Windows with an AMD or Intel GPU neither answers and it would plan without a GPU
        # (nvidia-smi.exe can stay installed after an NVIDIA card is gone: seen on the desktop PC)
        g = gpu
        if g and g[2] > 0:
            a.vram = f"{g[2] / 1024:.2f}"
            say(f"moe-cache: GPU {g[1]} ({g[0]}): {a.vram} GiB free (llama-server --list-devices; --vram GiB to override)")
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
    if a.api_key and "--api-key" not in extra:
        extra += ["--api-key", a.api_key]
    if (WINDOWS and gpu and gpu[0].startswith("Vulkan") and re.search(r"AMD|Radeon", gpu[1])
            and arg_after(extra, "-ctxcp", "--ctx-checkpoints", "--swa-checkpoints") is None):
        try:
            sys.path.insert(0, os.path.join(HERE, "planner"))
            from gguf_plan import model_key
            if (model_key(model) or "").startswith("qwen4exp"):
                # desktop PC, 2026-10-26: llama-server (stock too) crashes in AMD's Windows Vulkan driver (amdvlk64.dll, access violation) when it
                # saves a context checkpoint at the end of a Flash-Next prompt; --ctx-checkpoints 0 avoids it (Qwen3.6 / 3.5-122B are not affected)
                extra += ["--ctx-checkpoints", "0"]
                say("moe-cache: AMD GPU on Windows + this model: context checkpoints off (--ctx-checkpoints 0; they crash AMD's Vulkan driver here)")
        except Exception:
            pass
    alias = None
    slotdir = None
    if a.agent:
        if not any(x in ("--jinja", "--no-jinja") for x in extra):
            extra.append("--jinja")         # the default in current llama.cpp; explicit for older builds (tool calls need the model's own template)
        if arg_after(extra, "-ctxcp", "--ctx-checkpoints", "--swa-checkpoints") is None:
            # models with recurrent layers keep a copy of that state per checkpoint in host RAM (about 60 MiB on Qwen3.6-35B), outside the cache's budget
            extra += ["--ctx-checkpoints", "8"]
        alias = arg_after(extra, "-a", "--alias")
        if alias is None:
            alias = re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", os.path.basename(model))
            extra += ["--alias", alias]
        if not a.no_warm and not a.project:
            if arg_after(extra, "--slot-save-path") is None:
                slotdir = os.path.join(CACHE_DIR, "slots", "agent"); os.makedirs(slotdir, exist_ok=True)
                extra += ["--slot-save-path", slotdir]
            else:
                slotdir = arg_after(extra, "--slot-save-path")
    host = arg_after(extra, "--host") or ""
    if not (host == "" or host.startswith("127.") or host in ("localhost", "::1")) and "--api-key" not in extra:
        say(f"WARNING: llama-server will listen on '{host}' without --api-key: anyone who can reach this machine can use the model and the web UI. "
            "Add --api-key KEY after the -- or bind to 127.0.0.1.", err=True)
    pname = a.profile or ""
    if pname and not NAME_RE.match(pname):
        raise SystemExit("--profile NAME may only contain letters, digits, _ and -")
    if a.project:
        if not NAME_RE.match(a.project):
            raise SystemExit("--project NAME may only contain letters, digits, _ and -")
        pname = pname or a.project
        slotdir = os.path.join(CACHE_DIR, "slots", a.project); os.makedirs(slotdir, exist_ok=True)
        extra += ["--slot-save-path", slotdir]
    profile = os.path.join(CACHE_DIR, os.path.basename(model) + "." + (pname + "." if pname else "") + "usage")
    if not os.path.isfile(profile) and not a.no_starter and not a.dry_run:
        # idea I18: no profile yet: start from the shipped one for this model family (a learned profile made the first request 2.1-2.4x faster, RESULTS 16)
        try:
            sys.path.insert(0, os.path.join(HERE, "planner"))
            from gguf_plan import model_key
            key = model_key(model)
            starter = os.path.join(HERE, "profiles", key + ".usage") if key else None
            if starter and os.path.isfile(starter):
                os.makedirs(CACHE_DIR, exist_ok=True)
                shutil.copyfile(starter, profile)
                say(f"moe-cache: no usage profile yet: starting from the shipped one for {key} (it learns from your prompts from now on)")
        except Exception as e:
            say(f"moe-cache: could not use a starter profile ({e})", err=True)
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
    # --fast also stores the KV cache as q8_0 (idea I6: KL 0.0045 on its own, RESULTS 53): half the KV memory, which at agent-sized contexts is the
    # VRAM a bigger micro-batch needs. Your own -ctk / -ctv after the -- win.
    kv_q8 = bool(a.fast) and not any(x in ("-ctk", "--cache-type-k", "-ctv", "--cache-type-v") for x in extra)
    if kv_q8:
        extra += ["-ctk", "q8_0", "-ctv", "q8_0"]
    if a.fast:
        env.append(("MOE_CACHE_ROUTE", "skip"))
        if a.fast == "2":
            env.append(("MOE_CACHE_ROUTE_DROP", "4"))
            say("moe-cache: --fast=2: up to 4 low-ranked experts per token that are not in RAM are skipped (for memory much smaller than the model: "
                "Flash-Next at 12 GB +43 %; about 1 top token in 28 differs from stock, KL 0.022; RESULTS 53)")
        else:
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
            tuned = tune(model, ctx, free=free, want_cache=gcache != "off", kv_q8=kv_q8, **kw)
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
        env.append(("MOE_CACHE_HOST_RESERVE_MIB", str(max(int(cram), 0) + (1024 if a.agent else 0))))
    elif a.agent:
        env.append(("MOE_CACHE_HOST_RESERVE_MIB", "1024"))     # told to the plugin directly instead of it parsing our command line
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
    # Windows has no cgroups: --ram becomes a job memory limit on llama-server when the plugin runs (WP3). Stock runs are not limited by it (their
    # model lives in the file cache, which job limits do not count), so it is left off there.
    win_limit = float(a.ram) * 1e9 if (WINDOWS and a.ram and method != "none (stock)") else None
    if a.dry_run:
        if win_limit:
            say(f"moe-cache: (would set a Windows job memory limit of {win_limit / 1e9:.1f} GB on llama-server)")
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
    if a.agent and not a.no_warm and not a.project and slotdir:
        # a saved state is only valid for the same model file, context and KV settings: all of them go into the name
        tag = re.sub(r"[^A-Za-z0-9_-]", "_", f"{alias}-c{ctx}-" + "-".join(x for x in (arg_after(extra, "-ctk", "--cache-type-k"), arg_after(extra, "-ctv", "--cache-type-v")) if x))
        threading.Thread(target=warm_keeper, args=(port, api_key, slotdir, tag.rstrip("-_")), daemon=True).start()
    if a.agent:
        def snippets():
            for _ in range(1800):
                if healthy(port, api_key):
                    say(agent_snippets(port, alias, ctx, api_key, host), err=True)
                    return
                time.sleep(1)
        threading.Thread(target=snippets, daemon=True).start()
    if a.open:
        def opener():
            for _ in range(600):
                if healthy(port, api_key):
                    import webbrowser
                    webbrowser.open(f"http://127.0.0.1:{port}")
                    return
                time.sleep(1)
        threading.Thread(target=opener, daemon=True).start()
    return run_child(cmd, dict(os.environ, **dict(env)), port=port, expect_plugin=method != "none (stock)", mem_limit=win_limit)


def warm_keeper(port, api_key, slotdir, tag, poll=10, idle_polls=2, min_tokens=2048, keep=2):
    """Ideas I29 / I34: at start, restore the saved conversations (one file per slot, at most `keep`, longest first) into the slots they came from; afterwards
    save every idle slot of min_tokens or more (with its context checkpoints) once it has been unchanged for idle_polls polls. Each agent keeps
    its own slot (llama-server picks the slot whose prompt matches best), so Hermes and OpenCode both start warm after a restart. Saves go to a
    temporary name first, so a stop during a save never leaves a broken file. When the shared KV pool fills, llama-server clears idle slots itself."""
    url = f"http://127.0.0.1:{port}"
    for _ in range(3600):
        if healthy(port, api_key):
            break
        time.sleep(1)
    else:
        return
    files = [f for f in glob.glob(os.path.join(slotdir, tag + "-s*.bin")) if re.search(r"-s(\d+)\.bin$", f) and not f.endswith(".new.bin")]
    for f in sorted(files, key=os.path.getsize, reverse=True)[:keep]:
        to = int(re.search(r"-s(\d+)\.bin$", f).group(1))      # back into the slot it came from, so later saves replace this file
        t0 = time.time()
        st, body = http(f"{url}/slots/{to}?action=restore", {"filename": os.path.basename(f)}, api_key, timeout=600)
        try:
            n = json.loads(body).get("n_restored") if st == 200 else None
        except ValueError:
            n = None
        say(f"moe-cache: warm session: restored a saved agent conversation ({n} tokens) into slot {to} in {time.time() - t0:.1f} s" if st == 200
            else f"moe-cache: warm session: could not restore {os.path.basename(f)} (HTTP {st}): {body[:160]}", err=True)
    saved, seen = {}, {}
    while True:
        time.sleep(poll)
        st, body = http(f"{url}/slots", api_key=api_key)
        if st != 200:
            if st == 0 and not healthy(port, api_key):
                return       # the server is gone
            continue
        try:
            slots = json.loads(body)
        except ValueError:
            continue
        if any(x.get("is_processing") for x in slots):
            seen = {}
            continue
        for x in slots:
            if (x.get("n_prompt_tokens") or 0) < min_tokens:
                continue
            sid = x["id"]
            key = (x.get("id_task"), x.get("n_prompt_tokens"))
            seen[(sid, key)] = seen.get((sid, key), 0) + 1
            if saved.get(sid) == key or seen[(sid, key)] < idle_polls:
                continue
            fname, tmp = f"{tag}-s{sid}.bin", f"{tag}-s{sid}.new.bin"
            t0 = time.time()
            st, body = http(f"{url}/slots/{sid}?action=save", {"filename": tmp}, api_key, timeout=600)
            saved[sid] = key
            if st != 200:
                say(f"moe-cache: warm session: save failed (HTTP {st}): {body[:160]}", err=True)
                continue
            try:
                os.replace(os.path.join(slotdir, tmp), os.path.join(slotdir, fname))
                say(f"moe-cache: warm session: saved slot {sid} ({x.get('n_prompt_tokens')} tokens, "
                    f"{os.path.getsize(os.path.join(slotdir, fname)) / 1e6:.0f} MB) in {time.time() - t0:.1f} s", err=True)
            except OSError as e:
                say(f"moe-cache: warm session: save failed ({e})", err=True)


def server_info(url, api_key=None):
    """(model name, context) of a running llama-server, from /v1/models and /props; (None, None) when it cannot be reached."""
    st, body = http(url.rstrip("/") + "/v1/models", api_key=api_key)
    if st != 200:
        return None, None
    try:
        name = json.loads(body)["data"][0]["id"]
    except (ValueError, KeyError, IndexError):
        name = None
    ctx = None
    st, body = http(url.rstrip("/") + "/props", api_key=api_key)
    if st == 200:
        try:
            ctx = int(json.loads(body)["default_generation_settings"]["n_ctx"])
        except (ValueError, KeyError, TypeError):
            pass
    return name, ctx


def connect(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py connect", description="point a coding agent at a running moe-cache server (started with --agent). "
                                 "Prints the config; --write applies it (Hermes through 'hermes config set'; OpenCode by merging opencode.json, old file kept as .bak)")
    ap.add_argument("agent", choices=["hermes", "opencode"]); ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY")); ap.add_argument("--write", action="store_true")
    ap.add_argument("--config", help="opencode.json to merge into (default: ~/.config/opencode/opencode.json)")
    a = ap.parse_args(argv)
    name, ctx = server_info(a.url, a.api_key)
    if not name:
        raise SystemExit(f"no llama-server answers at {a.url} (start one with: moe-cache-server MODEL.gguf --agent)")
    ctx = ctx or 65536
    base = a.url.rstrip("/") + "/v1"
    if a.agent == "hermes":
        if ctx < 64000:
            say(f"moe-cache: warning: the server's context is {ctx}; Hermes Agent refuses less than 64000 (restart with --agent or --ctx 65536)", err=True)
        sets = [("model.provider", "custom"), ("model.base_url", base), ("model.default", name), ("model.context_length", str(ctx)),
                ("model.api_key", a.api_key or "none")]
        if not a.write:
            say(agent_snippets(None, name, ctx, a.api_key, base=base, only="hermes"))
            say("apply with: " + " && ".join(f"hermes config set {k} {v}" for k, v in sets))
            return 0
        h = shutil.which("hermes")
        if not h:
            raise SystemExit("hermes is not on PATH (install: https://github.com/NousResearch/hermes-agent)")
        for k, v in sets:
            r = subprocess.run([h, "config", "set", k, v], capture_output=True, text=True, timeout=60)
            if r.returncode:
                raise SystemExit(f"hermes config set {k} failed: {(r.stderr or r.stdout).strip()[-300:]}")
        say(f"Hermes Agent now uses {name} at {base} (context {ctx}); run: hermes")
        return 0
    cfg = a.config or os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config"), "opencode", "opencode.json")
    entry = {"npm": "@ai-sdk/openai-compatible", "name": "moe-cache", "options": dict({"baseURL": base}, **({"apiKey": a.api_key} if a.api_key else {})),
             "models": {name: {"name": name, "tool_call": True, "limit": {"context": ctx, "output": min(32768, max(4096, ctx // 4))}}}}
    if not a.write:
        say(agent_snippets(None, name, ctx, a.api_key, base=base, only="opencode"))
        say(f"apply with: moe_cache.py connect opencode --write  (merges into {cfg})")
        return 0
    data = {"$schema": "https://opencode.ai/config.json"}
    if os.path.isfile(cfg):
        try:
            data = json.load(open(cfg, encoding="utf-8"))
        except ValueError:
            raise SystemExit(f"{cfg} is not plain JSON (comments?): add the block printed by 'moe_cache.py connect opencode' by hand")
        shutil.copy2(cfg, cfg + ".bak")
    data.setdefault("provider", {})["moe-cache"] = entry
    data["model"] = f"moe-cache/{name}"
    os.makedirs(os.path.dirname(cfg), exist_ok=True)
    with open(cfg + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(cfg + ".tmp", cfg)
    say(f"OpenCode now uses moe-cache/{name} at {base} (context {ctx}); config {cfg}" + (" (old file: .bak)" if os.path.isfile(cfg + ".bak") else ""))
    return 0


def agent_snippets(port, model, ctx, api_key, host="", base=None, only=None):
    """What an agent needs to use this server: the OpenAI-compatible base URL, the model name and the context, as config blocks for Hermes Agent
    (~/.hermes/config.yaml) and OpenCode (opencode.json)."""
    if base is None:
        h = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
        base = f"http://{h}:{port}/v1"
    key = api_key or "none"
    out = min(32768, max(4096, ctx // 4))     # room left for the answer
    head = ["moe-cache: agent mode ready. OpenAI-compatible endpoint:",
            f"  base URL {base}   model {model}   context {ctx}   API key {'(set)' if api_key else 'none'}"]
    hermes = [
        "  Hermes Agent, ~/.hermes/config.yaml:",
        "    model:",
        "      provider: custom",
        f"      base_url: {base}",
        f"      default: {model}",
        f"      context_length: {ctx}",
        f"      api_key: {key}"]
    opencode = [
        "  OpenCode, opencode.json:",
        '    {"$schema": "https://opencode.ai/config.json", "provider": {"moe-cache": {"npm": "@ai-sdk/openai-compatible", "name": "moe-cache",',
        f'      "options": {{"baseURL": "{base}"' + (f', "apiKey": "{api_key}"' if api_key else "") + '},',
        f'      "models": {{"{model}": {{"name": "{model}", "tool_call": true, "limit": {{"context": {ctx}, "output": {out}}}}}}}}}}},',
        f'     "model": "moe-cache/{model}"}}']
    if only == "hermes":
        return "\n".join(hermes)
    if only == "opencode":
        return "\n".join(opencode)
    return "\n".join(head + hermes + opencode + ["  or let the launcher write them: moe_cache.py connect hermes|opencode --write" + (f" --url http://127.0.0.1:{port}" if port and port != 8080 else "")])


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
    ap.add_argument("--run", action="store_true", help="also start llama-server with the plugin and a small cache for one short request (about a minute): "
                    "proves the cache really computes on this machine")
    a = ap.parse_args(argv)
    run_load = None
    n = {"fail": 0, "warn": 0}
    def ok(k, m): say(f"  [ OK ] {k:<22} {m}")
    def warn(k, m): say(f"  [WARN] {k:<22} {m}"); n["warn"] += 1
    def fail(k, m): say(f"  [FAIL] {k:<22} {m}"); n["fail"] += 1
    say("moe-cache check")
    import platform
    if sys.platform.startswith("linux"):
        ok("system", "Linux " + platform.release())
    elif WINDOWS:
        warn("system", f"Windows {platform.release()}: experimental (identity-tested in CI on a Windows runner; use --run to try the cache here)")
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
                run_load = load
                ok("plugin loads", f"yes, via {method}" + ("" if a.run else " (loaded and listed; the cache itself first runs with the first request: "
                   "add --run to try it)"))
            else:
                why = [l.strip() for l in probe.splitlines() if "load_backend" in l or "moe-cache:" in l][:3]
                fail("plugin loads", f"llama-server did not list the MOE_CACHE device (method tried: {method})" + ("; llama-server said: " + " | ".join(why) if why else "")
                     + ("; on Windows an empty 'failed to load' reason usually means a missing DLL dependency or a DLL built for another ggml" if WINDOWS else ""))
            g = pick_gpu(list_devices(lsp))
            if g:
                ok("gpu", f"{g[0]}: {g[1]}" + (f", {g[2]} MiB free" if g[2] else ""))
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
    if a.run:
        if not (a.model and run_load and lsp):
            fail("cache runs", "needs a model, a llama-server and a plugin that loads (see above)")
        else:
            r = run_check(lsp, a.model, run_load)
            (ok if r[0] else fail)("cache runs", r[1])
    say(f"summary: {n['fail']} failed, {n['warn']} warnings")
    return 1 if n["fail"] else 0


def run_check(lsp, model, load, port=8097):
    """Start llama-server with the plugin in cache mode (0.25 GiB cache, experts on the CPU, the rest on the CPU too, small context) for one 8-token
    request (idea I26: the Claw's check said "plugin loads: yes" while cache mode crashed at the first request). (passed, message)."""
    if port_busy(port):
        return False, f"port {port} is in use: stop what runs there and try again"
    env = dict(os.environ, **dict(load), MOE_CACHE_SIZE_GIB="0.25", MOE_CACHE_GGUF=model, MOE_CACHE_STATS="1")
    env.pop("MOE_CACHE_PROFILE", None)
    cmd = [lsp, "-m", model, "-ngl", "0", "--n-cpu-moe", "999", "-c", "512", "--no-warmup", "--port", str(port), "-np", "1"]
    kw = {"env": env, "stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE}
    if sys.platform.startswith("linux"):
        kw["preexec_fn"] = _die_with_parent_linux
    p = subprocess.Popen(cmd, **kw)
    if WINDOWS:
        try:
            _kill_with_parent_windows(p)
        except Exception:
            pass
    lines = []
    threading.Thread(target=lambda: [lines.append(l.decode(errors="replace").rstrip()) for l in iter(p.stderr.readline, b"")], daemon=True).start()
    try:
        ready = False
        for _ in range(600):
            if p.poll() is not None:
                break
            if healthy(port):
                ready = True
                break
            time.sleep(1)
        st, body = (http(f"http://127.0.0.1:{port}/completion", {"prompt": "The capital of France is", "n_predict": 8, "temperature": 0}, timeout=600)
                    if ready else (0, ""))
        time.sleep(1)
        mc = [l for l in lines if "moe-cache:" in l]
        if p.poll() is not None:
            return False, exit_reason(p.returncode, ready) + (f"; last plugin line: {mc[-1]}" if mc else "")
        if not any("cache mode" in l for l in mc):
            return False, "the server ran but the plugin's cache was not on (no 'moe-cache: cache mode' line): " + (mc[-1] if mc else "no plugin lines at all")
        if st != 200:
            return False, f"the request failed (HTTP {st})"
        try:
            text = json.loads(body).get("content", "")
        except ValueError:
            text = ""
        return True, f"yes: one request in cache mode with a 0.25 GiB cache, answer {text.strip()[:40]!r}"
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()


def main():
    cmds = {"server": server, "check": check, "learn": learn, "project": project, "connect": connect}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        say(__doc__.strip())
        return 0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 1
    return cmds[sys.argv[1]](sys.argv[2:]) or 0


if __name__ == "__main__":
    sys.exit(main())
