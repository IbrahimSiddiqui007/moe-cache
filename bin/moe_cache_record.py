#!/usr/bin/env python3
"""moe_cache_record.py: the one benchmark record (docs/BENCHMARK_TEMPLATE.md, docs/bench_record.schema.json). Linux and Windows; standard library only.

  collect   machine, software and model facts as JSON (no run)               python3 bin/moe_cache_record.py collect --model M.gguf --llama-server LS
  monitor   watch a running llama-server (--pid N) or run a command (-- CMD) and sample VRAM, process RAM, free RAM and CPU until it ends
  assemble  build the full record from the facts, the runs a harness measured and the protocol
  validate  check a record against the schema (required fields present, every value a number / text or "not recorded")
  ingest, claim, board, lint, replay   runs, claims and the board of benchmarks/ (bin/moe_cache_claims.py)
  card      print the record as the benchmark card's four detail lines, a table row, a RESULTS section or an issue body (--as env: the working-desktop line)
  env-from-gtm   the environment block (BENCH-1150: RAM and VRAM of the whole machine) from chain's .gtm.jsonl

A value that could not be read is written "not recorded" (NR), never guessed. bench-report (bin/moe_cache_bench.py) uses the Sampler and the record
builder, so its report carries the same fields. The OS-specific readers are small probes (LinuxProbe, WindowsProbe) that tests drive with fake files
and fake command output (tests/bench_record_test.py)."""
import argparse, csv, datetime, glob, hashlib, io, json, os, platform, re, shutil, subprocess, sys, threading, time

NR = "not recorded"
RECORD_SCHEMA = 1
HERE = os.path.dirname(os.path.abspath(__file__))
IS_WIN = os.name == "nt"
SCHEMA_PATH = os.path.join(HERE, "..", "docs", "bench_record.schema.json")
MiB = 1024 * 1024


def nr(v):
    """The value, or "not recorded" for None / empty."""
    return NR if v is None or v == "" or v == [] or v == {} else v


def rnd(x, n=1):
    return NR if x is None else round(x, n)


# ---------------------------------------------------------------- OS access (tests replace RUN and READ)

def _run(cmd, timeout=20, both=False):
    """stdout of a command ('' when it is missing, fails or is too slow); both=True adds stderr (llama-server prints --version there)."""
    try:
        if not shutil.which(cmd[0]):
            return ""
        flags = 0x08000000 if IS_WIN else 0              # CREATE_NO_WINDOW
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, creationflags=flags) if IS_WIN \
            else subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return ((r.stdout or "") + (r.stderr or "") if both else r.stdout) if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


def _read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _glob(pat):
    return sorted(glob.glob(pat))


RUN, READ, GLOB = _run, _read, _glob


def ps_json(script):
    """A PowerShell one-liner whose output is ConvertTo-Json -> Python value (None when PowerShell is missing or it fails)."""
    out = RUN(["powershell", "-NoProfile", "-NonInteractive", "-Command", script + " | ConvertTo-Json -Compress"], timeout=30)
    try:
        return json.loads(out) if out.strip() else None
    except ValueError:
        return None


def as_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


# ---------------------------------------------------------------- parsers (pure functions; tests feed them fake text)

def parse_cpuinfo(text):
    """(model name, physical cores, logical CPUs) from /proc/cpuinfo."""
    name, cores, logical = None, set(), 0
    phys = None
    for line in text.splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "model name" and name is None:
            name = v
        elif k == "processor":
            logical += 1
        elif k == "physical id":
            phys = v
        elif k == "core id":
            cores.add((phys, v))
    return name, (len(cores) or None), (logical or None)


def parse_meminfo(text):
    """{'total', 'available'} in bytes from /proc/meminfo."""
    d = {}
    for line in text.splitlines():
        m = re.match(r"(MemTotal|MemAvailable):\s+(\d+) kB", line)
        if m:
            d[m.group(1)] = int(m.group(2)) * 1024
    return {"total": d.get("MemTotal"), "available": d.get("MemAvailable")}


def parse_dmidecode(text):
    """(type, speed MT/s, module count, module size GiB) from `dmidecode -t memory`; only populated slots count."""
    types, speeds, sizes = [], [], []
    for block in text.split("Memory Device"):
        m = re.search(r"Size:\s+(\d+)\s*(MB|GB)", block)
        if not m:
            continue
        sizes.append(int(m.group(1)) / (1024 if m.group(2) == "MB" else 1))
        t = re.search(r"\n\s*Type:\s+(DDR\d\w*|LPDDR\d\w*)", block)
        if t:
            types.append(t.group(1))
        s = re.search(r"Configured (?:Memory|Clock) Speed:\s+(\d+)", block) or re.search(r"\n\s*Speed:\s+(\d+)\s*(?:MT/s|MHz)", block)
        if s:
            speeds.append(int(s.group(1)))
    return (types[0] if types else None, min(speeds) if speeds else None, len(sizes), (sizes[0] if sizes else None))


WIN_MEM_TYPE = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 30: "LPDDR4", 34: "DDR5", 35: "LPDDR5"}


def parse_nvidia_smi(text):
    """[{'name', 'vram_mib', 'driver'}] from `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`."""
    out = []
    for line in text.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3 and p[1].replace(".", "").isdigit():
            out.append({"name": p[0], "vram_mib": float(p[1]), "driver": p[2]})
    return out


def parse_list_devices(text):
    """[{'dev', 'name', 'total_mib', 'free_mib'}] from `llama-server --list-devices` ("  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15890 MiB free)")."""
    out = []
    listing = text.split("Available devices", 1)[1] if "Available devices" in text else ""
    for line in listing.splitlines():
        m = re.match(r"\s*(\w+?\d*):\s+(.*?)\s+\((\d+) MiB(?:, (\d+) MiB free)?\)", line)
        if m and not m.group(1).upper().startswith("CPU"):
            out.append({"dev": m.group(1), "name": m.group(2), "total_mib": float(m.group(3)), "free_mib": float(m.group(4)) if m.group(4) else None})
    return out


def parse_vulkaninfo(text):
    """driverInfo (Mesa / AMD driver version) of the first GPU from `vulkaninfo --summary`."""
    m = re.search(r"driverInfo\s*=\s*(.+)", text)
    return m.group(1).strip() if m else None


def parse_typeperf(text):
    """{instance: bytes} from `typeperf "<counter>(*)\\..." -sc 1`: the header line names the counter instances, the next CSV line holds the values."""
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    hdr = next((r for r in rows if r and r[0].startswith("(PDH-CSV")), None)
    val = next((r for r in rows if hdr and r is not hdr and len(r) == len(hdr) and re.match(r"\d\d/\d\d/\d{4}", r[0])), None)
    if not hdr or not val:
        return {}
    out = {}
    for name, v in zip(hdr[1:], val[1:]):
        m = re.search(r"\(([^)]*)\)\\", name)
        try:
            out[m.group(1) if m else name] = float(v)
        except ValueError:
            pass
    return out


def parse_llama_log(lines):
    """Facts from llama-server's own log: n_layer, offloaded layers, buffers per device (MiB), threads, flash attention.
    {'n_layer', 'offloaded', 'offload_total', 'buffers': {device: {'model','KV','compute',...}}, 'vram_from_log_mib'}."""
    d = {"n_layer": None, "offloaded": None, "offload_total": None, "buffers": {}, "threads": None, "plugin_layers": None, "plugin_cpu_expert_layers": None}
    for l in lines:
        m = re.search(r"\bn_threads\s*=\s*(\d+)", l)            # 'llama threadpool init, n_threads = 16' (b12000+ prints this at the default verbosity)
        if m and d["threads"] is None:
            d["threads"] = int(m.group(1))
        m = re.search(r"\((\d+) layers x \d+ experts\)", l)       # the plugin's 'profile ... loaded (48 layers x 512 experts)' (stdout/stderr of the server)
        if m and d["plugin_layers"] is None:
            d["plugin_layers"] = int(m.group(1))
        m = re.search(r"experts of (\d+) layers stay on the CPU", l)
        if m and d["plugin_cpu_expert_layers"] is None:
            d["plugin_cpu_expert_layers"] = int(m.group(1))
        m = re.search(r"\bn_layer\s*=\s*(\d+)", l)
        if m and d["n_layer"] is None:
            d["n_layer"] = int(m.group(1))
        m = re.search(r"offloaded (\d+)/(\d+) layers", l)
        if m:
            d["offloaded"], d["offload_total"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"(\S+?)\s+(model|KV|compute|output|RS|Compute) buffer size\s*=\s*([\d.]+) MiB", l)
        if m:
            dev = m.group(1).strip(":")
            d["buffers"].setdefault(dev, {})[m.group(2).lower() if m.group(2) != "KV" else "KV"] = float(m.group(3))
    gpu = [v for k, v in d["buffers"].items() if not re.match(r"(CPU|HOST)", k, re.I)]
    d["vram_from_log_mib"] = round(sum(sum(b.values()) for b in gpu), 1) if gpu else None
    return d


def parse_cmdline(line):
    """The settings a llama-server command line carries: ctx, ub, b, n_cpu_moe, ngl, kv types, threads, flash attention."""
    def flag(*names):
        for n in names:
            m = re.search(r"(?:^|\s)" + re.escape(n) + r"[\s=]+(\S+)", line)
            if m:
                return m.group(1)
        return None
    num = lambda s: int(s) if s is not None and s.isdigit() else s
    mm = re.search(r'(?:^|\s)(?:-m|--model)[\s=]+("([^"]+)"|\S+)', line)
    return {"model": (mm.group(2) or mm.group(1)) if mm else None, "ctx": num(flag("-c", "--ctx-size")), "ub": num(flag("-ub", "--ubatch-size")), "b": num(flag("-b", "--batch-size")),
            "n_cpu_moe": 0 if re.search(r"(?:^|\s)(--cpu-moe|-cmoe)(\s|$)", line) and flag("--n-cpu-moe") is None else num(flag("--n-cpu-moe", "-ncmoe")),
            "all_experts_on_cpu": bool(re.search(r"(?:^|\s)(--cpu-moe|-cmoe)(\s|$)", line)),
            "ngl": num(flag("-ngl", "--n-gpu-layers")), "kv_k": flag("-ctk", "--cache-type-k"), "kv_v": flag("-ctv", "--cache-type-v"),
            "threads": num(flag("-t", "--threads")), "flash_attn": flag("-fa", "--flash-attn"), "mtp": bool(re.search(r"--spec|--mtp|draft", line))}


def env_in_line(line):
    """{NAME: value} for the MOE_CACHE_* / GGML_* / LLAMA_* assignments in a command line printed as 'env A=1 B=2 llama-server ...'."""
    return {m.group(1): m.group(2) for m in re.finditer(r"\b((?:MOE_CACHE|GGML|LLAMA)_[A-Z0-9_]+)=(\S+)", line)}


def parse_powercfg(text):
    m = re.search(r"\(([^)]+)\)\s*$", text.strip().splitlines()[0]) if text.strip() else None
    return m.group(1) if m else None


def parse_filetime(x):
    return x / 1e7                                   # 100 ns -> s


# ---------------------------------------------------------------- probes: one per OS, same small interface

class LinuxProbe:
    os_name = "linux"

    def __init__(self):
        try:
            self.hz = os.sysconf("SC_CLK_TCK")
        except (AttributeError, ValueError, OSError):
            self.hz = 100
        self.ncpu = os.cpu_count() or 1

    # processes
    def tree(self, pid):
        """pid and all its descendants, from /proc/*/stat."""
        kids = {}
        for p in GLOB("/proc/[0-9]*/stat"):
            f = self.parse_stat(READ(p))
            if f:
                kids.setdefault(f["ppid"], []).append(f["pid"])
        out, todo = [], [pid]
        while todo:
            x = todo.pop()
            out.append(x)
            todo += kids.get(x, [])
        return out

    @staticmethod
    def parse_stat(text):
        """{'pid','ppid','cpu_ticks'} from /proc/PID/stat (the command name may hold spaces and parentheses: split after the last ')')."""
        if ")" not in text:
            return None
        head, _, rest = text.rpartition(")")
        f = rest.split()
        try:
            return {"pid": int(head.split()[0]), "ppid": int(f[1]), "cpu_ticks": int(f[11]) + int(f[12])}
        except (ValueError, IndexError):
            return None

    @staticmethod
    def parse_status(text):
        """{'rss','anon','hwm'} bytes from /proc/PID/status."""
        d = {}
        for line in text.splitlines():
            m = re.match(r"(VmRSS|RssAnon|VmHWM):\s+(\d+) kB", line)
            if m:
                d[m.group(1)] = int(m.group(2)) * 1024
        return {"rss": d.get("VmRSS"), "anon": d.get("RssAnon"), "hwm": d.get("VmHWM")}

    def proc(self, pids):
        """Summed over the tree: {'rss','private','cpu_s','hwm'} (bytes, bytes, s, bytes)."""
        rss = anon = hwm = 0
        cpu = 0.0
        seen = False
        for p in pids:
            st = self.parse_status(READ(f"/proc/{p}/status"))
            f = self.parse_stat(READ(f"/proc/{p}/stat"))
            if st["rss"] is not None:
                seen = True
                rss += st["rss"]; anon += st["anon"] or 0; hwm += st["hwm"] or 0
            if f:
                cpu += f["cpu_ticks"] / self.hz
        return {"rss": rss, "private": anon, "cpu_s": cpu, "hwm": hwm} if seen else None

    def sys_cpu(self):
        """(busy seconds, total seconds) of all CPUs since boot, /proc/stat."""
        line = next((l for l in READ("/proc/stat").splitlines() if l.startswith("cpu ")), "")
        v = [int(x) for x in line.split()[1:] if x.isdigit()]
        if len(v) < 5:
            return None
        total = sum(v[:8]); idle = v[3] + v[4]
        return (total - idle) / self.hz, total / self.hz

    def mem(self):
        return parse_meminfo(READ("/proc/meminfo"))

    def commit_free(self):
        """CommitLimit - Committed_AS in bytes (/proc/meminfo); the kernel's own number, advisory when overcommit is on."""
        d = {m.group(1): int(m.group(2)) * 1024 for m in re.finditer(r"(CommitLimit|Committed_AS):\s+(\d+) kB", READ("/proc/meminfo"))}
        return d["CommitLimit"] - d["Committed_AS"] if len(d) == 2 else None

    def uptime_s(self):
        t = READ("/proc/uptime").split()
        return float(t[0]) if t else None

    # GPU memory: (system-wide used MiB, this process tree's MiB or None, source)
    def vram(self, pids):
        smi = RUN(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=10)
        used = [float(x) for x in smi.split() if x.replace(".", "").isdigit()]
        if used:
            apps = RUN(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], timeout=10)
            mine = 0.0; found = False
            for line in apps.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) == 2 and p[0].isdigit() and int(p[0]) in pids and p[1].replace(".", "").isdigit():
                    mine += float(p[1]); found = True
            return max(used), (mine if found else None), "nvidia-smi"
        best = None
        for d in GLOB("/sys/class/drm/card[0-9]*/device"):
            if READ(d + "/vendor").strip() != "0x1002":
                continue
            u = READ(d + "/mem_info_vram_used").strip()
            if u.isdigit():
                best = max(best or 0, int(u) / MiB)
        return (best, None, "amdgpu sysfs mem_info_vram_used") if best is not None else (None, None, None)

    # hardware facts
    def hardware(self, model_path):
        name, cores, logical = parse_cpuinfo(READ("/proc/cpuinfo"))
        h = {"cpu_model": nr(name), "cpu_cores": nr(cores), "cpu_threads": nr(logical)}
        mt, ms, mod, msz = parse_dmidecode(RUN(["dmidecode", "-t", "memory"], timeout=10))
        h.update(ram_gib=rnd((self.mem()["total"] or 0) / 2 ** 30 or None, 1), ram_type=nr(mt), ram_speed_mts=nr(ms))
        h.update(storage=self.storage(model_path), power_plan=self.power_plan(), os=f"{platform.system()} {platform.release()}",
                 os_build=platform.version())
        return h

    def storage(self, path):
        """{'model','interface','drive'}: the disk holding the model file, from findmnt + lsblk."""
        src = RUN(["findmnt", "-no", "SOURCE", "--target", path or "/"], timeout=10).strip()
        src = re.sub(r"\[.*\]$", "", src)                      # btrfs: /dev/nvme0n1p3[/@]
        if not src.startswith("/dev/"):
            return {"model": NR, "interface": NR, "drive": nr(src or None)}
        parent = RUN(["lsblk", "-ndo", "PKNAME", src], timeout=10).strip().splitlines()
        dev = "/dev/" + parent[0].strip() if parent and parent[0].strip() else src
        out = RUN(["lsblk", "-dno", "MODEL,TRAN,ROTA", dev], timeout=10).strip()
        m = re.match(r"(.*?)\s+(\w+)\s+([01])$", out) or re.match(r"(.*?)\s+([01])$", out)
        model = m.group(1).strip() if m else None
        tran = m.group(2) if m and m.lastindex == 3 else None
        rota = m.group(m.lastindex) if m else None
        if tran is None and dev.startswith("/dev/nvme"):
            tran = "nvme"
        return {"model": nr(model), "interface": nr(tran or ("hdd (rotational)" if rota == "1" else None)), "drive": src}

    def power_plan(self):
        p = RUN(["powerprofilesctl", "get"], timeout=5).strip()
        g = READ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").strip()
        return nr(", ".join(x for x in (p and "profile " + p, g and "governor " + g) if x))

    def gpu(self, listing):
        smi = parse_nvidia_smi(RUN(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], timeout=10))
        if smi:
            return {"gpu_model": smi[0]["name"], "vram_mib": smi[0]["vram_mib"], "gpu_driver": smi[0]["driver"]}
        dev = parse_list_devices(listing or "")
        drv = parse_vulkaninfo(RUN(["vulkaninfo", "--summary"], timeout=20))
        if dev:
            return {"gpu_model": dev[0]["name"], "vram_mib": dev[0]["total_mib"], "gpu_driver": nr(drv)}
        return {"gpu_model": NR, "vram_mib": NR, "gpu_driver": NR}

    def temperatures(self):
        t = {}
        for d in GLOB("/sys/class/hwmon/hwmon*"):
            n = READ(d + "/name").strip()
            v = READ(d + "/temp1_input").strip()
            if n in ("k10temp", "coretemp") and v.isdigit():
                t["cpu_c"] = int(v) / 1000
        g = RUN(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], timeout=10).split()
        if g and g[0].isdigit():
            t["gpu_c"] = float(g[0])
        return t


class WindowsProbe:
    """Windows 10/11. Memory and CPU counters through ctypes (kernel32 / psapi: no PowerShell per sample); VRAM through typeperf (the GPU performance counters);
    machine facts through PowerShell Get-CimInstance. The ctypes parts need Windows, so the tests drive the parsers and the typeperf path with fake text."""
    os_name = "windows"

    def __init__(self):
        self.ncpu = os.cpu_count() or 1
        self._vram_adapter_fail = False

    # processes (ctypes; every failure -> None)
    def tree(self, pid):
        try:
            import ctypes
            from ctypes import wintypes
            class PE(ctypes.Structure):
                _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                            ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                            ("szExeFile", ctypes.c_wchar * 260)]
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
            snap = k.CreateToolhelp32Snapshot(2, 0)
            kids, e = {}, PE()
            e.dwSize = ctypes.sizeof(PE)
            ok = k.Process32FirstW(wintypes.HANDLE(snap), ctypes.byref(e))
            while ok:
                kids.setdefault(e.th32ParentProcessID, []).append(e.th32ProcessID)
                ok = k.Process32NextW(wintypes.HANDLE(snap), ctypes.byref(e))
            k.CloseHandle(wintypes.HANDLE(snap))
            out, todo = [], [pid]
            while todo:
                x = todo.pop(); out.append(x); todo += kids.get(x, [])
            return out
        except Exception:
            return [pid]

    def proc(self, pids):
        """Summed over the tree: working set and private bytes (psapi GetProcessMemoryInfo), CPU seconds (GetProcessTimes)."""
        try:
            import ctypes
            from ctypes import wintypes
            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                            ("PrivateUsage", ctypes.c_size_t)]
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            ps = ctypes.WinDLL("psapi", use_last_error=True)
            k.OpenProcess.restype = wintypes.HANDLE
            rss = priv = hwm = 0
            cpu = 0.0
            seen = False
            for p in pids:
                h = k.OpenProcess(0x1000 | 0x0010, False, p)             # QUERY_LIMITED_INFORMATION | VM_READ
                if not h:
                    continue
                c = PMC(); c.cb = ctypes.sizeof(PMC)
                if ps.GetProcessMemoryInfo(wintypes.HANDLE(h), ctypes.byref(c), c.cb):
                    seen = True
                    rss += c.WorkingSetSize; priv += c.PrivateUsage; hwm += c.PeakWorkingSetSize
                ft = [wintypes.FILETIME() for _ in range(4)]
                if k.GetProcessTimes(wintypes.HANDLE(h), *[ctypes.byref(x) for x in ft]):
                    cpu += parse_filetime(((ft[2].dwHighDateTime << 32) | ft[2].dwLowDateTime) + ((ft[3].dwHighDateTime << 32) | ft[3].dwLowDateTime))
                k.CloseHandle(wintypes.HANDLE(h))
            return {"rss": rss, "private": priv, "cpu_s": cpu, "hwm": hwm} if seen else None
        except Exception:
            return None

    def sys_cpu(self):
        try:
            import ctypes
            from ctypes import wintypes
            idle, kern, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
            if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return None
            q = lambda f: parse_filetime((f.dwHighDateTime << 32) | f.dwLowDateTime)
            total = q(kern) + q(user)                                      # kernel time includes idle time
            return (total - q(idle)) / self.ncpu, total / self.ncpu         # per-CPU seconds, like /proc/stat's jiffies divided by the CPU count
        except Exception:
            return None

    def mem(self):
        try:
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                            ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return {"total": m.ullTotalPhys, "available": m.ullAvailPhys}
        except Exception:
            pass
        return {"total": None, "available": None}

    def commit_free(self):
        """Free commit in bytes: ullAvailPageFile of GlobalMemoryStatusEx (commit limit minus commit charge)."""
        try:
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                            ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            return int(m.ullAvailPageFile) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)) else None
        except Exception:
            return None

    def uptime_s(self):
        try:
            import ctypes
            return ctypes.windll.kernel32.GetTickCount64() / 1000.0
        except Exception:
            return None

    def vram(self, pids):
        """(adapter dedicated GPU memory in use MiB: the largest adapter, this process tree's MiB or None, source). typeperf takes about a second per call.
        \\GPU Process Memory(pid_N_*)\\Dedicated Usage is the llama-server's own share; \\GPU Adapter Memory(*)\\Dedicated Usage is the whole card."""
        adapters = parse_typeperf(RUN(["typeperf", r"\GPU Adapter Memory(*)\Dedicated Usage", "-sc", "1"], timeout=20))
        used = max(adapters.values()) / MiB if adapters else None
        # count the process tree on that adapter only: an iGPU next to the card is another luid, and summing both read 2360 MiB for a
        # tree on a card that held 2172 in all (PC, 2026-11-03)
        big = max(adapters, key=adapters.get) if adapters else ""
        luid = (re.search(r"(luid_0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)", big) or [None, None])[1] if big else None
        procs = parse_typeperf(RUN(["typeperf", r"\GPU Process Memory(*)\Dedicated Usage", "-sc", "1"], timeout=20))
        mine = None
        if procs:
            mine = sum(v for k, v in procs.items() if (m := re.match(r"pid_(\d+)_", k)) and int(m.group(1)) in pids
                       and (luid is None or luid.lower() in k.lower())) / MiB
        if used is None and mine is None:
            return None, None, None
        if mine is not None and used is not None and mine > used:
            # Windows counts an allocation shared between processes (desktop surfaces) in each of them: all processes together read 3749 MiB
            # on a card holding 2131 (PC 2026-11-03). llama-server's own buffers are not shared, so this only happens for trees of desktop apps
            return used, used, "typeperf GPU Process Memory (capped at the card's use: shared allocations counted twice)"
        return used, mine, "typeperf GPU Process Memory" if mine is not None else "typeperf GPU Adapter Memory"

    # hardware facts
    def hardware(self, model_path):
        cpu = as_list(ps_json("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors"))
        mem = as_list(ps_json("Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed,ConfiguredClockSpeed,SMBIOSMemoryType"))
        c0 = cpu[0] if cpu else {}
        m0 = mem[0] if mem else {}
        total = self.mem()["total"] or (sum(m.get("Capacity", 0) for m in mem) or None)
        speed = m0.get("ConfiguredClockSpeed") or m0.get("Speed")
        h = {"cpu_model": nr((c0.get("Name") or "").strip()), "cpu_cores": nr(c0.get("NumberOfCores")), "cpu_threads": nr(c0.get("NumberOfLogicalProcessors")),
             "ram_gib": rnd(total / 2 ** 30 if total else None, 1), "ram_type": nr(WIN_MEM_TYPE.get(m0.get("SMBIOSMemoryType"))), "ram_speed_mts": nr(speed)}
        h.update(storage=self.storage(model_path), power_plan=nr(parse_powercfg(RUN(["powercfg", "/getactivescheme"], timeout=10))),
                 os=f"{platform.system()} {platform.release()}", os_build=platform.version())
        return h

    def storage(self, path):
        m = re.match(r"([A-Za-z]):", path or "")
        letter = m.group(1) if m else ""
        if not letter:
            return {"model": NR, "interface": NR, "drive": NR}
        d = ps_json(f"Get-Partition -DriveLetter {letter} | Get-Disk | Select-Object FriendlyName,BusType") or {}
        d = as_list(d)[0] if d else {}
        return {"model": nr(d.get("FriendlyName")), "interface": nr(d.get("BusType")), "drive": letter.upper() + ":"}

    def gpu(self, listing):
        dev = parse_list_devices(listing or "")
        vc = as_list(ps_json("Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion"))
        name = dev[0]["name"] if dev else NR
        drv = next((v.get("DriverVersion") for v in vc if dev and (v.get("Name") or "") and (v["Name"] in name or name in v["Name"])), None) \
            or (vc[0].get("DriverVersion") if vc and not dev else None)
        return {"gpu_model": name, "vram_mib": dev[0]["total_mib"] if dev else NR, "gpu_driver": nr(drv)}

    def temperatures(self):
        return {}                                   # no standard Windows source without a vendor tool or administrator rights


def probe():
    return WindowsProbe() if IS_WIN else LinuxProbe()


# ---------------------------------------------------------------- the environment block (BENCH-1150)
# A published number is real-use: it was measured on a working desktop. The block says what else was running: RAM (total, in use, moe-cache, the rest, the
# five biggest programs), VRAM per process with the spill, free commit. Sampled at the start, at the peak (highest RAM in use) and at the end of a run.
# Source: gputaskman --json (Windows PC: C:\tools\gputaskman\...; laptop: gtm) and our own counters where it is missing; the block names which it used.
# GB in this block are GiB (2^30), the way the launcher prints them.

GIB = 2 ** 30
ENV_EVERY_S = 30.0      # one gputaskman call every 30 s (chain's own sampler did the same); a speed run is never disturbed more often
ENV_TOP = 5


def gb(x):
    return round(x / GIB, 2) if isinstance(x, (int, float)) and not isinstance(x, bool) else NR


def find_gtm():
    """Path of gputaskman: MOE_CACHE_GTM, then `gputaskman` / `gtm` on PATH, then the Windows install folder; None when there is none."""
    p = os.environ.get("MOE_CACHE_GTM")
    if p:
        return shutil.which(p) or (p if os.path.exists(p) else None)
    for n in ("gputaskman", "gtm"):
        w = shutil.which(n)
        if w:
            return w
    if IS_WIN:
        hits = GLOB(r"C:\tools\gputaskman\*\gputaskman.exe")
        return hits[-1] if hits else None
    return None


def read_gtm(exe, timeout=25):
    """One `gputaskman --json` snapshot as a dict, or None (missing, failed, not JSON)."""
    if not exe:
        return None
    try:
        d = json.loads(RUN([exe, "--json"], timeout=timeout) or "null")
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def prog_name(n):
    n = str(n or "?")
    return n[:-4] if n.lower().endswith(".exe") else n


def top_by_name(rows, skip_pids=(), n=ENV_TOP):
    """[(name, bytes)] of the n biggest programs; one program's processes (a browser's helpers) are added up. rows: dicts with pid, name, bytes."""
    tot = {}
    for r in rows:
        if r.get("pid") in skip_pids or not isinstance(r.get("bytes"), (int, float)):
            continue
        k = prog_name(r.get("name"))
        tot[k] = tot.get(k, 0) + r["bytes"]
    return sorted(tot.items(), key=lambda kv: -kv[1])[:n]


def linux_top_ram():
    """Fallback for the program list without gputaskman: every /proc/PID/status (VmRSS) and /proc/PID/comm."""
    rows = []
    for p in GLOB("/proc/[0-9]*/status"):
        pid = int(p.split("/")[2])
        st = LinuxProbe.parse_status(READ(p))
        if st["rss"]:
            rows.append({"pid": pid, "name": READ(f"/proc/{pid}/comm").strip() or "?", "bytes": st["rss"]})
    return rows


def env_snapshot(pr, pids, g=None, t_s=None):
    """The block's fields at this moment. pr: probe, pids: the server's process tree (moe-cache), g: a gputaskman dict or None."""
    pids = set(pids)
    sysd = (g or {}).get("system") or {}
    ram = sysd.get("ram") or {}
    total, avail = ram.get("total"), ram.get("available")
    if total is None or avail is None:
        m = pr.mem()
        total, avail = m.get("total"), m.get("available")
    used = total - avail if total is not None and avail is not None else None
    own = pr.proc(list(pids))
    mine = own["rss"] if own else None                    # working set (Windows) / VmRSS (Linux): private plus the mapped model pages it holds
    rows = [{"pid": r.get("pid"), "name": r.get("name"), "bytes": r.get("rss")} for r in sysd.get("top_ram") or []]
    top_src = "gputaskman"
    if not rows and not g and getattr(pr, "os_name", "") == "linux":
        rows, top_src = linux_top_ram(), "/proc"
    top = top_by_name(rows, pids)
    snap = {"t_s": None if t_s is None else round(t_s, 1), "ram_total_gb": gb(total), "ram_used_total_gb": gb(used), "ram_moe_cache_gb": gb(mine),
            "ram_rest_of_system_gb": gb(max(used - mine, 0)) if used is not None and mine is not None else NR,
            "top_programs": [[n, gb(b)] for n, b in top] if top else NR}
    # VRAM: per process from gputaskman; without it the counters give this server and the card, the rest is "other programs"
    vt, per = NR, NR
    procs = (g or {}).get("processes")
    if procs:
        vt = gb((g or {}).get("vram_total"))
        spill = bool((g or {}).get("shared_is_spill"))
        rows_v = []
        for p in procs:
            b = p.get("vram_bytes") if p.get("vram_bytes") is not None else p.get("vram_inferred_bytes")
            if isinstance(b, (int, float)):
                rows_v.append((p.get("pid"), prog_name(p.get("name")), b, p.get("shared_bytes") if spill and isinstance(p.get("shared_bytes"), (int, float)) else None))
        add = lambda x, y: x if y is None else (y if x is None else x + y)          # None = unknown: stays unknown until some process gives a figure
        mv = sum(b for pid, _, b, _ in rows_v if pid in pids)
        ms = None
        for pid, _, _, sp in rows_v:
            if pid in pids:
                ms = add(ms, sp)
        agg = {}
        for pid, name, b, sp in rows_v:
            if pid in pids:
                continue
            a = agg.setdefault(name, [0, None])
            a[0] += b; a[1] = add(a[1], sp)
        oth = sorted(agg.items(), key=lambda kv: -kv[1][0])[:ENV_TOP]
        per = [["moe-cache", gb(mv), gb(ms)]] + [[n, gb(v[0]), gb(v[1])] for n, v in oth]
    else:
        try:
            sysv, procv, _src = pr.vram(pids)
        except Exception:
            sysv = procv = None
        if procv is not None:
            per = [["moe-cache", round(procv / 1024, 2), NR]] + ([["other programs", round(max(sysv - procv, 0) / 1024, 2), NR]] if sysv is not None else [])
    cf = getattr(pr, "commit_free", None)
    cfree = cf() if cf else None
    snap.update({"vram_total_gb": vt, "vram_per_process": per, "commit_free_gb": gb(cfree)})
    snap["source"] = "gputaskman --json" if g else "own counters (no gputaskman)" + (" + /proc for the programs" if top_src == "/proc" else "")
    return snap


class EnvTrack:
    """Keeps the first, the peak (highest RAM in use) and the last snapshot of a run; add() takes the dicts env_snapshot returns."""

    def __init__(self):
        self.start = self.peak = self.end = None
        self.n = 0

    def add(self, snap):
        self.n += 1
        self.start = self.start or snap
        self.end = snap
        key = lambda x: x["ram_used_total_gb"] if isinstance(x.get("ram_used_total_gb"), (int, float)) else -1
        if self.peak is None or key(snap) > key(self.peak):
            self.peak = snap

    def block(self):
        """The structured block: the peak snapshot's fields on top, `sampled_at` holds all three."""
        if not self.n:
            return NR
        out = {k: v for k, v in self.peak.items() if k != "t_s"}
        out["basis"] = "the snapshot with the highest RAM in use during the run"
        out["sampled_at"] = {"start": self.start, "peak": self.peak, "end": self.end}
        out["samples"] = self.n
        return out


def snapshot_from_gtm_line(d):
    """One line of chain's gtm_sampler.ps1 (.gtm.jsonl) -> the block's fields. A chain item holds several runs: filter the lines by the run's start and end
    time first (env_from_lines)."""
    g = d.get("gputaskman") if isinstance(d.get("gputaskman"), dict) else None
    llama = d.get("llama") or []
    pids = {x.get("pid") for x in llama}
    mine = sum(x.get("ws_gb") or 0 for x in llama)
    total, used = d.get("total_ram_gb"), d.get("used_ram_gb")
    if g and (g.get("system") or {}).get("ram"):
        r = g["system"]["ram"]
        if r.get("total") and r.get("available") is not None:
            total, used = r["total"] / GIB, (r["total"] - r["available"]) / GIB
    rows = [{"pid": x.get("pid"), "name": x.get("name"), "bytes": (x.get("ws_gb") or 0) * GIB} for x in d.get("top_ram") or []]
    if g and (g.get("system") or {}).get("top_ram"):
        rows = [{"pid": x.get("pid"), "name": x.get("name"), "bytes": x.get("rss")} for x in g["system"]["top_ram"]]
    top = top_by_name(rows, pids)
    snap = {"t_s": None, "ram_total_gb": round(total, 2) if isinstance(total, (int, float)) else NR, "ram_used_total_gb": round(used, 2) if isinstance(used, (int, float)) else NR,
            "ram_moe_cache_gb": round(mine, 2) if llama else NR,
            "ram_rest_of_system_gb": round(max(used - mine, 0), 2) if isinstance(used, (int, float)) and llama else NR,
            "top_programs": [[n, gb(b)] for n, b in top] if top else NR, "vram_total_gb": NR, "vram_per_process": NR, "commit_free_gb": NR,
            "source": "gputaskman --json via chain gtm_sampler.ps1" if g else "chain gtm_sampler.ps1 (no gputaskman data in this line)"}
    if g and g.get("processes") is not None:
        s = env_snapshot(_NoProbe(), pids, g)
        snap.update({"vram_total_gb": s["vram_total_gb"], "vram_per_process": s["vram_per_process"]})
    return snap


class _NoProbe:
    os_name = "none"

    def mem(self): return {"total": None, "available": None}

    def proc(self, pids): return None

    def vram(self, pids): return None, None, None


def env_from_lines(lines, start=None, end=None):
    """The block from the lines of a chain .gtm.jsonl (text lines). start / end: 'YYYY-MM-DDTHH:MM:SS' bounds of one run (inclusive); None = all lines."""
    tr = EnvTrack()
    for ln in lines:
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        t = str(d.get("t") or "")
        if (start and t < start) or (end and t > end):
            continue
        tr.add(snapshot_from_gtm_line(d))
    return tr.block()


def env_line(env, label="working desktop", server="moe-cache"):
    """One compact line for a published number, or None when the block is missing:
    'working desktop (at peak): 31 GB RAM, moe-cache 9.0 GB, other programs 8.4 GB (Claude 0.8, Dia 0.4...), GPU: moe-cache 10.9 GB, desktop 1.8 GB'."""
    if not isinstance(env, dict) or not isinstance(env.get("ram_total_gb"), (int, float)):
        return None
    f = lambda x: f"{x:.1f} GB" if isinstance(x, (int, float)) else NR
    parts = [f"{env['ram_total_gb']:.0f} GB RAM"]
    if isinstance(env.get("ram_moe_cache_gb"), (int, float)):
        parts.append(f"{server} {f(env['ram_moe_cache_gb'])}")
    if isinstance(env.get("ram_rest_of_system_gb"), (int, float)):
        t = env.get("top_programs")
        names = ""
        if isinstance(t, list) and t:
            shown = ", ".join(f"{n} {v:.1f}" if isinstance(v, (int, float)) else f"{n}" for n, v in t[:3])
            names = f" ({shown}{'...' if len(t) > 3 or env['ram_rest_of_system_gb'] > sum(v for _, v in t[:3] if isinstance(v, (int, float))) + 0.05 else ''})"
        parts.append(f"other programs {f(env['ram_rest_of_system_gb'])}{names}")
    v = env.get("vram_per_process")
    if isinstance(v, list) and v:
        mc = next((x for x in v if x[0] == "moe-cache"), None)
        rest = sum(x[1] for x in v if x[0] != "moe-cache" and isinstance(x[1], (int, float)))
        g = []
        if mc and isinstance(mc[1], (int, float)):
            g.append(f"{server} {f(mc[1])}" + (f" ({mc[2]:.1f} GB spilled)" if len(mc) > 2 and isinstance(mc[2], (int, float)) and mc[2] >= 0.05 else ""))
        if len(v) > (1 if mc else 0):
            g.append(f"desktop {f(rest)}")
        if g:
            parts.append("GPU: " + ", ".join(g))
    return f"{label} (at peak): " + ", ".join(parts)


def env_of(rec, arm=None):
    """(block, number of runs that have one) for an arm of a record: the run with the most RAM in use at its peak, because a number is only as clean as its worst run."""
    best, n = None, 0
    for r in rec.get("runs", []):
        e = r.get("environment")
        if (arm is None or r.get("arm") == arm) and r.get("ok", True) and isinstance(e, dict):
            n += 1
            if best is None or (e.get("ram_used_total_gb") or 0) > (best.get("ram_used_total_gb") or 0):
                best = e
    return best, n


def env_line_for(rec, arm=None):
    """The line for one arm: the stock arm's server is plain llama-server, so the line says that instead of 'moe-cache'."""
    e, n = env_of(rec, arm)
    line = env_line(e, server="stock llama-server" if str(arm or "").lower().startswith("stock") else "moe-cache")
    return line + (f" [the busiest of {n} runs]" if line and n > 1 else "") if line else None


# ---------------------------------------------------------------- Sampler

class Sampler(threading.Thread):
    """Polls a process tree while a run goes on. Peaks: process RAM (working set / RSS, which includes the mapped model file) and private RAM (anonymous
    memory / private bytes), the lowest free system RAM, VRAM (process share when the OS tells it, else the whole card), CPU % of all logical CPUs as
    an average over the run and as the highest interval, for the process tree and for the whole system. stop() returns the metrics dict."""

    def __init__(self, pid, pr=None, interval=0.5, vram_interval=2.0, log_lines=None, env_every=ENV_EVERY_S, gtm=None):
        super().__init__(daemon=True)
        self.pid, self.pr, self.interval, self.vram_interval = pid, pr or probe(), interval, vram_interval
        self.log_lines = log_lines if log_lines is not None else []
        self._stop_ev = threading.Event()
        self.rss_peak = self.priv_peak = 0
        self.free_min = None
        self.vram_proc_peak = self.vram_sys_peak = None
        self.vram_source = None
        self.proc_cpu, self.sys_cpu = [], []
        self.samples = 0
        self.t0 = self.t1 = None
        self.cpu_total_s = None
        self.vram_before = None
        self.env_every = env_every                       # BENCH-1150: the environment block (RAM and VRAM of the whole machine) at start, peak and end; None = off
        self.gtm = gtm if gtm is not None else (find_gtm() if env_every else None)
        self.env = EnvTrack()

    def _env_loop(self):
        # its own thread too: a gputaskman call (or the typeperf fallback) takes 0.1-2 s
        while not self._stop_ev.is_set():
            self.env_sample()
            self._stop_ev.wait(self.env_every)

    def env_sample(self):
        try:
            self.env.add(env_snapshot(self.pr, set(self.pr.tree(self.pid)), read_gtm(self.gtm) if self.gtm else None, time.time() - (self.t0 or time.time())))
        except Exception:
            pass                                          # a failed read leaves a gap, never ends the run

    def _vram_loop(self):
        # its own thread: on Windows a typeperf call takes 1-2 s and stalled the RAM and CPU samples (one sample in 2 s on the CI runner, 2026-11-03)
        while not self._stop_ev.is_set():
            try:
                self.sample_vram(set(self.pr.tree(self.pid)))
            except Exception:
                pass
            self._stop_ev.wait(self.vram_interval)

    def run(self):
        box = [None]                                    # (time, proc cpu s, sys busy s, sys total s) of the previous sample
        self.t0 = time.time()
        vt = threading.Thread(target=self._vram_loop, daemon=True)
        vt.start()
        et = threading.Thread(target=self._env_loop, daemon=True) if self.env_every else None
        if et:
            et.start()
        while not self._stop_ev.is_set():
            try:
                self.sample_once(box, False)
            except Exception:                           # one unreadable sample must not end the sampling (a process may exit between two reads)
                pass
            self._stop_ev.wait(self.interval)
        vt.join(timeout=25)
        if et:
            et.join(timeout=25)
            self.env_sample()                              # the END sample, after the run's last request
        self.t1 = time.time()

    def sample_once(self, last_box, do_vram=True):
        pr = self.pr
        now = time.time()
        pids = pr.tree(self.pid)
        p = pr.proc(pids)
        if p:
            self.samples += 1
            self.rss_peak = max(self.rss_peak, p["rss"], p.get("hwm") or 0)
            self.priv_peak = max(self.priv_peak, p["private"])
        av = pr.mem()["available"]
        if av is not None:
            self.free_min = av if self.free_min is None else min(self.free_min, av)
        s = pr.sys_cpu()
        last = last_box[0]
        if last and p and s:
            dt = now - last[0]
            if dt > 0:
                self.proc_cpu.append(max(0.0, 100.0 * (p["cpu_s"] - last[1]) / dt / pr.ncpu))
                db = s[1] - last[3]
                if db > 0:
                    self.sys_cpu.append(max(0.0, min(100.0, 100.0 * (s[0] - last[2]) / db)))
        if p:
            self.cpu_total_s = p["cpu_s"]
        if p and s:
            last_box[0] = (now, p["cpu_s"], s[0], s[1])
        elif s:
            last_box[0] = (now, 0.0, s[0], s[1])
        if do_vram:
            self.sample_vram(set(pids))

    def sample_vram(self, pids):
        sysv, procv, src = self.pr.vram(pids)
        if sysv is not None:
            self.vram_sys_peak = max(self.vram_sys_peak or 0, sysv)
            self.vram_source = src
        if procv is not None:
            self.vram_proc_peak = max(self.vram_proc_peak or 0, procv)

    def stop(self):
        self._stop_ev.set()
        self.join(timeout=60 if self.env_every else 30)       # the end snapshot comes after the loops end: a slow gputaskman must not cut it off
        return self.result()

    def result(self):
        avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else NR
        top = lambda xs: round(max(xs), 1) if xs else NR
        have_proc = self.vram_proc_peak is not None
        vram = self.vram_proc_peak if have_proc else self.vram_sys_peak
        src = (self.vram_source + (" (this server)" if have_proc else " (whole card, includes other programs)")) if self.vram_source and vram is not None else None
        if vram is None:
            lg = parse_llama_log(self.log_lines).get("vram_from_log_mib")
            vram, src = lg, ("llama-server log buffer sizes (model + KV + compute; not a measured peak)" if lg is not None else None)
        return {"vram_used_peak_mib": rnd(vram, 0), "vram_source": nr(src), "vram_system_peak_mib": rnd(self.vram_sys_peak, 0),
                "process_ram_peak_mib": rnd(self.rss_peak / MiB if self.samples else None, 0),
                "process_private_peak_mib": rnd(self.priv_peak / MiB if self.samples else None, 0),
                "system_ram_free_min_mib": rnd(self.free_min / MiB if self.free_min is not None else None, 0),
                "cpu_avg_pct": avg(self.proc_cpu), "cpu_peak_pct": top(self.proc_cpu), "system_cpu_avg_pct": avg(self.sys_cpu), "system_cpu_peak_pct": top(self.sys_cpu),
                "samples": self.samples, **({"environment": self.env.block()} if self.env.n else {})}


# ---------------------------------------------------------------- the record

def sha256_head(path, n=16 * MiB):
    """sha256 of the first n bytes (a model file is tens of GB: the prefix plus the size identifies it; --hash-model hashes the whole file)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(n))
        return h.hexdigest()
    except OSError:
        return None


def sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * MiB), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def git_commit(root):
    """Short commit of the checkout at `root`; in an unpacked kit (no .git) the kit's REVISION file, else the 'built from commit XXXX' line of README_WINDOWS.txt."""
    out = RUN(["git", "-C", root, "rev-parse", "--short=9", "HEAD"], timeout=10).strip()
    if out:
        return out
    for rel, rx in (("REVISION", r"^\s*([0-9a-f]{7,40})\s*$"), ("README_WINDOWS.txt", r"built from commit ([0-9a-f]{7,40})")):
        m = re.search(rx, READ(os.path.join(root, rel)) or "", re.M)
        if m:
            return m.group(1)
    return None


def llama_build(version_line):
    """(build number, commit) from `llama-server --version` text: 'version: 0.8.0-dev (build 12064, commit 51fe42c5d)' (b12000 and later) or the old
    'version: 11966 (ff0d396e3)'."""
    m = re.search(r"\(build\s+(\d+),\s*commit\s+([0-9a-f]{6,})\)", version_line or "") or re.search(r"version:\s*(\d+)\s*\(([0-9a-f]{6,})\)", version_line or "")
    return (int(m.group(1)), m.group(2)) if m else (None, None)


def collect_environment(model=None, llama_server=None, listing=None, version_line=None, backend=None, pr=None, lib=None, hash_model=False, model_url=None, root=None,
                        version=None, identity=None):
    """Hardware, Software and Model sections for this machine. `listing` / `version_line` / `backend` are llama-server's `--list-devices` and `--version` text and
    the backend name (bench-report already has them); when they are None and llama_server is given they are read here."""
    pr = pr or probe()
    root = root or os.path.abspath(os.path.join(HERE, ".."))
    if llama_server and listing is None:
        listing = RUN([llama_server, "--list-devices"], timeout=60) + ""
    if llama_server and version_line is None:
        out = RUN([llama_server, "--version"], timeout=60, both=True)
        version_line = next((l for l in out.splitlines() if "version" in l), "")
    hw = pr.hardware(model)
    hw.update(pr.gpu(listing))
    if version is None:
        m = re.search(r'^VERSION\s*=\s*"([^"]+)"', READ(os.path.join(HERE, "moe_cache.py")), re.M)
        version = m.group(1) if m else None
    if lib is None:
        for pat in (os.path.join(HERE, "build", "libggml-moe-cache.so"), os.path.join(HERE, "build", "win", "ggml-moe-cache.dll"),
                    os.path.join(HERE, "ggml-moe-cache*.dll"), os.path.join(HERE, "..", "ggml-moe-cache*.dll"), os.path.join(HERE, "build-*", "libggml-moe-cache.so")):
            hit = [p for p in GLOB(pat) if os.path.isfile(p)]
            if hit:
                lib = max(hit, key=os.path.getmtime)
                break
    lib_hash = sha256_file(lib) if lib else None
    build, commit = llama_build(version_line)
    sw = {"moe_cache_version": nr(version), "moe_cache_commit": nr(git_commit(root)), "plugin_file": os.path.basename(lib) if lib else NR,
          "plugin_sha256_prefix": lib_hash[:12] if lib_hash else NR, "llama_build": nr(build), "llama_commit": nr(commit),
          "llama_backend": nr(backend), "python": platform.python_version()}
    md = {"file": NR, "quant": NR, "size_gb": NR, "sha256_head16MiB": NR, "sha256_full": NR, "source_url": nr(model_url)}
    if model and os.path.isfile(model):
        parts = [model]
        m = re.search(r"-\d{5}-of-(\d{5})\.gguf$", model)
        if m:
            parts = sorted(glob.glob(glob.escape(model[:m.start()]) + "-[0-9][0-9][0-9][0-9][0-9]-of-" + m.group(1) + ".gguf")) or parts
        head = sha256_head(model)
        md.update(file=os.path.basename(model), size_gb=round(sum(os.path.getsize(p) for p in parts) / 1e9, 2), sha256_head16MiB=nr(head and head[:16]),
                  quant=nr(identity.get("quant") if identity else _file_quant(model)))
        if hash_model:
            full = sha256_file(model)
            md["sha256_full"] = nr(full and full[:16])
    return {"hardware": hw, "software": sw, "model": md}


def _file_quant(model):
    """The quant of a GGUF: its general.file_type name, else the type holding most of the expert bytes (Q2_0 files carry no file_type name). None if unreadable."""
    try:
        sys.path.insert(0, HERE)
        import moe_cache_model as mm
        m = mm.identity(model)
        return m.file_quant or m.expert_quant or None
    except Exception:
        return None


def _gguf_layers(model):
    """Block count (n_layer) from the GGUF header of `model`, or None where the file is not here / not readable (the PC that ran the server has it)."""
    try:
        sys.path.insert(0, HERE)
        import moe_cache_model as mm
        return mm.identity(model).n_layers or None
    except Exception:
        return None


KV_DEFAULT = "f16 (llama.cpp default: no -ctk/-ctv on the command line)"


def settings_from(cmdline, env=None, log_lines=None, extra=None, model=None):
    """The Settings section: the full launcher command line, every MOE_CACHE_* / GGML_* / LLAMA_* variable (from `env` and from an 'env A=1 ...' prefix of the
    line), ctx / ub / b / n-cpu-moe / GPU expert layers / KV type / flags, the server log's layer counts. Every value comes from a fact: the command line, the
    server log (stderr and stdout lines together), or the model's own GGUF header (`model`, default: the -m path of the command line). `fact_sources` says which.
    A value none of them gives stays "not recorded"."""
    cmdline = cmdline or ""
    p = parse_cmdline(cmdline)
    lg = parse_llama_log(log_lines or [])
    envs = {k: v for k, v in sorted((env or os.environ).items()) if re.match(r"(MOE_CACHE|GGML|LLAMA)_", k) and k != "LLAMA_API_KEY"}
    envs.update(env_in_line(cmdline))
    src = {}
    n_layer = lg["n_layer"]
    if n_layer:
        src["n_layer"] = "server log (n_layer)"
    elif lg["offload_total"]:
        n_layer, src["n_layer"] = lg["offload_total"] - 1, "server log (offloaded layers - 1)"
    elif lg["plugin_layers"]:
        n_layer, src["n_layer"] = lg["plugin_layers"], "plugin log line (layers x experts)"
    else:
        mp = model or p["model"]
        n_layer = _gguf_layers(mp) if mp and os.path.isfile(mp) else None
        if n_layer:
            src["n_layer"] = "GGUF header block_count of " + os.path.basename(mp)
    n_cpu_moe = p["n_cpu_moe"]
    if n_cpu_moe is None and lg["plugin_cpu_expert_layers"] is not None:
        n_cpu_moe, src["n_cpu_moe"] = lg["plugin_cpu_expert_layers"], "plugin log line (experts of N layers stay on the CPU)"
    gpu_exp = NR
    if isinstance(n_cpu_moe, int) and n_layer:
        gpu_exp = max(0, n_layer - n_cpu_moe) if not p["all_experts_on_cpu"] else 0
        src["gpu_expert_layers"] = "n_layer - n_cpu_moe"
    kv_k, kv_v = p["kv_k"], p["kv_v"]
    if cmdline.strip():                      # a real command line without -ctk / -ctv runs llama.cpp's default cache type
        if kv_k is None:
            kv_k, src["kv_type_k"] = KV_DEFAULT, "llama.cpp default (not on the command line)"
        if kv_v is None:
            kv_v, src["kv_type_v"] = KV_DEFAULT, "llama.cpp default (not on the command line)"
    threads = p["threads"]
    if threads is None and lg["threads"]:
        threads, src["threads"] = lg["threads"], "server log (llama threadpool init, n_threads)"
    s = {"command_line": nr(cmdline), "environment": envs or {}, "ctx": nr(p["ctx"]), "ubatch": nr(p["ub"]), "batch": nr(p["b"]), "n_cpu_moe": nr(n_cpu_moe),
         "n_layer": nr(n_layer), "gpu_expert_layers": gpu_exp, "ngl": nr(p["ngl"]), "kv_type_k": nr(kv_k), "kv_type_v": nr(kv_v),
         "flash_attn": nr(p["flash_attn"]), "mtp": "yes" if p["mtp"] else "no", "fast_flag": "yes" if "--fast" in cmdline else "no", "threads": nr(threads)}
    if src:
        s["fact_sources"] = src
    s.update(extra or {})
    return s


def run_entry(arm, index, r, mon=None):
    """One run's Metrics: `r` is what a harness measured (bench-report's run_one keys or the schema's own names), `mon` the Sampler's result."""
    g = lambda *ks: next((r[k] for k in ks if r.get(k) is not None), None)
    e = {"arm": arm, "index": index, "ok": not r.get("error"),
         "prompt_tps": rnd(g("prompt_tps", "prefill"), 2), "decode_tps": rnd(g("decode_tps", "decode"), 2), "ttft_s": rnd(g("ttft_s"), 2),
         "wall_s": rnd(g("wall_s"), 1), "load_s": rnd(g("load_s"), 1), "prompt_tokens": nr(g("prompt_tokens")), "cache_hit_pct": rnd(g("cache_hit_pct", "hit_rate"), 1),
         "disk_read_gb": rnd(g("disk_read_gb", "gb_read"), 2), "gpu_blocks": nr(g("gpu_blocks")), "error": r.get("error") or ""}
    for k in ("vram_used_peak_mib", "vram_source", "vram_system_peak_mib", "process_ram_peak_mib", "process_private_peak_mib", "system_ram_free_min_mib",
              "cpu_avg_pct", "cpu_peak_pct", "system_cpu_avg_pct", "system_cpu_peak_pct"):
        e[k] = (mon or r).get(k, NR) if (mon or r).get(k) is not None else NR
    env = (mon or r).get("environment")
    if isinstance(env, dict):
        e["environment"] = env                              # BENCH-1150: optional, additive (schema 1 stays valid)
    t = r.get("text")
    e["text_sha256"] = sha256_text(t)[:16] if isinstance(t, str) and t else nr(r.get("text_sha256"))
    return e


def summarize_runs(runs):
    """Mean per arm of the numeric metrics (runs that failed are left out)."""
    out = {}
    for arm in sorted({r["arm"] for r in runs}):
        rs = [r for r in runs if r["arm"] == arm and r["ok"]]
        d = {"runs": len(rs)}
        for k in ("prompt_tps", "decode_tps", "ttft_s", "vram_used_peak_mib", "process_ram_peak_mib", "system_ram_free_min_mib", "cpu_avg_pct", "cache_hit_pct"):
            xs = [r[k] for r in rs if isinstance(r[k], (int, float))]
            d[k] = round(sum(xs) / len(xs), 2) if xs else NR
        d["cpu_peak_pct"] = max([r["cpu_peak_pct"] for r in rs if isinstance(r["cpu_peak_pct"], (int, float))], default=NR)
        out[arm] = d
    return out


def outputs_from(runs, baseline_arm="stock"):
    """Text hash per arm, and whether every other arm's text is identical to the baseline's ('yes' / 'no' / 'not checked')."""
    per = {}
    for r in runs:
        if r["ok"] and r["text_sha256"] != NR:
            per.setdefault(r["arm"], set()).add(r["text_sha256"])
    ident = "not checked"
    if baseline_arm in per and len(per) > 1:
        ident = "yes" if all(v == per[baseline_arm] and len(v) == 1 for v in per.values()) else "no"
    return {"text_sha256_by_arm": {a: sorted(v) for a, v in per.items()} or NR, "identical_to_baseline": ident, "baseline_arm": baseline_arm}


def assemble(env, settings, conditions, protocol, runs, reproduce=None, notes=None, now=None):
    benv = env_of({"runs": runs})[0]
    if benv and isinstance(conditions, dict):
        conditions = {**conditions, "environment": benv}   # the busiest run's block: what the number's "other programs" line is built from
    rec = {"record_schema": RECORD_SCHEMA, "created_utc": (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%d %H:%M"),
           "hardware": env["hardware"], "software": env["software"], "model": env["model"], "settings": settings, "conditions": conditions, "protocol": protocol,
           "runs": runs, "summary": summarize_runs(runs), "outputs": outputs_from(runs), "reproduce": reproduce or {}, "notes": notes or []}
    return rec


def conditions_now(pr=None, cold=None, baseline=None, other_programs=None, temps=True):
    """Conditions at the start: uptime, free RAM and VRAM in use, CPU busy over one second, temperatures. cold: True / False / None; baseline / other_programs: text the
    person or harness gives (a baseline guard result like 'PASS 49.4 tok/s')."""
    pr = pr or probe()
    c0 = pr.sys_cpu(); time.sleep(1.0); c1 = pr.sys_cpu()
    busy = round(100 * (c1[0] - c0[0]) / (c1[1] - c0[1]), 1) if c0 and c1 and c1[1] > c0[1] else NR
    up = pr.uptime_s()
    av = pr.mem()["available"]
    vsys, _, _ = pr.vram(set())
    t = pr.temperatures() if temps else {}
    return {"uptime_h": rnd(up / 3600 if up else None, 1), "baseline_check": nr(baseline), "other_programs": nr(other_programs),
            "system_cpu_busy_before_pct": busy, "system_ram_free_before_mib": rnd(av / MiB if av else None, 0), "vram_in_use_before_mib": rnd(vsys, 0),
            "file_cache": {True: "cold", False: "warm"}.get(cold, NR), "cpu_temp_c": t.get("cpu_c", NR), "gpu_temp_c": t.get("gpu_c", NR)}


# ---------------------------------------------------------------- renderers

def _gib(mib):
    return f"{mib / 1024:.1f} GiB" if isinstance(mib, (int, float)) else NR


def details_lines(rec, section=None, date=None, arm="moe-cache"):
    """The benchmark card's four detail lines (docs/benchmark_card.html, per-row `d`), from the record."""
    s, sw, p, c = rec["settings"], rec["software"], rec["protocol"], rec["conditions"]
    runs = [r for r in rec["runs"] if r["arm"] == arm and r["ok"]] or [r for r in rec["runs"] if r["ok"]]
    pick = lambda k: max([r[k] for r in runs if isinstance(r.get(k), (int, float))], default=NR)
    avg = lambda k: (lambda xs: round(sum(xs) / len(xs), 1) if xs else NR)([r[k] for r in runs if isinstance(r.get(k), (int, float))])
    n_runs = max([sum(1 for r in rec["runs"] if r["arm"] == a) for a in {r["arm"] for r in rec["runs"]}] or [0])
    pct = lambda x: f"{x} %" if isinstance(x, (int, float)) else NR
    plen = p.get("prompt_tokens", NR)
    d = rec["created_utc"][5:10]
    return [f"GPU expert layers: {s['gpu_expert_layers']} (n-cpu-moe {s['n_cpu_moe']}) · -ub {s['ubatch']}",
            f"ctx {s['ctx']} · prompt {plen} tokens · {n_runs} runs each, order {p.get('order', NR)} · {c['file_cache']} file cache",
            f"VRAM peak {_gib(pick('vram_used_peak_mib'))} · process RAM peak {_gib(pick('process_ram_peak_mib'))} (system free min {_gib(pick('system_ram_free_min_mib'))}) · "
            f"CPU {pct(avg('cpu_avg_pct'))} avg, {pct(pick('cpu_peak_pct'))} peak",
            f"launcher {sw['moe_cache_commit']}, DLL {sw['plugin_sha256_prefix'][:7] if sw['plugin_sha256_prefix'] != NR else NR} · llama.cpp {sw['llama_build']} "
            f"({sw['llama_commit']}, {sw['llama_backend']}) · {date or d}" + (f" · §{section}" if section else "")] + \
           ([env_line_for(rec, arm) or env_line_for(rec)] if env_line_for(rec, arm) or env_line_for(rec) else [])


def card_row(rec):
    """One markdown table row for a RESULTS section: model, arms' decode and prompt tok/s, ratio, identity."""
    m, sm = rec["model"], rec["summary"]
    arms = sorted(sm)
    cells = [f"{a}: {sm[a]['decode_tps']} / {sm[a]['prompt_tps']}" for a in arms]
    ratio = NR
    if "stock" in sm and "moe-cache" in sm and all(isinstance(sm[a]["decode_tps"], (int, float)) for a in ("stock", "moe-cache")) and sm["stock"]["decode_tps"]:
        ratio = f"{sm['moe-cache']['decode_tps'] / sm['stock']['decode_tps']:.2f}x"
    return f"| {m['file']} ({m['size_gb']} GB) | " + " | ".join(cells) + f" | {ratio} | {rec['outputs']['identical_to_baseline']} |"


RESULTS_TEMPLATE = """## Result N. TITLE

| Date | Machine | Model + quant | Kit / commit | llama.cpp build | Settings | Clean |
|---|---|---|---|---|---|---|
| {date} | {machine_short} | {model_short} | {kit} | {build} | {settings_short} | {clean} |

{machine}. Model: {model}. Order {order}, {runs} runs per arm, first / second shown. Baseline check: {baseline}. Up {uptime} h.

| Case | {arms} |
|---|---|
{rows}

Details (the benchmark card's lines): {details}

Settings: `{cmd}`. Environment: {env}.
Conditions: {cond}. Text identical to baseline: {ident}. Record: {record_file}.
{envline}"""


def results_section(rec, title="TITLE", number="N", record_file="records/NAME.json"):
    h, c, s, pr = rec["hardware"], rec["conditions"], rec["settings"], rec["protocol"]
    arms = sorted(rec["summary"])
    rows = "\n".join(f"| {a} | decode " + " / ".join(str(r["decode_tps"]) for r in rec["runs"] if r["arm"] == a) + "; prompt " +
                     " / ".join(str(r["prompt_tps"]) for r in rec["runs"] if r["arm"] == a) + " |" for a in arms)
    machine = f"{h['os']}, {h['cpu_model']}, {h['gpu_model']} {_gib(h['vram_mib'])}, {h['ram_gib']} GB {h['ram_type']}, model on {h['storage']['drive']} ({h['storage']['model']}, {h['storage']['interface']})"
    sw, st, md = rec["software"], rec["settings"], rec["model"]
    txt = RESULTS_TEMPLATE.format(date=str(rec["created_utc"])[:10], machine_short=f"{h['os']}, {h['cpu_model']}, {h['gpu_model']}".replace("|", "/"),
                                  model_short=f"{md['file']} ({md['quant']})", kit=f"{sw['moe_cache_version']} ({sw['moe_cache_commit']})",
                                  build=f"{sw['llama_build']} ({sw['llama_commit']})", settings_short=f"ctx {st['ctx']}, ub {st['ubatch']}, {pr.get('prompt_tokens', NR)}-token prompt",
                                  clean=f"other programs: {c['other_programs']}; baseline: {c['baseline_check']}", machine=machine, model=f"{rec['model']['file']} ({rec['model']['quant']}, {rec['model']['size_gb']} GB)", order=pr.get("order", NR),
                                  runs=pr.get("runs_per_arm", NR), baseline=c["baseline_check"], uptime=c["uptime_h"], arms="results", rows=rows,
                                  details=" | ".join(details_lines(rec)), cmd=s["command_line"], env=json.dumps(s["environment"]) if s["environment"] else "none",
                                  cond=f"other programs {c['other_programs']}, file cache {c['file_cache']}, CPU temp {c['cpu_temp_c']}, GPU temp {c['gpu_temp_c']}",
                                  ident=rec["outputs"]["identical_to_baseline"], record_file=record_file,
                                  envline=(lambda l: f"Environment: {l}.\n" if l else "")(env_line_for(rec)))
    return txt.replace("## Result N.", f"## Result {number}.").replace("TITLE", title, 1)


def issue_body(rec):
    """What a person pastes into the benchmark issue: the card lines, then the whole record in a collapsed block."""
    return "\n".join(["### Details", ""] + ["- " + l for l in details_lines(rec)] + ["", card_row(rec), "", "<details><summary>Record (JSON)</summary>", "", "```json",
                                                                                        json.dumps(rec, indent=1, ensure_ascii=False), "```", "", "</details>"]) + "\n"


# ---------------------------------------------------------------- validation (a small JSON-Schema subset: type, enum, const, required, properties, items, anyOf, $ref)

def load_schema(path=None):
    """The schema file (docs/bench_record.schema.json), or None when this install has no docs folder."""
    try:
        with open(path or SCHEMA_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def validate(value, schema, root=None, path="$"):
    """List of error strings (empty = valid)."""
    root = root or schema
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return validate(value, node, root, path)
    errs = []
    if "anyOf" in schema:
        if not any(not validate(value, s, root, path) for s in schema["anyOf"]):
            return [f"{path}: {json.dumps(value)[:60]} matches none of the allowed forms"]
        return []
    if "const" in schema and value != schema["const"]:
        return [f"{path}: expected {schema['const']!r}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path}: {value!r} is not one of {schema['enum']}"]
    t = schema.get("type")
    if t:
        ok = {"string": isinstance(value, str), "number": isinstance(value, (int, float)) and not isinstance(value, bool), "integer": isinstance(value, int) and not isinstance(value, bool),
              "boolean": isinstance(value, bool), "object": isinstance(value, dict), "array": isinstance(value, list), "null": value is None}
        if not any(ok[x] for x in (t if isinstance(t, list) else [t])):
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}.{k}: required field missing (write \"{NR}\" if it was not recorded)")
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                errs += validate(value[k], sub, root, f"{path}.{k}")
        ap = schema.get("additionalProperties")
        if isinstance(ap, dict):
            for k, v in value.items():
                if k not in schema.get("properties", {}):
                    errs += validate(v, ap, root, f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errs += validate(v, schema["items"], root, f"{path}[{i}]")
    return errs


def not_recorded_fields(rec, path="$"):
    """Paths of the fields that say "not recorded" (the report lists them so a reader sees the gaps)."""
    out = []
    if isinstance(rec, dict):
        for k, v in rec.items():
            out += not_recorded_fields(v, f"{path}.{k}")
    elif isinstance(rec, list):
        for i, v in enumerate(rec):
            out += not_recorded_fields(v, f"{path}[{i}]")
    elif rec == NR:
        out.append(path)
    return out


# ---------------------------------------------------------------- command line

def _log_tail(proc, lines):
    for l in iter(proc.stdout.readline, b""):
        lines.append(l.decode(errors="replace").rstrip())


def cmd_monitor(a):
    lines = []
    child = None
    if a.command:
        child = subprocess.Popen(a.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        threading.Thread(target=_log_tail, args=(child, lines), daemon=True).start()
        pid = child.pid
    else:
        pid = a.pid
    smp = Sampler(pid, interval=a.interval, log_lines=lines)
    smp.start()
    t0 = time.time()
    try:
        while True:
            if child is not None and child.poll() is not None:
                break
            if child is None and not smp.pr.proc([pid]):
                break
            if a.seconds and time.time() - t0 >= a.seconds:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    res = smp.stop()
    res["wall_s"] = round(time.time() - t0, 1)
    res["log"] = parse_llama_log(lines)
    if child is not None and child.poll() is None:
        child.terminate()
    out = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(out)
    print(out)
    return 0


def cmd_assemble(a):
    env = json.load(open(a.env, encoding="utf-8")) if a.env else collect_environment(a.model, a.llama_server, backend=a.backend, model_url=a.model_url, hash_model=a.hash_model)
    raw = json.load(open(a.runs, encoding="utf-8"))
    runs = []
    counts = {}
    for r in raw:
        arm = r.get("arm", "moe-cache")
        counts[arm] = counts.get(arm, 0) + 1
        mon = r.get("monitor")
        if isinstance(mon, str):
            mon = json.load(open(mon, encoding="utf-8"))
        runs.append(run_entry(arm, r.get("index", counts[arm]), r, mon))
    prompt = open(a.prompt_file, encoding="utf-8").read() if a.prompt_file else None
    proto = {"arms": sorted(counts), "order": a.order, "runs_per_arm": max(counts.values()) if counts else NR, "prompt_set": nr(a.prompt_set),
             "prompt_sha256": sha256_text(prompt)[:16] if prompt else NR, "prompt_tokens": nr(a.prompt_tokens), "n_predict": nr(a.n_predict), "temperature": a.temperature,
             "seed": a.seed, "sampler": nr(a.sampler)}
    st = settings_from(a.command_line, log_lines=None)
    cond = conditions_now(cold={"cold": True, "warm": False}.get(a.cache), baseline=a.baseline, other_programs=a.other_programs)
    rec = assemble(env, st, cond, proto, runs, reproduce={"commands": [a.command_line] if a.command_line else [], "kit": nr(a.kit_url), "model_url": nr(a.model_url)})
    sch = load_schema()
    errs = validate(rec, sch) if sch else []
    for e in errs:
        print("record: " + e, file=sys.stderr)
    text = json.dumps(rec, indent=1, ensure_ascii=False)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
    else:
        print(text)
    return 1 if errs else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] and (argv[0] in ("ingest", "ingest-issue", "claim", "board", "lint", "replay") or argv[0] == "--dir"):          # benchmarks/: runs, claims, board
        import moe_cache_claims
        return moe_cache_claims.main(argv)
    ap = argparse.ArgumentParser(prog="moe_cache_record.py", description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="machine, software and model facts as JSON")
    c.add_argument("--model"); c.add_argument("--llama-server"); c.add_argument("--backend"); c.add_argument("--model-url"); c.add_argument("--hash-model", action="store_true"); c.add_argument("--out")
    m = sub.add_parser("monitor", help="sample a running llama-server (--pid) or a command (-- CMD)")
    m.add_argument("--pid", type=int); m.add_argument("--seconds", type=float, default=0); m.add_argument("--interval", type=float, default=0.5); m.add_argument("--out")
    m.add_argument("command", nargs=argparse.REMAINDER)
    s = sub.add_parser("assemble", help="build the record from runs a harness measured")
    s.add_argument("--runs", required=True, help="JSON list of runs: arm, decode_tps, prompt_tps, ttft_s, text or text_sha256, monitor (file or dict) ...")
    s.add_argument("--env", help="a `collect` file (default: collect now)"); s.add_argument("--model"); s.add_argument("--llama-server"); s.add_argument("--backend")
    s.add_argument("--model-url"); s.add_argument("--hash-model", action="store_true"); s.add_argument("--command-line", default="", help="the FULL launcher command line, with its env prefix")
    s.add_argument("--order", default="ABBA", help="ABBA or single"); s.add_argument("--prompt-set"); s.add_argument("--prompt-file"); s.add_argument("--prompt-tokens", type=int)
    s.add_argument("--n-predict", type=int); s.add_argument("--temperature", type=float, default=0.0); s.add_argument("--seed", type=int, default=42); s.add_argument("--sampler")
    s.add_argument("--cache", choices=["cold", "warm"]); s.add_argument("--baseline"); s.add_argument("--other-programs"); s.add_argument("--kit-url"); s.add_argument("--out")
    v = sub.add_parser("validate", help="check a record against docs/bench_record.schema.json"); v.add_argument("record")
    r = sub.add_parser("card", help="print a record as card lines / row / RESULTS section / issue body"); r.add_argument("record")
    r.add_argument("--as", dest="fmt", choices=["details", "row", "results", "issue", "gaps", "env"], default="details"); r.add_argument("--section"); r.add_argument("--title", default="TITLE")
    e = sub.add_parser("env-from-gtm", help="the environment block (BENCH-1150) from chain's .gtm.jsonl lines, optionally one run's time window")
    e.add_argument("file"); e.add_argument("--start"); e.add_argument("--end"); e.add_argument("--line", action="store_true", help="print the one-line text instead of the JSON")
    a = ap.parse_args(argv)
    if a.cmd == "env-from-gtm":
        blk = env_from_lines(READ(a.file).splitlines(), a.start, a.end)
        print((env_line(blk) or NR) if a.line else json.dumps(blk, indent=1, ensure_ascii=False))
        return 0 if blk != NR else 1
    if a.cmd == "collect":
        env = collect_environment(a.model, a.llama_server, backend=a.backend, model_url=a.model_url, hash_model=a.hash_model)
        text = json.dumps(env, indent=1, ensure_ascii=False)
        (open(a.out, "w", encoding="utf-8").write(text) if a.out else print(text))
        return 0
    if a.cmd == "monitor":
        if a.command and a.command[0] == "--":
            a.command = a.command[1:]
        if not a.command and not a.pid:
            ap.error("monitor needs --pid N or a command after --")
        return cmd_monitor(a)
    if a.cmd == "assemble":
        return cmd_assemble(a)
    rec = json.load(open(a.record, encoding="utf-8"))
    if a.cmd == "validate":
        sch = load_schema()
        if not sch:
            print("validate: docs/bench_record.schema.json not found next to this install"); return 2
        errs = validate(rec, sch)
        for e in errs:
            print("FAIL " + e)
        gaps = not_recorded_fields(rec)
        print(f"{'valid' if not errs else 'INVALID'}: {len(errs)} problems, {len(gaps)} fields say \"{NR}\"")
        return 1 if errs else 0
    if a.fmt == "gaps":
        print("\n".join(not_recorded_fields(rec)))
    elif a.fmt == "env":
        print(env_line_for(rec) or NR)
    elif a.fmt == "row":
        print(card_row(rec))
    elif a.fmt == "results":
        print(results_section(rec, a.title, a.section or "N"))
    elif a.fmt == "issue":
        print(issue_body(rec))
    else:
        print("\n".join(details_lines(rec, a.section)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
