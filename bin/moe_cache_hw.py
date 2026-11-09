"""Hardware readings for the Monitor view and `moe_cache.py top` (GUI-046). Python 3.8+, standard library only, no admin rights.

Every reading is None when this machine cannot give it (no nvidia-smi, a GPU temperature or power on Windows, a process that is not ours): the screens print "n/a" and nothing
else changes. Parsers are separate from the readers so tests run them on recorded output.
"""
import glob, os, re, subprocess, time

WINDOWS = os.name == "nt"
NVIDIA_QUERY = "name,utilization.gpu,temperature.gpu,power.draw,power.limit,memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current"


def _num(x):
    try:
        return float(str(x).strip())
    except ValueError:
        return None                                 # "[N/A]", "[Not Supported]", ""


def parse_nvidia(text):
    """nvidia-smi --query-gpu=NVIDIA_QUERY --format=csv,noheader,nounits -> the busiest card's {name, util, temp, power, power_limit, mem_used, mem_total (bytes), pcie_gen, pcie_width, cards}."""
    best, n = None, 0
    for line in (text or "").splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 9:
            continue
        n += 1
        d = {"name": f[0], "util": _num(f[1]), "temp": _num(f[2]), "power": _num(f[3]), "power_limit": _num(f[4]),
             "mem_used": _num(f[5]) * 2**20 if _num(f[5]) is not None else None, "mem_total": _num(f[6]) * 2**20 if _num(f[6]) is not None else None,
             "pcie_gen": _num(f[7]), "pcie_width": _num(f[8])}
        if best is None or (d["util"] or 0) > (best["util"] or 0):
            best = d
    if best:
        best["cards"] = n
    return best


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def amd_sysfs(root="/sys/class/drm"):
    """The AMD card with the most VRAM, from sysfs (Linux, no admin): {name, util, temp, power, power_limit, mem_used, mem_total, pcie_gen, pcie_width}."""
    best = None
    for dev in glob.glob(os.path.join(root, "card[0-9]*", "device")):
        if _read(dev + "/vendor") != "0x1002":
            continue
        tot, used = _num(_read(dev + "/mem_info_vram_total")), _num(_read(dev + "/mem_info_vram_used"))
        if not tot or tot < 2**30:
            continue                                                 # an integrated GPU's small carve-out is not the card
        hw = (glob.glob(dev + "/hwmon/hwmon*") or [None])[0]
        temp = _num(_read(hw + "/temp1_input")) if hw else None
        pw = _num(_read(hw + "/power1_average") or _read(hw + "/power1_input")) if hw else None
        cap = _num(_read(hw + "/power1_cap")) if hw else None
        speed = _read(dev + "/current_link_speed") or ""
        m = re.search(r"([\d.]+)\s*GT/s", speed)
        gen = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}.get(float(m.group(1))) if m else None
        d = {"name": "AMD GPU", "util": _num(_read(dev + "/gpu_busy_percent")), "temp": temp / 1000 if temp is not None else None,
             "power": pw / 1e6 if pw is not None else None, "power_limit": cap / 1e6 if cap else None, "mem_used": used, "mem_total": tot,
             "pcie_gen": gen, "pcie_width": _num(_read(dev + "/current_link_width")), "cards": 1}
        if best is None or tot > best["mem_total"]:
            best = d
    return best


def gpu_stats(timeout=4):
    """The GPU's {name, util %, temp C, power W, power_limit W, mem_used, mem_total, pcie_gen, pcie_width}: NVIDIA through nvidia-smi, AMD on Linux through
    sysfs. None otherwise (AMD or Intel on Windows: nothing readable without admin; the screens say n/a)."""
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader,nounits"], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        if r.returncode == 0:
            d = parse_nvidia(r.stdout)
            if d:
                return d
    except (OSError, subprocess.SubprocessError):
        pass
    if WINDOWS:
        return win_gpu_stats()          # R15-02: any vendor, no admin: PDH counters + DXGI (temperature and power stay None: only vendor libraries give them, R15-07)
    return amd_sysfs()


# ---- Windows GPU and memory counters without admin (R15-02): PDH "GPU Engine" and "GPU Adapter Memory", DXGI adapter list and video-memory budget, commit ----
# Parsers first (tests run them on recorded instance names), the ctypes readers below them (Windows CI runs them for real; the PC check is a later chain item).

_LUID_RE = re.compile(r"luid_(0x[0-9a-fA-F]+)_(0x[0-9a-fA-F]+)")


def luid_key(text):
    """'luid_0x00000000_0x0000A1B2_phys_0' (any PDH instance name with a LUID) -> (high, low) as ints, or None."""
    m = _LUID_RE.search(text or "")
    return (int(m.group(1), 16), int(m.group(2), 16)) if m else None


def parse_gpu_engine(items):
    """[(instance, percent)] of the PDH counter 'GPU Engine(*)\\Utilization Percentage' -> {luid: percent}. The instances are per process and per engine
    ('pid_123_luid_..._phys_0_eng_0_engtype_3D'): per (adapter, engine index, engine type) the values of all processes are added, the adapter's figure is the busiest
    such engine (capped at 100), which is what Task Manager's GPU column shows for its busiest engine."""
    per_engine = {}
    for name, val in items or []:
        k = luid_key(name)
        m = re.search(r"eng_(\d+)_engtype_(\w+)", name or "")
        if not k or not m or val is None:
            continue
        per_engine[(k, m.group(1), m.group(2))] = per_engine.get((k, m.group(1), m.group(2)), 0.0) + float(val)
    out = {}
    for (k, _eng, _typ), v in per_engine.items():
        out[k] = min(100.0, max(out.get(k, 0.0), v))
    return out


def parse_adapter_memory(items):
    """[(instance, bytes)] of '\\GPU Adapter Memory(*)\\Dedicated Usage' or 'Shared Usage' -> {luid: bytes} (the instance names carry the LUID and 'phys_N')."""
    out = {}
    for name, val in items or []:
        k = luid_key(name)
        if k and val is not None:
            out[k] = out.get(k, 0.0) + float(val)
    return out


def pick_adapter(adapters):
    """The adapter the model runs on: the one with the most dedicated video memory (a discrete card), else the first with a budget; software adapters (Microsoft Basic Render)
    only when nothing else exists. adapters: [{name, luid, dedicated, shared, budget, software}]."""
    real = [a for a in adapters or [] if not a.get("software")]
    pool = real or list(adapters or [])
    if not pool:
        return None
    return max(pool, key=lambda a: (a.get("dedicated") or 0, a.get("shared") or 0))


def merge_gpu(adapter, util, dedicated_used, shared_used, usage, n_adapters):
    """The gpu_stats() dict for one adapter. mem_total is the dedicated video memory (an integrated GPU: its shared-system budget), mem_used the PDH 'Dedicated Usage'
    (an integrated GPU: 'Shared Usage'); temp / power / pcie stay None. Extra keys: mem_shared, mem_budget (DXGI), mem_dedicated_total."""
    if not adapter:
        return None
    ded = adapter.get("dedicated") or 0
    integrated = ded < 2**30                         # a carve-out of a few hundred MiB is not the memory the model sees
    total = adapter.get("budget") if integrated and adapter.get("budget") else (ded or adapter.get("budget"))
    used = shared_used if integrated and shared_used is not None else dedicated_used
    if used is None and usage is not None:
        used = usage                                   # DXGI CurrentUsage of this process group when PDH had nothing
    return {"name": adapter.get("name"), "util": util, "temp": None, "power": None, "power_limit": None, "mem_used": used, "mem_total": total,
            "pcie_gen": None, "pcie_width": None, "cards": n_adapters, "mem_shared": shared_used, "mem_budget": adapter.get("budget"), "mem_dedicated_total": ded or None}


class _WinCounters:
    """PDH through ctypes (pdh.dll, English counter names regardless of the UI language): one query kept between samples because rate counters need two collections."""
    def __init__(self, paths):
        import ctypes
        from ctypes import wintypes
        self.ct, self.wt = ctypes, wintypes
        self.pdh = pdh = ctypes.windll.pdh
        self.h = ctypes.c_void_p()
        pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        pdh.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        if pdh.PdhOpenQueryW(None, None, ctypes.byref(self.h)) != 0:
            raise OSError("PdhOpenQuery failed")
        self.counters = {}
        for key, path in paths.items():
            c = ctypes.c_void_p()
            if pdh.PdhAddEnglishCounterW(self.h, path, None, ctypes.byref(c)) == 0:
                self.counters[key] = c                  # a counter the machine does not have (no GPU counters in a VM) is just missing
        self.collect()

    def collect(self):
        return self.pdh.PdhCollectQueryData(self.h) == 0

    def read(self, key):
        """[(instance name, float value)] of one wildcard counter, [] when missing or not yet readable."""
        c = self.counters.get(key)
        if c is None:
            return []
        ct, wt = self.ct, self.wt
        class FmtValue(ct.Structure):
            _fields_ = [("CStatus", wt.DWORD), ("pad", wt.DWORD), ("doubleValue", ct.c_double)]
        class Item(ct.Structure):
            _fields_ = [("szName", wt.LPWSTR), ("FmtValue", FmtValue)]
        size, count = wt.DWORD(0), wt.DWORD(0)
        PDH_FMT_DOUBLE = 0x200
        r = self.pdh.PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE, ct.byref(size), ct.byref(count), None)
        if r != 0x800007D2 or not size.value:                          # PDH_MORE_DATA is the answer to a size query
            return []
        buf = ct.create_string_buffer(size.value)
        if self.pdh.PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE, ct.byref(size), ct.byref(count), buf) != 0:
            return []
        arr = ct.cast(buf, ct.POINTER(Item))
        return [(arr[i].szName, arr[i].FmtValue.doubleValue) for i in range(count.value) if arr[i].FmtValue.CStatus in (0, 1)]

    def close(self):
        try:
            self.pdh.PdhCloseQuery(self.h)
        except Exception:
            pass


def dxgi_adapters():
    """The DXGI adapters: [{name, luid, dedicated, shared, budget, usage, software}] (budget / usage: IDXGIAdapter3::QueryVideoMemoryInfo, local segment; None when the
    interface is missing). ctypes COM calls through the vtable; any failure gives what was read so far (never raises)."""
    out = []
    try:
        import ctypes
        from ctypes import wintypes
        ct = ctypes

        class GUID(ct.Structure):
            _fields_ = [("a", wintypes.DWORD), ("b", wintypes.WORD), ("c", wintypes.WORD), ("d", ct.c_ubyte * 8)]

        def guid(txt):
            h = txt.replace("-", "")
            g = GUID()
            g.a, g.b, g.c = int(h[0:8], 16), int(h[8:12], 16), int(h[12:16], 16)
            for i in range(8):
                g.d[i] = int(h[16 + 2 * i:18 + 2 * i], 16)
            return g

        class Desc1(ct.Structure):
            _fields_ = [("Description", ct.c_wchar * 128), ("VendorId", wintypes.UINT), ("DeviceId", wintypes.UINT), ("SubSysId", wintypes.UINT), ("Revision", wintypes.UINT),
                        ("DedicatedVideoMemory", ct.c_size_t), ("DedicatedSystemMemory", ct.c_size_t), ("SharedSystemMemory", ct.c_size_t),
                        ("LuidLow", wintypes.DWORD), ("LuidHigh", wintypes.LONG), ("Flags", wintypes.UINT)]

        class MemInfo(ct.Structure):
            _fields_ = [("Budget", ct.c_uint64), ("CurrentUsage", ct.c_uint64), ("AvailableForReservation", ct.c_uint64), ("CurrentReservation", ct.c_uint64)]

        def method(obj, index, restype, *argtypes):
            vtbl = ct.cast(ct.cast(obj, ct.POINTER(ct.c_void_p))[0], ct.POINTER(ct.c_void_p))
            return ct.WINFUNCTYPE(restype, ct.c_void_p, *argtypes)(vtbl[index])

        dxgi = ct.windll.dxgi
        dxgi.CreateDXGIFactory1.argtypes = [ct.POINTER(GUID), ct.POINTER(ct.c_void_p)]
        factory = ct.c_void_p()
        if dxgi.CreateDXGIFactory1(ct.byref(guid("770aae78-f26f-4dba-a829-253c83d1b387")), ct.byref(factory)) != 0:
            return out
        try:
            enum_adapters1 = method(factory, 12, ct.c_long, wintypes.UINT, ct.POINTER(ct.c_void_p))
            i = 0
            while i < 16:
                ad = ct.c_void_p()
                if enum_adapters1(factory, i, ct.byref(ad)) != 0:
                    break
                i += 1
                try:
                    d = Desc1()
                    if method(ad, 10, ct.c_long, ct.POINTER(Desc1))(ad, ct.byref(d)) != 0:
                        continue
                    row = {"name": d.Description, "luid": (d.LuidHigh & 0xFFFFFFFF, d.LuidLow), "dedicated": int(d.DedicatedVideoMemory), "shared": int(d.SharedSystemMemory),
                           "budget": None, "usage": None, "software": bool(d.Flags & 2)}           # DXGI_ADAPTER_FLAG_SOFTWARE
                    ad3 = ct.c_void_p()
                    if method(ad, 0, ct.c_long, ct.POINTER(GUID), ct.POINTER(ct.c_void_p))(ad, ct.byref(guid("645967a4-1392-4310-a798-8053ce3e93fd")), ct.byref(ad3)) == 0:
                        try:
                            mi = MemInfo()
                            if method(ad3, 14, ct.c_long, wintypes.UINT, wintypes.UINT, ct.POINTER(MemInfo))(ad3, 0, 0, ct.byref(mi)) == 0:      # node 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL
                                row["budget"], row["usage"] = int(mi.Budget), int(mi.CurrentUsage)
                        finally:
                            method(ad3, 2, ct.c_ulong)(ad3)                                         # Release
                    out.append(row)
                finally:
                    method(ad, 2, ct.c_ulong)(ad)
        finally:
            method(factory, 2, ct.c_ulong)(factory)
    except Exception:
        pass
    return out


def parse_memory_status(total_page, avail_page, total_phys, avail_phys):
    """GlobalMemoryStatusEx fields -> {commit_total, commit_free, commit_used, ram_total, ram_free} (bytes). ullTotalPageFile is the commit limit (RAM + page file)."""
    return {"commit_total": float(total_page), "commit_free": float(avail_page), "commit_used": float(total_page) - float(avail_page), "ram_total": float(total_phys), "ram_free": float(avail_phys)}


def win_memory_status():
    try:
        import ctypes
        from ctypes import wintypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MS)]
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return parse_memory_status(m.ullTotalPageFile, m.ullAvailPageFile, m.ullTotalPhys, m.ullAvailPhys)
    except Exception:
        return None


_WIN_GPU = {}


def win_gpu_stats(counters=None, adapters=None):
    """gpu_stats() on Windows for any vendor: PDH GPU Engine utilisation + GPU Adapter Memory + DXGI. `counters` (an object with collect() and read(key)) and `adapters`
    (a list as dxgi_adapters() gives) are injectable for tests. The first call has no utilisation yet (rate counters need two collections): util None. None when no adapter."""
    try:
        if counters is None:
            if "c" not in _WIN_GPU:
                _WIN_GPU["c"] = _WinCounters({"engine": r"\GPU Engine(*)\Utilization Percentage", "dedicated": r"\GPU Adapter Memory(*)\Dedicated Usage",
                                              "shared": r"\GPU Adapter Memory(*)\Shared Usage"})
            counters = _WIN_GPU["c"]
        if adapters is None:
            if time.time() - _WIN_GPU.get("t", 0) > 30 or "a" not in _WIN_GPU:       # the adapter list and budget change slowly; the PDH numbers every call
                _WIN_GPU["a"], _WIN_GPU["t"] = dxgi_adapters(), time.time()
            adapters = _WIN_GPU["a"]
        ok = counters.collect()
        ad = pick_adapter(adapters)
        if not ad:
            return None
        k = ad["luid"]
        util = parse_gpu_engine(counters.read("engine")).get(k) if ok else None
        ded = parse_adapter_memory(counters.read("dedicated")).get(k)
        sh = parse_adapter_memory(counters.read("shared")).get(k)
        return merge_gpu(ad, util, ded, sh, ad.get("usage"), len([a for a in adapters if not a.get("software")]) or len(adapters))
    except Exception:
        return None


# ---- CPU and disk of the server process (and the whole machine) ----

def parse_proc_stat_total(text):
    """/proc/stat first line -> (busy jiffies, total jiffies)."""
    f = [int(x) for x in text.splitlines()[0].split()[1:9]]
    idle = f[3] + f[4]
    return sum(f) - idle, sum(f)


def parse_pid_stat(text):
    """/proc/PID/stat -> utime + stime in jiffies (the comm field may hold spaces and brackets: split after the last ')')."""
    f = text.rsplit(")", 1)[1].split()
    return int(f[11]) + int(f[12])


def parse_pid_io(text):
    d = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        d[k.strip()] = int(v)
    return d.get("read_bytes", 0), d.get("write_bytes", 0)


def logical_cpus():
    return os.cpu_count() or 1


def physical_cores():
    """Physical cores on Linux (cpuinfo's physical id + core id pairs); None elsewhere."""
    if WINDOWS:
        return None
    try:
        ids = set()
        phys = core = None
        for line in open("/proc/cpuinfo"):
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
            elif not line.strip() and core is not None:
                ids.add((phys, core)); phys = core = None
        return len(ids) or None
    except OSError:
        return None


class _Win:
    """ctypes calls for the Windows readings (kernel32 only, nothing to install)."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ct, self.wt = ctypes, wintypes
        self.k = k = ctypes.windll.kernel32
        k.OpenProcess.restype, k.OpenProcess.argtypes = wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.CloseHandle.argtypes = [wintypes.HANDLE]                           # without these a 64-bit handle can be cut to 32 bits on the way in
        P = ctypes.POINTER(wintypes.FILETIME)
        k.GetSystemTimes.argtypes = [P, P, P]
        k.GetProcessTimes.argtypes = [wintypes.HANDLE, P, P, P, P]

    def filetime(self, ft):
        return (ft.dwHighDateTime << 32 | ft.dwLowDateTime)

    def system_times(self):
        ft = [self.wt.FILETIME() for _ in range(3)]
        if not self.k.GetSystemTimes(*[self.ct.byref(x) for x in ft]):
            return None
        idle, kern, user = (self.filetime(x) for x in ft)
        return kern + user - idle, kern + user          # kernel time includes idle; 100 ns units

    def process_times(self, pid):
        h = self.k.OpenProcess(0x1000, False, pid)      # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            ft = [self.wt.FILETIME() for _ in range(4)]
            if not self.k.GetProcessTimes(h, *[self.ct.byref(x) for x in ft]):
                return None
            return self.filetime(ft[2]) + self.filetime(ft[3])
        finally:
            self.k.CloseHandle(h)

    def process_io(self, pid):
        class IO(self.ct.Structure):
            _fields_ = [(n, self.ct.c_uint64) for n in ("ro", "wo", "oo", "rb", "wb", "ob")]
        h = self.k.OpenProcess(0x1000, False, pid)
        if not h:
            return None
        try:
            io = IO()
            if not self.k.GetProcessIoCounters(h, self.ct.byref(io)):
                return None
            return io.rb, io.wb
        finally:
            self.k.CloseHandle(h)


class HostMeter:
    """CPU use of the machine and of one process, and that process's disk traffic, between two sample() calls.
    sample() -> {cpu_sys (% of all threads), cpu_proc (cores busy, e.g. 3.4), io_r, io_w (MB/s)}; each None when unknown."""
    def __init__(self, pid=None):
        self.pid, self.prev, self.win = pid, None, _Win() if WINDOWS else None

    def read(self):
        t = time.time()
        try:
            if WINDOWS:
                st = self.win.system_times()
                pt = self.win.process_times(self.pid) if self.pid else None
                io = self.win.process_io(self.pid) if self.pid else None
                return t, st, pt, io, 1e7                              # 100 ns ticks per second
            hz = os.sysconf("SC_CLK_TCK")
            st = parse_proc_stat_total(open("/proc/stat").read())
            pt = parse_pid_stat(open(f"/proc/{self.pid}/stat").read()) if self.pid else None
            io = parse_pid_io(open(f"/proc/{self.pid}/io").read()) if self.pid else None
            return t, st, pt, io, float(hz)
        except (OSError, ValueError, IndexError):
            return t, None, None, None, 1.0

    def sample(self):
        cur = self.read()
        prev, self.prev = self.prev, cur
        out = {"cpu_sys": None, "cpu_proc": None, "io_r": None, "io_w": None}
        if not prev:
            return out
        dt = max(cur[0] - prev[0], 1e-3)
        if cur[1] and prev[1] and cur[1][1] > prev[1][1]:
            out["cpu_sys"] = 100.0 * (cur[1][0] - prev[1][0]) / (cur[1][1] - prev[1][1])
        if cur[2] is not None and prev[2] is not None:
            out["cpu_proc"] = max(0.0, (cur[2] - prev[2]) / cur[4] / dt)
        if cur[3] and prev[3]:
            out["io_r"] = max(0, cur[3][0] - prev[3][0]) / dt / 1e6
            out["io_w"] = max(0, cur[3][1] - prev[3][1]) / dt / 1e6
        return out


# ---- which process listens on a port ----

def parse_netstat(text, port):
    """netstat -ano -p TCP output -> the pid listening on 127.0.0.1:port (or any address) or None."""
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 5 and f[0].upper() == "TCP" and f[3].upper() == "LISTENING" and f[1].rsplit(":", 1)[-1] == str(port):
            try:
                return int(f[4])
            except ValueError:
                return None
    return None


def pid_of_port(port):
    """The pid of the process that listens on `port` (this user's processes on Linux; any on Windows through netstat), or None."""
    try:
        if WINDOWS:
            from moe_cache_wintool import win_tool
            r = subprocess.run([win_tool("netstat"), "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=10)
            return parse_netstat(r.stdout, port)
        inodes = set()
        for fn in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                for line in open(fn).read().splitlines()[1:]:
                    f = line.split()
                    if f[3] == "0A" and int(f[1].rsplit(":", 1)[1], 16) == port:       # state LISTEN
                        inodes.add(f[9])
            except OSError:
                pass
        for fd in glob.glob("/proc/[0-9]*/fd/*"):
            try:
                if os.readlink(fd).startswith("socket:[") and os.readlink(fd)[8:-1] in inodes:
                    return int(fd.split("/")[2])
            except OSError:
                continue
    except Exception:
        pass
    return None
