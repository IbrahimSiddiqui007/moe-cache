"""Hardware readings for the Monitor view and `moe_cache.py top` (GUI-046). Python 3.8+, standard library only, no admin rights.

Every reading is None when this machine cannot give it (no nvidia-smi, an AMD card on Windows, a process that is not ours): the screens print "n/a" and nothing
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
    return None if WINDOWS else amd_sysfs()


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
