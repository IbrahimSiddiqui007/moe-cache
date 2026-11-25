#!/usr/bin/env python3
"""ram_cap_decision_test: the real decision function bin/moe_cache.ram_cap_decision / avail_reserve_gib (no stubs of the function itself; only the machine's probes are fixed numbers).
The plugin's RAM is sized from the RAM available at start minus a reserve for the OS and the user's programs, on Windows and Linux; a model file bigger than the cache keeps
extra back for its memory-mapped pages. The user's --ram, MOE_CACHE_RAM_GIB, stock, and a Linux cgroup limit are untouched. Numbers: the PC on 2026-11-17 (31.1 GiB, 17.4 GiB
available with the owner's programs open, Flash-Next IQ3_XXS 70.6 GB file). Built, not measured: the stream term is an assumption until the PC re-run."""
import os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
sys.path.insert(0, os.path.join(HERE, "..", "planner"))
import moe_cache as mc  # noqa: E402

GIB = 2 ** 30
FAILS = []


def check(c, w):
    print(("  ok   " if c else "  FAIL ") + w)
    if not c:
        FAILS.append(w)


D = mc.ram_cap_decision
TOT = 31.1 * GIB
# 1. the PC case, a model that does not fit: 17.4 - 4.0 - 2.0 stream = 11.4
d = D(17.4 * GIB, TOT, windows=True, model_bytes=70.6e9)
check(d and abs(d["gib"] - 11.4) < 0.01 and d["stream"] == 2.0 and "memory-mapped" in d["why"] and "Windows" in d["why"], f"PC, file bigger than the cache: 17.4 - 4 - 2 = 11.4 GiB ({d and d['gib']})")
# 2. a model that fits next to the reserve: no stream term
d = D(30 * GIB, TOT, windows=True, model_bytes=10e9)
check(d and abs(d["gib"] - 26.0) < 0.01 and d["stream"] == 0 and "memory-mapped" not in d["why"], "small model: no stream term")
# 3. Linux uses MemAvailable and says so
d = D(17.4 * GIB, TOT, windows=False, model_bytes=70.6e9)
check(d and abs(d["gib"] - 11.4) < 0.01 and "Linux MemAvailable" in d["why"], "Linux: same arithmetic, source named")
# 4. reserve: max(4 GiB, 12.5 %) of the RAM, so 12.5 % on a 64 GiB machine and at least the owner's example max(3 GB, 10 %)
check(mc.ram_keep_bytes(31.1 * GIB) == 4 * GIB and mc.ram_keep_bytes(64 * GIB) == 8 * GIB, "reserve 4 GiB at 31 GiB, 8 GiB at 64 GiB")
check(all(mc.ram_keep_bytes(t * GIB) >= max(3e9, 0.10 * t * GIB) for t in (8, 16, 31.1, 64, 128)), "never below the brief's max(3 GB, 10 %)")
# 5. untouched cases
check(D(17.4 * GIB, TOT, windows=True, ram_flag="10") is None, "--ram wins")
check(D(17.4 * GIB, TOT, windows=True, env_ram="9") is None, "MOE_CACHE_RAM_GIB wins")
check(D(17.4 * GIB, TOT, windows=True, stock=True) is None, "stock: nothing set")
check(D(17.4 * GIB, TOT, windows=False, cgroup_limit=20 * GIB) is None, "Linux cgroup limit: the plugin reads it itself")
check(D(17.4 * GIB, TOT, windows=True, cgroup_limit=20 * GIB) is not None, "a cgroup figure means nothing on Windows")
check(D(None, TOT, windows=True) is None and D(17.4 * GIB, None, windows=True) is None, "unknown figures: nothing")
# 6. little RAM: floor 4 GiB (the plugin then floors its own cache at 1 GiB); never above what is available
d = D(5 * GIB, TOT, windows=True, model_bytes=70e9)
check(d is None or d["gib"] * GIB < 5 * GIB, "5 GiB available: never plans more than is available")
d = D(6 * GIB, TOT, windows=True, model_bytes=70e9)
check(d and d["gib"] == 4.0, "6 GiB available: the 4 GiB floor")
# 7. through avail_reserve_gib with a real file on disk and fixed probes
class M(mc.Machine):
    def __init__(self, avail, windows, limit=None): self.avail, self._w, self.limit = avail, windows, limit
    @property
    def windows(self): return self._w
    def mem_for_server(self, ram_gb): return self.avail
    def mem_available(self): return self.avail
    def mem_total(self): return TOT
    def cgroup_limit(self): return self.limit
a, _, _ = mc.parse_server(["m.gguf", "--no-pin"])
with tempfile.TemporaryDirectory() as td:
    big = os.path.join(td, "m.gguf")
    with open(big, "wb") as f:
        f.truncate(30 * GIB)             # sparse: 30 GiB file, bigger than 17.4 - 4
    saved = os.environ.pop("MOE_CACHE_RAM_GIB", None)
    try:
        for win in (True, False):
            r = mc.avail_reserve_gib(a, M(17.4 * GIB, win), model=big)
            check(r and abs(r[0] - 11.4) < 0.01 and r[3]["stream"] == 2.0, f"avail_reserve_gib reads the file size ({'Windows' if win else 'Linux'}): {r and round(r[0], 2)}")
        r = mc.avail_reserve_gib(a, M(17.4 * GIB, False, limit=20 * GIB), model=big)
        check(r is None, "Linux with a cgroup limit: None")
        ax, _, _ = mc.parse_server(["m.gguf", "--no-pin", "--exact"])
        check(mc.avail_reserve_gib(ax, M(17.4 * GIB, False), model=big) is None, "--exact on Linux: as before (none)")
        rx = mc.avail_reserve_gib(ax, M(17.4 * GIB, True), model=big)
        check(rx and abs(rx[0] - 13.4) < 0.01 and rx[3]["stream"] == 0, "--exact on Windows: as before (13.4, no stream term)")
        class Clip(M):
            def mem_for_server(self, r): return 28 * GIB          # clipped at the whole-server cap
            def mem_available(self): return 30 * GIB
        rc = mc.avail_reserve_gib(a, Clip(30 * GIB, True))
        check(rc and abs(rc[0] - 26.0) < 0.01, f"the reserve is taken once: 30 GiB available -> 26, not the clipped 28 - 4 = 24 ({rc and rc[0]})")
        os.environ["MOE_CACHE_STREAM_GIB"] = "0"
        r = mc.avail_reserve_gib(a, M(17.4 * GIB, True), model=big)
        check(r and abs(r[0] - 13.4) < 0.01, "MOE_CACHE_STREAM_GIB=0 gives the old 13.4")
    finally:
        os.environ.pop("MOE_CACHE_STREAM_GIB", None)
        if saved is not None:
            os.environ["MOE_CACHE_RAM_GIB"] = saved
# 8. RAM-LIVE: the keep the launcher hands to the plugin
K = mc.live_keep_decision
check(abs(K(31.1 * GIB) - 4.0) < 1e-9 and abs(K(64 * GIB) - 8.0) < 1e-9, "live keep = the start reserve: 4 GiB at 31 GiB, 8 GiB at 64 GiB")
check(K(31.1 * GIB, stock=True) is None and K(31.1 * GIB, exact=True) is None and K(31.1 * GIB, env_keep="6") is None and K(31.1 * GIB, off="0") is None and K(None) is None,
      "live keep off: stock, --exact, the user's own value, MOE_CACHE_LIVE_FOLLOW=0, no RAM figure")
check(K(31.1 * GIB, windows=True) is None and K(31.1 * GIB, windows=True, off="0") is None and abs(K(31.1 * GIB, windows=True, off="1") - 4.0) < 1e-9
      and abs(K(31.1 * GIB, windows=False) - 4.0) < 1e-9 and K(31.1 * GIB, windows=True, off="1", stock=True) is None,
      "0.3.9: Windows live follow is OFF by default, MOE_CACHE_LIVE_FOLLOW=1 turns it on, Linux unchanged")
import contextlib, io
class M2(M):
    def load_env(self, ls, lib, method="auto"): return [("LD_PRELOAD", lib)], "preload"
    def mem_cap(self): return TOT - 4 * GIB
os.environ.pop("MOE_CACHE_AVAIL_KEEP_GIB", None); os.environ.pop("MOE_CACHE_LIVE_FOLLOW", None)
out = io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
    envd = dict(mc.memory_env_stage(a, "llama-server", "lib.so", "m.gguf", "p", False, "off", M2(17.4 * GIB, True))[3])
os.environ.pop("MOE_CACHE_LIVE_FOLLOW", None)
check("MOE_CACHE_AVAIL_KEEP_GIB" not in envd and "live RAM follow: off (default on Windows" in out.getvalue(), "Windows default: no MOE_CACHE_AVAIL_KEEP_GIB, the plan line says off and names the switch")
os.environ["MOE_CACHE_LIVE_FOLLOW"] = "1"
out = io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
    envd = dict(mc.memory_env_stage(a, "llama-server", "lib.so", "m.gguf", "p", False, "off", M2(17.4 * GIB, True))[3])
check(envd.get("MOE_CACHE_AVAIL_KEEP_GIB") == "4.000" and "live RAM follow" in out.getvalue(), "Windows with MOE_CACHE_LIVE_FOLLOW=1: MOE_CACHE_AVAIL_KEEP_GIB is passed and the plan line is printed")
os.environ.pop("MOE_CACHE_LIVE_FOLLOW", None)
out = io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
    envl = dict(mc.memory_env_stage(a, "llama-server", "lib.so", "m.gguf", "p", False, "off", M2(17.4 * GIB, False))[3])
check(envl.get("MOE_CACHE_AVAIL_KEEP_GIB") == "4.000", "Linux default unchanged: MOE_CACHE_AVAIL_KEEP_GIB is set")
ax2, _, _ = mc.parse_server(["m.gguf", "--no-pin", "--exact"])
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    envx = dict(mc.memory_env_stage(ax2, "llama-server", "lib.so", "m.gguf", "p", False, "off", M2(17.4 * GIB, True))[3])
check("MOE_CACHE_AVAIL_KEEP_GIB" not in envx, "--exact: not set")
# --- LAUNCHER-1277: the plan line says off exactly where the plugin's guard (mclive::follows) is off: Linux file-view (default), on with MOE_CACHE_LINUX_VIEW=0 and on Windows ---
import re as _re
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "mc_livecap.h")).read()
_m = _re.search(r"constexpr bool follows\(bool file_view, bool is_windows\) \{\s*return ([^;]+);", _src)
check(bool(_m), "mc_livecap.h has the follows() rule")
def _plugin_follows(file_view, is_windows):          # the header's expression, evaluated
    return eval(_m.group(1).replace("!", " not ").replace("||", " or ").replace("&&", " and ").replace("file_view", str(file_view)).replace("is_windows", str(is_windows)))
for _win in (False, True):
    for _env in (None, "0", "1"):
        if _env is None: os.environ.pop("MOE_CACHE_LINUX_VIEW", None)
        else: os.environ["MOE_CACHE_LINUX_VIEW"] = _env
        _mach = M2(17.4 * GIB, _win)
        os.environ.pop("MOE_CACHE_AVAIL_KEEP_GIB", None)
        _o = io.StringIO()
        with contextlib.redirect_stdout(_o), contextlib.redirect_stderr(_o):
            mc.memory_env_stage(a, "llama-server", "lib.so", "m.gguf", "p", False, "off", _mach)
        _says_off = "live RAM follow: off on Linux's default path" in _o.getvalue()
        _file_view = (_env != "0")                     # Linux default = file view; the Windows default view does not matter (is_windows wins)
        check(_says_off == (not _plugin_follows(_file_view, _win)), f"plan line matches the plugin guard (windows={_win}, MOE_CACHE_LINUX_VIEW={_env}): says off {_says_off}")
os.environ.pop("MOE_CACHE_LINUX_VIEW", None)

# the plugin's two use sites call the one rule (a copy of the old inline guard would not fail the truth table above); live_keep is set only where it follows (critic 2026-11-23: with it set and the probe skipped, growth stopped)
_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_tune = open(os.path.join(_base, "mc_state_profile.inl")).read(); _init = open(os.path.join(_base, "mc_state.inl")).read()
check("if (live_keep && mclive::follows(win_view, plat::is_windows))" in _tune, "tune(): the RAM-LIVE guard is mclive::follows()")
_i = _init.index("if (!mclive::follows(win_view, plat::is_windows)) {"); _blk = _init[_i:_i + 1800]
check(_blk.index("live RAM follow is OFF") < _blk.index("live_keep = k") if "live_keep = k" in _blk else False, "start-up: live_keep is set only in the branch where it follows (off branch leaves it 0)")
check(_init.count("live_keep = k") == 1, "live_keep is assigned once, from the launcher's value")

print("ram cap decision test: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
sys.exit(1 if FAILS else 0)
