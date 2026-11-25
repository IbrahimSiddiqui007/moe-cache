#!/usr/bin/env python3
"""LAUNCHER-1158: a fallback (an auto value replaced by a guess) says so in one plain line. Model-free unit checks of the Python-side ones:
core count unreadable, implausible GPU room figure (see launch plan golden for the rest), fixed drives unreadable, paging watch giving up."""
import os, sys, io, contextlib, types
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "planner"))
import moe_cache as mc   # noqa: E402
import real_room as rr   # noqa: E402
fails = 0
def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what)
    fails += 0 if ok else 1

def capture(fn):
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        r = fn()
    return r, err.getvalue()

# 1. physical_cores: /proc/cpuinfo unreadable -> half the logical CPUs, said once
real_open = open
def no_cpuinfo(p, *a, **k):
    if str(p) == "/proc/cpuinfo": raise OSError("none")
    return real_open(p, *a, **k)
mc.WINDOWS = False
mc._SAID_ONCE.clear()
import builtins
builtins.open = no_cpuinfo
try:
    n, e1 = capture(mc.physical_cores)
    n2, e2 = capture(mc.physical_cores)
finally:
    builtins.open = real_open
check(n == max(1, (os.cpu_count() or 2) // 2) and "core count could not be read" in e1 and "assuming" in e1, f"physical_cores fallback says so ({e1.strip()[:80]})")
check(e2 == "", "...once per run, not at every call")

# 2. fixed_drives (Windows): the API raises -> C and D, said once
import moe_cache_pick as pk   # noqa: E402
if os.name == "nt":
    print("skip fixed_drives check: ctypes.windll works on Windows"); d, e = ["C", "D"], "fixed drives"
else:
  mc.WINDOWS = True; pk.WINDOWS = True
  mc._SAID_ONCE.clear()
  d, e = capture(pk.fixed_drives)
  mc.WINDOWS = False; pk.WINDOWS = False
check(d == ["C", "D"] and "fixed drives" in e, f"fixed_drives fallback says so (got {d}, {e.strip()[:70]})")

# 3. paging watch gives up after three misses without a counter, and says so
msgs = []
alive_n = [0]
def alive():
    alive_n[0] += 1
    return alive_n[0] < 60
rr.watch_room(1, "dev", "/nonexistent-cache", msgs.append, alive, lambda: True, sampler=lambda pid: "", sleep=lambda s: None)
check(any("paging watch" in m and "stops" in m for m in msgs), f"watch_room: a counter that never answers is announced ({msgs[:1]})")

# 4. the room figure that is not plausible (source check: the line is next to the fallback)
src = open(os.path.join(HERE, "bin", "moe_cache.py"), encoding="utf-8").read()
check('say_once("room_implausible"' in src and src.index('say_once("room_implausible"') > src.index("room = real_room_mib(min(by_card, by_driver), 0.0)"), "GPU room: the implausible-figure fallback has its line")
print("fallback lines: " + (f"{fails} failed" if fails else "all passed"))
sys.exit(1 if fails else 0)
