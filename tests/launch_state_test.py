#!/usr/bin/env python3
"""LAUNCHER-003: launch_plan() must not change process-wide state that control-page threads share (MOE_CACHE_MAX_RAM_GB, notes, umask).
No model, nothing started: launch_plan is replaced by a stub for _launch_plan.   python3 tests/launch_state_test.py"""
import os, sys, threading, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin"))
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

class Plan: pass
class A:
    def __init__(self, max_ram=None): self.max_ram = max_ram

os.environ.pop("MOE_CACHE_MAX_RAM_GB", None)
seen = {}
def fake(a, extra, argv, machine):
    if a.max_ram:
        mc._TL.max_ram = a.max_ram                  # what _launch_plan does
    mc.say("hello " + (a.max_ram or "-"))
    seen[a.max_ram] = mc.mem_cap()
    time.sleep(0.05)
    seen[(a.max_ram, "after")] = mc.mem_cap()
    return Plan()
mc._launch_plan = fake

# 1. --max-ram is local to the plan: the environment is not touched, the cap is the plan's own, it is gone afterwards
p = mc.launch_plan(A("7"), [], [], machine=object(), quiet=True)
check("MOE_CACHE_MAX_RAM_GB" not in os.environ, "launch_plan does not set MOE_CACHE_MAX_RAM_GB in the environment")
check(seen["7"] == 7e9, "while planning, mem_cap() is the plan's --max-ram (7 GB)")
check(getattr(mc._TL, "max_ram", None) is None, "after planning, the plan's cap is gone")
check(p.notes == [("hello 7", False)], "the plan's notes hold its lines")

# 2. two threads plan at once: each sees its own cap and its own notes
res = {}
def run(n):
    res[n] = mc.launch_plan(A(n), [], [], machine=object(), quiet=True)
ts = [threading.Thread(target=run, args=(n,)) for n in ("5", "9")]
[t.start() for t in ts]; [t.join() for t in ts]
check(seen["5"] == 5e9 and seen[("5", "after")] == 5e9 and seen["9"] == 9e9 and seen[("9", "after")] == 9e9, "two threads: 5 GB and 9 GB caps stay apart")
check(res["5"].notes == [("hello 5", False)] and res["9"].notes == [("hello 9", False)], "two threads: each plan has only its own lines")

# 3. quiet is per thread: a quiet plan in one thread does not silence another thread's say()
import io, contextlib
quiet_in = threading.Event(); out_done = threading.Event()
def slow(a, extra, argv, machine):
    quiet_in.set(); out_done.wait(2); return Plan()
mc._launch_plan = slow
t = threading.Thread(target=lambda: mc.launch_plan(A(), [], [], machine=object(), quiet=True)); t.start(); quiet_in.wait(2)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    mc.say("loud")
out_done.set(); t.join()
check("loud" in buf.getvalue(), "a quiet plan in another thread does not silence this thread")

# 4. umask: two quiet plans in turn leave the caller's umask alone, even when they overlap
if os.name != "nt":
    mc._launch_plan = lambda a, e, v, m: (os.umask(0o077), time.sleep(0.05), Plan())[2]
    old = os.umask(0o022); os.umask(old)
    os.umask(0o022)
    ts = [threading.Thread(target=lambda: mc.launch_plan(A(), [], [], machine=object(), quiet=True)) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    now = os.umask(0o022)
    check(now == 0o022, f"four overlapping quiet plans restore the umask (0o{now:03o})")
    os.umask(old)

# ENGINE-029: the figures the page shows come from the plan alone
lp = mc.LaunchPlan()
lp.planner_info, lp.tuned, lp.n_cpu_moe, lp.ub, lp.kv_q8 = None, None, "20", "1024", False
check(lp.numbers() is None, "numbers(): no planner answer, no numbers")
pl = {"arch": "qwen3moe", "layers": 48, "experts": 128, "used": 8, "n_cpu_moe": 40, "cache_GiB": 6.0, "cpu_experts_GiB": 18.0, "free_vram_GiB": 7.5, "note": ""}
lp.planner_info = pl
n = lp.numbers()
check(n["n_cpu_moe"] == 20 and n["gpu_expert_layers"] == 28 and n["layers"] == 48 and n["ub"] == 1024 and n["kv_cache"] == "f16" and n["cache_GiB"] == 6.0, "numbers(): the launcher's n_cpu_moe and -ub, the planner's model sizes")
lp.n_cpu_moe, lp.ub, lp.tuned, lp.kv_q8 = "auto", "", {"ub": 2048}, True
n = lp.numbers()
check(n["n_cpu_moe"] == 40 and n["gpu_expert_layers"] == 8 and n["ub"] == 2048 and n["kv_cache"] == "q8_0", "numbers(): a non-numeric n_cpu_moe falls back to the planner's, -ub to the tuner's, KV q8_0")
lp.n_cpu_moe = "999"
check(lp.numbers()["gpu_expert_layers"] == 0 and lp.numbers()["n_cpu_moe"] == 999, "numbers(): more CPU layers than layers: no GPU layers")
sys.exit(1 if fails else 0)
