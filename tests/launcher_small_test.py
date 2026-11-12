#!/usr/bin/env python3
"""LAUNCHER-005: the small launcher fixes of the third audit that are still open, each on its own (A3-17, 20, 21, 22, 23, 25, 26, 27, 28). No model, no server.
   python3 tests/launcher_small_test.py"""
import json, os, sys, tempfile, time, types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-small-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, os.path.join(HERE, "planner"))
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# A3-21: exit status of a signalled server
if os.name != "nt":
    check(mc.exit_status(-9) == 137 and mc.exit_status(-15) == 143 and mc.exit_status(0) == 0 and mc.exit_status(3) == 3 and mc.exit_status(None) is None,
          "exit status: -9 -> 137, -15 -> 143, others unchanged")

# A3-22: MOE_CACHE_MAX_RAM_GB read like the plugin's strtod
check(mc.leading_number("20GB") == "20" and mc.leading_number(" 1.5e1x") == "1.5e1" and mc.leading_number(".5") == ".5", "leading number: 20GB -> 20, 1.5e1x -> 1.5e1, .5")
for bad in ("GB", "", "0x10", "inf", "nan", "abc20"):
    try:
        mc.leading_number(bad); ok = False
    except ValueError:
        ok = True
    check(ok, f"leading number: {bad!r} is a typo (default cap)")
os.environ["MOE_CACHE_MAX_RAM_GB"] = "20GB"
check(mc.mem_cap() == 20e9, "mem_cap(): MOE_CACHE_MAX_RAM_GB=20GB is 20 GB, as in the plugin")
os.environ["MOE_CACHE_MAX_RAM_GB"] = "0.0GB"
check(mc.mem_cap() is None, "mem_cap(): 0GB means no cap")
os.environ.pop("MOE_CACHE_MAX_RAM_GB")

# A3-23: the folder above the kit is not searched
for win in (False, True):
    mc.WINDOWS = win
    d = mc.llama_search_dirs()
    check(mc.HERE in d and os.path.dirname(mc.HERE) not in d, f"search folders ({'Windows' if win else 'Linux'}): the kit, not the folder above it")
mc.WINDOWS = os.name == "nt"

# A3-20: job handles of ended children are closed
class Child:
    def __init__(self, alive): self.alive = alive
    def poll(self): return None if self.alive else 0
jobs, closed = [(11, Child(False)), (12, Child(True)), (13, Child(False))], []
n = mc.prune_jobs(jobs, closed.append)
check(n == 2 and closed == [11, 13] and [j for j, _ in jobs] == [12], "jobs: handles of ended children are closed, the running one stays")
def boom(h): raise OSError("x")
jobs = [(1, Child(False))]
check(mc.prune_jobs(jobs, boom) == 1 and jobs == [], "jobs: a handle that cannot be closed is dropped without an error")

# A3-26: notes are written all at once
f = os.path.join(TMP, "server.id")
mc.write_text_atomic(f, "abc\n")
check(open(f).read() == "abc\n" and not [x for x in os.listdir(TMP) if x.endswith(".tmp")], "atomic note: written, no temp file left")
try:
    mc.write_text_atomic(os.path.join(TMP, "no-such-folder", "x"), "1")
    ok = False
except OSError:
    ok = True
check(ok and not [x for x in os.listdir(TMP) if x.endswith(".tmp")], "atomic note: a failing write raises and leaves no temp file")

# A3-27: expert_count read as a key
import launch_plan_test as lpt  # noqa: E402
m = os.path.join(TMP, "moe.gguf"); lpt.make_synthetic_model(m)
check(mc.header_has_experts(m), "check: a MoE header has expert_count (read as keys)")
open(os.path.join(TMP, "dense.gguf"), "wb").write(b"GGUF" + b"\0" * 64)
check(not mc.header_has_experts(os.path.join(TMP, "dense.gguf")) and not mc.header_has_experts(os.path.join(TMP, "missing.gguf")), "check: no experts / unreadable: False, no error")

# A3-28: stale .new.bin files
sd = os.path.join(TMP, "slots"); os.makedirs(sd)
old, new, keep = (os.path.join(sd, n) for n in ("t-s0.new.bin", "t-s1.new.bin", "t-s0.bin"))
for p_ in (old, new, keep):
    open(p_, "w").write("x")
os.utime(old, (time.time() - 600, time.time() - 600))
check(mc.clean_stale_saves(sd, "t") == 1 and not os.path.exists(old) and os.path.exists(new) and os.path.exists(keep),
      "stale saves: the old .new.bin goes; a recent one (a save in progress) and the real file stay")

# A3-25: learn does not start with a fixed 20 GB on a small machine
mc.mem_for_server = lambda ram: 9.3e9
check(mc.learn_default_ram() == "9.3", "learn: default --ram is what the machine can give (9.3 GB), not 20")
mc.mem_for_server = lambda ram: 64e9
check(mc.learn_default_ram() == "20.0", "learn: at most 20 GB")
mc.mem_for_server = lambda ram: None
check(mc.learn_default_ram() == "20", "learn: unknown memory keeps 20")

# A3-17: --leave-room says only what it did
import contextlib, io  # noqa: E402
class M(lpt.FakeMachine):
    def physical_cores(self): return 8
tmp2 = tempfile.mkdtemp(prefix="moe-lr-")
for f_ in ("llama-server", "libggml-moe-cache.so"):
    open(os.path.join(tmp2, f_), "w").close()
mc.CACHE_DIR = os.path.join(tmp2, "cache"); mc.KEY_FILE = os.path.join(mc.CACHE_DIR, "api.key"); mc.WINDOWS = False
mc.default_llama_server = lambda refresh=False: os.path.join(tmp2, "llama-server")
def notes(*extra_args):
    a, ex, argv = mc.parse_server([m, "--llama-server", os.path.join(tmp2, "llama-server"), "--leave-room", "--no-pin", "--always-plugin", "--plan-json", *extra_args])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return [t for t, _ in mc.launch_plan(a, ex, argv, machine=M(tmp2), quiet=True).notes if "--leave-room" in t]
n1 = notes()
check(bool(n1) and "keeps 2 GB of GPU" not in n1[0] and "leaves 2 CPU cores free" not in n1[0], "leave-room: the first line no longer claims GPU memory and cores before they are decided")
check(any("CPU cores left free" in t for t in n1), f"leave-room: cores are reported when -t was set ({n1})")
def plan_env(*extra_args):
    a, ex, argv = mc.parse_server([m, "--llama-server", os.path.join(tmp2, "llama-server"), "--no-pin", "--always-plugin", "--plan-json", *extra_args])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return dict(mc.launch_plan(a, ex, argv, machine=M(tmp2), quiet=True).env)
check("MOE_CACHE_SCHED_TUNING" not in plan_env(), "the scheduling tuning is gone: nothing is set by default")
check("MOE_CACHE_SCHED_TUNING" not in plan_env("--no-sched-tuning"), "--no-sched-tuning is accepted and does nothing (kept for one release so old commands still run)")
n2 = notes("--", "-t", "4")
check(not any("CPU cores left free" in t for t in n2), "leave-room: with your own -t no cores claim")
# CACHE-181: the loading line names the size and a seconds range; a missing file still gives a line
lf = os.path.join(tmp2, "m.gguf")
with open(lf, "wb") as fh:
    fh.truncate(6_000_000_000)
ll = mc.loading_line(lf)
check("loading 6.0 GB" in ll and "about 2-12 s" in ll, f"loading line: size and range ({ll})")
check("size unknown" in mc.loading_line(os.path.join(tmp2, "none.gguf")), "loading line: missing file gives a line, no crash")
# DOCS-047: the --fast help gives the PC figure beside the laptop figure
check("0.0120 on the PC" in mc.SERVER_HELP, "--fast help quotes the PC KL value")
# DOCS-058 item 19: the registry name and the listed name need not match exactly
D = {"AMD Radeon(TM) RX 9070 XT": 16 << 30, "AMD Radeon(TM) Graphics": 512 << 20}
check(mc.dedicated_for("AMD Radeon(TM) RX 9070 XT", D) == 16 << 30, "dedicated VRAM: exact name")
check(mc.dedicated_for("AMD Radeon RX 9070 XT (RADV GFX1201)", D) == 16 << 30, "dedicated VRAM: listed name has no (TM) and a driver suffix")
check(mc.dedicated_for("NVIDIA GeForce RTX 4090", D) == -1, "dedicated VRAM: unknown adapter -1")
lst = "Available devices:\n  Vulkan0: AMD Radeon Graphics (2048 MiB, 2000 MiB free)\n  Vulkan1: AMD Radeon RX 9070 XT (16304 MiB, 15000 MiB free)\n"
mc.win_dedicated_vram = lambda: D
check(mc.pick_gpu_full(lst)[1] == "AMD Radeon RX 9070 XT", "the discrete card wins although the registry spells its name differently")
check(mc.dedicated_vram_of("AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO)", D) == 512 << 20 and mc.dedicated_vram_of("NVIDIA GeForce RTX 4090", D) is None and mc.dedicated_vram_of("x", None) is None,
      "dedicated_vram_of: the fuzzy match, None when the adapter is not in the registry list")
# the launcher's GPU stage (LAUNCHER-991 review): an integrated GPU listed under a name the registry spells differently is planned without VRAM, not with its shared RAM
class _M:
    windows = True
    def list_devices(self, ls): return "Available devices:\n  Vulkan0: AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO) (16000 MiB, 15000 MiB free)\n"
    def pick_gpu_full(self, listing): return ("Vulkan0", "AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO)", 15000, 16000)
    def nvidia_reports(self): return False
    def win_dedicated_vram(self): return {"AMD Radeon(TM) Graphics": 512 << 20}
import types as _t
_a = _t.SimpleNamespace(vram=None, cpu_only=False, gpu_margin=None)
_said = []; _sv = mc.say; mc.say = lambda t, err=False: _said.append(t)
try:
    mc.gpu_stage(_a, "llama-server", True, _M())
finally:
    mc.say = _sv
check(_a.vram is None and any("is integrated" in x for x in _said), f"gpu_stage: the iGPU under a different adapter name is recognised: planned without VRAM (a.vram {_a.vram!r})")
# ENGINE-029 (critic BR-16): two threads asking for a plan at once load planner/plan.py once (a lock around the module cache)
import threading
_pd = os.path.join(TMP, "fakehere"); os.makedirs(os.path.join(_pd, "planner"))
open(os.path.join(_pd, "planner", "plan.py"), "w").write("import time, builtins\nbuiltins._PLAN_LOADS = getattr(builtins, '_PLAN_LOADS', 0) + 1\ntime.sleep(0.3)\ndef plan_info(model, ctx=None, ram_gib=None, vram_gib=None):\n    return {'ok': 1}\n")
_oh, _om = mc.HERE, list(mc._PLAN_MODULE)
mc.HERE = _pd; del mc._PLAN_MODULE[:]
try:
    _ts = [threading.Thread(target=mc.planner_answer, args=("m", 4096)) for _ in range(6)]
    [t_.start() for t_ in _ts]; [t_.join() for t_ in _ts]
    import builtins
    _loads = getattr(builtins, "_PLAN_LOADS", 0)
finally:
    mc.HERE = _oh; mc._PLAN_MODULE[:] = _om
check(_loads == 1, f"ENGINE-029: six threads asking for a plan at once load plan.py once (loads {_loads})")
# WINDOWS-023: QuickEdit off (a fake kernel32: the mode word is read, QuickEdit cleared, EXTENDED_FLAGS set)
import ctypes
class _K:
    def __init__(self, mode, console=True): self.mode, self.console, self.set = mode, console, None
    def GetStdHandle(self, n): return 7
    def GetConsoleMode(self, h, ref):
        if not self.console: return 0
        ctypes.cast(ref, ctypes.POINTER(ctypes.c_uint32))[0] = self.mode; return 1
    def SetConsoleMode(self, h, m): self.set = m; return 1
_k = _K(0x01F7); _r = mc.win_console_no_quickedit(_k)
check(_r is True and _k.set is not None and not _k.set & 0x0040 and _k.set & 0x0080 and (_k.set & 0x0137) == (0x01F7 & 0x0137), f"WINDOWS-023: QuickEdit is cleared and EXTENDED_FLAGS set, the other bits stay ({hex(_k.set or 0)})")
_u = mc._QE_UNDO.pop(); _u()
check(_k.set == 0x01F7, f"WINDOWS-023: the old console mode is put back when moe-cache ends ({hex(_k.set or 0)})")
_k = _K(0x01F7); os.environ["MOE_CACHE_QUICKEDIT"] = "1"
check(mc.win_console_no_quickedit(_k) is False and _k.set is None, "WINDOWS-023: MOE_CACHE_QUICKEDIT=1 leaves the console mode alone")
del os.environ["MOE_CACHE_QUICKEDIT"]
_k = _K(0x0137); check(mc.win_console_no_quickedit(_k) is False and _k.set is None, "WINDOWS-023: already off: nothing written")
_k = _K(0x01F7, console=False); check(mc.win_console_no_quickedit(_k) is False and _k.set is None, "WINDOWS-023: input is not a console (service, pipe): nothing written")
sys.exit(1 if fails else 0)
