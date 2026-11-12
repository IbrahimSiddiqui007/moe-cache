#!/usr/bin/env python3
"""load_mode_test (LAUNCHER-1133 follow-up, prepared not merged): a dense model that does not fit the GPU keeps mmap, unless the RAM the server may use is under the model file
plus 4 GiB: then --load-mode none. Fake machine and a sparse file, no model."""
import os, sys, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
sys.path.insert(0, os.path.join(HERE, "..", "planner"))
import moe_cache as mc  # noqa: E402

FAILS = []
GIB = 2 ** 30


def check(c, w):
    print(("  ok   " if c else "  FAIL ") + w)
    if not c:
        FAILS.append(w)


class M:
    def __init__(self, windows=True, avail=20 * GIB, total=32 * GIB, cap=28 * GIB):
        self.windows, self.avail, self.total, self.cap = windows, avail, total, cap

    def mem_for_server(self, ram): return self.avail

    def mem_total(self): return self.total

    def mem_cap(self): return self.cap


d = tempfile.mkdtemp()
f = os.path.join(d, "m.gguf")
open(f, "wb").close()
FAKE_SIZE = {f: int(13.5 * GIB)}                       # sizes are faked (a sparse file is not sparse on a Windows runner: 40 GiB failed with "No space left")
_real_getsize = os.path.getsize
os.path.getsize = lambda p: FAKE_SIZE[p] if p in FAKE_SIZE else _real_getsize(p)
A = lambda **k: types.SimpleNamespace(**{"exact": False, "stock": False, "ram": None, **k})
cmd = ["llama-server", "-m", f, "--load-mode", "mmap"]
r = lambda m=M(), a=None, c=cmd, e=(), mod=f, sp=1.0: mc.load_mode_low_ram(a or A(), c, list(e), m, mod, sp)

why = r(M(avail=14 * GIB))
check(why and "13.5 GiB" in why and "14.0 GiB of RAM is available" in why and "--load-mode none" in why and "1.0 GiB of it stays in RAM" in why, f"14 GiB available for a 13.5 GiB file: none, and the note says the figures ({why and why[:80]})")
check(r(M(avail=17 * GIB)) is not None, "17 GiB available (under file + 4 = 17.5): none")
check(r(M(avail=18 * GIB)) is None, "18 GiB available (over file + 4): mmap stays")
check(r(M(windows=False, avail=1 * GIB)) is None, "Linux: mmap stays (page cache serves it, no spike measured)")
check(r(M(avail=14 * GIB), A(exact=True)) is None and r(M(avail=14 * GIB), A(stock=True)) is None, "--exact and --stock: nothing changed")
check(r(M(avail=14 * GIB), e=["--load-mode", "mmap"]) is None and r(M(avail=14 * GIB), e=["--no-mmap"]) is None and r(M(avail=14 * GIB), e=["-lm=none"]) is None, "a load flag of your own: kept")
check(r(M(avail=14 * GIB), c=cmd + ["-ot", "x"]) is None and r(M(avail=14 * GIB), c=cmd + ["-ngl", "0"]) is None, "-ot or -ngl 0 (FFN in RAM / CPU-only plans): mmap stays")
check(r(M(avail=14 * GIB), mod=os.path.join(d, "missing.gguf")) is None and r(M(avail=None)) is None and r(M(avail=14 * GIB), sp=None) is None, "unreadable file, unknown RAM or unknown spill: mmap stays (no guess)")
# critic R1: a file much bigger than the RAM keeps mmap (none would copy the spilled part into private memory)
big = os.path.join(d, "big.gguf")
open(big, "wb").close()
FAKE_SIZE[big] = 40 * GIB
check(r(M(avail=14 * GIB), mod=big, sp=26.0) is None, "R1: 40 GiB file, 14 GiB available, 26 GiB would stay in RAM: mmap stays (the spill + reserve does not fit)")
check(r(M(avail=30 * GIB), mod=big, sp=8.0) is not None, "40 GiB file, 30 GiB available, 8 GiB stays in RAM: none")
# critic R4: the reserve is what the machine keeps free (RAM minus mem_cap), not a fixed 4 GiB, on a bigger machine
m64 = M(avail=19 * GIB, total=64 * GIB, cap=56 * GIB)
check(r(m64, sp=1.0) is not None and r(M(avail=19 * GIB, total=32 * GIB, cap=28 * GIB), sp=1.0) is None, "R4: 13.5 GiB file, 19 GiB available: 64 GiB machine (8 GiB reserve) -> none, 32 GiB machine (4 GiB reserve) -> mmap")
check("8.0 GiB the machine keeps free" in r(m64, sp=1.0) and "available to the server" in r(m64, sp=1.0), "the note names the margin and says 'available to the server' (R4, R6)")
# the PC case through the real command_stage (main's request, 2026-11-12): dense, the planner says "does not fit", 12 GiB available for a 13.5 GiB file
import contextlib, io
class PCMachine(mc.Machine):
    windows = True
    def __init__(self, avail): self.avail = avail
    def mem_total(self): return 32 * GIB
    def mem_cap(self): return 28 * GIB
    def mem_for_server(self, ram): return self.avail
    def run(self, cmd, **kw): return types.SimpleNamespace(returncode=0, stdout="--fit-target\n", stderr="")


def stage(avail, fits, extra=(), windows=True, args=(), over=1.0):
    a, ex, _ = mc.parse_server([f, "--no-pin", *args])
    saved = (mc.dense_gpu_fit_pair, mc.dense_ffn_override)
    mc.dense_gpu_fit_pair = lambda *x, **k: [{"fits": fits, "over_gib": 0.0 if fits else over}] * 4
    mc.dense_ffn_override = lambda *x, **k: None
    m = PCMachine(avail); m.windows = windows
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            cmd, _ = mc.command_stage(a, "llama-server", f, 65536, 8080, "0", list(extra), [], None, True, "preload", "p", m)
    finally:
        mc.dense_gpu_fit_pair, mc.dense_ffn_override = saved
    return cmd[cmd.index("--load-mode") + 1], out.getvalue()


mode, say = stage(12 * GIB, fits=False)
check(mode == "none" and "13.5 GiB" in say and "12.0 GiB of RAM is available" in say, f"PC case through command_stage: dense, does not fit, 12 GiB available -> --load-mode none with the plain line ({mode})")
mode, say = stage(30 * GIB, fits=False)
check(mode == "mmap" and "load-mode none" not in say, f"does not fit, 30 GiB available -> mmap stays ({mode})")
mode, say = stage(12 * GIB, fits=True)
check(mode == "none" and "fits fully on the GPU" in say and "less than the file" not in say, f"fits fully -> the older rule (LAUNCHER-1065) decides, not the new one ({mode})")
mode, say = stage(12 * GIB, fits=False, extra=["--load-mode", "mmap"])
check(mode == "mmap" and "less than the file" not in say, f"a --load-mode of your own after -- is kept ({mode})")
mode, say = stage(12 * GIB, fits=False, windows=False)
check(mode == "mmap", f"Linux: mmap stays ({mode})")
mode, say = stage(12 * GIB, fits=False, args=["--exact"])
check(mode == "mmap", f"--exact: mmap stays ({mode})")
mode, say = stage(14 * GIB, fits=False, over=26.0)
check(mode == "mmap", f"plan level, R1: the planner says 26 GiB would stay in RAM, 14 GiB available -> mmap ({mode})")
mode, say = stage(12 * GIB, fits=False, extra=["-ot", "x"])
check(mode == "mmap", f"plan level: -ot in the command (FFN-in-RAM plan) -> no NameError, mmap stays ({mode})")
print("load mode test: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
sys.exit(1 if FAILS else 0)
