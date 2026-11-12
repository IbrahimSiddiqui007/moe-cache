#!/usr/bin/env python3
"""load_mode_test (LAUNCHER-1133 follow-up, prepared not merged): a dense model that does not fit the GPU keeps mmap, unless the RAM the server may use is under the model file
plus 4 GiB: then --load-mode none. Fake machine and a sparse file, no model."""
import inspect, os, sys, tempfile, types

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
    def __init__(self, windows=True, avail=20 * GIB):
        self.windows, self.avail = windows, avail

    def mem_for_server(self, ram): return self.avail


d = tempfile.mkdtemp()
f = os.path.join(d, "m.gguf")
with open(f, "wb") as fh:
    fh.truncate(int(13.5 * GIB))                      # sparse: no disk used
A = lambda **k: types.SimpleNamespace(**{"exact": False, "stock": False, "ram": None, **k})
cmd = ["llama-server", "-m", f, "--load-mode", "mmap"]
r = lambda m=M(), a=None, c=cmd, e=(), mod=f: mc.load_mode_low_ram(a or A(), c, list(e), m, mod)

why = r(M(avail=14 * GIB))
check(why and "13.5 GiB" in why and "14.0 GiB of RAM is available" in why and "--load-mode none" in why, f"14 GiB available for a 13.5 GiB file: none, and the note says the figures ({why and why[:80]})")
check(r(M(avail=17 * GIB)) is not None, "17 GiB available (under file + 4 = 17.5): none")
check(r(M(avail=18 * GIB)) is None, "18 GiB available (over file + 4): mmap stays")
check(r(M(windows=False, avail=1 * GIB)) is None, "Linux: mmap stays (page cache serves it, no spike measured)")
check(r(M(avail=14 * GIB), A(exact=True)) is None and r(M(avail=14 * GIB), A(stock=True)) is None, "--exact and --stock: nothing changed")
check(r(M(avail=14 * GIB), e=["--load-mode", "mmap"]) is None and r(M(avail=14 * GIB), e=["--no-mmap"]) is None and r(M(avail=14 * GIB), e=["-lm=none"]) is None, "a load flag of your own: kept")
check(r(M(avail=14 * GIB), c=cmd + ["-ot", "x"]) is None and r(M(avail=14 * GIB), c=cmd + ["-ngl", "0"]) is None, "-ot or -ngl 0 (FFN in RAM / CPU-only plans): mmap stays")
check(r(M(avail=14 * GIB), mod=os.path.join(d, "missing.gguf")) is None and r(M(avail=None)) is None, "unreadable file or unknown RAM: mmap stays (no guess)")
src = inspect.getsource(mc.command_stage)
check("load_mode_low_ram(a, cmd, extra, machine, model)" in src and src.index("load_mode_fit_none(") < src.index("load_mode_low_ram("), "command_stage calls it after the fits-fully rule, for dense models only")
print("load mode test: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
sys.exit(1 if FAILS else 0)
