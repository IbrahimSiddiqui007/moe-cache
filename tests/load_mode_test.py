#!/usr/bin/env python3
"""load_mode_test (LAUNCHER-1133 follow-up, prepared not merged): a dense model that does not fit the GPU keeps mmap, unless the RAM the server may use is under the model file
plus 4 GiB: then --load-mode none. Fake machine and a sparse file, no model."""
import os, sys, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
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
r = lambda m=M(), a=None, c=cmd, e=(), mod=f, sp=5.0: mc.load_mode_low_ram(a or A(), c, list(e), m, mod, sp)

why = r(M(avail=14 * GIB))
check(why and "13.5 GiB" in why and "14.0 GiB of RAM is available" in why and "stage about 8.5 GiB" in why and "--load-mode none" in why and "5.0 GiB stays in RAM" in why, f"14 GiB available for a 13.5 GiB file: none, and the note says the figures ({why and why[:80]})")
check(r(M(avail=16 * GIB)) is not None and r(M(avail=17 * GIB)) is None and r(M(avail=19 * GIB)) is None, "critic L3 (the default): staging 8.5 GiB + 2 x 4 reserve = 16.5: 16 GiB available -> none, 17 and 19 -> mmap (the whole-file rule is off by default)")
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
check(r(m64) is not None and r(M(avail=20 * GIB, total=32 * GIB, cap=28 * GIB)) is None, "R4: 13.5 GiB file: 19 GiB available on the 64 GiB machine (8 GiB reserve) -> none; 20 GiB on a 32 GiB machine (4 GiB reserve) -> mmap")
check("8.0 GiB the machine keeps free" in r(m64) and "available to the server" in r(m64), "the note names the margin and says 'available to the server' (R4, R6)")
# MMAP-SPIKE (opt-in, MOE_CACHE_MMAP_SPIKE=1; critic 2026-11-17: a default change needs the owner, and `none` is not measured at a 5-9 GiB spill): the whole file plus the prompt cache against the reserve
E2 = ["--cache-ram", "2048"]                     # command_stage has put the launcher's own value there
os.environ[mc.MMAP_SPIKE_ENV] = "1"
check(r(M(avail=19 * GIB), e=E2) is not None and r(M(avail=20 * GIB), e=E2) is None and r(M(avail=18 * GIB), e=E2) is not None, "opt-in on: the whole file 13.5 + prompt cache 2 + reserve 4 = 19.5: 19 GiB available -> none, 20 -> mmap")
check(mc.mmap_working_set_short(13.3, 17.4, 2.0, 4.0) and not mc.mmap_working_set_short(13.3, 20.0, 2.0, 4.0), "mmap_working_set_short: 13.3 file + 2 prompt cache + 4 reserve = 19.3 GiB: 17.4 short, 20 not")
pcg = mc.prompt_cache_gib
check(pcg(["--cache-ram", "512"]) == 0.5 and pcg(["--cache-ram", "2048"]) == 2.0 and pcg(["-cram", "1024"]) == 1.0, "prompt cache GiB: the --cache-ram MiB in the command")
check(pcg(["--cache-ram", "0"]) == 0.0, "critic fix 4: --cache-ram 0 is OFF (0 GiB), not 2")
check(pcg(["--cache-ram", "-1"]) == 8.0 and pcg([]) == 8.0, "-1 (no limit) and no flag count as llama's own 8 GiB")
check(mc.cram_gib("0", 0.0) == 0.0 and mc.cram_gib(None, 0.0) == 0.0 and mc.cram_gib("-1", 0.0) == 8.0 and mc.cram_gib("2048", 0.0) == 2.0, "one parser: load_mode_cache_none shares it (missing flag stays 0 there)")
pcw = r(M(avail=17.4 * GIB, total=31.1 * GIB, cap=27.1 * GIB), e=E2, mod=f, sp=9.0)          # large spill: stage 4.5 + 8 = 12.5 fits, the old rule kept mmap
check(pcw and "whole 13.5 GiB file" in pcw and "--load-mode none" in pcw and "MOE_CACHE_MMAP_SPIKE=1" in pcw and "chain" not in pcw and "guard" not in pcw and "1.2 GB" not in pcw, "large spill, 17.4 GiB available: the staging rule kept mmap; the working-set rule (opt-in) picks none, says so, no internal words, no unrecorded incident")
check(r(M(avail=24 * GIB, total=31.1 * GIB, cap=27.1 * GIB), e=E2, mod=f, sp=9.0) is None, "24 GiB available: mmap stays")
mw = M(avail=17.6 * GIB, total=31.1 * GIB, cap=27.1 * GIB)
check(r(mw, e=["--cache-ram", "0"], mod=f, sp=9.0) is None and r(mw, e=E2, mod=f, sp=9.0) is not None and r(mw, e=["--cache-ram", "-1"], mod=f, sp=9.0) is not None, "--cache-ram 0 counts 0 (13.5 + 4 = 17.5 fits 17.6: mmap), 2048 and -1 count more (none)")
# critic fix 8: the RAM available now (not clipped at mem_cap, which already holds the reserve) and every shard
class MU(M):
    def mem_available(self): return 40 * GIB
check(r(MU(avail=17.6 * GIB), e=E2, mod=f, sp=9.0) is None, "fix 8: a machine with 40 GiB available (the server's figure clipped to 17.6) does not take the reserve twice: mmap stays")
sh = [os.path.join(d, f"s-0000{i}-of-00002.gguf") for i in (1, 2)]
for x in sh:
    open(x, "wb").close(); FAKE_SIZE[x] = int(7 * GIB)
check(mc.model_file_bytes(sh[0]) == 14 * GIB and mc.model_file_bytes(f) == int(13.5 * GIB), "fix 8: a split GGUF counts every shard, a single file its own size")
check(r(M(avail=19 * GIB), e=E2, mod=sh[0], sp=9.0) is not None and r(M(avail=20.5 * GIB), e=E2, mod=sh[0], sp=9.0) is None, "fix 8: 14 GiB in two shards: the whole-file rule sees 14, not 7")
del os.environ[mc.MMAP_SPIKE_ENV]
check(r(M(avail=17.4 * GIB, total=31.1 * GIB, cap=27.1 * GIB), e=E2, mod=f, sp=9.0) is None, "opt-in off (default): the same plan keeps mmap")
# the PC case through the real command_stage (main's request, 2026-11-12): dense, the planner says "does not fit", 12 GiB available for a 13.5 GiB file
import contextlib, io
class PCMachine(mc.Machine):
    windows = True
    def __init__(self, avail): self.avail = avail
    def mem_total(self): return 32 * GIB
    def mem_cap(self): return 28 * GIB
    def mem_for_server(self, ram): return self.avail
    def mem_available(self): return self.avail
    def run(self, cmd, **kw): return types.SimpleNamespace(returncode=0, stdout="--fit-target\n", stderr="")


def stage(avail, fits, extra=(), windows=True, args=(), over=5.0):
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
check(mode == "none" and "13.5 GiB" in say and "12.0 GiB of RAM is available" in say, f"low RAM through command_stage: dense, does not fit (5 GiB stays), 12 GiB available -> --load-mode none with the plain line ({mode})")
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
# chain's rn038_loadcheck28b (28b80a9, 2026-11-13): 27B, planner 0.7 GiB over, 19 GB available, apps open: mmap staged ~11.5 GB through RAM and the guard stopped the load
mode, say = stage(19 * GIB, fits=False, over=0.7)
check(mode == "none" and "takes almost all" in say and "0.7 GiB stays in RAM" in say, f"the PC case through command_stage: small spill (0.7 GiB), 19 GiB available -> --load-mode none ({mode})")
mode, say = stage(30 * GIB, fits=False, over=4.0)
check(mode == "mmap", f"a spill over 3 GiB with plenty of RAM keeps mmap ({mode})")
check(r(M(avail=30 * GIB), sp=3.0) is not None and r(M(avail=30 * GIB), sp=3.1) is None, "the small-spill limit is 3 GiB (13.5 GiB file: 25 % = 3.4)")
smallf = os.path.join(d, "small.gguf"); open(smallf, "wb").close(); FAKE_SIZE[smallf] = 4 * GIB
check(r(M(avail=30 * GIB), mod=smallf, sp=1.5) is None and r(M(avail=30 * GIB), mod=smallf, sp=0.9) is not None, "and at most 25 % of the file (4 GiB file: 1.0)")
mode, say = stage(19 * GIB, fits=False, over=0.7, windows=False)
check(mode == "mmap", f"Linux: mmap stays ({mode})")
mode, say = stage(19 * GIB, fits=False, over=0.7, extra=["--load-mode", "mmap"])
check(mode == "mmap" and "takes almost all" not in say, f"a --load-mode of your own: kept ({mode})")
# critic L2: the commit check must not turn the launcher's own --load-mode none back into mmap
mode, say = stage(19 * GIB, fits=False, over=0.7)
check(mode == "none" and mc.LOAD_NONE_BY_RULE[0] is True, "the rule's own none is marked for the commit check")
mode, say = stage(30 * GIB, fits=False, over=4.0)
check(mode == "mmap" and mc.LOAD_NONE_BY_RULE[0] is False, "mmap: not marked")
# critic fix 6: Windows, dense, the planner's spill (9 GiB at 128k), 17.4 GiB available, the launcher's own --cache-ram 2048, through command_stage
mode, say = stage(17.4 * GIB, fits=False, over=9.0, extra=["--cache-ram", "2048"])
check(mode == "mmap" and "whole" not in say, f"command_stage, default: Windows dense + 9 GiB spill, 17.4 GiB available -> mmap stays ({mode})")
os.environ[mc.MMAP_SPIKE_ENV] = "1"
try:
    mode, say = stage(17.4 * GIB, fits=False, over=9.0, extra=["--cache-ram", "2048"])
    check(mode == "none" and "whole 13.5 GiB file" in say and "MOE_CACHE_MMAP_SPIKE=1" in say and "chain" not in say and "guard" not in say and mc.LOAD_NONE_BY_RULE[0] is True, f"command_stage, opt-in: -> none, the plain line, marked for the commit check ({mode})")
    mode, say = stage(20 * GIB, fits=False, over=9.0, extra=["--cache-ram", "0"])
    check(mode == "mmap", f"command_stage, opt-in, prompt cache off, 20 GiB available: mmap ({mode})")
    mode, say = stage(17.4 * GIB, fits=False, over=9.0, extra=["--cache-ram", "2048"], windows=False)
    check(mode == "mmap", f"command_stage, opt-in, Linux: mmap ({mode})")
    mode, say = stage(17.4 * GIB, fits=False, over=9.0, extra=["--cache-ram", "2048"], args=["--exact"])
    check(mode == "mmap", f"command_stage, opt-in, --exact: mmap ({mode})")
finally:
    del os.environ[mc.MMAP_SPIKE_ENV]
print("load mode test: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
sys.exit(1 if FAILS else 0)
