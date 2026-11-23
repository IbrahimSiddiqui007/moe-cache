#!/usr/bin/env python3
"""INKLING-SUPPORT-1 + LOOP-SPEED-1: inkling_loop_stage (bin/moe_cache.py) without a model: a mocked GGUF header. Standard library only.
   python3 tests/inkling_loop_test.py"""
import os, sys, types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "planner"))
import moe_cache as mc  # noqa: E402
import gguf_plan  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

META = {"general.architecture": "inkling2_loop", "inkling2_loop.block_count": 24, "inkling2_loop.loop.start": 6, "inkling2_loop.loop.end": 18, "inkling2_loop.loop.count": 3,
        "inkling2_loop.attention.head_count": [24] * 24}
gguf_plan.read_meta = lambda p: dict(META)
MiB = 1048576
gguf_plan.model_layout = lambda p: {"expert_bytes_by_layer": {i: (668 if i in (20, 21) else 547) * MiB for i in range(24)}}   # Inkling-2-Mini UD-IQ4_XS sizes
mc.ls_build = lambda ls: None
mc.model_parts = lambda m: [m]
mc.say = lambda *a, **k: None
A = lambda **kw: types.SimpleNamespace(**{"exact": False, "cpu_only": False, "n_cpu_moe": None, "ub": "auto", **kw})

extra = ["-ub", "2048", "-b", "2048"]
ub, nc = mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "12", None, False)
check(nc == "0" and ub == "128" and extra[:4] == ["-ub", "128", "-b", "2048"] and "--no-op-offload" in extra, "default: ub 128, no-op-offload, placement through -ot")
rx = extra[extra.index("-ot") + 1]
check(rx.startswith("blk\\.(0|1|2|3|4|5|18|19|20|21|22|23)\\.ffn_") and rx.endswith("=CPU"), "12 CPU layers = the once-used layers (0-5, 18-23)")
extra = ["-ub", "2048"]
mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "19", None, False)
check("blk\\.(0|1|2|3|4|5|6|7|8|9|10|11|18|19|20|21|22|23)\\." in extra[extra.index("-ot") + 1] or "6|7" in extra[extra.index("-ot") + 1], "19 CPU layers: once-used first, then loop layers from the front (6-12)")
extra = []
check(mc.inkling_loop_stage(A(exact=True), extra, "m.gguf", "2048", "12", None, False) == ("2048", "12") and extra == [], "--exact: untouched")
extra = []
check(mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "12", "10", False) == ("128", "12") and "-ot" not in extra and "--no-op-offload" in extra, "your --n-cpu-moe stands (placement), the other two rules still apply")
extra = ["-ub", "512"]
check(mc.inkling_loop_stage(A(ub="512"), extra, "m.gguf", "512", "12", None, False)[0] == "512" and extra[1] == "512", "your --ub stands")
extra = ["--no-op-offload"]
mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "12", None, False)
check(extra.count("--no-op-offload") == 1, "no second --no-op-offload")
META["general.architecture"] = "qwen3moe"
extra = []
check(mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "12", None, False) == ("2048", "12") and extra == [], "other architectures: untouched")
META["general.architecture"] = "inkling2_loop"
check(mc.inkling_loop_stage(A(), [], "m.gguf", "2048", "0", None, False) == ("2048", "0") and mc.inkling_loop_stage(A(), [], "m.gguf", "2048", "999", None, False) == ("2048", "999"),
      "nothing on the CPU (0) or a cache plan (999): untouched")

# LOOP-SPEED-1: any arch with loop keys counts; count 1 does not
def rename(arch, count=3):
    META.clear(); META.update({"general.architecture": arch, f"{arch}.block_count": 24, f"{arch}.loop.start": 6, f"{arch}.loop.end": 18, f"{arch}.loop.count": count,
                               f"{arch}.attention.head_count": [24] * 24})
rename("future_looped")
extra = ["-ub", "2048"]
check(mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "12", None, False)[1] == "0" and "-ot" in extra, "generic: another arch with loop.* keys gets the placement")
rename("future_looped", count=1)
extra = []
check(mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "12", None, False) == ("2048", "12") and extra == [], "generic: loop.count 1 = not looped, untouched")
rename("inkling2_loop")
T64 = {"compute_mib": 4660}            # tune() at 64k, IQ4_XS, 15.3 GiB free, long-prompt mode: -ub 4096, 13 CPU layers (checked 2026-11-23)
mc.ls_build = lambda ls: 12651
extra = ["-ub", "4096", "-b", "4096"]
ub, nc = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check("--no-op-offload" not in extra and ub == "256" and extra[1] == "256", f"b12651 at 64k: op-offload on, -ub 256 (3072 bias + 690 + 547 staged = 4309 <= 4660) [got ub {ub}]")
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=131072, tuned={"compute_mib": 7958}, ls="ls")
check("--no-op-offload" not in extra and ub == "256", f"b12651 at 128k: -ub 256 fits (6144 + 960 + 547 = 7651 <= 7958), 512 does not [got {ub}]")
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=262144, tuned={"compute_mib": 7958}, ls="ls")
check("--no-op-offload" in extra and ub == "128", f"b12651 at 256k with the same budget: even -ub 128 + staging (6144 + 1403 + 547) is over: op-offload off, -ub 128 [got {ub}]")
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=None, ls="ls")
check("--no-op-offload" in extra and ub == "128", "no planner budget: op-offload off, -ub 128 (the old rule)")
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned={"compute_mib": 900}, ls="ls")
check("--no-op-offload" in extra and ub == "128", "budget too small for staging at -ub 128: op-offload off")
mc.ls_build = lambda ls: 12207
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check("--no-op-offload" in extra and ub == "256", f"b12207 (no PR 32807): op-offload off; -ub still sized to the budget without staging [got {ub}]")
mc.ls_build = lambda ls: 12651
os.environ["MOE_CACHE_INKLING_OP_OFFLOAD"] = "0"
extra = ["-ub", "4096"]
mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check("--no-op-offload" in extra, "MOE_CACHE_INKLING_OP_OFFLOAD=0 forces it off")
os.environ["MOE_CACHE_INKLING_OP_OFFLOAD"] = "1"
mc.ls_build = lambda ls: None
extra = ["-ub", "4096"]
mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check("--no-op-offload" not in extra, "MOE_CACHE_INKLING_OP_OFFLOAD=1 forces it on (unknown build)")
del os.environ["MOE_CACHE_INKLING_OP_OFFLOAD"]
mc.ls_build = lambda ls: 12651
extra = ["-ub", "4096", "--no-op-offload"]
mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check(extra.count("--no-op-offload") == 1, "your --no-op-offload stands, once")
extra = ["-ub", "4096", "-ot", "blk\\.(6|7)\\.ffn_.*=CPU"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check(extra.count("-ot") == 1 and ub == "256" and "--no-op-offload" in extra, f"your -ot: all 12 loop layers counted as staged (errs high): op-offload off, -ub sized without staging [got {ub}]")

print("inkling_loop_test: " + ("all passed" if not fails else f"{len(fails)} failed"))
sys.exit(1 if fails else 0)
