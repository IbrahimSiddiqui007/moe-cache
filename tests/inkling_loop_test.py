#!/usr/bin/env python3
"""INKLING-SUPPORT-1: inkling_loop_stage (bin/moe_cache.py) without a model: a mocked GGUF header. Standard library only.
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

META = {"general.architecture": "inkling2_loop", "inkling2_loop.block_count": 24, "inkling2_loop.loop.start": 6, "inkling2_loop.loop.end": 18, "inkling2_loop.loop.count": 3}
gguf_plan.read_meta = lambda p: dict(META)
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
print("inkling_loop_test: " + ("all passed" if not fails else f"{len(fails)} failed"))
sys.exit(1 if fails else 0)
