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
mc.ls_build = lambda ls: None; mc._LS_BUILD.clear()
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
extra = ["-ub", "2048"]
r0 = mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "0", None, False)
check(r0 == ("2048", "0") and extra == ["-ub", "2048"],
      f"all on the GPU (0), default (no MOE_CACHE_LOOP_FIT): untouched, exactly 9114606's plan (main24 2026-11-23) [got {r0}]")
extra = ["-ub", "2048"]
r9 = mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "999", None, False)
check(r9 == ("2048", "999") and extra == ["-ub", "2048"], f"GPU-cache plan (999): placeholder, the GPU-cache stage resolves it later: untouched (loop-speed-2 critic) [got {r9}, {extra}]")

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
os.environ["MOE_CACHE_LOOP_FIT"] = "1"   # opt-in until PC item P1
T64 = {"compute_mib": 4660}            # tune() at 64k, IQ4_XS, 15.3 GiB free, long-prompt mode: -ub 4096, 13 CPU layers (checked 2026-11-23)
mc.ls_build = lambda ls: 12651; mc._LS_BUILD.clear()
extra = ["-ub", "4096", "-b", "4096"]
ub, nc = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check("--no-op-offload" not in extra and ub == "256" and extra[1] == "256", f"b12651 at 64k: op-offload on, -ub 256 (3072 bias + 690 + 547 staged = 4309 <= 4660) [got ub {ub}]")
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=131072, tuned={"compute_mib": 7958}, ls="ls")
check("--no-op-offload" not in extra and ub == "128", f"b12651 at 128k: -ub 256 fits the budget (7651 <= 7958) but its bias tensor (131072 x 256 x 24 x 4 B = 3 GiB) is over the 2 GiB per-tensor cap -> -ub 128 [got {ub}]")
# TOOLS-1309 c: the plan line that names the 2 GiB cap carries its R19 source label (assumed constant, not read from the device)
_said = []; _old_say = mc.say; mc.say = lambda m, *a, **k: _said.append(str(m))
try:
    mc.inkling_loop_stage(A(), ["-ub", "4096"], "m.gguf", "4096", "13", None, False, ctx=262144, tuned={"compute_mib": 20000}, ls="ls")      # -ub 128 alone: 3 GiB bias, over the cap, under the budget
finally:
    mc.say = _old_say
_cap = [l for l in _said if "per-tensor cap" in l]
check(len(_cap) == 1 and "our constant, assumed" in _cap[0] and "not read from the device" in _cap[0] and "2 GiB per-tensor cap" in _cap[0], f"the per-tensor cap line says it is an assumed constant [got {_cap}]")
_said.clear(); mc.say = lambda m, *a, **k: _said.append(str(m))
try:
    mc.inkling_loop_stage(A(), ["-ub", "4096"], "m.gguf", "4096", "13", None, False, ctx=65536, tuned={"compute_mib": 12000}, ls="ls")
finally:
    mc.say = _old_say
check(any("-ub 512 was skipped" in l and "our constant, assumed" in l for l in _said), f"a -ub kept out by the cap says so in the plan line [got {_said}]")
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=262144, tuned={"compute_mib": 7000}, ls="ls")
check("--no-op-offload" in extra and ub == "128", f"b12651 at 256k, budget 7000 MiB: even -ub 128 + staging (6144 bias + 577 loop_base_mib + 547) is over: op-offload off, -ub 128 [got {ub}]")
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=None, ls="ls")
check("--no-op-offload" in extra and ub == "128", "no planner budget: op-offload off, -ub 128 (the old rule)")
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned={"compute_mib": 900}, ls="ls")
check("--no-op-offload" in extra and ub == "128", "budget too small for staging at -ub 128: op-offload off")
mc.ls_build = lambda ls: 12207; mc._LS_BUILD.clear()
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check("--no-op-offload" in extra and ub == "256", f"b12207 (no PR 32807): op-offload off; -ub still sized to the budget without staging [got {ub}]")
mc.ls_build = lambda ls: 12651; mc._LS_BUILD.clear()
os.environ["MOE_CACHE_INKLING_OP_OFFLOAD"] = "0"
extra = ["-ub", "4096"]
mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check("--no-op-offload" in extra, "MOE_CACHE_INKLING_OP_OFFLOAD=0 forces it off")
os.environ["MOE_CACHE_INKLING_OP_OFFLOAD"] = "1"
mc.ls_build = lambda ls: None; mc._LS_BUILD.clear()
extra = ["-ub", "4096"]
mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check("--no-op-offload" not in extra, "MOE_CACHE_INKLING_OP_OFFLOAD=1 forces it on (unknown build)")
del os.environ["MOE_CACHE_INKLING_OP_OFFLOAD"]
mc.ls_build = lambda ls: 12651; mc._LS_BUILD.clear()
extra = ["-ub", "4096", "--no-op-offload"]
mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check(extra.count("--no-op-offload") == 1, "your --no-op-offload stands, once")
extra = ["-ub", "4096", "-ot", "blk\\.(6|7)\\.ffn_.*=CPU"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check(extra.count("-ot") == 1 and ub == "256" and "--no-op-offload" in extra, f"your -ot: all 12 loop layers counted as staged (errs high): op-offload off, -ub sized without staging [got {ub}]")

# engine-review-2 (2026-11-23): all on the GPU / cache plan / dense looped, at 64k with a budget
extra = ["-ub", "2048"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "0", None, False, ctx=32768, tuned={"compute_mib": 4660}, ls="ls")
check(ub == "512" and "--no-op-offload" not in extra, f"all on the GPU at 32k, budget 4660: -ub 512 (3072 + 770) [got {ub}]")
extra = ["-ub", "2048"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "999", None, False, ctx=65536, tuned=T64, ls="ls")
check(ub == "2048" and extra == ["-ub", "2048"], f"cache plan (999) at 64k on b12651, even with MOE_CACHE_LOOP_FIT=1: untouched, not decided before gpu_cache_stage [got {ub}]")
extra = ["-ub", "2048"]
ub, nc = mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "", None, False, ctx=65536, tuned=T64, ls="ls", dense=True)
check(ub == "256" and "--no-op-offload" not in extra and "-ot" not in extra, f"dense looped on b12651: no expert placement, op-offload on, -ub fit without staging [got {ub}]")
mc.ls_build = lambda ls: None; mc._LS_BUILD.clear()
extra = ["-ub", "2048"]
mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "", None, False, ctx=65536, tuned=T64, ls="ls", dense=True)
check("--no-op-offload" in extra, "dense looped, unknown build: op-offload off")
# loop-speed-2 critic fixes
mc.ls_build = lambda ls: 12651; mc._LS_BUILD.clear()
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=262144, tuned={"compute_mib": 40000}, ls="ls")
check("--no-op-offload" not in extra and ub == "128", f"256k, budget 40000: the 2 GiB tensor cap makes -ub 128 but does NOT turn op-offload off (it said 'no room' before) [got {ub}, {extra}]")
mc.ls_build = lambda ls: 12651; mc._LS_BUILD.clear()
a13 = A(); extra = ["-ub", "4096"]
mc.inkling_loop_stage(a13, extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check("--no-op-offload" not in extra and getattr(a13, "loop_min_batch", None) == int(extra[extra.index("-ub") + 1]), "op-offload on: loop_min_batch = -ub (stock_env_stage lowers GGML_OP_OFFLOAD_MIN_BATCH to it)")
META["inkling2_loop.attention.head_count"] = 0
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check(ub == "128" and "--no-op-offload" in extra, f"head count unknown: no fit, -ub 128, op-offload off [got {ub}]")
META["inkling2_loop.attention.head_count"] = [24] * 24
for v in ("false", "off", " 0"):
    os.environ["MOE_CACHE_INKLING_OP_OFFLOAD"] = v; extra = ["-ub", "4096"]
    mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
    check("--no-op-offload" in extra, f"MOE_CACHE_INKLING_OP_OFFLOAD={v!r} = off")
del os.environ["MOE_CACHE_INKLING_OP_OFFLOAD"]
os.environ["MOE_CACHE_LOOP_FIT"] = "true"; extra = ["-ub", "4096"]
mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check("--no-op-offload" not in extra, "MOE_CACHE_LOOP_FIT=true = on")
# default path (LOOP_FIT off): all on the GPU keeps the planner's -ub when its bias fits, drops to 128 when not
del os.environ["MOE_CACHE_LOOP_FIT"]
extra = ["-ub", "2048"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "0", None, False, ctx=4096, tuned={"compute_mib": 20000}, ls="ls")
check(ub == "2048" and extra == ["-ub", "2048"], f"default, all on the GPU, bias fits the budget: planner -ub kept [got {ub}]")
extra = ["-ub", "2048"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "0", None, False, ctx=32768, tuned={"compute_mib": 4660}, ls="ls")
check(ub == "2048", f"default, all on the GPU, 12 GiB bias over the budget: still untouched without MOE_CACHE_LOOP_FIT [got {ub}]")
extra = ["-ub", "2048"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "", None, False, ctx=65536, tuned=T64, ls="ls", dense=True)
check(ub == "2048" and extra == ["-ub", "2048"], "default, dense looped: untouched")
os.environ["MOE_CACHE_LOOP_FIT"] = "1"
extra = ["-ub", "2048"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "0", None, False, ctx=32768, tuned={"compute_mib": 4660}, ls="ls")
check(ub == "512", f"MOE_CACHE_LOOP_FIT=1, all on the GPU at 32k, budget 4660: -ub 512 fit [got {ub}]")
del os.environ["MOE_CACHE_LOOP_FIT"]
extra = ["-ub", "2048"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "13", None, False, ctx=65536, tuned=T64, ls="ls")
check(ub == "128" and "--no-op-offload" in extra, "default, experts on the CPU: old rule, -ub 128 + --no-op-offload (unchanged)")
os.environ["MOE_CACHE_LOOP_FIT"] = "1"
# stock_env_stage: the 512 threshold goes down to the looped model's -ub
env, _m = mc.stock_env_stage(types.SimpleNamespace(stock=False, api_stock=None, loop_min_batch=256), [], True, "13", 10e9, 40e9, "x")
check(("GGML_OP_OFFLOAD_MIN_BATCH", "256") in env, f"stock_env_stage: GGML_OP_OFFLOAD_MIN_BATCH <= the looped -ub [{env}]")
env, _m = mc.stock_env_stage(types.SimpleNamespace(stock=False, api_stock=None, loop_min_batch=256), [], True, "0", 10e9, 40e9, "x")
check(("GGML_OP_OFFLOAD_MIN_BATCH", "256") in env, "stock_env_stage through the ncmoe '0' placement path: threshold follows the looped -ub")
env, _m = mc.stock_env_stage(types.SimpleNamespace(stock=False, api_stock=None), [], True, "0", 10e9, 40e9, "x")
check(not any(k == "GGML_OP_OFFLOAD_MIN_BATCH" for k, v in env), "stock_env_stage, other models at ncmoe 0: unchanged (no threshold)")
os.environ["MOE_CACHE_INKLING_OP_OFFLOAD"] = "0"; extra = ["-ub", "2048"]
mc.inkling_loop_stage(A(), extra, "m.gguf", "2048", "999", None, False, ctx=65536, tuned=T64, ls="ls")
check(extra == ["-ub", "2048", "--no-op-offload"], f"999 with a forced MOE_CACHE_INKLING_OP_OFFLOAD=0: --no-op-offload only [{extra}]")
del os.environ["MOE_CACHE_INKLING_OP_OFFLOAD"]
# model_type: one line, read once; dense looped header (synthetic)
LAYOUT = {"n_layers": 24, "expert_bytes_per_layer": 0, "mtp_blocks": 0, "expert_bytes_by_layer": {}}
gguf_plan.model_layout = lambda p: dict(LAYOUT)
mt = mc.model_type("m.gguf")
check(mt and mt["dense"] and mt["loop"][2:5] == (6, 18, 3) and "dense (24 layers)" in mt["line"] and "looped: layers 6-17 x3" in mt["line"] and "source: the GGUF header" in mt["line"],
      f"model_type: a dense looped header -> dense + looped, one line with its source [{mt and mt['line']}]")
# dense_ffn_override: a dense looped model sends once-used FFN layers to RAM first
mc.planner_path = lambda: None
gguf_plan.dense_ffn_split = lambda *a, **k: {"n_layers": 24, "first_layer": 10, "n_off": 14, "ffn_gib": 0.3, "need_gib": 20, "free_gib": 15, "kv_gib": 2, "fits_all": False}
a = A(vram="16", gpu_margin=None, no_ffn_ram=False, cpu_gpu_split=None); a.model_type = mt
ot = mc.dense_ffn_override(a, "m.gguf", 65536, False, None, types.SimpleNamespace(free_vram_bytes=lambda: 16 * 2**30), [])
check(ot == "blk\\.(0|1|2|3|4|5|16|17|18|19|20|21|22|23)\\.ffn_(up|down|gate|gate_up)\\.weight=CPU",
      f"dense looped spill: 12 once-used layers + loop layers 16,17 (from the back) in RAM [{ot}]")
a.model_type = dict(mt, loop=None)
ot = mc.dense_ffn_override(a, "m.gguf", 65536, False, None, types.SimpleNamespace(free_vram_bytes=lambda: 16 * 2**30), [])
check(ot and ot.startswith("blk\\.(10|11|"), f"dense, not looped: unchanged (the last 14 layers) [{ot}]")
META.update({"inkling2_loop.loop.count": 1})
check(mc.model_type("m.gguf")["loop"] is None and "looped: no" in mc.model_type("m.gguf")["line"], "model_type: loop.count 1 -> not looped")

# LAUNCHER-1285: with MOE_CACHE_LOOP_FIT=1 the stage runs after gpu_cache_stage, without it before (default command unchanged)
src = open(os.path.join(HERE, "bin", "moe_cache.py")).read()
k = src.index("    if loop_fit_on():\n        ncmoe = gpu_cache_stage(")
check(src.index("ncmoe = gpu_cache_stage(", k) < src.index("ub, ncmoe = inkling_loop_stage(", k) < src.index("    else:\n        ub, ncmoe = inkling_loop_stage(", k) < src.index("ncmoe = gpu_cache_stage(", src.index("    else:\n        ub, ncmoe = inkling_loop_stage(", k)),
      "LAUNCHER-1285: LOOP_FIT=1 runs gpu_cache_stage then the loop stage; default keeps the old order (loop stage first)")
# LAUNCHER-1284: the stage's budget excludes the bias the planner already counted
rename("inkling2_loop"); gguf_plan.model_layout = lambda p: {"expert_bytes_by_layer": {i: (668 if i in (20, 21) else 547) * MiB for i in range(24)}}; os.environ["MOE_CACHE_LOOP_FIT"] = "1"; mc.ls_build = lambda ls: 12651; mc._LS_BUILD.clear()
extra = ["-ub", "4096"]
ub, _ = mc.inkling_loop_stage(A(), extra, "m.gguf", "4096", "13", None, False, ctx=65536, tuned={"compute_mib": 4660 + 3072, "loop_bias_mib": 3072}, ls="ls")
check(ub == "256", f"tuned with loop_bias_mib: the whole compute budget counts (LAUNCHER-1303: the bias is no longer taken off it); -ub 512 stays out by the 2 GiB per-tensor cap [got {ub}]")
del os.environ["MOE_CACHE_LOOP_FIT"]
# plugin long-prompt block and --no-op-offload (LOOP-SPEED-2)
Ah = lambda **kw: types.SimpleNamespace(**{"hybrid": False, **kw})
check(mc.plugin_no_op_offload_stage(Ah(), [], ["--no-op-offload"], False) == [("MOE_CACHE_GPU_MIN_TOKENS", "0")], "your --no-op-offload: plugin block off (MIN_TOKENS=0)")
check(mc.plugin_no_op_offload_stage(Ah(), [("MOE_CACHE_GPU_MIN_TOKENS", "64")], ["-nopo"], False) == [("MOE_CACHE_GPU_MIN_TOKENS", "0")], "-nopo replaces the 64 set by --fast")
check(mc.plugin_no_op_offload_stage(Ah(), [], ["-ub", "512"], False) == [], "no flag: env untouched (our own looped-model --no-op-offload never reaches user_extra)")
check(mc.plugin_no_op_offload_stage(Ah(hybrid=True), [], ["--no-op-offload"], False) == [] and mc.plugin_no_op_offload_stage(Ah(), [], ["--no-op-offload"], True) == [], "hybrid and stock runs untouched")
os.environ["MOE_CACHE_GPU_MIN_TOKENS"] = "32"
check(mc.plugin_no_op_offload_stage(Ah(), [], ["--no-op-offload"], False) == [], "your own MOE_CACHE_GPU_MIN_TOKENS wins")
del os.environ["MOE_CACHE_GPU_MIN_TOKENS"]

print("inkling_loop_test: " + ("all passed" if not fails else f"{len(fails)} failed"))
sys.exit(1 if fails else 0)
