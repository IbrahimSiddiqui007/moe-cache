"""LAUNCHER-1303/1236: the plugin's long-prompt block room at short contexts (planner/gguf_plan.py block_weight_share / small_ctx_block, launcher block_cap_stage,
plugin MOE_CACHE_GPU_BLOCK_CAP_MIB). Synthetic sparse GGUF headers (no weights, no model download):
  - a 35B-A3B-like model (the header twin of Qwen3.6-35B-A3B UD-IQ3_XXS, tests/twin35b.py);
  - an Inkling-2-Mini-like model (24 layers, ~558 MiB of experts per layer, ~1.1 GiB of other GPU weights), plain and looped (<arch>.loop.count 3, 24 heads).
Checks: on a 16 GB card (13.5-15.5 GiB of room) at 4k / 8k the plugin plan never keeps more expert layers on the CPU than before and keeps fewer somewhere in that range;
from 32k on (64k, 128k, 256k) nothing changes; the block room never falls under the batch's results + the plugin's headroom + 1/4 of a layer; the planned use never exceeds
the room; a looped model whose bias buffer the compute estimate does not count keeps its full block room; stock plans and GPU-cache plans are unchanged; the launcher passes
the cut room to the plugin (and only then); the plugin source caps the block with it. Forecasts of the planner, nothing here is measured on a GPU."""
import os
import struct
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
sys.path.insert(0, os.path.join(HERE, "tests"))
sys.path.insert(0, os.path.join(HERE, "bin"))
import gguf_plan   # noqa: E402
from twin35b import make_twin   # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails += 1


MiB, GiB = 1 << 20, 1 << 30


def write_gguf(path, meta, tensors):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = []
    for k, v in meta.items():
        kv.append(s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str) else struct.pack("<I", 4) + struct.pack("<I", v)))
    infos, off = [], 0
    for name, size in tensors:
        infos.append(s(name) + struct.pack("<I", 1) + struct.pack("<Q", 1) + struct.pack("<IQ", 0, off))
        off += size
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(kv)) + b"".join(kv) + b"".join(infos)
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head)
        f.truncate(len(head) + off)        # sparse: no weights written


def inkling_like(path, looped):
    """Sizes of Inkling-2-Mini-UD-IQ4_XS (read from the real header on the laptop 2026-11-25: 24 layers, 584811349 B of experts per layer, 1686 MB other, 492 MB token_embd)."""
    a = "loopmoe" if looped else "plainmoe"
    meta = {"general.architecture": a, f"{a}.block_count": 24, f"{a}.embedding_length": 2304, f"{a}.expert_count": 128, f"{a}.expert_used_count": 8,
            f"{a}.expert_feed_forward_length": 1152, f"{a}.attention.head_count": 24, f"{a}.attention.head_count_kv": 8, f"{a}.attention.key_length": 128,
            f"{a}.attention.value_length": 128, f"{a}.context_length": 1048576}
    if looped:
        meta.update({f"{a}.loop.count": 3, f"{a}.loop.start": 6, f"{a}.loop.end": 18})
    t = [("token_embd.weight", 492106752), ("output.weight", 492106752)]
    per = 584811349 // 3
    for i in range(24):
        t += [(f"blk.{i}.ffn_gate_exps.weight", per), (f"blk.{i}.ffn_up_exps.weight", per), (f"blk.{i}.ffn_down_exps.weight", per),
              (f"blk.{i}.attn_q.weight", (1686304128 - 2 * 492106752) // 24)]
    write_gguf(path, meta, t)


tmp = tempfile.mkdtemp(prefix="block-room-")
TWIN = os.path.join(tmp, "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf")
make_twin(TWIN)
INK = os.path.join(tmp, "inkling-like.gguf")
inkling_like(INK, looped=False)
INKL = os.path.join(tmp, "inkling-like-looped.gguf")
inkling_like(INKL, looped=True)


def plan(path, ctx, free_gib, old=False, **kw):
    """tune() as the launcher calls it for a static plan on the room figure; old=True: the rule before LAUNCHER-1303 (the full layer at every context)."""
    keep = gguf_plan.BLOCK_MIN_WEIGHT_SHARE
    if old:
        gguf_plan.BLOCK_MIN_WEIGHT_SHARE = 1.0
    try:
        return gguf_plan.tune(path, ctx, free=free_gib * GiB, want_cache=kw.pop("want_cache", False), plugin=kw.pop("plugin", True), free_source="room", **kw)
    finally:
        gguf_plan.BLOCK_MIN_WEIGHT_SHARE = keep


def block_mib(t):
    return next(r["value"] for r in t["inputs"] if r["setting"] == "long_prompt_block_mib")


check(gguf_plan.block_weight_share(4096) == 0.25 and gguf_plan.block_weight_share(16384) == 0.5 and gguf_plan.block_weight_share(32768) == 1.0
      and gguf_plan.block_weight_share(262144) == 1.0 and gguf_plan.block_weight_share(None) == 1.0, "weight share: 1/4 floor, ctx/32k below 32k, 1 from 32k on and when unknown")
check(gguf_plan.loop_heads(INKL) == 24 and gguf_plan.loop_heads(INK) == 0 and gguf_plan.loop_heads(TWIN) == 0, "looped model read from the header (loop.count > 1), not from a name")

FREES = [13.5 + 0.1 * i for i in range(21)]          # 13.5 .. 15.5 GiB of room on a 16 GB card
for name, path in (("35B-like", TWIN), ("Inkling-like", INK)):
    lay = gguf_plan.model_layout(path)
    for ctx in (4096, 8192):
        fewer, worse, over, small = 0, [], [], []
        for f in FREES:
            o, n = plan(path, ctx, f, old=True), plan(path, ctx, f)
            fewer += n["n_cpu_moe"] < o["n_cpu_moe"]
            if n["n_cpu_moe"] > o["n_cpu_moe"]:
                worse.append(f)
            used = (gguf_plan.gpu_fixed_weights_bytes(lay) + n["kv_mib"] * MiB + n["compute_mib"] * MiB + block_mib(n) * MiB + 0.5e9
                    + gguf_plan.gpu_expert_bytes(lay, n["n_cpu_moe"]))
            if used > f * GiB + 2 * MiB:
                over.append(f)
            floor_b = (0.25 + 0.6) * lay["expert_bytes_per_layer"] + 416 * MiB          # results (0.6 of a layer at -ub 2048) + 160 + 256 MiB + 1/4 layer
            if n["ub"] == 2048 and block_mib(n) * MiB < floor_b - 2 * MiB:
                small.append(f)
            if f == 14.0:
                print(f"     forecast {name} ctx {ctx} room 14.0 GiB: --n-cpu-moe {o['n_cpu_moe']} -> {n['n_cpu_moe']}, block {block_mib(o)} -> {block_mib(n)} MiB, cap {n['block_cap_mib']}")
        check(not worse, f"{name} ctx {ctx}: never more CPU layers than before over 13.5-15.5 GiB ({worse or 'none worse'})")
        check(fewer > 0, f"{name} ctx {ctx}: fewer CPU layers than before at {fewer} of {len(FREES)} room figures")
        check(not over, f"{name} ctx {ctx}: planned use (weights, KV, compute, block, 0.5 GB reserve, GPU experts) within the room ({over or 'all within'})")
        check(not small, f"{name} ctx {ctx}: the block room keeps 1/4 layer + the batch's results + the plugin's headroom ({small or 'all'})")
    for ctx in (32768, 65536, 131072, 262144):
        diff = [f for f in FREES if (lambda o, n: (o["n_cpu_moe"], o["ub"], block_mib(o)) != (n["n_cpu_moe"], n["ub"], block_mib(n)) or n["block_cap_mib"])(plan(path, ctx, f, old=True), plan(path, ctx, f))]
        check(not diff, f"{name} ctx {ctx}: the same plan as before, no cap ({diff or 'identical'})")

# the cut is kept only when it frees an expert layer (critic 2026-11-25): a cap exactly where the plan has fewer CPU layers than the full-room plan; elsewhere the
# full-room plan itself (same --n-cpu-moe, same block, no cap). cut_block=False (--exact, MOE_CACHE_GPU_CHUNKS=0) is the full-room plan at every room figure.
for name, path in (("35B-like", TWIN), ("Inkling-like", INK), ("looped Inkling-like", INKL)):
    for ctx in (4096, 8192, 31744):
        bad, offbad = [], []
        for f in FREES + [2.0, 6.0]:
            o, n, x = plan(path, ctx, f, old=True), plan(path, ctx, f), plan(path, ctx, f, cut_block=False)
            if bool(n["block_cap_mib"]) != (n["n_cpu_moe"] < o["n_cpu_moe"]) or (not n["block_cap_mib"] and (n["n_cpu_moe"], n["ub"], block_mib(n)) != (o["n_cpu_moe"], o["ub"], block_mib(o))):
                bad.append(f)
            if (x["n_cpu_moe"], x["ub"], block_mib(x), x["block_cap_mib"]) != (o["n_cpu_moe"], o["ub"], block_mib(o), 0):
                offbad.append(f)
        check(not bad, f"{name} ctx {ctx}: a cap only where the cut frees a layer, else the full-room plan ({bad or 'all'})")
        check(not offbad, f"{name} ctx {ctx}: cut_block=False (--exact / MOE_CACHE_GPU_CHUNKS=0) = the full-room plan, no cap ({offbad or 'all'})")
allcpu = plan(TWIN, 4096, 2.0)
check(allcpu["n_cpu_moe"] == gguf_plan.model_layout(TWIN)["n_layers"] and not allcpu["block_cap_mib"], f"all expert layers on the CPU (2 GiB room): no cap ({allcpu['n_cpu_moe']})")

# a looped model whose bias buffer is not counted (loop_n_head=0, as --exact plans it): the room the plan cannot see is never given to expert layers
for ctx in (4096, 8192):
    same = all(plan(INKL, ctx, f, loop_n_head=0)["n_cpu_moe"] == plan(INKL, ctx, f, old=True, loop_n_head=0)["n_cpu_moe"] for f in FREES)
    n = plan(INKL, ctx, 14.0, loop_n_head=0)
    check(same and block_mib(n) == block_mib(plan(INKL, ctx, 14.0, old=True, loop_n_head=0)) and n["block_cap_mib"] == 0,
          f"looped Inkling-like ctx {ctx} -ub {n['ub']}: bias {gguf_plan.loop_bias_bytes(ctx, n['ub'], 24) // MiB} MiB uncounted >= the cut: full block, same pick, no cap")
# LAUNCHER-1303 critic: by default (loop_n_head None, no loop_ubs) the bias is not counted (CUDA has no dense bias buffer): the same plan as loop_n_head=0, as before 2026-11-25
dflt = plan(INKL, 4096, 14.2)
check(dflt["loop_bias_mib"] == 0 and dflt == plan(INKL, 4096, 14.2, loop_n_head=0), f"looped, default without loop_ubs: bias not counted, = loop_n_head 0 (n_cpu_moe {dflt['n_cpu_moe']}, cap {dflt['block_cap_mib']})")
# ... and when the bias is counted (loop_n_head given), the cut applies again
nc = plan(INKL, 4096, 14.2, loop_n_head=24)
check(block_mib(nc) < block_mib(plan(INKL, 4096, 14.2, old=True, loop_n_head=24)) and nc["block_cap_mib"] > 0, f"looped, bias counted: the block room is cut ({block_mib(nc)} MiB, cap {nc['block_cap_mib']})")
# a measured compute buffer already holds the bias: the cut applies
nm = plan(INKL, 4096, 14.2, measured={"buffers": {2048: {"compute_mib": 2600}}})
check(nm["block_cap_mib"] > 0, "looped, compute buffer measured at the last load: the cut applies")

# stock and GPU-cache plans are untouched
for path in (TWIN, INK):
    for ctx in (4096, 8192):
        s_same = all(plan(path, ctx, f, plugin=False)["n_cpu_moe"] == plan(path, ctx, f, plugin=False, old=True)["n_cpu_moe"] for f in FREES)
        c_new, c_old = plan(path, ctx, 7.6, want_cache=True), plan(path, ctx, 7.6, want_cache=True, old=True)
        check(s_same and c_new["gpu_cache_mib"] == c_old["gpu_cache_mib"] and c_new["n_cpu_moe"] == c_old["n_cpu_moe"] and not c_new["block_cap_mib"],
              f"{os.path.basename(path)} ctx {ctx}: stock plan and GPU-cache plan unchanged, no cap")

# the rule path (MOE_CACHE_PLAN_RULE=1) uses the same cut
os.environ["MOE_CACHE_PLAN_RULE"] = "1"
try:
    r_new, r_old = plan(TWIN, 4096, 14.0), plan(TWIN, 4096, 14.0, old=True)
    check(block_mib(r_new) < block_mib(r_old) and r_new["n_cpu_moe"] <= r_old["n_cpu_moe"] and abs(r_new["block_cap_mib"] - block_mib(r_new)) <= 1,
          f"plan rule: the block room is cut the same way ({block_mib(r_old)} -> {block_mib(r_new)} MiB)")
    r64 = plan(TWIN, 65536, 14.0)
    check(block_mib(r64) == block_mib(plan(TWIN, 65536, 14.0, old=True)) and not r64["block_cap_mib"], "plan rule at 64k: unchanged")
finally:
    del os.environ["MOE_CACHE_PLAN_RULE"]

# multigpu.plan() on one card places as tune() does (same block)
import multigpu   # noqa: E402
for ctx in (4096, 65536):
    m1, t1 = multigpu.plan(TWIN, ctx, [14 * GiB]), plan(TWIN, ctx, 14.0)
    check(m1 is not None and m1["n_cpu_moe"] == t1["n_cpu_moe"], f"multigpu one card ctx {ctx}: {m1 and m1['n_cpu_moe']} = tune {t1['n_cpu_moe']}")

# launcher: block_cap_stage
import moe_cache as mc   # noqa: E402
t = plan(TWIN, 4096, 14.2)
env = mc.block_cap_stage([("A", "1")], t, False, None)
check(("MOE_CACHE_GPU_BLOCK_CAP_MIB", str(t["block_cap_mib"])) in env, f"launcher passes the cut room to the plugin ({env[-1]})")
check(mc.block_cap_stage([("A", "1")], t, True, None) == [("A", "1")], "stock: no cap")
check(mc.block_cap_stage([("A", "1")], t, False, "3") == [("A", "1")], "your --n-cpu-moe: no cap (the plan's room is not what runs)")
check(mc.block_cap_stage([("A", "1")], plan(TWIN, 65536, 14.0), False, None) == [("A", "1")], "64k: no cap (full room, as before)")
check(mc.block_cap_stage([("MOE_CACHE_GPU_BLOCK_CAP_MIB", "9")], t, False, None) == [("MOE_CACHE_GPU_BLOCK_CAP_MIB", "9")], "a value already set wins")
check(mc.block_cap_stage([], None, False, None) == [], "no plan: nothing")
check(mc.block_cap_stage([("A", "1")], t, False, None, str(t["n_cpu_moe"])) == [("A", "1"), ("MOE_CACHE_GPU_BLOCK_CAP_MIB", str(t["block_cap_mib"]))], "the plan's own --n-cpu-moe runs: cap")
check(mc.block_cap_stage([("A", "1")], t, False, None, "999") == [("A", "1")], "--hybrid / a GPU cache (n-cpu-moe 999 runs, not the plan's): no cap")
check(mc.block_cap_stage([("A", "1")], t, False, None, str(t["n_cpu_moe"]), exact=True) == [("A", "1")], "--exact: no cap")
check(mc.block_cap_stage([("MOE_CACHE_GPU_CHUNKS", "0")], t, False, None) == [("MOE_CACHE_GPU_CHUNKS", "0")], "MOE_CACHE_GPU_CHUNKS=0 in the launch env: no cap (a capped block never fits)")
os.environ["MOE_CACHE_GPU_CHUNKS"] = "0"
try:
    check(mc.block_cap_stage([], t, False, None) == [] and not mc.block_cut_on(types.SimpleNamespace(exact=False)), "MOE_CACHE_GPU_CHUNKS=0 in your environment: no cut, no cap")
finally:
    del os.environ["MOE_CACHE_GPU_CHUNKS"]
check(mc.block_cut_on(types.SimpleNamespace(exact=False)) and not mc.block_cut_on(types.SimpleNamespace(exact=True)), "the launcher asks for the cut by default, not with --exact")

# plugin source: the cap goes through the tested free_for_block helper, on the first card only
src = open(os.path.join(HERE, "src", "mc_backend.inl"), encoding="utf-8").read()
check('getenv("MOE_CACHE_GPU_BLOCK_CAP_MIB")' in src and "gpu_free = mcshared::free_for_block(gpu_free, held, (size_t) block_cap_mib * 1048576);" in src
      and src.index("MOE_CACHE_GPU_BLOCK_CAP_MIB") < src.index("auto fits = [&](size_t need)"), "plugin: the block's free figure is capped before fits() decides whole / split")
doc = open(os.path.join(HERE, "docs", "reference", "configuration.md"), encoding="utf-8").read()
check("`MOE_CACHE_GPU_BLOCK_CAP_MIB`" in doc, "configuration.md documents MOE_CACHE_GPU_BLOCK_CAP_MIB")

print(f"{fails} failed" if fails else "all passed")
sys.exit(1 if fails else 0)
