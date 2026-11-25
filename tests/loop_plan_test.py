"""LAUNCHER-1303 / 1284: the planner's numbers for a LOOPED model (Inkling 2 Mini today), model-free (synthetic sparse GGUF headers, no weights, no GPU).
  - KV cache + conv state (planner/gguf_plan.py loop_kv_bytes): one cache per pass through a loop layer, sliding-window layers capped at the window x slots + -ub,
    against the PC's -lv 4 buffer lines (2026-11-25, chain29, RX 9070 XT, Inkling 2 Mini UD-IQ4_XS, f16): 768 MiB KV + 14.6 MiB state at 4k, 2742.6 MiB at 64k.
  - the dense relative-position bias is counted by default (loop_n_head None = read from the header), never twice (a measured load, llama-fit-params);
  - loop_plan: when the launcher's looped-model rule runs a small -ub with experts on the CPU, the plan is made at that -ub (PC 4k: our plan kept 6 layers on the CPU
    where stock ran 4 at the same -ub 128 and left 3.2 GiB unused); a plan that keeps every expert on the GPU or picks a GPU cache is left alone;
  - the compute buffer estimate at -ub 128 against the PC (153.3 MiB at 4k, 1699.2 MiB at 64k).
Forecasts of the planner checked against recorded buffer sizes; no speed is measured here."""
import os
import shutil
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
sys.path.insert(0, os.path.join(HERE, "bin"))
import gguf_plan   # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails += 1


MiB, GiB = 1 << 20, 1 << 30


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _val(v):
    if isinstance(v, str):
        return struct.pack("<I", 8) + _s(v)
    if isinstance(v, bool):
        return struct.pack("<I", 7) + struct.pack("<?", v)
    if isinstance(v, list):
        if all(isinstance(x, bool) for x in v):
            return struct.pack("<I", 9) + struct.pack("<IQ", 7, len(v)) + b"".join(struct.pack("<?", x) for x in v)
        return struct.pack("<I", 9) + struct.pack("<IQ", 4, len(v)) + b"".join(struct.pack("<I", x) for x in v)
    return struct.pack("<I", 4) + struct.pack("<I", v)


def write_gguf(path, meta, tensors):
    kv = [_s(k) + _val(v) for k, v in meta.items()]
    infos, off = [], 0
    for name, size in tensors:
        infos.append(_s(name) + struct.pack("<I", 1) + struct.pack("<Q", 1) + struct.pack("<IQ", 0, off))
        off += size
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(kv)) + b"".join(kv) + b"".join(infos)
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head)
        f.truncate(len(head) + off)


PATTERN = [(i % 6) != 5 for i in range(24)]          # Inkling 2 Mini: every 6th layer full attention, the rest a 1024-token window


def inkling(path, looped=True, swa=True):
    """The header facts of Inkling-2-Mini-UD-IQ4_XS that the planner reads (read from the real file on the laptop 2026-11-25): 24 layers, 24 heads, 8 KV heads,
    128 key/value length, window 1024, loop layers 6-17 x3, shortconv l_cache 4; ~558 MiB of experts per layer, 1686 MB other weights (492 MB token_embd)."""
    a = "inkling2_loop" if looped else "plainswa"
    meta = {"general.architecture": a, f"{a}.block_count": 24, f"{a}.embedding_length": 2304, f"{a}.expert_count": 128, f"{a}.expert_used_count": 8,
            f"{a}.expert_feed_forward_length": 1152, f"{a}.attention.head_count": [24] * 24, f"{a}.attention.head_count_kv": [8] * 24,
            f"{a}.attention.key_length": 128, f"{a}.attention.value_length": 128, f"{a}.context_length": 1048576, f"{a}.shortconv.l_cache": 4}
    if swa:
        meta.update({f"{a}.attention.sliding_window": 1024, f"{a}.attention.sliding_window_pattern": PATTERN})
    if looped:
        meta.update({f"{a}.loop.count": 3, f"{a}.loop.start": 6, f"{a}.loop.end": 18})
    t = [("token_embd.weight", 492106752), ("output.weight", 492106752)]
    per = 584811349 // 3
    for i in range(24):
        t += [(f"blk.{i}.ffn_gate_exps.weight", per), (f"blk.{i}.ffn_up_exps.weight", per), (f"blk.{i}.ffn_down_exps.weight", per),
              (f"blk.{i}.attn_q.weight", (1686304128 - 2 * 492106752) // 24)]
    write_gguf(path, meta, t)


tmp = tempfile.mkdtemp(prefix="loop-plan-")
INK = os.path.join(tmp, "inkling-loop.gguf")
inkling(INK)
PLAIN = os.path.join(tmp, "plain-swa.gguf")
inkling(PLAIN, looped=False)

# 1. KV cache + state, against the PC's buffer lines
kv4, kv64 = gguf_plan.kv_bytes(INK, 4096) / MiB, gguf_plan.kv_bytes(INK, 65536) / MiB
check(782.6 <= kv4 <= 782.6 * 1.02, f"4k f16: {kv4:.1f} MiB vs 768 KV + 14.6 state = 782.6 measured on the PC (within +2 %, never under)")
check(2742.6 <= kv64 <= 2742.6 * 1.03, f"64k f16: {kv64:.1f} MiB vs 2742.6 MiB recorded on the PC (within +3 %, never under)")
check(gguf_plan.kv_bytes(INK, 4096, True) / MiB < kv4 * 0.6, "q8_0 KV: about half")
check(gguf_plan.kv_bytes(PLAIN, 4096) is None, "a sliding-window model that is not looped keeps the old answer (None: flat 40 KB per token), unchanged")

# 2. the bias is counted by default, never twice
free_pc = 15419 * MiB              # PC: Vulkan1 15419 MiB free at the 4k item (chain29 -lv 4)
t0 = gguf_plan.tune(INK, 4096, free=free_pc, want_cache=False, ub=128)
check(t0["loop_bias_mib"] == 96, f"default: bias read from the header, 2 x 4096 x 128 x 24 x 4 B = 96 MiB ({t0['loop_bias_mib']})")
check(gguf_plan.tune(INK, 4096, free=free_pc, want_cache=False, ub=128, loop_n_head=0)["loop_bias_mib"] == 0, "loop_n_head=0 (--exact): not counted")
tm = gguf_plan.tune(INK, 4096, free=free_pc, want_cache=False, ub=128, measured={"buffers": {128: {"compute_mib": 153}}})
check(tm["loop_bias_mib"] == 0 and tm["compute_mib"] == 153, "a measured load at that -ub already holds the bias: not added twice")
keep = (gguf_plan.plan_rule_active, gguf_plan.fit_params_buffer_mib)
try:
    gguf_plan.plan_rule_active = lambda fp: True
    gguf_plan.fit_params_buffer_mib = lambda fp, path, ctx, u, kv_q8=False: 150.0
    tf = gguf_plan.tune(INK, 4096, free=free_pc, want_cache=False, ub=128, fit_params="/fake/llama-fit-params")
finally:
    gguf_plan.plan_rule_active, gguf_plan.fit_params_buffer_mib = keep
check(tf["loop_bias_mib"] == 0 and tf["compute_mib"] == 150.0, f"llama-fit-params measured this model (it holds the bias): not added twice ({tf['compute_mib']})")
inp = {r["setting"]: r for r in t0["inputs"]}
check("loop_bias_mib" in inp and inp["loop_bias_mib"]["value"] == 96, "the plan file names the bias (its own inputs row, R17)")

# 3. compute buffer at -ub 128 against the PC
c4 = gguf_plan.tune(INK, 4096, free=free_pc, want_cache=False, ub=128)["compute_mib"]
c64 = gguf_plan.tune(INK, 65536, free=free_pc, want_cache=False, ub=128)["compute_mib"]
check(153.3 <= c4 <= 153.3 * 1.5, f"4k -ub 128: {c4:.0f} MiB vs 153.3 measured on the PC (never under, at most +50 %)")
check(1699.2 <= c64 <= 1699.2 * 1.15, f"64k -ub 128: {c64:.0f} MiB vs 1699.2 recorded on the PC (never under, at most +15 %)")

# 4. loop_plan: the plan at the rule's -ub
old = gguf_plan.tune(INK, 4096, free=free_pc, want_cache=False)                   # the usual ladder (what the launcher planned before; the rule then ran -ub 128)
new = gguf_plan.tune(INK, 4096, free=free_pc, want_cache=False, loop_ubs=(128,))
check(new["ub"] == 128 and new["n_cpu_moe"] < old["n_cpu_moe"], f"4k, 15419 MiB: planned at -ub 128 -> {new['n_cpu_moe']} CPU layers (usual ladder at -ub {old['ub']}: {old['n_cpu_moe']})")
check(new["n_cpu_moe"] <= 4, f"4k on the PC's free figure: no more CPU layers than stock's 4 ({new['n_cpu_moe']})")
lay = gguf_plan.model_layout(INK)
blk = next(r["value"] for r in new["inputs"] if r["setting"] == "long_prompt_block_mib")
at_load = gguf_plan.gpu_fixed_weights_bytes(lay) + new["kv_mib"] * MiB + new["compute_mib"] * MiB + gguf_plan.gpu_expert_bytes(lay, new["n_cpu_moe"])
check(at_load + blk * MiB + 0.5e9 <= free_pc, f"weights + KV + compute + block room ({blk} MiB) + 0.5 GB reserve fit the free figure ({(free_pc - at_load) / MiB:.0f} MiB free at load)")
check(any("looped-model rule runs -ub 128" in w for w in new["why"]) and any(r["setting"] == "loop_plan_ub" for r in new["inputs"]), "the plan says why -ub 128 and records it")
big = gguf_plan.tune(INK, 4096, free=40 * GiB, want_cache=False, loop_ubs=(128,))
check(big["n_cpu_moe"] == 0 and big["ub"] == old["ub"] if old["ub"] else True, f"every expert fits: the usual plan stands (-ub {big['ub']}, the rule does not apply with nothing on the CPU)")
cache = gguf_plan.tune(INK, 4096, free=8 * GiB, want_cache=True, loop_ubs=(128,))
check(cache["n_cpu_moe"] in (999,) or cache.get("gpu_cache_mib") == 0, "a GPU-cache plan is left to the cache stage")
fit = gguf_plan.tune(INK, 8192, free=free_pc, want_cache=False, loop_ubs=(512, 256, 128))
r128 = gguf_plan.tune(INK, 8192, free=free_pc, want_cache=False, ub=128)
check(fit["ub"] in (512, 256, 128) and fit["n_cpu_moe"] == r128["n_cpu_moe"], f"MOE_CACHE_LOOP_FIT ladder: the fewest CPU layers, then the largest -ub (-ub {fit['ub']}, {fit['n_cpu_moe']} layers)")
for f in (7.5, 11.0, 15.06, 23.0):
    lp = gguf_plan.tune(INK, 4096, free=f * GiB, want_cache=False, loop_ubs=(128,))
    od = gguf_plan.tune(INK, 4096, free=f * GiB, want_cache=False)
    check(lp["n_cpu_moe"] <= od["n_cpu_moe"], f"{f} GiB free: never more CPU layers than the usual ladder ({lp['n_cpu_moe']} vs {od['n_cpu_moe']})")

# 5. the launcher passes the rule's -ub (source check: tune_launch and inkling_loop_stage share LOOP_RULE_UB)
src = open(os.path.join(HERE, "bin", "moe_cache.py"), encoding="utf-8").read()
check('kw["loop_ubs"] = LOOP_UBS if loop_fit_on() else (LOOP_RULE_UB,)' in src and "pick = LOOP_RULE_UB" in src, "the launcher plans at the -ub its looped-model rule runs")
check('kw["loop_n_head"] = 0' in src, "--exact / --cpu-only keep the old plan (bias not counted)")

shutil.rmtree(tmp, ignore_errors=True)
print("loop plan checks: " + (f"{fails} failed" if fails else "all passed"))
sys.exit(1 if fails else 0)
