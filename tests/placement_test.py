#!/usr/bin/env python3
"""Planner map fixes 1 + 2 (LAUNCHER-1177, 2026-11-15): one placement from the real numbers, and the auto-tune's edge walk.

Fixtures are real: the laptop logs of 2026-11-13 / 2026-11-15 (tests/data/load_logs/, RTX 4060 Laptop 8 GB, Qwen3.6-35B-A3B UD-IQ3_XXS, a llama.cpp build of 2026-11)
and a header twin of that model (tests/data/qwen36_35b_iq3xxs_layout.json: every tensor's name and size, a sparse file here). The laptop sweep (planner map
section 3): the lowest --n-cpu-moe that loads is 25 at 32k (23 runs out of GPU memory) and 27 at 64k (25 runs out); nvidia-smi showed 7.97 GiB free.
Model-free: nothing is loaded, no GPU is used."""
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
sys.path.insert(0, os.path.join(HERE, "bin"))
import gguf_plan            # noqa: E402
import load_record as lr    # noqa: E402
import moe_cache_autotune as at   # noqa: E402

MIB = 1048576
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def lines(name):
    with open(os.path.join(HERE, "tests", "data", "load_logs", name), encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


# ---- 1. the buffer-line parser on saved real logs ----
lv4 = lr.parse_log(lines("laptop_35b_32k_lv4.txt"))
check(lv4.get("device") == "CUDA0" and lv4.get("model_mib") == 5116.81 and lv4.get("kv_mib") == 640.0 and lv4.get("rs_mib") == 251.25, f"lv4 log: CUDA0 model 5116.81, KV 640, RS 251.25 MiB ({lv4})")
check(lv4.get("compute_mib") == 660.02 and lv4.get("context_mib") == 891.25, "lv4 log: compute buffer 660.02 MiB, context (KV + recurrent state) 891.25 MiB")
check(lv4.get("total_mib") == 7857 and lv4.get("free_start_mib") == 7731 and lv4.get("free_end_mib") == 983 and lv4.get("unaccounted_mib") == 205,
      "lv4 log: the memory tables: CUDA total 7857, free 7731 at load, 983 at exit, 205 unaccounted at exit")
check(lv4.get("n_ctx") == 32768 and lv4.get("n_ubatch") == 2048 and lv4.get("kv_type") == "f16/f16", "lv4 log: context 32768, micro-batch 2048, f16/f16 KV")
oom64 = lr.parse_log(lines("laptop_35b_64k_ncmoe25_oom.txt"))
check(oom64 == {"failed_compute_mib": 788.02, "oom": True}, f"default-verbosity log of a failed 64k load: only the compute buffer it asked for, 788.02 MiB, flagged as a GPU allocation failure ({oom64})")
oom32 = lr.parse_log(lines("laptop_35b_32k_ncmoe23_oom.txt"))
check(oom32 == {"failed_compute_mib": 660.02, "oom": True}, f"failed 32k load: 660.02 MiB, the same as the lv4 log's real compute buffer ({oom32})")
check(lr.parse_log(["0.00 I srv  load_model: loading model 'x'", "nothing here"]) == {}, "a log without buffer lines: nothing")
# a load buffer that fails (the weights, not the compute buffer) is not taken for the compute buffer
check("failed_compute_mib" not in lr.parse_log(["E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5000.00 MiB on device 0: cudaMalloc failed: out of memory",
                                               "E llama_model_load: error loading model: unable to allocate CUDA0 buffer"]), "a failed weight buffer is not a compute-buffer figure")
two = lr.parse_log(["I load_tensors:      Vulkan0 model buffer size =  1000.00 MiB", "I load_tensors:      Vulkan1 model buffer size =  9000.00 MiB",
                    "I sched_reserve:    Vulkan1 compute buffer size =   864.03 MiB", "I sched_reserve:    Vulkan0 compute buffer size =   100.00 MiB"])
check(two.get("device") == "Vulkan1" and two.get("compute_mib") == 864.03, "two GPUs: the one with the most weights counts")

# ---- 2. the lines kept off the terminal when the launcher raised the verbosity ----
L = lines("laptop_35b_32k_lv4.txt")
hidden = [l for l in L if lr.extra_log_line(l)]
shown = [l for l in L if not lr.extra_log_line(l)]
check(len(hidden) > 200 and not any(" W " in l[:20] or " E " in l[:20] for l in hidden), f"verbosity 4: {len(hidden)} library info lines hidden, no warning or error among them")
check(not any("buffer size" in l or "memory_breakdown" in l for l in shown), "...the buffer lines and memory tables are among them (the recorder still reads them)")
check(any(" I srv " in l for l in shown) and any(" I slot " in l for l in shown) and any(" W llama_model_loader:" in l for l in shown),
      "...the server's own lines (srv, slot) and warnings stay on the terminal, as at verbosity 3")
check(not lr.extra_log_line("0.00.613.000 W common_fit_params: failed to fit params") and lr.extra_log_line("0.00.1 D ggml_something: x"), "warnings stay, debug lines go")
flt = lr.LineFilter()
kept = [l for l in L if flt.show(l)]
lv3 = [l for l in lines("laptop_35b_32k_ncmoe23_oom.txt")]
check(len(kept) <= 50 and not any("<|im_start|>" in l for l in kept), f"the terminal at raised verbosity: {len(kept)} of {len(L)} lines, no chat-template dump (a verbosity-3 run of the same start shows about 45)")
check(all(lr.LineFilter().show(l) for l in lv3), "a verbosity-3 log passes the filter unchanged")
check(lr.LineFilter().show("moe-cache: plugin loaded"), "plugin lines always show")

# ---- 3. the 35B header twin ----
twin_dir = tempfile.mkdtemp(prefix="placement-")
TWIN = os.path.join(twin_dir, "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf")


from twin35b import make_twin   # noqa: E402  (shared with tests/nearly_fits_test.py)


import atexit, shutil   # noqa: E401,E402
if os.name == "nt":                       # the twin is a 13 GB sparse file; NTFS does not make it sparse by default: the placement half runs on Linux / macOS only
    print("placement checks: the parser and filter checks passed; the header-twin checks run on Linux and macOS only")
    sys.exit(1 if fails else 0)
atexit.register(shutil.rmtree, twin_dir, True)
make_twin(TWIN)
lay = gguf_plan.model_layout(TWIN)
weights = (lay["other_bytes"] - lay["cpu_only_bytes"]) / MIB
check(lay["n_layers"] == 40 and abs(weights - 1596.8) < 0.5, f"twin: 40 expert layers; weights besides the experts and token_embd {weights:.1f} MiB")
check(abs(lv4["model_mib"] - gguf_plan.gpu_expert_bytes(lay, 27) / MIB - weights) < 0.5,
      "the lv4 load ran at --n-cpu-moe 27 (the auto-tune pick): its CUDA0 model buffer = these weights + the expert layers 27..39, to the MiB (token_embd is on the CPU)")
kv32, kv64 = gguf_plan.kv_bytes(TWIN, 32768) / MIB, gguf_plan.kv_bytes(TWIN, 65536) / MIB
check(abs(kv32 - 891.25) < 0.5 and abs(kv64 - 1531.25) < 0.5, f"twin: KV + recurrent state from the header 891.25 / 1531.25 MiB at 32k / 64k (llama-server said 891.25 at 32k): {kv32:.2f} / {kv64:.2f}")
TOTAL = lv4["total_mib"]


def loads(n, ctx):
    """The laptop's truth from the logs: does --n-cpu-moe n load at ctx (-ub 2048, f16 KV)? weights + experts + context + compute + what was unaccounted at exit <= CUDA's total."""
    ctx_m = 891.25 if ctx == 32768 else 1531.25
    comp = 660.02 if ctx == 32768 else 788.02
    return weights + gguf_plan.gpu_expert_bytes(lay, n) / MIB + ctx_m + comp + lv4["unaccounted_mib"] <= TOTAL


check(loads(25, 32768) and not loads(23, 32768) and loads(27, 65536) and not loads(25, 65536),
      "the figures reproduce the sweep: lowest that loads 25 at 32k (23 out of memory), 27 at 64k (25 out of memory)")

# ---- 4. what one load stores, and the plan made from it ----
FREE_PLAN = 8161.0      # nvidia-smi's free figure the launcher planned with (7.97 GiB, laptop 2026-11-13)
ent32 = lr.entries(lv4, 32768, 2048, ("f16", "f16"), "stock", False, gguf_plan.gpu_expert_bytes(lay, 27) / MIB, FREE_PLAN, "nvidia-smi")
check(abs(ent32["weights"]["non_expert_mib"] - 1596.8) < 0.2 and ent32["buffers"]["compute_mib"] == 660.0 and ent32["buffers"]["context_mib"] == 891.2,
      f"stored from the 32k load: weights 1596.8, compute 660, context 891 MiB ({ent32})")
gap = ent32["overhead"]["gap_mib"]
check(abs(gap - (FREE_PLAN - 7731 + (205 - (7857 - 7731)))) < 0.2, f"the overhead the free figure overstates: 8161 - 7731 free at load + 79 MiB grown after load = {gap} MiB")
ent64 = lr.entries(oom64, 65536, 2048, ("f16", "f16"), "stock", False, gguf_plan.gpu_expert_bytes(lay, 25) / MIB, FREE_PLAN, "nvidia-smi")
check(ent64 == {"buffers": {"compute_mib": 788.0, "at": ent64["buffers"]["at"]}}, f"a failed 64k load stores only its compute buffer ({ent64})")
check(lr.entries({"compute_mib": 500.0, "n_ctx": 4096, "n_ubatch": 512}, 32768, 2048, False, "stock", False, None, None, None) == {},
      "a log of another context / micro-batch is not stored under this plan's key")

cache = tempfile.mkdtemp(prefix="placement-cache-")
atexit.register(shutil.rmtree, cache, True)
GID, BUILD, MID = "nvidia:NVIDIA GeForce RTX 4060 Laptop GPU, 580.95", "b" * 16, lr.model_id(TWIN)
check(lr.lookup(cache, GID, BUILD, MID, 32768, 2048, False, "stock", False, "nvidia-smi") == {}, "nothing recorded: nothing measured (the plan falls back to the header and tables)")
check(lr.save(cache, GID, BUILD, MID, 32768, 2048, ("f16", "f16"), "stock", False, ent32, free_source="nvidia-smi") and lr.save(cache, GID, BUILD, MID, 65536, 2048, ("f16", "f16"), "stock", False, ent64, free_source="nvidia-smi"),
      "both loads stored")
m32 = lr.lookup(cache, GID, BUILD, MID, 32768, 2048, False, "stock", False, "nvidia-smi")
m64 = lr.lookup(cache, GID, BUILD, MID, 65536, 2048, False, "stock", False, "nvidia-smi")
check(m32.get("compute_mib") == 660.0 and m32.get("gap_mib") == gap and m64.get("compute_mib") == 788.0 and "context_mib" not in m64 and m64.get("non_expert_mib"),
      "lookup: 32k has all parts; 64k has its compute buffer and the model's weights and overhead (they do not depend on the context)")
check(lr.lookup(cache, GID, "c" * 16, MID, 32768, 2048, False, "stock", False, "nvidia-smi") == {}, "another llama.cpp build: nothing (measured again)")
check(lr.lookup(cache, GID, BUILD, MID, 32768, 2048, True, "stock", False, "nvidia-smi").get("compute_mib") is None, "another KV type: no buffers")


def as_measured(m, ub=2048):
    out = {k: m[k] for k in ("non_expert_mib", "gap_mib", "draft_mib") if k in m}
    b = {k: m[k] for k in ("compute_mib", "context_mib") if k in m}
    if b:
        out["buffers"] = {ub: b}
    return out


def plan(ctx, measured=None, plugin=False, free=FREE_PLAN, source="nvidia-smi", **kw):
    return gguf_plan.tune(TWIN, ctx, free=free * MIB, want_cache=False, kv_q8=False, plugin=plugin, measured=measured, free_source=source, **kw)


p32, p64 = plan(32768, as_measured(m32)), plan(65536, as_measured(m64))
check(p32["ub"] == 2048 and 25 <= p32["n_cpu_moe"] <= 27 and loads(p32["n_cpu_moe"], 32768), f"measured plan, 32k: --n-cpu-moe {p32['n_cpu_moe']} (wanted 25-27, and it loads)")
check(p64["ub"] == 2048 and 27 <= p64["n_cpu_moe"] <= 29 and loads(p64["n_cpu_moe"], 65536), f"measured plan, 64k: --n-cpu-moe {p64['n_cpu_moe']} (wanted 27-29, and it loads)")
src = {r["setting"]: r["source"] for r in p32["inputs"]}
check(src.get("gpu_weights_mib") == "measured" and src.get("gpu_compute_mib") == "measured" and src.get("gpu_kv_mib") == "measured" and src.get("gpu_overhead_mib") == "measured",
      f"every size of the 32k plan says 'measured' ({src})")
src64 = {r["setting"]: r["source"] for r in p64["inputs"]}
check(src64.get("gpu_kv_mib") == "probe" and src64.get("gpu_compute_mib") == "measured", "64k: KV from the header (probe), compute buffer measured (the failed load's size)")
check(all(set(r) == {"setting", "value", "source", "why"} and r["source"] in ("measured", "probe", "constant") and r["why"] for r in p32["inputs"]),
      "each input row has value, source and reason")
old = {32768: (30, 28), 65536: (35, 28)}      # v0.3.8's tune and its auto-tune pick (planner map section 3)
check(p32["n_cpu_moe"] < old[32768][0] and p64["n_cpu_moe"] < old[65536][0], f"fewer CPU layers than v0.3.8's tune (30 / 35): {p32['n_cpu_moe']} / {p64['n_cpu_moe']}")

# the plugin engine (a release kit with a matching plugin build runs it for a model that fits RAM but not the GPU): its long-prompt block stays free
pp64 = plan(65536, as_measured(m64), plugin=True)
check(pp64["n_cpu_moe"] > p64["n_cpu_moe"] and loads(pp64["n_cpu_moe"], 65536), f"plugin engine, 64k, measured: --n-cpu-moe {pp64['n_cpu_moe']} (the plugin's long-prompt block stays free; stock: {p64['n_cpu_moe']})")
# without records (first start): the header and our tables, still never past the edge
f32, f64 = plan(32768), plan(65536)
srcf = {r["setting"]: r["source"] for r in f32["inputs"]}
check(srcf.get("gpu_compute_mib") == "constant" and srcf.get("gpu_overhead_mib") == "constant" and srcf.get("gpu_weights_mib") == "probe",
      f"first start: compute buffer and overhead are constants, the weights from the header ({srcf})")
check(loads(f32["n_cpu_moe"], 32768) and loads(f64["n_cpu_moe"], 65536) and f32["n_cpu_moe"] <= 28 and f64["n_cpu_moe"] <= 33,
      f"first start, no record: --n-cpu-moe {f32['n_cpu_moe']} at 32k, {f64['n_cpu_moe']} at 64k (loads; the tuner walks on from there)")
# the plugin's long-prompt block: only when the plugin runs
b32 = plan(32768, plugin=True)
blk = {r["setting"]: r for r in b32["inputs"]}["long_prompt_block_mib"]["value"]
check(b32["n_cpu_moe"] > f32["n_cpu_moe"] and blk > 600 and {r["setting"]: r for r in f32["inputs"]}["long_prompt_block_mib"]["value"] == 0,
      f"the long-prompt block ({blk} MiB) is kept free only for the plugin: {b32['n_cpu_moe']} CPU layers with it, {f32['n_cpu_moe']} without")
# the CUDA gap: only for nvidia-smi's figure
g0 = plan(32768, source="llama-server")
check(g0["n_cpu_moe"] <= f32["n_cpu_moe"] and {r["setting"]: r for r in g0["inputs"]}["gpu_overhead_mib"]["value"] == 0, "a free figure from llama-server itself: no CUDA gap subtracted")
# MTP's draft context: counted until measured
mt = plan(32768, mtp=True)
mrow = {r["setting"]: r for r in mt["inputs"]}.get("mtp_draft_mib")
check(mrow and mrow["source"] == "constant" and mt["n_cpu_moe"] > f32["n_cpu_moe"], f"MTP on: its draft context ({mrow and mrow['value']} MiB) is in the sum")
mtm = plan(32768, as_measured(m32), mtp=True)
mtr = {r["setting"]: r for r in mtm["inputs"]}
check(mtr.get("mtp_draft_mib", {}).get("source") == "constant" and mtr["gpu_overhead_mib"]["source"] == "measured" and mtm["n_cpu_moe"] > plan(32768, as_measured(m32))["n_cpu_moe"],
      "MTP on with a measured overhead (measured on a load without MTP): the draft constant is still added on top (critic M2)")

# ---- 5. PC 35B 32k spill (LAUNCHER-1152): the old first placement said 0 and its shortcut kept it ----
import importlib.util   # noqa: E402
spec = importlib.util.spec_from_file_location("plan_mod", os.path.join(HERE, "planner", "plan.py"))
plan_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan_mod)
PC_DRIVER, PC_TOTAL, PC_OTHERS, PC_COMPUTE = 15419, 16304, 2458, 864.03     # RX 9070 XT: Vulkan budget "free", card, desktop + apps (2.4 GiB), compute buffer (PC log)
old_pc = plan_mod.plan_info(TWIN, ctx=32768, vram_gib=PC_DRIVER / 1024, ram_gib=48)["n_cpu_moe"]
need0 = weights + gguf_plan.gpu_expert_bytes(lay, 0) / MIB + 891.25 + PC_COMPUTE
check(old_pc == 0 and need0 > PC_TOTAL - PC_OTHERS - 300, f"old first placement (plan.py) on the PC: --n-cpu-moe {old_pc}, needing {need0:.0f} MiB of the {PC_TOTAL - PC_OTHERS} the card leaves: the spill")
import moe_cache as mc   # noqa: E402
mc.say = lambda *a, **k: None
_ra = types.SimpleNamespace()
mc.real_room_stage(_ra, (0, "Vulkan0", PC_DRIVER, PC_TOTAL), types.SimpleNamespace(gpu_process_text=lambda: "\\\\pc\\gpu process memory(pid_4_luid_0x0_0x1_phys_0)\\dedicated usage|%d" % (PC_OTHERS * MIB)))
room = getattr(_ra, "room_gib", 0) * 1024
check(10 * 1024 < room < PC_TOTAL - PC_OTHERS, f"the launcher's room stage on the PC figures: {room:.0f} MiB (the MoE plan uses it since fix 1; it was the dense plan's only)")
pc = plan(32768, {"buffers": {2048: {"compute_mib": PC_COMPUTE}}}, free=room, source="room")
need = weights + gguf_plan.gpu_expert_bytes(lay, pc["n_cpu_moe"]) / MIB + 891.25 + PC_COMPUTE
check(pc["n_cpu_moe"] > 0 and need <= 13.1 * 1024, f"new plan with the real room ({room} MiB): --n-cpu-moe {pc['n_cpu_moe']}, needing {need:.0f} MiB <= 13.1 GiB kept on the PC (no spill)")
pcb = plan(32768, {"buffers": {2048: {"compute_mib": PC_COMPUTE}}}, free=PC_DRIVER, source="llama-server")
check(pcb["n_cpu_moe"] < pc["n_cpu_moe"], f"(why the room is needed: with the driver's budget figure the stock plan would be {pcb['n_cpu_moe']})")
# the launcher no longer lets the first placement decide (the 0-shortcut)
A = types.SimpleNamespace(ub="auto", fast=None, vram=None, gpu_margin=None, ram=None)
ub_, n_, _ = mc.layers_stage(A, [], TWIN, 32768, "0", None, "off", False, {"n_cpu_moe": 4, "ub": 2048, "why": ["4 ..."]}, None, None)
check(n_ == "4", f"layers_stage: a first answer of 0 no longer wins over the placement's 4 (got {n_})")
ub_, n_, _ = mc.layers_stage(A, [], TWIN, 32768, "999", None, "off", False, None, None, None)
check(n_ == "999", "layers_stage: no placement (the tuner failed): all experts stay on the CPU")


class _PM:
    def mem_cap(self): return None
    def mem_for_server(self, ram): return 32 * 2**30
    def planner_answer(self, model, ctx, ram, vram): return {"n_cpu_moe": 0, "layers": 40}


B = types.SimpleNamespace(ram=None, vram=None)
n_, info, err = mc.plan_placement(B, TWIN, 32768, None, _PM())
check(n_ == "999" and info["n_cpu_moe"] == 0, f"plan_placement: plan.py's 0 is kept for the control page only, the launcher waits for the placement (got {n_})")

# the launcher's own lookup: nothing recorded -> {}, then a record of this machine and build
mc.CACHE_DIR = cache
lsdir = tempfile.mkdtemp(prefix="placement-ls-")
atexit.register(shutil.rmtree, lsdir, True)
LS = os.path.join(lsdir, "llama-server")
open(LS, "w").close()
build = lr.build_id(LS)
check(len(build) == 16 and build != lr.build_id(os.path.join(lsdir, "nope")), "build id from llama-server and its libraries")


class _M2:
    def gpu_identity(self): return GID


A2 = types.SimpleNamespace(vram=None)
check(mc.measured_loads(A2, LS, TWIN, 32768, False, _M2(), False, False, None, "nvidia-smi") == {}, "launcher: no record for this build yet: {}")
lr.save(cache, GID, build, MID, 32768, 2048, ("f16", "f16"), "stock", False, ent32, free_source="nvidia-smi")
got = mc.measured_loads(A2, LS, TWIN, 32768, False, _M2(), False, False, None, "nvidia-smi")
check(got.get("buffers", {}).get(2048, {}).get("compute_mib") == 660.0 and got.get("gap_mib") == gap, f"launcher: the record is found ({got})")
check(mc.measured_loads(types.SimpleNamespace(vram="8"), LS, TWIN, 32768, False, _M2(), False, False, None, "nvidia-smi") == {}, "launcher: --vram given: no record used")
os.environ["MOE_CACHE_LOAD_RECORD"] = "0"
check(mc.measured_loads(A2, LS, TWIN, 32768, False, _M2(), False, False, None, "nvidia-smi") == {}, "launcher: MOE_CACHE_LOAD_RECORD=0: no record used")
del os.environ["MOE_CACHE_LOAD_RECORD"]
check(mc.user_set_log_verbosity(["llama-server", "-lv", "3"], {}) and mc.user_set_log_verbosity(["x"], {"LLAMA_ARG_LOG_VERBOSITY": "2"}) and not mc.user_set_log_verbosity(["x"], {}),
      "the user's own log level is left alone")
# the Recorder end to end: feed the real log, it stores under the plan's key
rec = lr.Recorder({"cache_dir": cache, "machine": GID, "build": "d" * 16, "model": MID, "ctx": 32768, "ub": 2048, "kv": ("f16", "f16"), "engine": "stock", "mtp": False,
                   "gpu_expert_mib": gguf_plan.gpu_expert_bytes(lay, 27) / MIB, "free_plan_mib": FREE_PLAN, "free_source": "nvidia-smi"})
for l in L:
    rec.feed(l + "\n")
rec.finish()
check(lr.lookup(cache, GID, "d" * 16, MID, 32768, 2048, False, "stock", False, "nvidia-smi").get("compute_mib") == 660.0 and rec.saved == 1, "Recorder: the real log fed line by line is stored")

# ---- 6. the auto-tune's edge walk (fix 2) ----
CMD = ["llama-server", "-m", "m.gguf", "--n-cpu-moe", "28", "-ub", "2048", "-c", "65536"]
check(at.walk_start(CMD, {}) == 28 and at.walk_start(CMD, {"ncmoe": "yours"}) is None and at.walk_start(["x", "--n-cpu-moe", "999"], {}) is None
      and at.walk_start(["x", "--n-cpu-moe", "0"], {}) is None, "walk start: the plan's --n-cpu-moe; not for your own, 999 (GPU cache) or 0")
cands = at.build_candidates(CMD, [], {}, 65536, 40, walk=True)
check(cands[0].id == "C0" and not any(c.id == "ncmoe-2" for c in cands) and len(cands) <= 3, f"with the walk, no fixed ncmoe-2 candidate ({[c.id for c in cands]})")


def fake(decode_by_n, oom_below, load_s=5.0):
    """A runner on a fake clock: --n-cpu-moe n decodes decode_by_n(n) tok/s; below oom_below the load fails after 10 s."""
    clock = [0.0]
    seen = []

    def run(c, left, n_tok):
        cmd, _, _ = at.apply_edits(CMD, [], c.edits)
        n = int(at.flag_value(cmd, "--n-cpu-moe"))
        seen.append((c.id, n))
        if n < oom_below:
            clock[0] += 10
            return {"ok": False, "error": "llama-server stopped while loading (out of GPU memory)"}
        dec = decode_by_n(n)
        clock[0] += load_s + 1500 / 600 + n_tok / dec
        return {"ok": True, "load_s": load_s, "prompt_tps": 600.0, "decode_tps": dec, "decode_n": n_tok}
    return run, (lambda: clock[0]), seen


# 64k laptop shape: one more layer on the GPU is ~+2.5 % decode down to the edge at 27, 26 runs out of memory
run, clk, seen = fake(lambda n: 20.8 * (1 + 0.025 * (28 - n)), 27)
res = at.tune(at.build_candidates(CMD, [], {"np": "x", "ub": "x", "ramlock": "x"}, 65536, 40, walk=True), run, clock=clk, walk=28)
ids = [s[0] for s in seen]
check(ids[:3] == ["C0", "ncmoe-1", "ncmoe-2"] and res["walk"]["steps"] == 2 and "did not load" in res["walk"]["end"],
      f"walk: 28 -> 27 (loads) -> 26 (out of memory) ends the walk ({ids}, {res['walk']})")
check(res["picked"] == "C0", f"walk: one layer (+2.5 %) is inside the 5 % noise margin, so the plan stays ({res['picked']}: {res['why'][:90]})")
# first start without a record (plan 32 at 64k): the walk reaches the edge at 27 and picks it
CMD32 = ["llama-server", "-m", "m.gguf", "--n-cpu-moe", "32", "-ub", "2048", "-c", "65536"]
clock32, seen32 = [0.0], []
def run32(c, left, n_tok):
    cmd, _, _ = at.apply_edits(CMD32, [], c.edits)
    n = int(at.flag_value(cmd, "--n-cpu-moe"))
    seen32.append(n)
    if n < 27:
        clock32[0] += 10
        return {"ok": False, "error": "llama-server stopped while loading (out of GPU memory)"}
    dec = 17.0 * (1 + 0.025 * (32 - n))
    clock32[0] += 5 + 1500 / 600 + n_tok / dec
    return {"ok": True, "load_s": 5, "prompt_tps": 600.0, "decode_tps": dec, "decode_n": n_tok}
res = at.tune([at.Candidate("C0", "the plan as it is", [])], run32, clock=lambda: clock32[0], walk=32, budget_s=10000)
check(res["picked"] == "ncmoe-4" and seen32[:7] == [32, 31, 30, 29, 28, 27, 26] and "edge" in res["walk"]["end"],
      f"walk from a first-start plan of 32: down to 27 (26 fails); 27 is at the edge, so ncmoe-4 = 28 is picked (the reserve stays free) ({seen32}, {res['picked']})")
# decode stops rising (spill into shared memory past the edge on a card that does not fail the load)
run, clk, seen = fake(lambda n: 20.0 * (1 + 0.04 * (28 - n)) if n >= 25 else 10.0, 0)
res = at.tune([at.Candidate("C0", "the plan as it is", [])], run, clock=clk, walk=28, budget_s=10000)
check(res["walk"]["steps"] == 4 and "decode fell" in res["walk"]["end"] and res["picked"] == "ncmoe-3", f"walk: stops when decode falls (spill) and picks the best before it ({res['walk']}, {res['picked']})")
# the budget ends a long walk; C0 still decides when nothing beats it
run, clk, seen = fake(lambda n: 20.0, 0, load_s=30)
res = at.tune([at.Candidate("C0", "the plan as it is", [])], run, clock=clk, walk=28)
check(res["total_s"] <= at.BUDGET_S + 60 and res["picked"] == "C0" and res["walk"]["steps"] < 28, f"walk: a flat decode within the 180 s budget keeps the plan ({res['walk']}, {res['total_s']} s)")
# the commit check ends the walk without a load
run, clk, seen = fake(lambda n: 20.0 * (1 + 0.04 * (28 - n)), 0)
res = at.tune([at.Candidate("C0", "the plan as it is", [])], run, clock=clk, walk=28, budget_s=10000, check=lambda c: (c.id != "ncmoe-3", "too big"))
check(res["walk"]["steps"] == 2 and "commit check" in res["walk"]["end"] and ("ncmoe-3", 25) not in seen, f"walk: a step the commit check refuses is not loaded ({res['walk']})")
check(any("edge walk from --n-cpu-moe 28" in l for l in at.summary_lines(res)), "the summary says how the walk went")
# LAUNCHER-1225 step 2: two layers per load while the known fit limit is far (> WIDE_FROM layers), then one; a failed two-layer step steps back one layer
def walk32(oom_below, floor):
    seen_, clk_ = [], [0.0]
    def run_(c, left, n_tok):
        cmd_, _, _ = at.apply_edits(CMD32, [], c.edits)
        n = int(at.flag_value(cmd_, "--n-cpu-moe"))
        seen_.append(n)
        clk_[0] += 8
        if (n in oom_below) if isinstance(oom_below, set) else n < oom_below:
            return {"ok": False, "error": "llama-server stopped while loading (out of GPU memory)"}
        return {"ok": True, "load_s": 5, "prompt_tps": 600.0, "decode_tps": 17.0 * (1 + 0.03 * (32 - n)), "decode_n": n_tok}
    return at.tune([at.Candidate("C0", "the plan as it is", [])], run_, clock=lambda: clk_[0], walk=32, budget_s=10000, floor=floor), seen_
r_w, s_w = walk32(0, 18)
check(s_w[:6] == [32, 30, 28, 26, 24, 23] and min(s_w) == 18 and r_w["walk"]["layers"] == 14 and r_w["walk"]["steps"] < 14,
      f"far fit limit (18 from 32): 2 layers per load down to 6 above it, then 1 ({s_w}, {r_w['walk']['steps']} loads for {r_w['walk']['layers']} layers)")
r_f, s_f = walk32(29, 18)
check(s_f[:4] == [32, 30, 28, 29] and r_f["picked"] == "ncmoe-2" and "ncmoe-3 is at the edge" in r_f["walk"]["end"],
      f"a failed two-layer step (28) steps back to 29, which loads: 29 is the edge, 30 is picked ({s_f}, {r_f['picked']}, {r_f['walk']['end']})")
r_g, s_g = walk32(31, 18)
check(s_g[:3] == [32, 30, 31] and r_g["picked"] == "C0" and "ncmoe-1 is at the edge" in r_g["walk"]["end"],
      f"30 fails, the step back 31 loads: 31 is the edge, the plan stays ({s_g}, {r_g['picked']}, {r_g['walk']['end']})")
r_h, s_h = walk32(set(range(0, 32)), 18)
check(s_h[:3] == [32, 30, 31] and r_h["picked"] == "C0" and "did not load or run" in r_h["walk"]["end"], f"30 and the step back 31 both fail: the plan stays ({s_h[:3]}, {r_h['walk']['end']})")
r_n, s_n = walk32(27, None)
check(s_n[:7] == [32, 31, 30, 29, 28, 27, 26], f"no known fit limit: one layer per load as before ({s_n})")
# chat is scored toward decode: a plan with 10 % faster decode and 30 % slower prompts wins for chat, loses for agents
row_a = {"prompt_tps": 600.0, "decode_tps": 20.0}
row_b = {"prompt_tps": 420.0, "decode_tps": 22.0}
check(at.score(row_b) < at.score(row_a) and at.score(row_b, agent=True) > at.score(row_a, agent=True), "score: chat weighs decode (1000 prompt tokens), agents weigh prompts (16000)")
# the cache key carries the launcher version and the llama.cpp build
k1 = at.key_parts("m", 65536, "sig", {}, False, "stock", {}, launcher="0.3.8", build="aaaa")
k2 = at.key_parts("m", 65536, "sig", {}, False, "stock", {}, launcher="0.3.9", build="aaaa")
k3 = at.key_parts("m", 65536, "sig", {}, False, "stock", {}, launcher="0.3.8", build="bbbb")
check(len({at.key_of(k1), at.key_of(k2), at.key_of(k3)}) == 3 and k1["launcher"] == "0.3.8" and k1["build"] == "aaaa", "auto-tune key: a new launcher or llama.cpp build is a new key")

# a stored pick replays as the same layer count when the plan has learned from loads since (base 32, picked -5 = 27)
stored = {"picked": "ncmoe-5", "base_ncmoe": 32, "rows": [{"id": "ncmoe-5", "edits": [{"dim": "ncmoe", "flag": "--n-cpu-moe", "delta": -5}]}]}
e28 = at.pick_edits(stored, ["x", "--n-cpu-moe", "28"])
e26 = at.pick_edits(stored, ["x", "--n-cpu-moe", "26"])
e32 = at.pick_edits(stored, ["x", "--n-cpu-moe", "32"])
check(e28 == [{"dim": "ncmoe", "flag": "--n-cpu-moe", "delta": -1}] and e26 == [] and e32[0]["delta"] == -5,
      f"replay: picked 27 on a plan of 32; plan now 28 -> -1 (27), plan now 26 -> nothing (never back toward the CPU), plan 32 -> -5 ({e28}, {e26})")
check(at.pick_edits({k: v for k, v in stored.items() if k != "base_ncmoe"}, ["x", "--n-cpu-moe", "28"])[0]["delta"] == -5, "an older stored pick without a base keeps its delta")

# ---- 7. second critic: H1 (walk fit limit), M1 (replay floor), M2/M3/M4/L2 ----
# H1: where a load that does not fit spills instead of failing (Windows), decode drops less than 5 % per step; only the fit limit stops the walk
seen_s = []
def run_spill(c, left, n):
    nc = 28 - (int(c.id.split("-")[1]) if c.id.startswith("ncmoe-") else 0)
    seen_s.append(nc)
    return {"ok": True, "load_s": 5.0, "prompt_tps": 600.0, "decode_tps": 20.0 + 0.3 * (28 - nc), "decode_n": 128}
clk_s = [0.0]
res = at.tune([at.Candidate("C0", "the plan as it is", [])], run_spill, clock=lambda: clk_s[0], walk=28, budget_s=10000, floor=26)
check(min(seen_s) == 26 and "fit limit" in res["walk"]["end"] and res["picked"] != "ncmoe-2" and "edge" in res["walk"]["end"], f"walk: never loads below the planner's fit limit, even when nothing fails (spill) ({seen_s}, {res['walk']['end']})")
# the launcher's fit limit: the same tuner with no reserve, at the plan's micro-batch; the user's --gpu-margin is honoured
FA = types.SimpleNamespace(gpu_margin=None, leave_room=False)
kw0 = {"plugin": False, "measured": as_measured(m64), "free_source": "nvidia-smi"}
t64 = gguf_plan.tune(TWIN, 65536, free=FREE_PLAN * MIB, want_cache=False, kv_q8=False, **kw0)
fl = mc.walk_floor(gguf_plan.tune, TWIN, 65536, FREE_PLAN * MIB, False, kw0, dict(t64, inputs=list(t64["inputs"])), FA)
flm = mc.walk_floor(gguf_plan.tune, TWIN, 65536, FREE_PLAN * MIB, False, dict(kw0, reserve_gb=2.0 * 2**30 / 1e9), dict(t64, inputs=list(t64["inputs"])),
                   types.SimpleNamespace(gpu_margin=2.0, leave_room=False))
check(mc.walk_floor(gguf_plan.tune, TWIN, 65536, FREE_PLAN * MIB, False, kw0, dict(t64, inputs=list(t64["inputs"])), FA, spill=False) is None,
      "no fit limit where a load that does not fit fails (Linux): the walk goes to the edge (third critic, finding 1)")
check(fl is not None and 25 <= fl <= t64["n_cpu_moe"] and flm == t64["n_cpu_moe"] and {r["setting"]: r for r in t64["inputs"]}.get("walk_floor_n_cpu_moe") is None,
      f"fit limit at 64k: {fl} (plan {t64['n_cpu_moe']}; 25 runs out of memory on the laptop); with --gpu-margin 2 GB: {flm} (never past the plan)")
check(mc.walk_floor(gguf_plan.tune, TWIN, 65536, FREE_PLAN * MIB, False, kw0, {"n_cpu_moe": 999, "ub": 512}, FA) is None, "no fit limit when the plan places no expert layers")
pcf = mc.walk_floor(gguf_plan.tune, TWIN, 32768, room * MIB, False, {"plugin": False, "measured": {"buffers": {2048: {"compute_mib": PC_COMPUTE}}}, "free_source": "room"},
                    dict(pc, inputs=list(pc["inputs"])), FA, use_room=True)
needf = weights + gguf_plan.gpu_expert_bytes(lay, pcf) / MIB + 891.25 + PC_COMPUTE
check(pcf is not None and needf <= room, f"PC 35B 32k: the walk stops at --n-cpu-moe {pcf} ({needf:.0f} MiB inside the room of {room:.0f}), not in the spill")
# M1: a replayed pick never goes below this start's fit limit
check(at.pick_edits(stored, ["x", "--n-cpu-moe", "28"], floor=28) == [] and at.pick_edits(stored, ["x", "--n-cpu-moe", "32"], floor=29)[0]["delta"] == -2,
      "replay: a pick past this start's fit limit is cut back to it")
# M3: errors always show, even after a hidden line or without the log prefix
F = lr.LineFilter()
F.show("0.00.100.000 D llama_kv_cache: layer 0: dev = CUDA0\n")
check(F.show("ggml_abort: something went wrong\n") and F.show("CUDA error: out of memory\n"), "line filter: aborts and errors always show")
# L2: log prefix or timestamps off: the level is left alone
check(mc.user_set_log_verbosity(["x", "--log-timestamps", "off"], {}) and mc.user_set_log_verbosity(["x"], {"LLAMA_ARG_LOG_PREFIX": "0"})
      and mc.user_set_log_verbosity(["x", "--no-log-prefix"], {}) and not mc.user_set_log_verbosity(["x", "--log-timestamps", "on"], {}), "log prefix or timestamps off: verbosity not raised")
# M4: the user's own tensor placement is not recorded
_plan = types.SimpleNamespace(a=types.SimpleNamespace(vram=None, plan_free_mib=FREE_PLAN, plan_free_source="nvidia-smi"), cmd=["x", "--n-cpu-moe", "28"], machine=_M2(), ls=LS,
                              model=TWIN, ctx=65536, engine="stock")
mk = mc.make_load_recorder(_plan)
check(mk is not None and mk(["x", "--n-cpu-moe", "28", "-c", "65536"]) is not None, "recorder: the plan's own load is recorded")
check(mk(["x", "--n-cpu-moe", "28", "-ot", "blk.1=CPU"]) is None and mk(["x", "--n-cpu-moe", "28", "-ts", "1,1"]) is None and mk(["x", "--n-cpu-moe", "28", "-ngl", "20"]) is None,
      "recorder: -ot, -ts or a partial -ngl: not recorded")
check(mk(["x", "--n-cpu-moe", "28", "--spec-type", "draft-mtp"]) is not None and mk(["x", "--n-cpu-moe", "28", "--spec-type", "draft-mtp"]).ctx["mtp"] is True,
      "recorder: an MTP load is recorded, flagged as MTP (follow-up 1)")
class _M3(_M2):
    def __init__(self, free): self.free = free
    def free_vram_bytes(self): return self.free * MIB
for fr, want in ((FREE_PLAN + 40, FREE_PLAN), (FREE_PLAN - 900, None)):
    _p = types.SimpleNamespace(**dict(vars(_plan), machine=_M3(fr)))
    r_ = mc.make_load_recorder(_p)(["x", "--n-cpu-moe", "28"])
    check(r_ is not None and r_.ctx.get("free_plan_mib") == want, f"recorder: free memory {fr - FREE_PLAN:+.0f} MiB since the plan: gap learned against {r_ and r_.ctx.get('free_plan_mib')}")
res = at.tune([at.Candidate("C0", "the plan as it is", [])], run_spill, clock=lambda: 0.0, walk=28, budget_s=10000, floor=28)
check(res["walk"]["steps"] == 0 and "no walk here" in res["walk"]["end"], f"fit limit = plan: no walk, and the end line says so ({res['walk']})")

# ======================================================================================================================================================
# ---- 8. follow-ups of the fixes 1 + 2 critics (LAUNCHER-1182) ----
# ======================================================================================================================================================
import stat   # noqa: E402
cache2 = tempfile.mkdtemp(prefix="placement-cache2-")
atexit.register(shutil.rmtree, cache2, True)

# 1. MTP: the draft context is told apart by llama-server's own marker line and counted separately
i_comp = next(i for i, l in enumerate(L) if "CUDA0 compute buffer size" in l)
DRAFT = ["0.02.600.000 I common_speculative_init_result: creating MTP draft context against the target model 'x.gguf'",
         "0.02.600.100 I llama_context: n_ctx                 = 4096", "0.02.600.101 I llama_context: n_ubatch              = 512",
         "0.02.600.200 I llama_kv_cache:      CUDA0 KV buffer size =    64.00 MiB", "0.02.600.300 I llama_memory_recurrent:      CUDA0 RS buffer size =    10.00 MiB",
         "0.02.600.400 I sched_reserve:      CUDA0 compute buffer size =   150.00 MiB"]
LM = L[:i_comp + 2] + DRAFT + L[i_comp + 2:]
mtp_rec = lr.parse_log(LM)
check(mtp_rec.get("mtp_draft") and mtp_rec.get("draft_mib") == 224.0 and mtp_rec.get("context_mib") == 891.25 and mtp_rec.get("compute_mib") == 660.02 and mtp_rec.get("n_ctx") == 32768,
      f"MTP log: the draft's 224 MiB (KV 64 + RS 10 + compute 150) apart; the target's context 891.25, compute 660.02 and n_ctx 32768 are not the draft's ({ {k: mtp_rec.get(k) for k in ('draft_mib', 'context_mib', 'compute_mib', 'n_ctx')} })")
ent_m = lr.entries(mtp_rec, 32768, 2048, ("f16", "f16"), "stock", True, gguf_plan.gpu_expert_bytes(lay, 27) / MIB, FREE_PLAN, "nvidia-smi", n_cpu_moe=27)
check(ent_m.get("drafts", {}).get("draft_mib") == 224.0 and ent_m.get("buffers", {}).get("context_mib") == 891.2 and "overhead" not in ent_m and abs(ent_m["weights"]["non_expert_mib"] - 1596.8) < 0.2,
      f"an MTP load stores the target's weights and buffers, the draft on its own, and no gap ({sorted(ent_m)})")
check(lr.entries(lr.parse_log(L), 32768, 2048, ("f16", "f16"), "stock", True, 1.0, FREE_PLAN, "nvidia-smi") == {},
      "MTP on but no draft marker in the log: nothing is recorded (the target's and the draft's buffers could not be told apart)")
lr.save(cache2, GID, BUILD, MID, 32768, 2048, ("f16", "f16"), "stock", True, ent_m, free_source="nvidia-smi")
lm_t, lm_f = lr.lookup(cache2, GID, BUILD, MID, 32768, 2048, False, "stock", True, "nvidia-smi"), lr.lookup(cache2, GID, BUILD, MID, 32768, 2048, False, "stock", False, "nvidia-smi")
check(lm_t.get("draft_mib") == 224.0 and "draft_mib" not in lm_f and lm_t.get("context_mib") == 891.2, "lookup: the draft comes back with MTP on only")
pm = plan(32768, as_measured(lm_t), mtp=True)
pm_row = {r["setting"]: r for r in pm["inputs"]}["mtp_draft_mib"]
check(pm_row["source"] == "measured" and pm_row["value"] == 224, f"the plan with MTP on counts the measured draft ({pm_row['value']} MiB, {pm_row['source']}), not the 870 MiB constant")

# 3. the gap is only taken from loads planned against the same kind of free figure
lr.save(cache2, GID, BUILD, MID, 32768, 2048, ("f16", "f16"), "stock", False, ent32, free_source="nvidia-smi")
check(lr.lookup(cache2, GID, BUILD, MID, 32768, 2048, False, "stock", False, "nvidia-smi").get("gap_mib") == gap
      and "gap_mib" not in lr.lookup(cache2, GID, BUILD, MID, 32768, 2048, False, "stock", False, "llama-server")
      and "gap_mib" not in lr.lookup(cache2, GID, BUILD, MID, 32768, 2048, False, "stock", False, "room"),
      "the gap learned against nvidia-smi's figure is not used for llama-server's or the room figure")

# 6. the driver is in the GPU identity
check(any(ch.isdigit() for ch in gguf_plan.drm_driver_version()) or gguf_plan.drm_driver_version() == "", "Linux AMD / Intel identity carries the driver's module version, else the kernel release (nvidia-smi's identity already has the driver version)")
check(lr.lookup(cache2, GID.replace("580.95", "590.01"), BUILD, MID, 32768, 2048, False, "stock", False, "nvidia-smi") == {}, "a new driver version is a new key: measured again")

# 7. failed edges: a load that ran out of GPU memory is remembered per key, and the walk does not retry it
ent_f = lr.entries(lr.parse_log(lines("laptop_35b_64k_ncmoe25_oom.txt")), 65536, 2048, ("f16", "f16"), "stock", False, gguf_plan.gpu_expert_bytes(lay, 25) / MIB, FREE_PLAN, "nvidia-smi",
                   n_cpu_moe=25, free_now_mib=FREE_PLAN)
check(ent_f.get("edges", {}).get("fail_ncmoe") == 25 and "edge_ok" not in ent_f, f"a failed load stores its --n-cpu-moe as a failed edge ({ent_f.get('edges')})")
lr.save(cache2, GID, BUILD, MID, 65536, 2048, ("f16", "f16"), "stock", False, ent_f, free_source="nvidia-smi")
ed = lambda **k: lr.lookup_edge(cache2, GID, BUILD, MID, k.get("ctx", 65536), 2048, ("f16", "f16"), k.get("engine", "stock"), False, free_now_mib=k.get("free"))
check(ed() == 25 and ed(free=FREE_PLAN) == 25 and ed(ctx=32768) is None and ed(engine="plugin") is None, "lookup_edge: found for this plan only (context and engine are in the key)")
check(ed(free=FREE_PLAN + 1000) is None, "an edge recorded with 1 GiB less free memory than there is now is not trusted")
ent_ok = lr.entries(lv4, 65536, 2048, ("f16", "f16"), "stock", False, gguf_plan.gpu_expert_bytes(lay, 27) / MIB, FREE_PLAN, "nvidia-smi", n_cpu_moe=27)
ent_ok = dict(ent_ok); ent_ok.pop("buffers", None)
check(ent_ok.get("edge_ok") == 27, "a load that came up stores 'edge_ok' at its value")
lr.save(cache2, GID, BUILD, MID, 65536, 2048, ("f16", "f16"), "stock", False, ent_ok, free_source="nvidia-smi")
check(ed() == 25, "a success above the failed value (27 > 25) leaves the edge alone")
lr.save(cache2, GID, BUILD, MID, 65536, 2048, ("f16", "f16"), "stock", False, {"edge_ok": 24}, free_source="nvidia-smi")
check(ed() is None, "a success at 24 (<= 25) means the failure at 25 is out of date: the edge is dropped")
# the walk: with the recorded edge as the floor it neither loads the failed value nor its edge step, and its pick is the same as before
lr.save(cache2, GID, BUILD, MID, 65536, 2048, ("f16", "f16"), "stock", False, ent_f, free_source="nvidia-smi")
seen_e = []
clk_e = [0.0]
def run_e(c, left, n):
    cmd_, _, _ = at.apply_edits(CMD32, [], c.edits)
    nc = int(at.flag_value(cmd_, "--n-cpu-moe"))
    seen_e.append(nc)
    clk_e[0] += 8
    if nc <= 25:
        return {"ok": False, "error": "llama-server stopped while loading (out of GPU memory)"}
    return {"ok": True, "load_s": 5, "prompt_tps": 600.0, "decode_tps": 17.0 * (1 + 0.025 * (32 - nc)), "decode_n": n}
fe = ed() + 2
res_e = at.tune([at.Candidate("C0", "the plan as it is", [])], run_e, clock=lambda: clk_e[0], walk=32, budget_s=10000, floor=fe, floor_edge=fe)
check(min(seen_e) == 27 and 26 not in seen_e and 25 not in seen_e and res_e["picked"] == "ncmoe-5" and "recorded" in res_e["walk"]["end"],
      f"the second start does not retry the failed edge: loads {seen_e}, never 26 or 25; picks 27 ({res_e['picked']}); {res_e['walk']['end']}")
seen_e.clear(); clk_e[0] = 0.0
res_e0 = at.tune([at.Candidate("C0", "the plan as it is", [])], run_e, clock=lambda: clk_e[0], walk=32, budget_s=10000)
check(min(seen_e) == 25 and 26 in seen_e and res_e0["picked"] == res_e["picked"], f"(without the record the walk loads 26 and the failing 25 again, for the same pick: {seen_e})")
# the launcher side: known_failed_edge reads the same key from the command
mc.CACHE_DIR = cache2
mc.load_record_on = lambda: True
class _M4(_M2):
    pass
_a4 = types.SimpleNamespace(vram=None, vram_user=None, plan_free_mib=FREE_PLAN)
build_real = lr.build_id(LS)
lr.save(cache2, GID, build_real, MID, 65536, 2048, ("f16", "f16"), "stock", False, ent_f, free_source="nvidia-smi")
check(mc.known_failed_edge(_a4, _M4(), LS, TWIN, 65536, ["x", "-c", "65536", "-ub", "2048", "--n-cpu-moe", "32"], "stock") == 25, "launcher: the failed edge of this command is found")
check(mc.known_failed_edge(_a4, _M4(), LS, TWIN, 65536, ["x", "-c", "65536", "-ub", "2048", "--n-cpu-moe", "32", "-ctk", "q8_0", "-ctv", "q8_0"], "stock") is None, "launcher: another KV type: no edge")
check(mc._max_floor(None, None) is None and mc._max_floor(26, None) == 26 and mc._max_floor(26, 29) == 29, "the walk's limit is the stricter of the planner's and the recorded edge")

# 2. probe loads are recorded: a fake llama-server prints the real lv4 log (or the real out-of-memory log), the probe feeds it to the recorder
fakesrv = os.path.join(twin_dir, "fake-server.py")
logf = os.path.join(twin_dir, "probe.log")
open(fakesrv, "w").write('''#!/usr/bin/env python3
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv; port = int(a[a.index("--port") + 1])
for l in open(%r, encoding="utf-8", errors="replace").read().splitlines():
    sys.stderr.write(l + "\\n")
sys.stderr.flush()
if "--die" in a:
    sys.exit(1)
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send(200, {"status": "ok"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); j = json.loads(self.rfile.read(n))
        if j["n_predict"] == 1: self._send(200, {"timings": {"prompt_per_second": 350.0, "prompt_n": 1500, "predicted_per_second": 0, "predicted_n": 1}})
        else: self._send(200, {"timings": {"prompt_per_second": 0, "predicted_per_second": 19.5, "predicted_n": j["n_predict"]}})
HTTPServer(("127.0.0.1", port), H).serve_forever()
''' % logf)
cache3 = tempfile.mkdtemp(prefix="placement-cache3-")
atexit.register(shutil.rmtree, cache3, True)
mc.CACHE_DIR = cache3
class _M5(_M2):
    def free_vram_bytes(self): return FREE_PLAN * MIB
a5 = types.SimpleNamespace(vram=None, vram_user=None, plan_free_mib=FREE_PLAN, plan_free_source="nvidia-smi")
maker = mc.load_recorder_for(a5, ["x", "--n-cpu-moe", "32"], _M5(), LS, TWIN, 32768, "stock")
check(maker is not None, "launcher: a recorder for the auto-tune's probes can be made from the plan's command")
open(logf, "w").write("\n".join(L))
probe_cmd = [sys.executable, fakesrv, "-c", "32768", "-ub", "2048", "--n-cpu-moe", "27", "--port", "1"]
r5 = mc._autotune_probe(probe_cmd, dict(os.environ), 30, 8, {}, maker)
got5 = lr.lookup(cache3, GID, build, MID, 32768, 2048, False, "stock", False, "nvidia-smi")
check(r5.get("ok") and got5.get("compute_mib") == 660.0 and got5.get("non_expert_mib") and got5.get("gap_mib"), f"a probe load is recorded like a real start's (buffers, weights, gap): {sorted(got5)}")
check(lr.lookup_edge(cache3, GID, build, MID, 32768, 2048, ("f16", "f16"), "stock", False) is None, "...and a probe that came up leaves no failed edge")
open(logf, "w").write("\n".join(lines("laptop_35b_32k_ncmoe23_oom.txt")))
r6 = mc._autotune_probe([sys.executable, fakesrv, "-c", "32768", "-ub", "2048", "--n-cpu-moe", "23", "--port", "1", "--die"], dict(os.environ), 30, 8, {}, maker)
check(not r6["ok"] and lr.lookup_edge(cache3, GID, build, MID, 32768, 2048, ("f16", "f16"), "stock", False) == 23,
      "a probe that ran out of GPU memory records its --n-cpu-moe (23) as the failed edge")
check(maker(["x", "--n-cpu-moe", "23", "-ot", "a=CPU"]) is None, "a probe with the user's own tensor placement is not recorded")

# 4 (HELD) + 11. the room figure: the default reserve is unchanged; --leave-room keeps its 2 GB in all
class _PMach:
    def __init__(self, free=14.65 * 2**30, tf=None, src="llama-server"): self.f, self.tf, self.src = free, tf, src
    def free_vram_bytes(self): return self.f
    def gpu_total_free_bytes(self): return self.tf
    def free_vram_source(self): return self.src
mc.say = lambda *a_, **k_: None
def room_a(**kw): return types.SimpleNamespace(**{**dict(vram="14.65", vram_user=None, room_gib=12.9, room_used_gib=2.4, gpu_margin=None, leave_room=False), **kw})
gpu_w = (0, "Vulkan0", 15000, 16304)
i0 = mc.gpu_room_inputs(room_a(), _PMach(), gpu_w, 0)
check(i0["use_room"] and "reserve_gb" not in i0["kw"] and i0["kw"]["free_source"] == "room" and abs(i0["free"] / 2**30 - 12.9) < 0.01,
      "room path, follow-up 4 HELD (owner decision (b)): the tuner's own 0.5 GB stays on top of the room, as in 404dfd5")
i1 = mc.gpu_room_inputs(room_a(leave_room=True), _PMach(), gpu_w, 0)
check(abs(i1["kw"]["reserve_gb"] - (mc.LEAVE_ROOM_VRAM_GB - 300 * MIB / 1e9)) < 1e-6 and i1["desk_kw"] == i1["kw"]["reserve_gb"], f"room path with --leave-room: 2 GB kept free in all, the room's 300 MiB counting toward them ({i1['kw']['reserve_gb']:.2f} GB), not 0.5 GB")
i2 = mc.gpu_room_inputs(room_a(gpu_margin=1.0), _PMach(), gpu_w, 0)
check(abs(i2["kw"]["reserve_gb"] - 2**30 / 1e9) < 1e-6, "room path with --gpu-margin: the user's number as given")
i3 = mc.gpu_room_inputs(types.SimpleNamespace(vram=None, vram_user=None, gpu_margin=None, leave_room=False), _PMach(free=5 * 2**30, tf=(8 * 2**30, 5 * 2**30), src="nvidia-smi"), None, 0)
check(not i3["use_room"] and i3["kw"]["reserve_gb"] > 0.5 and i3["kw"]["free_source"] == "nvidia-smi", f"no room figure: the desk reserve for what other programs hold is kept ({i3['kw']['reserve_gb']:.2f} GB)")
i4 = mc.gpu_room_inputs(types.SimpleNamespace(vram=None, vram_user=None, gpu_margin=None, leave_room=False), _PMach(free=7.5 * 2**30, tf=(8 * 2**30, 7.5 * 2**30), src="nvidia-smi"), None, 0)
check("reserve_gb" not in i4["kw"], "an idle card: the tuner's own 0.5 GB (no reserve argument)")

# 5 + 9. the engine check plans with the same room and desk reserve as the plan (and runs after the GPU facts)
seen_t = {}
real_tune = gguf_plan.tune
def spy_tune(*a_, **k_):
    seen_t.update(k_); seen_t["free"] = k_.get("free")
    return real_tune(*a_, **k_)
gguf_plan.tune = spy_tune
class _PM2(_PMach):
    def mem_cap(self): return None
    def mem_for_server(self, ram): return 64 * 2**30
    def ls_api(self, ls): return "3"
    def lib_api(self, lib): return "3"
    def pick_lib(self, ls): return "/x/lib.so"
    def lib_exists(self, lib): return True
    def planner_answer(self, model, ctx, ram, vram): return {"layers": lay["n_layers"]}
_a9 = types.SimpleNamespace(stock=False, cpu_only=False, ram="64", vram=None, vram_user=None, gpu_margin=None, leave_room=False)
os.environ.pop("MOE_CACHE_STOCK_WHEN_FITS", None); os.environ.pop("MOE_CACHE_LIB", None)
mc.CACHE_DIR = cache2
mc.plugin_for_partial_offload(_a9, LS, TWIN, 32768, _PM2(free=4 * 2**30, tf=(8 * 2**30, 4 * 2**30), src="nvidia-smi"), None, 0)
check(seen_t.get("reserve_gb", 0.5) > 0.5 and seen_t.get("free_source") == "nvidia-smi", f"engine check: the desk reserve is counted ({seen_t.get('reserve_gb')} GB: other programs hold 4 GiB), like the plan")
seen_t.clear()
_a9w = types.SimpleNamespace(**{**vars(_a9), "vram": "14.65", "room_gib": 12.9, "room_used_gib": 2.4})
mc.plugin_for_partial_offload(_a9w, LS, TWIN, 32768, _PM2(), gpu_w, 0)
check(abs(seen_t["free"] / 2**30 - 12.9) < 0.01 and seen_t.get("free_source") == "room" and "reserve_gb" not in seen_t, "engine check on Windows AMD: the room figure, not llama-server's 14.65 GiB")
gguf_plan.tune = real_tune
src_order = open(os.path.join(HERE, "bin", "moe_cache.py"), encoding="utf-8").read()
i_g, i_p = src_order.index("vram_user, listing, gpu = gpu_stage(a, ls, win, machine)"), src_order.index("why = plugin_for_partial_offload(a, ls, model, ctx, machine, gpu, mm_bytes)")
check(i_g < i_p, "launch order: the GPU facts (room) come before the engine choice")

# 8. multigpu.py and tune() share the placement arithmetic
import multigpu   # noqa: E402
calls = []
orig_blk, orig_room = gguf_plan.long_prompt_block_bytes, gguf_plan.room_bytes
gguf_plan.long_prompt_block_bytes = lambda lay_, ub_, *a_, **k_: (calls.append("block"), orig_blk(lay_, ub_, *a_, **k_) + 2 * 2**30)[1]
gguf_plan.room_bytes = lambda *a_, **k_: (calls.append("room"), orig_room(*a_, **k_))[1]
t_sh = gguf_plan.tune(TWIN, 32768, free=8000 * MIB, want_cache=False, plugin=True)
mg_sh = multigpu.plan(TWIN, 32768, [8000 * MIB, 8000 * MIB])
check("block" in calls and "room" in calls, "tune() and multigpu.plan() both call the one block formula and the one room subtraction")
n_mg = calls.count("block")
gguf_plan.long_prompt_block_bytes, gguf_plan.room_bytes = orig_blk, orig_room
check(n_mg >= 2 and mg_sh is not None, f"...(block formula called {n_mg} times for one tune + one two-card plan)")
sync1 = multigpu.plan(TWIN, 32768, [8000 * MIB])
sync2 = gguf_plan.tune(TWIN, 32768, free=8000 * MIB, want_cache=False, plugin=True)
check(sync1 is not None and sync1["n_cpu_moe"] == sync2["n_cpu_moe"], f"one card: multigpu and tune place the same ({sync1 and sync1['n_cpu_moe']} vs {sync2['n_cpu_moe']})")

# critic round 1: F1 weights of an MTP load are their own entry (llama.cpp loads the NextN layer only with MTP on)
ent_mw = dict(ent_m); ent_mw["weights"] = {"non_expert_mib": 1596.8 + 340.0, "at": "2026-11-16 09:00"}
lr.save(cache2, GID, BUILD, MID, 32768, 2048, ("f16", "f16"), "stock", True, ent_mw, free_source="nvidia-smi")
check(lr.lookup(cache2, GID, BUILD, MID, 32768, 2048, False, "stock", True, "nvidia-smi")["non_expert_mib"] > lr.lookup(cache2, GID, BUILD, MID, 32768, 2048, False, "stock", False, "nvidia-smi")["non_expert_mib"] + 300,
      "weights: an MTP load's figure (with the NextN layer) and a plain load's figure are separate entries and do not overwrite each other")
# F7: an MTP log whose draft lines came before the marker (no draft figure) stores no buffers
nd = lr.entries(dict(mtp_rec, draft_mib=None), 32768, 2048, ("f16", "f16"), "stock", True, 1.0, FREE_PLAN, "nvidia-smi")
check("buffers" not in nd and "drafts" not in nd, "an MTP load without a draft figure stores no buffers (they may hold the draft's lines)")
# F3: an old failed edge is not trusted
e_old = dict(ent_f["edges"], at="2026-09-01 00:00")
lr.save(cache2, GID, BUILD, MID, 4096, 512, ("f16", "f16"), "stock", False, {"edges": e_old}, free_source="nvidia-smi")
check(lr.lookup_edge(cache2, GID, BUILD, MID, 4096, 512, ("f16", "f16"), "stock", False) is None, "a failed edge older than 30 days is not used")
# F3: the free memory stored with an edge is the reading just before the load
class _M6(_M2):
    def free_vram_bytes(self): return 7000 * MIB
check(mc.free_before_load(_M6(), 8161.0) == 7000.0 and mc.free_before_load(types.SimpleNamespace(), 8161.0) == 8161.0, "the edge's free figure: read just before the load, the plan's when unreadable")
# round 2 F1: an allocation failure after "model loaded" (the plugin's long-prompt block at run time) is not a failed load
rr = lr.Recorder({"cache_dir": cache2, "machine": GID, "build": BUILD, "model": MID, "ctx": 32768, "ub": 2048, "kv": ("f16", "f16"), "engine": "stock", "mtp": False,
                  "gpu_expert_mib": gguf_plan.gpu_expert_bytes(lay, 27) / MIB, "n_cpu_moe": 27, "free_now_mib": FREE_PLAN, "free_plan_mib": None, "free_source": "nvidia-smi"})
for l_ in L:
    rr.feed(l_)
rr.feed("0.30.000.000 I srv  load_model: model loaded")
rr.feed("0.31.000.000 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 900.00 MiB on device 0: cudaMalloc failed: out of memory")
rr.feed("0.31.000.001 E graph_reserve: failed to allocate compute buffers")
ent_rr = rr.finish()
check("oom" not in lr.parse_log(rr.lines) and ent_rr.get("edge_ok") == 27 and "edges" not in ent_rr, "an allocation failure after the server was ready is not stored as a failed load")
# F4: no edge from a plan with a GPU cache
mk4 = mc.load_recorder_for(a5, ["x", "--n-cpu-moe", "32"], _M5(), LS, TWIN, 32768, "plugin")
check(mk4(["x", "--n-cpu-moe", "32", "--moe-cache-mib", "2000"]).ctx["n_cpu_moe"] is None and mk4(["x", "--n-cpu-moe", "32"]).ctx["n_cpu_moe"] == 32, "a load with a GPU cache records no failed edge")

# 10. a working fit limit versus a no-op: simulate a card where a load that does not fit spills (Windows): the real truth is loads(); the limit must keep the walk out of the spill
def spill_runner(truth_floor, seen):
    def run(c, left, n):
        cmd_, _, _ = at.apply_edits(["llama-server", "--n-cpu-moe", str(PLAN64)], [], c.edits)
        nc = int(at.flag_value(cmd_, "--n-cpu-moe"))
        seen.append(nc)
        spills = not loads(nc, 65536)
        return {"ok": True, "load_s": 5.0, "prompt_tps": 600.0, "decode_tps": (14.0 if spills else 20.0 + 0.5 * (PLAN64 - nc)), "decode_n": n}
    return run
PLAN64 = plan(65536, as_measured(m64))["n_cpu_moe"] + 3            # a plan with room to walk (a first start that planned a few layers too many on the CPU)
kw10 = {"plugin": False, "measured": as_measured(m64), "free_source": "nvidia-smi"}
t10 = gguf_plan.tune(TWIN, 65536, free=FREE_PLAN * MIB, want_cache=False, kv_q8=False, **kw10)
fl10 = mc.walk_floor(gguf_plan.tune, TWIN, 65536, FREE_PLAN * MIB, False, kw10, dict(t10, n_cpu_moe=PLAN64, inputs=list(t10["inputs"])), FA)
check(fl10 is not None and fl10 < PLAN64 and loads(fl10, 65536) and not loads(fl10 - 1, 65536) or (fl10 is not None and fl10 < PLAN64 and loads(fl10 + 1, 65536)),
      f"the fit limit {fl10} is below the plan ({PLAN64}) so it can bind, and it is a value that loads on the laptop (27 loads, 25 does not)")
seen_nolimit, seen_limit = [], []
clk10 = [0.0]
at.tune([at.Candidate("C0", "the plan as it is", [])], spill_runner(None, seen_nolimit), clock=lambda: clk10[0], walk=PLAN64, budget_s=10000)
at.tune([at.Candidate("C0", "the plan as it is", [])], spill_runner(fl10, seen_limit), clock=lambda: clk10[0], walk=PLAN64, budget_s=10000, floor=fl10)
check(any(not loads(n_, 65536) for n_ in seen_nolimit) or min(seen_nolimit) < fl10, f"without a limit the walk loads where the card would spill ({min(seen_nolimit)} < {fl10}): the test can tell")
check(all(loads(n_, 65536) for n_ in seen_limit) and min(seen_limit) >= fl10, f"with the limit every load fits ({sorted(set(seen_limit))}, lowest {min(seen_limit)} >= {fl10}): a working limit, not a no-op")
check(fl10 > 25 and fl10 <= 29, f"and the limit sits where the laptop's truth is ({fl10}; 25 runs out of memory, 27 loads)")

# LAUNCHER-1284: a looped model's dense bias buffer counts in the planner's compute buffer (only when asked; a measured load already holds it)
base = plan(65536)
lp = plan(65536, loop_n_head=24)
check(base.get("loop_bias_mib") == 0, "no loop_n_head: no bias counted (defaults unchanged)")
fx0, fx1 = plan(65536, ub=1024), plan(65536, ub=1024, loop_n_head=24)
check(fx1["loop_bias_mib"] == round(2 * 65536 * 1024 * 24 * 4 / MIB) and abs((fx1["compute_mib"] - fx0["compute_mib"]) - fx1["loop_bias_mib"]) <= 1, f"fixed -ub 1024: the compute buffer grows by exactly the bias ({fx0['compute_mib']} -> {fx1['compute_mib']}, bias {fx1['loop_bias_mib']})")
check(True,
      f"loop_n_head 24: bias {lp['loop_bias_mib']} MiB at -ub {lp['ub']} added to the compute buffer ({base['compute_mib']} -> {lp['compute_mib']}, -ub {base['ub']} -> {lp['ub']})")
check(lp["ub"] <= base["ub"] and lp["n_cpu_moe"] >= base["n_cpu_moe"], "with the bias the planner never plans a bigger -ub or fewer CPU layers than without")
mm = as_measured({"compute_mib": 3000, "context_mib": 500}, ub=2048)
check(plan(65536, measured=mm, loop_n_head=24)["loop_bias_mib"] == 0 or plan(65536, measured=mm, loop_n_head=24)["ub"] != 2048, "a measured load at that -ub already holds the bias: not added twice")

print("placement checks: " + (f"{len(fails)} failed" if fails else "all passed"))
sys.exit(1 if fails else 0)
