#!/usr/bin/env python3
"""Planner map fixes 1 + 2 (LAUNCHER-1177, 2026-11-15): one placement from the real numbers, and the auto-tune's edge walk.

Fixtures are real: the laptop logs of 2026-11-13 / 2026-11-15 (tests/data/load_logs/, RTX 4060 Laptop 8 GB, Qwen3.6-35B-A3B UD-IQ3_XXS, a llama.cpp build of 2026-11)
and a header twin of that model (tests/data/qwen36_35b_iq3xxs_layout.json: every tensor's name and size, a sparse file here). The laptop sweep (planner map
section 3): the lowest --n-cpu-moe that loads is 25 at 32k (23 runs out of GPU memory) and 27 at 64k (25 runs out); nvidia-smi showed 7.97 GiB free.
Model-free: nothing is loaded, no GPU is used."""
import json
import os
import struct
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
check(oom64 == {"failed_compute_mib": 788.02}, f"default-verbosity log of a failed 64k load: only the compute buffer it asked for, 788.02 MiB ({oom64})")
oom32 = lr.parse_log(lines("laptop_35b_32k_ncmoe23_oom.txt"))
check(oom32 == {"failed_compute_mib": 660.02}, f"failed 32k load: 660.02 MiB, the same as the lv4 log's real compute buffer ({oom32})")
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


def make_twin(path):
    d = json.load(open(os.path.join(HERE, "tests", "data", "qwen36_35b_iq3xxs_layout.json"), encoding="utf-8"))

    def string(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = []
    for k, v in d["meta"].items():
        if isinstance(v, str):
            kv.append(string(k) + struct.pack("<I", 8) + string(v))
        elif isinstance(v, float):
            kv.append(string(k) + struct.pack("<I", 6) + struct.pack("<f", v))
        else:
            kv.append(string(k) + struct.pack("<I", 4) + struct.pack("<I", v))
    infos, off = [], 0
    for name, size in d["tensors"]:
        infos.append(string(name) + struct.pack("<I", 1) + struct.pack("<Q", 1) + struct.pack("<IQ", 0, off))
        off += size
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(kv)) + b"".join(kv) + b"".join(infos)
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head)
        f.truncate(len(head) + off)          # sparse: no weights are written


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
check(lr.lookup(cache, GID, BUILD, MID, 32768, 2048, False, "stock", False) == {}, "nothing recorded: nothing measured (the plan falls back to the header and tables)")
check(lr.save(cache, GID, BUILD, MID, 32768, 2048, ("f16", "f16"), "stock", False, ent32) and lr.save(cache, GID, BUILD, MID, 65536, 2048, ("f16", "f16"), "stock", False, ent64),
      "both loads stored")
m32 = lr.lookup(cache, GID, BUILD, MID, 32768, 2048, False, "stock", False)
m64 = lr.lookup(cache, GID, BUILD, MID, 65536, 2048, False, "stock", False)
check(m32.get("compute_mib") == 660.0 and m32.get("gap_mib") == gap and m64.get("compute_mib") == 788.0 and "context_mib" not in m64 and m64.get("non_expert_mib"),
      "lookup: 32k has all parts; 64k has its compute buffer and the model's weights and overhead (they do not depend on the context)")
check(lr.lookup(cache, GID, "c" * 16, MID, 32768, 2048, False, "stock", False) == {}, "another llama.cpp build: nothing (measured again)")
check(lr.lookup(cache, GID, BUILD, MID, 32768, 2048, True, "stock", False).get("compute_mib") is None, "another KV type: no buffers")


def as_measured(m, ub=2048):
    out = {k: m[k] for k in ("non_expert_mib", "gap_mib") if k in m}
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
check(mc.measured_loads(A2, LS, TWIN, 32768, False, _M2(), False, False) == {}, "launcher: no record for this build yet: {}")
lr.save(cache, GID, build, MID, 32768, 2048, ("f16", "f16"), "stock", False, ent32)
got = mc.measured_loads(A2, LS, TWIN, 32768, False, _M2(), False, False)
check(got.get("buffers", {}).get(2048, {}).get("compute_mib") == 660.0 and got.get("gap_mib") == gap, f"launcher: the record is found ({got})")
check(mc.measured_loads(types.SimpleNamespace(vram="8"), LS, TWIN, 32768, False, _M2(), False, False) == {}, "launcher: --vram given: no record used")
os.environ["MOE_CACHE_LOAD_RECORD"] = "0"
check(mc.measured_loads(A2, LS, TWIN, 32768, False, _M2(), False, False) == {}, "launcher: MOE_CACHE_LOAD_RECORD=0: no record used")
del os.environ["MOE_CACHE_LOAD_RECORD"]
check(mc.user_set_log_verbosity(["llama-server", "-lv", "3"], {}) and mc.user_set_log_verbosity(["x"], {"LLAMA_ARG_LOG_VERBOSITY": "2"}) and not mc.user_set_log_verbosity(["x"], {}),
      "the user's own log level is left alone")
# the Recorder end to end: feed the real log, it stores under the plan's key
rec = lr.Recorder({"cache_dir": cache, "machine": GID, "build": "d" * 16, "model": MID, "ctx": 32768, "ub": 2048, "kv": ("f16", "f16"), "engine": "stock", "mtp": False,
                   "gpu_expert_mib": gguf_plan.gpu_expert_bytes(lay, 27) / MIB, "free_plan_mib": FREE_PLAN, "free_source": "nvidia-smi"})
for l in L:
    rec.feed(l + "\n")
rec.finish()
check(lr.lookup(cache, GID, "d" * 16, MID, 32768, 2048, False, "stock", False).get("compute_mib") == 660.0 and rec.saved == 1, "Recorder: the real log fed line by line is stored")

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
# M2 + M4: MTP loads and the user's own tensor placement are not recorded
_plan = types.SimpleNamespace(a=types.SimpleNamespace(vram=None, plan_free_mib=FREE_PLAN, plan_free_source="nvidia-smi"), cmd=["x", "--n-cpu-moe", "28"], machine=_M2(), ls=LS,
                              model=TWIN, ctx=65536, engine="stock")
mk = mc.make_load_recorder(_plan)
check(mk is not None and mk(["x", "--n-cpu-moe", "28", "-c", "65536"]) is not None, "recorder: the plan's own load is recorded")
check(mk(["x", "--n-cpu-moe", "28", "-ot", "blk.1=CPU"]) is None and mk(["x", "--n-cpu-moe", "28", "-ts", "1,1"]) is None and mk(["x", "--n-cpu-moe", "28", "-ngl", "20"]) is None,
      "recorder: -ot, -ts or a partial -ngl: not recorded")
check(mk(["x", "--n-cpu-moe", "28", "--spec-type", "draft-mtp"]) is None, "recorder: an MTP load is not recorded")
class _M3(_M2):
    def __init__(self, free): self.free = free
    def free_vram_bytes(self): return self.free * MIB
for fr, want in ((FREE_PLAN + 40, FREE_PLAN), (FREE_PLAN - 900, None)):
    _p = types.SimpleNamespace(**dict(vars(_plan), machine=_M3(fr)))
    r_ = mc.make_load_recorder(_p)(["x", "--n-cpu-moe", "28"])
    check(r_ is not None and r_.ctx.get("free_plan_mib") == want, f"recorder: free memory {fr - FREE_PLAN:+.0f} MiB since the plan: gap learned against {r_ and r_.ctx.get('free_plan_mib')}")
print("placement checks: " + (f"{len(fails)} failed" if fails else "all passed"))
sys.exit(1 if fails else 0)
