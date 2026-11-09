#!/usr/bin/env python3
"""LAUNCHER-991: the commit estimate, the pre-load decision (smaller layouts, then refusal) and the runtime guard, with fake commit figures and fake clocks.
   python3 tests/commit_guard_test.py
The incident: Qwen3.8-27B on the PC (RX 9070 XT, Vulkan; GPU allocations count as Windows commit) at 64k / 128k took llama-server to 15.7 / 17.6 GB of commit and
Windows killed claude.exe, dwm.exe and codex.exe (2026-11-09, twice). The model figures below are rounded stand-ins for that 27B (Q3_K_XL), not measurements."""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
import commit_guard as cg   # noqa: E402

GIB = 2**30
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def dense27(ctx_gib_f16_per_64k=4.0):
    """~27B hybrid dense: 13.4 GiB file, 0.6 GiB embedding, 64 layers of 0.12 GiB FFN, KV 4 GiB (f16) per 64k."""
    return {"total": 13.4 * GIB, "emb": 0.6 * GIB, "n_layers": 64, "dense": True, "expert_per_layer": 0.0, "ffn_per_layer": 0.12 * GIB, "ffn_layers": list(range(64)),
            "kv": lambda ctx, q8: ctx / 65536 * ctx_gib_f16_per_64k * GIB * (34 / 32 / 2 if q8 else 1.0)}

def moe():
    """A 35B MoE: 40 layers of 0.45 GiB experts, 3 GiB of other weights."""
    return {"total": 21 * GIB, "emb": 0.4 * GIB, "n_layers": 40, "dense": False, "expert_per_layer": 0.45 * GIB, "ffn_per_layer": 0.0, "ffn_layers": [],
            "kv": lambda ctx, q8: ctx * 20e3 * (0.53 if q8 else 1.0)}

def layout(**kw):
    L = {"ctx": 65536, "kv_q8": True, "ub": 512, "n_cpu_moe": 0, "ffn_off": 0, "cpu_only": False, "copy_weights": False, "cache_ram_mib": 2048, "plugin_private_gib": 0.0,
         "fit": False, "vram_free": 14.65 * GIB, "mm_bytes": 0.0}
    L.update(kw)
    return L

print("floors")
check(cg.soft_floor(48 * GIB) == 6 * GIB, "48 GB limit: soft floor is the 6 GB minimum")
check(abs(cg.soft_floor(96 * GIB) - 12 * GIB) < 1, "96 GB limit: 12.5 % = 12 GB")
check(cg.linux_floors(32 * GIB)[0] < 4 * GIB and cg.linux_floors(32 * GIB)[1] == GIB, "Linux soft line stays under the plugin's own 4 GiB RAM margin on 32 GB")

print("estimate")
f = dense27()
e = cg.estimate(f, layout())
ew = cg.estimate(f, layout(), windows=False)
check(e["total"] > ew["total"] + 10 * GIB, f"Windows counts the GPU allocations ({e['total'] / 1e9:.1f} GB), Linux does not ({ew['total'] / 1e9:.1f} GB)")
check(12e9 < e["gpu_total"] < 17e9, f"27B 64k q8_0 on the GPU: {e['gpu_total'] / 1e9:.1f} GB of GPU buffers (incident: llama-server at 15.7 GB)")
e128 = cg.estimate(f, layout(ctx=131072, kv_q8=False))
check(e128["total"] > e["total"] + 5 * GIB, f"128k f16 KV needs far more than 64k q8_0 ({e128['total'] / 1e9:.1f} vs {e['total'] / 1e9:.1f} GB)")
check(cg.estimate(f, layout(ffn_off=30, copy_weights=True))["total"] > cg.estimate(f, layout(ffn_off=30))["total"] + 3 * GIB, "copied weights count (those left in RAM); mapped ones do not")
check(cg.estimate(f, layout(cache_ram_mib=8192))["total"] - e["total"] > 5.9 * GIB, "the prompt cache (8 GiB default) counts")
check(cg.estimate(f, layout(ffn_off=20))["total"] < e["total"] - 2 * GIB, "FFN weights of 20 layers in RAM (mmapped) lower the commit")
cpu = cg.estimate(f, layout(cpu_only=True))
check(cpu["gpu_total"] == 0 and cpu["host_kv"] > 0, "CPU only: the KV cache is private host memory, so it counts")
sp = cg.estimate(f, layout(fit=True, vram_free=8 * GIB))
check(sp["gpu_total"] == 8 * GIB, "--fit: GPU buffers are capped by the free VRAM, the rest is mapped host memory")
# LAUNCHER-1029 review: the KV cache / compute buffers that --fit leaves in host memory are private memory, so they count even when the weights stay mapped
check(sp["spill_private"] > 0 and sp["spill_private"] <= sp["kv"] + sp["compute"], f"--fit spill: its KV/compute share counts as private host memory ({sp['spill_private'] / 1e9:.1f} GB)")
nosp = cg.estimate(f, layout(fit=True, vram_free=200 * GIB))
check(nosp["spill_private"] == 0 and sp["total"] > nosp["total"] - nosp["gpu_total"] + sp["gpu_total"], "--fit with room on the card: nothing spilled, nothing added")
spc = cg.estimate(f, layout(fit=True, vram_free=8 * GIB, copy_weights=True))
check(spc["spill_private"] == 0, "--fit with copied weights: the spill is already counted as copied weights, not twice")
check(sp["total"] > cg.estimate(f, layout()) ["total"] - cg.estimate(f, layout())["gpu_total"] + 8 * GIB, "--fit estimate is above the old one (GPU 8 GiB + host process only)")

print("guard: faster poll while the server is still loading (LAUNCHER-1029 review)")
_sl = []; _ld = [True, True, False, False]
_g = cg.CommitGuard(lambda: 10 * GIB, lambda t: None, lambda r: None, 6 * GIB, loading=lambda: _ld.pop(0) if _ld else False)
_al = iter([True] * 4 + [False]); _g.run(lambda: next(_al), sleep=_sl.append)
check(_sl == [0.25, 0.25, 2.0, 2.0], f"poll every 0.25 s while loading, every 2 s once ready ({_sl})")
_sl = []; _g = cg.CommitGuard(lambda: 10 * GIB, lambda t: None, lambda r: None, 6 * GIB); _al = iter([True, False]); _g.run(lambda: next(_al), sleep=_sl.append)
check(_sl == [2.0], "no loading() given: always 2 s")

print("decision: enough commit")
d = cg.decide(f, layout(), 30 * GIB, 48 * GIB)
check(d["ok"] and d["steps"] == [], "27B 64k q8_0 with 30 GB free of 48: starts as planned")

print("decision: tight commit, the owner's desktop with apps open")
d = cg.decide(f, layout(kv_q8=False, ub=2048, cache_ram_mib=8192, ctx=131072), 22 * GIB, 48 * GIB)
check(d["ok"] and d["steps"], f"27B 128k f16, 22 GB free: a smaller layout is found ({'; '.join(d['steps'])})")
check(d["layout"]["ctx"] == 131072, "the context is never shortened")
check(22 * GIB - d["est"]["total"] >= 6 * GIB, f"it leaves >= 6 GB free after the load ({(22 * GIB - d['est']['total']) / 1e9:.1f} GB)")
check(d["steps"][0].startswith("-ub") and any("q8_0" in s for s in d["steps"]), "order: smaller -ub first, q8_0 KV before weights go to RAM")
d2 = cg.decide(f, layout(kv_q8=False, ub=2048, cache_ram_mib=8192, ctx=131072), 36 * GIB, 48 * GIB, locked=("ub", "kv", "cache"))
check(d2["ok"] and d2["layout"]["ffn_off"] > 0 and not d2["layout"]["kv_q8"] and d2["layout"]["ub"] == 2048 and d2["layout"]["cache_ram_mib"] == 8192,
      f"user-fixed -ub / KV type / prompt cache stay; FFN of {d2['layout']['ffn_off']} layers go to RAM instead")
d3 = cg.decide(f, layout(kv_q8=False), 22 * GIB, 48 * GIB, allow_q8=False)
check(not any("q8_0" in s for s in d3["steps"]), "--exact: no q8_0 KV step")
d4 = cg.decide(f, layout(ffn_off=30, copy_weights=True), 21 * GIB, 48 * GIB)
check(d4["ok"] and d4["steps"][0].startswith("mmap"), "copied weights: mmap is the first fix")
d5 = cg.decide(moe(), layout(n_cpu_moe=10, ub=2048, kv_q8=False), 13 * GIB, 40 * GIB)
check(d5["ok"] and d5["layout"]["n_cpu_moe"] >= 10 and "n_cpu_moe" in d5["layout"], f"MoE: {'; '.join(d5['steps'])}")

print("decision: refusal")
d = cg.decide(f, layout(kv_q8=False, ub=2048, cache_ram_mib=8192, ctx=131072), 8 * GIB, 48 * GIB)
check(not d["ok"], "27B 128k with 8 GB free: refused")
txt = cg.refusal_text(d, 8 * GIB, 48 * GIB, ram_total=32 * GIB)
check("needs" in txt and "free commit" in txt and "you have 8.6 GB" in txt and "page file" in txt and "close apps" in txt, "refusal says needs N, you have M, close apps or set the page file")
print("     " + txt)
import re
m = re.search(r"needs (\d+\.\d) GB free commit", txt); pf = re.search(r"page file to at least (\d+) GB", txt)
check(bool(m) and bool(pf) and int(pf.group(1)) * 1e9 + 32e9 > float(m.group(1)) * 1e9 - 8e9, "the page-file size it names would cover the shortfall")
check(cg.decide(f, layout(cpu_only=True, kv_q8=False, ctx=131072), 5 * GIB, 48 * GIB)["ok"] is False, "CPU-only with the KV in private memory and no commit: refused, no weights lever")

print("command round trip")
cmd = ["llama-server", "-m", "m.gguf", "-ngl", "99", "--n-cpu-moe", "12", "-c", "65536", "-fa", "on", "--load-mode", "mmap", "-ub", "2048", "-b", "2048", "--cache-ram", "2048"]
Lc = cg.layout_from_cmd(cmd, [], moe())
check(Lc["ctx"] == 65536 and Lc["ub"] == 2048 and Lc["n_cpu_moe"] == 12 and Lc["cache_ram_mib"] == 2048 and not Lc["copy_weights"] and not Lc["kv_q8"] and not Lc["fit"], "layout read from a MoE command")
new = dict(Lc, ub=512, kv_q8=True, n_cpu_moe=20, cache_ram_mib=1024)
c2, ncmoe, ub, q8, ffn = cg.apply_layout_to_cmd(cmd, Lc, new, moe())
check(c2[c2.index("--n-cpu-moe") + 1] == "20" and c2[c2.index("-ub") + 1] == "512" and c2[c2.index("--cache-ram") + 1] == "1024" and "-ctk" in c2 and c2[c2.index("-ctv") + 1] == "q8_0" and ncmoe == 20 and ub == 512 and q8,
      "ub, --n-cpu-moe, --cache-ram and the q8_0 KV flags are written into the command")
dcmd = ["llama-server", "-m", "m.gguf", "-c", "131072", "-fa", "on", "--load-mode", "mmap", "-fitt", "256"]
Ld = cg.layout_from_cmd(dcmd, [], f, vram_free=14 * GIB)
check(Ld["fit"] and Ld["ffn_off"] == 0, "a dense command without -ngl / -ot is the --fit layout")
c3, _, _, _, ch = cg.apply_layout_to_cmd(dcmd, Ld, dict(Ld, ffn_off=10, fit=False), f)
check(ch and "-ot" in c3 and "-fitt" not in c3 and c3[c3.index("-ngl") + 1] == "99" and "blk\\.(54|55|56|57|58|59|60|61|62|63)\\.ffn_" in c3[c3.index("-ot") + 1], "--fit layout becomes -ngl 99 + FFN of the last 10 layers in RAM")
check(cg.layout_from_cmd(c3, [], f)["ffn_off"] == 10, "...and reads back as 10 layers")
Lm = cg.layout_from_cmd([x if x != "mmap" else "none" for x in cmd], [("MOE_CACHE_WIN_VIEW", "0"), ("MOE_CACHE_RAM_GIB", "9")], moe())
check(Lm["copy_weights"] and Lm["plugin_private_gib"] == 9.0, "--load-mode none is a copy; MOE_CACHE_WIN_VIEW=0 makes the plugin cache private memory")

print("runtime guard")
class Rig:
    def __init__(self, series, mode="stop", soft=6e9, hard=2e9):
        self.series, self.t, self.said, self.stops = list(series), 0.0, [], []
        self.g = cg.CommitGuard(lambda: self.series.pop(0) if self.series else None, self.said.append, self.stops.append, soft, hard, mode, clock=lambda: self.t)
    def step(self, dt=2.0):
        self.t += dt
        return self.g.poll()
r = Rig([20e9, 5.5e9, 5.4e9, 5.3e9, 4.0e9, 7.0e9])
res = [r.step() for _ in range(6)]
check(res == ["ok", "soft", "soft", "soft", "soft", "ok"], f"states {res}")
warn = [s for s in r.said if "WARNING" in s]
check(len(warn) == 2 and "5.5 GB" in warn[0] and "4.0 GB" in warn[1], "one warning when it drops under 6 GB, one more when it gets 1 GB worse; none for the small steps in between")
check(any("back to 7.0 GB" in s for s in r.said) and not r.stops, "a recovery line, and llama-server was not stopped")
r = Rig([5e9] * 20)
[r.step() for _ in range(16)]
check(len([s for s in r.said if "WARNING" in s]) == 2, "an unchanged shortage repeats its warning every 30 s (16 samples of 2 s = 2 warnings)")
r = Rig([5e9, 1.9e9, 1.8e9])
res = [r.step() for _ in range(3)]
check(res[:2] == ["soft", "soft"] and res[2] == "hard" and len(r.stops) == 1 and "hard floor" in r.stops[0] and "stopping llama-server" in r.stops[0], "under the hard floor for two samples: llama-server is stopped, with the reason")
check(any(s.startswith("moe-cache: WARNING") and "fell to 1.8 GB" in s for s in r.said), "the stop line is a 'moe-cache: WARNING' line the Monitor / control page lifts")
r = Rig([8e9, 0.9e9])
[r.step() for _ in range(2)]
check(len(r.stops) == 1, "a fall to under half the hard floor stops at once")
r = Rig([8e9, 1.9e9, 8e9, 1.9e9, 8e9])
[r.step() for _ in range(5)]
check(not r.stops, "one dip under the hard floor between healthy samples is not fatal")
r = Rig([1e9] * 5, mode="warn")
[r.step() for _ in range(5)]
check(not r.stops and r.said, "MOE_CACHE_COMMIT_GUARD=warn only warns")
r = Rig([None, None, 3e9])
check([r.step() for _ in range(3)] == [None, None, "soft"], "a failed probe is skipped")
check(cg.parse_mode(None) == "stop" and cg.parse_mode("OFF") == "off" and cg.parse_mode("nonsense") == "stop", "mode parsing")
r = Rig([5e9, 4e9])
alive = iter([True, True, True, False])
r.g.run(lambda: next(alive), sleep=lambda s: None)
check(not r.stops, "run() loops until the server is gone")
r = Rig([])
boom = []
r.g.probe = lambda: (_ for _ in ()).throw(OSError("probe broke"))
alive = iter([True] * 8 + [False])
r.g.run(lambda: next(alive), sleep=lambda s: None)
check(len([x for x in r.said if "cannot read" in x]) == 1 and "not protecting" in r.said[0], "a probe that always fails: ONE warning that the guard is not protecting the run")
d_ = cg.decide(dense27(), layout(fit=True, kv_q8=False, ctx=131072, vram_free=14.65 * GIB), 23.5 * GIB, 48 * GIB, locked=("kv", "ub", "cache"))
check(d_["ok"] and 0 < d_["layout"]["ffn_off"] < 64 and not d_["layout"]["fit"] and 23.5 * GIB - d_["est"]["total"] >= cg.soft_floor(48 * GIB),
      f"--fit layout: the weights step is sized on the uncapped layout (auditor case: {d_['layout']['ffn_off']} layers, {d_['est']['total'] / GIB:.1f} GiB), no false refusal")
d_ = cg.decide(dense27(), layout(fit=True, kv_q8=False, ctx=131072, vram_free=14.65 * GIB), 23.5 * GIB, 48 * GIB, locked=("ub", "cache"))
check(d_["ok"] and any("q8_0" in x for x in d_["steps"]) and d_["layout"]["kv_q8"], "--fit layout, KV free: the q8_0 step is taken, because the KV cache --fit leaves in RAM is private memory and a q8_0 one is smaller (LAUNCHER-1029 review; before, the spill was not counted)")
check("estimate, not yet measured" in cg.refusal_text(cg.decide(dense27(), layout(), 3 * GIB, 48 * GIB), 3 * GIB, 48 * GIB, 32 * GIB), "the refusal says the estimate is not measured yet")

print("launcher: the plan on a fake Windows PC (synthetic 18 GB MoE header, sparse file)")
import tempfile, types
sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, os.path.join(HERE, "bin"))
import launch_plan_test as lp   # noqa: E402
import moe_cache as mc          # noqa: E402
lp.MACHINES["win_roomy"] = dict(lp.MACHINES["win"], commit=(96e9, 80e9))
lp.MACHINES["win_tight"] = dict(lp.MACHINES["win"], commit=(48e9, 17e9))
lp.MACHINES["win_full"] = dict(lp.MACHINES["win"], commit=(48e9, 7e9))
with tempfile.TemporaryDirectory() as tmp:
    model = os.path.join(tmp, "synthetic-moe.gguf"); lp.make_synthetic_model(model)
    run = lambda m, opts=(), extra=(), flag="--dry-run": lp.run_case(m, ["--ram", "12"] + list(opts), list(extra), model, tmp, flag=flag)
    cmdline = lambda r: next((l for l in r["stdout"].splitlines() if "llama-server" in l and " -m " in l), "")
    txt = lambda r: r["stdout"] + "\n" + r["stderr"]
    roomy = run("win_roomy", ["--ctx", "65536"])
    check("free commit is tight" not in txt(roomy) and "not starting" not in txt(roomy), "plenty of commit: nothing said, plan unchanged")
    tight = run("win_tight", ["--ctx", "65536"])
    tight_line = next((l for l in tight["stderr"].splitlines() if "free commit is tight" in l), "")
    check(bool(tight_line) and ("(was " in tight_line or "more layer" in tight_line or "q8_0" in tight_line), f"17 GB free of 48: a smaller layout is chosen and the line names the change ({tight_line[100:200]})")
    check("MOE_CACHE_HOST_RESERVE_MIB" in tight["stdout"], "the plan still carries the plugin's host reserve")
    # (4) the reserve follows the prompt cache the stage chose, at every commit level
    import re as _re
    for free in (14e9, 16e9, 18e9, 20e9, 24e9, 30e9):
        lp.MACHINES["win_x"] = dict(lp.MACHINES["win"], commit=(48e9, free))
        r = run("win_x", ["--ctx", "65536"])
        c = _re.search(r"--cache-ram (\d+)", cmdline(r)); rs = _re.search(r'MOE_CACHE_HOST_RESERVE_MIB="(\d+)"', r["stdout"])
        if "not starting" in r["stderr"]:
            continue
        check(bool(c) and bool(rs) and c.group(1) == rs.group(1), f"{free / 1e9:.0f} GB free: --cache-ram {c and c.group(1)} and the plugin's host reserve {rs and rs.group(1)} agree")
    check("-c 65536" in cmdline(tight), "...the context is still 65536")
    pinned = run("win_tight", ["--ctx", "65536", "--no-commit-check"])
    check("free commit is tight" not in txt(pinned) and "not starting" not in txt(pinned) and "GPU plan" not in txt(pinned), "--no-commit-check: the first plan stands, nothing said, the old GPU-plan cap is not applied either")
    lp.MACHINES["win_nv"] = dict(lp.MACHINES["win"], commit=(48e9, 17e9), nvidia=True, listing=lp.LINUX_LIST.replace("CUDA0", "CUDA0"))
    _cb = mc.cuda_backend
    mc.cuda_backend = lambda ls: True
    try:
        nv = run("win_nv", ["--ctx", "65536"])
        check("free commit is tight" not in txt(nv) and "not starting" not in txt(nv), "Windows + NVIDIA + the CUDA backend next to llama-server: GPU memory is not counted as commit (only the host side), so 17 GB free of 48 changes nothing")
        mc.cuda_backend = lambda ls: False
        nvk = run("win_nv", ["--ctx", "65536"])
        check("free commit is tight" in txt(nvk) or "not starting" in txt(nvk), "Windows + NVIDIA card on the Vulkan build (no CUDA backend): GPU memory counts as commit like on AMD, so 17 GB free of 48 is tight")
    finally:
        mc.cuda_backend = _cb
    import tempfile as _t
    with _t.TemporaryDirectory() as _d:
        _ls = os.path.join(_d, "llama-server"); open(_ls, "w").close()
        check(not mc.cuda_backend(_ls), "cuda_backend: no ggml-cuda next to llama-server -> False")
        open(os.path.join(_d, "libggml-cuda.so"), "w").close()
        check(mc.cuda_backend(_ls), "cuda_backend: ggml-cuda next to llama-server -> True")
    full = run("win_full", ["--ctx", "131072"])
    check("not starting: this model at this context needs" in full["stderr"] and "free commit" in full["stderr"] and "page file" in full["stderr"] and "close apps" in full["stderr"], "7 GB free of 48: the plan carries the refusal text")
    userub = run("win_tight", ["--ctx", "65536", "--ub", "2048"])
    check(("-ub 2048" in cmdline(userub)) or "not starting" in userub["stderr"], "a --ub the user gave is not changed")
    # (1) the refusal comes before anything loads a model, the --approx canary included
    canary_calls = []
    saved_canary = mc.approx_canary
    mc.approx_canary = lambda plan: canary_calls.append(1)
    try:
        ap = run("win_full", ["--ctx", "131072", "--approx", "careful", "--approx-unverified"], flag="--no-warm")
    except SystemExit:
        ap = {"stderr": ""}
    finally:
        mc.approx_canary = saved_canary
    check(not canary_calls and "not starting: this model at this context needs" in ap["stderr"], f"--approx careful on a PC with no commit: refused before the canary runs (canary calls {len(canary_calls)})")
    lin = run("linux", ["--ctx", "65536"])
    check("commit" not in lin["stderr"], "Linux: no commit lines")
    # the refusal stops the real start
    pl = mc.LaunchPlan(); pl.commit_refusal = "moe-cache: not starting: test"
    pl.a, pl.extra, pl.machine = types.SimpleNamespace(dry_run=False), [], None
    for k in ("model", "ls", "port", "ctx", "host", "cmd", "env", "key_env", "method", "win_limit", "api_key", "alias", "slotdir", "kv_q8", "ncmoe_user"):
        setattr(pl, k, None)
    try:
        mc.run_launch_plan(pl); refused = False
    except SystemExit as e:
        refused = "not starting" in str(e.code)
    check(refused, "run_launch_plan refuses to start a plan with a commit refusal")

print("launcher: the 27B incident, dense model on the PC (synthetic ~27B header: 64 layers, 13 GiB, 2 KV heads)")
import struct
def make_dense27(path, n_layers=64, d=8192, kvh=2, heads=64):
    def string(x):
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    infos, off = [], 0
    def add(name, dims, ty, nbytes):
        nonlocal off
        infos.append((name, dims, ty, off)); off += (nbytes + 31) // 32 * 32
    add("token_embd.weight", [d, 32000], 12, d // 256 * 144 * 32000)
    for l in range(n_layers):
        add(f"blk.{l}.attn_q.weight", [d, d], 12, d // 256 * 144 * d)
        add(f"blk.{l}.ffn_up.weight", [d, 4 * d], 12, d // 256 * 144 * 4 * d)
    kv = [("general.architecture", 8, string("llama")), ("llama.block_count", 4, struct.pack("<I", n_layers)), ("llama.context_length", 4, struct.pack("<I", 262144)),
          ("llama.embedding_length", 4, struct.pack("<I", d)), ("llama.attention.head_count", 4, struct.pack("<I", heads)), ("llama.attention.head_count_kv", 4, struct.pack("<I", kvh))]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(kv))
    for k, t, v in kv:
        head += string(k) + struct.pack("<I", t) + v
    for name, dims, ty, o in infos:
        head += string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", x) for x in dims) + struct.pack("<IQ", ty, o)
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head); f.truncate(len(head) + off)
with tempfile.TemporaryDirectory() as tmp:
    dm = os.path.join(tmp, "dense27.gguf"); make_dense27(dm)
    ff = cg.facts_from_model(dm)
    check(ff["dense"] and ff["n_layers"] == 64 and 10 * GIB < ff["total"] < 16 * GIB and ff["ffn_per_layer"] > 0.1 * GIB, f"facts_from_model reads the dense header ({ff['total'] / GIB:.1f} GiB, {ff['n_layers']} layers, FFN {ff['ffn_per_layer'] / GIB:.2f} GiB per layer)")
    check(abs(ff["kv"](65536, False) / GIB - 4.0) < 0.1, f"KV f16 at 64k from the header: {ff['kv'](65536, False) / GIB:.1f} GiB")
    for name, commit in (("w_ok", (48e9, 40e9)), ("w_tight", (48e9, 17e9)), ("w_full", (48e9, 9e9))):
        lp.MACHINES[name] = dict(lp.MACHINES["win"], commit=commit)
    rund = lambda m, ctx, opts=(), extra=(): lp.run_case(m, ["--ram", "12", "--ctx", str(ctx)] + list(opts), list(extra), dm, tmp)
    cl = lambda r: next((l for l in r["stdout"].splitlines() if "llama-server" in l and " -m " in l), "")
    for ctx in (65536, 131072):
        ok = rund("w_ok", ctx)
        check("free commit is tight" not in ok["stderr"] and "not starting" not in ok["stderr"], f"27B at {ctx}: 40 GB free commit of 48, started as planned")
        t = rund("w_tight", ctx)
        check("GPU plan" not in t["stderr"], f"27B at {ctx}: the old fixed-4-GiB cap is not applied on top (the layout is cut once)")
        check(f"-c {ctx}" in cl(t), f"27B at {ctx}: 17 GB free commit of 48, context stays {ctx}")
        check(("free commit is tight" in t["stderr"]) or ("not starting" in t["stderr"]), f"27B at {ctx}: 17 GB free commit of 48 changes the plan or refuses ({(t['stderr'].splitlines() or [''])[-1][:150]})")
        fl = rund("w_full", ctx)
        check("not starting: this model at this context needs" in fl["stderr"], f"27B at {ctx}: 9 GB free commit of 48 is refused")
    t = rund("w_tight", 131072, ["--exact"])
    check("-ctk q8_0" not in cl(t), "27B 128k --exact: the q8_0 KV cache is not used to save commit")

print("launcher: the guard on a fake process")
class FakeProc:
    def __init__(self): self.alive, self.terminated = True, False
    def poll(self): return None if self.alive else 0
    def terminate(self): self.terminated, self.alive = True, False
said, saved_say = [], mc.say
mc.say = lambda m, err=False: said.append(m)
try:
    seq = iter([20e9, 5e9, 1.5e9, 1.4e9, 1.3e9])
    saved_w, saved_win, saved_mi = mc.WINDOWS, mc.win_commit, mc.meminfo_available
    mc.WINDOWS, mc.win_commit = True, lambda: (48e9, 20e9)
    p = FakeProc()
    g = mc.start_commit_guard(p, {}, probe=lambda: next(seq, 1e9))
    check(g is not None and g.soft == cg.soft_floor(48e9) and g.hard == cg.HARD_FLOOR, "Windows guard: soft line is max(6 GB, 12.5 % of the limit), hard line 2 GB")
    g.stopped = None
    g.stop_fn("x"); check(p.terminated, "the guard's stop ends the process (terminate)")
    check(mc.start_commit_guard(FakeProc(), {"MOE_CACHE_COMMIT_GUARD": "off"}) is None, "MOE_CACHE_COMMIT_GUARD=off: no guard")
    mc.WINDOWS = False
    mc.meminfo_available = lambda: 10e9
    g2 = mc.start_commit_guard(FakeProc(), {})
    check(g2 is not None and g2.label == "available memory" and g2.hard == GIB and g2.mode == "warn", "Linux guard watches MemAvailable with a 1 GiB hard floor and only warns by default")
    g3 = mc.start_commit_guard(FakeProc(), {"MOE_CACHE_COMMIT_GUARD": "stop"})
    check(g3 is not None and g3.mode == "stop", "Linux: MOE_CACHE_COMMIT_GUARD=stop turns the stop on")
    mc.WINDOWS = True
    check(mc.start_commit_guard(FakeProc(), {}, probe=lambda: 50e9).mode == "stop", "Windows: stop is the default")
    said.clear()
    r = Rig([5e9, 1.9e9, 1.8e9])
    [r.step() for _ in range(3)]
    mc.say_exit(1, {"guard": r.g, "ready": True, "last": ""}, False)
    check(any("stopped by the memory guard" in m for m in said) and not any("exited" in m for m in said), "say_exit: a guard stop is named as such, not as a crash")
    said.clear()
    r = Rig([5e9, 4e9]); [r.step() for _ in range(2)]
    mc.say_exit(0, {"guard": r.g, "ready": True, "last": ""}, False)
    check(any("lowest free commit seen" in m and "4.0 GB" in m for m in said), "say_exit: the lowest free commit seen is reported when it went under the warning line (calibration data)")
    said.clear()
    mc.win_commit = lambda: (None, None)
    mc.WINDOWS = True
    check(mc.start_commit_guard(FakeProc(), {}) is None and any("memory guard is off" in m for m in said), "no commit figure: the guard says it is off")
finally:
    mc.say, mc.WINDOWS, mc.win_commit, mc.meminfo_available = saved_say, saved_w, saved_win, saved_mi

print("FAIL" if fails else "all ok")
sys.exit(1 if fails else 0)
