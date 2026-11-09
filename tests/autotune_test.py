#!/usr/bin/env python3
"""LAUNCHER-994: measured auto-tune (bin/moe_cache_autotune.py + autotune_stage / autotune_probe in bin/moe_cache.py). Fake timings and a fake llama-server (a small
HTTP script); no model weights, no real server.
   python3 tests/autotune_test.py"""
import contextlib, io, json, os, stat, sys, tempfile, time, types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, os.path.join(HERE, "planner"))
import moe_cache as mc  # noqa: E402
import moe_cache_autotune as at  # noqa: E402
import launch_plan_test as lpt  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

BASE_CMD = ["llama-server", "-m", "m.gguf", "--port", "8080", "-c", "65536", "-ub", "4096", "-b", "4096", "--n-cpu-moe", "20", "-ctk", "q8_0", "-ctv", "q8_0"]
BASE_ENV = [("MOE_CACHE_PRELOAD", "1")]

# --- candidates ---
c = at.build_candidates(BASE_CMD, BASE_ENV, {}, 65536, n_layers=40)
check(c[0].id == "C0" and not c[0].edits and 2 <= len(c) <= 4, f"candidates: C0 first, 2-4 in all ({[x.id for x in c]})")
check(len({x.id for x in c}) == len(c), "candidate ids are unique")
c = at.build_candidates(BASE_CMD, BASE_ENV, {"ncmoe": "yours", "kv": "yours", "ub": "yours"}, 65536)
check([x.id for x in c] == ["C0", "ramlock", "np1"], f"locked dimensions are not tuned ({[x.id for x in c]})")
c = at.build_candidates(BASE_CMD, BASE_ENV, {d: "x" for d in ("ncmoe", "kv", "ub", "ramlock", "np")}, 65536)
check(len(c) == 1, "everything locked: only the plan as it is")
# LAUNCHER-1043: the 1.5k-token probe cannot judge the KV type or a batch above its own length: the measured long-prompt default (-ub 4096, q8_0 KV) is never undone
for ctx_ in (8192, 65536, 262144):
    check(not any("kv" in x.id or "kv" in x.dims for x in at.build_candidates(BASE_CMD, BASE_ENV, {}, ctx_, n_layers=40)), f"never a KV candidate (ctx {ctx_})")
for plan_ub, want in (("4096", []), ("2048", []), ("1024", []), ("750", ["ub1500"]), ("512", ["ub1024"]), ("256", ["ub512"]), ("128", ["ub256"])):
    cmd_ = [t if t != "4096" else plan_ub for t in BASE_CMD]
    got = [x.id for x in at.build_candidates(cmd_, BASE_ENV, {}, 65536, n_layers=40) if x.id.startswith("ub")]
    check(got == want, f"-ub candidates for a plan of {plan_ub}: {got} (only upward, never above the {at.PROMPT_TOKENS}-token probe prompt)")
check(not any(e.get("flag") in ("-ctk", "-ctv") for x in at.build_candidates(BASE_CMD, BASE_ENV, {}, 65536, n_layers=40) for e in x.edits), "no candidate edits -ctk / -ctv")

# --- apply_edits ---
cand = {x.id: x for x in at.build_candidates(BASE_CMD, BASE_ENV, {}, 65536, n_layers=40)}
cmd, env, sk = at.apply_edits(BASE_CMD, BASE_ENV, cand["ncmoe-2"].edits)
check(at.flag_value(cmd, "--n-cpu-moe") == "18" and not sk, "ncmoe-2 is a delta on the current plan (20 -> 18)")
cmd, env, sk = at.apply_edits(BASE_CMD[:-4] + ["--n-cpu-moe", "12"], BASE_ENV, cand["ncmoe-2"].edits)
check(at.flag_value(cmd, "--n-cpu-moe") == "10", "the same delta on a plan of 12 gives 10 (a changed GPU memory later cannot carry 18 over)")
cmd, env, sk = at.apply_edits(BASE_CMD, BASE_ENV, cand["ncmoe-2"].edits, locked={"ncmoe": "yours"})
check(cmd == BASE_CMD and sk == ["ncmoe"], "a locked dimension is skipped and named")
SMALL_CMD = [t if t != "4096" else "512" for t in BASE_CMD]
cmd, env, sk = at.apply_edits(SMALL_CMD, BASE_ENV, {x.id: x for x in at.build_candidates(SMALL_CMD, BASE_ENV, {}, 65536, n_layers=40)}["ub1024"].edits)
check(cmd.count("-ub") == 1 and at.flag_value(cmd, "-ub") == "1024", "ub edit replaces, never duplicates")
cmd, env, sk = at.apply_edits(BASE_CMD, BASE_ENV, at.Candidate("r", "r", [{"dim": "ramlock", "env": k, "value": v} for k, v in at.RAMLOCK_ENV]).edits)
check(dict(env).get("MOE_CACHE_RAM_LOCK") == "1" and dict(env).get("MOE_CACHE_PRELOAD") == "1", "env edits add to the environment")

# --- locked_dims ---
A = lambda **k: types.SimpleNamespace(**dict(dict(ub="auto", exact=False, n_cpu_moe=None, hybrid=False), **k))
check(at.locked_dims(A(), ["m.gguf"], {}) == {}, "no user flags: nothing locked")
l = at.locked_dims(A(), ["m.gguf", "--", "-ub", "512", "-np", "2", "--n-cpu-moe", "8", "-ctk", "f16"], {"MOE_CACHE_RAM_LOCK": "1"})
check(set(l) == {"ub", "np", "ncmoe", "kv", "ramlock"}, f"user flags and env lock their dimensions ({sorted(l)})")
check("ub" in at.locked_dims(A(ub="1024"), [], {}), "--ub N locks ub")
check("ramlock" in at.locked_dims(A(), [], {}, windows=True), "RAM_LOCK is Linux only")
check("np" in at.locked_dims(A(), [], {}, agent=True), "agent mode keeps its slots")

# --- tune(): picking ---
def timings(table):
    def runner(cand, left, n):
        r = table.get(cand.id)
        if isinstance(r, Exception):
            raise r
        return dict(r, decode_n=n) if r else {"ok": False, "error": "x"}
    return runner
class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
cands = at.build_candidates(BASE_CMD, BASE_ENV, {}, 65536, n_layers=40)
ids = [x.id for x in cands]
T = {"C0": dict(ok=True, load_s=10, prompt_tps=300, decode_tps=20)}
clk = Clock()
def run_t(table, step=20.0, budget=180.0, agent=False, clk=None):
    clk = clk or Clock()
    def runner(cand, left, n):
        clk.t += step
        return timings(table)(cand, left, n)
    return at.tune(cands, runner, agent=agent, budget_s=budget, clock=clk)
T[ids[1]] = dict(ok=True, load_s=10, prompt_tps=420, decode_tps=21)           # +40 % prompt: clearly faster
T[ids[2]] = dict(ok=True, load_s=10, prompt_tps=290, decode_tps=19)
r = run_t(T)
check(r["picked"] == ids[1] and r["changed"] and r["state"] == "tuned", f"the fastest candidate is picked ({r['picked']})")
check("faster" in r["why"] and "tok/s" in r["why"], "the reason names the numbers")
T2 = dict(T); T2[ids[1]] = dict(ok=True, load_s=10, prompt_tps=310, decode_tps=20.4)       # +2 %: inside the noise margin
r = run_t(T2)
check(r["picked"] == "C0" and not r["changed"] and "stays" in r["why"], "a gain under 5 % keeps the plan as it is")
T3 = dict(T); T3[ids[1]] = None; T3[ids[2]] = RuntimeError("boom")
r = run_t(T3)
check(r["picked"] == "C0" and sum(x["status"] == "failed" for x in r["rows"]) >= 2, "failed and raising candidates are dropped, the start is not")
r = run_t({"C0": None})
check(r["state"] == "too_slow" and r["picked"] == "C0" and len(r["rows"]) == 1, "C0 not measurable: too_slow, nothing picked, nothing else tried")
# budget: each probe 70 s; C0 takes 70, a second would end at 140 (fits), the third needs more than the 40 s left
r = run_t(T, step=70.0)
done = [x["id"] for x in r["rows"] if x["status"] == "measured"]
check(done == ids[:2] and any("no time left" in str(x.get("error")) for x in r["rows"]), f"the 180 s budget stops later candidates ({done})")
check(r["total_s"] <= 180.0 + 1e-6 + 70, "total time is within budget plus the last probe")
# short probe: C0 took 60 s with 512 tokens at 20 tok/s (25.6 s decode, 34 s load+prompt); at 140 s used, 40 s left -> room for ~ (40 - 39) s decode
T4 = {i: dict(T["C0"], prompt_tps=300 + 40 * k, decode_tps=20) for k, i in enumerate(ids)}
clk = Clock(); steps = iter([60, 60, 60, 60])
def runner4(cand, left, n):
    clk.t += next(steps); return dict(T4[cand.id], decode_n=n)
r = at.tune(cands, runner4, budget_s=130.0, clock=clk)
sp = [x for x in r["rows"] if x.get("short_probe")]
check(all((x["decode_n"] or 0) < at.DECODE_TOKENS for x in sp), "a decode probe cut short says so (short_probe, decode_n < 512)")
check(all((x["decode_n"] or 0) >= at.SHORT_MIN for x in r["rows"] if x["status"] == "measured"), "no decode figure below 64 tokens is used")
full = [x for x in run_t(T)["rows"] if x["status"] == "measured"]
check(all(x["decode_n"] == 512 and not x["short_probe"] for x in full), "with time to spare every decode probe is 512 tokens")
# agent mode weights the prompt more
Tq = {"C0": dict(ok=True, prompt_tps=100, decode_tps=30), ids[1]: dict(ok=True, prompt_tps=130, decode_tps=24), ids[2]: None}
check(run_t(Tq, agent=True)["picked"] == ids[1] and run_t(Tq, agent=False)["picked"] in ("C0", ids[1]), "agent mode scores a long prompt")

# --- the baseline is measured twice (C0 runs cold) ---
seen = {"C0": 0}
def cold_runner(cand, left, n):
    clk.t += 20
    if cand.id == "C0":
        seen["C0"] += 1
        return dict(ok=True, load_s=5, prompt_tps=250 if seen["C0"] == 1 else 340, decode_tps=17 if seen["C0"] == 1 else 23, decode_n=n)    # the first run is cold
    return dict(ok=True, load_s=5, prompt_tps=300, decode_tps=20, decode_n=n)                                                               # +20 % over the cold C0 only
clk = Clock()
r = at.tune(cands, cold_runner, clock=clk)
check([x["id"] for x in r["rows"]][-1] == "C0b" and r["rows"][-1]["status"] == "measured", "the plan as it is runs a second time (C0b) when the budget allows")
check(r["picked"] == "C0" and not r["changed"] and "moved" in r["why"], "a candidate that only beats the COLD baseline is not picked; the why names the baseline spread")
check(r.get("baseline_spread", 0) > 0.05, "the baseline spread is stored")
seen["C0"] = 0; clk = Clock()
r = at.tune(cands, cold_runner, clock=clk, budget_s=70.0)
check(all(x["id"] != "C0b" for x in r["rows"]), "no time for the second baseline run: not attempted")

# --- a gain inside twice the baseline's own spread is not a verdict (LAUNCHER-1045, margin part) ---
seen = {"C0": 0}
def noisy(cand, left, n):
    clk.t += 20
    if cand.id == "C0":
        seen["C0"] += 1
        return dict(ok=True, load_s=5, prompt_tps=300, decode_tps=20 if seen["C0"] == 1 else 18.8, decode_n=n)       # the baseline moves 6 %
    return dict(ok=True, load_s=5, prompt_tps=300, decode_tps=22.0, decode_n=n)                                         # about 6 % better score than the better baseline run: over 5 %, under 2 x the spread
clk = Clock()
r = at.tune(cands, noisy, clock=clk)
check(r["picked"] == "C0" and not r["changed"], f"a 6 % gain on a baseline that moves 6 % is not picked ({r['picked']}, spread {r.get('baseline_spread')})")
seen["C0"] = 0; clk = Clock()
def clear(cand, left, n):
    clk.t += 20
    if cand.id == "C0":
        seen["C0"] += 1
        return dict(ok=True, load_s=5, prompt_tps=300, decode_tps=20 if seen["C0"] == 1 else 18.8, decode_n=n)
    return dict(ok=True, load_s=5, prompt_tps=300, decode_tps=25, decode_n=n)                                            # +25 %: far outside the spread
r = at.tune(cands, clear, clock=clk)
check(r["changed"] and r["picked"] != "C0", "a 25 % gain is still picked")

# --- cache file ---
with tempfile.TemporaryDirectory() as tmp:
    parts = at.key_parts("m.gguf:1:2", 65536, {"cpu": "x"}, {}, False, "plugin")
    check(at.load(tmp, parts) is None, "no file: no stored pick")
    res = run_t(T)
    p = at.save(tmp, parts, res)
    check(os.path.isfile(p) and os.path.dirname(p).endswith("autotune"), "stored in <cache>/autotune/<key>.json")
    got = at.load(tmp, parts)
    check(got and got["picked"] == ids[1] and at.pick_edits(got) == at.Candidate.from_dict(got["rows"][1]).edits, "stored pick reads back")
    check(at.load(tmp, at.key_parts("m.gguf:1:2", 131072, {"cpu": "x"}, {}, False, "plugin")) is None, "another context: another key")
    check(at.load(tmp, at.key_parts("m.gguf:1:2", 65536, {"cpu": "y"}, {}, False, "plugin")) is None, "another machine: another key")
    check(at.load(tmp, at.key_parts("m.gguf:1:3", 65536, {"cpu": "x"}, {}, False, "plugin")) is None, "another model file: another key")
    check(at.load(tmp, at.key_parts("m.gguf:1:2", 65536, {"cpu": "x"}, {"ub": "yours"}, False, "plugin")) is None, "other user flags: another key")
    check(at.load(tmp, at.key_parts("m.gguf:1:2", 65536, {"cpu": "x"}, {}, False, "plugin", {"ram": "8"})) is None, "another --ram / --vram cap: another key")
    d = json.load(open(p)); d["version"] = 0; json.dump(d, open(p, "w"))
    check(at.load(tmp, parts) is None, "an old schema version is ignored")
    open(p, "w").write("{broken")
    check(at.load(tmp, parts) is None, "a broken file is ignored, never an error")

# --- autotune_stage through the launcher ---
tmp = tempfile.mkdtemp(prefix="autotune-")
model = os.path.join(tmp, "m.gguf"); lpt.make_synthetic_model(model)
ls = os.path.join(tmp, "llama-server"); open(ls, "w").close()
saved = (mc.CACHE_DIR, mc.port_busy)
mc.CACHE_DIR = os.path.join(tmp, "cache"); mc.port_busy = lambda port: False
class M(lpt.FakeMachine):
    def physical_cores(self): return 8
    def other_llama_servers(self): return []
    def list_devices(self, ls): return "Available devices:\n  CUDA0: Fake GPU (8188 MiB, 7600 MiB free)\n"
calls = []
def fake_runner(cmd, env, left, n):
    calls.append((cmd, env, left, n))
    ub = at.flag_value(cmd, "-ub"); nc = at.flag_value(cmd, "--n-cpu-moe")
    fast = (nc == "18")                                   # fewer CPU expert layers is the fast one in this fake
    return dict(ok=True, load_s=5, prompt_tps=400 if fast else 300, decode_tps=22 if fast else 20, decode_n=n)
def stage(argv=("--dry",), probe=True, retune=False, exact=False, stock=False, runner=fake_runner, machine=None, **k):
    a = types.SimpleNamespace(exact=exact, stock=stock, retune=retune, agent=False, project=None, ub="auto", n_cpu_moe=None, hybrid=False, port=8080, autotune_probe=probe, **k)
    out = []
    r = mc.autotune_stage(a, list(argv), list(BASE_CMD), list(BASE_ENV), [], model, 65536, ls, machine or M(tmp), "plugin", 40, say_fn=lambda t, err=False: out.append(t), runner=runner)
    return r + (out,)
try:
    cmd, env, info, out = stage(probe=False)
    check(info["state"] == "pending" and cmd == BASE_CMD and not calls and not out, "dry-run / plan-json: no stored pick -> nothing measured, nothing said, plan unchanged")
    cmd, env, info, out = stage()
    check(info["state"] == "tuned" and info["picked"] != "C0" and at.flag_value(cmd, "--n-cpu-moe") == "18", f"first real start measures and applies the fastest ({info['picked']})")
    check(len(calls) >= 2 and calls[0][0] == BASE_CMD, f"C0 was measured first with the plan's own command ({len(calls)} probes)")
    check(any("picked" in l for l in out) and any("tok/s" in l for l in out), "the launcher output says what was picked and why, with the numbers")
    n = len(calls)
    cmd, env, info, out = stage()
    check(info["state"] == "cached" and len(calls) == n and at.flag_value(cmd, "--n-cpu-moe") == "18", "second start uses the stored pick, no probe")
    check(any("cached" in l and "--retune" in l for l in out), "the cached line names --retune")
    cmd, env, info, out = stage(probe=False)
    check(info["state"] == "cached" and at.flag_value(cmd, "--n-cpu-moe") == "18", "dry-run / plan-json show and apply the stored pick")
    cmd, env, info, out = stage(retune=True)
    check(len(calls) > n and info["state"] == "tuned", "--retune measures again")
    n = len(calls)
    cmd, env, info, out = stage(exact=True)
    check(info is None and cmd == BASE_CMD and len(calls) == n, "--exact: the tuner is off, the plan is untouched")
    cmd, env, info, out = stage(stock=True)
    check(info is None and cmd == BASE_CMD, "--stock: the tuner is off")
    os.environ["MOE_CACHE_NO_AUTOTUNE"] = "1"
    check(stage()[2] is None, "MOE_CACHE_NO_AUTOTUNE=1 skips it (scripts, CI)")
    del os.environ["MOE_CACHE_NO_AUTOTUNE"]
    # a user flag wins even against a stored pick: tuned with ncmoe free, then the user passes --n-cpu-moe 20
    cmd, env, info, out = stage(argv=("--", "--n-cpu-moe", "20"))
    check(at.flag_value(cmd, "--n-cpu-moe") == "20", "your --n-cpu-moe after -- wins (its own key: measured without ncmoe)")
    check("ncmoe" in (info.get("locked") or {}), "the output lists what was left alone")
    calls.clear()
    cmd, env, info, out = stage(argv=("--", "--n-cpu-moe", "20", "-ub", "4096", "-ctk", "q8_0", "-np", "1"), runner=fake_runner)
    check(all("ncmoe-2" not in str(c) for c in calls) and all(at.flag_value(c[0], "--n-cpu-moe") == "20" and at.flag_value(c[0], "-ub") == "4096" for c in calls),
          "probes never touch a dimension you set")
    class Busy(M):
        def other_llama_servers(self): return [(1, 2**30)]
    cmd, env, info, out = stage(retune=True, machine=Busy(tmp))
    check(info["state"] == "skipped" and cmd == BASE_CMD, "another llama-server running: not measured, tried again next start")
    # too slow: stored, not retried, --retune retries
    def slow(cmd, env, left, n): return {"ok": False, "error": "the load did not finish in the time left"}
    for f in os.listdir(os.path.join(mc.CACHE_DIR, "autotune")):
        os.unlink(os.path.join(mc.CACHE_DIR, "autotune", f))
    cmd, env, info, out = stage(runner=slow)
    check(info["state"] == "too_slow" and cmd == BASE_CMD, "a model too slow to load in the budget: plan untouched")
    calls.clear()
    cmd, env, info, out = stage(runner=fake_runner)
    check(info["state"] == "too_slow" and not calls and any("not tried again" in l for l in out), "...and it is not retried at every start")
    cmd, env, info, out = stage(runner=fake_runner, retune=True)
    check(info["state"] == "tuned" and calls, "--retune tries it again")
finally:
    mc.CACHE_DIR, mc.port_busy = saved


# --- LAUNCHER-1042 / 1044 / TEST-1046: commit interplay, guard stop, model size ---
tmp2 = tempfile.mkdtemp(prefix="autotune-b-")
saved2 = (mc.CACHE_DIR, mc.port_busy)
mc.CACHE_DIR = os.path.join(tmp2, "cache"); mc.port_busy = lambda port: False
def stage2(runner, probe=True, machine=None, size=None, avail=None, fits=None, **k):
    a = types.SimpleNamespace(exact=False, stock=False, retune=False, agent=False, project=None, ub="auto", n_cpu_moe=None, hybrid=False, port=8080, autotune_probe=probe, **k)
    out = []
    r = mc.autotune_stage(a, ["--dry"], list(BASE_CMD), list(BASE_ENV), [], model, 65536, ls, machine or M(tmp), "plugin", 40, say_fn=lambda t, err=False: out.append(t),
                          runner=runner, size=size, avail=avail, fits=fits)
    return r + (out,)
def stored_files():
    d = os.path.join(mc.CACHE_DIR, "autotune")
    return os.listdir(d) if os.path.isdir(d) else []
try:
    # 1044: the model size
    seen_ = []
    def counting(cmd, env, left, n):
        seen_.append(cmd); return fake_runner(cmd, env, left, n)
    for size_, avail_, what in ((75.8e9, 90e9, "75 GB on a 32 GB computer (> 40 % of RAM)"), (5e9, 3e9, "bigger than the memory the server may use (spill)")):
        seen_.clear()
        cmd, env, info, out = stage2(counting, size=size_, avail=avail_)
        check(info["state"] == "skipped" and not seen_ and cmd == BASE_CMD and not stored_files() and any("skipped" in l and "budget" in l for l in out), f"{what}: no probe, one clear line, nothing stored")
    seen_.clear()
    cmd, env, info, out = stage2(counting, probe=False, size=75.8e9, avail=90e9)
    check(info["state"] == "skipped" and not out and "40 %" in info["lines"][0], "dry-run / plan-json / control page: the plan says the model will not be tuned (not 'the first start measures')")
    cmd, env, info, out = stage2(counting, size=5e9, avail=20e9)
    check(info["state"] == "tuned" and seen_, "a model of 5 GB on 32 GB RAM is tuned")
    for f in stored_files(): os.unlink(os.path.join(mc.CACHE_DIR, "autotune", f))
    # a stored pick is still applied for a big model (only the measuring is skipped)
    cmd, env, info, out = stage2(fake_runner, size=5e9, avail=20e9)
    seen_.clear()
    cmd, env, info, out = stage2(counting, size=75.8e9, avail=90e9)
    check(info["state"] == "cached" and not seen_, "a stored pick is still used when the size rule would skip measuring")
    for f in stored_files(): os.unlink(os.path.join(mc.CACHE_DIR, "autotune", f))
    # 1042: the commit check on every command that is loaded
    asked = []
    def fits_no_gpu_more(c_cmd, c_env):          # the candidate with 2 more expert layers on the GPU needs more commit than is free
        asked.append(at.flag_value(c_cmd, "--n-cpu-moe"))
        return (False, "free commit 4.1 GB after the load, under the 6 GB line") if at.flag_value(c_cmd, "--n-cpu-moe") == "18" else (True, "")
    seen_.clear()
    cmd, env, info, out = stage2(counting, fits=fits_no_gpu_more)
    check(asked and "18" in asked and all(at.flag_value(c, "--n-cpu-moe") != "18" for c in seen_), "a candidate the commit check refuses is never loaded")
    check(any("commit check" in l and "ncmoe-2" in l for l in out), "...and the line says which and why")
    check(info["state"] == "tuned" and at.flag_value(cmd, "--n-cpu-moe") != "18", "the pick is one of the candidates that passed")
    for f in stored_files(): os.unlink(os.path.join(mc.CACHE_DIR, "autotune", f))
    seen_.clear()
    cmd, env, info, out = stage2(counting, fits=lambda c, e: (False, "free commit 9 GB, the load needs 30 GB"))
    check(info["state"] == "skipped" and not seen_ and cmd == BASE_CMD and not stored_files(), "the plan as it is refused or shrunk by the commit check: nothing is loaded, nothing stored")
    # no commit check configured (Linux, --no-commit-check): everything fits
    seen_.clear()
    cmd, env, info, out = stage2(counting, fits=lambda c, e: (True, ""))
    check(info["state"] == "tuned" and len(seen_) >= 3, "a commit check that passes everything leaves the tune as before")
    for f in stored_files(): os.unlink(os.path.join(mc.CACHE_DIR, "autotune", f))
    # 1042: a guard stop aborts the whole tune and stores nothing
    runs_ = []
    def guard_runner(cmd, env, left, n):
        runs_.append(cmd)
        if len(runs_) == 2:
            return {"ok": False, "error": "llama-server stopped while loading (exit 1)", "guard_stop": "memory guard: free commit 1.2 GB, under the 2 GB line"}
        return fake_runner(cmd, env, left, n)
    cmd, env, info, out = stage2(guard_runner)
    check(len(runs_) == 2, f"after the first guard stop no further plan is loaded ({len(runs_)} loads)")
    check(info["state"] == "aborted" and info["picked"] == "C0" and cmd == BASE_CMD, "...the plan as it is runs")
    check(not stored_files(), "...and nothing is stored (the next start tries again)")
    check(any("memory guard" in l and "aborted" in l for l in out), "...and the launcher says so")
    c0calls = []
    ck_ = Clock()
    def guard_on_c0b(cand, left, n):
        ck_.t += 20
        if cand.id == "C0":
            c0calls.append(1)
            if len(c0calls) == 2:
                return {"ok": False, "guard_stop": "memory guard: free commit 1.0 GB"}
        return dict(T["C0"], decode_n=n)
    cands_ = at.build_candidates(BASE_CMD, BASE_ENV, {}, 65536, n_layers=40)
    r_ = at.tune(cands_[:2], guard_on_c0b, clock=ck_)
    check(r_["state"] == "aborted" and r_["rows"][-1]["id"] == "C0b" and r_["rows"][-1]["status"] == "stopped", "a guard stop on the second baseline run aborts the tune too")
    r_ = at.tune(cands, lambda c, left, n: {"ok": False, "guard_stop": "g"} if c.id == "C0" else dict(T["C0"], decode_n=n), clock=Clock())
    check(r_["state"] == "aborted" and len(r_["rows"]) == 1, "a guard stop on the first plan aborts too")
finally:
    mc.CACHE_DIR, mc.port_busy = saved2


# --- LAUNCHER-1045 (rest) / 1041 / TEST-1046 rest: failing stored pick, tuning lock, key coverage, progress, end state ---
# the lock
with tempfile.TemporaryDirectory() as lk:
    ok1, h1 = at.acquire_lock(lk)
    ok2, h2 = at.acquire_lock(lk)
    check(ok1 and not ok2 and h2 == os.getpid(), "tuning lock: the second launcher does not get it and learns who holds it")
    at.release_lock(lk)
    check(at.acquire_lock(lk)[0], "tuning lock: free again after release")
    at.release_lock(lk)
    # a lock of a dead process is taken over
    json.dump({"pid": 2 ** 22 + 12345, "t": time.time()}, open(at.lock_path(lk), "w"))
    check(at.acquire_lock(lk, alive=lambda pid: False)[0], "tuning lock: a dead holder's lock is taken over")
    at.release_lock(lk)
    json.dump({"pid": os.getpid(), "t": time.time() - at.LOCK_STALE_S - 5}, open(at.lock_path(lk), "w"))
    check(at.acquire_lock(lk)[0], "tuning lock: a lock older than the budget allows is taken over")
    at.release_lock(lk)
    json.dump({"pid": 1, "t": time.time()}, open(at.lock_path(lk), "w"))
    at.release_lock(lk)
    check(os.path.exists(at.lock_path(lk)), "release_lock leaves a lock that is not ours")

tmp3 = tempfile.mkdtemp(prefix="autotune-c-")
saved3 = (mc.CACHE_DIR, mc.port_busy)
mc.CACHE_DIR = os.path.join(tmp3, "cache"); mc.port_busy = lambda port: False
def stage3(runner, probe=True, retune=False, **k):
    a = types.SimpleNamespace(exact=False, stock=False, retune=retune, agent=False, project=None, ub="auto", n_cpu_moe=None, hybrid=False, port=8080, autotune_probe=probe, **k)
    out = []
    r = mc.autotune_stage(a, ["--dry"], list(BASE_CMD), list(BASE_ENV), [], model, 65536, ls, M(tmp), "plugin", 40, say_fn=lambda t, err=False: out.append(t), runner=runner)
    return r + (out,)
try:
    # the lock held by another launcher: nothing is measured
    at.acquire_lock(mc.CACHE_DIR)
    asked3 = []
    cmd, env, info, out = stage3(lambda c, e, l, n: asked3.append(c) or fake_runner(c, e, l, n))
    check(info["state"] == "skipped" and not asked3 and cmd == BASE_CMD and any("another start" in l for l in out), "concurrent tuners: the second start runs untuned and says why, no probe")
    check(not any(f.endswith(".json") for f in os.listdir(os.path.join(mc.CACHE_DIR, "autotune"))), "...and stores nothing")
    at.release_lock(mc.CACHE_DIR)
    cmd, env, info, out = stage3(fake_runner)
    check(info["state"] == "tuned" and not os.path.exists(at.lock_path(mc.CACHE_DIR)), "the lock is released after a tune")
    def boom(c, e, l, n): raise KeyboardInterrupt()
    for f in [f for f in os.listdir(os.path.join(mc.CACHE_DIR, "autotune")) if f.endswith(".json")]:
        os.unlink(os.path.join(mc.CACHE_DIR, "autotune", f))
    try:
        stage3(boom)
    except KeyboardInterrupt:
        pass
    check(not os.path.exists(at.lock_path(mc.CACHE_DIR)), "...and after an interrupt (Ctrl-C while measuring)")
    # the other start finished while this one waited: its pick is used, nothing measured twice
    for f in [f for f in os.listdir(os.path.join(mc.CACHE_DIR, "autotune")) if f.endswith(".json")]:
        os.unlink(os.path.join(mc.CACHE_DIR, "autotune", f))
    stage3(fake_runner)
    n_before = len(calls)
    cmd, env, info, out = stage3(fake_runner)
    check(info["state"] == "cached" and len(calls) == n_before, "a stored pick found: used, no second tuning")

    # progress lines for the control page
    for f in [f for f in os.listdir(os.path.join(mc.CACHE_DIR, "autotune")) if f.endswith(".json")]:
        os.unlink(os.path.join(mc.CACHE_DIR, "autotune", f))
    cmd, env, info, out = stage3(fake_runner)
    prog = [l for l in out if "trying plan" in l]
    check(len(prog) >= 3 and prog[0].startswith("moe-cache: auto-tune: trying plan 1 of ") and " of " in prog[1], f"progress lines 'trying plan N of M' ({prog[:2]})")
    import moe_cache_ui as ui
    check(not any(ui.is_warning(l) for l in prog), "...and the control page does not take them for warnings (it lists them under 'What the launcher decided')")

    # the key covers --cache-ram / --mtp / --fast / the KV type of the plan
    files0 = set(os.listdir(os.path.join(mc.CACHE_DIR, "autotune")))
    def keyed(cmd_, **k):
        a = types.SimpleNamespace(exact=False, stock=False, retune=False, agent=False, project=None, ub="auto", n_cpu_moe=None, hybrid=False, port=8080, autotune_probe=True, **k)
        mc.autotune_stage(a, ["--dry"], cmd_, list(BASE_ENV), [], model, 65536, ls, M(tmp), "plugin", 40, say_fn=lambda *x, **y: None, runner=fake_runner)
        return set(os.listdir(os.path.join(mc.CACHE_DIR, "autotune"))) - files0
    new0 = keyed(BASE_CMD + ["--cache-ram", "1536"])
    check(len(new0) == 0, "the planner's own --cache-ram (it follows the free commit on Windows) does not change the key")
    a_user = types.SimpleNamespace(exact=False, stock=False, retune=False, agent=False, project=None, ub="auto", n_cpu_moe=None, hybrid=False, port=8080, autotune_probe=True)
    before_n = len(os.listdir(os.path.join(mc.CACHE_DIR, "autotune")))
    mc.autotune_stage(a_user, ["--dry", "--cache-ram", "512"], list(BASE_CMD), list(BASE_ENV), [], model, 65536, ls, M(tmp), "plugin", 40, say_fn=lambda *x, **y: None, runner=fake_runner)
    check(len(os.listdir(os.path.join(mc.CACHE_DIR, "autotune"))) == before_n + 1, "a --cache-ram the user gave: another stored pick")
    new2 = keyed(BASE_CMD, mtp=True)
    check(len(new2) >= 1, "--mtp: another stored pick")
    new3 = keyed(BASE_CMD, fast="kv")
    check(len(new3) >= 1, "--fast: another stored pick")
    new4 = keyed([t if t != "q8_0" else "f16" for t in BASE_CMD])
    check(len(new4) >= 1, "another KV type in the plan: another stored pick")

    # a stored pick that is marked bad is not applied again
    for f in [f for f in os.listdir(os.path.join(mc.CACHE_DIR, "autotune")) if f.endswith(".json")]:
        os.unlink(os.path.join(mc.CACHE_DIR, "autotune", f))
    cmd, env, info, out = stage3(fake_runner)
    check(info["changed"] and at.flag_value(cmd, "--n-cpu-moe") == "18" and info.get("key_parts") and info.get("before") == (BASE_CMD, BASE_ENV), "a tuned start returns what the fall-back needs (key parts, the plan before the pick)")
    check(at.mark_bad(mc.CACHE_DIR, info["key_parts"], "llama-server stopped before it was ready (exit 1) with ncmoe-2"), "mark_bad writes into the stored file")
    n_before = len(calls)
    cmd, env, info, out = stage3(fake_runner)
    check(info["state"] == "bad" and cmd == BASE_CMD and not info["changed"] and len(calls) == n_before and any("not used again" in l and "--retune" in l for l in out),
          "a bad pick is not applied again, nothing is measured, the line names --retune")
    cmd, env, info, out = stage3(fake_runner, retune=True)
    check(info["state"] == "tuned" and info["changed"], "--retune replaces the bad pick")
finally:
    mc.CACHE_DIR, mc.port_busy = saved3

# the fall-back in run_child: a server that does not come up with the picked plan starts once with the plan as it is
fb = os.path.join(tmp, "fb-server.py")
open(fb, "w").write('''
import os, sys, time, json
from http.server import BaseHTTPRequestHandler, HTTPServer
mark, mode, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
n = int(open(mark).read()) if os.path.exists(mark) else 0
open(mark, "w").write(str(n + 1))
if mode == "dies_loading" or (mode == "dies_loading_once" and n == 0):
    sys.stderr.write("error: failed to load the model\\n"); sys.exit(1)
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
srv = HTTPServer(("127.0.0.1", port), H); srv.timeout = 0.2
t0 = time.time()
while time.time() - t0 < (3.5 if mode == "dies_late" else 2.5):
    srv.handle_request()
sys.exit(1 if mode == "dies_late" else 0)
''')
def free_port():
    import socket
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0)); return sk.getsockname()[1]
def run_fb(mode, tune=True, expect_plugin=False):
    mark = os.path.join(tmp, f"fb_{mode}_{tune}_{expect_plugin}"); port = free_port(); asked_ = []
    def tune_retry(rc, seen):
        asked_.append(rc); return [sys.executable, fb, mark, "ok", str(port)], dict(os.environ)
    with contextlib.redirect_stderr(io.StringIO()):
        rc = mc.run_child([sys.executable, fb, mark, mode, str(port)], dict(os.environ), port=port, expect_plugin=expect_plugin, allow_sleep=True, tune_retry=tune_retry if tune else None)
    return rc, len(asked_), int(open(mark).read())
rc, asked_n, starts = run_fb("dies_loading_once")
check(rc == 0 and asked_n == 1 and starts == 2, f"failing stored pick: the server dies loading, the plan as it is starts once and runs (rc {rc}, fall-back asked {asked_n}x, started {starts}x)")
rc, asked_n, starts = run_fb("dies_loading")
check(rc == 0 and asked_n == 1 and starts == 2, f"...the fall-back is asked once, not in a loop (rc {rc}, started {starts}x)")
rc, asked_n, starts = run_fb("dies_late")
check(rc == 1 and asked_n == 0 and starts == 1, f"a server that was ready and died later is NOT a bad pick (stock engine too): no fall-back (rc {rc}, asked {asked_n}x)")
import inspect
check("range(900)" not in inspect.getsource(mc.run_child_once).split("def check")[1].split("threading.Thread(target=check")[0], "the ready check has no 15-minute cap")
rc, asked_n, starts = run_fb("dies_loading_once", tune=False)
check(rc == 1 and starts == 1, "no pick applied (no tune_retry): the failure is reported as before")

# end to end through a launch plan: a stored pick, a server that does not load, the fall-back command and env, the bad mark, the end-state line
open(os.path.join(tmp, "libggml-moe-cache.so"), "w").close()
class Big(M):
    def mem_total(self): return 64 * 2 ** 30
saved4 = (mc.CACHE_DIR, mc.port_busy, mc.autotune_probe, mc.run_child, mc.WINDOWS)
mc.CACHE_DIR = os.path.join(tmp3, "cache2"); mc.port_busy = lambda port: False; mc.WINDOWS = False
ref = {"n": None}
def runner_rel(cmd_, env_, left_, n_):          # fewer CPU expert layers than the first (the plan as it is) is the fast one, whatever the planner chose
    nc = int(at.flag_value(cmd_, "--n-cpu-moe")); ref["n"] = nc if ref["n"] is None else ref["n"]
    fast = nc < ref["n"]
    return dict(ok=True, load_s=5, prompt_tps=400 if fast else 300, decode_tps=22 if fast else 20, decode_n=n_)
mc.autotune_probe = runner_rel
try:
    def make_plan():
        ref["n"] = None
        a, ex, argv = mc.parse_server([model, "--llama-server", ls, "--ram", "40", "--no-pin", "--always-plugin", "--ctx", "65536"])
        a.autotune_probe = True
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return mc.launch_plan(a, ex, argv, machine=Big(tmp), quiet=True)
    plan = make_plan()
    t = plan.autotune
    check(t and t["state"] == "tuned" and t["changed"] and "--n-cpu-moe" in plan.cmd, f"launch plan with a fresh tune: {t and t['state']} picked {t and t.get('picked')}")
    fbk = plan.tune_fallback() if plan.tune_fallback else None
    check(fbk and at.flag_value(fbk[0], "--n-cpu-moe") != at.flag_value(plan.cmd, "--n-cpu-moe") and at.flag_value(fbk[0], "--n-cpu-moe") == at.flag_value(t["before"][0], "--n-cpu-moe"),
          f"the fall-back command is the plan without the pick (n-cpu-moe {fbk and at.flag_value(fbk[0], '--n-cpu-moe')} instead of {at.flag_value(plan.cmd, '--n-cpu-moe')})")
    seen_run = {}
    def fake_run_child(cmd_, env_, **kw):
        seen_run["first"] = cmd_
        new = kw["tune_retry"](1, {"ready": False})
        seen_run["second"] = new
        return 0
    mc.run_child = fake_run_child
    plan2 = make_plan()
    plan2.a.dry_run = False
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            mc.run_launch_plan(plan2)
        except SystemExit:
            pass
    second = seen_run.get("second")
    check(second and at.flag_value(second[0], "--n-cpu-moe") == at.flag_value(t["before"][0], "--n-cpu-moe"), "run_launch_plan hands run_child a fall-back that starts the plan as it is")
    cmd_c, env_c, info_c, out_c = (None,) * 4
    a_c, ex_c, argv_c = mc.parse_server([model, "--llama-server", ls, "--ram", "40", "--no-pin", "--always-plugin", "--ctx", "65536", "--dry-run"])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        plan3 = mc.launch_plan(a_c, ex_c, argv_c, machine=Big(tmp), quiet=True)
    check(plan3.autotune["state"] == "bad" and f"--n-cpu-moe {at.flag_value(t['before'][0], '--n-cpu-moe')}" in " ".join(plan3.cmd), "after the fall-back the next start (dry-run too) runs the plan as it is and says the pick is bad")
    # the end-state line when the commit stage changes the picked plan
    for f in [f for f in os.listdir(os.path.join(mc.CACHE_DIR, "autotune")) if f.endswith(".json")]:
        os.unlink(os.path.join(mc.CACHE_DIR, "autotune", f))
    real_stage = mc.commit_stage
    # the probes' own commit check (commit_fits, quiet) passes every plan; the real stage after the pick moves the expert layers
    mc.commit_stage = lambda a_, cmd_, env_, argv_, model_, ctx_, ncmoe_, ub_, kv_, gpu_, machine_, mm_=0: (
        (list(cmd_), ncmoe_, ub_, kv_, None) if getattr(mc._TL, "quiet", 0) > 0 else (at.set_flag(list(cmd_), "--n-cpu-moe", "25"), "25", ub_, kv_, None))
    try:
        a, ex, argv = mc.parse_server([model, "--llama-server", ls, "--ram", "40", "--no-pin", "--always-plugin", "--ctx", "65536"])
        a.autotune_probe = True
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            plan4 = mc.launch_plan(a, ex, argv, machine=Big(tmp), quiet=False)
        check("the commit check changed the picked plan" in buf.getvalue() and plan4.autotune.get("final_differs"), "end state: a pick the commit check changes is said with the numbers that run")
    finally:
        mc.commit_stage = real_stage
finally:
    mc.CACHE_DIR, mc.port_busy, mc.autotune_probe, mc.run_child, mc.WINDOWS = saved4

# --- the real commit check (commit_fits over commit_stage) on a Windows fake machine ---
class WM(M):
    windows = True
    def __init__(self, tmp, limit, free): super().__init__(tmp); self.lim, self.fr = limit, free
    def win_commit(self): return (self.lim, self.fr)
    def win_memory(self): return (32e9, 20e9)
    def nvidia_reports(self): return False
a_w = types.SimpleNamespace(exact=False, stock=False, no_commit_check=False, ub="auto", n_cpu_moe=None, hybrid=False, vram=None)
mc.WINDOWS, wsave = True, mc.WINDOWS
try:
    wcmd = [ls, "-m", model, "--port", "8080", "-c", "65536", "-ub", "512", "-b", "2048", "--n-cpu-moe", "20", "--cache-ram", "2048", "-ngl", "99"]
    ok_big, _w = mc.commit_fits(a_w, wcmd, [], ["m"], model, 65536, "20", "512", False, None, WM(tmp, 80e9, 70e9))
    check(ok_big, f"commit_fits: plenty of free commit -> fits ({_w[:60]})")
    ok_small, why_ = mc.commit_fits(a_w, wcmd, [], ["m"], model, 65536, "20", "512", False, None, WM(tmp, 80e9, 3e9))
    check(not ok_small and why_, f"commit_fits: 3 GB free -> does not fit, with the reason ({why_[:70]})")
    n0 = len(getattr(mc._TL, "notes", ()))
    mc.commit_fits(a_w, wcmd, [], ["m"], model, 65536, "20", "512", False, None, WM(tmp, 80e9, 3e9))
    check(len(getattr(mc._TL, "notes", ())) == n0, "commit_fits says nothing and leaves the plan's notes alone")
    a_w.no_commit_check = True
    ok_nc, _w = mc.commit_fits(a_w, wcmd, [], ["m"], model, 65536, "20", "512", False, None, WM(tmp, 80e9, 3e9))
    check(ok_nc, "--no-commit-check: everything fits (the user's choice)")
finally:
    mc.WINDOWS = wsave

# --- the real probe reports a guard stop ---
dead = os.path.join(tmp, "dead-llama-server.py")
open(dead, "w").write("import sys, time\ntime.sleep(0.3)\nsys.exit(1)\n")
orig_guard = mc.start_commit_guard
mc.start_commit_guard = lambda p, env, *a, **k: types.SimpleNamespace(stopped="memory guard: free commit 1.0 GB", min_free=1e9, soft=6e9, label="commit")
try:
    r = mc.autotune_probe([sys.executable, dead, "--port", "1"], dict(os.environ), 20, 512)
    check(not r["ok"] and "memory guard" in r.get("guard_stop", ""), f"the real probe carries the guard's stop text ({r.get('guard_stop')!r})")
finally:
    mc.start_commit_guard = orig_guard
r = mc.autotune_probe([sys.executable, dead, "--port", "1"], dict(os.environ), 20, 512)
check(not r["ok"] and not r.get("guard_stop"), "a plain crash is a failed plan, not a guard stop")

# --- the real probe runner against a fake llama-server ---
fake = os.path.join(tmp, "fake-llama-server.py")
open(fake, "w").write('''#!/usr/bin/env python3
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv; port = int(a[a.index("--port") + 1]); ub = a[a.index("-ub") + 1] if "-ub" in a else "0"
time.sleep(0.5)
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send(200, {"status": "ok"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); j = json.loads(self.rfile.read(n))
        assert j.get("cache_prompt") is False
        if j["n_predict"] == 1: self._send(200, {"timings": {"prompt_per_second": 350.0 if ub == "2048" else 250.0, "prompt_n": 1500, "predicted_per_second": 0, "predicted_n": 1}})
        else: self._send(200, {"timings": {"prompt_per_second": 0, "predicted_per_second": 19.5, "predicted_n": j["n_predict"]}})
HTTPServer(("127.0.0.1", port), H).serve_forever()
''')
os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)
env = dict(os.environ)
r = mc.autotune_probe([sys.executable, fake, "-ub", "2048", "--port", "1", "--host", "0.0.0.0"], env, 30, 512)
check(r.get("ok") and r["prompt_tps"] == 350.0 and r["decode_tps"] == 19.5 and r["decode_n"] == 512, f"the real probe: prompt + 512-token decode through HTTP ({ {k: r.get(k) for k in ('ok', 'prompt_tps', 'decode_tps', 'decode_n', 'error')} })")
check(r["load_s"] >= 0.4, "the load time is the time until healthy")
r = mc.autotune_probe([sys.executable, fake, "-ub", "2048", "--port", "1"], env, 0.2, 512)
check(not r["ok"] and "time" in r["error"], "a load that outlasts the budget fails cleanly")
r = mc.autotune_probe(["/nonexistent/llama-server"], env, 5, 512)
check(not r["ok"] and "could not start" in r["error"], "a missing binary fails cleanly")
import shutil, subprocess
if shutil.which("pgrep"):
    time.sleep(1)
    left = subprocess.run(["pgrep", "-f", "[f]ake-llama-server.py"], capture_output=True, text=True).stdout.split()
    check(not left, "no probe server is left running")

print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
