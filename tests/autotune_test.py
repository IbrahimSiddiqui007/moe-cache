#!/usr/bin/env python3
"""LAUNCHER-994: measured auto-tune (bin/moe_cache_autotune.py + autotune_stage / autotune_probe in bin/moe_cache.py). Fake timings and a fake llama-server (a small
HTTP script); no model weights, no real server.
   python3 tests/autotune_test.py"""
import json, os, stat, sys, tempfile, time, types

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
c = at.build_candidates(BASE_CMD, BASE_ENV, {}, 8192, n_layers=40)
check(not any(x.id.startswith("kv") for x in c), "no KV candidate below 32k context")

# --- apply_edits ---
cand = {x.id: x for x in at.build_candidates(BASE_CMD, BASE_ENV, {}, 65536, n_layers=40)}
cmd, env, sk = at.apply_edits(BASE_CMD, BASE_ENV, cand["ncmoe-2"].edits)
check(at.flag_value(cmd, "--n-cpu-moe") == "18" and not sk, "ncmoe-2 is a delta on the current plan (20 -> 18)")
cmd, env, sk = at.apply_edits(BASE_CMD[:-4] + ["--n-cpu-moe", "12"], BASE_ENV, cand["ncmoe-2"].edits)
check(at.flag_value(cmd, "--n-cpu-moe") == "10", "the same delta on a plan of 12 gives 10 (a changed GPU memory later cannot carry 18 over)")
cmd, env, sk = at.apply_edits(BASE_CMD, BASE_ENV, cand["ncmoe-2"].edits, locked={"ncmoe": "yours"})
check(cmd == BASE_CMD and sk == ["ncmoe"], "a locked dimension is skipped and named")
cmd, env, sk = at.apply_edits(BASE_CMD, BASE_ENV, cand["ub2048"].edits)
check(cmd.count("-ub") == 1 and at.flag_value(cmd, "-ub") == "2048", "ub edit replaces, never duplicates")
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

# --- cache file ---
with tempfile.TemporaryDirectory() as tmp:
    parts = at.key_parts("m.gguf:1:2", 65536, {"cpu": "x"}, {}, False, "plugin")
    check(at.load(tmp, parts) is None, "no file: no stored pick")
    res = run_t(T)
    p = at.save(tmp, parts, res)
    check(os.path.isfile(p) and os.path.dirname(p).endswith("autotune"), "stored in <cache>/autotune/<key>.json")
    got = at.load(tmp, parts)
    check(got and got["picked"] == ids[1] and at.pick_edits(got) == cand[ids[1]].edits if False else got and got["picked"] == ids[1], "stored pick reads back")
    check(at.load(tmp, at.key_parts("m.gguf:1:2", 131072, {"cpu": "x"}, {}, False, "plugin")) is None, "another context: another key")
    check(at.load(tmp, at.key_parts("m.gguf:1:2", 65536, {"cpu": "y"}, {}, False, "plugin")) is None, "another machine: another key")
    check(at.load(tmp, at.key_parts("m.gguf:1:3", 65536, {"cpu": "x"}, {}, False, "plugin")) is None, "another model file: another key")
    check(at.load(tmp, at.key_parts("m.gguf:1:2", 65536, {"cpu": "x"}, {"ub": "yours"}, False, "plugin")) is None, "other user flags: another key")
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
