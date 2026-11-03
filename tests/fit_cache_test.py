#!/usr/bin/env python3
"""PREFILL-180 / PREFILL-181: llama-fit-params figures are remembered on disk (model, context, -ub, KV type, GPU, tool) and the missing -ub candidates are
measured at the same time. No model, no GPU, no tool: subprocess.run is replaced by a fake that counts calls and takes 0.3 s.
   python3 tests/fit_cache_test.py"""
import json, os, sys, tempfile, threading, time, types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
import gguf_plan as gp  # noqa: E402

TMP = tempfile.mkdtemp(prefix="moe-fit-test-")
os.environ["MOE_CACHE_FIT_CACHE"] = os.path.join(TMP, "cache", "fit.json")
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

tool = os.path.join(TMP, "llama-fit-params"); open(tool, "w").write("t")
model = os.path.join(TMP, "m.gguf"); open(model, "wb").write(b"GGUF" * 8)
gp.gpu_identity = lambda: "gpu-A"
calls, lock, active, peak = [], threading.Lock(), [0], [0]
def fake_run(cmd, **kw):
    with lock:
        calls.append(cmd); active[0] += 1; peak[0] = max(peak[0], active[0])
    time.sleep(0.3)
    with lock:
        active[0] -= 1
    ub = int(cmd[cmd.index("-ub") + 1])
    return types.SimpleNamespace(stdout=f"CUDA0 3664 1840 {1000 + ub // 2} \nHost 68651 0 222 \n", returncode=0)
gp.subprocess.run = fake_run

def fresh():                    # a new process: nothing in memory
    gp._FIT_CACHE.clear()

# 1. one measurement is written to disk and read by the next "process"
v = gp.fit_params_buffer_mib(tool, model, 65536, 1024)
fresh()
n = len(calls)
v2 = gp.fit_params_buffer_mib(tool, model, 65536, 1024)
check(v == v2 == 1512.0 and len(calls) == n, "the figure is read from the disk cache by the next process; the tool does not run again")
check(len(json.load(open(os.environ["MOE_CACHE_FIT_CACHE"]))) == 1, "one entry on disk")

# 2. any part of the key changes: measured again
for what, kw in (("context", dict(ctx=32768)), ("-ub", dict(ub=2048)), ("KV type", dict(kv_q8=True))):
    fresh(); n = len(calls)
    args = dict(ctx=65536, ub=1024, kv_q8=False); args.update(kw)
    gp.fit_params_buffer_mib(tool, model, args["ctx"], args["ub"], args["kv_q8"])
    check(len(calls) == n + 1, f"a different {what} is measured again")
fresh(); gp.gpu_identity = lambda: "gpu-B"; n = len(calls)
gp.fit_params_buffer_mib(tool, model, 65536, 1024)
check(len(calls) == n + 1, "another GPU is measured again")
gp.gpu_identity = lambda: "gpu-A"
open(model, "ab").write(b"more")
fresh(); n = len(calls)
gp.fit_params_buffer_mib(tool, model, 65536, 1024)
check(len(calls) == n + 1, "a changed model file is measured again")
os.utime(tool, ns=(time.time_ns(), time.time_ns() + 7_000_000_000)); fresh(); n = len(calls)
gp.fit_params_buffer_mib(tool, model, 65536, 1024)
check(len(calls) == n + 1, "a changed tool is measured again")

# 3. a failure and an unknown GPU are not written
gp.subprocess.run = lambda cmd, **kw: types.SimpleNamespace(stdout="no figures", returncode=1)
fresh(); before = json.load(open(os.environ["MOE_CACHE_FIT_CACHE"]))
check(gp.fit_params_buffer_mib(tool, model, 65536, 6144) is None and json.load(open(os.environ["MOE_CACHE_FIT_CACHE"])) == before, "a failed run is not written to disk")
gp.subprocess.run = fake_run
gp.gpu_identity = lambda: ""; fresh(); n = len(calls)
gp.fit_params_buffer_mib(tool, model, 65536, 512); fresh(); gp.fit_params_buffer_mib(tool, model, 65536, 512)
check(len(calls) == n + 2, "without a GPU identity nothing is cached on disk")
gp.gpu_identity = lambda: "gpu-A"

# 4. a damaged cache file is ignored
open(os.environ["MOE_CACHE_FIT_CACHE"], "w").write("{nope"); fresh(); n = len(calls)
check(gp.fit_params_buffer_mib(tool, model, 65536, 1024) == 1512.0 and len(calls) == n + 1, "a damaged cache file is ignored")

# 5. PREFILL-181: the missing candidates run at the same time, with the same figures as one by one
os.remove(os.environ["MOE_CACHE_FIT_CACHE"]); fresh(); calls.clear(); peak[0] = 0
t0 = time.time()
started = gp.prefetch_fit_params(tool, model, 65536, (512, 1024, 2048, 3072), False)
dt = time.time() - t0
check(started == 4 and peak[0] == 4 and dt < 0.9, f"4 candidates measured at the same time: {dt:.2f} s instead of about 1.2 s (peak {peak[0]} at once)")
t0 = time.time()
vals = [gp.fit_params_buffer_mib(tool, model, 65536, u) for u in (512, 1024, 2048, 3072)]
check(vals == [1256.0, 1512.0, 2024.0, 2536.0] and len(calls) == 4 and time.time() - t0 < 0.2, f"the prefetched figures are the same as one by one ({vals}) and need no more runs")
check(gp.prefetch_fit_params(tool, model, 65536, (512, 1024), False) == 0, "nothing left to do: no runs")
check(gp.prefetch_fit_params(None, model, 65536, (512, 1024), False) == 0, "no tool: nothing started")
sys.exit(1 if fails else 0)
