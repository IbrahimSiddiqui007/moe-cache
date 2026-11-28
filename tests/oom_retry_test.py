#!/usr/bin/env python3
"""GPU out-of-memory at load (desktop PC 2026-10-30: planned with 13.4 GiB free, another program took some, llama-server died with
ErrorOutOfDeviceMemory): run_child starts the server once more with the command oom_retry() returns. Fake servers, no model.
   python3 tests/oom_retry_test.py"""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bin"))
import moe_cache as mc  # noqa: E402

TMP = tempfile.mkdtemp(prefix="oom-test-")
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

for line in ("ggml_vulkan: Device memory allocation of size 940714752 failed.", "ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory",
             "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 940.00 MiB on device 0: cudaMalloc failed: out of memory",
             "CUDA error: out of memory", "llama_model_load: error loading model: unable to allocate Vulkan0 buffer"):
    check(bool(mc.GPU_OOM_RE.search(line)), "recognised: " + line[:60])
for line in ("llama_model_load: error loading model: unable to allocate CPU buffer", "srv  log: model loaded", "error: invalid argument"):
    check(not mc.GPU_OOM_RE.search(line), "not a GPU out-of-memory: " + line[:60])

fake = os.path.join(TMP, "fake.py")
open(fake, "w").write('''
import os, sys
mark = sys.argv[1]; mode = sys.argv[2]
n = int(open(mark).read()) if os.path.exists(mark) else 0
open(mark, "w").write(str(n + 1))
if mode == "oom_once" and n == 0:
    sys.stderr.write("ggml_vulkan: Device memory allocation of size 940714752 failed.\\nfailed to load model\\n"); sys.exit(1)
if mode == "oom_always":
    sys.stderr.write("ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory\\n"); sys.exit(1)
if mode == "kv_abort_once" and n == 0:       # llama.cpp b12064 + Granite 1B + q8_0 K and V: the scheduler's hash set is full in the --fit probe, at any context
    sys.stderr.write("/home/runner/work/llama.cpp/llama.cpp/ggml/src/ggml-impl.h:330: fatal error\\n"); os.abort()
if mode == "kv_abort_always":
    sys.stderr.write("/home/runner/work/llama.cpp/llama.cpp/ggml/src/ggml-impl.h:330: fatal error\\n"); sys.exit(1)
if mode == "other":
    sys.stderr.write("error: something else\\n"); sys.exit(1)
sys.stderr.write("moe-cache: fake ok, extra=%s\\n" % sys.argv[3:]); sys.exit(0)
''')
def run(mode, retry=True):
    mark = os.path.join(TMP, f"m_{mode}_{retry}")
    calls = []
    def oom_retry():
        calls.append(1)
        return [sys.executable, fake, mark, mode, "retried"]
    rc = mc.run_child([sys.executable, fake, mark, mode], dict(os.environ), oom_retry=oom_retry if retry else None)
    return rc, len(calls), int(open(mark).read())

rc, calls, starts = run("oom_once")
check(rc == 0 and calls == 1 and starts == 2, f"out of memory once: retried once and succeeded (rc {rc}, retry asked {calls}x, started {starts}x)")
rc, calls, starts = run("oom_always")
check(rc == 1 and calls == 1 and starts == 2, f"out of memory twice: one retry only (rc {rc}, started {starts}x)")
rc, calls, starts = run("other")
check(rc == 1 and calls == 0 and starts == 1, f"another failure: no retry (rc {rc}, started {starts}x)")
rc, calls, starts = run("oom_once", retry=False)
check(rc == 1 and starts == 1, f"no oom_retry given (user fixed the plan): fails as before (rc {rc})")

# the llama.cpp abort in the scheduler with a q8_0 KV cache the launcher chose (found with the 131072 default on Granite 1B, 2026-11-07): one retry with the f16 KV cache
check(bool(mc.GGML_ABORT_RE.search("D:/a/llama.cpp/llama.cpp/ggml/src\\ggml-impl.h:330: fatal error")) and bool(mc.GGML_ABORT_RE.search("/x/ggml/src/ggml-impl.h:301: fatal error")),
      "recognised: the ggml-impl.h fatal error line (Windows and Linux paths)")
check(not mc.GGML_ABORT_RE.search("ggml.c:1234: GGML_ASSERT(x) failed") and not mc.GGML_ABORT_RE.search("srv: fatal error in the web ui"), "not recognised: other asserts and unrelated text")
# the command of the f16 retry (the launcher's own kv_retry closure uses kv_retry_cmd; tests adopted from the outside review of 7c3d024): every quantized type of the KV ladder is dropped, not only q8_0
base = ["llama-server", "-m", "m.gguf", "-c", "65536"]
new, was = mc.kv_retry_cmd(base + ["-ctk", "q8_0", "-ctv", "q8_0", "--port", "8080"])
check(new == base + ["--port", "8080"] and was == "K q8_0 / V q8_0", f"f16 retry: q8_0 / q8_0 is dropped ({was})")
new, was = mc.kv_retry_cmd(base + ["-ctk", "q8_0", "-ctv", "q4_0"])
check(new == base and was == "K q8_0 / V q4_0", f"f16 retry: K q8_0 / V q4_0 drops both (V q4_0 used to stay: a f16 / q4_0 retry announced as f16) ({was})")
new, was = mc.kv_retry_cmd(base + ["-ctk", "q4_0", "-ctv", "q4_0"])
check(new == base and was == "K q4_0 / V q4_0", f"f16 retry: q4_0 / q4_0 is dropped (there used to be no retry at all) ({was})")
new, was = mc.kv_retry_cmd(base + ["--cache-type-k", "q8_0", "--cache-type-v", "q4_0"])
check(new == base and was == "K q8_0 / V q4_0", "f16 retry: the long option names count too")
new, was = mc.kv_retry_cmd(base + ["-ctk", "f16", "-ctv", "f16"])
check(new == base + ["-ctk", "f16", "-ctv", "f16"] and was == "", "f16 retry: an f16 cache gives nothing to drop (no retry)")
new, was = mc.kv_retry_cmd(base)
check(new == base and was == "", "f16 retry: no cache type in the command gives nothing to drop")
def run_kv(mode, retry=True):
    mark = os.path.join(TMP, f"kv_{mode}_{retry}")
    calls = []
    def kv_retry():
        calls.append(1)
        return [sys.executable, fake, mark, "kv_abort_always" if mode == "kv_abort_always" else "ok", "f16-kv"]
    rc = mc.run_child([sys.executable, fake, mark, mode], dict(os.environ), kv_retry=kv_retry if retry else None)
    return rc, len(calls), int(open(mark).read())
rc, calls, starts = run_kv("kv_abort_once")
check(rc == 0 and calls == 1 and starts == 2, f"llama.cpp abort with the launcher's q8_0 KV: retried once with f16 and succeeded (rc {rc}, asked {calls}x, started {starts}x)")
_said = []
_say0 = mc.say
mc.say = lambda text, *a_, **k_: (_said.append(text), _say0(text, *a_, **k_))[1]
os.remove(os.path.join(TMP, "kv_kv_abort_once_True"))          # the fake counts its starts in this file: start over so it aborts once more
run_kv("kv_abort_once")
mc.say = _say0
check(any("abort line that triggered the f16 KV retry" in x and "ggml-impl.h:330: fatal error" in x for x in _said), "the f16 KV retry logs which abort line matched")
rc, calls, starts = run_kv("kv_abort_always")
check(rc != 0 and calls == 1 and starts == 2, f"the abort twice: one retry only (rc {rc}, started {starts}x)")
rc, calls, starts = run_kv("kv_abort_once", retry=False)
check(rc != 0 and starts == 1, f"no kv_retry (your own -ctk/-ctv): fails as before, no retry (rc {rc})")
rc, calls, starts = run_kv("other")
check(rc == 1 and calls == 0 and starts == 1, f"another failure: no KV retry (rc {rc}, started {starts}x)")

# LAUNCHER-1029: the commit guard stopped llama-server (TerminateProcess = exit code 1 on Windows, not "quiet") and the log has an out-of-memory line: NO second load.
# run_child_once is replaced by a stub (the same on Windows and Linux): the first load ends with rc 1, an out-of-memory line seen, not ready.
class _Guard:
    def __init__(self, stopped): self.stopped, self.min_free, self.soft, self.label = stopped, 1.8e9, 6e9, "free commit"
def run_guard(stopped, kind="oom"):
    starts, asked = [], []
    def once(cmd, env, port=None, expect_plugin=False, mem_limit=None, allow_sleep=False, final=False):
        starts.append(cmd)
        seen = {"plugin": False, "disabled": "", "last": "", "ready": False, "oom": kind == "oom", "kvabort": "ggml-impl.h:330: fatal error" if kind == "kv" else "",
                "guard": _Guard(stopped)}
        return (1 if len(starts) == 1 else 0), seen
    saved = mc.run_child_once
    mc.run_child_once = once
    try:
        rc = mc._run_child(["srv"], {}, None, False, None, False,
                           lambda: (asked.append("oom"), ["srv", "again"])[1], lambda: (asked.append("kv"), ["srv", "kv"])[1])
    finally:
        mc.run_child_once = saved
    return rc, asked, len(starts)
rc, asked, n = run_guard("moe-cache: WARNING: free commit fell to 1.8 GB")
check(rc == 1 and asked == [] and n == 1, f"the commit guard stopped the server (exit 1, out-of-memory line in the log): no retry, one start (rc {rc}, retries asked {asked}, started {n}x)")
rc, asked, n = run_guard("moe-cache: WARNING: free commit fell to 1.8 GB", kind="kv")
check(rc == 1 and asked == [] and n == 1, f"...and no f16 KV retry either (rc {rc}, retries asked {asked}, started {n}x)")
rc, asked, n = run_guard(None)
check(rc == 0 and asked == ["oom"] and n == 2, f"control: the same failure with no guard stop is retried once, as before (rc {rc}, asked {asked}, started {n}x)")
rc, asked, n = run_guard(None, kind="kv")
check(rc == 0 and asked == ["kv"] and n == 2, f"control: the KV abort with no guard stop is retried once, as before (rc {rc}, asked {asked}, started {n}x)")

# LAUNCHER-1200: the cache the launcher chose does not fit at load -> one retry without it (static plan); never for your own --moe-cache-mib or --exact
check(bool(mc.GPU_OOM_RE.search("llama_init_from_model: failed to initialize the context: failed to allocate buffer for MoE cache")), "recognised: the cache buffer failure line")
base = ["llama-server", "-m", "m.gguf", "-ngl", "99", "--n-cpu-moe", "999", "-c", "65536", "--load-mode", "none", "-ub", "2048", "-b", "2048", "--moe-cache-mib", "6500", "--cache-ram", "2048"]
new = mc.cache_fallback_cmd(base, 30, 2048)
check(new is not None and "--moe-cache-mib" not in new and "6500" not in new and new[new.index("--n-cpu-moe") + 1] == "30" and "--cache-ram" in new and new[new.index("-ub") + 1] == "2048",
      "fallback command: cache flag and its value gone, static --n-cpu-moe 30, the rest kept")
new = mc.cache_fallback_cmd(base, 33, 1024)
check(new[new.index("-ub") + 1] == "1024" and new[new.index("-b") + 1] == "2048", "fallback command: a smaller -ub when the static plan needs it, -b stays 2048")
check(mc.cache_fallback_cmd([c for c in base if c not in ("--moe-cache-mib", "6500")], 30, 2048) is None, "no cache flag in the command: nothing to fall back from")
# end to end with a fake server: first start dies with the cache buffer failure, the retry (what oom_retry returns) starts without the cache
fake_cache = os.path.join(TMP, "fake_cache.py")
open(fake_cache, "w").write('''
import os, sys
mark = sys.argv[1]
n = int(open(mark).read()) if os.path.exists(mark) else 0
open(mark, "w").write(str(n + 1))
if "--moe-cache-mib" in sys.argv:
    sys.stderr.write("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 6500.00 MiB on device 0: cudaMalloc failed: out of memory\\nllama_init_from_model: failed to initialize the context: failed to allocate buffer for MoE cache\\n"); sys.exit(1)
sys.stderr.write("fake ok without cache\\n"); sys.exit(0)
''')
def run_cache(launcher_choice, exact=False, engine="stock"):
    """The decision is the launcher's real cache_oom_decision + cache_fallback_cmd (the same functions oom_retry calls), the child is a fake server."""
    mark = os.path.join(TMP, f"mc_{launcher_choice}_{exact}_{engine}")
    start = [sys.executable, fake_cache, mark, "--n-cpu-moe", "999", "-ub", "2048", "--moe-cache-mib", "6500"]
    asked = []
    def oom_retry():
        asked.append(1)
        d = mc.cache_oom_decision(start, launcher_choice, exact, engine)
        return mc.cache_fallback_cmd(start, 30, 2048) if d == "fallback" else None
    rc = mc.run_child(start, dict(os.environ), oom_retry=oom_retry)
    return rc, len(asked), int(open(mark).read())
rc, asked, n = run_cache(True)
check(rc == 0 and asked == 1 and n == 2, f"the launcher's cache fails at load: one retry without it starts (rc {rc}, asked {asked}, started {n}x)")
rc, asked, n = run_cache(False)
check(rc != 0 and n == 1, f"your own --moe-cache-mib after --: no retry at all (rc {rc}, started {n}x)")
rc, asked, n = run_cache(True, exact=True)
check(rc != 0 and n == 1, f"--gpu-cache N with --exact: no retry at all (rc {rc}, started {n}x)")
rc, asked, n = run_cache(True, engine="plugin")
check(rc != 0 and n == 1, f"the plugin engine with a cache: no cache fallback (rc {rc}, started {n}x)")
check(mc.cache_oom_decision(["x", "--n-cpu-moe", "20"], False, False, "stock") is None, "no cache in the command: the older retries apply (decision None)")
# remembered failure: the next plan is smaller or has no cache
check(mc.shrink_known_fail(3000, 0) == 3000 and mc.shrink_known_fail(2000, 3000) == 2000 and mc.shrink_known_fail(3000, 3000) == 2496 and mc.shrink_known_fail(1700, 1700) == 0,
      "shrink_known_fail: unchanged below the failed size, 15 % under it from there, none under 1536 MiB")
sys.path.insert(0, os.path.join(ROOT, "planner"))
import load_record as lr  # noqa: E402
cd = os.path.join(TMP, "lrcache"); kvf = ("f16", "f16")
LK = lambda free=7600, ctx=65536, b="b1", kv=kvf: lr.lookup_cache_fail(cd, "gpu", b, "m1", ctx, kv, "stock", free_now_mib=free)
check(LK() is None, "no failure recorded: None")
check(lr.save_cache_fail(cd, "gpu", "b1", "m1", 65536, kvf, "stock", 3000, 7600) and LK()["fail_mib"] == 3000 and LK()["fails"] == 1, "a failed cache of 3000 MiB is stored and found (1 failure)")
check(LK(8400) is None, "more free GPU memory now (+800 MiB): the failure is not trusted")
check(LK(None) is None, "free memory now unknown: not trusted (the free-memory check is not skipped)")
check(LK(ctx=32768) is None and LK(b="b2") is None and LK(kv=("q8_0", "q8_0")) is None, "another context, build or KV type: not found")
lr.save_cache_fail(cd, "gpu", "b1", "m1", 65536, kvf, "stock", 2500, 7600)
check(LK()["fail_mib"] == 2500 and LK()["fails"] == 2, "a later, smaller failure replaces it; two failures in a row are counted")
# plan_known_cache: the real decision
S = lambda fail, fails=1, ok=None: {"fail_mib": fail, "fails": fails, "ok_mib": ok}
check(mc.plan_known_cache(6500, S(6500)) == 5504 and mc.plan_known_cache(6500, S(6500), 3300) == 3264, "plan: 15 % under the failed size, never over the tuner's own size from today's memory")
check(mc.plan_known_cache(6500, S(6500, 2)) == 0, "plan: after two failures in a row, no cache")
check(mc.plan_known_cache(6500, S(6500, 1, 3200)) == 3200, "plan: a size that loaded before is used directly")
check(mc.plan_known_cache(3000, S(6500)) == 3000 and mc.plan_known_cache(3000, None) == 3000, "plan: a size under the failed one, or no record: unchanged")
# the size that loaded: kept under the failing size, record cleared at or over it
lr.save_cache_fail(cd, "gpu", "b1", "m1", 40000, kvf, "stock", 6500, 7600)
lr.save_cache_ok(cd, "gpu", "b1", "m1", 40000, kvf, "stock", 3200)
check(LK(ctx=40000)["ok_mib"] == 3200 and LK(ctx=40000)["fail_mib"] == 6500, "a size that loaded fine (under the failing one) is recorded next to the failure")
lr.save_cache_ok(cd, "gpu", "b1", "m1", 40000, kvf, "stock", 6500)
check(LK(ctx=40000) is None, "a load with the cache at or over the failing size clears the record")
check(not lr.save_cache_ok(cd, "gpu", "b9", "m1", 65536, kvf, "stock", 3000), "no failure recorded: a good load writes nothing")
# apply_cache_record on a final command, with a real record and a stub machine
class M:
    def gpu_identity(self): return "gpu"
class A: pass
a = A(); a.vram = None; a.vram_user = None; a.plan_free_mib = 7600
mc.load_record_on = lambda: True
mc.CACHE_DIR = os.path.join(TMP, "apply"); mc.LAUNCHER_CACHE[0] = True
mc_lr = lr; mc_lr.build_id = lambda ls: "bX"; mc_lr.model_id = lambda m: "mX"
base = ["llama-server", "-c", "65536", "--n-cpu-moe", "999", "-ub", "2048", "-b", "2048", "--moe-cache-mib", "6500", "-ctk", "q8_0", "-ctv", "q8_0"]
mc_lr.save_cache_fail(mc.CACHE_DIR, "gpu", "bX", "mX", 65536, mc.kv_key(base), "stock", 6500, 7600)
tuned = {"gpu_cache_mib": 3300, "static_n_cpu_moe": 34}
c = list(base); r = mc.apply_cache_record(a, c, M(), "ls", "model", 65536, tuned)
check(r == 3264 and c[c.index("--moe-cache-mib") + 1] == "3264", f"apply on the final command (q8_0 KV key): 6500 planned as the tuner's 3264 (got {r})")
c = [x if x != "q8_0" else "f16" for x in base]; r = mc.apply_cache_record(a, c, M(), "ls", "model", 65536, tuned)
check(r is None and "6500" in c, "a command with another KV type has another key: record not applied")
mc_lr.save_cache_fail(mc.CACHE_DIR, "gpu", "bX", "mX", 65536, mc.kv_key(base), "stock", 5000, 7600)
c = list(base); r = mc.apply_cache_record(a, c, M(), "ls", "model", 65536, tuned)
check(r == 0 and "--moe-cache-mib" not in c and c[c.index("--n-cpu-moe") + 1] == "34" and mc.LAUNCHER_CACHE[0] is False, "after two failures: no cache, the static layers (34), cache flag off")
mc.LAUNCHER_CACHE[0] = False
c = list(base); r = mc.apply_cache_record(a, c, M(), "ls", "model", 65536, tuned)
check(r is None and "6500" in c, "your own cache flag (not the launcher's): never edited")
# fail, ok, fail: the count restarts at 1 (a good load reset it to 0, which must not read as "no count")
lr.save_cache_fail(cd, "gpu", "b1", "m1", 30000, kvf, "stock", 6500, 7600)
lr.save_cache_ok(cd, "gpu", "b1", "m1", 30000, kvf, "stock", 3200)
lr.save_cache_fail(cd, "gpu", "b1", "m1", 30000, kvf, "stock", 6500, 7600)
check(LK(ctx=30000)["fails"] == 1 and mc.plan_known_cache(6500, LK(ctx=30000)) == 3200, f"fail, ok, fail: fails == 1, not 2 (got {LK(ctx=30000)['fails']}); the good size is still used")
# the auto-tune's retry (tune_retry) then the same failure: the other retries still apply to the plan-as-is load, except the KV retry on a cache command
def tune_case(extra_tuned, mode="oom_always"):
    mark = os.path.join(TMP, "m_tune_" + mode + str(len(extra_tuned))); asked = []; kvasked = []
    if os.path.exists(mark): os.remove(mark)
    def tune_retry(rc, seen):
        return ([sys.executable, fake, mark, mode, "tuned"] + extra_tuned, dict(os.environ))
    rc = mc.run_child([sys.executable, fake, mark, "oom_always"], dict(os.environ), oom_retry=lambda: asked.append(1) or [sys.executable, fake, mark, "oom_once_done", "third"],
                      kv_retry=lambda: kvasked.append(1) or None, tune_retry=tune_retry)
    return rc, int(open(mark).read()), len(asked), len(kvasked)
rc, n, asked, kvn = tune_case([])
check(n == 3 and asked == 1, f"tune retry on a command without the cache: the out-of-memory retry still applies (started {n}x, oom_retry asked {asked}x)")
rc, n, asked, kvn = tune_case(["--moe-cache-mib", "3000"])
check(n == 3 and asked == 1 and kvn == 0, f"tune retry on a cache command: the cache fallback (oom_retry) still applies, the KV retry does not (started {n}x, asked {asked}x, kv asked {kvn}x)")
# a KV abort on the plan-as-is load: the KV retry runs for a command without the cache, never for one with it
rc, n, asked, kvn = tune_case([], "kv_abort_always")
check(kvn == 1, f"tune retry, no cache, KV abort: the KV retry is asked (kv asked {kvn}x)")
rc, n, asked, kvn = tune_case(["--moe-cache-mib", "3000"], "kv_abort_always")
check(kvn == 0, f"tune retry, cache command, KV abort: the KV retry is not asked (kv asked {kvn}x)")
# cache_fallback_retry with no free-memory reading: the plan's static layers and ub, nothing recorded; no static plan: one line, no command
ccmd = ["llama-server", "-c", "65536", "--n-cpu-moe", "999", "-ub", "2048", "-b", "2048", "--moe-cache-mib", "3300"]
said = []; real_say = mc.say; mc.say = lambda m, err=False: said.append(m)
recorded = []
try:
    new = mc.cache_fallback_retry(ccmd, None, {"static_n_cpu_moe": 34, "ub": 1024}, lambda tf: 1 / 0, lambda mib, fr: recorded.append((mib, fr)) or True)
    check(new is not None and "--moe-cache-mib" not in new and new[new.index("--n-cpu-moe") + 1] == "34" and new[new.index("-ub") + 1] == "1024" and not recorded,
          "free memory unknown: the plan's static layers (34) and ub (1024), the cache flag gone, nothing recorded")
    check(any("free GPU memory unknown now" in m and "3300 MiB" in m for m in said), "free memory unknown: the line says so")
    new = mc.cache_fallback_retry(ccmd, None, {"static_n_cpu_moe": 34, "ub": 2048}, lambda tf: 1 / 0, lambda mib, fr: True)
    check(new is not None and new[new.index("-ub") + 1] == "2048", "static ub 2048 keeps -ub 2048 (not forced to 1024)")
    said.clear()
    check(mc.cache_fallback_retry(ccmd, None, {}, lambda tf: 1 / 0, lambda mib, fr: recorded.append(1)) is None and not recorded and any("no second try" in m for m in said),
          "no free-memory reading and no static plan: no command, one 'no second try' line, nothing recorded")
    new = mc.cache_fallback_retry(ccmd, (8 * 2**30, 7 * 2**30), None, lambda tf: (30, 1024), lambda mib, fr: recorded.append((mib, fr)) or True)
    check(new is not None and new[new.index("--n-cpu-moe") + 1] == "30" and new[new.index("-ub") + 1] == "1024" and recorded == [(3300, 7168.0)], "free memory known: the tuner's plan, and the failed size recorded with the free memory")
finally:
    mc.say = real_say

# --- LAUNCHER-1182 (b)+(c): the re-plans' free figure and reserve (one function for the MoE re-plan and the cache fallback) ---
import types  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "planner"))
import gguf_plan  # noqa: E402
_A = lambda room=None, **kw: types.SimpleNamespace(**dict(dict(gpu_margin=None, leave_room=False, room_gib=room), **kw))
_G = 2**30
_TF = (16 * _G, 14.65 * _G)          # (total, free) as llama-server reports them after the failed load
f, r = mc.oom_replan_free(_A(), _TF)
check(f == 14.65 * _G and r == max(gguf_plan.desk_reserve_gb(*_TF)[0], 1.0), "re-plan, no room figure: the fresh figure and the desk reserve (unchanged)")
f, r = mc.oom_replan_free(_A(13.08), _TF)
check(f == 13.08 * _G and r == 1.0, f"re-plan on Windows AMD with a measured room: the room, the 1.0 GB retry floor, no desk reserve on top ({f / _G:.2f} GiB, {r} GB)")
f, r = mc.oom_replan_free(_A(13.08, leave_room=True), _TF)
check(f == 13.08 * _G and r == mc.LEAVE_ROOM_VRAM_GB, f"re-plan on the room with --leave-room: {r} GB")
f, r = mc.oom_replan_free(_A(13.08, gpu_margin=3.0), _TF)
check(f == 13.08 * _G and r == 3.0 * _G / 1e9, "re-plan on the room with --gpu-margin: the user's number as given")
f, r = mc.oom_replan_free(_A(15.0), _TF)
check(f == 14.65 * _G, "re-plan: a room larger than the fresh figure is not used")

# --- LAUNCHER-1324: (1) the dense retry takes a fresh reading, a.vram only when the user gave it; (2) llama-server's FREE figure, not the total; (3) the projector's share ---
_lsfree = {"mib": (2 * 1024, 16 * 1024)}                       # free 2 GiB of 16 GiB total, after the failed load
_real = (mc.WINDOWS, mc.list_devices, mc.pick_gpu_full)
mc.WINDOWS, mc.list_devices, mc.pick_gpu_full = True, (lambda ls, ttl=None: "x"), (lambda listing: ("Vulkan0", "AMD Radeon", _lsfree["mib"][0], _lsfree["mib"][1]))
try:
    _M = types.SimpleNamespace(free_vram_bytes=lambda: None)
    _a = types.SimpleNamespace(vram=14.0, vram_user=None)     # Windows auto-detection set a.vram = 14 GiB at plan time; the user gave no --vram
    got = mc.dense_retry_free(_a, _M, "ls")
    check(got == 2 * _G, f"dense retry: stale 14 GiB at plan time, 2 GiB free now, 16 GiB total: the fresh free figure ({None if got is None else got / _G} GiB)")
    got = mc.dense_retry_free(types.SimpleNamespace(vram=14.0, vram_user=14.0), _M, "ls")
    check(got == 14.0 * _G, "dense retry: the user's own --vram wins")
    got = mc.dense_retry_free(_a, _M, "ls", 2 * _G)
    check(got == max(2 * _G - (2 * _G + mc.MMPROJ_VRAM_HEADROOM), 0.0) == 0.0, "dense retry: a 2 GiB projector comes off the fresh figure (floor 0)")
    _lsfree["mib"] = (10 * 1024, 16 * 1024)
    got = mc.dense_retry_free(_a, _M, "ls", 2 * _G)
    check(abs(got - (10 * _G - 2 * _G - mc.MMPROJ_VRAM_HEADROOM)) < 1, f"dense retry: 10 GiB free, 2 GiB projector: {got / _G:.2f} GiB")
finally:
    mc.WINDOWS, mc.list_devices, mc.pick_gpu_full = _real
f, r = mc.oom_replan_free(_A(), _TF, 2 * _G)
check(abs(f - (14.65 * _G - 2 * _G - mc.MMPROJ_VRAM_HEADROOM)) < 1, f"MoE re-plan: the 2 GiB projector and its headroom come off the fresh figure ({f / _G:.2f} GiB)")
f, r = mc.oom_replan_free(_A(13.08), _TF, 2 * _G)
check(abs(f - (13.08 * _G - 2 * _G - mc.MMPROJ_VRAM_HEADROOM)) < 1, "MoE re-plan on the measured room: the projector comes off the room too")

# --- LAUNCHER-1324 critic: the three retry reads of the plan's projector bytes are direct reads (a missing attribute fails loudly) and none was dropped ---
import inspect  # noqa: E402
_src = inspect.getsource(mc.run_launch_plan)
check(_src.count("plan.mm_bytes") == 4 and 'getattr(plan, "mm_bytes"' not in _src,
      f"run_launch_plan: oom_replan_free (x2), dense_retry_free and the retry line each read plan.mm_bytes directly ({_src.count('plan.mm_bytes')} reads)")
check("oom_replan_free(a, tf_, plan.mm_bytes)" in _src and "oom_replan_free(a, tf, plan.mm_bytes)" in _src and "dense_retry_free(a, machine, ls, plan.mm_bytes, raw)" in _src,
      "run_launch_plan: each of the three re-plan sites passes the plan's projector bytes")
check("free_source=retry_free_source(machine)" in _src and _src.count("retry_free_source(machine)") == 3,
      "run_launch_plan: all three re-plans name the source of the fresh reading (the gap matches the plan's)")
# the gap follows the reading's label
_gp_run = gguf_plan.subprocess.run
gguf_plan.subprocess.run = lambda cmd, **k: types.SimpleNamespace(stdout="8188, 7742\n")
try:
    gguf_plan.gpu_total_free_bytes()
    check(gguf_plan.LAST_FREE_SOURCE[0] == "nvidia-smi-free", "gpu_total_free_bytes labels its reading (the re-plan's gap source)")
finally:
    gguf_plan.subprocess.run = _gp_run
check(mc.retry_free_source(types.SimpleNamespace(free_vram_source=lambda: "nvidia-smi-free")) == "nvidia-smi-free", "retry_free_source: the machine's label on Linux (also with the user's --vram: the retry reads the card again)")
_win_saved = mc.WINDOWS
mc.WINDOWS = True
try:
    check(mc.retry_free_source(types.SimpleNamespace(free_vram_source=lambda: "nvidia-smi")) == "llama-server", "retry_free_source: llama-server's own figure on Windows")
finally:
    mc.WINDOWS = _win_saved
# dense_oom_replan passes the label on: a memory.free label takes the 0.08 GiB gap, none the 0.40 GiB gap (a figure between the two answers differently)
_seen_src = []
_real_dfs = gguf_plan.dense_ffn_split
gguf_plan.dense_ffn_split = lambda *a_, **k_: _seen_src.append(k_.get("free_source")) or None
try:
    mc.dense_oom_replan(["llama-server", "-m", "x"], "x", 8192, False, 4 * _G, 0.25, False, free_source="nvidia-smi-free")
finally:
    gguf_plan.dense_ffn_split = _real_dfs
check(_seen_src == ["nvidia-smi-free"], f"dense_oom_replan hands the free_source to dense_ffn_split ({_seen_src})")
_raw = []
mc.WINDOWS, mc.list_devices, mc.pick_gpu_full = True, (lambda ls, ttl=None: "x"), (lambda listing: ("Vulkan0", "AMD Radeon", 10 * 1024, 16 * 1024))
try:
    _fr = mc.dense_retry_free(types.SimpleNamespace(vram=None, vram_user=None), types.SimpleNamespace(free_vram_bytes=lambda: None), "ls", 2 * _G, _raw)
finally:
    mc.WINDOWS, mc.list_devices, mc.pick_gpu_full = _real
check(_raw == [10 * _G] and _fr < _raw[0], "dense retry: the raw free figure (10 GiB) is kept for the retry line, before the projector's share")

# --- LAUNCHER-1324 round 2: DRIVE the oom_retry closure of run_launch_plan (fake plan and machine; dense_oom_replan, tune and the readings stubbed) ---
import threading  # noqa: E402
_MiB = 2**20


def _drive(cmd, *, mm_bytes, vram, vram_user, windows, free_mib=10 * 1024, src="nvidia-smi-free", tf=(16 * _G, 10 * _G), ncmoe_user=None):
    """run run_launch_plan on a fake plan up to run_child (stubbed: it hands back oom_retry), call oom_retry() once. -> dict(said, dense (free, free_source) or None, tune (free, free_source) or None, new)"""
    out = {"said": [], "dense": None, "tune": None, "new": None}
    a = types.SimpleNamespace(dry_run=False, project=None, agent=False, open=False, no_warm=True, exact=False, vram=vram, vram_user=vram_user, gpu_margin=None, leave_room=False,
                              room_gib=None, allow_sleep=False, cpu_gpu_split=None)
    machine = types.SimpleNamespace(free_vram_bytes=lambda: None if windows else free_mib * _MiB, free_vram_source=lambda: src)
    pl = mc.LaunchPlan()
    pl.a, pl.extra, pl.machine, pl.commit_refusal = a, [], machine, None
    pl.model, pl.ls, pl.port, pl.ctx, pl.host, pl.cmd, pl.env, pl.key_env = "m.gguf", "ls", 8099, 8192, "127.0.0.1", list(cmd), [], []
    pl.method, pl.win_limit, pl.api_key, pl.alias, pl.slotdir, pl.kv_q8, pl.ncmoe_user = "none (stock)", None, "k", "al", None, False, ncmoe_user
    pl.engine, pl.tuned, pl.autotune, pl.tune_fallback, pl.mm_bytes = "stock", None, None, None, mm_bytes
    saved = {n: getattr(mc, n) for n in ("WINDOWS", "port_busy", "loading_line", "llama_patch_line", "run_child", "list_devices", "pick_gpu_full", "dense_oom_replan", "say", "plan_fit_params", "kv_plan_type", "make_load_recorder", "_LOAD_RECORDER")}
    gsave = {n: getattr(gguf_plan, n) for n in ("tune", "gpu_total_free_bytes", "model_layout")}
    def fake_dense(c, model, ctx, kvq, free, margin, user_ot, free_source=None):
        out["dense"] = (free, free_source); return list(c) + ["-ot", "x"], "FFN in RAM"
    def fake_tune(model, ctx, **kw):
        out["tune"] = (kw.get("free"), kw.get("free_source")); return {"n_cpu_moe": 30, "ub": 512}
    mc.WINDOWS, mc.port_busy, mc.loading_line, mc.llama_patch_line = windows, (lambda p: False), (lambda m: ""), (lambda ls: "")
    mc.list_devices, mc.pick_gpu_full = (lambda ls, ttl=None: "x"), (lambda listing: ("Vulkan0", "AMD Radeon", free_mib, 16 * 1024))
    mc.dense_oom_replan, mc.plan_fit_params, mc.kv_plan_type, mc.make_load_recorder = fake_dense, (lambda ls: None), (lambda extra, kv: None), (lambda p: None)
    mc.say = lambda msg, err=False: out["said"].append(msg)
    mc.run_child = lambda c, e, **kw: out.__setitem__("new", kw["oom_retry"]()) or 0
    gguf_plan.tune, gguf_plan.gpu_total_free_bytes, gguf_plan.model_layout = fake_tune, (lambda: tf), (lambda m: {"n_layers": 40})
    try:
        mc.run_launch_plan(pl)
    finally:
        for n, v in saved.items():
            setattr(mc, n, v)
        for n, v in gsave.items():
            setattr(gguf_plan, n, v)
    return out


_dense_cmd = ["llama-server", "-m", "m.gguf", "-c", "8192"]
_proj = 600 * _MiB
_share = _proj + mc.MMPROJ_VRAM_HEADROOM
r1 = _drive(_dense_cmd, mm_bytes=_proj, vram=14.0, vram_user=None, windows=True)             # Windows: stale 14 GiB plan reading, llama-server says 10 GiB free now
check(r1["dense"] is not None and abs(r1["dense"][0] - (10 * _G - _share)) < 1, f"driven dense retry, projector: dense_oom_replan gets 10 GiB minus the projector share ({r1['dense'] and r1['dense'][0] / _G:.2f} GiB)")
check(r1["dense"] and r1["dense"][1] == "llama-server", f"driven dense retry on Windows, no user --vram: free_source llama-server ({r1['dense'] and r1['dense'][1]})")
_line = next((m for m in r1["said"] if "trying again once" in m), "")
check("now 10.0 GiB free" in _line and "after the vision projector" in _line and f"{(10 * _G - _share) / _G:.1f} GiB of it for the model" in _line, f"driven dense retry line: raw 10.0 GiB first, then the model's share ({_line[-170:]})")
r2 = _drive(_dense_cmd, mm_bytes=0, vram=14.0, vram_user=None, windows=True)
check("after the vision projector" not in next((m for m in r2["said"] if "trying again once" in m), ""), "driven dense retry, no projector: the line has no projector clause")
r3 = _drive(_dense_cmd, mm_bytes=_proj, vram=6.0, vram_user=6.0, windows=False)             # the user's own --vram wins, the label is None like the plan's
check(r3["dense"] and abs(r3["dense"][0] - (6 * _G - _share)) < 1 and r3["dense"][1] is None, f"driven dense retry with the user's --vram: their 6 GiB minus the share, free_source None ({r3['dense']})")
r4 = _drive(_dense_cmd, mm_bytes=_proj, vram=None, vram_user=None, windows=False, free_mib=5 * 1024)
check(r4["dense"] and abs(r4["dense"][0] - (5 * _G - _share)) < 1 and r4["dense"][1] == "nvidia-smi-free", f"driven dense retry on Linux, no --vram: the fresh reading and its label ({r4['dense']})")
_moe_cmd = ["llama-server", "-m", "m.gguf", "-c", "8192", "--n-cpu-moe", "20", "-ub", "1024"]
r5 = _drive(_moe_cmd, mm_bytes=_proj, vram=6.0, vram_user=6.0, windows=False, tf=(16 * _G, 9 * _G), src="nvidia-smi-free")
check(r5["tune"] and abs(r5["tune"][0] - (9 * _G - _share)) < 1 and r5["tune"][1] == "nvidia-smi-free", f"driven MoE re-plan with a user --vram: the fresh reading minus the share, labelled by the reading, not 'user' ({r5['tune']})")
r6 = _drive(_moe_cmd, mm_bytes=_proj, vram=None, vram_user=None, windows=True, tf=(16 * _G, 9 * _G))
check(r6["tune"] and r6["tune"][1] == "llama-server", f"driven MoE re-plan on Windows: free_source llama-server ({r6['tune']})")
check(r5["new"] and "--n-cpu-moe" in r5["new"] and r5["new"][r5["new"].index("--n-cpu-moe") + 1] == "30", "driven MoE re-plan: the new command has the tuner's layer count")


sys.exit(1 if fails else 0)
