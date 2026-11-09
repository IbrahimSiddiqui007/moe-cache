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
stopfake = os.path.join(TMP, "stopfake.py")
open(stopfake, "w").write('''
import os, signal, sys, time
mark = sys.argv[1]
def die(*_):                      # what Windows' TerminateProcess looks like to the launcher: a plain exit code 1, with the out-of-memory text in the log
    sys.stderr.write("ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory\\n"); sys.stderr.flush(); sys.exit(1)
signal.signal(signal.SIGTERM, die)
n = int(open(mark).read()) if os.path.exists(mark) else 0
open(mark, "w").write(str(n + 1))
if n == 0:
    time.sleep(30)
sys.stderr.write("moe-cache: fake ok\\n"); sys.exit(0)
''')
def run_guard(guard_mode):
    mark = os.path.join(TMP, "m_guard_" + guard_mode)
    if os.path.exists(mark):
        os.remove(mark)
    calls = []
    def oom_retry():
        calls.append(1)
        return [sys.executable, stopfake, mark]
    saved = (mc.meminfo_available, os.environ.get("MOE_CACHE_COMMIT_GUARD"))
    mc.meminfo_available = lambda: 0.1e9 if os.path.exists(mark) else 50e9      # far under the hard line once the fake server runs: the guard stops it at its next sample
    os.environ["MOE_CACHE_COMMIT_GUARD"] = guard_mode
    try:
        rc = mc.run_child([sys.executable, stopfake, mark], dict(os.environ, MOE_CACHE_COMMIT_GUARD=guard_mode), oom_retry=oom_retry)
    finally:
        mc.meminfo_available = saved[0]
        os.environ.pop("MOE_CACHE_COMMIT_GUARD", None) if saved[1] is None else os.environ.__setitem__("MOE_CACHE_COMMIT_GUARD", saved[1])
    return rc, len(calls), int(open(mark).read())
rc, calls, starts = run_guard("stop")
check(rc == 1 and calls == 0 and starts == 1, f"the commit guard stopped the server (exit 1, out-of-memory line in the log): no retry, one start (rc {rc}, retry asked {calls}x, started {starts}x)")
sys.exit(1 if fails else 0)
