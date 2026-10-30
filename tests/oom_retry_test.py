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
sys.exit(1 if fails else 0)
