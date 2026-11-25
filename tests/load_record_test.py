#!/usr/bin/env python3
"""planner-fixes-3 (LAUNCHER-1182 known limits): Vulkan / HIP out-of-memory lines are seen as failed edges, and two launcher PROCESSES writing loads.json at
the same time do not lose each other's entries. Model-free.
   python3 tests/load_record_test.py"""
import json, multiprocessing, os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "planner"))
import load_record as lr  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# 1. out-of-memory lines of other backends set "oom" in the parsed load, and Recorder.feed keeps them (it filters lines by text)
OOM_LINES = ["ggml_vulkan: Device memory allocation of size 940714752 failed.", "ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory",
             "llama_model_load: error loading model: unable to allocate Vulkan0 buffer", "hipMalloc failed: out of memory", "CUDA error: out of memory",
             "llama_init_from_model: failed to allocate Vulkan0 compute buffer"]
NOT_OOM = ["llama_model_load: error loading model: unable to allocate CPU buffer", "srv  log: model loaded", "ggml_vulkan: Found 1 Vulkan devices:"]
for line in OOM_LINES:
    check(bool(lr.parse_log([line]).get("oom")), "parse_log: oom from: " + line[:60])
for line in NOT_OOM:
    check(not lr.parse_log([line]).get("oom"), "parse_log: not oom: " + line[:60])
CT = tempfile.mkdtemp(prefix="loadrec-ent-")
ctx = {"cache_dir": CT, "machine": "m", "build": "b", "model": "mod", "ctx": 65536, "ub": 512, "kv": True, "engine": "stock", "mtp": False, "gpu_expert_mib": 3000, "free_plan_mib": 7000, "free_source": "nvidia-smi", "n_cpu_moe": 27, "free_now_mib": 6900}
for line in OOM_LINES:
    r = lr.Recorder(dict(ctx))
    r.feed("load_tensors:   Vulkan0 model buffer size =  5000.00 MiB")
    r.feed(line)
    ent = r.finish()
    check(ent.get("edges", {}).get("fail_ncmoe") == 27, "Recorder: a failed edge at --n-cpu-moe 27 is recorded from: " + line[:60])
r = lr.Recorder(dict(ctx))                                  # the running server's failure after "model loaded" is not a failed load
r.feed("srv  load_model: model loaded"); r.feed(OOM_LINES[0])
check("edges" not in r.finish(), "Recorder: an out-of-memory line after 'model loaded' records no edge")

# 2. two processes saving different entries at the same time: nothing is lost
def worker(args):
    cache, i = args
    for j in range(15):
        ok = lr.save(cache, "m", "b", f"model{i}", 4096 + j, 512, True, "stock", False, {"buffers": {"compute_mib": float(i * 100 + j), "at": "2026-11-25 12:00"}})
        if not ok:
            sys.exit(3)
    sys.exit(0)

def run_workers(cache, n=6):
    ps = [multiprocessing.Process(target=worker, args=((cache, i),)) for i in range(n)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(120)
    return all(p.exitcode == 0 for p in ps)

if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="loadrec-")
    check(run_workers(tmp), "6 processes x 15 saves all returned True")
    d = json.load(open(lr.store_path(tmp)))
    check(len(d["buffers"]) == 6 * 15, f"no entry lost across processes ({len(d['buffers'])} of 90)")
    check(not [f for f in os.listdir(tmp) if f.endswith(".tmp")], "no temp file left behind")
    # a lock file that cannot be created (the cache dir is a file path inside a read-only spot) never raises and still saves or returns False
    bad = os.path.join(tmp, "loads.json.lock")
    os.remove(bad)
    os.mkdir(bad)                                            # opening a directory as the lock file fails: save goes on without the file lock
    ok = lr.save(tmp, "m", "b", "modelX", 1, 512, True, "stock", False, {"buffers": {"compute_mib": 1.0, "at": "2026-11-25 12:00"}})
    check(ok is True, "an unusable lock file does not stop a save")
    print("FAIL: %d" % len(fails) if fails else "all ok")
    sys.exit(1 if fails else 0)
