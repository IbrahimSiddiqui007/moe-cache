#!/usr/bin/env python3
"""DECODE-917 (patch 0091) launcher stage: --cpu-gpu-split only touches the dense FFN-in-RAM plan on a kit whose llama.cpp carries patch 0091.
   python3 tests/cpu_gpu_split_test.py"""
import os, sys, tempfile, types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests"))
sys.path.insert(0, os.path.join(HERE, "planner"))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc                # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def kit(patches):
    d = tempfile.mkdtemp()
    ls = os.path.join(d, "llama-server")
    open(ls, "w").close()
    if patches is not None:
        with open(os.path.join(d, "PATCHES-moe-cache.txt"), "w") as f:
            f.write(f"llama.cpp b12064 + moe-cache patches {patches} built in, all off\n")
    return ls

CMD = ["llama-server", "-m", "m.gguf", "-ngl", "99", "-c", "8192", "-fa", "on", "--load-mode", "mmap", "--no-warmup", "-ot", "blk\\.(22|23)\\.ffn_(up|down|gate)\\.weight=CPU"]
A = lambda v: types.SimpleNamespace(cpu_gpu_split=v)

env = []
out = mc.cpu_gpu_split_stage(A(None), env, CMD, kit("0004,0091"))
check(out == CMD and not env, "flag off: command and environment untouched")

env = []
out = mc.cpu_gpu_split_stage(A("auto"), env, CMD, kit("0004,0020,0091"))
check(env == [("GGML_CPU_GPU_SPLIT", "auto")], "auto: GGML_CPU_GPU_SPLIT=auto in the environment")
check(out[out.index("--load-mode") + 1] == "none" and "--no-repack" in out and CMD[-2:] == out[out.index("-ot"):out.index("-ot") + 2], "auto: --load-mode none, --no-repack, the -ot stays")
check(CMD[CMD.index("--load-mode") + 1] == "mmap", "the input command is not changed in place")

env = []
out = mc.cpu_gpu_split_stage(A("0.35"), env, CMD, kit("0091"))
check(env == [("GGML_CPU_GPU_SPLIT", "0.35")], "a fraction is passed as given")

for bad in ("0.9", "0.01", "x"):
    try:
        mc.cpu_gpu_split_stage(A(bad), [], CMD, kit("0091")); check(False, f"{bad} refused")
    except SystemExit:
        check(True, f"{bad} refused")

env = []
out = mc.cpu_gpu_split_stage(A("auto"), env, CMD, kit("0004,0090"))
check(out == CMD and not env, "kit without 0091: not applied")
env = []
out = mc.cpu_gpu_split_stage(A("auto"), env, CMD, kit(None))
check(out == CMD and not env, "official build (no patch note): not applied")
env = []
plain = [c for c in CMD[:-2]]
out = mc.cpu_gpu_split_stage(A("auto"), env, plain, kit("0091"))
check(out == plain and not env, "no -ot (not the FFN-in-RAM plan): not applied")

sys.exit(1 if fails else 0)
