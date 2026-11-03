#!/usr/bin/env python3
"""LINUX-011: the launcher's CPU-only mode really is CPU-only: `--device none` and `-ngl 0` (with -ngl 0 alone llama.cpp can still put the KV cache on a
loaded GPU backend). Model-free: a synthetic GGUF header (sparse file) and a fake machine, as in tests/launch_plan_test.py.
   python3 tests/cpu_only_test.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_plan_test as T
mc = T.mc

bad = 0
def check(ok, what):
    global bad
    print(("  ok   " if ok else "  FAIL ") + what)
    bad += 0 if ok else 1

GPU = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB, 7600 MiB free)\n"
NONE = "ggml_backend_load_best: ...\nAvailable devices:\n"
D = mc.cpu_only_decision
check(D(False, [], GPU) == (False, None), "decision: a GPU is listed -> not CPU-only")
check(D(False, [], NONE)[0] is True, "decision: the device list answered and has no GPU -> CPU-only")
check(D(False, [], "")[0] is False and D(False, [], None)[0] is False, "decision: no answer from --list-devices -> nothing changes")
check(D(False, [], "some error text")[0] is False, "decision: output without 'Available devices' -> nothing changes")
check(D(True, [], GPU) == (True, "--cpu-only"), "decision: --cpu-only forces it even with a GPU")
check(D(True, ["--device", "CUDA0"], GPU)[0] is False and D(False, ["--device=Vulkan0"], NONE)[0] is False, "decision: your own --device wins")
check(D(False, [], NONE, vram_given=True)[0] is False, "decision: --vram given -> not auto CPU-only")

with tempfile.TemporaryDirectory() as tmp:
    model = os.path.join(tmp, "m.gguf")
    T.make_synthetic_model(model)
    T.MACHINES["nogpu"] = dict(T.MACHINES["linux"], listing=NONE, nvidia=False, free_vram=0, total_vram=0)
    def cmdline(machine, opts, extra=()):
        r = T.run_case(machine, ["--ram", "12", *opts], list(extra), model, tmp)
        line = next((l for l in r["stdout"].splitlines() if "llama-server" in l and "-m" in l.split()), "")
        return line.split(), r
    c, r = cmdline("nogpu", [])
    check(c and c[c.index("-ngl") + 1] == "0" and "--device" in c and c[c.index("--device") + 1] == "none", f"no GPU device: -ngl 0 --device none in the command ({' '.join(c)[:120]})")
    check("CPU-only (llama-server lists no GPU device)" in r["stdout"] + r["stderr"], "...and one line says why")
    c, r = cmdline("linux", [])
    check(c and c[c.index("-ngl") + 1] == "99" and "--device" not in c, "with a GPU: unchanged (-ngl 99, no --device)")
    c, r = cmdline("linux", ["--cpu-only"])
    check(c and c[c.index("-ngl") + 1] == "0" and c[c.index("--device") + 1] == "none", "--cpu-only on a GPU machine: -ngl 0 --device none")
    c, r = cmdline("nogpu", [], ["--device", "CUDA0"])
    check(c and c.count("--device") == 1 and c[c.index("--device") + 1] == "CUDA0", "your --device after -- is kept, no second --device")
print("cpu_only_test:", "all passed" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
