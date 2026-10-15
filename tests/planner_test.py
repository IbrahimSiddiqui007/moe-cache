#!/usr/bin/env python3
"""Planner / tuner / launcher checks on a real MoE GGUF (I13; CI runs it on Granite 1B). No GPU needed: VRAM is passed in.
   python3 tests/planner_test.py MODEL.gguf
Checks rules that must hold for any model: more VRAM never puts more expert layers on the CPU, the margin is respected, the micro-batch is one the
tuner knows, plan.py's answer is in range, and the launcher passes --vram to the tuner in GiB (audit A12) and --gpu-margin through (audit A05)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
import gguf_plan  # noqa: E402

model = sys.argv[1]
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

lay = gguf_plan.model_layout(model)
check(lay["n_layers"] > 0 and lay["expert_bytes_per_layer"] > 0, f"layout: {lay['n_layers']} layers, {lay['expert_bytes_per_layer'] / 1e6:.1f} MB experts per layer")

GIB = 2**30
prev = None
for gib in (2, 3, 4, 6, 8, 12, 16, 24):
    t = gguf_plan.tune(model, 8192, free=gib * GIB, want_cache=False)
    check(t["ub"] in (512, 1024, 2048), f"tune at {gib} GiB: ub {t['ub']} is a known micro-batch")
    ncm = min(t["n_cpu_moe"], lay["n_layers"])
    if prev is not None:
        check(ncm <= prev, f"tune at {gib} GiB: {ncm} CPU expert layers <= {prev} with less VRAM")
    prev = ncm

# a larger margin can only keep the same or more expert layers on the CPU
a = gguf_plan.tune(model, 8192, free=8 * GIB, want_cache=False, reserve_gb=0.5)
b = gguf_plan.tune(model, 8192, free=8 * GIB, want_cache=False, reserve_gb=3.0)
check(min(b["n_cpu_moe"], lay["n_layers"]) >= min(a["n_cpu_moe"], lay["n_layers"]), "a 3 GB margin keeps at least as many expert layers on the CPU as 0.5 GB")

# plan.py (the launcher's first opinion) answers in range
out = subprocess.run([sys.executable, os.path.join(HERE, "planner", "plan.py"), model, "--ctx", "8192", "--vram-gib", "8", "--ram-gib", "16", "--json"],
                     capture_output=True, text=True, timeout=120)
try:
    info = json.loads(out.stdout)
    check(0 <= info["n_cpu_moe"] <= lay["n_layers"], f"plan.py: n_cpu_moe {info['n_cpu_moe']} within 0..{lay['n_layers']}")
except Exception as e:
    check(False, f"plan.py --json: {e} {out.stderr[-300:]}")

# the launcher's dry run: --vram is GiB for the tuner too, --gpu-margin reaches the tuner
def dry(*extra):
    env = dict(os.environ)
    env.setdefault("MOE_CACHE_LIB", os.path.join(HERE, "build", "libggml-moe-cache.so"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "server", model, "--ram", "4", "--always-plugin", "--dry-run", *extra],
                       capture_output=True, text=True, timeout=300, env=env)
    return r.stdout + r.stderr
def ncmoe(text):
    import re
    m = re.search(r"--n-cpu-moe (\d+)", text)
    return int(m.group(1)) if m else None
for v in ("2", "4", "8"):
    want = min(gguf_plan.tune(model, 8192, free=float(v) * GIB, want_cache=False)["n_cpu_moe"], 999)
    got = ncmoe(dry("--vram", v))
    check(got == want, f"launcher --vram {v}: --n-cpu-moe {got} equals the tuner at {v} GiB ({want})")
lo, hi = ncmoe(dry("--vram", "8", "--gpu-margin", "0.2")), ncmoe(dry("--vram", "8", "--gpu-margin", "3"))
check(lo is not None and hi is not None and hi >= lo, f"launcher --gpu-margin 3 keeps at least as many CPU expert layers ({hi}) as 0.2 ({lo})")

print(f"planner checks: {len(fails)} failed")
sys.exit(1 if fails else 0)
