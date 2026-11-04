#!/usr/bin/env python3
"""planner/plan.py smi(): a hanging nvidia-smi (a wedged driver) must not hang the planner. The fake nvidia-smi sleeps 40 s in a child process (not exec'd, so
the child keeps the pipe open like the real tool's helpers); plan.py has to give up after its 5 s timeout and carry on without a GPU reading.
   python3 tests/planner_smi_test.py"""
import os, stat, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests"))
import launch_plan_test as lpt  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

if os.name == "nt":
    print("  skip: the fake nvidia-smi is a shell script")
    sys.exit(0)

TMP = tempfile.mkdtemp(prefix="moe-smi-test-")
model = os.path.join(TMP, "m.gguf")
lpt.make_synthetic_model(model)
bindir = os.path.join(TMP, "bin"); os.makedirs(bindir)
fake = os.path.join(bindir, "nvidia-smi")
open(fake, "w").write("#!/bin/sh\nsleep 40\necho 8192\n")
os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)
env = dict(os.environ, PATH=bindir + os.pathsep + os.environ.get("PATH", ""))

t0 = time.time()
try:
    r = subprocess.run([sys.executable, os.path.join(HERE, "planner", "plan.py"), model, "--ram-gib", "16"], capture_output=True, text=True, env=env, timeout=60)
    rc, out = r.returncode, r.stdout + r.stderr
except subprocess.TimeoutExpired:
    rc, out = None, ""
took = time.time() - t0
check(rc == 0, f"plan.py finishes (rc {rc}) although nvidia-smi hangs")
check(took < 20, f"...within the 5 s smi timeout plus slack ({took:.1f} s; the fake sleeps 40 s)")
check("8192" not in out, "...and does not use a reading that never arrived")
import shutil; shutil.rmtree(TMP, ignore_errors=True)
print("\nFAILED: " + "; ".join(fails) if fails else "\nplanner smi tests passed")
sys.exit(1 if fails else 0)
