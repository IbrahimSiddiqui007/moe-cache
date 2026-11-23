#!/usr/bin/env python3
"""LAUNCHER-1234 row 15: run_plain_child (the `diffusion` child) is tied to the launcher: SIGTERM to the launcher reaches the child, SIGKILL of the launcher kills it (Linux)."""
import os, sys, subprocess, time, signal, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mc", os.path.join(here, "..", "bin", "moe_cache.py")); mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)
FAILS = []
def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg); (None if ok else FAILS.append(msg))
check(mc.run_plain_child([sys.executable, "-c", "import sys; sys.exit(3)"], dict(os.environ)) == 3, "exit code passed through")
if sys.platform.startswith("linux"):
    drv = "import sys,os; sys.path.insert(0,%r); import importlib.util as u; s=u.spec_from_file_location('mc',%r); m=u.module_from_spec(s); s.loader.exec_module(m); print('CHILDPID', flush=True); m.run_plain_child(['sleep','60'], dict(os.environ))" % (here, os.path.join(here, "..", "bin", "moe_cache.py"))
    for sig, name in ((signal.SIGTERM, "SIGTERM"), (signal.SIGKILL, "SIGKILL")):
        l = subprocess.Popen([sys.executable, "-c", drv], stdout=subprocess.PIPE, text=True); l.stdout.readline(); time.sleep(0.7)
        kids = subprocess.run(["pgrep", "-P", str(l.pid)], capture_output=True, text=True).stdout.split()
        check(len(kids) == 1, f"{name}: the launcher has one child ({kids})")
        os.kill(l.pid, sig); l.wait(); time.sleep(1.0)
        alive = [k for k in kids if os.path.exists(f"/proc/{k}") and open(f"/proc/{k}/stat").read().split()[2] != "Z"]
        check(not alive, f"{name} to the launcher: the child is gone ({alive})")
        for k in alive: os.kill(int(k), 9)
print("run_plain_child test:", "FAIL" if FAILS else "PASS"); sys.exit(1 if FAILS else 0)
