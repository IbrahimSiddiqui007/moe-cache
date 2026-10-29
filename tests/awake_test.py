#!/usr/bin/env python3
"""Keep-awake check (idea I47): while the server runs the machine must not sleep, and nothing else changes. No model, no llama-server.
   python3 tests/awake_test.py
Linux: `systemd-inhibit --what=sleep:idle` (skipped quietly when systemd-inhibit/logind is not there, e.g. a container),
a helper process holds the lock (shown in `systemd-inhibit --list`), llama-server itself is not wrapped, and the lock goes when the launcher releases it or is killed.
Windows: SetThreadExecutionState is set (the returned release call clears it). --allow-sleep and MOE_CACHE_ALLOW_SLEEP=1 change nothing."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

cmd = [sys.executable, "-c", "import time; time.sleep(4)"]
os.environ.pop("MOE_CACHE_ALLOW_SLEEP", None)

c, rel, note = mc.keep_awake(cmd, allow_sleep=True)
check(c == cmd and "may sleep" in note, f"--allow-sleep: command unchanged, one line says so ({note[:60]})")
os.environ["MOE_CACHE_ALLOW_SLEEP"] = "1"
c, rel, note = mc.keep_awake(cmd)
check(c == cmd and "may sleep" in note, "MOE_CACHE_ALLOW_SLEEP=1 does the same")
del os.environ["MOE_CACHE_ALLOW_SLEEP"]

c, rel, note = mc.keep_awake(cmd)
def locks():
    return subprocess.run(["systemd-inhibit", "--list", "--no-pager"], capture_output=True, text=True).stdout
if mc.WINDOWS:
    check("keeping the PC awake" in note and c == cmd, f"Windows: power request set, command unchanged ({note[:50]})")
    rel()
    check(True, "release call runs")
elif note:
    check(c == cmd and "keeping the machine awake" in note, "Linux: the command stays llama-server's own (not wrapped)")
    check("llama-server is running" in locks(), "the lock is listed while the server runs")
    rel()
    time.sleep(0.5)
    check("llama-server is running" not in locks(), "the lock is gone after release")
    # a launcher that dies without releasing (kill -9) must not leave the lock: run the keep-awake in a child process and kill it
    code = "import sys, time; sys.path.insert(0, %r); import moe_cache as mc; c, r, n = mc.keep_awake(['x']); print('up', flush=True); time.sleep(60)" % os.path.join(HERE, "bin")
    q = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    q.stdout.readline()
    check("llama-server is running" in locks(), "a second launcher holds a lock")
    q.kill(); q.wait()
    time.sleep(1.0)
    check("llama-server is running" not in locks(), "killed launcher (kill -9): the lock is released by the closing pipe")
else:
    check(c == cmd, "no systemd-inhibit / logind here: skipped quietly, command unchanged")

print("awake test: " + ("PASS" if not fails else "FAILED: " + "; ".join(fails)))
sys.exit(1 if fails else 0)
