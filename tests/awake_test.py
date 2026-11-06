#!/usr/bin/env python3
"""Keep-awake check (idea I47): while the server runs the machine must not sleep, and nothing else changes. No model, no llama-server.
   python3 tests/awake_test.py
Linux: `systemd-inhibit --what=sleep:idle` (skipped quietly when systemd-inhibit/logind is not there, e.g. a container),
a helper process holds the lock (shown in `systemd-inhibit --list`), llama-server itself is not wrapped, and the lock goes when the launcher releases it or is killed.
Windows: SetThreadExecutionState is set (the returned release call clears it). --allow-sleep and MOE_CACHE_ALLOW_SLEEP=1 change nothing."""
import os, re, subprocess, sys, time

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


def locks():
    try:
        return subprocess.run(["systemd-inhibit", "--list", "--no-pager"], capture_output=True, text=True).stdout
    except OSError:
        return ""


def lock_pids():
    """PIDs of the moe-cache keep-awake locks listed now (the PID column is the systemd-inhibit process)."""
    pids = set()
    for line in locks().splitlines():
        if "llama-server is running" in line:
            m = re.match(r"\S+\s+\d+\s+\S+\s+(\d+)\s", line)
            if m:
                pids.add(int(m.group(1)))
    return pids


def ppid(pid):
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return int(fh.read().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return -1


def own_locks(root):
    """The locks whose helper is a child of `root`. A real launcher of another session (or another test run) on the same machine holds a lock with the
    same text and may start or stop it at any moment, so counting all of them made this test fail once (pc-ci-script session); only children of the
    process this test started count."""
    return {p for p in lock_pids() if ppid(p) == root}


def wait_for(cond, limit=20.0):
    """Poll until cond() is true (systemd-inhibit registers its lock, and drops it, a moment after the process starts or the pipe closes)."""
    end = time.time() + limit
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


if lock_pids():
    print(f"  note: {len(lock_pids())} moe-cache keep-awake lock(s) of other launchers are held on this machine; counting only this test's own")
c, rel, note = mc.keep_awake(cmd)
if mc.WINDOWS:
    check("keeping the PC awake" in note and c == cmd, f"Windows: power request set, command unchanged ({note[:50]})")
    rel()
    check(True, "release call runs")
elif note:
    check(c == cmd and "keeping the machine awake" in note, "Linux: the command stays llama-server's own (not wrapped)")
    me = os.getpid()
    check(wait_for(lambda: len(own_locks(me)) == 1), "the lock is listed while the server runs (one of ours)")
    mine = own_locks(me)
    rel()
    check(wait_for(lambda: not (mine & lock_pids())), "the lock is gone after release")
    # a launcher that dies without releasing (kill -9) must not leave the lock: run the keep-awake in a child process and kill it
    code = "import sys, time; sys.path.insert(0, %r); import moe_cache as mc; c, r, n = mc.keep_awake(['x']); print('up', flush=True); time.sleep(60)" % os.path.join(HERE, "bin")
    q = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    q.stdout.readline()
    check(wait_for(lambda: len(own_locks(q.pid)) == 1), "a second launcher holds a lock")
    mine = own_locks(q.pid)
    q.kill(); q.wait()
    check(wait_for(lambda: not (mine & lock_pids())), "killed launcher (kill -9): the lock is released by the closing pipe")
else:
    check(c == cmd, "no systemd-inhibit / logind here: skipped quietly, command unchanged")

print("awake test: " + ("PASS" if not fails else "FAILED: " + "; ".join(fails)))
sys.exit(1 if fails else 0)
