#!/usr/bin/env python3
"""doctor / check preflight (N34): `doctor` is `check`, ends with an overview table of the machine and a list of what to fix. No model, no llama-server.
   python3 tests/doctor_test.py"""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH = os.path.join(HERE, "bin", "moe_cache.py")
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def run(*args):
    env = dict(os.environ)
    env.pop("LLAMA_SERVER", None)
    env["XDG_CACHE_HOME"] = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, LAUNCH, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, env=env)
    return r.returncode, r.stdout + r.stderr

missing = os.path.join(tempfile.mkdtemp(), "no-such-llama-server")
rc, out = run("doctor", "--llama-server", missing)
check(rc == 1, f"doctor with a missing llama-server exits 1 ({rc})")
check("this machine at a glance" in out and "what to fix" in out, "the overview table and the fix list are printed")
check("[FAIL] llama-server" in out and "not found" in out, "the missing llama-server is a FAIL line")
check(any(l.strip().startswith("[FAIL] llama-server") for l in out.split("what to fix")[1].splitlines()), "...and is repeated under 'what to fix' with its remedy")
check("system" in out.split("at a glance")[1] and "ram" in out.split("at a glance")[1], "system and RAM rows are filled")
rc2, out2 = run("check", "--llama-server", missing)
check(rc2 == rc and "this machine at a glance" in out2, "check prints the same (one command, doctor is its name)")
rc3, out3 = run("doctor", "--help")
check(rc3 == 0 and "--llama-server" in out3, "doctor --help works")
rc4, out4 = run("--help")
check("doctor" in out4, "the top help names doctor")
print("doctor test: " + ("PASS" if not fails else "FAILED: " + "; ".join(fails)))
sys.exit(1 if fails else 0)
