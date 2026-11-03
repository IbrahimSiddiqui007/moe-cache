#!/usr/bin/env python3
"""PACKAGING-027: `llama-server --help` and `ldd` are not asked again on every launch. No llama-server is started: a fake runner counts the calls.
   python3 tests/probe_cache_test.py"""
import os, sys, tempfile, time, types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-probe-test-")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc  # noqa: E402

mc.CACHE_DIR = os.path.join(TMP, "cache")
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

calls = []
def runner(text):
    def run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout=text, stderr="")
    return run

ls = os.path.join(TMP, "llama-server")
open(ls, "w").write("x")
a = mc.ls_help(ls, runner("--cache-ram N\n"))
b = mc.ls_help(ls, runner("SHOULD NOT RUN"))
check(a == b == "--cache-ram N\n" and len(calls) == 1, "the second ask is answered from the cache file; llama-server ran once")
check(os.path.isfile(os.path.join(mc.CACHE_DIR, mc.PROBE_CACHE_NAME)), "the answer is in probe-cache.json")

# the binary changes: asked again
os.utime(ls, ns=(time.time_ns(), time.time_ns() + 5_000_000_000))
c = mc.ls_help(ls, runner("--cache-ram N\n--spec-type draft-mtp\n"))
check("draft-mtp" in c and len(calls) == 2, "a new build of llama-server (changed file time) is asked again")

# a ggml library next to it changes: asked again
lib = os.path.join(TMP, "libggml-base.so")
open(lib, "w").write("1")
mc.ls_help(ls, runner("v1"))                       # fingerprint now includes the library
n = len(calls)
mc.ls_help(ls, runner("ignored"))
check(len(calls) == n, "with the library present the answer is cached again")
os.utime(lib, ns=(time.time_ns(), time.time_ns() + 9_000_000_000))
mc.ls_help(ls, runner("v2"))
check(len(calls) == n + 1, "a changed ggml library next to llama-server invalidates the answer")

# a fake path is never cached
mc.ls_help(os.path.join(TMP, "nothing-here"), runner("fake"))
mc.ls_help(os.path.join(TMP, "nothing-here"), runner("fake"))
check(calls[-1] == [os.path.join(TMP, "nothing-here"), "--help"] and sum(1 for c in calls if c[0].endswith("nothing-here")) == 2, "a llama-server that does not exist is not cached")

# a broken or unwritable cache file does not stop a launch
open(os.path.join(mc.CACHE_DIR, mc.PROBE_CACHE_NAME), "w").write("{not json")
n = len(calls)
check(mc.ls_help(ls, runner("fresh")) in ("fresh", "v2") and len(calls) == n + 1, "a damaged cache file is ignored and rewritten")
mc.CACHE_DIR = os.path.join(TMP, "file-not-folder"); open(mc.CACHE_DIR, "w").write("x")
try:
    mc.ls_help(ls, runner("fresh2")); ok = True
except Exception as e:
    ok = False
check(ok, "a cache folder that cannot be created does not raise")

# the cache keeps at most PROBE_CACHE_KEEP binaries
mc.CACHE_DIR = os.path.join(TMP, "cache2")
for i in range(mc.PROBE_CACHE_KEEP + 3):
    p = os.path.join(TMP, f"ls{i}"); open(p, "w").write(str(i))
    mc.ls_help(p, runner(f"h{i}"))
import json
check(len(json.load(open(os.path.join(mc.CACHE_DIR, mc.PROBE_CACHE_NAME)))) == mc.PROBE_CACHE_KEEP, f"the file keeps the newest {mc.PROBE_CACHE_KEEP} binaries only")

# ldd: one call for several asks inside the memo time (Linux only: Windows has no ldd)
if os.name != "nt":
    import subprocess
    seen = []
    real = subprocess.run
    subprocess.run = lambda cmd, **kw: (seen.append(cmd), types.SimpleNamespace(stdout="libggml-base.so => /x/libggml-base.so.0.27.0 (0x1)\n"))[1]
    try:
        mc._LDD_MEMO.clear()
        outs = [mc.ldd(ls) for _ in range(3)]
        check(len(seen) == 1 and outs[0] == outs[2], "ldd runs once for three asks")
        mc._LDD_MEMO[mc.which(ls)] = (time.time() - mc.LDD_MEMO_SECONDS - 1, "old")
        mc.ldd(ls)
        check(len(seen) == 2, "after the memo time ldd runs again (a long-lived control page)")
    finally:
        subprocess.run = real
sys.exit(1 if fails else 0)
