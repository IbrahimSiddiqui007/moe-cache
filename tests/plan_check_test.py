#!/usr/bin/env python3
"""tools/plan_check.py and conversation_bench_pair.sh --plan-check (test-fixes 2026-11-17): two arms must run the same plan apart from what they declare. Fake launchers that print
a recorded plan (tests/plan_check_linux_stock.txt = a real laptop --dry-run of 2026-11-17); no model, no server.
   python3 tests/plan_check_test.py"""
import os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
import plan_check as pc
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

REAL = open(os.path.join(HERE, "tests", "plan_check_linux_stock.txt"), encoding="utf-8").read()
p = pc.parse_plan(REAL)
check((p["engine"], p["load_mode"], p["ub"], p["n_cpu_moe"], p["ngl"], p["ctx"], p["cache_ram"]) == ("stock", "none", "2048", "999", "99", "65536", "2048"), "parses the real laptop plan: stock engine, --load-mode none, -ub 2048, --n-cpu-moe 999")
check(p["gpu_cache_gib"] == 2.34 and p["moe_cache_gib"] is None and p["np"] == "default", "GPU expert cache 2.34 GiB (VRAM, not host RAM); -np not given = default")
plug = REAL.replace("running stock llama-server without the plugin", "plugin loaded by path").replace("-fa on", "-fa on -np 1")
q = pc.parse_plan(plug)
check(q["engine"] == "plugin" and q["np"] == "1", "plugin text -> engine plugin, -np 1")
probs, notes = pc.compare(p, q, set())
check(any(x.startswith("engine:") for x in probs) and any(x.startswith("np:") for x in probs), "engine path and -np differences are FAILs when not declared")
probs, notes = pc.compare(p, q, {"engine", "np"})
check(not probs and len(notes) >= 2, "declared differences are notes")
r = pc.parse_plan(REAL.replace("--load-mode none", "--load-mode mmap"))
check(any(x.startswith("load_mode:") for x in pc.compare(p, r, set())[0]), "--load-mode difference is a FAIL")
check(pc.compare(p, pc.parse_plan(REAL.replace("--cache-ram 2048", "--cache-ram 512")), set())[0] == [], "--cache-ram follows free memory: a note, not a FAIL")
check(not pc.parse_plan("nothing")["has_server_line"], "no server line = nothing to compare")

tmp = tempfile.mkdtemp()
def root(name, text):
    d = os.path.join(tmp, name); os.makedirs(os.path.join(d, "bin"))
    open(os.path.join(d, "plan.txt"), "w").write(text)
    open(os.path.join(d, "bin", "moe_cache.py"), "w").write('import sys, os\na = sys.argv[1:]\nsep = a.index("--") if "--" in a else len(a)\nif "--dry-run" in a[sep:]:\n    sys.exit(7)       # a dry-run flag after the separator goes to llama-server: the real launcher exits 1 (fixtures/plan/wrong_order)\nif "--dry-run" in a:\n    print(open(os.path.join(os.path.dirname(__file__), "..", "plan.txt")).read())\n')
    return d
RA, RB, RC = root("a", REAL), root("b", REAL), root("c", REAL.replace("--load-mode none", "--load-mode mmap"))
base = ["--model", "m.gguf", "--llama-server", "ls", "--ctx", "65536"]
check(pc.main(base + ["--root-a", RA, "--root-b", RB]) == 0, "same plan in both arms: OK")
check(pc.main(base + ["--root-a", RA, "--root-b", RC]) == 1, "load-mode differs and is not declared: FAIL (exit 1)")
check(pc.main(base + ["--root-a", RA, "--root-b", RC, "--allow", "load_mode"]) == 0, "load-mode declared with --allow: OK")
check(pc.main(base + ["--root-a", RA, "--root-b", RB, "--allow", "bogus"]) == 2, "unknown --allow key is refused")
check(pc.main(base + ["--root-a", RA, "--root-b", RB, "--args-a", "--n-cpu-moe 24 -- --alias x"]) == 0, "--dry-run is inserted BEFORE the -- separator (the fake launcher exits 7 otherwise)")

if os.name != "nt" and shutil.which("bash"):
    out = os.path.join(tmp, "out")
    drv = ["bash", os.path.join(HERE, "tools", "conversation_bench_pair.sh"), "--out", out, "--model", os.path.join(tmp, "m.gguf"), "--llama-server", os.path.join(tmp, "ls"), "--ctx", "4096",
           "--no-evict", "--no-cool", "--plan-check", "--root-a", RA, "--root-b", RC, "--port", "8097"]
    open(os.path.join(tmp, "m.gguf"), "wb").write(b"x"); open(os.path.join(tmp, "ls"), "w").write("#!/bin/sh\n"); os.chmod(os.path.join(tmp, "ls"), 0o755)
    r = subprocess.run(drv, capture_output=True, text=True, timeout=120)
    check(r.returncode == 1 and "plan check FAILED" in r.stdout and "nothing was started" in r.stdout and "PLAN-FAIL load_mode" in r.stdout, "pair driver --plan-check stops before any arm when the plans differ")
    check(not any(f.startswith("server-") or f.startswith("cache-") for f in os.listdir(out)), "no arm directory was created")
shutil.rmtree(tmp, ignore_errors=True)
print("FAILED: %s" % fails if fails else "all ok")
sys.exit(1 if fails else 0)
