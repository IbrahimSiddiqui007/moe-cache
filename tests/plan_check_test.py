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
plug = REAL.replace("env GGML_OP_OFFLOAD_MIN_BATCH=512", "env GGML_BACKEND_PATH=/path/to/plugin.so GGML_OP_OFFLOAD_MIN_BATCH=512").replace("-fa on", "-fa on -np 1")
q = pc.parse_plan(plug)
check(q["engine"] == "plugin" and q["load_method"] == "path" and q["np"] == "1", "plugin text (GGML_BACKEND_PATH in the command line) -> engine plugin, method path, -np 1")
probs, notes = pc.compare(p, q, set())
check(any(x.startswith("engine:") for x in probs) and any(x.startswith("np:") for x in probs), "engine path and -np differences are FAILs when not declared")
probs, notes = pc.compare(p, q, {"engine", "np"})
check(not probs and len(notes) >= 2, "declared differences are notes")
r = pc.parse_plan(REAL.replace("--load-mode none", "--load-mode mmap"))
check(any(x.startswith("load_mode:") for x in pc.compare(p, r, set())[0]), "--load-mode difference is a FAIL")
check(pc.compare(p, pc.parse_plan(REAL.replace("--cache-ram 2048", "--cache-ram 512")), set())[0] == [], "--cache-ram follows free memory: a note, not a FAIL")
check(not pc.parse_plan("nothing")["has_server_line"], "no server line = nothing to compare")

# the plan as data: real laptop plugin plans (--plan-json, Qwen3-30B-A3B Q4_K_M, --ram 4, --always-plugin; paths replaced), preload and path form
import json
PRE = json.load(open(os.path.join(HERE, "tests", "plan_check_plugin_preload.json"))); PATH = json.load(open(os.path.join(HERE, "tests", "plan_check_plugin_path.json")))
jp, jh = pc.parse_json_plan(PRE), pc.parse_json_plan(PATH)
check((jp["engine"], jp["load_method"], jh["load_method"], jp["ub"], jp["n_cpu_moe"], jp["load_mode"], jp["fa"], jp["cpu_mask"]) == ("plugin", "preload", "path", "2048", "38", "mmap", "on", "0x555"), "plugin plan from --plan-json: engine plugin, preload vs path, -ub 2048, --n-cpu-moe 38, -fa on, -C 0x555")
check("MOE_CACHE_EXEC_ECORES=4" in jp["env"] and "PROFILE" not in jp["env"] and "GGUF" not in jp["env"] and jp["moe_cache_gib"] is None, "speed env is listed (MOE_CACHE_EXEC_ECORES), path env (PROFILE, GGUF) is not")
check(any(x.startswith("load_method:") for x in pc.compare(jp, jh, set())[0]) and pc.compare(jp, jh, {"load_method"})[0] == [], "preload vs path is a FAIL unless declared")
def with_args(d, extra, env=()):
    d = json.loads(json.dumps(d)); d["argv"] += extra; d["env"] += [list(e) for e in env]; return d
check(pc.parse_json_plan(with_args(PRE, ["--cache-type-k", "q8_0"]))["kv"] == "q8_0/f16", "long flag --cache-type-k is read as -ctk")
for flag, val, key in [("--ctx-size", "32768", "ctx"), ("--ubatch-size", "512", "ub"), ("--batch-size", "512", "batch"), ("--n-gpu-layers", "20", "ngl"), ("--threads", "6", "threads"), ("-ot", "x=CPU", "ot"), ("--flash-attn", "off", "fa"), ("-C", "0xff", "cpu_mask"), ("--no-mmap", None, "no_mmap"), ("--mlock", None, "mlock")]:
    q = pc.parse_json_plan(with_args(PRE, [flag] + ([val] if val else [])))
    check(q[key] != jp[key] and any(x.startswith(key + ":") for x in pc.compare(jp, q, set())[0]), "%s changes %s: HARD FAIL" % (flag, key))
check(any(x.startswith("env:") for x in pc.compare(jp, pc.parse_json_plan(with_args(PRE, [], [("GGML_OP_OFFLOAD_MIN_BATCH", "512")])), set())[0]), "GGML_OP_OFFLOAD_MIN_BATCH differs: FAIL")
check(pc.compare(jp, pc.parse_json_plan(with_args(PRE, [], [("MOE_CACHE_LOG", "1"), ("MOE_CACHE_RECORD", "x")])), set())[0] == [], "MOE_CACHE log/record env does not count")
check(pc.compare(jp, pc.parse_json_plan(PRE, [("GGML_VK_X", "1")]), set())[0] != [], "an arm's own --env words (GGML_VK_*) are compared")
check(any(x.startswith("fa:") for x in pc.compare({"fa": "on"}, {"fa": "off"}, set())[0]), "fa is a HARD difference")

# fix-batch-1 2026-11-24: plugin off is the declared plan key `plugin` (MOE_CACHE_DISABLE=1 in the arm's env), not a loose env word
off = pc.parse_json_plan(PRE, [("MOE_CACHE_DISABLE", "1")])
check(jp["plugin"] == "on" and off["plugin"] == "off" and off["env"] == jp["env"], "MOE_CACHE_DISABLE=1 -> plugin off, and it is not also an env difference")
check([x.split(":")[0] for x in pc.compare(jp, off, set())[0]] == ["plugin"], "plugin on vs off is a FAIL when not declared")
check(pc.compare(jp, off, {"plugin"})[0] == [] and any(x.startswith("plugin:") for x in pc.compare(jp, off, {"plugin"})[1]), "declared with --allow plugin: a note")
check(pc.parse_json_plan(PRE, [("MOE_CACHE_DISABLE", "0")])["plugin"] == "on", "MOE_CACHE_DISABLE=0 is plugin on")

tmp = tempfile.mkdtemp()
FAKE = ('import sys, os, json\na = sys.argv[1:]\nsep = a.index("--") if "--" in a else len(a)\nhere = os.path.dirname(__file__)\n'
        'if "--dry-run" in a[sep:] or "--plan-json" in a[sep:]:\n    sys.exit(7)       # a flag after the separator goes to llama-server: the real launcher exits 1 (fixtures/plan/wrong_order)\n'
        'if "--plan-json" in a and os.path.exists(os.path.join(here, "..", "plan.json")):\n    d = json.load(open(os.path.join(here, "..", "plan.json"))); d["argv"] += a[sep + 1:]; print(json.dumps(d))\n'
        'elif "--plan-json" in a:\n    sys.exit(2)          # an old launcher: unrecognized arguments\n'
        'elif "--dry-run" in a:\n    print(open(os.path.join(here, "..", "plan.txt")).read())\n')
def root(name, text=None, js=None):
    d = os.path.join(tmp, name); os.makedirs(os.path.join(d, "bin"))
    if text is not None: open(os.path.join(d, "plan.txt"), "w").write(text)
    if js is not None: json.dump(js, open(os.path.join(d, "plan.json"), "w"))
    open(os.path.join(d, "bin", "moe_cache.py"), "w").write(FAKE)
    return d
RA, RB, RC = root("a", REAL), root("b", REAL), root("c", REAL.replace("--load-mode none", "--load-mode mmap"))      # old launchers: text fallback
JA, JB, JC = root("ja", js=PRE), root("jb", js=PRE), root("jc", js=PATH)
base = ["--model", "m.gguf", "--llama-server", "ls", "--ctx", "65536"]
check(pc.main(base + ["--root-a", RA, "--root-b", RB]) == 0, "old launcher (dry-run text fallback), same plan in both arms: OK")
check(pc.main(base + ["--root-a", RA, "--root-b", RC]) == 1, "load-mode differs and is not declared: FAIL (exit 1)")
check(pc.main(base + ["--root-a", RA, "--root-b", RC, "--allow", "load_mode"]) == 0, "load-mode declared with --allow: OK")
check(pc.main(base + ["--root-a", RA, "--root-b", RB, "--allow", "bogus"]) == 2, "unknown --allow key is refused")
check(pc.main(base + ["--root-a", RA, "--root-b", RB, "--args-a", "--n-cpu-moe 24 -- --alias x"]) == 0, "--dry-run is inserted BEFORE the -- separator (the fake launcher exits 7 otherwise)")
check(pc.main(base + ["--root-a", JA, "--root-b", JB]) == 0, "--plan-json launcher, same plan: OK")
check(pc.main(base + ["--root-a", JA, "--root-b", JB, "--args-a", "--n-cpu-moe 24 -- --alias x"]) == 0, "--plan-json is inserted BEFORE the -- separator")
check(pc.main(base + ["--root-a", JA, "--root-b", JB, "--args-b", "-- --cache-type-k q8_0"]) == 1, "--args-b \"-- --cache-type-k q8_0\" must FAIL (KV type differs)")
check(pc.main(base + ["--root-a", JA, "--root-b", JB, "--args-b", "-- --cache-type-k q8_0", "--allow", "kv"]) == 0, "... and passes when kv is declared")
check(pc.main(base + ["--root-a", JA, "--root-b", JC]) == 1 and pc.main(base + ["--root-a", JA, "--root-b", JC, "--allow", "load_method"]) == 0, "preload vs path launcher plans: FAIL unless declared")
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pc.main(base + ["--root-a", JA, "--root-b", JB])
check("PRE-TUNE" in buf.getvalue(), "without a seed the output says plainly that the check is pre-tune")
buf = io.StringIO(); seed = os.path.join(tmp, "seed"); os.makedirs(seed)
with contextlib.redirect_stdout(buf):
    pc.main(base + ["--root-a", JA, "--root-b", JB, "--seed-a", seed, "--seed-b", seed])
check("TUNED plans" in buf.getvalue() and "PRE-TUNE" not in buf.getvalue(), "with the prepared caches as seeds the output says it checked the tuned plans")

if os.name != "nt" and shutil.which("bash"):
    out = os.path.join(tmp, "out")
    drv = ["bash", os.path.join(HERE, "tools", "conversation_bench_pair.sh"), "--out", out, "--model", os.path.join(tmp, "m.gguf"), "--llama-server", os.path.join(tmp, "ls"), "--ctx", "4096",
           "--no-evict", "--no-cool", "--plan-check", "--root-a", RA, "--root-b", RC, "--port", "8097"]
    open(os.path.join(tmp, "m.gguf"), "wb").write(b"x"); open(os.path.join(tmp, "ls"), "w").write("#!/bin/sh\n"); os.chmod(os.path.join(tmp, "ls"), 0o755)
    r = subprocess.run(drv, capture_output=True, text=True, timeout=120)
    check(r.returncode == 1 and "plan check (pre-tune) FAILED" in r.stdout and "nothing was run further" in r.stdout and "PLAN-FAIL load_mode" in r.stdout, "pair driver --plan-check stops before any arm when the plans differ")
    check(not any(f.startswith("server-") or f.startswith("cache-") for f in os.listdir(out)), "no arm directory was created")
shutil.rmtree(tmp, ignore_errors=True)
print("FAILED: %s" % fails if fails else "all ok")
sys.exit(1 if fails else 0)
