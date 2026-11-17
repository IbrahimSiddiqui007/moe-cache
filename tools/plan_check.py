#!/usr/bin/env python3
"""plan_check.py: do the two arms of a pair run the SAME plan, apart from what they declare? (test-fixes 2026-11-17, critic report_testing.md fix 7)
Runs the launcher's --dry-run for arm A and arm B (no model load; --dry-run goes BEFORE any `--`), reads the plan facts from the llama-server command line and the launcher's plan lines,
and compares: engine path (plugin or stock llama-server), --load-mode, -np, -ub, -b, KV types, --n-cpu-moe, -ngl, -c, device. A difference that is not named in --allow is a FAIL
(yesterday's void laptop A/B differed in more than the switch under test, and nothing compared the plans). Soft facts that follow free memory (--cache-ram, the expert-cache size) are only reported.
Called by conversation_bench_pair.sh --plan-check; usable alone:
  python3 tools/plan_check.py --model M.gguf --llama-server LS --ctx 65536 [--root-a DIR --args-a "..." --env-a "K=V ..."] [--root-b ...] [--allow load_mode,n_cpu_moe] [--cache-seed DIR]
Exit 0 = same plan (or only allowed differences), 1 = a difference that was not declared, 2 = a launcher dry run failed. The same parser, with Windows lines too, lives in handover/tools/chainkit/ckit/plan.py."""
import argparse, os, re, shutil, subprocess, sys, tempfile

HARD = ["engine", "load_mode", "np", "ub", "batch", "kv", "n_cpu_moe", "ngl", "ctx", "device"]
SOFT = ["cache_ram", "gpu_cache_gib", "moe_cache_gib", "fa"]
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def parse_flags(line):
    toks, out, i = line.split(), {}, 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("-") and not NUM.match(t):
            key = t.lstrip("-")
            if i + 1 < len(toks) and (not toks[i + 1].startswith("-") or NUM.match(toks[i + 1])):
                out[key] = toks[i + 1]; i += 2; continue
            out[key] = True
        i += 1
    return out


def parse_plan(text):
    text = text.replace("\r", "").replace("\x00", "")
    lines = text.split("\n")
    srv = [l for l in lines if re.search(r"llama-server(\.exe)?\"?\s", l) and re.search(r"\s-m\s", l)]
    fl = parse_flags(re.sub(r"^.*llama-server(\.exe)?\"?", "", srv[-1])) if srv else {}
    plugin = bool(re.search(r"GGML_BACKEND_PATH=|MOE_CACHE_LIB=|plugin loaded by path|MOE_CACHE_SIZE_GIB=", text)) and "without the plugin" not in text
    s = lambda k: fl.get(k) if isinstance(fl.get(k), str) else None
    m_gpu = re.search(r"GPU expert cache of (\d+) MiB", text) or None
    m_mib = re.search(r"--moe-cache-mib\s+(\d+)", text)
    m_ram = re.search(r'MOE_CACHE_SIZE_GIB="?([\d.]+)"?', text) or re.search(r"(?<!GPU )expert cache(?: of| size|:)?\s*~?\s*([\d.]+)\s*GiB", text, re.I)
    np_val = next((fl[k] for k in ("np", "parallel", "n-parallel") if isinstance(fl.get(k), str)), None)
    return {"has_server_line": bool(srv), "engine": ("plugin" if plugin else "stock") if srv else None,
            "load_mode": s("load-mode") or ("default" if srv else None), "np": np_val or ("default" if srv else None),
            "ub": s("ub") or (re.search(r"(?:micro-batch -ub|n_ubatch\s*=)\s*(\d+)", text) or [None, None])[1], "batch": s("b"),
            "kv": ("%s/%s" % (s("ctk") or "f16", s("ctv") or "f16")) if srv else None,
            "n_cpu_moe": s("n-cpu-moe"), "ngl": s("ngl"), "ctx": s("c"), "device": s("device") or ("default" if srv else None),
            "cache_ram": s("cache-ram"), "fa": s("fa"),
            "gpu_cache_gib": (round(int((m_gpu or m_mib).group(1)) / 1024.0, 2) if (m_gpu or m_mib) else None),
            "moe_cache_gib": float(m_ram.group(1)) if m_ram else None}


def dry_run(root, model, ls, ctx, port, args, env_words, seed):
    """the launcher's --dry-run for one arm; --dry-run is inserted before a `--` separator in the arm's arguments."""
    a = args.split()
    if "--" in a:
        a = a[:a.index("--")] + ["--dry-run"] + a[a.index("--"):]
    else:
        a = a + ["--dry-run"]
    cache = tempfile.mkdtemp(prefix="plan_check_cache_")
    try:
        if seed:
            shutil.copytree(seed, cache, dirs_exist_ok=True, symlinks=True)
        env = dict(os.environ, XDG_CACHE_HOME=cache)
        env.update(dict(w.split("=", 1) for w in env_words.split() if "=" in w))
        r = subprocess.run([sys.executable, os.path.join(root, "bin", "moe_cache.py"), "server", model, "--llama-server", ls, "--ctx", str(ctx), "--port", str(port)] + a,
                           capture_output=True, text=True, env=env, timeout=180)
        return r.returncode, (r.stdout or "") + "\n" + (r.stderr or "")
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def compare(pa, pb, allow):
    """-> (problems, notes)"""
    problems, notes = [], []
    for k in HARD:
        if pa.get(k) != pb.get(k):
            (notes if k in allow else problems).append("%s: arm A %s, arm B %s%s" % (k, pa.get(k), pb.get(k), " (declared with --allow)" if k in allow else " and it is not declared: the arms would not differ only in the switch under test"))
    for k in SOFT:
        if pa.get(k) != pb.get(k):
            notes.append("%s: arm A %s, arm B %s (soft: follows free memory, not compared)" % (k, pa.get(k), pb.get(k)))
    return problems, notes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--llama-server", required=True); ap.add_argument("--ctx", type=int, default=65536); ap.add_argument("--port", type=int, default=8099)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for x in "ab":
        ap.add_argument("--root-" + x, default=here); ap.add_argument("--args-" + x, default=""); ap.add_argument("--env-" + x, default="")
    ap.add_argument("--allow", default="", help="plan keys that differ on purpose: " + ",".join(HARD))
    ap.add_argument("--cache-seed", default="")
    a = ap.parse_args(argv)
    allow = {x.strip() for x in a.allow.split(",") if x.strip()}
    bad_allow = allow - set(HARD)
    if bad_allow:
        print("PLAN-CHECK FAIL: --allow names unknown keys %s (known: %s)" % (sorted(bad_allow), ",".join(HARD))); return 2
    plans = {}
    for x in "ab":
        code, text = dry_run(getattr(a, "root_" + x), a.model, a.llama_server, a.ctx, a.port, getattr(a, "args_" + x), getattr(a, "env_" + x), a.cache_seed)
        if code != 0:
            print("PLAN-CHECK FAIL: arm %s launcher --dry-run exited %s: %s" % (x.upper(), code, " | ".join(text.strip().split("\n")[-2:])[:300])); return 2
        plans[x] = parse_plan(text)
        if not plans[x]["has_server_line"]:
            print("PLAN-CHECK FAIL: arm %s: the dry run printed no llama-server command line (nothing to compare)" % x.upper()); return 2
        p = plans[x]
        print("PLAN arm %s: engine %s, load-mode %s, -np %s, -ub %s, -b %s, KV %s, --n-cpu-moe %s, -ngl %s, -c %s, device %s, cache-ram %s, GPU expert cache %s GiB, host expert cache %s" % (
            x.upper(), p["engine"], p["load_mode"], p["np"], p["ub"], p["batch"], p["kv"], p["n_cpu_moe"], p["ngl"], p["ctx"], p["device"], p["cache_ram"],
            p["gpu_cache_gib"] if p["gpu_cache_gib"] is not None else "not recorded", ("%s GiB" % p["moe_cache_gib"]) if p["moe_cache_gib"] is not None else "not recorded"))
    problems, notes = compare(plans["a"], plans["b"], allow)
    for n in notes:
        print("PLAN-NOTE", n)
    for p in problems:
        print("PLAN-FAIL", p)
    print("PLAN-CHECK", "FAIL" if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
