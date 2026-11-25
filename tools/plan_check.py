#!/usr/bin/env python3
"""plan_check.py: do the two arms of a pair run the SAME plan, apart from what they declare? (test-fixes 2026-11-17, critic report_testing.md fix 7)
Runs the launcher's `--plan-json` for arm A and arm B (no model load; the flag goes BEFORE any `--`) and compares the plan as data: engine (plugin or stock), load method (preload or path),
--load-mode, -np, -ub, -b, KV types, --n-cpu-moe, -ngl, -c, device, flash attention, -t, -C, -ot, --no-mmap, --mlock, and the environment variables that change speed
(GGML_OP_OFFLOAD_MIN_BATCH, GGML_VK_*, MOE_CACHE_* except log/record/path ones). Short and long flag forms are read alike (-ctk = --cache-type-k). Old launchers without --plan-json
fall back to parsing the `--dry-run` text (same as bin/moe_cache_bench.py). A difference that is not named in --allow is a FAIL (a void laptop A/B once differed in more than the switch
under test, and nothing compared the plans). Soft facts that follow free memory (--cache-ram, the expert-cache size) are only reported.
Called by conversation_bench_pair.sh --plan-check; usable alone:
  python3 tools/plan_check.py --model M.gguf --llama-server LS --ctx 65536 [--root-a DIR --args-a "..." --env-a "K=V ..."] [--root-b ...] [--allow load_mode,n_cpu_moe] [--cache-seed DIR | --seed-a DIR --seed-b DIR]
With --seed-a/--seed-b (the prepared, tuned caches of a --prep run) the check sees the tuned plans; without a seed it is PRE-TUNE (the launcher's stored auto-tune is not there yet) and says so.
Exit 0 = same plan (or only allowed differences), 1 = a difference that was not declared, 2 = a launcher dry run failed."""
import argparse, json, os, re, shutil, subprocess, sys, tempfile

HARD = ["engine", "load_method", "load_mode", "np", "ub", "batch", "kv", "n_cpu_moe", "ngl", "ctx", "device", "fa", "threads", "cpu_mask", "ot", "no_mmap", "mlock", "env", "plugin"]
SOFT = ["cache_ram", "gpu_cache_gib", "moe_cache_gib"]
NUM = re.compile(r"^-?\d+(\.\d+)?$")
# flag spellings -> plan key (the mapping of bin/moe_cache_monitor.py plan_from_args, plus the load flags)
ALIAS = {"ctk": "ctk", "cache-type-k": "ctk", "ctv": "ctv", "cache-type-v": "ctv", "c": "c", "ctx-size": "c", "ub": "ub", "ubatch-size": "ub", "b": "b", "batch-size": "b",
         "ngl": "ngl", "n-gpu-layers": "ngl", "gpu-layers": "ngl", "np": "np", "parallel": "np", "n-parallel": "np", "t": "t", "threads": "t", "C": "C", "cpu-mask": "C",
         "ot": "ot", "override-tensor": "ot", "fa": "fa", "flash-attn": "fa", "lm": "load-mode", "load-mode": "load-mode", "n-cpu-moe": "n-cpu-moe", "ncmoe": "n-cpu-moe",
         "dev": "device", "device": "device", "cache-ram": "cache-ram", "cram": "cache-ram", "no-mmap": "no-mmap", "mlock": "mlock", "moe-cache-mib": "moe-cache-mib"}
BOOL = {"no-mmap", "mlock"}
ENV_SPEED = re.compile(r"^(GGML_OP_OFFLOAD_MIN_BATCH|GGML_VK_.*|MOE_CACHE_.*)$")
ENV_SKIP = re.compile(r"LOG|RECORD|TRACE|PROFILE|GGUF|LIB$|DIR$|PATH$|^MOE_CACHE_(SIZE_GIB|RAM_GIB|PRELOAD|DISABLE)$")      # log/record/path settings do not change speed; the sizes follow free memory (soft); PRELOAD is the load method, compared on its own


def parse_flags(line_or_toks):
    toks = line_or_toks.split() if isinstance(line_or_toks, str) else list(line_or_toks)
    out, i = {}, 0
    while i < len(toks):
        t = str(toks[i])
        if t.startswith("-") and not NUM.match(t):
            key, val = t.lstrip("-"), None
            if t.startswith("--") and "=" in t:
                key, val = key.split("=", 1)
            key = ALIAS.get(key, key)
            if val is not None:
                out[key] = val
            elif key in BOOL:
                out[key] = True
            elif i + 1 < len(toks) and (not str(toks[i + 1]).startswith("-") or NUM.match(str(toks[i + 1]))):
                out[key] = str(toks[i + 1]); i += 1
            else:
                out[key] = True
        i += 1
    return out


def speed_env(pairs):
    """the environment words that change speed, as one sorted string; sizes that follow free memory are returned apart"""
    d = {k: v for k, v in pairs if ENV_SPEED.match(k) and not ENV_SKIP.search(k)}
    return ",".join("%s=%s" % kv for kv in sorted(d.items())) or "none"


def facts(fl, engine, method, env_pairs, has_server, text=""):
    s = lambda k: fl.get(k) if isinstance(fl.get(k), str) else None
    sizes = dict(env_pairs)
    m_ram = sizes.get("MOE_CACHE_SIZE_GIB")
    m_ram = float(m_ram) if m_ram and NUM.match(m_ram) else None
    mib = s("moe-cache-mib")
    off = str(dict(env_pairs).get("MOE_CACHE_DISABLE", "")) not in ("", "0")       # fix-batch-1: "plugin off" is the plan key `plugin` (on|off), declared with --plan-diff plugin; not an env word
    return {"plugin": ("off" if off else "on") if has_server else None, "has_server_line": has_server, "engine": engine, "load_method": method if has_server else None, "load_mode": s("load-mode") or ("default" if has_server else None),
            "np": s("np") or ("default" if has_server else None), "ub": s("ub"), "batch": s("b"),
            "kv": ("%s/%s" % (s("ctk") or "f16", s("ctv") or "f16")) if has_server else None,
            "n_cpu_moe": s("n-cpu-moe"), "ngl": s("ngl"), "ctx": s("c"), "device": s("device") or ("default" if has_server else None), "fa": s("fa"),
            "threads": s("t"), "cpu_mask": s("C"), "ot": s("ot"), "no_mmap": bool(fl.get("no-mmap")), "mlock": bool(fl.get("mlock")), "env": speed_env(env_pairs) if has_server else None,
            "cache_ram": s("cache-ram"), "gpu_cache_gib": round(int(mib) / 1024.0, 2) if mib and mib.isdigit() else None, "moe_cache_gib": m_ram}


def parse_json_plan(d, env_words=()):
    """the plan as data (bin/moe_cache.py to_dict: engine, method, argv, env, micro_batch, kv_cache, n_cpu_moe); the arm's own env words count too"""
    argv = [str(x) for x in d.get("argv") or []]
    fl = parse_flags(argv[1:] if argv else [])
    if d.get("micro_batch") and "ub" not in fl:
        fl["ub"] = str(d["micro_batch"])
    pairs = [(k, v) for k, v in d.get("env") or []] + list(env_words)
    if d.get("n_cpu_moe") not in (None, "") and "n-cpu-moe" not in fl:
        fl["n-cpu-moe"] = str(d["n_cpu_moe"])
    return facts(fl, d.get("engine"), d.get("method"), pairs, bool(argv))


def parse_plan(text, env_words=()):
    """FALLBACK for a launcher without --plan-json: the llama-server line and the notes of the `--dry-run` text"""
    text = text.replace("\r", "").replace("\x00", "")
    lines = text.split("\n")
    srv = [l for l in lines if re.search(r"llama-server(\.exe)?\"?\s", l) and re.search(r"\s-m\s", l)]
    fl = parse_flags(re.sub(r"^.*llama-server(\.exe)?\"?", "", srv[-1])) if srv else {}
    pairs = re.findall(r"(?:^|\s)([A-Z][A-Z0-9_]+)=(\S+)", srv[-1].split("llama-server")[0]) if srv else []
    plugin = any(k in ("GGML_BACKEND_PATH", "LD_PRELOAD", "MOE_CACHE_LIB") for k, _ in pairs) or "MOE_CACHE_SIZE_GIB=" in text
    method = "preload" if any(k == "LD_PRELOAD" for k, _ in pairs) else ("path" if plugin else None)
    if "ub" not in fl:
        m = re.search(r"(?:micro-batch -ub|n_ubatch\s*=)\s*(\d+)", text)
        if m:
            fl["ub"] = m.group(1)
    m_gpu = re.search(r"GPU expert cache of (\d+) MiB", text)
    if m_gpu and "moe-cache-mib" not in fl:
        fl["moe-cache-mib"] = m_gpu.group(1)
    return facts(fl, ("plugin" if plugin else "stock") if srv else None, method, list(pairs) + list(env_words), bool(srv))


def launch(root, model, ls, ctx, port, args, env_words, seed, flag):
    """one launcher call with `flag` (--plan-json or --dry-run) inserted BEFORE a `--` separator in the arm's arguments; XDG_CACHE_HOME is a copy of the seed (the tuned cache of a --prep run), never the real one"""
    a = args.split()
    if "--" in a:
        a = a[:a.index("--")] + [flag] + a[a.index("--"):]
    else:
        a = a + [flag]
    cache = tempfile.mkdtemp(prefix="plan_check_cache_")
    try:
        if seed:
            shutil.copytree(seed, cache, dirs_exist_ok=True, symlinks=True)
        env = dict(os.environ, XDG_CACHE_HOME=cache)
        env.update(dict(w.split("=", 1) for w in env_words.split() if "=" in w))
        r = subprocess.run([sys.executable, os.path.join(root, "bin", "moe_cache.py"), "server", model, "--llama-server", ls, "--ctx", str(ctx), "--port", str(port)] + a,
                           capture_output=True, text=True, env=env, timeout=180)
        return r.returncode, r.stdout or "", r.stderr or ""
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def get_plan(root, model, ls, ctx, port, args, env_words, seed):
    """-> (exit code, plan facts or None, error text, how it was read: 'plan-json' or 'dry-run text')"""
    ew = [tuple(w.split("=", 1)) for w in env_words.split() if "=" in w]
    code, out, err = launch(root, model, ls, ctx, port, args, env_words, seed, "--plan-json")
    if code == 0:
        try:
            d = json.loads(out)
            if isinstance(d, dict) and d.get("argv"):
                return 0, parse_json_plan(d, ew), "", "plan-json"
        except ValueError:
            pass
    code, out, err = launch(root, model, ls, ctx, port, args, env_words, seed, "--dry-run")      # a launcher from before --plan-json (or its JSON was unreadable)
    if code != 0:
        return code, None, " | ".join((out + "\n" + err).strip().split("\n")[-2:])[:300], ""
    return 0, parse_plan(out + "\n" + err, ew), "", "dry-run text"


def compare(pa, pb, allow):
    """-> (problems, notes)"""
    problems, notes = [], []
    if "engine" in allow:
        allow = set(allow) | {"load_method"}          # plugin vs stock engine differs in the load method by definition
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
        ap.add_argument("--root-" + x, default=here); ap.add_argument("--args-" + x, default=""); ap.add_argument("--env-" + x, default=""); ap.add_argument("--seed-" + x, default="")
    ap.add_argument("--allow", default="", help="plan keys that differ on purpose: " + ",".join(HARD))
    ap.add_argument("--cache-seed", default="", help="cache folder both arms start from (an arm's --seed-a/--seed-b wins)")
    a = ap.parse_args(argv)
    allow = {x.strip() for x in a.allow.split(",") if x.strip()}
    bad_allow = allow - set(HARD)
    if bad_allow:
        print("PLAN-CHECK FAIL: --allow names unknown keys %s (known: %s)" % (sorted(bad_allow), ",".join(HARD))); return 2
    plans, tuned = {}, True
    for x in "ab":
        seed = getattr(a, "seed_" + x) or a.cache_seed
        code, p, err, how = get_plan(getattr(a, "root_" + x), a.model, a.llama_server, a.ctx, a.port, getattr(a, "args_" + x), getattr(a, "env_" + x), seed)
        if code != 0:
            print("PLAN-CHECK FAIL: arm %s launcher dry run exited %s: %s" % (x.upper(), code, err)); return 2
        if not p["has_server_line"]:
            print("PLAN-CHECK FAIL: arm %s: the dry run printed no llama-server command line (nothing to compare)" % x.upper()); return 2
        plans[x] = p
        tuned = tuned and bool(seed)
        print("PLAN arm %s (read from %s): engine %s (%s), load-mode %s, -np %s, -ub %s, -b %s, KV %s, --n-cpu-moe %s, -ngl %s, -c %s, -fa %s, -t %s, -C %s, -ot %s, no-mmap %s, mlock %s, device %s, env %s, cache-ram %s, GPU expert cache %s, host expert cache %s" % (
            x.upper(), how, p["engine"], p["load_method"] or "no plugin", p["load_mode"], p["np"], p["ub"], p["batch"], p["kv"], p["n_cpu_moe"], p["ngl"], p["ctx"], p["fa"], p["threads"], p["cpu_mask"],
            p["ot"], p["no_mmap"], p["mlock"], p["device"], p["env"], p["cache_ram"],
            ("%s GiB" % p["gpu_cache_gib"]) if p["gpu_cache_gib"] is not None else "not recorded", ("%s GiB" % p["moe_cache_gib"]) if p["moe_cache_gib"] is not None else "not recorded"))
    print("PLAN-NOTE basis: %s" % ("TUNED plans (the arms' prepared caches were the seed)" if tuned else "PRE-TUNE plans: no prepared cache was given, so the launcher's stored auto-tune is not in these plans; it may still move -ub, KV or the cache size at run time"))
    problems, notes = compare(plans["a"], plans["b"], allow)
    for n in notes:
        print("PLAN-NOTE", n)
    for p in problems:
        print("PLAN-FAIL", p)
    print("PLAN-CHECK", "FAIL" if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
