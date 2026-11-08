#!/usr/bin/env python3
"""QUALITY-970: --approx safe|careful|aggressive. Tier-to-switch mapping, gate records and refusals, gate numbers equal to tools/quality_gate.py and docs/QUALITY_GATE.md,
canary logic (step-down, unreadable measurement) and the launcher's canary stage with a fake llama-perplexity. No model, no GPU.
   python3 tests/approx_test.py"""
import json, os, stat, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "tools")); sys.path.insert(0, os.path.join(HERE, "planner"))
import moe_cache_approx as ax  # noqa: E402
import launch_plan_test as lp  # noqa: E402
import quality_gate as qg  # noqa: E402
mc = lp.mc

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# --- the gate numbers are the contract's numbers ---
for lvl, (kmax, tmin, tol) in ax.GATE_LEVELS.items():
    check(qg.LEVELS[lvl][2:] == (kmax, tmin, tol), f"gate level {lvl} equals tools/quality_gate.py")
doc = open(os.path.join(HERE, "docs", "QUALITY_GATE.md"), encoding="utf-8").read()
for lvl, (kmax, tmin, tol) in ax.GATE_LEVELS.items():
    if lvl != "exact":
        check(f"| `{lvl}` |" in doc and f"| {kmax:.3f} | {int(tmin)} % | {tol} |" in doc, f"gate level {lvl} is in docs/QUALITY_GATE.md with the same numbers")

# --- resolve: mapping, refusal, off, unverified, user --fast ---
R = lambda tier, model="Qwen3.8-Flash-Next-IQ3_XXS.gguf", **k: ax.resolve(tier, model, records=k.pop("records", []), **k)
r = R("safe");        check((r["fast"], r["kv"], r["spec"]) == (None, True, True), "safe: 8-bit KV + own drafting, no expert skipping")
r = R("careful");     check((r["fast"], r["kv"], r["spec"]) == ("1", False, True), "careful: --fast (level 1) + drafting")
r = R("aggressive");  check((r["fast"], r["spec"]) == ("2", True), "aggressive: --fast=2 + drafting")
r = R("exact");       check((r["fast"], r["kv"], r["spec"], r["modes"]) == (None, False, False, []), "exact: nothing on")
check(all("NOT identical" in R(t)["lines"][0] for t in ("safe", "careful", "aggressive")), "every tier says plainly that answers are not identical")
check(all(R(t)["gate"] == ax.TIERS[t]["gate"] for t in ax.TIERS) and [R(t)["gate"] for t in ("safe", "careful", "aggressive")] == ["kv", "fast", "fewer"], "tier -> gate level: kv, fast, fewer")
try:
    R("safe", "unknown-model.gguf"); check(False, "an unknown model is refused")
except ax.ApproxError as e:
    check("no gate record" in str(e) and "--approx-unverified" in str(e), "an unknown model is refused and the message says how to go on")
r = R("safe", "unknown-model.gguf", unverified=True)
check(any("NO gate record" in l for l in r["lines"]) and r["kv"], "--approx-unverified runs it with a loud line")
r = R("safe", "unknown-model.gguf", off=["kv8"])
check(r["kv"] is False and r["spec"] is True, "--approx-off kv8 removes the mode (spec-exact needs no model record)")
try:
    R("safe", off=["nope"]); check(False, "unknown --approx-off refused")
except ax.ApproxError:
    check(True, "unknown --approx-off mode is refused")
rec = [{"match": "flash-next", "modes": {"kv8": {"pass": False}}}]
try:
    R("safe", records=rec); check(False, "a failed gate record refuses the mode")
except ax.ApproxError as e:
    check("FAILED" in str(e), "a failed gate record refuses the mode, even with --approx-unverified" if True else "")
try:
    R("safe", records=rec, unverified=True); check(False, "failed record also refuses with --approx-unverified")
except ax.ApproxError:
    check(True, "failed record also refuses with --approx-unverified")
r = R("careful", user_fast="2")
check(r["fast"] is None and any("wins" in l for l in r["lines"]), "your own --fast wins over the tier and is reported")
check(ax.mode_status("kv8", "Granite-1B.gguf", []) == "verified", "granite counts as verified (CI smoke guard)")

# --- canary logic ---
seq = {"aggressive": (0.09, 85.0), "careful": (0.03, 91.0), "safe": (0.004, 97.0)}
f, h, n = ax.canary_decide("aggressive", lambda t: seq[t])
check(f == "careful" and [x[3] for x in h] == [False, True] and n is None, "aggressive fails (KL 0.09 > 0.06), careful passes: steps down to careful")
f, h, n = ax.canary_decide("aggressive", lambda t: {"aggressive": (0.2, 99.0), "careful": (0.2, 99.0), "safe": (0.3, 99.0)}[t])
check(f == "exact" and "running exact" in n, "all tiers fail: exact")
f, h, n = ax.canary_decide("careful", lambda t: (0.03, 85.0))
check(f == "safe" or f == "exact", "top-1 below the gate fails even when KL is fine")
check(ax.canary_decide("careful", lambda t: (0.03, 85.0))[1][0][3] is False, "careful with top-1 85 % < 90 % does not pass")
f, h, n = ax.canary_decide("careful", lambda t: None)
check(f == "careful" and h == [] and "could not run" in n, "a canary that cannot run keeps the tier and says nothing was checked")
check(not ax.passes("kv", None, 99.0) and not ax.passes("kv", 0.01, None) and ax.passes("kv", 0.015, 95.0), "gate edges: unreadable never passes, limits are inclusive")
check(ax.canary_decide("exact", lambda t: 1 / 0)[0] == "exact", "exact needs no measurement")
txt = "Mean    KLD:   0.012345 ±   0.000100\nSame top p: 95.500000 ±  0.2\n"
check(ax.parse_ppl(txt) == (0.012345, 95.5) and ax.parse_ppl("nothing") == (None, None), "llama-perplexity output is parsed")
cmd = ["llama-server", "-m", "x.gguf", "-ngl", "99", "--n-cpu-moe", "20", "-ctk", "q8_0", "-ctv", "q8_0", "--no-mmap", "-c", "8192", "--port", "8080"]
check(ax.model_args_from_cmd(cmd) == ["-ngl", "99", "--n-cpu-moe", "20", "--no-mmap"], "placement flags are carried to the canary, context and port are not")
check(os.path.getsize(ax.CANARY_TEXT) > 1500, "the canary text exists")
a1, a2 = ax.canary_key("/nonexistent/m.gguf", "", "safe"), ax.canary_key("/nonexistent/m.gguf", "", "careful")
check(a1 != a2, "the stamp key differs per tier")

# --- the launcher ---
cmdline = lambda r: next((l for l in (r["stdout"] + r["stderr"]).split("\n") if " -m " in l and "llama-server" in l), "")
with tempfile.TemporaryDirectory() as t:
    m = os.path.join(t, "synthetic-moe.gguf"); lp.make_synthetic_model(m)
    base = ["--ram", "12", "--ctx", "65536", "--no-approx-canary"]
    off = lp.run_case("win", ["--ram", "12", "--ctx", "65536"], [], m, t)
    check("approx" not in off["stdout"].lower() and "NOT identical" not in off["stdout"], "no --approx: OFF, no approx line at all")
    ref = lp.run_case("win", ["--ram", "12", "--ctx", "65536"], [], m, t)
    check(cmdline(off) == cmdline(ref), "no --approx: the command is the default one")
    refused = lp.run_case("win", base + ["--approx", "safe"], [], m, t)
    check(refused["rc"] != 0 and "no gate record" in refused["stderr"], "synthetic model without a record: --approx safe refused")
    U = base + ["--approx-unverified"]
    safe = lp.run_case("win", U + ["--approx", "safe"], [], m, t)
    check(safe["rc"] == 0 and "-ctk q8_0 -ctv q8_0" in cmdline(safe) and "MOE_CACHE_ROUTE" not in safe["stdout"], "safe: q8_0 KV cache, no route skipping")
    check("--approx safe" in safe["stdout"] and "NOT identical" in safe["stdout"], "safe: the launcher says the answers are not identical")
    care = lp.run_case("win", U + ["--approx", "careful"], [], m, t)
    check('MOE_CACHE_ROUTE="skip"' in care["stdout"] and "-ctk q8_0" in cmdline(care), "careful: route skipping + q8_0 KV")
    agg = lp.run_case("win", U + ["--approx", "aggressive"], [], m, t)
    check('MOE_CACHE_ROUTE_DROP="4"' in agg["stdout"], "aggressive: --fast=2 (drop up to 4)")
    byhand = lp.run_case("win", base + ["--fast=2"], [], m, t)
    check('MOE_CACHE_ROUTE_DROP="4"' in byhand["stdout"] and 'MOE_CACHE_ROUTE="skip"' in byhand["stdout"], "--fast=2 by hand gives the same switches")
    own = lp.run_case("win", U + ["--approx", "careful", "--fast=kv"], [], m, t)
    check("wins over the tier" in own["stdout"] and "MOE_CACHE_ROUTE" not in own["stdout"], "--fast=kv given: wins, reported")
    for bad in ("--exact", "--stock"):
        r = lp.run_case("win", U + ["--approx", "safe", bad], [], m, t)
        check(r["rc"] != 0 and "does not go with" in r["stderr"], f"--approx safe {bad}: refused")
    r = lp.run_case("win", base + ["--approx-unverified"], [], m, t)
    check(r["rc"] != 0 and "need --approx" in r["stderr"], "--approx-unverified alone is refused")
    ex = lp.run_case("win", U + ["--approx", "exact"], [], m, t)
    check(ex["rc"] == 0 and "nothing approximate" in ex["stdout"], "--approx exact: nothing on")
    pj = lp.run_case("win", U + ["--approx", "careful"], [], m, t, flag="--plan-json")
    try:
        j = json.loads(pj["stdout"][pj["stdout"].index("{"):])
        check(j.get("approx", {}).get("tier") == "careful" and j["approx"]["gate"] == "fast", "--plan-json carries the approx block")
    except ValueError:
        check(False, "--plan-json carries the approx block")

    # --- the canary stage with a fake llama-perplexity (Linux/macOS shell script) ---
    if os.name != "nt":
        bindir = os.path.join(t, "fakebin"); os.makedirs(bindir)
        ls = os.path.join(bindir, "llama-server"); open(ls, "w").close()
        def fake_perp(kld_by_route, top1):
            p = os.path.join(bindir, "llama-perplexity")
            open(p, "w").write("#!/bin/sh\n[ -n \"$MOE_CACHE_DROP_TEST\" ] && :\n"
                               "case \" $* \" in *--kl-divergence\\ *|*\" --kl-divergence\") ;; *) exit 0;; esac\n"
                               f"if [ \"$MOE_CACHE_ROUTE_DROP\" = 4 ]; then echo 'Mean    KLD:   {kld_by_route[2]}'; elif [ \"$MOE_CACHE_ROUTE\" = skip ]; then echo 'Mean    KLD:   {kld_by_route[1]}'; else echo 'Mean    KLD:   {kld_by_route[0]}'; fi\n"
                               f"echo 'Same top p: {top1}'\n")
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
        class A: pass
        class P: pass
        def stage(tier, kl, top1, fast="default"):
            fake_perp(kl, top1)
            a = A(); a.approx = tier; a.no_approx_canary = False; a.fast = {"safe": "kv", "careful": "1", "aggressive": "2"}[tier] if fast == "default" else fast; a.approx_info = {"tier": tier, "canary": None}
            plan = P(); plan.a = a; plan.ls = ls; plan.lib = ""; plan.model = m; plan.cmd = [ls, "-m", m, "-ngl", "99"]; plan.env = [("MOE_CACHE_ROUTE", "skip")]
            old = mc.CACHE_DIR; mc.CACHE_DIR = os.path.join(t, "cache" + tier + str(kl)); out = []
            mc.say = lambda s, err=False: out.append(s)
            try:
                return mc.approx_canary(plan), a.approx_info, out
            finally:
                mc.CACHE_DIR = old
        final, info, out = stage("aggressive", (0.0, 0.03, 0.09), 90.0)
        check(final == "careful" and "PAST its gate" in " ".join(out), "canary stage: aggressive KL 0.09 is past the gate, steps down to careful (0.03 passes)")
        final, info, out = stage("careful", (0.0, 0.01, 0.09), 96.0)
        check(final == "careful" and info["canary"] == "passed", "canary stage: careful within its gate stays")
        final, info, out = stage("safe", (0.0, 0.01, 0.09), 80.0)
        check(final == "exact", "canary stage: top-1 80 % is past every gate: exact")
        final, info, out = stage("careful", (0.0, 0.5, 0.5), 50.0, fast="kv")
        check(final is None and "skipped" in info["canary"], "canary stage: your own --fast replaced the expert mode: no canary, said so")
        os.remove(os.path.join(bindir, "llama-perplexity"))
print("FAILED: " + "; ".join(fails) if fails else "approx_test: all passed")
sys.exit(1 if fails else 0)
