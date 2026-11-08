#!/usr/bin/env python3
"""LAUNCHER-984: the table of token-changing modes (TOKEN_MODES in bin/moe_cache_approx.py). The default launch enables exactly the rows whose status is "won"; --exact enables none and
gives stock-identical settings; the old --fast / --approx names still work. Fake machine and a synthetic MoE header (as tests/long_prompt_test.py); no model, no GPU.
   python3 tests/token_modes_test.py"""
import os, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "planner"))
import launch_plan_test as lp  # noqa: E402
import moe_cache_approx as ax  # noqa: E402
import gguf_plan  # noqa: E402
mc = lp.mc

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# --- the table ---
ids = [m["id"] for m in ax.TOKEN_MODES]
check(len(ids) == len(set(ids)), "mode ids are unique")
check(all(m["status"] in ("won", "lost", "unmeasured") and m["reason"] and m["what"] and m["switch"] for m in ax.TOKEN_MODES), "every row has a status (won / lost / unmeasured), a reason, a description and its switch")
check(ax.default_ids() == ["kv8", "long-prompt", "mtp"], f"today's won rows: q8_0 KV from 128k, long-prompt from 32k, MTP in agent mode ({ax.default_ids()})")
check(all(m["status"] == "unmeasured" for m in ax.TOKEN_MODES if m["id"] in ("fast1", "fast2", "fewer", "approx-tiers")), "expert skipping and the tiers are unmeasured, so off by default")

def cmd(r):
    return next((l for l in (r["stdout"] + r["stderr"]).split("\n") if " -m " in l and "llama-server" in l), "")
def on(r):
    c = cmd(r)
    return {"kv": "-ctk q8_0 -ctv q8_0" in c, "ub4096": " -ub 4096 " in c + " ", "route": 'MOE_CACHE_ROUTE="skip"' in r["stdout"], "drop": "MOE_CACHE_ROUTE_DROP" in r["stdout"], "fewer": "MOE_CACHE_FEWER" in r["stdout"]}

with tempfile.TemporaryDirectory() as t:
    m = os.path.join(t, "synthetic-moe.gguf"); lp.make_synthetic_model(m)
    run = lambda *o: lp.run_case("win", ["--ram", "12", *o], [], m, t)
    d64, d128, d8 = run("--ctx", "65536"), run("--ctx", "131072"), run("--ctx", "8192")
    check(on(d64)["kv"] and on(d64)["ub4096"] and not on(d64)["route"], "default at 64k: long-prompt (q8_0 KV, -ub 4096), no expert skipping")
    check(on(d128)["kv"], "default at 128k: q8_0 KV cache")
    check(not on(d8)["kv"] and not on(d8)["ub4096"], "default at 8k: nothing token-changing")
    for ctx in ("8192", "65536", "131072"):
        ex = run("--ctx", ctx, "--exact")
        o = on(ex)
        check(ex["rc"] == 0 and not any(o.values()), f"--exact at {ctx}: no q8_0 KV, no -ub 4096, no expert skipping (stock-identical settings)")
    check(cmd(run("--ctx", "65536", "--exact")) == cmd(run("--ctx", "65536", "--exact", "--no-long-prompt")), "--exact gives the same command as --exact --no-long-prompt")
    for bad in ("--fast", "--mtp"):
        r = run("--ctx", "65536", "--exact", bad)
        check(r["rc"] != 0 and "does not go with" in r["stderr"], f"--exact {bad}: refused (exact means every mode off)")
    r = run("--ctx", "65536", "--exact", "--approx", "safe", "--approx-unverified")
    check(r["rc"] != 0, "--exact --approx safe: refused")


    # --- LAUNCHER-985: a token mode only adds its own environment on top of the normal plan (long-prompt -ub / KV, same layers) ---
    import re
    def split(r):
        c = cmd(r)
        env = dict(re.findall(r'\$env:(\w+)="([^"]*)"', c))
        return env, c.split("& ", 1)[1] if "& " in c else c
    base_env, base_argv = split(d64)
    check(" -ub 4096 " in base_argv + " " and "--n-cpu-moe" in base_argv, "reference: the default plan at 64k has -ub 4096 and q8_0 KV")
    EXTRA_OK = {"MOE_CACHE_ROUTE", "MOE_CACHE_ROUTE_DROP", "MOE_CACHE_FEWER", "MOE_CACHE_GPU_MIN_TOKENS"}
    for name, opts in (("--fast", ["--fast"]), ("--fast=2", ["--fast=2"]), ("--fast=fewer", ["--fast=fewer"]), ("--fast=kv", ["--fast=kv"]),
                       ("--approx careful", ["--approx", "careful", "--approx-unverified", "--no-approx-canary"]),
                       ("--approx aggressive", ["--approx", "aggressive", "--approx-unverified", "--no-approx-canary"]),
                       ("--approx safe", ["--approx", "safe", "--approx-unverified", "--no-approx-canary"])):
        r = run("--ctx", "65536", *opts)
        env, argv = split(r)
        extra_keys = set(env) - set(base_env)
        check(r["rc"] == 0 and argv == base_argv and extra_keys <= EXTRA_OK and all(env.get(k) == v for k, v in base_env.items()),
              f"{name} at 64k: the same command as the default plan (-ub 4096, q8_0 KV, same layers) plus only its own environment {sorted(extra_keys)}")
        check("changes nothing" not in r["stdout"], f"{name}: no 'changes nothing' line")
    check(split(run("--ctx", "65536", "--fast"))[0].get("MOE_CACHE_ROUTE") == "skip" and split(run("--ctx", "65536", "--fast=2"))[0].get("MOE_CACHE_ROUTE_DROP") == "4"
          and split(run("--ctx", "65536", "--fast=fewer"))[0].get("MOE_CACHE_FEWER") == "2", "each mode's own environment is there (skip, drop 4, fewer 2)")
    low = lp.run_case("linux", ["--ram", "12", "--ctx", "65536", "--fast"], [], m, t)
    check(low["rc"] == 0 and "-ub 4096" not in cmd(low) and "-ctk q8_0" in cmd(low), "8 GB card, --fast: -ub 4096 does not fit, the 8-bit KV cache of --fast stays with the usual micro-batch")
    check(not any(on(run("--ctx", "65536", "--exact")).values()), "--exact is still all off")

    # the old names still work (aliases that force the mode on)
    f1 = run("--ctx", "8192", "--fast"); f2 = run("--ctx", "8192", "--fast=2")
    check(on(f1)["route"] and on(f1)["kv"] and on(f2)["drop"], "--fast and --fast=2 still switch their modes on")
    ap = run("--ctx", "8192", "--approx", "careful", "--approx-unverified", "--no-approx-canary")
    check(on(ap)["route"], "--approx careful still works")
    check("DEPRECATED" in mc.SERVER_HELP.split("--fast ")[1][:60] and "--approx safe|careful|aggressive  DEPRECATED" in mc.SERVER_HELP, "--fast and --approx are marked deprecated in the help")

    # the default follows the table: flip a row and the default follows
    try:
        ax.token_mode("kv8")["status"] = "lost"
        check(not on(run("--ctx", "131072", "--no-long-prompt"))["kv"] and on(d64)["kv"], "kv8 marked lost: no q8_0 KV by default at 128k")
        ax.token_mode("long-prompt")["status"] = "lost"
        check(not on(run("--ctx", "65536"))["kv"] and not on(run("--ctx", "65536"))["ub4096"], "long-prompt marked lost: not on by default at 64k")
    finally:
        ax.token_mode("kv8")["status"] = "won"; ax.token_mode("long-prompt")["status"] = "won"
    try:
        ax.token_mode("fast1")["status"] = "won"
        w = run("--ctx", "8192"); wex = run("--ctx", "8192", "--exact")
        check(on(w)["route"] and not on(wex)["route"], "fast1 marked won: on by default; --exact turns it off")
        ax.token_mode("fast2")["status"] = "won"
        check(on(run("--ctx", "8192"))["drop"], "fast2 marked won as well: the stronger level is the default")
    finally:
        ax.token_mode("fast1")["status"] = "unmeasured"; ax.token_mode("fast2")["status"] = "unmeasured"
    check(cmd(run("--ctx", "65536")) == cmd(d64), "the table is back to today's state after the flips")

# --- MTP: default in agent mode follows the table; --exact turns it off ---
gguf_plan_mtp, mc_ls_help = gguf_plan.mtp_layers, mc.ls_help
gguf_plan.mtp_layers = lambda model: (1, 1); mc.ls_help = lambda ls: "--spec-type draft-mtp"
mc.say = lambda s, err=False: None
try:
    check(mc.mtp_args(None, True, "ls", "m.gguf", []) == mc.MTP_ARGS, "MTP head + agent mode: on by default (row mtp is won)")
    check(mc.mtp_args(None, True, "ls", "m.gguf", [], exact=True) == [], "--exact: no MTP")
    check(mc.mtp_args(None, False, "ls", "m.gguf", []) == [], "no agent mode: MTP stays off by default")
    ax.token_mode("mtp")["status"] = "lost"
    check(mc.mtp_args(None, True, "ls", "m.gguf", []) == [] and mc.mtp_args(True, False, "ls", "m.gguf", []) == mc.MTP_ARGS, "mtp marked lost: off by default, --mtp still asks for it")
finally:
    ax.token_mode("mtp")["status"] = "won"
    gguf_plan.mtp_layers, mc.ls_help = gguf_plan_mtp, mc_ls_help
print("FAILED: " + "; ".join(fails) if fails else "token_modes_test: all passed")
sys.exit(1 if fails else 0)
