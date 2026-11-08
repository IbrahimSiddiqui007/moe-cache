#!/usr/bin/env python3
"""The long-prompt mode (--long-prompt; the DEFAULT from 32k context since 2026-11-08, --no-long-prompt opts out): -ub 4096 with the q8_0 KV cache, only when the memory plan of the card leaves room for -ub 4096; otherwise nothing
changes. Fake machines (a Windows PC with a 16 GB card, a Linux box with an 8 GB card) and a synthetic MoE header, as tests/launch_plan_test.py; no model, no GPU.
   python3 tests/long_prompt_test.py"""
import os, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, os.path.join(HERE, "planner"))
import launch_plan_test as lp  # noqa: E402
import gguf_plan  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def cmd(r):
    return next((l for l in (r["stdout"] + r["stderr"]).split("\n") if " -m " in l and "llama-server" in l), "")

with tempfile.TemporaryDirectory() as t:
    m = os.path.join(t, "synthetic-moe.gguf"); lp.make_synthetic_model(m)
    base = ["--ram", "12", "--ctx", "65536"]
    plain = lp.run_case("win", base + ["--no-long-prompt"], [], m, t)
    dflt = lp.run_case("win", base, [], m, t)
    on = lp.run_case("win", base + ["--long-prompt"], [], m, t)
    check(" -ub 4096 " in cmd(dflt) + " " and "-ctk q8_0 -ctv q8_0" in cmd(dflt) and "long-prompt mode (the default" in dflt["stdout"], "16 GB card, 64k: the long-prompt mode is the DEFAULT: -ub 4096, the q8_0 KV cache, and the line says so")
    check(cmd(dflt) == cmd(on), "the default and --long-prompt give the same command at 64k")
    check("-ub 4096" not in cmd(plain) and "-ctk" not in cmd(plain) and "long-prompt mode" not in plain["stdout"], "--no-long-prompt gives the old setting (-ub 2048, f16 KV) and says nothing")
    check("fast=kv" not in dflt["stdout"] and "--fast=kv: q8_0" not in dflt["stdout"], "no second 'fast=kv' line: the long-prompt line says what happened")
    short = lp.run_case("win", ["--ram", "12", "--ctx", "8192"], [], m, t)
    check("-ub 4096" not in cmd(short) and "-ctk" not in cmd(short), "8k context: the default does not switch it on (prompts shorter than a chunk, q8_0 would cost exact answers for nothing)")
    short_on = lp.run_case("win", ["--ram", "12", "--ctx", "8192", "--long-prompt"], [], m, t)
    check("-ub 4096" in cmd(short_on) and "-ctk q8_0" in cmd(short_on), "8k context with --long-prompt: asked for, so it applies")
    small = lp.run_case("linux", base + ["--long-prompt"], [], m, t)
    plain_small = lp.run_case("linux", base + ["--no-long-prompt"], [], m, t)
    default_small = lp.run_case("linux", base, [], m, t)
    check("-ub 4096" not in cmd(small) and "-ctk" not in cmd(small) and "does not fit" in small["stdout"], "8 GB card: -ub 4096 does not fit, nothing changes and the --long-prompt line says why")
    check(cmd(default_small) == cmd(plain_small) and "does not fit" not in default_small["stdout"], "8 GB card, default: the normal plan, no warning line (the flag was not asked for)")
    ub_of = lambda r: (cmd(r).split(" -ub ")[1].split()[0] if " -ub " in cmd(r) else None)
    check(ub_of(small) == ub_of(plain_small), f"8 GB card: the micro-batch is the usual one ({ub_of(small)})")
    check(small["stdout"].count("other programs hold") <= plain_small["stdout"].count("other programs hold"), "the second plan does not repeat the 'other programs hold' line")
    own = lp.run_case("win", base + ["--ub", "2048"], [], m, t)
    check(" -ub 2048" in cmd(own) and "-ctk" not in cmd(own), "your own --ub wins and the long-prompt mode changes nothing")
    ex = lp.run_case("win", base + ["--long-prompt", "--exact"], [], m, t)
    check(ex["rc"] != 0 and "does not go with --exact" in ex["stderr"], "--exact and --long-prompt together are refused")
    ex2 = lp.run_case("win", base + ["--exact"], [], m, t)
    check(ex2["rc"] == 0 and "-ctk" not in cmd(ex2) and " -ub 2048" in cmd(ex2), "--exact alone: the default long-prompt mode stays off (stock-identical output)")
    st = lp.run_case("win", base + ["--stock"], [], m, t)
    check("-ctk" not in cmd(st), "--stock: no q8_0 KV cache")
    fast = lp.run_case("win", base + ["--long-prompt", "--fast=kv"], [], m, t)
    check(" -ub 4096 " in cmd(fast) + " " and "-ctk q8_0" in cmd(fast) and "changes nothing" not in fast["stdout"], "LAUNCHER-985: with --fast=kv given the long-prompt mode still applies (-ub 4096, same q8_0 KV cache)")

    # --no-long-prompt gives the f16 KV cache at 128k+ too (SERVER_HELP: "-ub 2048 and the f16 KV cache"); the 128k q8_0 default is for everyone else
    big = ["--ram", "12", "--ctx", "131072"]
    d128 = lp.run_case("win", big, [], m, t)
    n128 = lp.run_case("win", big + ["--no-long-prompt"], [], m, t)
    check("-ctk q8_0 -ctv q8_0" in cmd(d128), "128k, default: the q8_0 KV cache")
    check(n128["rc"] == 0 and "-ctk" not in cmd(n128) and " -ub 4096" not in cmd(n128) + " " and "q8_0 KV cache by default" not in n128["stdout"],
          "128k with --no-long-prompt: the f16 KV cache (no -ctk), -ub 2048 (SERVER_HELP)")
    check("-ctk" not in cmd(lp.run_case("win", big + ["--exact"], [], m, t)), "128k with --exact: still the f16 KV cache")
    # --long-prompt with --exact or --stock is an error whatever else is set (--ub / --fast used to hide the conflict)
    for opts in (["--exact"], ["--exact", "--ub", "2048"], ["--exact", "--fast=kv"], ["--stock", "--ub", "2048"]):
        r = lp.run_case("win", base + ["--long-prompt"] + opts, [], m, t)
        check(r["rc"] != 0 and "does not go with --exact or --stock" in r["stderr"], f"--long-prompt {' '.join(opts)}: refused")

    # server.id records the FINAL KV type (it was written before the long-prompt stage could undo the q8_0 cache, and before the f16 retry)
    import io, contextlib, types
    mc = lp.mc
    saved = (mc.CACHE_DIR, mc.KEY_FILE, mc.default_llama_server, mc.WINDOWS)
    for f in ("llama-server", "libggml-moe-cache.so"):
        open(os.path.join(t, f), "w").close()
    mc.CACHE_DIR = os.path.join(t, "cache"); mc.KEY_FILE = os.path.join(mc.CACHE_DIR, "api.key"); mc.WINDOWS = False
    mc.default_llama_server = lambda refresh=False: os.path.join(t, "llama-server")
    def plan_id(vram, free, name, *opts):
        a, extra, argv = mc.parse_server([m, "--llama-server", os.path.join(t, "llama-server"), "--ram", "12", "--no-pin", "--always-plugin", "--ctx", "65536", "--project", name, *opts])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            plan = mc.launch_plan(a, extra, argv, machine=lp.FakeMachine(t, vram, free))
        ident = open(os.path.join(mc.CACHE_DIR, "slots", name, "server.id")).read().strip()
        return plan, ident
    try:
        p16, id16 = plan_id(16384, 15000, "big")                      # room for -ub 4096: q8_0 stays
        p8, id8 = plan_id(8188, 7600, "small")                        # no room: the stage undid the q8_0 KV cache
        pn, idn = plan_id(16384, 15000, "off", "--no-long-prompt")
        q8, f16 = mc.model_identity(m, 65536, "q8_0", "q8_0"), mc.model_identity(m, 65536, None, None)
        check(p16.kv_q8 and "-ctk" in p16.cmd and id16 == q8, "server.id, long-prompt mode applied: records the q8_0 KV cache")
        check(not p8.kv_q8 and "-ctk" not in p8.cmd and id8 == f16 and id8 != q8, f"server.id, long-prompt mode undone (no room on an 8 GB card): records the f16 KV cache (kv_q8 {p8.kv_q8})")
        check("-ctk" not in pn.cmd and idn == f16, "server.id, --no-long-prompt: f16")
        # the f16 retry (llama.cpp abort at load) rewrites it: write_server_id is what the retry calls with the new command
        sd = os.path.join(mc.CACHE_DIR, "slots", "big")
        mc.write_server_id(sd, m, 65536, [x for x in p16.cmd if x not in ("-ctk", "-ctv", "q8_0")])
        check(open(os.path.join(sd, "server.id")).read().strip() == f16, "server.id after the f16 KV retry: rewritten to the f16 identity")
    finally:
        mc.CACHE_DIR, mc.KEY_FILE, mc.default_llama_server, mc.WINDOWS = saved

free = 15.4e9
P = lp.__dict__.get("MODEL_FOR_TUNE")
tn = gguf_plan.tune.__doc__ or ""
check("long_prompt" in tn, "the tuner documents the long_prompt switch")
print("FAILED: " + "; ".join(fails) if fails else "long_prompt_test: all passed")
sys.exit(1 if fails else 0)
