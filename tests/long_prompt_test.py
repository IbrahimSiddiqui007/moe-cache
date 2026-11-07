#!/usr/bin/env python3
"""The opt-in long-prompt mode (--long-prompt, v0.3.8): -ub 4096 with the q8_0 KV cache, only when the memory plan of the card leaves room for -ub 4096; otherwise nothing
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
    plain = lp.run_case("win", base, [], m, t)
    on = lp.run_case("win", base + ["--long-prompt"], [], m, t)
    check(" -ub 4096 " in cmd(on) + " " and "-ctk q8_0 -ctv q8_0" in cmd(on) and "long-prompt" in on["stdout"], "16 GB card: --long-prompt gives -ub 4096 and the q8_0 KV cache, and says so")
    check("-ub 4096" not in cmd(plain) and "-ctk" not in cmd(plain), "without the flag nothing changes (the default stays)")
    check("fast=kv" not in on["stdout"] and "--fast=kv: q8_0" not in on["stdout"], "no second 'fast=kv' line: the long-prompt line says what happened")
    small = lp.run_case("linux", base + ["--long-prompt"], [], m, t)
    plain_small = lp.run_case("linux", base, [], m, t)
    check("-ub 4096" not in cmd(small) and "-ctk" not in cmd(small) and "does not fit" in small["stdout"], "8 GB card: -ub 4096 does not fit, nothing changes and the line says why")
    ub_of = lambda r: (cmd(r).split(" -ub ")[1].split()[0] if " -ub " in cmd(r) else None)
    check(ub_of(small) == ub_of(plain_small), f"8 GB card: the micro-batch is the usual one ({ub_of(small)})")
    check(small["stdout"].count("other programs hold") <= plain_small["stdout"].count("other programs hold"), "the second plan does not repeat the 'other programs hold' line")
    own = lp.run_case("win", base + ["--long-prompt", "--ub", "2048"], [], m, t)
    check(" -ub 2048" in cmd(own) and "-ctk" not in cmd(own) and "your own --ub wins" in own["stdout"], "your own --ub wins and the long-prompt mode changes nothing")
    ex = lp.run_case("win", base + ["--long-prompt", "--exact"], [], m, t)
    check(ex["rc"] != 0 and "does not go with --exact" in ex["stderr"], "--exact and --long-prompt together are refused")
    fast = lp.run_case("win", base + ["--long-prompt", "--fast=kv"], [], m, t)
    check("-ub 4096" not in cmd(fast) and "-ctk q8_0" in cmd(fast) and "changes nothing" in fast["stdout"], "with --fast already given the long-prompt mode adds nothing")

free = 15.4e9
P = lp.__dict__.get("MODEL_FOR_TUNE")
tn = gguf_plan.tune.__doc__ or ""
check("long_prompt" in tn, "the tuner documents the long_prompt switch")
print("FAILED: " + "; ".join(fails) if fails else "long_prompt_test: all passed")
sys.exit(1 if fails else 0)
