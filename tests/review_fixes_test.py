#!/usr/bin/env python3
"""Fixes from the outside review of 2026-11-18 (handover/sessions/main4_2026-11-17/files/glm_review/REVIEW.md): findings 1-2 (`--flag=value` after `--`), 8 (KV undo),
21 (--max-ram edge cases), 22 (--port range), 23 (--override-tensor). No model, no llama-server.   python3 tests/review_fixes_test.py"""
import argparse, os, sys, types
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# findings 1-2: one reading of the flags after `--`
sp = mc.split_equals_flags
check(sp(["--host=0.0.0.0", "--cache-ram=512", "-ngl=1", "--alias", "a=b", "-ot", "x=CPU"]) == ["--host", "0.0.0.0", "--cache-ram", "512", "-ngl=1", "--alias", "a=b", "-ot", "x=CPU"],
      "split_equals_flags: long flags split, short flags and plain values (with =) untouched")
check(sp(["--chat-template-kwargs={\"a\":1}", "--override-kv=x=int:3"]) == ["--chat-template-kwargs", "{\"a\":1}", "--override-kv", "x=int:3"], "split_equals_flags: only the first = splits; JSON values survive")
check(sp(sp(["--host=1"])) == ["--host", "1"] and sp([]) == [], "split_equals_flags: idempotent, empty ok")
ex = sp(["--cache-ram=512", "--api-key=KEY", "-ngl", "0", "--n-gpu-layers=7"])
check(mc.arg_after(ex, "--cache-ram", "-cram") == "512" and mc.arg_after(ex, "--n-gpu-layers") == "7", "arg_after sees the equals form after the split (cache-ram, n-gpu-layers)")
check(mc.take_api_key_from_args(ex) == "KEY" and "KEY" not in ex and "--api-key" not in ex, "take_api_key_from_args: the key leaves the argument list")
a, extra, argv = mc.parse_server(["m.gguf", "--", "--host=0.0.0.0", "-c", "4096"])
check(extra == ["--host", "0.0.0.0", "-c", "4096"], "parse_server hands the flags after -- over in the split form")

# finding 21: --max-ram
p = lambda *v: mc.parse_server(["m.gguf", *v])[0].max_ram
check(p("--max-ram", "0") == "0" and p("--max-ram", "0.0") == "0" and p("--max-ram", "0.00") == "0" and p("--max-ram", "20") == "20", "--max-ram: 0, 0.0, 0.00 all mean no cap")
for bad in ("inf", "nan", "-1"):
    try:
        p("--max-ram", bad); check(False, f"--max-ram {bad} refused")
    except SystemExit:
        check(True, f"--max-ram {bad} refused")
mc._TL.max_ram = "inf"
os.environ.pop("MOE_CACHE_MAX_RAM_GB", None)
mc.mem_cap.warned = True
m_inf = mc.mem_cap()
mc._TL.max_ram = None
check(m_inf is None or m_inf != float("inf"), "mem_cap: inf never becomes an infinite cap (default cap used)")

# finding 22: --port
for ok in ("1", "8080", "65535"):
    check(mc.parse_server(["m.gguf", "--port", ok])[0].port == int(ok), f"--port {ok} accepted")
for bad in ("0", "70000", "-5", "http"):
    try:
        mc.parse_server(["m.gguf", "--port", bad]); check(False, f"--port {bad} refused")
    except SystemExit:
        check(True, f"--port {bad} refused")

# finding 23: --override-tensor is a custom RAM layout like -ot
class M:
    windows = False
    def mem_for_server(self, r): return 64e9
    def mem_total(self): return 64e9
    def mem_cap(self): return None
a_ = types.SimpleNamespace(exact=False, stock=False, ram=None)
import tempfile
mp = os.path.join(tempfile.mkdtemp(prefix="moe-rf-"), "m.gguf"); open(mp, "wb").write(b"x")
check(mc.load_mode_cache_none(a_, ["ls", "-ot", "x=CPU"], [], M(), mp, 0) is None and mc.load_mode_cache_none(a_, ["ls", "--override-tensor", "x=CPU"], [], M(), mp, 0) is None,
      "load_mode_cache_none: -ot and --override-tensor both keep the user's layout (None)")

# finding 8: the long-prompt fallback removes the KV pair the launcher added, whatever its types
def lp(types_):
    a = types.SimpleNamespace(long_prompt=True, lp_explicit=False, lp_kv=True, ub="auto", approx_kv=False, lp_with_fast=False, fast="kv", dense_kv_types=types_)
    extra = ["-t", "8", "-ctk", types_[0] if types_ else "q8_0", "-ctv", types_[1] if types_ else "q8_0"]
    old = mc.tune_launch
    mc.tune_launch = lambda *args, **kw: ({"ub": 2048}, {}, None)
    try:
        mc.long_prompt_stage(a, extra, "ls", "m", 65536, None, None, None, True, 0, {"ub": 2048}, {}, None)
    finally:
        mc.tune_launch = old
    return extra
check(lp(None) == ["-t", "8"], "long-prompt fallback: q8_0/q8_0 pair undone")
check(lp(("q8_0", "q4_0")) == ["-t", "8"], "long-prompt fallback: a dense K q8_0 / V q4_0 pair is undone too (finding 8)")

import moe_cache_on_demand as od  # noqa: E402
check(od.model_alias("/m/big.gguf", ["x", "--", "--alias=foo"]) == "foo" and od.model_alias("/m/big.gguf", ["x", "--", "--alias", "foo"]) == "foo", "on-demand routes by --alias=foo like --alias foo")

print("PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
