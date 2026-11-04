#!/usr/bin/env python3
"""LAUNCHER-674: the launcher's list of tested llama.cpp commits (check / doctor) agrees with what CI tests (identity.yml, tested "yes") and with the plugin's
   MOE_CACHE_TESTED_GGML (src/moe_cache_plugin.cpp). No llama-server needed.
   python3 tests/tested_commits_test.py"""
import os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "planner"))
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

yml = open(os.path.join(HERE, ".github", "workflows", "identity.yml"), encoding="utf-8").read()
ci = {m.group(1)[:9] for m in re.finditer(r'sha: "([0-9a-f]{40})"[^}]*tested: "yes"', yml)}
check(len(ci) >= 4, f"identity.yml: {len(ci)} tested commits found")
check(ci == set(mc.TESTED_COMMITS), f"launcher TESTED_COMMITS {sorted(mc.TESTED_COMMITS)} == identity.yml tested rows {sorted(ci)}")

cpp = open(os.path.join(HERE, "src", "moe_cache_plugin.cpp"), encoding="utf-8").read()
plug = set()
for m in re.finditer(r'#define MOE_CACHE_TESTED_GGML "([^"]*)"', cpp):
    plug |= set(re.findall(r"@([0-9a-f]{7,})", m.group(1)))
check(len(plug) >= 2, f"plugin: {sorted(plug)} named by commit")
for c in sorted(plug):
    check(mc.tested_commit("commit " + c), f"plugin commit {c} is accepted by the launcher")
for c in mc.TESTED_COMMITS:
    check(mc.tested_commit("commit " + c) and mc.tested_commit(c + "0123456789abc"), f"launcher accepts {c} (short and long)")
check(not mc.tested_commit("commit abcdef1") and not mc.tested_commit(""), "an unknown commit and no commit are not accepted")
sys.exit(1 if fails else 0)
