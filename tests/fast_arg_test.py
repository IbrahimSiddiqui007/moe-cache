#!/usr/bin/env python3
"""LAUNCHER-004: `moe-cache-server --fast MODEL` must not read the model path as the value of --fast. No model, nothing started.
   python3 tests/fast_arg_test.py"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin"))
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def parse(*argv):
    a, extra, _ = mc.parse_server(list(argv))
    return a.fast, a.model, extra

check(parse("--fast", "m.gguf") == ("1", "m.gguf", []), "--fast MODEL: level 1, model kept")
check(parse("m.gguf", "--fast") == ("1", "m.gguf", []), "MODEL --fast: level 1")
check(parse("--fast", "C:\\models\\big model.gguf")[:2] == ("1", "C:\\models\\big model.gguf"), "--fast with a Windows path")
for lvl in ("1", "2", "kv", "fewer"):
    check(parse("--fast", lvl, "m.gguf") == (lvl, "m.gguf", []), f"--fast {lvl} MODEL still takes the level")
    check(parse(f"--fast={lvl}", "m.gguf")[:2] == (lvl, "m.gguf"), f"--fast={lvl} MODEL")
check(parse("m.gguf")[0] is None, "no --fast: off")
check(parse("--fast", "--", "-x") == ("1", None, ["-x"]), "--fast before --")
a, _, argv = mc.parse_server(["--fast", "m.gguf"])
check(argv == ["--fast", "m.gguf"], "the launcher's own argument list is returned unchanged")
sys.exit(1 if fails else 0)
