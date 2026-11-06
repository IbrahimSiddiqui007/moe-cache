#!/usr/bin/env python3
"""GUI-697: the page's "llama.cpp not found" box must not run any file named llama-server. A script, a file outside the usual folders and a
file other users can write to are refused before --version is run. Fake files only.
   python3 tests/llama_path_test.py"""
import os, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_ui as ui  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

home = os.path.realpath(os.path.expanduser("~"))
d = tempfile.mkdtemp(prefix="moe-llpath-", dir=home)
def make(name, data, mode=0o755):
    p = os.path.join(d, name); open(p, "wb").write(data); os.chmod(p, mode); return p

check(ui.llama_path_refusal(make("elf", b"\x7fELF" + b"\0" * 16)) is None, "an ELF program under the home folder is allowed")
check(ui.llama_path_refusal(make("pe.exe", b"MZ" + b"\0" * 16)) is None, "a PE program is allowed")
check("compiled" in (ui.llama_path_refusal(make("script", b"#!/bin/sh\necho llama version\n")) or ""), "a shell script named llama-server is refused")
if os.name == "posix":
    check("write" in (ui.llama_path_refusal(make("wide", b"\x7fELF" + b"\0" * 16, 0o777)) or ""), "a program other users can write to is refused")
    out = tempfile.mkdtemp(prefix="moe-llpath-out-")
    p = os.path.join(out, "llama-server"); open(p, "wb").write(b"\x7fELF" + b"\0" * 16); os.chmod(p, 0o755)
    if not os.path.realpath(out).startswith(home + os.sep):
        check("outside" in (ui.llama_path_refusal(p) or ""), "a program outside home, program and kit folders is refused")
    os.remove(p); os.rmdir(out)
check("cannot be read" in (ui.llama_path_refusal(os.path.join(d, "missing")) or ""), "a missing file is refused, not a crash")

for f in os.listdir(d):
    os.remove(os.path.join(d, f))
os.rmdir(d)
print("FAILED: " + "; ".join(fails) if fails else "llama_path_test: all passed")
sys.exit(1 if fails else 0)
