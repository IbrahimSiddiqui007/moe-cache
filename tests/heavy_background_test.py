#!/usr/bin/env python3
# The launcher names game launchers from tasklist output and stays quiet without them.
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("moe_cache", os.path.join(HERE, "..", "bin", "moe_cache.py"))
mc = importlib.util.module_from_spec(spec)
sys.modules["moe_cache"] = mc
spec.loader.exec_module(mc)

row = '"{}","1234","Console","1","45,000 K"'
ok = True
for text, want in [("", []), (row.format("notepad.exe"), []), (row.format("EADesktop.exe"), ["EA app"]), ("\n".join([row.format("steam.exe"), row.format("Battle.net.exe"), row.format("upc.exe")]), ["Battle.net", "Ubisoft Connect"]), ('INFO: No tasks are running', [])]:
    got = mc.heavy_background_apps(text)
    if got != want:
        ok = False
        print(f"FAIL: {text!r}: {got} != {want}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
