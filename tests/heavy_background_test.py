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
# busy programs: lines of the PowerShell counters -> only what is above the limits, our own and system programs left out
lines = "\n".join(["gpu\tEACefSubProcess\t3D\t28.6", "gpu\tEACefSubProcess\tCopy\t40", "gpu\tdwm\t3D\t9.9", "gpu\tllama-server\tCompute\t60", "gpu\tgame\tCompute\t3",
                   "mem\tgame\t2048", "mem\tsmall\t200", "cpu\tbrowser\t120", "cpu\tidle\t3000", "cpu\tquiet\t10", "junk line"])
got = [(i["name"], i["kind"]) for i in mc.parse_busy(lines)]
if got != [("EACefSubProcess", "gpu"), ("game", "mem"), ("browser", "cpu")]:
    ok = False
    print(f"FAIL: parse_busy: {got}")
note = mc.busy_note(mc.parse_busy(lines)[0])
if "23 %" not in note:
    ok = False
    print(f"FAIL: EA note: {note}")
note2 = mc.busy_note(mc.parse_busy(lines)[2])
if "not measured" not in note2 or "23 %" in note2:
    ok = False
    print(f"FAIL: other note: {note2}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
