#!/usr/bin/env python3
"""LAUNCHER-1244b (R19): the long-prompt mode (-ub 4096 + q8_0 KV) is a per-model planner rule and the plan says where the pick came from.
ON by default for a model bigger than the memory it may use (Flash-Next, PC 64k item 361: prompt +50.7 %, decode +5.3 %, one A B B A = forecast);
OFF by default for a model that fits in RAM (35B, PC 64k item 351: prompt -42 %, decode -4.5 %, one A B B A = forecast). Both: LAUNCHER-1236 row.
Model-free: a 35B header twin (sparse file; --ram 12 makes it 'bigger than RAM' the way Flash-Next is on the PC), fake machines, dry-runs. Linux / macOS only."""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(HERE, "tests"), os.path.join(HERE, "planner"), os.path.join(HERE, "bin")]
import launch_plan_test as lp   # noqa: E402
from twin35b import make_twin   # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


if os.name == "nt":
    print("long-prompt per-model checks: sparse header twin, Linux / macOS only")
    sys.exit(0)

FLASH = os.path.expanduser("~/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS.gguf")      # the real header when this machine has it (dev laptop); skipped otherwise

with tempfile.TemporaryDirectory() as tmp:
    twin = os.path.join(tmp, "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf")
    make_twin(twin)

    def plan(machine, model, opts, extra=()):
        r = lp.run_case(machine, ["--ctx", "65536"] + opts, list(extra), model, tmp)
        out = r["stdout"] + r["stderr"]
        cmd = next((ln for ln in out.splitlines() if "llama-server -m" in ln), "")
        g = lambda f: (cmd.split(f, 1)[1].split()[0] if f in cmd else None)     # noqa: E731
        return dict(ub=g("-ub "), ctk=g("-ctk "), out=out, on="long-prompt mode (the default" in out, off_line="long-prompt mode: off by default" in out)

    # 35B, fits in RAM: OFF, and the plan says why (item 351)
    for label, opts in (("PC card (stock path)", ["--vram", "15"]), ("8 GiB card (plugin for a partial offload)", ["--vram", "8"])):
        p = plan("linux", twin, opts)
        check(not p["on"] and p["ub"] == "2048" and p["ctk"] is None and p["off_line"] and "off by default for a model that fits in RAM" in p["out"],
              f"35B fits in RAM, {label}: long-prompt OFF (-ub {p['ub']}, KV {p['ctk'] or 'f16'}) and the plan says why")
    p = plan("linux", twin, ["--vram", "15", "--long-prompt"])
    check(p["ub"] == "4096" and p["ctk"] == "q8_0" and not p["off_line"], "35B: --long-prompt still asks for it (-ub 4096, q8_0 KV) and prints no 'off by default' line")
    p = plan("linux", twin, ["--vram", "15", "--no-long-prompt"])
    check(p["ub"] == "2048" and p["ctk"] is None and not p["off_line"], "35B: --no-long-prompt prints no 'off by default' line (the user chose)")
    p = plan("linux", twin, ["--vram", "15", "--exact"])
    check(p["ub"] == "2048" and p["ctk"] is None and not p["off_line"], "35B: --exact keeps it off without that line")
    check(plan("linux", twin, ["--vram", "15"])["ub"] == "2048", "35B again: the pick is stable")
    # model bigger than the memory it may use: ON
    p = plan("linux", twin, ["--vram", "15", "--ram", "12"])
    check(p["on"] and p["ub"] == "4096" and p["ctk"] == "q8_0" and "Flash-Next" in p["out"],
          f"model bigger than RAM (--ram 12): long-prompt ON by default (-ub {p['ub']}, KV {p['ctk']}), line cites its measurement")
    if os.path.exists(FLASH):
        for m in ("win",):
            p = plan(m, FLASH, [])
            check(p["on"] and p["ub"] == "4096" and not p["off_line"], f"Flash-Next on the PC machine model: long-prompt ON (-ub {p['ub']}); line cites its measurement")
        p = plan("win", FLASH, ["--no-long-prompt"])
        check(p["ub"] != "4096" and not p["on"], "Flash-Next --no-long-prompt: off")
    else:
        print("  skip Flash-Next real-header checks (model not on this machine)")

print("long-prompt per-model checks: " + (f"{len(fails)} failed" if fails else "all passed"))
sys.exit(1 if fails else 0)
