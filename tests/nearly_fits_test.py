#!/usr/bin/env python3
"""LAUNCHER-1236: a MoE model that nearly fits the GPU gets at least stock llama-server's plan.

PC evidence (RX 9070 XT 16 GB, Qwen3.6-35B-A3B UD-IQ3_XXS, 64k, 6-turn replay, b12207, apps closed, n=1 per arm = forecast): stock --n-cpu-moe 3 / -ub 2048 / f16 KV decoded
98.2 tok/s and read prompts at 2603 tok/s; the launcher's plugin plan (--n-cpu-moe 6, -ub 4096, q8_0 KV) 92.5 / 1859. Two causes, both fixed: (1) for a model that fits in RAM
the plugin ran and kept room for its long-prompt GPU block (3 more expert layers on the CPU on that card); (2) the long-prompt mode (-ub 4096 + q8_0 KV) was on by default
although it was measured only on a model far bigger than RAM (Flash-Next). Model-free: a header twin (sparse file), fake machines, dry-runs only. Linux / macOS only (sparse file)."""
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
    print("nearly-fits checks: the header twin is a 13 GB sparse file; these checks run on Linux and macOS only")
    sys.exit(0)

with tempfile.TemporaryDirectory() as tmp:
    twin = os.path.join(tmp, "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf")
    make_twin(twin)

    def plan(machine, opts, ctx=65536):
        r = lp.run_case(machine, ["--ctx", str(ctx)] + opts, [], twin, tmp)
        out = r["stdout"] + r["stderr"]
        cmd = next((ln for ln in out.splitlines() if "llama-server -m" in ln), "")
        g = lambda f: (cmd.split(f, 1)[1].split()[0] if f in cmd else None)     # noqa: E731
        return dict(ncmoe=g("--n-cpu-moe "), ub=g("-ub "), ctk=g("-ctk "), plugin="LD_PRELOAD" in cmd or "MOE_CACHE_GGUF" in cmd, out=out, cmd=cmd)

    # the PC: 16 GB RX 9070 XT, apps closed (about 15 GiB free); Windows fake machine with --vram
    p = plan("win", ["--vram", "15"])
    check(p["ncmoe"] == "3" and p["ub"] == "2048" and p["ctk"] is None and not p["plugin"] and "nearly fits the GPU" in p["out"],
          f"PC card, 35B at 64k: stock's plan (--n-cpu-moe {p['ncmoe']}, -ub {p['ub']}, KV {p['ctk'] or 'f16'}, plugin {p['plugin']}; stock ran 3 / 2048 / f16)")
    p = plan("linux", ["--vram", "15"])
    check(p["ncmoe"] == "3" and p["ub"] == "2048" and p["ctk"] is None and not p["plugin"],
          f"the same on Linux (--n-cpu-moe {p['ncmoe']}, -ub {p['ub']}, KV {p['ctk'] or 'f16'})")
    p = plan("linux", ["--vram", "14"])
    check(p["ub"] == "2048" and p["ctk"] is None and not p["plugin"] and p["ncmoe"] is not None and int(p["ncmoe"]) <= 7,
          f"--vram 14 (the brief's laptop check): stock plan, --n-cpu-moe {p['ncmoe']}, -ub {p['ub']}, KV {p['ctk'] or 'f16'}")
    p = plan("linux", ["--vram", "15", "--always-plugin"])
    check(p["plugin"] and "nearly fits the GPU" not in p["out"],
          f"--always-plugin keeps the plugin (--n-cpu-moe {p['ncmoe']}, -ub {p['ub']}, KV {p['ctk'] or 'f16'})")
    p = plan("linux", ["--vram", "15", "--long-prompt", "--always-plugin"])
    check(p["ub"] == "4096" and p["ctk"] == "q8_0", f"--long-prompt asked for still gives -ub 4096 + q8_0 KV (-ub {p['ub']}, KV {p['ctk']})")
    # a model far over the card (8 GiB): the plugin rule is unchanged (10 of 40 layers fit the GPU either way: within the executor's best measured gain)
    p = plan("linux", ["--vram", "8"])
    check(p["plugin"] and "nearly fits the GPU" not in p["out"], f"8 GiB card: the plugin still runs (--n-cpu-moe {p['ncmoe']} of 40 on the CPU)")
    p = plan("linux", ["--vram", "8"], ctx=32768)
    check(p["plugin"] and p["ub"] == "2048" and p["ctk"] is None,
          f"8 GiB card at 32k, model fits in RAM: no long-prompt default (-ub {p['ub']}, KV {p['ctk'] or 'f16'}, --n-cpu-moe {p['ncmoe']}; was 40 / 4096 / q8_0)")

print("nearly-fits checks: " + (f"{len(fails)} failed" if fails else "all passed"))
sys.exit(1 if fails else 0)
