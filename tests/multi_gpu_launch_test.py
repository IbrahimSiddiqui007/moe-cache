#!/usr/bin/env python3
"""--multi-gpu (PROTOTYPE) through the real launcher on FAKE machines (no GPU, no model download: the synthetic MoE header of launch_plan_test).
   python3 tests/multi_gpu_launch_test.py
Checks: off by default and byte-identical (also on a 2-GPU machine); two cards give --device/-ts/--n-cpu-moe and the prototype line; an iGPU is left out;
one card, user flags, --vram and --gpu-cache paths leave the one-card plan; a Windows PC with 9070 XT + iGPU stays single-card;
Linux and Windows give the same plan for the same cards (only the command-line quoting differs)."""
import copy, os, re, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests"))
sys.path.insert(0, os.path.join(HERE, "planner"))
import launch_plan_test as lp         # noqa: E402
import multigpu                       # noqa: E402

GIB = 2**30
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

TWO = "Available devices:\n  CUDA0: NVIDIA RTX A (8188 MiB, 7600 MiB free)\n  CUDA1: NVIDIA RTX B (8188 MiB, 7600 MiB free)\n"
UNEQ = "Available devices:\n  CUDA0: NVIDIA RTX A (8188 MiB, 7000 MiB free)\n  CUDA1: NVIDIA RTX B (12288 MiB, 11000 MiB free)\n"
WIN_TWO = ("Available devices:\n  Vulkan0: AMD Radeon(TM) Graphics (512 MiB, 400 MiB free)\n  Vulkan1: AMD Radeon RX 9070 XT (16304 MiB, 15000 MiB free)\n"
           "  Vulkan2: AMD Radeon RX 9070 XT (16304 MiB, 15000 MiB free)\n")

def machine(base, listing, **kw):
    m = copy.deepcopy(lp.MACHINES[base]); m["listing"] = listing; m.update(kw); return m

def plan_line(r):
    m = re.search(r"^.*llama-server\S* -m .*$", r["stdout"], re.M)
    return m.group(0) if m else ""

def flag(line, name):
    m = re.search(re.escape(name) + r" (\S+)", line)
    return m.group(1) if m else None

with tempfile.TemporaryDirectory() as tmp:
    model = os.path.join(tmp, "synthetic-moe.gguf")
    lp.make_synthetic_model(model)
    lp.MACHINES["linux2"] = machine("linux", TWO)
    lp.MACHINES["linux_uneq"] = machine("linux", UNEQ)
    lp.MACHINES["win2"] = machine("win", WIN_TWO)
    run = lambda mname, opts, extra=(): lp.run_case(mname, ["--ram", "12", "--always-plugin"] + opts, list(extra), model, tmp)

    off = run("linux2", [])
    check("--device" not in plan_line(off), "default off: a 2-GPU machine gets no --device/-ts, as before")
    check("PROTOTYPE" not in off["stdout"], "default off: nothing printed about multi-GPU")
    one = run("linux", [])
    one_on = run("linux", ["--multi-gpu"])
    check(plan_line(one) == plan_line(one_on), "one card with --multi-gpu: the same command line as without")

    on = run("linux2", ["--multi-gpu"])
    ln = plan_line(on)
    check(flag(ln, "--device") == "CUDA0,CUDA1" and flag(ln, "-ts") is not None, f"two cards: --device CUDA0,CUDA1 -ts {flag(ln, '-ts')}")
    check("PROTOTYPE" in on["stdout"], "the launcher says it is a prototype")
    check("MOE_CACHE_GPU_DEV2=auto" in on["stdout"], "N3: the plugin gets MOE_CACHE_GPU_DEV2=auto when the plan uses two cards")
    check("MOE_CACHE_GPU_DEV2" not in off["stdout"] and "MOE_CACHE_GPU_DEV2" not in one_on["stdout"], "N3: not set without --multi-gpu, nor with one card")
    check(int(flag(ln, "--n-cpu-moe")) < int(flag(plan_line(off), "--n-cpu-moe")), "the second card keeps more expert layers on the GPU than the one-card plan")
    ts = [int(x) for x in flag(ln, "-ts").split(",")]
    check(sum(ts) == 41, f"-ts counts add up to layers + 1 (40 + 1): {ts}")

    un = plan_line(run("linux_uneq", ["--multi-gpu"]))
    check(flag(un, "--device") == "CUDA1,CUDA0", f"unequal cards: the larger card is listed first: {flag(un, '--device')}")

    for what, opts, extra in (("--n-cpu-moe", ["--n-cpu-moe", "20"], []), ("--vram", ["--vram", "8"], []), ("--gpu-cache", ["--vram", "8", "--gpu-cache", "auto"], []),
                              ("user -ts", [], ["-ts", "1,1"]), ("user --device", [], ["--device", "CUDA1"])):
        r = run("linux2", ["--multi-gpu"] + opts, extra)
        ln = plan_line(r)
        check("not applied" in r["stdout"] and ln.count("-ts") == (1 if what == "user -ts" else 0), f"{what}: the user's choice stands, the line says why")

    w = run("win2", ["--multi-gpu"])
    wl = plan_line(w)
    check(flag(wl, "--device") == "Vulkan1,Vulkan2", f"Windows: the 512 MiB iGPU is left out ({flag(wl, '--device')}) and the two real cards are used")
    check("left out: Vulkan0" in w["stdout"], "the iGPU is named in the GPU line")
    wo = run("win", ["--multi-gpu"])        # the standard Windows fake: 9070 XT + iGPU
    check("--device" not in plan_line(wo) and plan_line(wo) == plan_line(run("win", [])), "Windows 9070 XT + iGPU: one usable card, plan unchanged")
    check(flag(wl, "-ts") is not None and len(flag(wl, "-ts").split(",")) == 2, "Windows plan has two shares")

    # same cards, same plan on both systems (the differences allowed: quoting, Windows commit cap)
    wm = machine("win", WIN_TWO.replace("Vulkan", "CUDA"), commit=(200e9, 190e9))
    lp.MACHINES["win2b"] = wm
    lp.MACHINES["lin2b"] = machine("linux", WIN_TWO.replace("Vulkan", "CUDA"))
    a = plan_line(run("win2b", ["--multi-gpu"])); b = plan_line(run("lin2b", ["--multi-gpu"]))
    check(flag(a, "-ts") == flag(b, "-ts") and flag(a, "--n-cpu-moe") == flag(b, "--n-cpu-moe") and flag(a, "--device") == flag(b, "--device"),
          "same cards: Linux and Windows get the same --device, -ts and --n-cpu-moe")

# pure helpers
cards = multigpu.list_cards("Available devices:\n  CUDA0: A (8000 MiB, 7000 MiB free)\n  Vulkan0: A (8000 MiB, 7000 MiB free)\n  Vulkan1: B (4000 MiB, 3900 MiB free)\n")
check([c["dev"] for c in cards] == ["CUDA0"] or [c["dev"] for c in cards] == ["Vulkan0", "Vulkan1"], f"a card listed by two backends is not counted twice: {[c['dev'] for c in cards]}")
check(multigpu.list_cards("") == [], "empty device list: no cards")

# MG3: the check summary
import moe_cache as mc
lines = []
mc.gpu_summary(WIN_TWO, lambda k, m: lines.append((k, m)))
check(len(lines) == 1 and "9070 XT" in lines[0][1] and "left out: Vulkan0" in lines[0][1], f"moe-cache-check names the cards used and the iGPU left out: {lines}")
lines.clear(); mc.gpu_summary(lp.LINUX_LIST, lambda k, m: lines.append((k, m)))
check(lines == [], "moe-cache-check: one device lists nothing extra")

print("FAILED: " + "; ".join(fails) if fails else "all ok")
sys.exit(1 if fails else 0)
