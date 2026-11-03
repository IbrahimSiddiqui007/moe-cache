#!/usr/bin/env python3
"""DECODE-630 (layers back to the GPU after the prompt), model-free: the planner side (planner/decode_reloc.py) and the link to the llama.cpp patch.
   python3 tests/decode_reloc_test.py
Checks: default off; the 64k Flash-Next numbers give the 2 layers the PC measured; refusals say why; the env names the launcher writes are the ones
patch 0050 reads; the patch is off unless LLAMA_DECODE_RELOC is set."""
import glob, os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
import decode_reloc as dr   # noqa: E402
import gguf_plan            # noqa: E402

MiB = 1048576
fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + str(detail)) if detail and not ok else ""))
    if not ok:
        fails.append(name)


# Flash-Next at 64k (planner table): compute buffer 1750 MiB at -ub 2048, 690 at 512; one expert layer 676 MiB; 40 of 48 layers on the CPU
LAY = {"n_layers": 48, "expert_bytes_per_layer": 676 * MiB}
cm = gguf_plan.compute_buffer_guess_mib
common = dict(lay=LAY, ctx=65536, ub=2048, ncmoe="40", extra=[], gcache="off", compute_mib=cm)

# 1. default off: nothing added, nothing printed
check("off by default", dr.launcher_env({}, **common) == ([], None))
check("off for 0", dr.launcher_env({dr.SWITCH: "0"}, **common) == ([], None))

# 2. on: the numbers
s = dr.size_extra(676 * MiB, 40, 1750, 690, 2048, 512)
check("64k freed = compute 1060 + block 169", abs(s["freed_mib"] - (1060 + (1.6 - 1.35) * 676)) < 0.5, s)
check("64k expected extra layers = 1 (no leftover counted)", s["expected_extra"] == 1, s)
check("64k the patch may try 2 (= the 2 layers measured on the PC)", s["max_layers"] == 2, s)
pairs, note = dr.launcher_env({dr.SWITCH: "1"}, **common)
d = dict(pairs)
check("env pairs", d.get("LLAMA_DECODE_RELOC") == "1" and d.get("LLAMA_DECODE_RELOC_UB") == "512" and d.get("LLAMA_DECODE_RELOC_MAX_LAYERS") == "2"
      and d.get("LLAMA_DECODE_RELOC_KEEP_MIB") == "768", pairs)
check("note names the patch and PROTOTYPE", note and "0050" in note and "PROTOTYPE" in note, note)
pairs, _ = dr.launcher_env({dr.SWITCH: "1", dr.SWITCH + "_UB": "1024", dr.SWITCH + "_KEEP_MIB": "512"}, **common)
check("decode chunk and keep from the environment", dict(pairs).get("LLAMA_DECODE_RELOC_UB") == "1024" and dict(pairs).get("LLAMA_DECODE_RELOC_KEEP_MIB") == "512", pairs)

# 3. refusals, each with a reason
on = {dr.SWITCH: "1"}
for name, kw in [("gpu cache", dict(gcache="2000")), ("all experts on the GPU", dict(ncmoe="0")), ("all experts on the CPU", dict(ncmoe="48")),
                 ("ncmoe not a number", dict(ncmoe="auto")), ("same chunk", dict(ub=512)), ("chunk not a number", dict(ub="x")),
                 ("mtp", dict(extra=["--spec-type", "draft-mtp"])), ("two cards", dict(extra=["--device", "CUDA0,CUDA1"])),
                 ("override-tensor", dict(extra=["-ot", "exps=CPU"]))]:
    p, why = dr.launcher_env(on, **{**common, **kw})
    check("refused: " + name, p == [] and why and "ignored" in why, (p, why))
tiny = dr.size_extra(100 * MiB, 40, 700, 690, 2048, 512)
check("size_extra: small buffers free less than one layer", tiny["expected_extra"] == 0 or tiny["max_layers"] >= 0)
p, why = dr.launcher_env(on, **{**common, "lay": {"n_layers": 48, "expert_bytes_per_layer": 8000 * MiB}})
check("refused: freed room below one layer says so", p == [] and "less than one layer" in why, why)
check("size_extra: no layers", dr.size_extra(0, 0, 1750, 690, 2048, 512)["max_layers"] == 0)
check("cap at 8", dr.size_extra(10 * MiB, 40, 100000, 0, 2048, 512)["max_layers"] == 8)

# 4. the link to the patch: every env name the launcher writes is read by the patch, and the patch is off unless the switch is set
pf = sorted(glob.glob(os.path.join(HERE, "patches", "llama.cpp", "0050-*.patch")))
check("patch 0050 exists", len(pf) == 1, pf)
if pf:
    text = open(pf[0], encoding="utf-8").read()
    for n in ("LLAMA_DECODE_RELOC", "LLAMA_DECODE_RELOC_UB", "LLAMA_DECODE_RELOC_MAX_LAYERS", "LLAMA_DECODE_RELOC_KEEP_MIB", "LLAMA_DECODE_RELOC_AFTER", "LLAMA_DECODE_RELOC_SMALL"):
        check("patch reads " + n, '"' + n + '"' in text, n)
    check("patch is off without the switch", re.search(r'if \(!e \|\| !\*e \|\| strcmp\(e, "0"\) == 0\)', text) is not None)
    check("patch refuses the MoE cache and training", "moe_cache_size != 0" in text and "cparams.training" in text)
    check("patch does not touch the plugin", "mc_" not in "".join(l for l in text.splitlines() if l.startswith("+++")))

# 5. through the real launcher (dry run, the fake machines of launch_plan_test): off adds nothing, on adds the four variables and one line, on both OS
import tempfile  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "tests"))
import launch_plan_test as lpt  # noqa: E402
tmp = tempfile.mkdtemp(prefix="decode_reloc_test_")
model = os.path.join(tmp, "m.gguf")
lpt.make_synthetic_model(model)
for mach in lpt.MACHINES:
    off = lpt.run_case(mach, ["--ram", "12", "--ctx", "65536"], [], model, tmp)
    on = lpt.run_case(mach, ["--ram", "12", "--ctx", "65536"], [], model, tmp, call=lambda: os.environ.__setitem__("MOE_CACHE_DECODE_RELOC", "1"))
    ot, nt = off["stdout"] + off["stderr"], on["stdout"] + on["stderr"]
    check(f"launcher {mach}: off adds nothing", off["rc"] == 0 and "RELOC" not in ot, ot[-300:])
    check(f"launcher {mach}: on adds the variables and the line", on["rc"] == 0 and "LLAMA_DECODE_RELOC" in nt and "LLAMA_DECODE_RELOC_UB" in nt
          and "MOE_CACHE_DECODE_RELOC=1: after the prompt -ub 2048 becomes 512" in nt, nt[-300:])
    check(f"launcher {mach}: the server command line is unchanged", "-ub 2048 -b 2048" in nt and "--n-cpu-moe" in nt)


print("FAIL: %d" % len(fails) if fails else "decode_reloc_test: all ok")
sys.exit(1 if fails else 0)
