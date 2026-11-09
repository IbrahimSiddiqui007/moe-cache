#!/usr/bin/env python3
"""LAUNCHER-850: a dense model runs out of GPU memory at load (a game or browser took memory after the plan was made): the launcher plans one step smaller from the free
memory measured again, and starts once more. Synthetic dense GGUF, fake free-memory figures, no GPU, no model.
   python3 tests/dense_oom_test.py"""
import os, re, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, os.path.join(HERE, "planner")); sys.path.insert(0, os.path.join(HERE, "bin"))
import dense_kv_fit_test as dk    # noqa: E402,F401  (its checks run on import; its make_dense builds the model)
import moe_cache as mc            # noqa: E402

GIB = 2**30
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def layers_of(cmd):
    m = re.search(r"blk\\\.\(([0-9|]+)\)", cmd[cmd.index("-ot") + 1])
    return len(m.group(1).split("|")) if m else 0

with tempfile.TemporaryDirectory() as tmp:
    model = os.path.join(tmp, "dense-7b-Q4.gguf"); dk.make_dense(model, n_layers=24)
    ctx = 8192
    base = ["llama-server", "-m", model, "-ngl", "99", "-c", str(ctx), "-ub", "512", "-b", "2048"]
    # no FFN plan yet (the model fitted): 3 GiB are gone now -> some layers' FFN go to RAM, at least one
    new, what = mc.dense_oom_replan(base, model, ctx, True, 2.0 * GIB, 0.25)
    check(new is not None and "-ot" in new and layers_of(new) >= 1, f"fitted plan, memory gone: FFN of {layers_of(new) if new else 0} layers to RAM ({what})")
    check(new is not None and new[:len(base)] == base, "the rest of the command is unchanged")
    # an FFN plan with 5 layers in RAM runs out again: more layers than before, at least one more
    ot5 = "blk\\.(19|20|21|22|23)\\.ffn_(up|down|gate|gate_up)\\.weight=CPU"
    cmd5 = base + ["-ot", ot5]
    new, what = mc.dense_oom_replan(cmd5, model, ctx, True, 64 * GIB, 0.25)
    check(new is not None and layers_of(new) == 6, f"plenty free now (the estimate was wrong): exactly one more layer, 6 (got {layers_of(new) if new else None})")
    new, what = mc.dense_oom_replan(cmd5, model, ctx, True, 0.5 * GIB, 0.25)
    check(new is not None and layers_of(new) == 24 and new[new.index("-ub") + 1] == "256", f"almost nothing free: all 24 layers and -ub 256 (got {layers_of(new) if new else None})")
    new, what = mc.dense_oom_replan(cmd5, model, ctx, True, 1.3 * GIB, 0.25)
    check(new is not None and 6 < layers_of(new) < 24, f"some memory gone: several more layers (got {layers_of(new) if new else None})")
    check(new is not None and new.count("-ot") == 1, "one -ot only")
    # no free-memory figure: an eighth of the layers more (24 / 8 = 3)
    new, what = mc.dense_oom_replan(cmd5, model, ctx, True, None, 0.25)
    check(new is not None and layers_of(new) == 8, f"no memory figure: 3 layers more, 8 (got {layers_of(new) if new else None})")
    # every layer already in RAM: the micro-batch is halved, -b follows but stays at least 2048
    all24 = "blk\\.(" + "|".join(str(i) for i in range(24)) + ")\\.ffn_(up|down|gate|gate_up)\\.weight=CPU"
    new, what = mc.dense_oom_replan(base + ["-ot", all24], model, ctx, True, 1.0 * GIB, 0.25)
    check(new is not None and new[new.index("-ub") + 1] == "256" and new[new.index("-b") + 1] == "2048", f"everything in RAM already: -ub 256 ({what})")
    # --fit places the layers (no -ot): a larger --fit margin; hybrid or unknown layout: same
    fit = ["llama-server", "-m", os.path.join(tmp, "missing.gguf"), "-c", "4096", "-fitt", "256"]
    new, what = mc.dense_oom_replan(fit, fit[2], 4096, True, 8 * GIB, 0.25)
    check(new is not None and new[new.index("-fitt") + 1] == "768", f"--fit run: margin 256 -> 768 MiB ({what})")
    # your own -ot stands; nothing left to shrink -> None with a reason
    new, what = mc.dense_oom_replan(cmd5, model, ctx, True, 8 * GIB, 0.25, user_ot=True)
    check(new is None, f"own -ot: untouched ({what})")
    new, what = mc.dense_oom_replan(["llama-server", "-m", "x"], "x", 4096, True, 8 * GIB, 0.25)
    check(new is None and what, f"nothing to shrink: {what}")
    # never mutates its input
    keep = list(cmd5); mc.dense_oom_replan(cmd5, model, ctx, True, 1 * GIB, 0.25)
    check(cmd5 == keep, "the original command is not changed")

if fails:
    print(f"{len(fails)} FAILED"); sys.exit(1)
print("dense_oom_test: all passed")
