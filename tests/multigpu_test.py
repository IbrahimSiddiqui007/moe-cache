#!/usr/bin/env python3
"""Checks for planner/multigpu.py (prototype, no GPU and no model download: a synthetic MoE header and fixed card sizes).
   python3 tests/multigpu_test.py
Rules that must hold: one card gives the same N as gguf_plan.tune(); a second card never puts more experts on the CPU; every card stays inside its own
memory; the split counts add up to layers + 1 and reproduce llama.cpp's own slot-to-card rule; an iGPU-sized second card changes little."""
import os, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
sys.path.insert(0, os.path.join(HERE, "tests"))
import gguf_plan, multigpu            # noqa: E402
import launch_plan_test as lp         # noqa: E402

GIB = 2**30
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def llama_cpp_cards(counts_ts, n_layers):
    """llama.cpp's rule (src/llama-model.cpp): slot il goes to the first card whose normalised cumulative split is above il / (n_layers + 1)."""
    tot = sum(counts_ts)
    cum, s = [], 0.0
    for c in counts_ts:
        s += c
        cum.append(s)
    cum = [x / tot for x in cum]
    out = []
    for il in range(n_layers + 1):
        f = il / (n_layers + 1)
        out.append(next(k for k, x in enumerate(cum) if x > f))
    return out

with tempfile.TemporaryDirectory() as tmp:
    model = os.path.join(tmp, "m.gguf")
    lp.make_synthetic_model(model)              # 40 layers, 3 x [2048, 512, 256] Q4_K experts each, small attention
    lay = gguf_plan.model_layout(model)
    L, e = lay["n_layers"], lay["expert_bytes_per_layer"]
    check(L == 40 and e > 0, f"synthetic layout: {L} layers, {e / 1e6:.0f} MB experts per layer")

    # one card = today's tune()
    for gib in (4, 6, 8, 12, 16, 24):
        t = gguf_plan.tune(model, 8192, free=gib * GIB, want_cache=False)
        m = multigpu.plan(model, 8192, [gib * GIB])
        check(m is not None and m["n_cpu_moe"] == min(t["n_cpu_moe"], L), f"one {gib} GiB card: N {m['n_cpu_moe']} = tune() {min(t['n_cpu_moe'], L)}")

    # a second card never hurts, and the cards stay inside their memory
    for f0, f1 in ((8, 8), (8, 4), (16, 8), (24, 12), (12, 24), (6, 6)):
        one = multigpu.plan(model, 8192, [f0 * GIB])
        two = multigpu.plan(model, 8192, [f0 * GIB, f1 * GIB])
        check(two["n_cpu_moe"] <= one["n_cpu_moe"], f"{f0}+{f1} GiB: N {two['n_cpu_moe']} <= {one['n_cpu_moe']} with the first card alone")
        counts = [int(x) for x in two["tensor_split"].split(",")]
        check(sum(counts) == L + 1, f"{f0}+{f1} GiB: -ts {two['tensor_split']} adds up to {L + 1} slots")
        check(all(p * GIB <= f * GIB for p, f in zip(two["per_card_gib"], (f0, f1))), f"{f0}+{f1} GiB: planned use {two['per_card_gib']} inside the cards")
        cards = llama_cpp_cards(counts, L)
        check([cards.count(k) for k in range(2)] == counts, f"{f0}+{f1} GiB: llama.cpp's slot rule gives exactly the planned counts {counts}")

    # two cards of one size hold about what one card of the sum would (minus one more compute buffer + reserve)
    one16 = multigpu.plan(model, 8192, [16 * GIB])
    two8 = multigpu.plan(model, 8192, [8 * GIB, 8 * GIB])
    check(0 <= two8["n_cpu_moe"] - one16["n_cpu_moe"] <= 8, f"2 x 8 GiB: N {two8['n_cpu_moe']} vs one 16 GiB card {one16['n_cpu_moe']} (the second card pays its own buffers)")

    # not enough memory for the rest of the model: all experts on the CPU, no split
    tiny = multigpu.plan(model, 8192, [1 * GIB, 1 * GIB])
    check(tiny["n_cpu_moe"] == L and tiny["tensor_split"] == "", "1 + 1 GiB: all experts stay on the CPU, no -ts")

    # a dense model (no experts) is not this planner's job
    check(multigpu.plan(model, 8192, [8 * GIB], layout={"n_layers": 0, "expert_bytes_per_layer": 0, "other_bytes": 1e9}) is None, "dense model: None (llama-fit-params / --fit)")

if fails:
    print(f"\n{len(fails)} FAILED")
    sys.exit(1)
print("\nall ok")
