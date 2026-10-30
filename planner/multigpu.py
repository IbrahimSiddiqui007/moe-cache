"""Placement for more than one GPU (PROTOTYPE, avenue 2 of handover/parts/multi_model_gpu_20261030.md; not wired into the launcher yet).

Why: llama.cpp assigns layers to devices by index, in the order of its device list, in proportion to -ts (default: each device's free memory), and
`--n-cpu-moe N` keeps the experts of layers 0..N-1 on the CPU. The launcher plans with ONE card's free memory and passes neither -ts nor --device, so
with two cards the planner counts one card, and the early layers (light: their experts are on the CPU) go to card 0 while the heavy ones pile up on
card 1 (docs of the report, section "what a 2-GPU user gets now").

What this does: fill the cards in order, first the N light layers (experts on the CPU), then the heavy ones, each card up to its own free memory
(minus its compute buffer and reserve; the first card also keeps the long-prompt block, because the plugin runs that block on the first GPU). The
number of layers each card gets is the -ts to pass (llama.cpp splits slot i of L+1 slots (L layers + the output layer) to the card whose cumulative
share is above i/(L+1), so counts that add up to L+1 reproduce the fill exactly). N is the smallest value for which everything fits.

Same arithmetic as gguf_plan.tune() for one card (tests/multigpu_test.py checks that), so a single-GPU machine gets the same answer as today.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_plan  # noqa: E402

MIB = 1048576


def fill(n_layers, n_cpu_moe, light, heavy, caps):
    """Counts of slots (layers 0..n_layers-1 and the output slot n_layers) per card, filled in card order; None when they do not fit.
    light = bytes of a slot whose experts are on the CPU (also the output slot), heavy = bytes of a slot with its experts on the card."""
    counts = [0] * len(caps)
    dev, used = 0, 0.0
    for i in range(n_layers + 1):
        w = heavy if (i < n_layers and i >= n_cpu_moe) else light
        while dev < len(caps) and used + w > caps[dev]:
            dev, used = dev + 1, 0.0
        if dev >= len(caps):
            return None
        counts[dev] += 1
        used += w
    return counts


def plan(path, ctx, frees, kv_q8=False, ub=None, reserve_gb=0.5, layout=None, kv=None, compute_mib=None):
    """frees: free bytes per card in the order llama.cpp will list them (pass --device in this order). Returns a dict like gguf_plan.tune() with
    n_cpu_moe, ub, plus tensor_split ("a,b", counts of slots), per_card_gib (planned use per card), ub; or None when no card has memory information.
    layout/kv/compute_mib: for tests (a layout dict as gguf_plan.model_layout returns, KV bytes, the compute buffer in MiB)."""
    if not frees:
        return None
    lay = layout or gguf_plan.model_layout(path)
    L, e = lay["n_layers"], lay["expert_bytes_per_layer"]
    if not L or not e:
        return None                                   # dense model: nothing to split by experts (llama.cpp's --fit / llama-fit-params does layers)
    kv = kv if kv is not None else gguf_plan._kv_or_flat(path, ctx, kv_q8)
    # 1) the answer for one pool of memory: what tune() gives for the sum, with each extra card paying its own compute buffer + reserve
    probe = gguf_plan.tune(path, ctx, free=max(frees), kv_q8=kv_q8, want_cache=False, reserve_gb=reserve_gb, ub=ub) if layout is None else {"ub": ub or 1024, "compute_mib": compute_mib or 400}
    buf = (compute_mib if compute_mib is not None else probe["compute_mib"]) * MIB
    u = probe["ub"]
    block = (1.6 if u > 1024 else 1.35) * e + (160 + 256) * MIB          # the plugin's long-prompt block: same formula as tune(); on the first card only
    caps = [f - buf - reserve_gb * 1e9 - (block if k == 0 else 0) for k, f in enumerate(frees)]
    light = (lay["other_bytes"] + kv) / (L + 1)                          # the same fixed part tune() counts, spread over the slots
    heavy = light + e
    for n in range(0, L + 1):
        counts = fill(L, n, light, heavy, caps)
        if counts is not None:
            break
    else:
        counts, n = None, L
    why = []
    if counts is None:
        why.append("even without any experts the cards' memory is too small for the rest of the model: all experts stay on the CPU, layers as llama.cpp splits them")
        return {"n_cpu_moe": L, "ub": u, "tensor_split": "", "per_card_gib": [], "gpu_cache_mib": 0, "why": why}
    per = []
    s = 0
    for c in counts:
        heavy_here = sum(1 for i in range(s, s + c) if n <= i < L)
        per.append(round((c * light + heavy_here * e) / 2**30, 2))
        s += c
    used = [k for k, c in enumerate(counts) if c]
    why.append(f"{L - n} of {L} layers of experts on the GPUs; layers per card (of {L + 1} slots incl. the output layer): {counts}; planned use per card GiB: {per}")
    return {"n_cpu_moe": n, "ub": u, "tensor_split": ",".join(str(c) for c in counts), "per_card_gib": per, "cards_used": used, "gpu_cache_mib": 0, "why": why}
