#!/usr/bin/env python3
"""U09 arithmetic: what would lending the output head's GPU memory to the prompt gain, and what does keeping it off the GPU cost?

Reads only the GGUF header (no model load, no GPU). Run:
  python3 tools/head_lend_math.py MODEL.gguf [--cpu-gbs 40] [--tok-s 20] [--layer-pct 2.5] [--vocab N --ub 2048]

The numbers it prints are arithmetic from the file sizes plus three assumptions you can change (CPU memory bandwidth,
decode speed, decode gain per GPU layer). Nothing here is measured; see patches/llama.cpp/README.md, section U09.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "planner"))
import gguf_plan  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--cpu-gbs", type=float, default=40.0, help="CPU memory bandwidth in GB/s (assumption)")
    ap.add_argument("--tok-s", type=float, default=20.0, help="decode speed in tokens/s (assumption)")
    ap.add_argument("--layer-pct", type=float, default=2.5, help="decode gain per GPU layer in percent (the docs say 1.5-4)")
    ap.add_argument("--vocab", type=int, default=0, help="vocabulary size, to also print the logits matrix of the reserved graph")
    ap.add_argument("--ub", type=int, default=2048, help="n_ubatch for the logits line")
    a = ap.parse_args()

    tensors = dict(gguf_plan.read_tensors(a.model))
    head = tensors.get("output.weight") or tensors.get("token_embd.weight")
    if head is None:
        print("moe-cache: no output.weight in the header, nothing to compute")
        return 1
    tied = "output.weight" not in tensors
    # one GPU layer = the expert tensors of one block (what --n-cpu-moe moves), averaged
    per_blk = {}
    for name, size in tensors.items():
        m = re.match(r"blk\.(\d+)\..*_exps\.weight$", name)
        if m:
            per_blk[int(m.group(1))] = per_blk.get(int(m.group(1)), 0) + size
    layer = sum(per_blk.values()) / len(per_blk) if per_blk else 0
    mib = 1024.0 * 1024.0
    cpu_ms = head / (a.cpu_gbs * 1e9) * 1000
    tok_ms = 1000.0 / a.tok_s
    rows = [
        ("head tensor", "token_embd (tied)" if tied else "output.weight", f"{head / mib:.0f} MiB"),
        ("one GPU layer (experts)", f"{len(per_blk)} layers", f"{layer / mib:.0f} MiB"),
        ("head in layers", "", f"{head / layer:.2f}" if layer else "n/a"),
        ("decode gain if the head's room became layers", f"{a.layer_pct}% per layer", f"{head / layer * a.layer_pct:.1f} %" if layer else "n/a"),
        ("decode cost of the head on the CPU", f"{a.cpu_gbs:g} GB/s, {a.tok_s:g} tok/s", f"+{cpu_ms:.1f} ms of {tok_ms:.0f} ms = {cpu_ms / tok_ms * 100:.0f} % slower"),
    ]
    if a.vocab:
        rows.append(("logits matrix in the reserved graph", f"{a.vocab} x {a.ub} x 4 B", f"{a.vocab * a.ub * 4 / mib:.0f} MiB (LLAMA_N_OUTPUTS_MAX shrinks it)"))
    w = [max(len(r[i]) for r in rows) for i in range(3)]
    for r in rows:
        print("| " + " | ".join(r[i].ljust(w[i]) for i in range(3)) + " |")
    if layer and head / layer * a.layer_pct < cpu_ms / tok_ms * 100:
        print("moe-cache: the head on the CPU costs more decode than the room gains; only a lend that is back on the GPU before decode could pay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
