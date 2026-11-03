#!/usr/bin/env python3
"""LAUNCHER-001: the launcher turns the plugin's slow-path log lines into one plain sentence each. The lines are the ones src/mc_backend.inl prints (checked
against the source here, so a reworded plugin message breaks this test instead of the warning).   python3 tests/slow_path_test.py"""
import os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

lines = {
    "no-mem-1": "moe-cache: a long-prompt block needs 539 MiB of GPU memory but only 538 MiB are free: it stays on the CPU (a smaller --moe-cache-mib or --gpu-margin leaves room; about 704 MiB per layer for this model)",
    "no-mem-big": "moe-cache: a long-prompt block needs 2300 MiB of GPU memory but only 900 MiB are free: it stays on the CPU (a smaller ...)",
    "no-op": "moe-cache: the GPU backend does not support MUL_MAT_ID with iq2_xxs weights: such expert blocks stay on the CPU",
    "no-ram": "moe-cache: a long-prompt block would upload whole expert tensors with experts not in RAM: it stays on the CPU",
    "alloc": "moe-cache: could not allocate GPU memory for a long-prompt block: blocks stay on the CPU for 64 graphs, then the GPU is tried again",
    "no-pin": "moe-cache: no pinned host buffer for the long-prompt block's result (out of memory): it is read back the plain way",
}
kinds = {k: (mc.slow_path_note(v) or (None, ""))[0] for k, v in lines.items()}
check(kinds == {"no-mem-1": "no-mem", "no-mem-big": "no-mem", "no-op": "no-op", "no-ram": "no-ram", "alloc": "alloc", "no-pin": "no-pin"}, f"all five kinds are recognised ({kinds})")
check("short by 1 MiB" in mc.slow_path_note(lines["no-mem-1"])[1] and "539" in mc.slow_path_note(lines["no-mem-1"])[1], "538 vs 539 MiB says it is short by 1 MiB")
check("short by" not in mc.slow_path_note(lines["no-mem-big"])[1] and "lower --vram" in mc.slow_path_note(lines["no-mem-big"])[1], "a big gap does not say 'short by'")
check(mc.slow_path_note("moe-cache: long-prompt block for 8192 tokens: uploading only the 40 of 256 experts it uses") is None, "'uploading only the n of m experts' is expected and stays quiet")
check(mc.slow_path_note("llama-server: prompt processing 100 %") is None, "other lines stay quiet")
# the patterns still match the plugin source's format strings
src = open(os.path.join(HERE, "src", "mc_backend.inl"), encoding="utf-8").read()
src = re.sub(r'"\s*\n\s*"', "", src)
for kind, text in (("no-mem", "a long-prompt block needs %.0f MiB of GPU memory but only %.0f MiB are free"), ("no-op", "such expert blocks stay on the CPU"),
                   ("no-ram", "would upload whole expert tensors with experts not in RAM: it stays on the CPU"),
                   ("alloc", "could not allocate GPU memory for a long-prompt block: blocks stay on the CPU for 64 graphs"), ("no-pin", "no pinned host buffer for the long-prompt block")):
    check(text in src, f"src/mc_backend.inl still prints the {kind} line")
sys.exit(1 if fails else 0)
