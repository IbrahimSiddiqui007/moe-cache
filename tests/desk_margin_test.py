#!/usr/bin/env python3
"""The GPU margin grows with the VRAM other programs hold (desktop, browser): headless card keeps the default, a busy desktop leaves more, capped."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner"))
from gguf_plan import desk_reserve_gb, tune  # noqa: E402
GIB = 2**30
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)
r, u = desk_reserve_gb(16 * GIB, 15.8 * GIB)
check(r == 0.5 and abs(u - 0.2) < 0.01, f"idle card (0.2 GiB used): default {r} GB")
r, u = desk_reserve_gb(16 * GIB, 13.2 * GIB)
check(abs(u - 2.8) < 0.01 and abs(r - 1.65) < 0.01, f"dwm + browser (2.8 GiB used): {r:.2f} GB")
r, u = desk_reserve_gb(16 * GIB, 6 * GIB)
check(r == 2.5, f"10 GiB held by others: capped at {r} GB")
check(desk_reserve_gb(None, None) == (0.5, 0.0) and desk_reserve_gb(16 * GIB, None) == (0.5, 0.0), "no GPU numbers: default")
sys.exit(1 if fails else 0)
