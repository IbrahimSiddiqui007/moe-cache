#!/usr/bin/env python3
"""Checks for bin/moe_cache_budget.py (prototype): the shares never add up to more than the cap, nobody gets more than it can use, dense models are
all-or-nothing, a model that can hold its whole file is stock, impossible combinations say 'swap'.   python3 tests/budget_test.py"""
import os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_budget as b      # noqa: E402

GIB = 2**30
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

CAP = 28 * GIB          # a 32 GiB PC under the owner's cap

# owner's example: OCR + vision (dense) that fit together: both stock, sum under the cap
r = b.split_ram(CAP, [{"name": "ocr", "size": 6 * GIB, "kind": "dense"}, {"name": "vision", "size": 9 * GIB, "kind": "dense", "extra": 1 * GIB}])
check(r["ok"] and [m["engine"] for m in r["models"]] == ["stock", "stock"], "2 dense models that fit: both stock")
check(sum(m["ram_bytes"] for m in r["models"]) <= CAP, "sum of shares <= cap")

# two dense models that do not fit together: no split
r = b.split_ram(CAP, [{"name": "a", "size": 20 * GIB, "kind": "dense"}, {"name": "b", "size": 12 * GIB, "kind": "dense"}])
check(not r["ok"] and r["models"][0]["engine"] == "swap", "2 dense models of 32 GiB in 28 GiB: swap, say so")

# MoE 40 GiB + dense 6 GiB: the dense one is whole, the MoE gets the rest as cache
r = b.split_ram(CAP, [{"name": "moe", "size": 40 * GIB}, {"name": "ocr", "size": 6 * GIB, "kind": "dense"}])
m = {x["name"]: x for x in r["models"]}
check(m["ocr"]["engine"] == "stock" and m["moe"]["engine"] == "plugin", "MoE bigger than RAM + small dense: plugin + stock")
check(abs(m["ocr"]["ram_bytes"] - 7 * GIB) < 2 and m["moe"]["ram_bytes"] <= CAP - 7 * GIB + 2, f"dense gets file+1 GiB, MoE the rest ({m['moe']['ram_bytes'] / GIB:.1f} GiB)")

# two MoE models bigger than RAM: split by weight, heavier use gets more, both floors respected
for w in ((1, 1), (3, 1), (1, 3)):
    r = b.split_ram(CAP, [{"name": "a", "size": 60 * GIB, "weight": w[0]}, {"name": "b", "size": 50 * GIB, "weight": w[1]}])
    a, bb = r["models"][0]["ram_bytes"], r["models"][1]["ram_bytes"]
    check(a + bb <= CAP + 2 and a >= 4.5 * GIB and bb >= 4.5 * GIB, f"two big MoE, weights {w}: {a / GIB:.1f} + {bb / GIB:.1f} GiB <= {CAP / GIB:.0f}, floors kept")
    check((a > bb) == (w[0] > w[1]) or w[0] == w[1], f"weights {w}: the heavier use gets the larger share")

# a share is never above what the model can use
r = b.split_ram(CAP, [{"name": "small", "size": 3 * GIB}, {"name": "big", "size": 90 * GIB}])
m = {x["name"]: x for x in r["models"]}
check(m["small"]["ram_bytes"] <= 4 * GIB + 2 and m["small"]["engine"] == "stock", "a small MoE gets file+headroom, not more, and runs stock")
check(m["big"]["ram_bytes"] >= CAP - 4 * GIB - 2, "the big one gets the rest")

# vram split
v = b.split_vram(16 * GIB, [{"name": "a", "weight": 3}, {"name": "b", "weight": 1}, {"name": "cpu_only", "gpu": False}])
check(set(v) == {"a", "b"} and v["a"] == 3 * v["b"] and sum(v.values()) <= 15 * GIB, f"VRAM: 15 GiB shared 3:1 ({v['a'] / GIB:.2f} / {v['b'] / GIB:.2f})")

if fails:
    print(f"\n{len(fails)} FAILED")
    sys.exit(1)
print("\nall ok")
