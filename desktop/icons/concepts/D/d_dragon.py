#!/usr/bin/env python3
"""Concept D: the owner's dragon-curled-around-a-cache design (handover/parts/icon_reference2, NOT committed) rebuilt on a clean pixel grid.
d64_snap.txt / d32_snap.txt: the design snapped to the 4-colour palette and reduced to a 64x64 / 32x32 grid (majority colour per cell);
d16_hand.txt: hand-simplified 16x16 (silhouette C, red block, two spikes). This script removes stray single pixels from the snaps, adds the stepped
rounded-square corners and writes d64.txt, d32.txt, d16.txt, the maps the sheet script reads."""
import os, math
HERE = os.path.dirname(os.path.abspath(__file__))

def load(n):
    return [list(r) for r in open(os.path.join(HERE, n)).read().split()]

def despeckle(g, passes=2):
    n = len(g)
    for _ in range(passes):
        out = [r[:] for r in g]
        for y in range(n):
            for x in range(n):
                nb = [g[y + dy][x + dx] for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= x + dx < n and 0 <= y + dy < n]
                if nb and all(c != g[y][x] for c in nb):
                    out[y][x] = max(set(nb), key=nb.count)
        g = out
    return g

def corners(g, r):
    n = len(g)
    for y in range(r):
        o = r - int(math.sqrt(max(0, r * r - (r - y - 0.5) ** 2)) + 0.5)
        for x in range(o):
            for gx, gy in ((x, y), (n - 1 - x, y), (x, n - 1 - y), (n - 1 - x, n - 1 - y)):
                g[gy][gx] = "."
    return g

def save(g, name):
    open(os.path.join(HERE, name), "w").write("\n".join("".join(r) for r in g) + "\n")

save(corners(despeckle(load("d64_snap.txt")), 12), "d64.txt")
save(corners(despeckle(load("d32_snap.txt")), 6), "d32.txt")
save(corners(load("d16_hand.txt"), 3), "d16.txt")
