#!/usr/bin/env python3
"""Concept E: the owner's robot dinosaur (handover/parts/icon_ref_E_robodino.png, NOT committed) rebuilt on clean pixel grids.
e64_snap.txt: the design snapped to the 4-colour palette on a 64x64 grid (majority colour per cell; the screenshot's black border and "Preview" button
were cropped / painted out). This script removes stray single pixels, builds the 32x32 from the 64x64 with outline-and-red priority (a cell is dark
brown or red when 2 of its 4 pixels are; thin outlines and the small red parts survive), and adds the stepped rounded-square corners.
e16_hand.txt is hand-simplified (open jaw, one or two blocks, red chest). Writes e64.txt, e32.txt, e16.txt (the maps the sheet scripts read)."""
import os, math
HERE = os.path.dirname(os.path.abspath(__file__))
def load(n): return [list(r) for r in open(os.path.join(HERE, n)).read().split()]
def save(g, name): open(os.path.join(HERE, name), "w").write("\n".join("".join(r) for r in g) + "\n")

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

def pool2(g):
    n = len(g) // 2; out = []
    for j in range(n):
        row = []
        for i in range(n):
            cell = [g[2 * j + dy][2 * i + dx] for dy in (0, 1) for dx in (0, 1)]
            c = "R" if cell.count("R") >= 2 else "K" if cell.count("K") >= 2 else "C" if cell.count("C") >= 2 else max(set(cell), key=cell.count)
            row.append(c)
        out.append(row)
    return out

def corners(g, r):
    n = len(g)
    for y in range(r):
        o = r - int(math.sqrt(max(0, r * r - (r - y - 0.5) ** 2)) + 0.5)
        for x in range(o):
            for gx, gy in ((x, y), (n - 1 - x, y), (x, n - 1 - y), (n - 1 - x, n - 1 - y)):
                g[gy][gx] = "."
    return g

g64 = despeckle(load("e64_snap.txt"))
g32 = despeckle(pool2(g64), 1)
if os.path.exists(os.path.join(HERE, "e32_edits.txt")):                 # optional hand edits to the pooled 32: lines "x y C" (x right, y down)
    for line in open(os.path.join(HERE, "e32_edits.txt")).read().splitlines():
        if line.strip() and not line.startswith("#"):
            x, y, c = line.split(); g32[int(y)][int(x)] = c
save(corners([r[:] for r in g64], 12), "e64.txt")
save(corners(g32, 6), "e32.txt")
save(corners(load("e16_hand.txt"), 3), "e16.txt")
