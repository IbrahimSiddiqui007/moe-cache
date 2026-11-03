"""Tiny helpers shared by the concept scripts: a grid of palette letters, rectangles, stepped corners, writing a text map.
Palette: S sky #6b99b3, C cream #f8efc9, K dark brown #2a1b14, R red-orange #c8442a, M mid #c6ceb8, '.' transparent (corner steps)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))

def grid(n):
    return [["S"] * n for _ in range(n)]

def put(g, x, y, w, h, c):
    for j in range(y, y + h):
        for i in range(x, x + w):
            if 0 <= i < len(g) and 0 <= j < len(g):
                g[j][i] = c

def dot(g, x, y, c):
    put(g, x, y, 1, 1, c)

def block(g, x, y, w, h, band=None, top="M"):
    """a lit block: cream, a mid-tone top row, a dark shade column on the right, optional red band row"""
    put(g, x, y, w - 1, h, "C"); put(g, x + w - 1, y, 1, h, "K")
    if top and h > 2:
        put(g, x, y, w - 1, 1, top)
    if band is not None:
        put(g, x, band, w - 1, 1, "R")

def corners(g, off):
    n = len(g)
    for y, o in enumerate(off):
        for x in range(o):
            for gx, gy in ((x, y), (n - 1 - x, y), (x, n - 1 - y), (n - 1 - x, n - 1 - y)):
                g[gy][gx] = "."

def save(g, name):
    open(os.path.join(HERE, name), "w").write("\n".join("".join(r) for r in g) + "\n")
    print(name, "\n" + "\n".join("".join(r) for r in g))

CORNERS = {32: [6, 4, 3, 2, 1, 1, 0], 16: [3, 1, 1, 0]}
