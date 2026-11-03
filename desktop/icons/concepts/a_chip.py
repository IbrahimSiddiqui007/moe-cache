"""Concept A: a pixel memory/CPU chip (pins on the sides), cache blocks inside (L2 row, L3 row, RAM row; red = hot experts), draft 1's tower rising from it."""
from pix import *
g = grid(32)
dot(g, 15, 1, "K"); put(g, 15, 2, 1, 2, "K")                       # antenna
put(g, 14, 4, 2, 6, "C"); put(g, 16, 4, 1, 6, "K")                  # mast
block(g, 10, 5, 4, 4); block(g, 18, 5, 4, 4, band=7)                # two little panels
put(g, 14, 6, 3, 1, "R")
put(g, 6, 10, 19, 17, "C"); put(g, 25, 10, 1, 17, "K"); put(g, 6, 27, 20, 1, "K")   # chip body, shade right and bottom
for y in range(12, 26, 2):                                          # side pins
    put(g, 4, y, 2, 1, "M"); put(g, 26, y, 2, 1, "K")
for x in range(8, 24, 2):                                           # bottom pins
    put(g, x, 28, 1, 2, "M")
for x in (8, 10, 12, 19, 21, 23):                                   # top pins (beside the mast)
    dot(g, x, 9, "M")
put(g, 9, 13, 14, 12, "K")                                          # the die
for x, c in ((10, "C"), (14, "R"), (18, "C")): put(g, x, 15, 3, 2, c)        # L2 row
for x, c in ((10, "C"), (16, "C")): put(g, x, 18, 6, 2, c)                   # L3 row
put(g, 16, 18, 6, 2, "R") if False else None
for x, c in ((10, "C"), (13, "C"), (16, "R"), (19, "C")): put(g, x, 21, 2, 2, c)  # RAM row
dot(g, 21, 14, "M"); dot(g, 10, 14, "M")
corners(g, CORNERS[32]); save(g, "a_chip32.txt")
h = grid(16)
dot(h, 8, 0, "K"); put(h, 7, 1, 2, 4, "C"); put(h, 9, 1, 1, 4, "K")
put(h, 3, 5, 10, 8, "C"); put(h, 13, 5, 1, 8, "K"); put(h, 3, 13, 11, 1, "K")
for y in (7, 9, 11): dot(h, 2, y, "M"); dot(h, 14, y, "K")
for x in (4, 6, 8, 10, 12): dot(h, x, 14, "M")
put(h, 5, 7, 6, 5, "K")
put(h, 5, 8, 2, 1, "C"); put(h, 8, 8, 2, 1, "R"); put(h, 5, 10, 2, 1, "R"); put(h, 8, 10, 2, 1, "C")
corners(h, CORNERS[16]); save(h, "a_chip16.txt")
