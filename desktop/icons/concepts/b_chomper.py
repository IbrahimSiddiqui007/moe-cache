"""Concept B: an original boxy creature (own shape: a chip-head with teeth and horns, not a round wedge-mouth) chomping a row of RAM/expert blocks that get smaller toward its mouth."""
from pix import *
g = grid(32)
put(g, 15, 7, 15, 18, "C"); put(g, 29, 7, 1, 18, "K"); put(g, 15, 24, 15, 1, "K")      # head, shade right/bottom
put(g, 17, 4, 2, 3, "C"); put(g, 18, 4, 1, 3, "K"); put(g, 25, 4, 2, 3, "C"); put(g, 26, 4, 1, 3, "K")   # horns
put(g, 15, 13, 8, 6, "K")                                                            # mouth
for x in (15, 17, 19, 21): put(g, x, 13, 1, 2, "C")                                  # upper teeth
for x in (16, 18, 20, 22): put(g, x, 17, 1, 2, "C")                                  # lower teeth
put(g, 18, 16, 4, 1, "R")                                                            # tongue
put(g, 24, 9, 2, 2, "K"); dot(g, 24, 9, "M"); put(g, 24, 15, 4, 1, "R")              # eye, cheek band
put(g, 26, 19, 3, 1, "K")
put(g, 17, 25, 3, 2, "K"); put(g, 24, 25, 3, 2, "K")                                 # feet
block(g, 1, 13, 6, 6, band=15); block(g, 8, 14, 4, 4); put(g, 13, 15, 2, 2, "R")     # 6, 4, 2 wide: eaten from the left
for x, y in ((12, 12), (13, 19), (11, 20)): dot(g, x, y, "M")                        # crumbs
corners(g, CORNERS[32]); save(g, "b_chomper32.txt")
h = grid(16)
put(h, 7, 3, 8, 10, "C"); put(h, 14, 3, 1, 10, "K"); put(h, 7, 12, 8, 1, "K")
put(h, 8, 1, 1, 2, "C"); put(h, 12, 1, 1, 2, "C")
put(h, 7, 6, 4, 4, "K"); dot(h, 7, 6, "C"); dot(h, 9, 6, "C"); dot(h, 8, 9, "C"); dot(h, 10, 9, "C"); dot(h, 9, 8, "R")
put(h, 12, 4, 2, 2, "K"); dot(h, 12, 4, "M")
put(h, 1, 6, 3, 3, "C"); put(h, 3, 6, 1, 3, "K"); put(h, 5, 7, 1, 2, "R")
corners(h, CORNERS[16]); save(h, "b_chomper16.txt")
