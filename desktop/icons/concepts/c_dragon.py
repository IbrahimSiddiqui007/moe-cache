"""Concept C: a small side-view dragon (cream, dark-brown shading, red belly and eye) biting a RAM stick."""
from pix import *
g = grid(32)
# the RAM stick: sage board, dark chips, one hot chip, pins underneath
put(g, 0, 15, 14, 4, "M")
for x in (2, 5, 8): put(g, x, 16, 2, 2, "K")
put(g, 11, 16, 2, 2, "R")
for x in range(1, 10, 2): dot(g, x, 19, "C")
# head: upper jaw, mouth, lower jaw
put(g, 9, 11, 10, 4, "C"); put(g, 9, 14, 10, 1, "K"); dot(g, 9, 12, "K")                 # upper jaw + nostril
put(g, 11, 15, 8, 4, "K")                                                                # mouth
for x in (12, 14, 16): dot(g, x, 15, "C")                                                # teeth
put(g, 10, 19, 9, 3, "C"); put(g, 10, 21, 9, 1, "K"); dot(g, 13, 18, "C"); dot(g, 15, 18, "C")
put(g, 15, 11, 2, 2, "K"); dot(g, 15, 11, "R")                                           # eye
dot(g, 17, 10, "C"); dot(g, 18, 9, "C"); dot(g, 19, 8, "C"); dot(g, 17, 9, "K"); dot(g, 18, 8, "K")   # horn
# neck and body
put(g, 18, 12, 5, 5, "C"); put(g, 18, 16, 12, 9, "C"); put(g, 29, 16, 1, 9, "K"); put(g, 18, 24, 12, 1, "K")
put(g, 19, 17, 2, 7, "R")                                                                # belly
for y in (18, 20, 22): put(g, 19, y, 2, 1, "K")
put(g, 22, 8, 1, 4, "K"); put(g, 23, 9, 1, 4, "K"); put(g, 24, 10, 1, 4, "K"); put(g, 25, 11, 1, 5, "K"); put(g, 26, 12, 1, 4, "K")   # wing
dot(g, 22, 8, "C"); dot(g, 23, 9, "C"); dot(g, 24, 10, "C")
put(g, 20, 25, 3, 3, "C"); put(g, 20, 27, 3, 1, "K"); put(g, 26, 25, 3, 3, "C"); put(g, 26, 27, 3, 1, "K")   # legs
put(g, 29, 25, 1, 1, "K"); put(g, 30, 26, 1, 1, "K")
for y, x in ((22, 30), (23, 30)): dot(g, x, y, "C")                                     # tail start
corners(g, CORNERS[32]); save(g, "c_dragon32.txt")
h = grid(16)
put(h, 0, 7, 6, 2, "M"); dot(h, 2, 7, "K"); dot(h, 4, 7, "R"); dot(h, 1, 9, "C"); dot(h, 3, 9, "C")
put(h, 4, 5, 5, 2, "C"); put(h, 4, 6, 5, 1, "K"); put(h, 5, 9, 4, 2, "C"); put(h, 5, 10, 4, 1, "K")
put(h, 5, 7, 4, 2, "K"); dot(h, 6, 7, "C"); dot(h, 7, 5, "R")
put(h, 8, 7, 6, 7, "C"); put(h, 13, 7, 1, 7, "K"); put(h, 8, 13, 6, 1, "K"); put(h, 8, 8, 1, 4, "R")
put(h, 10, 3, 1, 3, "K"); put(h, 11, 4, 1, 3, "K"); put(h, 12, 5, 1, 2, "K"); dot(h, 10, 3, "C")
corners(h, CORNERS[16]); save(h, "c_dragon16.txt")
