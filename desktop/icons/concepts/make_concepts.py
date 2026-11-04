#!/usr/bin/env python3
"""Render concepts.png from the text maps (a_chip, b_chomper, c_dragon: *32.txt and *16.txt). Run the concept scripts first if a map changed.
Per concept: 16 (the hand-placed 16x16 map), 32, 64 (x2) and 256 (x8) px, nearest-neighbour only, on a light and a dark row."""
import os
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
PAL = {"S": (0x6B, 0x99, 0xB3, 255), "C": (0xF8, 0xEF, 0xC9, 255), "K": (0x2A, 0x1B, 0x14, 255), "R": (0xC8, 0x44, 0x2A, 255), "M": (0xC6, 0xCE, 0xB8, 255), ".": (0, 0, 0, 0)}

def load(name):
    rows = open(os.path.join(HERE, name)).read().split()
    im = Image.new("RGBA", (len(rows), len(rows)))
    im.putdata([PAL[c] for r in rows for c in r])
    return im

def up(im, z):
    return im.resize((im.width * z, im.height * z), Image.NEAREST)

CONCEPTS = [("A", "chip", "a_chip"), ("B", "chomper", "b_chomper"), ("C", "dragon", "c_dragon")]
GAP = 36
group_w = 16 + 32 + 64 + 256 + GAP * 5
W = group_w * 3 + GAP
LABEL = 54
ROW = 256 + 2 * GAP
sheet = Image.new("RGB", (W, LABEL + ROW * 2))
d = ImageDraw.Draw(sheet)
FONT = ImageFont.load_default(size=30)
for r, bg in enumerate([(246, 248, 250), (22, 27, 34)]):
    d.rectangle([0, LABEL + r * ROW, W, LABEL + (r + 1) * ROW], fill=bg)
d.rectangle([0, 0, W, LABEL], fill=(246, 248, 250))
for i, (letter, word, stem) in enumerate(CONCEPTS):
    big, small = load(stem + "32.txt"), load(stem + "16.txt")
    x0 = GAP + i * group_w
    d.text((x0, 6), f"{letter}  {word}", fill=(30, 30, 30), font=FONT)
    for r in range(2):
        x = x0
        for im in (small, big, up(big, 2), up(big, 8)):
            y = LABEL + r * ROW + GAP + (256 - im.height) // 2
            sheet.paste(im, (x, y), im); x += im.width + GAP
sheet.save(os.path.join(HERE, "concepts.png"))
# a second, enlarged strip of the 16 and 32 px versions so the small sizes can be judged
zoom = Image.new("RGB", (W, 16 * 8 + 2 * GAP), (246, 248, 250))
for i, (letter, word, stem) in enumerate(CONCEPTS):
    x = GAP + i * group_w
    for im, z in ((load(stem + "16.txt"), 8), (load(stem + "32.txt"), 4)):
        t = up(im, z); zoom.paste(t, (x, GAP), t); x += t.width + GAP
full = Image.new("RGB", (W, sheet.height + zoom.height)); full.paste(sheet, (0, 0)); full.paste(zoom, (0, sheet.height)); full.save(os.path.join(HERE, "concepts.png"))
print("concepts.png", full.size)
