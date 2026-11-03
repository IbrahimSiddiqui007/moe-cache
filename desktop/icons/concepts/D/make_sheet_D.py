#!/usr/bin/env python3
"""concepts_D.png: the owner's image (read from --ref, never copied into the repo) scaled down beside concept D at 16, 32, 64 and 256 px,
light and dark rows. Usage: python3 make_sheet_D.py [--ref PATH_TO_IMAGE]"""
import os, sys
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
PAL = {"S": (0x6B, 0x99, 0xB3, 255), "C": (0xF8, 0xEF, 0xC9, 255), "K": (0x2A, 0x1B, 0x14, 255), "R": (0xC8, 0x44, 0x2A, 255), "M": (0xC6, 0xCE, 0xB8, 255), ".": (0, 0, 0, 0)}
def load(name):
    rows = open(os.path.join(HERE, name)).read().split(); im = Image.new("RGBA", (len(rows), len(rows))); im.putdata([PAL[c] for r in rows for c in r]); return im
def up(im, z): return im.resize((im.width * z, im.height * z), Image.NEAREST)
ref = None
if "--ref" in sys.argv:
    ref = Image.open(sys.argv[sys.argv.index("--ref") + 1]).convert("RGB")
d16, d32, d64 = load("d16.txt"), load("d32.txt"), load("d64.txt")
GAP, LABEL, ROW = 36, 54, 256 + 72
tiles = [d16, d32, d64, up(d64, 4)]            # 16, 32, 64, 256 px
W = GAP + (256 + GAP if ref else 0) + sum(t.width + GAP for t in tiles)
H = LABEL + ROW * 2 + 16 * 8 + 2 * GAP
sheet = Image.new("RGB", (W, H)); d = ImageDraw.Draw(sheet); F = ImageFont.load_default(size=26)
for r, bg in enumerate([(246, 248, 250), (22, 27, 34)]):
    d.rectangle([0, LABEL + r * ROW, W, LABEL + (r + 1) * ROW], fill=bg)
d.rectangle([0, 0, W, LABEL], fill=(246, 248, 250)); d.rectangle([0, LABEL + 2 * ROW, W, H], fill=(246, 248, 250))
d.text((GAP, 10), "D  owner's design" + (" (left: scaled image)" if ref else "") + "  |  16, 32, 64, 256 px", fill=(30, 30, 30), font=F)
for r in range(2):
    x = GAP
    if ref:
        t = ref.resize((256, 256), Image.LANCZOS); sheet.paste(t, (x, LABEL + r * ROW + GAP)); x += 256 + GAP
    for im in tiles:
        y = LABEL + r * ROW + GAP + (256 - im.height) // 2; sheet.paste(im, (x, y), im); x += im.width + GAP
x = GAP; y0 = LABEL + 2 * ROW + GAP
for im, z in ((d16, 8), (d32, 4)):
    t = up(im, z); sheet.paste(t, (x, y0), t); x += t.width + GAP
sheet.save(os.path.join(HERE, "..", "concepts_D.png")); print("concepts_D.png", sheet.size)
