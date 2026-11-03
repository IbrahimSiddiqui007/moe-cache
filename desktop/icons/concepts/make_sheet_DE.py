#!/usr/bin/env python3
"""concepts_DE.png: D and E as two columns, each with the owner's original scaled down beside it, then 16, 32, 64 and 256 px, light and dark rows,
and an enlarged strip of the 16 / 32 px versions. The originals are read from --ref-d / --ref-e (never copied into the repo; E needs the same crop as e_robodino).
Usage: python3 make_sheet_DE.py --ref-d IMAGE --ref-e IMAGE"""
import os, sys
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
PAL = {"S": (0x6B, 0x99, 0xB3, 255), "C": (0xF8, 0xEF, 0xC9, 255), "K": (0x2A, 0x1B, 0x14, 255), "R": (0xC8, 0x44, 0x2A, 255), "M": (0xC6, 0xCE, 0xB8, 255), ".": (0, 0, 0, 0)}
def load(path):
    rows = open(os.path.join(HERE, path)).read().split(); im = Image.new("RGBA", (len(rows), len(rows))); im.putdata([PAL[c] for r in rows for c in r]); return im
def up(im, z): return im.resize((im.width * z, im.height * z), Image.NEAREST)
def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None
refs = {"D": (arg("--ref-d"), None), "E": (arg("--ref-e"), (59, 55, 529, 525))}      # E: the screenshot's blue card (crop box, px)
cols = []
for letter, name, folder in (("D", "curled dragon", "D"), ("E", "robot dino", "E")):
    path, box = refs[letter]
    ref = Image.open(path).convert("RGB") if path else None
    if ref is not None and box:
        ref = ref.crop(box)
        ImageDraw.Draw(ref).rectangle([425 - box[0], 57 - box[1], 530 - box[0], 100 - box[1]], fill=(103, 147, 173))     # the screenshot's "Preview" button
    cols.append((letter, name, ref, load(f"{folder}/{folder.lower()}16.txt"), load(f"{folder}/{folder.lower()}32.txt"), load(f"{folder}/{folder.lower()}64.txt")))
GAP, LABEL, ROW = 36, 54, 256 + 72
colw = GAP + 256 + sum(w + GAP for w in (16, 32, 64, 256))
W = GAP + colw * 2
H = LABEL + ROW * 2 + 16 * 8 + 2 * GAP
sheet = Image.new("RGB", (W, H)); d = ImageDraw.Draw(sheet); F = ImageFont.load_default(size=28)
for r, bg in enumerate([(246, 248, 250), (22, 27, 34)]):
    d.rectangle([0, LABEL + r * ROW, W, LABEL + (r + 1) * ROW], fill=bg)
d.rectangle([0, 0, W, LABEL], fill=(246, 248, 250)); d.rectangle([0, LABEL + 2 * ROW, W, H], fill=(246, 248, 250))
for c, (letter, name, ref, d16, d32, d64) in enumerate(cols):
    x0 = GAP + c * colw
    d.text((x0, 10), f"{letter}  {name}:  original | 16 | 32 | 64 | 256 px", fill=(30, 30, 30), font=F)
    for r in range(2):
        x = x0
        if ref is not None:
            sheet.paste(ref.resize((256, 256), Image.LANCZOS), (x, LABEL + r * ROW + GAP))
        x += 256 + GAP
        for im in (d16, d32, d64, up(d64, 4)):
            sheet.paste(im, (x, LABEL + r * ROW + GAP + (256 - im.height) // 2), im); x += im.width + GAP
    x = x0 + 256 + GAP
    for im, z in ((d16, 8), (d32, 4)):
        t = up(im, z); sheet.paste(t, (x, LABEL + 2 * ROW + GAP), t); x += t.width + GAP
sheet.save(os.path.join(HERE, "concepts_DE.png")); print("concepts_DE.png", sheet.size)
