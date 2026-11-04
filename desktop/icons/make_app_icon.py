#!/usr/bin/env python3
"""Build the app icon files from the chosen concept D (concepts/D/d64.txt, d32.txt, d16.txt), nearest-neighbour only (no smoothing):
64x64 map for 64 px and up (1024 = x16), the 32x32 map for 32-48 px, the hand-made 16x16 for 16-24 px.
Writes the PNGs and icon.ico (ImageMagick `magick`, 7 frames) into ../src-tauri/icons/. icon.icns (macOS, not in version 1) comes from `npx tauri icon`.
Run:  python3 desktop/icons/make_app_icon.py     (needs Pillow and ImageMagick); then, once, for the .icns:  npx tauri icon <1024.png> -o <tmp> and copy icon.icns"""
import os, subprocess, tempfile
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "src-tauri", "icons")
PAL = {"S": (0x6B, 0x99, 0xB3, 255), "C": (0xF8, 0xEF, 0xC9, 255), "K": (0x2A, 0x1B, 0x14, 255), "R": (0xC8, 0x44, 0x2A, 255), "M": (0xC6, 0xCE, 0xB8, 255), ".": (0, 0, 0, 0)}
def load(n):
    rows = open(os.path.join(HERE, "concepts", "D", n)).read().split()
    im = Image.new("RGBA", (len(rows), len(rows))); im.putdata([PAL[c] for r in rows for c in r]); return im
m16, m32, m64 = load("d16.txt"), load("d32.txt"), load("d64.txt")
def render(px):
    src = m16 if px <= 24 else m32 if px <= 48 else m64
    return src.resize((px, px), Image.NEAREST)
os.makedirs(OUT, exist_ok=True)
for px, name in ((32, "32x32.png"), (128, "128x128.png"), (256, "128x128@2x.png"), (512, "icon.png")):
    render(px).save(os.path.join(OUT, name))
render(1024).save(os.path.join(os.path.dirname(HERE), "icons", "app_icon_1024.png")) if False else None
with tempfile.TemporaryDirectory() as t:
    frames = []
    for px in (16, 24, 32, 48, 64, 128, 256):
        p = os.path.join(t, f"{px}.png"); render(px).save(p); frames.append(p)
    subprocess.run(["magick", *frames, os.path.join(OUT, "icon.ico")], check=True)
print("wrote", sorted(os.listdir(OUT)))
