#!/usr/bin/env python3
"""Render the top of docs/benchmark_card.html (title, headline numbers, decode chart) to PNGs for the README (GitHub shows no HTML pages inline):
docs/benchmark_card.png (light) and docs/benchmark_card-dark.png. Run it after every change to the card (the README rule: numbers, card, image in one step).

    python3 tools/card_image.py [--chrome PATH] [--sections 2] [--width 940]

Needs Google Chrome or Chromium (headless) and Pillow. The card is copied to a temporary file with the later sections hidden and the theme fixed."""
import argparse, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CARD = os.path.join(HERE, "..", "docs", "benchmark_card.html")


def find_chrome():
    for c in ("google-chrome-stable", "google-chrome", "chromium", "chromium-browser"):
        if shutil.which(c):
            return shutil.which(c)
    return None


def render(chrome, html, theme, sections, width, out):
    s = re.sub(r"<html([^>]*)>", lambda m: f'<html{m.group(1)} data-theme="{theme}">', html, count=1) if "<html" in html else f'<html data-theme="{theme}">' + html
    hide = (f"<style>section:nth-of-type(n+{sections + 1}), footer {{ display: none !important; }} body {{ padding-bottom: 8px !important; }}"
            f" section:nth-of-type({sections}) > :not(h2):not(.chart) {{ display: none !important; }}</style>")    # the last shown section: heading and chart only
    s = s.replace("</head>", hide + "</head>", 1) if "</head>" in s else hide + s
    with tempfile.TemporaryDirectory() as tmp:
        page = os.path.join(tmp, "card.html")
        with open(page, "w", encoding="utf-8") as f:
            f.write(s)
        raw = os.path.join(tmp, "raw.png")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size={width},2400",
                        "--virtual-time-budget=4000", f"--screenshot={raw}", "file://" + page], check=True, capture_output=True, timeout=120)
        from PIL import Image
        im = Image.open(raw).convert("RGB")
        bg = im.getpixel((im.width - 2, im.height - 2))
        w, h = im.size
        px = im.load()
        bottom = h
        while bottom > 1 and all(px[x, bottom - 1] == bg for x in range(0, w, 4)):    # trim the empty page below the last shown section
            bottom -= 1
        im.crop((0, 0, w, min(h, bottom + 24))).save(out, optimize=True)
        return im.width, bottom


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chrome", default=find_chrome())
    ap.add_argument("--sections", type=int, default=2, help="how many <section>s of the card to show (default 2: headline numbers + decode chart)")
    ap.add_argument("--width", type=int, default=940)
    a = ap.parse_args()
    if not a.chrome:
        sys.exit("no Chrome / Chromium found: pass --chrome PATH")
    html = open(CARD, encoding="utf-8").read()
    for theme, name in (("light", "benchmark_card.png"), ("dark", "benchmark_card-dark.png")):
        out = os.path.join(HERE, "..", "docs", name)
        w, h = render(a.chrome, html, theme, a.sections, a.width, out)
        print(f"{os.path.normpath(out)}: {w}x{h + 24} px ({theme})")


if __name__ == "__main__":
    main()
