#!/usr/bin/env python3
"""Draw docs/img/results.svg (the README chart) from the ROWS table below. Every number must come from docs/RESULTS.md: the section is in each row.
   python3 tools/make_results_chart.py            (writes docs/img/results.svg)"""
import os

# (title, setup line, stock tok/s, moe-cache tok/s, RESULTS section / version measured with)
ROWS = [
    ("GLM-5.3-Flash (102 GB) in 24 GB RAM", "8 GB GPU, ctx 4k, exact output", 0.32, 1.75, "§26, v0.2.3.2"),
    ("Qwen3.8-Flash-Next (68 GB) in 24 GB RAM", "8 GB GPU, launcher defaults, ctx 4k", 5.2, 12.5, "§44, v0.2.3.4+"),
    ("gpt-oss-120b (63 GB) in 24 GB RAM", "2 expert layers on GPU, one run each", 0.37, 4.47, "§12, v0.1"),
    ("Qwen3-Next-80B (48.5 GB) in 24 GB RAM", "all experts on CPU, GPU ~2.6 GB, ctx 8k", 2.3, 10.6, "§6, v0.1"),
    ("Qwen3.6-35B (21.7 GB) in 12 GB RAM", "6 expert layers on GPU, ctx 32k, launcher", 6.9, 22.6, "§44, v0.2.3.4+"),
    ("Qwen3.6-35B, fits in RAM (24 GB)", "warm; the launcher now runs stock here", 42.9, 36.6, "§44, v0.2.3.4+"),
]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "img", "results.svg")
GREY, BLUE, TXT = "#9aa0a6", "#2f81f7", "#808080"


def main():
    rows = [r for r in ROWS if r[2] is not None and r[3] is not None]
    vmax = max(max(r[2], r[3]) for r in rows)
    x0, wmax, top, step = 372, 380, 84, 54
    h = top + step * len(rows) + 60
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="860" height="{h}" viewBox="0 0 860 {h}" font-family="Helvetica,Arial,sans-serif">',
         f'<text x="20" y="32" font-size="20" font-weight="bold" fill="{TXT}">Generation speed, tokens per second (higher is better)</text>',
         f'<rect x="20" y="44" width="14" height="14" fill="{GREY}"/><text x="40" y="56" font-size="13" fill="{TXT}">stock llama.cpp</text>',
         f'<rect x="170" y="44" width="14" height="14" fill="{BLUE}"/><text x="190" y="56" font-size="13" fill="{TXT}">with moe-cache</text>']
    for i, (title, setup, s, m, src) in enumerate(rows):
        y = top + i * step
        o.append(f'<text x="20" y="{y + 22}" font-size="14" fill="{TXT}">{title}</text>')
        o.append(f'<text x="20" y="{y + 39}" font-size="11" fill="{TXT}">{setup} ({src})</text>')
        for k, (v, col, bold) in enumerate(((s, GREY, False), (m, BLUE, True))):
            w = max(2.0, wmax * v / vmax)
            yy = y + k * 22
            fw = ' font-weight="bold"' if bold else ''
            o.append(f'<rect x="{x0}" y="{yy}" width="{w:.1f}" height="18" rx="3" fill="{col}"/>'
                     f'<text x="{x0 + w + 8:.1f}" y="{yy + 14}" font-size="13"{fw} fill="{TXT}">{v:g}</text>')
        ratio = m / s
        o.append(f'<text x="840" y="{y + 30}" font-size="13" font-weight="bold" text-anchor="end" fill="{TXT}">{ratio:.1f}x</text>')
    y = top + len(rows) * step + 12
    o.append(f'<text x="20" y="{y}" font-size="12" fill="{TXT}">Clean runs on one laptop (i7-13620H, RTX 4060 8 GB, NVMe), paired where marked in docs/RESULTS.md (section, version).</text>')
    o.append(f'<text x="20" y="{y + 20}" font-size="12" fill="{TXT}">"N GB RAM" = memory limit set with a cgroup on a 31 GB machine; the model file is evicted from the page cache before limited stock runs.</text>')
    o.append('</svg>')
    open(OUT, "w").write("\n".join(o) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
