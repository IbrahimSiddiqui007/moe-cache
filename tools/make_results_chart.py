#!/usr/bin/env python3
"""Draw docs/img/results.svg (the README chart) from the ROWS table below. Every number must come from docs/RESULTS.md: the section is in each row.
   python3 tools/make_results_chart.py            (writes docs/img/results.svg)"""
import os

# (title, GPU / context line, stock tok/s, moe-cache tok/s (exact), source, optional (fastest opt-in mode tok/s, its name))
# GPU line, same form for every row: what sits in the 8 GB of VRAM (expert layers with their size, the rest of the model), then the context
ROWS = [
    ("GLM-5.3-Flash (102 GB) in 24 GB RAM", "GPU: 0 expert layers + rest 6.7 GB, ctx 4k", 0.32, 1.75, "§26 v0.2.3.2", (2.13, "--fast")),
    ("Qwen3.8-Flash-Next (68 GB) in 24 GB RAM", "GPU: 2 expert layers 1.5 GB + rest 3.8 GB, ctx 4k", 5.2, 12.5, "§44 v0.2.3.4", (17.6, "--hybrid")),
    ("Qwen3.5-122B IQ2_XXS (36.6 GB) in 24 GB RAM", "GPU: 1 expert layer 0.7 GB + rest 4.3 GB, ctx 4k", 1.89, 7.72, "§45 v0.2.3.4 (stock: shorter requests)", (8.97, "--fast")),
    ("gpt-oss-120b (63 GB) in 24 GB RAM", "GPU: 2 expert layers + rest, 6.0 GB in use, ctx 8k", 0.37, 4.47, "§12b, §12d v0.1 (one run each)", None),
    ("Qwen3-Next-80B (48.5 GB) in 24 GB RAM", "GPU: 0 expert layers + rest 2.6 GB, ctx 8k", 2.3, 10.6, "§6 v0.1", None),
    ("Qwen3.6-35B (21.7 GB) in 12 GB RAM", "GPU: 6 expert layers 2.9 GB + rest 1.7 GB, ctx 32k", 6.9, 22.6, "§44 v0.2.3.4", None),
    ("Qwen3.6-35B (21.7 GB), fits in 24 GB RAM", "GPU: expert cache 4.9 GB + rest 1.7 GB, ctx 4k", 42.9, 36.6, "§44 v0.2.3.4", None),
]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "img", "results.svg")
GREY, BLUE, LIGHT, TXT = "#9aa0a6", "#2f81f7", "#8ab4f8", "#808080"


def main():
    rows = [r for r in ROWS if r[2] is not None and r[3] is not None]
    vmax = max(max(r[2], r[3], (r[5] or (0,))[0]) for r in rows)
    x0, wmax, top = 372, 380, 84
    heights = [80 if r[5] else 64 for r in rows]
    h = top + sum(heights) + 60
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="860" height="{h}" viewBox="0 0 860 {h}" font-family="Helvetica,Arial,sans-serif">',
         f'<text x="20" y="32" font-size="20" font-weight="bold" fill="{TXT}">Generation speed, tokens per second (higher is better)</text>',
         f'<rect x="20" y="44" width="14" height="14" fill="{GREY}"/><text x="40" y="56" font-size="13" fill="{TXT}">stock llama.cpp</text>',
         f'<rect x="170" y="44" width="14" height="14" fill="{BLUE}"/><text x="190" y="56" font-size="13" fill="{TXT}">with moe-cache (same tokens as stock)</text>',
         f'<rect x="460" y="44" width="14" height="14" fill="{LIGHT}"/><text x="480" y="56" font-size="13" fill="{TXT}">moe-cache, fastest opt-in mode (approximate)</text>']
    y = top
    for i, (title, setup, s, m, src, opt) in enumerate(rows):
        o.append(f'<text x="20" y="{y + 22}" font-size="14" fill="{TXT}">{title}</text>')
        o.append(f'<text x="20" y="{y + 39}" font-size="11" fill="{TXT}">{setup}</text>')
        o.append(f'<text x="20" y="{y + 54}" font-size="10" fill="{TXT}">{src}</text>')
        bars = [(s, GREY, False, ""), (m, BLUE, True, "")] + ([(opt[0], LIGHT, False, " " + opt[1])] if opt else [])
        for k, (v, col, bold, tag) in enumerate(bars):
            w = max(2.0, wmax * v / vmax)
            yy = y + k * 22
            fw = ' font-weight="bold"' if bold else ''
            o.append(f'<rect x="{x0}" y="{yy}" width="{w:.1f}" height="18" rx="3" fill="{col}"/>'
                     f'<text x="{x0 + w + 8:.1f}" y="{yy + 14}" font-size="13"{fw} fill="{TXT}">{v:g}{tag}</text>')
        ratio = m / s
        o.append(f'<text x="840" y="{y + 30}" font-size="13" font-weight="bold" text-anchor="end" fill="{TXT}">{ratio:.1f}x</text>')
        y += heights[i]
    y += 12
    o.append(f'<text x="20" y="{y}" font-size="12" fill="{TXT}">Clean runs on one laptop (i7-13620H, RTX 4060 8 GB, NVMe), paired where marked in docs/RESULTS.md (section, version).</text>')
    o.append(f'<text x="20" y="{y + 20}" font-size="12" fill="{TXT}">"N GB RAM" = memory limit set with a cgroup on a 31 GB machine; the model file is evicted from the page cache before limited stock runs.</text>')
    o.append('</svg>')
    open(OUT, "w").write("\n".join(o) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
