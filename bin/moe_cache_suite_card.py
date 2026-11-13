#!/usr/bin/env python3
"""TEST-1142: renders suite records as one self-contained HTML page: one card per model (columns), the first one is the reference, every tile shows the value and its change against the reference
(green better, red worse, grey = no difference outside the noise of one run); sections measured with another protocol or on another server setup show '-' and a note. No external files, no scripts; every field from a record is escaped.
  moe_cache.py bench-suite card REC.json [REC2.json ...] --out card.html [--title TEXT]"""
import argparse, html, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import moe_cache_suite as S      # noqa: E402

CSS = """:root{--bg:#f6f4ef;--card:#fff;--ink:#1d1b16;--mute:#6d6a60;--line:#e2ded2;--good:#1e7b43;--bad:#b3261e;--flat:#8b887c;--accent:#b4531f}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#16150f;--card:#201f17;--ink:#f0ede2;--mute:#a09c8e;--line:#35332a;--good:#5fc486;--bad:#ef7d74;--flat:#9a9789;--accent:#e08a55}}
:root[data-theme=dark]{--bg:#16150f;--card:#201f17;--ink:#f0ede2;--mute:#a09c8e;--line:#35332a;--good:#5fc486;--bad:#ef7d74;--flat:#9a9789;--accent:#e08a55}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif;padding:20px 16px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mute);margin:0 0 16px;font-size:13px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 14px 10px}
.card h2{font-size:16px;margin:0}.meta{color:var(--mute);font-size:12px;margin:2px 0 10px}.ref{color:var(--accent);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.sec{margin:10px 0 2px;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);border-top:1px solid var(--line);padding-top:8px}
.row{display:flex;justify-content:space-between;gap:8px;padding:2px 0}.row .v{font-variant-numeric:tabular-nums;text-align:right}
.better{color:var(--good)}.worse{color:var(--bad)}.flat{color:var(--flat)}.d{font-size:12px;margin-left:6px}
.tag{color:var(--mute);font-size:11px;margin-left:6px}.note{margin-top:14px;color:var(--mute);font-size:12px}.priv{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:11px;color:var(--mute)}"""


def fmt(v, unit):
    if v is None:
        return "-"
    t = f"{v:,.0f}" if isinstance(v, (int, float)) and abs(v) >= 100 else (f"{v:g}" if isinstance(v, (int, float)) else str(v))
    return t + (f" {unit}" if unit else "")


def render(records, title="Model comparison"):
    res = S.compare(records); e = html.escape; cols = []
    for i, rec in enumerate(records):
        m = rec["model"]; rows_html = []; last = None
        for r in res["rows"]:
            if r["section"] != last:
                rows_html.append(f'<div class="sec">{e({"speed": "speed", "memory": "memory", "quality": "KV cache cost (KL vs the same model with f16 KV)", "tasks": "tasks", "tasks_on": "tasks", "hard": "hard reasoning"}.get(r["section"], r["section"]))}</div>'); last = r["section"]
            d, mk = r["deltas_pct"][i], r["marks"][i]
            chip = "" if d is None or not mk else f'<span class="d {"flat" if mk == "=" else mk}">{"=" if mk == "=" else f"{d:+.0f} %"}</span>'
            tag = "" if r["values"][i] is None else (f' <span class="tag">n={r["ns"][i]}</span>' if r["kind"] == "pass" and r["ns"][i] else (' <span class="tag">1 run</span>' if r["kind"] in ("run", "info") else ""))
            rows_html.append(f'<div class="row"><span>{e(r["label"])}</span><span class="v">{e(fmt(r["values"][i], r["unit"]))}{chip}{tag}</span></div>')
        sv = rec.get("served") or {}; mc = rec.get("machine") or {}
        meta = " · ".join(x for x in [m.get("quant", ""), f'{m["size_gb"]} GB' if m.get("size_gb") else "", (mc.get("gpu_model") or mc.get("cpu_model") or mc.get("platform") or "")[:40], rec.get("created_utc", "")] if x)
        served = " · ".join(x for x in [f'ctx {sv["ctx"]}' if sv.get("ctx") else "", f'KV {sv["ctk"]}/{sv["ctv"]}' if sv.get("ctk") else "", f'{sv["np"]} slots' if sv.get("np") else "", sv.get("build") or ""] if x) or "server settings not recorded"
        cols.append(f'<section class="card"><div class="ref">{"reference" if i == 0 else "vs " + e(res["records"][0])}</div><h2>{e(m.get("name", "?"))}</h2><div class="meta">{e(meta)}'
                    f' <span class="priv">{"published" if rec.get("published") else "private"}</span>{" <span class=priv>partial</span>" if rec.get("partial") else ""}</div><div class="meta">{e(served)}</div>{"".join(rows_html)}</section>')
    notes = "".join(f"<div>{e(s)}</div>" for s in res["skipped"] + res["warnings"])
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title><style>{CSS}</style></head><body>'
            f'<h1>{e(title)}</h1><p class="sub">The first card is the reference. Every number is one run. Green or red only where the difference is outside the noise of one run: pass counts use a 95 % interval, speed and memory need 10 % or more. Where the server settings differ (see each card) the notes below say so.</p><div class="grid">{"".join(cols)}</div>'
            f'<div class="note">{notes}</div></body></html>')


def main(argv):
    ap = argparse.ArgumentParser(prog="bench-suite card"); ap.add_argument("files", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--title", default="Model comparison")
    a = ap.parse_args(argv); recs = [S.load(f) for f in a.files]
    for f, r in zip(a.files, recs):
        pr = S.validate_record(r)
        if pr:
            print(f"{f}: invalid: {'; '.join(pr)}", file=sys.stderr); return 1
    open(a.out, "w", encoding="utf-8").write(render(recs, a.title)); print("wrote", a.out); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
