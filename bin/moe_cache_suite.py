#!/usr/bin/env python3
"""TEST-1142: the benchmark-suite record and the comparison of suite records (standard library only; Linux and Windows).
One suite record = one model + quant + settings on one machine, with these sections (each optional, each with its own protocol hash):
  speed    decode / prompt tok/s by depth band, TTFT, decode at batch 1 and 4      memory  RSS, VRAM, commit
  quality  KL and same-top-token against a named reference                          tasks   the 202-task set (thinking on or off)
  hard     the hard long-reasoning set: pass, thinking tokens, loops, cap hits
Records are private by default (benchmarks/suites/); `publish` marks one as published (nothing else reads them yet). `compare` puts records side by side, the first one is the reference,
and refuses to compare a section whose protocol hash differs (it says which). Schema: docs/suite_record.schema.json.
  moe_cache_suite.py validate REC.json [...]        moe_cache_suite.py compare REC.json REC2.json [...] [--json]        moe_cache_suite.py show REC.json
  moe_cache_suite.py new --model NAME [--quant Q] [--size-gb N] [--section NAME=FILE.json ...] --out REC.json     (sections from the converters below, or hand-written)
  moe_cache_suite.py publish REC.json"""
import argparse, datetime, hashlib, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE_SCHEMA = 1
SCHEMA_PATH = os.path.join(HERE, "..", "docs", "suite_record.schema.json")
SECTIONS = ("speed", "memory", "quality", "tasks", "tasks_on", "hard")      # tasks = thinking off, tasks_on = thinking on (the section's `think` says which)


def protocol_hash(**parts):
    """Short stable hash of everything that must be equal for two section results to be comparable (task file hash, windows file hash, request settings, cap, KV type, thinking mode...)."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:12]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def new_record(model, machine=None, software=None, settings=None, sections=None, profile="", notes=None, now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    sections = {k: v for k, v in (sections or {}).items() if k in SECTIONS}
    ident = hashlib.sha256(json.dumps([model, settings, sections], sort_keys=True, default=str).encode()).hexdigest()[:6]
    return {"suite_record_schema": SUITE_SCHEMA, "created_utc": now.strftime("%Y-%m-%d %H:%M"), "id": f"{now.strftime('%Y-%m-%d')}_{re.sub(r'[^A-Za-z0-9._-]+', '-', model.get('name', 'model'))}_{ident}",
            "published": False, "profile": profile, "model": model, "machine": machine or {}, "software": software or {}, "settings": settings or {}, "sections": sections, "notes": notes or []}


# ---- converters from what the existing tools write ----------------------------------------------------------------------------------------------
def pctl(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else 0


def section_hard(run, tasks_hash=""):
    """From a tools/hard_tasks_run.py output."""
    it = [x for x in run["items"] if x]; th = [x["think_tokens"] for x in it]; cats = {}
    for x in it:
        c = cats.setdefault(x["category"], [0, 0]); c[0] += int(x["pass"]); c[1] += 1
    return {"protocol": protocol_hash(kind="hard", tasks=tasks_hash, cap=run.get("cap", 0), temp=0), "n": len(it), "pass": sum(x["pass"] for x in it), "think_median": statistics.median(th) if th else 0, "think_p90": pctl(th, .9),
            "think_max": max(th) if th else 0, "loops": sum(x["loop"] for x in it), "cap_hits": sum(x["cap_hit"] for x in it), "cap": run.get("cap", 0), "by_category": cats}


def section_tasks(run, think, tasks_hash="", cap=0):
    """From a tools/quality_yardstick.py run output (the 202-task set)."""
    sys.path.insert(0, os.path.join(HERE, "..", "tools")); import quality_yardstick as Y
    its = [x for x in run["items"] if x.get("kind") == "task"]
    ok = [bool(any(re.search(p, Y.strip_think(x["text"]).lower()) for p in x["answers"])) for x in its]
    return {"protocol": protocol_hash(kind="tasks", tasks=tasks_hash, think=think, cap=cap, temp=0), "n": len(its), "pass": sum(ok), "think": think, "cap": cap}


_KL = {"kl_mean": r"Mean\s+KLD:\s+(-?[\d.]+)", "kl_median": r"Median\s+KLD:\s+(-?[\d.]+)", "kl_p90": r"90\.0%\s+KLD:\s+(-?[\d.]+)", "top1_pct": r"Same top p:\s+([\d.]+)"}


def section_quality(log_text, ref, windows_hash="", windows=0, tokens=0, kv=""):
    """From a llama-perplexity --kl-divergence log."""
    out = {"ref": ref, "windows": windows, "tokens": tokens, "protocol": protocol_hash(kind="quality", ref=ref, windows=windows_hash, kv=kv)}
    for k, pat in _KL.items():
        m = re.search(pat, log_text)
        if m:
            out[k] = float(m.group(1))
    if "top1_pct" not in out:
        raise ValueError("no 'Same top p' line in the KL log")
    return out


# ---- validate ----------------------------------------------------------------------------------------------------------------------------------
def validate_record(rec):
    """List of problems (empty when valid): the schema file plus the cross-checks a schema cannot say."""
    sys.path.insert(0, HERE); import moe_cache_record as R
    schema = json.load(open(SCHEMA_PATH, encoding="utf-8")); probs = list(R.validate(rec, schema))
    s = rec.get("sections", {})
    if "hard" in s and s["hard"].get("pass", 0) > s["hard"].get("n", 0):
        probs.append("hard: pass > n")
    for k in ("tasks", "tasks_on"):
        if k in s and s[k].get("pass", 0) > s[k].get("n", 0):
            probs.append(f"{k}: pass > n")
    if "quality" in s and not 0 <= s["quality"].get("top1_pct", 0) <= 100:
        probs.append("quality: top1_pct outside 0-100")
    return probs


# ---- compare -----------------------------------------------------------------------------------------------------------------------------------
# (section, label, getter, unit, better: +1 higher is better, -1 lower is better)
def _band(depth, key):
    def g(s):
        for b in s.get("bands", []):
            if b["depth"] == depth:
                return b.get(key)
    return g


def _n(records, sec):
    """The task count of the first record that has the section (the protocol hash makes it the same task set in every compared record)."""
    for r in records:
        if sec in r["sections"]:
            return r["sections"][sec].get("n", "?")
    return "?"


def _metrics(records):
    rows = [("speed", "ttft", lambda s: s.get("ttft_ms"), "ms", -1), ("speed", "decode batch 1", lambda s: s.get("decode_b1"), "tok/s", 1), ("speed", "decode batch 4", lambda s: s.get("decode_b4"), "tok/s", 1)]
    depths = sorted({b["depth"] for r in records for b in r["sections"].get("speed", {}).get("bands", [])})
    for d in depths:
        rows.insert(0, ("speed", f"decode at {d // 1024}k", _band(d, "decode_tps"), "tok/s", 1)); rows.insert(1, ("speed", f"prompt at {d // 1024}k", _band(d, "prompt_tps"), "tok/s", 1))
    rows += [("memory", "server RSS", lambda s: s.get("rss_gb"), "GB", -1), ("memory", "VRAM", lambda s: s.get("vram_gb"), "GB", -1),
             ("quality", "KL mean", lambda s: s.get("kl_mean"), "", -1), ("quality", "same top token", lambda s: s.get("top1_pct"), "%", 1),
             ("tasks", f"tasks passed of {_n(records, 'tasks')}, thinking off", lambda s: s.get("pass"), "", 1), ("tasks_on", f"tasks passed of {_n(records, 'tasks_on')}, thinking on", lambda s: s.get("pass"), "", 1),
             ("hard", "hard tasks passed", lambda s: s.get("pass"), "", 1), ("hard", "thinking tokens, median", lambda s: s.get("think_median"), "", -1), ("hard", "thinking tokens, p90", lambda s: s.get("think_p90"), "", -1),
             ("hard", "loops", lambda s: s.get("loops"), "", -1), ("hard", "cap hits", lambda s: s.get("cap_hits"), "", -1)]
    return rows


def compare(records, noise_pct=2.0):
    """Returns {"records": [labels], "rows": [{section, label, unit, values, deltas, marks}], "skipped": [text]}; the first record is the reference. A section is compared only between records whose
    protocol hash equals the reference's; the others show '-' and a line in `skipped` says which section and why."""
    labels = [f"{r['model'].get('name', '?')} {r['model'].get('quant', '')}".strip() for r in records]; skipped = []; rows = []
    for sec, label, get, unit, better in _metrics(records):
        ref_s = records[0]["sections"].get(sec); vals = []
        for i, r in enumerate(records):
            s = r["sections"].get(sec); v = None
            if s is not None:
                if ref_s is not None and s.get("protocol") != ref_s.get("protocol"):
                    msg = f"{sec}: {labels[i]} was measured with a different protocol than {labels[0]} ({s.get('protocol')} against {ref_s.get('protocol')}); not compared"
                    if msg not in skipped: skipped.append(msg)
                else:
                    v = get(s)
            vals.append(v)
        if all(v is None for v in vals):
            continue
        base = vals[0]; deltas = []; marks = []
        for i, v in enumerate(vals):
            if i == 0 or v is None or base is None:
                deltas.append(None); marks.append(""); continue
            d = v - base; pct = (d / base * 100) if base else None; deltas.append(pct)
            marks.append("=" if d == 0 or (pct is not None and abs(pct) <= noise_pct) else ("better" if d * better > 0 else "worse"))
        rows.append({"section": sec, "label": label, "unit": unit, "values": vals, "deltas_pct": deltas, "marks": marks})
    return {"records": labels, "rows": rows, "skipped": skipped}


def compare_text(res):
    w = max(len(x) for x in res["records"]) + 2; lines = [f"{'':26s}" + "".join(f"{x:>{w}s}" for x in res["records"])]
    for r in res["rows"]:
        cells = []
        for v, d, m in zip(r["values"], r["deltas_pct"], r["marks"]):
            if v is None: cells.append("-")
            else:
                t = f"{v:g}" if isinstance(v, (int, float)) else str(v)
                if d is not None and m: t += f" ({d:+.0f} % {m})" if m != "=" else " (=)"
                cells.append(t)
        lines.append(f"{r['label'] + (' ' + r['unit'] if r['unit'] else ''):26s}" + "".join(f"{c:>{w}s}" for c in cells))
    return "\n".join(lines + [""] + res["skipped"])


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---- the records folder (private by default) and what the control page needs ---------------------------------------------------------------------
def default_dir():
    """benchmarks/suites in a checkout (git-ignored), else a folder in the user's data directory. Private to this user."""
    repo = os.path.join(HERE, "..", "benchmarks")
    if os.path.isdir(repo):
        return os.path.abspath(os.path.join(repo, "suites"))
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "moe-cache", "suites")


ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
MAX_RECORD_BYTES = 5_000_000


def record_path(d, rid):
    if not ID_RE.match(rid or ""):
        raise ValueError("bad record id")
    return os.path.join(d, rid + ".json")


def list_records(d):
    out = []
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not f.endswith(".json"):
            continue
        try:
            r = load(os.path.join(d, f))
        except (OSError, ValueError):
            continue
        if r.get("suite_record_schema") != SUITE_SCHEMA:
            continue
        m = r.get("model", {})
        out.append({"id": r.get("id", f[:-5]), "name": m.get("name", "?"), "quant": m.get("quant", ""), "created": r.get("created_utc", ""), "profile": r.get("profile", ""), "published": bool(r.get("published")), "sections": sorted(r.get("sections", {}))})
    return sorted(out, key=lambda x: x["created"], reverse=True)


def import_text(d, text):
    """Validate a pasted or dropped record and store it; returns its id. Never overwrites a record with another content (the id carries a hash of the content)."""
    if len(text) > MAX_RECORD_BYTES:
        raise ValueError("record too large")
    rec = json.loads(text); probs = validate_record(rec) if isinstance(rec, dict) else ["not an object"]
    if probs:
        raise ValueError("invalid record: " + "; ".join(probs[:3]))
    rid = rec["id"]; path = record_path(d, rid)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(path) and load(path) != rec:
        raise ValueError(f"a different record with the id {rid} exists")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return rid


def set_published(d, rid, value=True):
    path = record_path(d, rid); rec = load(path); rec["published"] = bool(value)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return rec["published"]


def compare_ids(d, ids, noise=2.0):
    recs = [load(record_path(d, i)) for i in ids]
    res = compare(recs, noise); res["meta"] = [{"id": r["id"], "name": r["model"].get("name", "?"), "quant": r["model"].get("quant", ""), "size_gb": r["model"].get("size_gb"), "created": r.get("created_utc", ""), "published": bool(r.get("published")), "machine": r.get("machine", {}).get("platform", "")} for r in recs]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate"); v.add_argument("files", nargs="+")
    c = sp.add_parser("compare"); c.add_argument("files", nargs="+"); c.add_argument("--json", action="store_true"); c.add_argument("--noise", type=float, default=2.0)
    s = sp.add_parser("show"); s.add_argument("file")
    n = sp.add_parser("new"); n.add_argument("--model", required=True); n.add_argument("--quant", default=""); n.add_argument("--size-gb", type=float); n.add_argument("--section", action="append", default=[]); n.add_argument("--out", required=True); n.add_argument("--profile", default="")
    p = sp.add_parser("publish"); p.add_argument("file")
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        bad = 0
        for f in a.files:
            pr = validate_record(load(f)); bad += bool(pr); print(f"{f}: " + ("ok" if not pr else "; ".join(pr)))
        return 1 if bad else 0
    if a.cmd == "compare":
        recs = [load(f) for f in a.files]; res = compare(recs, a.noise); print(json.dumps(res, indent=1) if a.json else compare_text(res)); return 0
    if a.cmd == "show":
        print(compare_text(compare([load(a.file)]))); return 0
    if a.cmd == "new":
        secs = {}
        for item in a.section:
            name, _, path = item.partition("="); secs[name] = load(path)
        model = {"name": a.model, "quant": a.quant}; model.update({"size_gb": a.size_gb} if a.size_gb else {})
        rec = new_record(model, sections=secs, profile=a.profile); pr = validate_record(rec)
        if pr: print("invalid:", "; ".join(pr), file=sys.stderr); return 1
        json.dump(rec, open(a.out, "w"), indent=1); print("wrote", a.out, rec["id"]); return 0
    if a.cmd == "publish":
        rec = load(a.file); rec["published"] = True; json.dump(rec, open(a.file, "w"), indent=1); print("marked published:", a.file, "(nothing reads published suite records yet; the board needs the owner's review)"); return 0


if __name__ == "__main__":
    sys.exit(main())
