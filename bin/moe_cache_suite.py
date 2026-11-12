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


def new_record(model, machine=None, software=None, settings=None, sections=None, profile="", notes=None, now=None, served=None, partial=False):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    sections = {k: v for k, v in (sections or {}).items() if k in SECTIONS}
    ident = hashlib.sha256(json.dumps([model, settings, served, sections, partial], sort_keys=True, default=str).encode()).hexdigest()[:6]
    return {"suite_record_schema": SUITE_SCHEMA, "created_utc": now.strftime("%Y-%m-%d %H:%M"), "id": f"{now.strftime('%Y-%m-%d')}_{re.sub(r'[^A-Za-z0-9._-]+', '-', model.get('name', 'model'))}_{ident}",
            "published": False, "profile": profile, "model": model, "machine": machine or {}, "software": software or {}, "served": served or {}, "settings": settings or {}, "sections": sections, "notes": notes or [], "partial": bool(partial)}


# ---- converters from what the existing tools write ----------------------------------------------------------------------------------------------
def pctl(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else 0


def section_hard(run, tasks_hash="", ctx=None):
    """From a tools/hard_tasks_run.py output. Tasks that ended in an error (not a wrong answer) are counted in `errors` and are not in n or pass. ctx = the context the server ran with: with no cap the
    context is the only limit, so it is part of the protocol."""
    it = [x for x in run["items"] if x and not x.get("error")]; err = sum(1 for x in run["items"] if x and x.get("error")); th = [x["think_tokens"] for x in it]; cats = {}
    for x in it:
        c = cats.setdefault(x["category"], [0, 0]); c[0] += int(x["pass"]); c[1] += 1
    return {"protocol": protocol_hash(kind="hard", tasks=tasks_hash, cap=run.get("cap", 0), ctx=ctx, loop=run.get("loop_rule", "weak-12x3"), temp=0), "n": len(it), "pass": sum(x["pass"] for x in it), "errors": err,
            "think_median": statistics.median(th) if th else 0, "think_p90": pctl(th, .9), "think_max": max(th) if th else 0, "loops": sum(x["loop"] for x in it), "loops_in_passed": sum(1 for x in it if x["loop"] and x["pass"]), "think_loops": sum(1 for x in it if x.get("think_loop")), "think_repeat_share_median": round(statistics.median(x.get("think_repeat_share", 0) for x in it), 3) if it else 0,
            "cap_hits": sum(x["cap_hit"] for x in it), "timeouts": sum(1 for x in it if x.get("stop") == "timeout"), "no_think_tag": sum(1 for x in it if x.get("think_tag") is False), "cap": run.get("cap", 0), "by_category": cats}


def task_answer_ok(text, patterns, stop="", think="off"):
    """(passed, cut_inside_thinking): the answer is looked for only after the last </think>. A text that ends (stop=limit) while the think block is still open has no answer: it is a cut,
    not a pass, even when the reasoning happens to mention the answer."""
    cut = ("<think>" in text and "</think>" not in text) or (think == "on" and stop == "limit" and "</think>" not in text)
    if cut:
        return False, True
    body = text.split("</think>")[-1] if "</think>" in text else text
    body = re.sub(r"<think>.*?</think>", "", body, flags=re.S).lower()
    return any(re.search(p, body) for p in patterns), False


def section_tasks(run, think, tasks_hash="", cap=0):
    """From a tools/quality_yardstick.py run output (the 202-task set). Scored on the text after </think> only (task_answer_ok); `cut` = answers cut off inside the thinking, `thinking_seen` = answers that
    contain a think block at all (with thinking 'off' that is a model that ignores the switch)."""
    sys.path.insert(0, os.path.join(HERE, "..", "tools"))
    its = [x for x in run["items"] if x.get("kind") == "task"]; res = [task_answer_ok(x["text"], x["answers"], x.get("stop", ""), think) for x in its]
    return {"protocol": protocol_hash(kind="tasks", tasks=tasks_hash, think=think, cap=cap, temp=0), "n": len(its), "pass": sum(1 for ok, _ in res if ok), "think": think, "cap": cap,
            "cut": sum(1 for _, c in res if c), "thinking_seen": sum(1 for x in its if "</think>" in x["text"] or "<think>" in x["text"])}


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
# What must be the same for two records to be compared (the record's `served` block says what the server really ran; machine and kit come from the record).
# A section whose results depend on a setting is NOT compared across a known difference in it (it shows '-' and a line says which); the others only get a warning line.
ENV_KEYS = ("machine", "kit", "build", "ctx", "np", "ctk", "ctv")
REFUSE = {"speed": ENV_KEYS, "memory": ENV_KEYS, "hard": ("ctx", "np", "ctk", "ctv", "build"), "tasks": (), "tasks_on": (), "quality": ()}
SINGLE_RUN_NOISE_PCT = 10.0      # one run per number: no spread is known, so only a change of 10 % or more gets a colour


def machine_id(m):
    return " | ".join(str(m[k]) for k in ("cpu_model", "ram_gib", "gpu_model", "vram_mib", "os", "platform", "ram_gb", "cpu_count") if m.get(k) not in (None, "not recorded")) or None


def env_of(rec):
    sv = rec.get("served") or {}
    return {"machine": machine_id(rec.get("machine") or {}), "kit": (rec.get("software") or {}).get("kit"), "build": sv.get("build"), "ctx": sv.get("ctx"), "np": sv.get("np"), "ctk": sv.get("ctk"), "ctv": sv.get("ctv")}


def env_diff(ra, rb):
    """({key: (a, b)} known differences, [keys not recorded in one of the two])."""
    ea, eb = env_of(ra), env_of(rb); diff, unknown = {}, []
    for k in ENV_KEYS:
        if ea[k] is None or eb[k] is None:
            unknown.append(k)
        elif ea[k] != eb[k]:
            diff[k] = (ea[k], eb[k])
    return diff, unknown


def wilson(k, n, z=1.96):
    """95 % Wilson interval (low, high) of k passes in n tasks, as fractions."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / d
    return (max(0.0, c - h), min(1.0, c + h))


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


# row: (section, label, getter, unit, better (+1 higher is better, -1 lower), kind): pass = a pass count of n tasks (Wilson interval), count = a small count (loops, cap hits), run = one measured number
def _metrics(records):
    rows = [("speed", "ttft", lambda s: s.get("ttft_ms"), "ms", -1, "run"), ("speed", "decode batch 1", lambda s: s.get("decode_b1"), "tok/s", 1, "run"), ("speed", "decode batch 4", lambda s: s.get("decode_b4"), "tok/s", 1, "run")]
    depths = sorted({b["depth"] for r in records for b in r["sections"].get("speed", {}).get("bands", [])})
    for d in depths:
        rows.insert(0, ("speed", f"decode at {d // 1024}k", _band(d, "decode_tps"), "tok/s", 1, "run")); rows.insert(1, ("speed", f"prompt at {d // 1024}k", _band(d, "prompt_tps"), "tok/s", 1, "run"))
    rows += [("memory", "server RSS", lambda s: s.get("rss_gb"), "GB", -1, "run"), ("memory", "VRAM", lambda s: s.get("vram_gb"), "GB", -1, "run"),
             ("quality", "KL mean (KV cost)", lambda s: s.get("kl_mean"), "", -1, "kl"), ("quality", "same top token (KV cost)", lambda s: s.get("top1_pct"), "%", 1, "kl"),
             ("tasks", f"tasks passed of {_n(records, 'tasks')}, thinking off", lambda s: s.get("pass"), "", 1, "pass"), ("tasks_on", f"tasks passed of {_n(records, 'tasks_on')}, thinking on", lambda s: s.get("pass"), "", 1, "pass"),
             ("hard", f"hard tasks passed of {_n(records, 'hard')}", lambda s: s.get("pass"), "", 1, "pass"), ("hard", "thinking tokens, median", lambda s: s.get("think_median"), "", -1, "run"), ("hard", "thinking tokens, p90", lambda s: s.get("think_p90"), "", -1, "run"),
             ("hard", "loops in the final answer", lambda s: s.get("loops"), "", -1, "count"), ("hard", "thinking repetition share (median)", lambda s: s.get("think_repeat_share_median"), "", -1, "run"), ("hard", "cap hits", lambda s: s.get("cap_hits"), "", -1, "count")]
    return rows


def _same_model(records):
    key = lambda r: (r["model"].get("name"), r["model"].get("quant"), r["model"].get("file"))
    return all(key(r) == key(records[0]) for r in records)


def compare(records, noise_pct=SINGLE_RUN_NOISE_PCT):
    """Returns {"records": [labels], "rows": [{section, label, unit, kind, values, ns, deltas_pct, marks}], "skipped": [text], "warnings": [text]}; the first record is the reference.
    A section is compared only between records whose protocol hash equals the reference's and whose known server settings (REFUSE) are equal; the others show '-' and a line in `skipped` says
    which section and why. Marks: pass counts use a 95 % Wilson interval (overlap = '='); counts of 0-1 apart are '='; one-run numbers need `noise_pct` (default 10 %); the KL rows carry no
    mark between different models (KL against the model's own f16 KV is what the KV cache costs, not how good the model is)."""
    labels = [f"{r['model'].get('name', '?')} {r['model'].get('quant', '')}".strip() for r in records]; skipped = []; warnings = []; rows = []; same_model = _same_model(records)
    for i in range(1, len(records)):
        diff, unknown = env_diff(records[0], records[i])
        for k, (x, y) in diff.items():
            warnings.append(f"{labels[i]} was measured with another {k} than {labels[0]} ({y} against {x})")
        if unknown:
            warnings.append(f"{labels[i]}: {', '.join(unknown)} not recorded in one of the two records, so it could not be checked")
    for sec, label, get, unit, better, kind in _metrics(records):
        ref_s = records[0]["sections"].get(sec); vals = []; ns = []
        for i, r in enumerate(records):
            s = r["sections"].get(sec); v = None
            if s is not None:
                why = None
                if ref_s is not None and s.get("protocol") != ref_s.get("protocol"):
                    why = f"{sec}: {labels[i]} was measured with a different protocol than {labels[0]} ({s.get('protocol')} against {ref_s.get('protocol')}); not compared"
                elif i > 0 and ref_s is not None:
                    bad = [f"{k} ({x} against {y})" for k, (x, y) in env_diff(records[0], r)[0].items() if k in REFUSE.get(sec, ())]
                    if bad:
                        why = f"{sec}: {labels[i]} not compared with {labels[0]}: another " + ", ".join(bad)
                if why:
                    if why not in skipped: skipped.append(why)
                else:
                    v = get(s)
            vals.append(v); ns.append(s.get("n") if (s is not None and v is not None) else None)
        if all(v is None for v in vals):
            continue
        base = vals[0]; deltas = []; marks = []
        for i, v in enumerate(vals):
            if i == 0 or v is None or base is None:
                deltas.append(None); marks.append(""); continue
            d = v - base; pct = (d / base * 100) if base else None; deltas.append(pct)
            if kind == "kl" and not same_model:
                mk = ""
            elif kind == "pass" and ns[0] and ns[i]:
                a, b = wilson(base, ns[0]), wilson(v, ns[i]); mk = "=" if not (b[0] > a[1] or a[0] > b[1]) else ("better" if d * better > 0 else "worse")
            elif kind == "count":
                mk = "=" if abs(d) <= 1 else ("better" if d * better > 0 else "worse")
            else:
                mk = "=" if d == 0 or (pct is not None and abs(pct) < noise_pct) else ("better" if d * better > 0 else "worse")
            marks.append(mk)
        rows.append({"section": sec, "label": label, "unit": unit, "kind": kind, "values": vals, "ns": ns, "deltas_pct": deltas, "marks": marks})
    return {"records": labels, "rows": rows, "skipped": skipped, "warnings": warnings}


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
    return "\n".join(lines + [""] + res["skipped"] + res.get("warnings", []) + ["(one run per number; pass counts: 95 % interval; speed: 10 % or more; counts: 2 or more apart)"])


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


RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}      # Windows device names


def record_path(d, rid):
    if not ID_RE.match(rid or "") or rid.split(".")[0].lower() in RESERVED:
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
    rec["published"] = False                 # an imported record is private here: publishing is a separate, explicit step of this user
    rid = rec["id"]; path = record_path(d, rid)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(path) and {**load(path), "published": False} != rec:
        raise ValueError(f"a different record with the id {rid} exists")
    if not os.path.exists(path):
        _write_json(path, rec)
    return rid


def _write_json(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, path)


def set_published(d, rid, value=True):
    path = record_path(d, rid); rec = load(path); rec["published"] = bool(value)
    _write_json(path, rec)
    return rec["published"]


def compare_ids(d, ids, noise=SINGLE_RUN_NOISE_PCT):
    recs = [load(record_path(d, i)) for i in ids]
    res = compare(recs, noise); res["meta"] = [{"id": r["id"], "name": r["model"].get("name", "?"), "quant": r["model"].get("quant", ""), "size_gb": r["model"].get("size_gb"), "created": r.get("created_utc", ""), "published": bool(r.get("published")), "partial": bool(r.get("partial")), "served": " · ".join(x for x in [f'ctx {(r.get("served") or {}).get("ctx")}' if (r.get("served") or {}).get("ctx") else "", f'KV {(r.get("served") or {}).get("ctk")}/{(r.get("served") or {}).get("ctv")}' if (r.get("served") or {}).get("ctk") else "", f'{(r.get("served") or {}).get("np")} slots' if (r.get("served") or {}).get("np") else ""] if x), "machine": (r.get("machine") or {}).get("gpu_model") or (r.get("machine") or {}).get("cpu_model") or (r.get("machine") or {}).get("platform", "")} for r in recs]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate"); v.add_argument("files", nargs="+")
    c = sp.add_parser("compare"); c.add_argument("files", nargs="+"); c.add_argument("--json", action="store_true"); c.add_argument("--noise", type=float, default=SINGLE_RUN_NOISE_PCT)
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
        _write_json(a.out, rec); print("wrote", a.out, rec["id"]); return 0
    if a.cmd == "publish":
        rec = load(a.file); rec["published"] = True; _write_json(a.file, rec); print("marked published:", a.file, "(nothing reads published suite records yet; the board needs the owner's review)"); return 0


if __name__ == "__main__":
    sys.exit(main())
