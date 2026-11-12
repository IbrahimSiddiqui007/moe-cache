#!/usr/bin/env python3
"""moe_cache_claims.py: runs, claims and the board (benchmarks/, docs/BENCHMARK_TEMPLATE.md). Standard library only.

  benchmarks/runs/DATE_MACHINE_SHORTID.json   one bench-record record per run set; append-only, never edited (the raw evidence)
  benchmarks/claims/ID.md                     a number we publish: the statement, the runs behind it, and what it is valid for
  benchmarks/BOARD.md, JOURNAL.md             generated: what is true now, and what was run when

  ingest RECORD.json            copy a record into runs/ under its canonical name
  ingest-issue N | --file BODY  the record a person pasted into a GitHub issue (the report's JSON block): validate it, write runs/DATE_community-N_ID.json;
                                --check only validates and prints the comment for the issue (the board row it would give). Reads text, runs nothing from it.
  claim add|show|list           create a claim from run ids; show one with its computed values; list all with their status
  board                         write BOARD.md and JOURNAL.md (--check: exit 1 if they are out of date)
  lint [FILES]                  README.md, docs/RESULTS_CURRENT.md and docs/benchmark_card.html: every performance number needs a `<!-- claim:ID -->` of a current claim
  replay CLAIM                  the commands that reproduce a claim; with --record NEW.json compare a new record with the claim inside a noise band

A claim's status is computed, never typed: current / stale (the moe-cache version or the llama.cpp pin moved past what the claim is valid for) / superseded (a
newer claim covers the same machine, model, setting, arm and metric) / withdrawn (written by hand, with the reason). Every number on the board is the mean of
runs in a run file; a value that was never recorded stays `not recorded`."""
import argparse, datetime, decimal, glob, hashlib, json, os, re, shlex, subprocess, sys

import moe_cache_record as rr

NR = rr.NR
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
BENCH = os.path.join(ROOT, "benchmarks")
METRICS = {"prompt_tps": "prompt tok/s", "decode_tps": "decode tok/s"}
MARK = re.compile(r"<!--\s*claim:([A-Za-z0-9._-]+)\s*-->")
NOISE_MIN_PCT = 5.0


def isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ---------------------------------------------------------------- repo facts the status is computed against

def vtuple(v):
    try:
        return tuple(int(p) for p in str(v).split("."))
    except ValueError:
        return None


def current_version(root=ROOT):
    m = re.search(r'^VERSION\s*=\s*"([^"]+)"', open(os.path.join(root, "bin", "moe_cache.py"), encoding="utf-8").read(), re.M)
    return m.group(1) if m else NR


def current_pin(root=ROOT):
    """The llama.cpp build the kits are built with: LLAMA_TAG in tools/make_windows_kit.sh."""
    try:
        m = re.search(r"LLAMA_TAG=b(\d+)", open(os.path.join(root, "tools", "make_windows_kit.sh"), encoding="utf-8").read())
    except OSError:
        return None
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- runs

def canonical(rec):
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def default_machine(rec):
    g = rec["hardware"].get("gpu_model", NR)
    words = [w for w in re.split(r"\s+", str(g)) if w.lower() not in ("amd", "nvidia", "geforce", "radeon", "laptop", "gpu", "intel", "graphics")]
    return slug(" ".join(words)) or "machine"


def run_parts(run_id):
    date, machine, short = run_id.split("_", 2)
    return date, machine, short


def load_runs(bdir=BENCH):
    runs = {}
    for f in sorted(glob.glob(os.path.join(bdir, "runs", "*.json"))):
        runs[os.path.basename(f)[:-5]] = json.load(open(f, encoding="utf-8"))
    return runs


def ingest(path, bdir=BENCH, machine=None, date=None, force=False):
    """Copy a record into runs/ as DATE_MACHINE_SHORTID.json (SHORTID = 6 hex of the content hash). Same content again: no change. Returns (run id, written?)."""
    rec = json.load(open(path, encoding="utf-8"))
    sch = rr.load_schema()
    errs = rr.validate(rec, sch) if sch else []
    if errs and not force:
        raise ValueError("the record does not validate (use `bench-record validate`): " + "; ".join(errs[:3]))
    date = date or str(rec.get("created_utc", ""))[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError("no date: the record's created_utc is not YYYY-MM-DD..., pass --date")
    machine = slug(machine) if machine else default_machine(rec)
    short = hashlib.sha256(canonical(rec).encode("utf-8")).hexdigest()[:6]
    rid = f"{date}_{machine}_{short}"
    dest = os.path.join(bdir, "runs", rid + ".json")
    if os.path.exists(dest):
        return rid, False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    return rid, True


# ---------------------------------------------------------------- records from issues (community)

REPO = "IbrahimSiddiqui007/moe-cache"
MAX_ISSUE_BODY = 2_000_000
COMMENT_MARK = "<!-- moe-cache-record-check -->"


def extract_record(body):
    """The record a report or the issue form's JSON box carries: the biggest ```json block (or bare JSON object) whose top-level key is record_schema.
    Returns (record or None, why not). Only json.loads reads the text: nothing in an issue is executed."""
    body = str(body or "")
    if len(body) > MAX_ISSUE_BODY:
        return None, "the issue text is bigger than 2 MB"
    cands = re.findall(r"```(?:json)?\s*\n(.*?)\n```", body, re.S) or [body]
    best, why = None, "no JSON block with a benchmark record in the issue (paste the whole report, or the file from `bench-record assemble`)"
    for c in cands:
        c = c.strip()
        if not c.startswith("{"):
            continue
        try:
            d = json.loads(c)
        except ValueError as e:
            why = f"a JSON block does not parse ({str(e)[:80]}): was it cut when pasting?"
            continue
        if isinstance(d, dict) and "record_schema" in d and (best is None or len(c) > len(json.dumps(best))):
            best = d
    return best, ("" if best else why)


def board_rows(rec):
    """[(metric label, stock mean, moe-cache mean, ratio, runs per arm, noise %, verdict)] for the record: the numbers a board row would show, with the 10 % rule."""
    rows = []
    for metric, label in METRICS.items():
        a, na, sa = arm_stats(rec, "stock", metric)
        b, nb, sb = arm_stats(rec, "moe-cache", metric)
        if not (isnum(a) and isnum(b) and a):
            rows.append((label, a, b, None, (na, nb), None, "no ratio (one arm has no measurement)"))
            continue
        x = b / a
        noise = max(10.0, sa, sb)
        v = "within noise: do not claim" if abs(x - 1) * 100 < noise else ("faster" if x > 1 else "slower")
        rows.append((label, a, b, x, (na, nb), noise, v))
    return rows


def cm(s, cap=80):
    """Record-derived text for a comment on a public issue: md() plus no @mentions, no auto-linked URLs or #N references, no emphasis, capped length."""
    s = str(s)
    s = (s[:cap] + "...") if len(s) > cap else s
    s = md(s).replace("@", "@\u200b").replace("://", ":\u200b//").replace("www.", "www\u200b.").replace("#", "#\u200b")
    return s.replace("*", "\\*").replace("_", "\\_").replace("~", "\\~").replace("!", "\\!")


def issue_comment(rec, errs, why="", number=None):
    """The text the Action posts on the issue: valid with its board row, or the problems."""
    out = [COMMENT_MARK]
    if rec is None:
        return "\n".join(out + ["**Record check: no record found.**", "", cm(why, 200), "", "Paste the whole `.md` report that `bench-report` wrote (it contains the record), or the record file."]) + "\n"
    if errs:
        return "\n".join(out + ["**Record check: not valid.**", "", *[f"- {cm(e, 200)}" for e in errs[:8]], "", "Run `python bin/moe_cache.py bench-record validate FILE` on your machine, "
                                 "or paste the report again as it was written. A value you could not measure must say `not recorded`."]) + "\n"
    hw, sw, m = rec.get("hardware", {}), rec.get("software", {}), rec.get("model", {})
    head = f"**Record check: valid.** moe-cache {cm(sw.get('moe_cache_version', NR), 40)}, {cm(m.get('file', NR))}, {cm(hw.get('gpu_model', NR))}, {cm(hw.get('ram_gib', NR), 20)} GiB RAM, {cm(rec.get('conditions', {}).get('file_cache', NR), 40)} start."
    lines = [head, "", "| metric | stock | moe-cache | ratio | runs (stock / moe-cache) | noise band | reading |", "|---|---|---|---|---|---|---|"]
    for label, a, b, x, n, noise, v in board_rows(rec):
        mt = "decode_tps" if "decode" in label else "prompt_tps"
        lines.append(f"| {label} | {fmt(mt, a)} | {fmt(mt, b)} | {'-' if x is None else f'{x:.2f}x'} | {n[0]} / {n[1]} | {'-' if noise is None else f'{noise:.0f} %'} | {v} |")
    lines += ["", "This is the number the record gives; a claim on the board is made by hand after the runs are read. Thank you for measuring."]
    return "\n".join(out + lines) + "\n"


def fetch_issue_body(number, repo=REPO):
    """The text of a public issue (GitHub API, no login needed for a public repository). Tests replace this."""
    import urllib.request
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/issues/{int(number)}", headers={"Accept": "application/vnd.github+json", "User-Agent": "moe-cache-bench-record"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read(MAX_ISSUE_BODY + 1).decode("utf-8", "replace"))
    return data.get("body") or ""


def ingest_issue(body, number, bdir=BENCH, machine=None, write=True):
    """-> (run id or None, comment). A valid record is written to runs/ as DATE_community-N_SHORTID.json (write=False: only checked)."""
    rec, why = extract_record(body)
    if rec is None:
        return None, issue_comment(None, [], why)
    sch = rr.load_schema()
    errs = rr.validate(rec, sch) if sch else []
    if errs:
        return None, issue_comment(rec, errs)
    date = str(rec.get("created_utc", ""))[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return None, issue_comment(rec, ["created_utc is not YYYY-MM-DD ..."])
    rid = None
    if write:
        tmp = os.path.join(bdir, "runs", f".incoming-{os.getpid()}.json")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        try:
            rid, _ = ingest(tmp, bdir, machine=machine or (f"community-{int(number)}" if number else "community"))
        finally:
            os.remove(tmp)
    return rid, issue_comment(rec, [])


def setting_of(rec):
    """The setting label of a run set: what distinguishes one cell of the board from another (only recorded parts)."""
    s, p = rec["settings"], rec["protocol"]
    parts = []
    for label, v in (("ctx", s.get("ctx")), ("-ub", s.get("ubatch")), ("n-cpu-moe", s.get("n_cpu_moe")), ("GPU expert layers", s.get("gpu_expert_layers")),
                     ("KV", s.get("kv_type_k"))):
        if v not in (None, NR):
            parts.append(f"{label} {v}")
    if p.get("prompt_set", NR) != NR:
        parts.append(str(p["prompt_set"]))
    return ", ".join(parts) or NR


def model_of(rec):
    m = rec["model"]
    return m["file"] if m.get("quant") in (NR, None) or str(m["quant"]) in m["file"] else f"{m['file']} ({m['quant']})"


def arm_stats(rec, arm, metric):
    """(mean, runs counted, relative spread %) of the ok runs of an arm; mean is NR when no run has the metric."""
    xs = [r[metric] for r in rec["runs"] if r["arm"] == arm and r.get("ok", True) and isnum(r.get(metric))]
    if not xs:
        return NR, 0, 0.0
    mean = sum(xs) / len(xs)
    return mean, len(xs), (100 * (max(xs) - min(xs)) / mean if len(xs) > 1 and mean else 0.0)


def fmt(metric, v):
    """Rounded half up on the 6-digit value (so a mean of 22.155 prints 22.16, as a person would round it)."""
    if not isnum(v):
        return NR
    q = decimal.Decimal("0.01" if metric == "decode_tps" else "0.1")
    return str(decimal.Decimal(repr(round(v, 6))).quantize(q, rounding=decimal.ROUND_HALF_UP))


def pct(a, b):
    return f"{100 * (a / b - 1):+.1f} %" if isnum(a) and isnum(b) and b else NR


# ---------------------------------------------------------------- claims

def parse_claim(text):
    """Front matter (`key: JSON value` lines between two `---`) and the body; the first `# ` line of the body is the title."""
    m = re.match(r"---\n(.*?)\n---\n(.*)\Z", text, re.S)
    if not m:
        raise ValueError("claim file needs a front matter block between two --- lines")
    c = {}
    for line in m.group(1).splitlines():
        if line.strip():
            k, _, v = line.partition(": ")
            c[k.strip()] = json.loads(v)
    body = m.group(2).strip("\n")
    t = re.match(r"# (.*)\n?", body)
    c["title"] = t.group(1).strip() if t else c.get("id", "")
    c["statement"] = body[t.end():].strip() if t else body
    return c


def render_claim(c):
    keys = ("id", "runs", "cells", "valid_moe_cache", "valid_llama_builds", "withdrawn")
    fm = "\n".join(f"{k}: {json.dumps(c[k], ensure_ascii=False)}" for k in keys)
    return f"---\n{fm}\n---\n# {c['title']}\n\n{c['statement'].strip()}\n"


def load_claims(bdir=BENCH):
    claims = {}
    for f in sorted(glob.glob(os.path.join(bdir, "claims", "*.md"))):
        c = parse_claim(open(f, encoding="utf-8").read())
        c["file"] = os.path.basename(f)
        claims[c["id"]] = c
    return claims


def claim_date(c):
    return max((run_parts(r)[0] for r in c["runs"]), default="")


def cell_key(c, cell, runs):
    rec = runs[cell["run"]]
    return (run_parts(cell["run"])[1], model_of(rec), setting_of(rec), cell["arm"], cell["metric"])


def compute_status(claims, runs, version=None, pin=None):
    """{claim id: (status, reason)} and the conflicts [(key, [claim ids])]. Order of checks: withdrawn, superseded, stale, current."""
    version = version or current_version()
    pin = pin if pin is not None else current_pin()
    by_key = {}
    for cid, c in claims.items():
        if c.get("withdrawn"):
            continue
        for cell in c["cells"]:
            by_key.setdefault(cell_key(c, cell, runs), []).append(cid)
    out, conflicts = {}, []
    for key, ids in by_key.items():
        vals = {round(arm_stats(runs[cell["run"]], cell["arm"], cell["metric"])[0] or 0, 2) for cid in ids for cell in claims[cid]["cells"]
                if cell_key(claims[cid], cell, runs) == key}
        newest = max(claim_date(claims[i]) for i in ids)
        top = [i for i in ids if claim_date(claims[i]) == newest]
        if len(top) > 1 and len(vals) > 1:
            conflicts.append((key, sorted(top)))
    for cid, c in claims.items():
        if c.get("withdrawn"):
            out[cid] = ("withdrawn", c["withdrawn"])
            continue
        newer = 0
        for cell in c["cells"]:
            ids = by_key[cell_key(c, cell, runs)]
            if any(claim_date(claims[i]) > claim_date(c) for i in ids):
                newer += 1
        if c["cells"] and newer == len(c["cells"]):
            out[cid] = ("superseded", "every cell is covered by a newer claim")
            continue
        vm, vl = c.get("valid_moe_cache"), c.get("valid_llama_builds")
        hi = vtuple(vm[1]) if vm and vm[1] != NR else None
        cur = vtuple(version)
        if hi is None:
            out[cid] = ("stale", "validity range (moe-cache version) not recorded")
        elif cur and cur > hi:
            out[cid] = ("stale", f"moe-cache is now {version}, the claim is valid up to {vm[1]}")
        elif vl and isnum(vl[1]) and pin and pin > vl[1]:
            out[cid] = ("stale", f"the llama.cpp pin is now b{pin}, the claim is valid up to b{vl[1]}")
        else:
            out[cid] = ("current", "")
    return out, conflicts


def auto_cells(run_id, rec, vs=None):
    cells = []
    for arm in dict.fromkeys(r["arm"] for r in rec["runs"]):
        for metric in METRICS:
            if arm_stats(rec, arm, metric)[1]:
                cell = {"run": run_id, "arm": arm, "metric": metric}
                if vs and vs != arm:
                    cell["vs"] = vs
                cells.append(cell)
    return cells


def cmd_claim(a, bdir):
    claims, runs = load_claims(bdir), load_runs(bdir)
    if a.action == "add":
        if not re.fullmatch(r"[A-Za-z0-9._-]+", a.id):
            raise ValueError("claim id: letters, digits, . _ - only")
        if a.id in claims:
            raise ValueError(f"claim {a.id} exists (claims are edited by hand, not re-added)")
        cells = []
        for rid in a.run:
            if rid not in runs:
                raise ValueError(f"no run {rid} in {os.path.join(bdir, 'runs')}")
            cells += auto_cells(rid, runs[rid], a.vs)
        for spec in a.cell or []:
            p = spec.split(":")
            if len(p) not in (3, 4) or p[0] not in runs or p[2] not in METRICS:
                raise ValueError("--cell RUN:ARM:METRIC[:VSARM], METRIC = prompt_tps or decode_tps")
            cells.append({"run": p[0], "arm": p[1], "metric": p[2], **({"vs": p[3]} if len(p) == 4 else {})})
        if not cells:
            raise ValueError("no cells: give --run RUN (all arms and metrics) or --cell")
        first = runs[(a.run or [cells[0]["run"]])[0]]
        vm = a.valid_moe.split("-", 1) if a.valid_moe else [first["software"]["moe_cache_version"]] * 2
        vl = [int(x) if x.isdigit() else NR for x in a.valid_llama.split("-", 1)] if a.valid_llama else [first["software"]["llama_build"]] * 2
        c = {"id": a.id, "title": a.title or a.id, "statement": a.statement, "runs": sorted({x["run"] for x in cells}), "cells": cells,
             "valid_moe_cache": vm if len(vm) == 2 else [vm[0], vm[0]], "valid_llama_builds": vl if len(vl) == 2 else [vl[0], vl[0]], "withdrawn": ""}
        os.makedirs(os.path.join(bdir, "claims"), exist_ok=True)
        with open(os.path.join(bdir, "claims", a.id + ".md"), "w", encoding="utf-8") as f:
            f.write(render_claim(c))
        print(f"claim {a.id}: {len(cells)} cells from {len(c['runs'])} run(s) -> {os.path.join(bdir, 'claims', a.id + '.md')}")
        return 0
    status, conflicts = compute_status(claims, runs)
    if a.action == "list":
        for cid, c in sorted(claims.items(), key=lambda kv: (claim_date(kv[1]), kv[0]), reverse=True):
            print(f"{status[cid][0]:<10} {claim_date(c)}  {cid}  {c['title']}")
        return 0
    if a.id not in claims:
        raise ValueError(f"no claim {a.id}")
    print(show_claim(claims[a.id], runs, status[a.id]))
    return 0


def show_claim(c, runs, st):
    lines = [f"# {c['title']}", f"id: {c['id']}    status: {st[0]}" + (f" ({st[1]})" if st[1] else ""), "", c["statement"], ""]
    first = runs[c["cells"][0]["run"]] if c["cells"] else None
    if first:
        h, sw, s = first["hardware"], first["software"], first["settings"]
        lines += [f"Machine: {run_parts(c['cells'][0]['run'])[1]} ({h['os']}, {h['cpu_model']}, {h['gpu_model']}). Model: {model_of(first)}. Setting: {setting_of(first)}.",
                  f"Measured with: moe-cache {sw['moe_cache_version']}, llama.cpp b{sw['llama_build']} ({sw['llama_backend']}). Command: {s['command_line']}"]
    lines += [f"Valid for: moe-cache {c['valid_moe_cache'][0]} to {c['valid_moe_cache'][1]}; llama.cpp b{c['valid_llama_builds'][0]} to b{c['valid_llama_builds'][1]}.", "",
              "| Arm | Metric | Value (mean) | Runs | Spread | Against | Run |", "|---|---|---|---|---|---|---|"]
    for cell in c["cells"]:
        rec = runs[cell["run"]]
        m, n, sp = arm_stats(rec, cell["arm"], cell["metric"])
        vs = ""
        if cell.get("vs"):
            vs = f"{cell['vs']}: {pct(m, arm_stats(rec, cell['vs'], cell['metric'])[0])}"
        lines.append(f"| {cell['arm']} | {METRICS[cell['metric']]} | {fmt(cell['metric'], m)} | {n} | {sp:.1f} % | {vs} | {cell['run']} |")
    gaps = sorted({g for r in c["runs"] for g in rr.not_recorded_fields(runs[r])})
    for line in dict.fromkeys(filter(None, (rr.env_line_for(runs[cell["run"]], cell["arm"]) for cell in c["cells"]))):
        lines += ["", f"Measured on a {line}"]
    if not any(rr.env_of(runs[r])[0] for r in c["runs"]):
        lines += ["", f"Environment: {NR} (the run predates the environment block, or no sampler ran)."]
    lines += ["", f"{len(gaps)} fields say `{NR}` in the runs behind this claim." + (" First ones: " + ", ".join(gaps[:6]) if gaps else "")]
    return "\n".join(lines)


# ---------------------------------------------------------------- board and journal

_MD_ESC = str.maketrans({"\\": "\\\\", "|": "\\|", "`": "\\`", "[": "\\[", "]": "\\]", "<": "&lt;", ">": "&gt;", "&": "&amp;", "\n": " ", "\r": " "})


def md(s):
    """Record text (a community record's machine, model, arm or setting names) as plain text in a Markdown table: no table breaks, links, code or HTML."""
    return str(s).translate(_MD_ESC)


def machine_label(runs, machine):
    for rid, rec in runs.items():
        if run_parts(rid)[1] == machine:
            h = rec["hardware"]
            return md(f"{machine}: {h['os']}, {h['cpu_model']}, {h['gpu_model']}")
    return md(machine)


def board_text(claims, runs, version=None, pin=None):
    version = version or current_version()
    pin = pin if pin is not None else current_pin()
    status, conflicts = compute_status(claims, runs, version, pin)
    win = {}                                  # (machine, model, setting, arm, metric) -> (claim date, claim id, cell)
    for cid, c in claims.items():
        if c.get("withdrawn"):
            continue
        for cell in c["cells"]:
            k = cell_key(c, cell, runs)
            if k not in win or (claim_date(c), cid) > win[k][:2]:
                win[k] = (claim_date(c), cid, cell)
    out = ["# Benchmark board: what is true now", "",
           f"Generated by `python3 bin/moe_cache.py bench-record board`; do not edit. Computed against moe-cache {version} and the llama.cpp pin "
           f"{'b' + str(pin) if pin else NR}. Every number is the mean of the runs in a run file ([runs/](runs/)) and links to the claim that publishes it ([claims/](claims/)).", "",
           "One row is a machine, model, setting and arm. The number shown is the one from the newest claim for that cell. **Status** is computed: *current* = valid for this "
           "version and pin; *stale* = the moe-cache version or llama.cpp pin has moved past what the claim is valid for (the number may still hold, nobody has re-measured); "
           "*superseded* = a newer claim covers the same cell. `not recorded` = nobody measured it. Change figures are against the arm named, my arithmetic from the means.", ""]
    rows = {}
    for (machine, model, setting, arm, metric), (d, cid, cell) in win.items():
        rows.setdefault((machine, model, setting, arm), {})[metric] = (d, cid, cell)
    has_env = any(rr.env_of(rec)[0] for rec in runs.values())             # BENCH-1150: the column appears once a run carries an environment block
    for machine in sorted({k[0] for k in rows}):
        out += [f"## {machine_label(runs, machine)}", "", "| Model | Setting | Arm | Prompt tok/s | Decode tok/s | Date | Status | Claims |" + (" Working desktop |" if has_env else ""),
                "|---|---|---|---|---|---|---|---|" + ("---|" if has_env else "")]
        for key in sorted(k for k in rows if k[0] == machine):
            cells, parts = rows[key], {}
            for metric in METRICS:
                if metric not in cells:
                    parts[metric] = NR
                    continue
                d, cid, cell = cells[metric]
                rec = runs[cell["run"]]
                m = arm_stats(rec, cell["arm"], metric)[0]
                parts[metric] = fmt(metric, m) + (f" ({pct(m, arm_stats(rec, cell['vs'], metric)[0])} vs {md(cell['vs'])})" if cell.get("vs") else "")
            ids = sorted({v[1] for v in cells.values()})
            sts = sorted({status[i][0] for i in ids})
            out.append(f"| {md(key[1])} | {md(key[2])} | {md(key[3])} | {parts['prompt_tps']} | {parts['decode_tps']} | {max(v[0] for v in cells.values())} | {', '.join(sts)} | "
                       + ", ".join(f"[{i}](claims/{claims[i]['file']})" for i in ids) + " |"
                       + ((" " + md(next(filter(None, (rr.env_line_for(runs[cl["run"]], cl["arm"]) for cl in cells.values() for cl in [cl[2]])), NR)) + " |") if has_env else ""))
        out.append("")
    bad = [(cid, st) for cid, st in sorted(status.items()) if st[0] != "current"]
    out += ["## Claims that are not current", ""]
    out += ["| Claim | Status | Why |", "|---|---|---|"] + [f"| [{cid}](claims/{claims[cid]['file']}) | {st[0]} | {md(st[1])} |" for cid, st in bad] if bad else ["None."]
    out += ["", "## Conflicts", ""]
    if conflicts:
        out += ["Two claims of the same date give different numbers for the same cell. Measure again, or withdraw one.", ""]
        out += [f"- {' / '.join(md(x) for x in k)}: " + ", ".join(ids) for k, ids in conflicts]
    else:
        out.append("None.")
    return "\n".join(out) + "\n"


def journal_text(claims, runs):
    status, _ = compute_status(claims, runs)
    out = ["# Benchmark journal", "", "Generated by `python3 bin/moe_cache.py bench-record board`; do not edit. One line per run set, newest first, and the claims that cite it.", ""]
    for rid in sorted(runs, key=lambda r: (run_parts(r)[0], r), reverse=True):
        rec, (d, machine, _) = runs[rid], run_parts(rid)
        arms = list(dict.fromkeys(r["arm"] for r in rec["runs"]))
        cl = sorted(cid for cid, c in claims.items() if rid in c["runs"])
        gaps = len(rr.not_recorded_fields(rec))
        out.append(f"- {d} {md(machine)}: {md(model_of(rec))}, {md(setting_of(rec))}; arms {', '.join(md(x) for x in arms)}, {len(rec['runs'])} runs; {gaps} fields not recorded. [{md(rid)}](runs/{rid}.json)"
                   + (f". Claims: {', '.join(f'{c} ({status[c][0]})' for c in cl)}" if cl else ". No claim yet"))
    return "\n".join(out) + "\n"


def cmd_board(a, bdir):
    claims, runs = load_claims(bdir), load_runs(bdir)
    files = {"BOARD.md": board_text(claims, runs), "JOURNAL.md": journal_text(claims, runs)}
    stale = [n for n, t in files.items() if not os.path.exists(os.path.join(bdir, n)) or open(os.path.join(bdir, n), encoding="utf-8").read() != t]
    if a.check:
        print("board: up to date" if not stale else "board: out of date: " + ", ".join(stale) + " (run `bench-record board`)")
        return 1 if stale else 0
    for n, t in files.items():
        with open(os.path.join(bdir, n), "w", encoding="utf-8") as f:
            f.write(t)
    print(f"wrote {os.path.join(bdir, 'BOARD.md')} and JOURNAL.md: {len(runs)} runs, {len(claims)} claims")
    return 0


# ---------------------------------------------------------------- lint

PERF = re.compile(r"(?<![\w.])[+\-−–≈]?\d[\d,]*(?:\.\d+)?\s?(?:tok/s|tokens/s)|(?<![\w.])[+\-−–]\d+(?:[.,]\d+)?\s?%|(?<![\w.])\d+(?:\.\d+)?\s?(?:×|x)(?![\w])")


def perf_numbers(line):
    """The performance numbers a line shows (tok/s values, signed percents, N× multipliers), outside HTML tags and comments."""
    text = re.sub(r"<!--.*?-->", " ", line)
    text = re.sub(r"<[^>]+>", " ", text)
    return [m.group(0).strip() for m in PERF.finditer(text)]


def lint_file(path, claims, status, root=ROOT):
    """Problems [(line, kind, text)] and the count of lines that show numbers."""
    probs, shown, ok = [], 0, 0
    fence = skip = False
    for n, line in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
        if path.endswith((".md", ".markdown")) and line.lstrip().startswith("```"):
            fence = not fence
            continue
        if re.search(r"<(script|style)\b", line, re.I):
            skip = not re.search(r"</(script|style)>", line, re.I)
            continue
        if re.search(r"</(script|style)>", line, re.I):
            skip = False
            continue
        if fence or skip:
            continue
        nums = perf_numbers(line)
        if not nums:
            continue
        shown += 1
        ids = MARK.findall(line)
        if not ids:
            probs.append((n, "unclaimed", ", ".join(nums[:4])))
            continue
        line_ok = False
        for i in ids:
            if i not in claims:
                probs.append((n, "unknown claim", i))
            elif status[i][0] != "current":
                probs.append((n, f"claim {status[i][0]}", f"{i}: {status[i][1]}"))
            else:
                line_ok = True
        ok += line_ok
    return probs, shown, ok


def cmd_lint(a, bdir):
    claims, runs = load_claims(bdir), load_runs(bdir)
    status, _ = compute_status(claims, runs)
    files = a.files or [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "docs", "RESULTS_CURRENT.md"), os.path.join(ROOT, "docs", "benchmark_card.html")]
    total = 0
    for f in files:
        probs, shown, ok = lint_file(f, claims, status)
        rel = os.path.relpath(f, ROOT) if f.startswith(ROOT) else f
        for n, kind, text in probs:
            print(f"{rel}:{n}: {kind}: {text}")
        print(f"{rel}: {shown} lines show performance numbers; {ok} with a current claim; {len(probs)} problems")
        total += len(probs)
    if total and a.warn:
        print("lint: WARN mode, exit 0 (add `<!-- claim:ID -->` to the line, or `--strict`)")
        return 0
    return 1 if total else 0


# ---------------------------------------------------------------- replay

def split_command(cm):
    """shlex for POSIX; on Windows backslashes in paths (C:\\models\\x.gguf) must survive, so non-POSIX splitting with the quotes removed."""
    if os.name != "nt":
        return shlex.split(cm)
    return [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'" else t for t in shlex.split(cm, posix=False)]


def cmd_replay(a, bdir):
    claims, runs = load_claims(bdir), load_runs(bdir)
    if a.claim not in claims:
        raise ValueError(f"no claim {a.claim}")
    c = claims[a.claim]
    status, _ = compute_status(claims, runs)
    print(f"claim {c['id']} ({status[c['id']][0]}): {c['title']}")
    cmds = []
    for rid in c["runs"]:
        for cm in runs[rid]["reproduce"].get("commands", []) or []:
            if cm and cm != NR and cm not in cmds:
                cmds.append(cm)
    first = runs[c["runs"][0]]
    print(f"Match these conditions: {run_parts(c['runs'][0])[1]}; {model_of(first)}; {setting_of(first)}; file cache {first['conditions']['file_cache']}; "
          f"protocol {first['protocol']['order']}, {first['protocol']['runs_per_arm']} runs per arm.")
    if cmds:
        print("Commands:")
        for cm in cmds:
            print("  " + cm)
    else:
        print(f"Commands: {NR} for the runs behind this claim (a converted run); use bench-report or bench-record with the conditions above.")
    if a.run:
        if not cmds:
            print("replay: nothing to run", file=sys.stderr)
            return 2
        for cm in cmds:
            print("$ " + cm)
            # the commands come from a benchmark record, which may be contributed by someone else: ask before each one (AUD-037)
            if not a.yes:
                try:
                    ok = input("  run this command? [y/N] ").strip().lower() in ("y", "yes") if sys.stdin.isatty() else False
                except EOFError:
                    ok = False
                if not ok:
                    print("replay: not run (answer y at the question, or pass --yes after reading the commands above)", file=sys.stderr)
                    return 2
            rc = subprocess.call(split_command(cm))
            if rc:
                print(f"replay: command failed with exit {rc}", file=sys.stderr)
                return rc
    if not a.record:
        return 0
    new = json.load(open(a.record, encoding="utf-8"))
    amap = dict(x.split("=", 1) for x in (a.arm_map or []))
    if setting_of(new) != setting_of(first):
        print(f"warning: setting differs: claim '{setting_of(first)}', new '{setting_of(new)}'")
    print("", "| Arm | Metric | Claim | New | Change | Band | Within band |", "|---|---|---|---|---|---|---|", sep="\n")
    worst = 0
    for cell in c["cells"]:
        old, _, spread = arm_stats(runs[cell["run"]], cell["arm"], cell["metric"])
        nv, nn, nspread = arm_stats(new, amap.get(cell["arm"], cell["arm"]), cell["metric"])
        band = max(a.band, spread, nspread)
        if not isnum(old) or not isnum(nv):
            print(f"| {cell['arm']} | {METRICS[cell['metric']]} | {fmt(cell['metric'], old)} | {fmt(cell['metric'], nv)} | {NR} | {band:.1f} % | not compared |")
            continue
        ch = 100 * (nv / old - 1)
        within = abs(ch) <= band
        worst += not within
        print(f"| {cell['arm']} | {METRICS[cell['metric']]} | {fmt(cell['metric'], old)} | {fmt(cell['metric'], nv)} | {ch:+.1f} % | {band:.1f} % | {'yes' if within else 'NO'} |")
    print(f"\nBand = the larger of {a.band} %, the claim's own first/second spread and the new record's spread. "
          + ("All compared cells are inside the band: the claim holds." if not worst else f"{worst} cell(s) outside the band: measure again, then add a newer claim or withdraw this one."))
    return 1 if worst else 0


# ---------------------------------------------------------------- command line

def main(argv=None, prog_cmd=None):
    ap = argparse.ArgumentParser(prog="bench-record", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=BENCH, help="the benchmarks folder (default: benchmarks/ of this checkout)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="copy a record into runs/ under its canonical name")
    i.add_argument("record"); i.add_argument("--machine", help="short machine name (default: from the GPU)"); i.add_argument("--date"); i.add_argument("--force", action="store_true", help="ingest although the schema check fails")
    g = sub.add_parser("ingest-issue", help="the record in a GitHub issue -> runs/ (or --check: only validate and print the comment)")
    g.add_argument("number", nargs="?", type=int, help="issue number in the public repository (read from api.github.com)"); g.add_argument("--file", help="the issue text from a file (no network)")
    g.add_argument("--repo", default=REPO); g.add_argument("--check", action="store_true", help="validate only; print the comment for the issue; exit 1 when the record is missing or invalid"); g.add_argument("--machine")
    c = sub.add_parser("claim", help="add | show | list")
    c.add_argument("action", choices=["add", "show", "list"]); c.add_argument("id", nargs="?")
    c.add_argument("--run", action="append", default=[], help="a run id (file name without .json); all its arms and metrics become cells"); c.add_argument("--cell", action="append", help="RUN:ARM:METRIC[:VSARM]")
    c.add_argument("--vs", help="the arm the others are compared with (e.g. v0.3.4, stock)"); c.add_argument("--title"); c.add_argument("--statement", default="")
    c.add_argument("--valid-moe", help="MIN-MAX moe-cache versions (default: the runs' version)"); c.add_argument("--valid-llama", help="MIN-MAX llama.cpp builds (default: the runs' build)")
    b = sub.add_parser("board", help="write BOARD.md and JOURNAL.md"); b.add_argument("--check", action="store_true")
    l = sub.add_parser("lint", help="every performance number in README.md / the card needs a current claim marker"); l.add_argument("files", nargs="*")
    l.add_argument("--warn", action="store_true", help="print the problems, exit 0"); l.add_argument("--strict", action="store_true", help="exit 1 on a problem (default)")
    r = sub.add_parser("replay", help="commands to reproduce a claim; --record NEW.json compares")
    r.add_argument("claim"); r.add_argument("--run", action="store_true", help="execute the commands (asks before each one)"); r.add_argument("--yes", action="store_true", help="with --run: do not ask (read the commands first)"); r.add_argument("--record"); r.add_argument("--band", type=float, default=NOISE_MIN_PCT)
    r.add_argument("--arm-map", action="append", help="CLAIMARM=NEWARM when the new record names the arms differently")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "ingest":
            rid, wrote = ingest(a.record, a.dir, a.machine, a.date, a.force)
            print(f"{rid}" + ("" if wrote else "  (already there, unchanged)"))
            return 0
        if a.cmd == "ingest-issue":
            if not a.number and not a.file:
                ap.error("ingest-issue needs an issue number or --file")
            body = open(a.file, encoding="utf-8", errors="replace").read(MAX_ISSUE_BODY + 1) if a.file else fetch_issue_body(a.number, a.repo)
            rid, comment = ingest_issue(body, a.number, a.dir, a.machine, write=not a.check)
            if a.check:
                print(comment, end="")
                return 0 if "Record check: valid" in comment else 1
            if rid is None:
                print(comment, file=sys.stderr, end="")
                return 1
            print(rid)
            return 0
        if a.cmd == "claim":
            if a.action != "list" and not a.id:
                ap.error("claim add|show needs an ID")
            return cmd_claim(a, a.dir)
        if a.cmd == "board":
            return cmd_board(a, a.dir)
        if a.cmd == "lint":
            return cmd_lint(a, a.dir)
        return cmd_replay(a, a.dir)
    except (ValueError, OSError, KeyError) as e:
        print(f"bench-record {a.cmd}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
