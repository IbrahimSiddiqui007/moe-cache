#!/usr/bin/env python3
"""The release gate for published numbers: `bench-record lint` must be clean, or docs/claims_waiver.txt (the owner's decision, never created by a tool)
must cover every problem it prints. A waiver is ONE line:

  waived: owner YYYY-MM-DD; expires YYYY-MM-DD; claims ID_OR_GLOB[,ID_OR_GLOB...]; files FILE[,FILE...]; reason: TEXT

  claims   the claim ids (or globs such as chain1103-*) whose "claim stale" problems are waived; a bare * is refused
  files    the files whose "unclaimed" numbers are waived (exact paths); leave the field out if there are none
  expires  not in the past, and at most 30 days after the owner date
Anything else lint reports (a withdrawn or unknown claim, a stale id the waiver does not name, an unclaimed number in a file it does not name) fails the gate.

    python3 tools/claims_gate.py [--lint-file F] [--waiver docs/claims_waiver.txt] [--today YYYY-MM-DD]      exit 0 pass (maybe with a waiver warning), 1 fail"""
import argparse, datetime, fnmatch, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBLEM = re.compile(r"^(?P<file>\S+?):(?P<line>\d+): (?P<kind>claim stale|unclaimed|[^:]+)(?:: (?P<rest>.*))?$")


def parse_waiver(text, today):
    """-> (waiver dict, None) or (None, why it is refused)"""
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) != 1:
        return None, "the waiver must be exactly one line"
    m = re.match(r"^waived: owner (\d{4}-\d{2}-\d{2}); expires (\d{4}-\d{2}-\d{2}); claims ([^;]+)(?:; files ([^;]+))?; reason: (.{8,})$", lines[0])
    if not m:
        return None, "the waiver line does not match: waived: owner DATE; expires DATE; claims IDS; [files FILES;] reason: TEXT"
    try:
        owner, exp = (datetime.date.fromisoformat(m.group(i)) for i in (1, 2))
    except ValueError:
        return None, "a date in the waiver is not a real date"
    if exp < today:
        return None, f"the waiver expired on {exp}"
    if (exp - owner).days > 30:
        return None, "the waiver may run at most 30 days from its owner date"
    claims = [c.strip() for c in m.group(3).split(",") if c.strip()]
    if not claims or any(c.strip("*") == "" for c in claims):
        return None, "the waiver must name claim ids (or globs like chain1103-*), not a bare *"
    files = [f.strip() for f in (m.group(4) or "").split(",") if f.strip()]
    return {"claims": claims, "files": files, "line": lines[0]}, None


def uncovered(problems, waiver):
    out = []
    for p in problems:
        if p["kind"] == "claim stale":
            cid = (p["rest"] or "").split(":")[0].strip()
            if any(fnmatch.fnmatchcase(cid, g) for g in waiver["claims"]):
                continue
        elif p["kind"] == "unclaimed" and p["file"] in waiver["files"]:
            continue
        out.append(p)
    return out


def run(lint_text, lint_rc, waiver_text, today):
    problems = [m.groupdict() for m in (PROBLEM.match(l.strip()) for l in lint_text.split("\n") if "lines show" not in l) if m]
    if lint_rc == 0:
        return 0, "claims lint: clean"
    if waiver_text is None:
        return 1, "claims lint failed and there is no docs/claims_waiver.txt: fix the claims, or (owner only) add a waiver, see tools/claims_gate.py"
    w, why = parse_waiver(waiver_text, today)
    if w is None:
        return 1, "claims lint failed and the waiver is refused: " + why
    left = uncovered(problems, w)
    if left or not problems:
        shown = "\n".join(f"  {p['file']}:{p['line']}: {p['kind']} {p['rest'] or ''}" for p in left[:20]) or "  (lint failed with no parsable problem line)"
        return 1, f"claims lint failed; the waiver does not cover {len(left)} problem(s):\n{shown}"
    return 0, "::warning::claims lint has problems; released under the waiver: " + w["line"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lint-file")
    ap.add_argument("--waiver", default=os.path.join(ROOT, "docs", "claims_waiver.txt"))
    ap.add_argument("--today", default=None)
    a = ap.parse_args()
    today = datetime.date.fromisoformat(a.today) if a.today else datetime.date.today()
    if a.lint_file:
        text, rc = open(a.lint_file, encoding="utf-8").read(), 1
        rc = 0 if "problems" not in text and "problem" not in text else 1
    else:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "bin", "moe_cache.py"), "bench-record", "lint"], capture_output=True, text=True)
        text, rc = r.stdout + r.stderr, r.returncode
    waiver = open(a.waiver, encoding="utf-8").read() if os.path.exists(a.waiver) else None
    code, msg = run(text, rc, waiver, today)
    print(msg)
    sys.exit(code)


if __name__ == "__main__":
    main()
