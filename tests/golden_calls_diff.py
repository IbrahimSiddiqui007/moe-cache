#!/usr/bin/env python3
"""Compares two launch-plan golden files field by field (ENGINE-029): every case must have the same rc, stdout and stderr, and the same `calls` except for the entries
that contain one given text (default: the planner/plan.py subprocess call) - at most one such entry per case, and removed, not changed. Anything else that differs
(placement, -ub, layers, environment, command line, a message) fails.
   python3 tests/golden_calls_diff.py OLD.json [NEW.json] [--allow-removed TEXT]        NEW defaults to tests/launch_plan_golden_synth.json
   python3 tests/golden_calls_diff.py                                                   no arguments: the checker against made-up files (runs in quick)"""
import json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compare(old, new, allow="planner/plan.py"):
    """-> (problems, removed): problems are text lines; removed is how many calls entries were taken out."""
    problems, removed = [], 0
    if old.get("model") != new.get("model"):
        problems.append(f"model fingerprint differs: {old.get('model')} vs {new.get('model')}")
    oc, nc = old.get("cases", {}), new.get("cases", {})
    for k in sorted(set(oc) | set(nc)):
        if k not in oc or k not in nc:
            problems.append(f"{k}: only in the {'new' if k not in oc else 'old'} file")
            continue
        o, n = oc[k], nc[k]
        for f in sorted(set(o) | set(n)):
            if f == "calls":
                continue
            if o.get(f) != n.get(f):
                problems.append(f"{k}: field '{f}' differs")
        oldcalls, newcalls = list(o.get("calls", [])), list(n.get("calls", []))
        gone = [c for c in oldcalls if allow in c]
        keep = [c for c in oldcalls if allow not in c]
        if len(gone) > 1:
            problems.append(f"{k}: {len(gone)} calls contain '{allow}' in the old file (expected at most one)")
        if any(allow in c for c in newcalls):
            problems.append(f"{k}: the new file still has a call with '{allow}'")
        if keep != [c for c in newcalls]:
            problems.append(f"{k}: calls differ beyond the '{allow}' entry")
        removed += len(gone)
    return problems, removed


def self_test():
    base = {"model": "m", "cases": {"a": {"rc": 0, "stdout": "x", "stderr": "", "calls": ["python3 <HERE>/planner/plan.py M --json", "ldd L"]},
                                    "b": {"rc": 0, "stdout": "y", "stderr": "", "calls": ["ldd L"]}}}
    def mod(f):
        n = json.loads(json.dumps(base)); f(n); return n
    ok_new = mod(lambda n: n["cases"]["a"].update(calls=["ldd L"]))
    cases = [
        ("only the plan.py entry removed: passes", ok_new, True),
        ("the plan.py entry still there: fails", mod(lambda n: None), False),
        ("stdout changed: fails", mod(lambda n: (n["cases"]["a"].update(calls=["ldd L"]), n["cases"]["a"].update(stdout="x2"))), False),
        ("another call removed too: fails", mod(lambda n: n["cases"]["a"].update(calls=[])), False),
        ("a call changed: fails", mod(lambda n: n["cases"]["a"].update(calls=["ldd L2"])), False),
        ("a case missing: fails", mod(lambda n: (n["cases"]["a"].update(calls=["ldd L"]), n["cases"].pop("b"))), False),
        ("rc changed: fails", mod(lambda n: (n["cases"]["a"].update(calls=["ldd L"]), n["cases"]["b"].update(rc=1))), False),
        ("model fingerprint changed: fails", mod(lambda n: (n["cases"]["a"].update(calls=["ldd L"]), n.update(model="other"))), False),
    ]
    bad = 0
    for name, new, want_ok in cases:
        problems, _ = compare(base, new)
        ok = (not problems) == want_ok
        bad += not ok
        print(("  ok   " if ok else "  FAIL ") + "golden diff checker: " + name)
    base2 = json.loads(json.dumps(base)); base2["cases"]["a"]["calls"].append("python3 <HERE>/planner/plan.py second")
    problems, _ = compare(base2, mod(lambda n: n["cases"]["a"].update(calls=["ldd L"])))
    ok = any("expected at most one" in p for p in problems)
    bad += not ok
    print(("  ok   " if ok else "  FAIL ") + "golden diff checker: two plan.py calls in one case: fails")
    return bad


def main(argv):
    if "--self-test" in argv or not argv:
        return 1 if self_test() else 0
    allow = "planner/plan.py"
    if "--allow-removed" in argv:
        i = argv.index("--allow-removed"); allow = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    old = json.load(open(argv[0]))
    new = json.load(open(argv[1] if len(argv) > 1 else os.path.join(HERE, "tests", "launch_plan_golden_synth.json")))
    problems, removed = compare(old, new, allow)
    for p in problems[:40]:
        print("  FAIL " + p)
    n = len(old.get("cases", {}))
    print(f"{n} cases compared: {'only the ' + repr(allow) + ' call differs (' + str(removed) + ' entries removed)' if not problems else str(len(problems)) + ' problems'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
