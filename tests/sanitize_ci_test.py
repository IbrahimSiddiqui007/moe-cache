#!/usr/bin/env python3
"""QUALITY-017: the slow sanitize workflow has a concurrency group per ref that queues instead of cancelling a running build. Text check, no yaml library.
   python3 tests/sanitize_ci_test.py"""
import os, re, sys

p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".github", "workflows", "sanitize.yml")
t = open(p, encoding="utf-8").read()
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

m = re.search(r"^concurrency:\n((?:[ ]+.*\n)+)", t, re.M)
body = m.group(1) if m else ""
check(bool(m), "sanitize.yml has a top-level concurrency block")
check(re.search(r"^\s+group:\s*\S*\$\{\{\s*github\.ref\s*\}\}", body, re.M) is not None, "the group includes the ref (a push to another branch is not held back)")
check(re.search(r"^\s+cancel-in-progress:\s*false\s*$", body, re.M) is not None, "a running build is not cancelled by a newer run (they queue)")
check(t.index("concurrency:") < t.index("\njobs:"), "the block is above the jobs")
sys.exit(1 if fails else 0)
