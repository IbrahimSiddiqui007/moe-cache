#!/usr/bin/env python3
"""Public files must not talk about other projects (owner rule, DOCS-676): fails if one of the names below appears in a tracked text file, except
the README credit line and the fixed KL-test input text. Needs git (or run from a checkout); standard library only."""
import os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = re.compile(r"strata(?![a-z])|bigmoeonedge|ktransformers|project[ -]maya|niko1221|mw00", re.I)
ALLOWED = {
    ("README.md", "Credit for ideas on tiering"),      # the one plain credit line: name and link, no comparison
}
EXEMPT_FILES = {
    "tests/public_names_test.py",                      # this file names them
    "tests/kl_batch_text.txt",                         # fixed prose input of the KL test (changing it changes the measurement); owner decides
}
files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
bad = []
for f in files:
    if not f or f in EXEMPT_FILES:
        continue
    try:
        data = open(os.path.join(ROOT, f), "rb").read()
    except OSError:
        continue
    if b"\0" in data:
        continue                                       # binary (images, the PDF)
    for n, line in enumerate(data.decode("utf-8", "replace").split("\n"), 1):
        if NAMES.search(line) and not any(f == af and key in line for af, key in ALLOWED):
            bad.append(f"{f}:{n}: {line.strip()[:120]}")
if bad:
    print("public text names another project (owner rule: no claims or comparisons; credit line in README only):")
    print("\n".join(bad))
    sys.exit(1)
print("public names test: PASS (no other project named outside the README credit line)")
