#!/usr/bin/env python3
"""Public files must not talk about other projects (owner rule, DOCS-676): fails if one of the names below appears in a tracked file (text, and PDFs
through pdftotext, else `strings`), except the README credit line (an exact-line allow-list) and the fixed KL-test input text.
Needs git (or run from a checkout); standard library only. "Big MoE" is not on the list: it is ordinary English in our own texts (the name check
looks for the project spelling BigMoE...)."""
import os, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = re.compile(r"strata(?![a-z])|bigmoe|ktransformers|project[ -]maya|niko1221|mw00|hypura|moe-infinity|ik_llama|neurall", re.I)
ALLOWED_LINES = {          # exact, whole-line matches only: (file, line)
    ("README.md", "- Credit for ideas on tiering experts across VRAM, RAM and SSD: [Strata](https://github.com/Niko1221/Strata) (MIT, by Niko1221) and "
                  "[Project Maya](https://github.com/mw00/project-maya) (MIT, by mw00)."),
}
EXEMPT_FILES = {
    "tests/public_names_test.py",      # this file names them
    "tests/kl_batch_text.txt",         # fixed prose input of the KL test (changing it changes the measurement); owner decides
}


def text_of(path, data):
    if path.lower().endswith(".pdf"):
        for cmd in (["pdftotext", "-q", path, "-"], ["strings", "-n", "6", path]):
            if shutil.which(cmd[0]):
                return subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout
        return data.decode("latin-1")          # no tool: raw bytes still show uncompressed names
    if b"\0" in data:
        return ""                              # other binaries (images)
    return data.decode("utf-8", "replace")


def find_bad(files, root=ROOT, exempt=EXEMPT_FILES, allowed=ALLOWED_LINES):
    bad = []
    for f in files:
        if not f or f in exempt:
            continue
        p = os.path.join(root, f)
        try:
            data = open(p, "rb").read()
        except OSError:
            continue
        for n, line in enumerate(text_of(p, data).split("\n"), 1):
            if NAMES.search(line) and (f, line.rstrip("\r")) not in allowed:
                bad.append(f"{f}:{n}: {line.strip()[:120]}")
    return bad


if __name__ == "__main__":
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    # self-test of the checker: a planted name must be found, the exact credit line must pass, a changed credit line must not
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        open(os.path.join(t, "a.md"), "w").write("see Hypura and the rest\n")
        open(os.path.join(t, "README.md"), "w").write(next(iter(ALLOWED_LINES))[1] + "\n")
        open(os.path.join(t, "b.md"), "w").write(next(iter(ALLOWED_LINES))[1] + " It is faster.\n")
        got = find_bad(["a.md", "README.md", "b.md"], root=t)
        assert len(got) == 2 and got[0].startswith("a.md") and got[1].startswith("b.md"), got
    bad = find_bad(files)
    if bad:
        print("public files name another project (owner rule: no claims or comparisons; one credit line in the README only):")
        print("\n".join(bad))
        sys.exit(1)
    print("public names test: PASS (no other project named outside the README credit line; PDFs scanned)")
