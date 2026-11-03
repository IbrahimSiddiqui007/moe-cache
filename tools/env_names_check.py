#!/usr/bin/env python3
"""List every MOE_CACHE_ name the code reads (src/, bin/, planner/) and whether the docs mention it (README.md, docs/*.md).

  tools/env_names_check.py            table: name, files that use it, documented yes/no; then a count
  tools/env_names_check.py --missing  only the undocumented names, one per line (with the first file using it)
  tools/env_names_check.py --config-only  count a mention only in README.md and docs/CONFIGURATION.md (default: anywhere in README.md and docs/*.md)
  tools/env_names_check.py --strict   exit 1 when something is undocumented (the default always exits 0: a warning only)
A name counts as read when it appears in code; names that end in "_" (prefixes built at run time) are skipped.
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = re.compile(r"MOE_CACHE_[A-Z0-9_]*[A-Z0-9]")
CODE_DIRS = ("src", "bin", "planner")


def scan(paths, exts):
    found = {}
    for top in paths:
        full = os.path.join(ROOT, top)
        files = [full] if os.path.isfile(full) else [os.path.join(d, f) for d, _, fs in os.walk(full) for f in fs]
        for f in sorted(files):
            if "__pycache__" in f or (exts and not f.endswith(exts)):
                continue
            try:
                text = open(f, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for n in NAME.findall(text):
                found.setdefault(n, set()).add(os.path.relpath(f, ROOT))
    return found


def main(argv):
    code = scan(CODE_DIRS, None)
    docs = scan(["README.md", "docs/CONFIGURATION.md"] if "--config-only" in argv else ["README.md", "docs"], (".md",))
    rows = [(n, sorted(fs), n in docs) for n, fs in sorted(code.items())]
    missing = [r for r in rows if not r[2]]
    if "--missing" in argv:
        for n, fs, _ in missing:
            print(f"{n}\t{fs[0]}")
    else:
        for n, fs, ok in rows:
            print(f"{'yes' if ok else 'NO ':3}  {n:40} {', '.join(fs[:3])}")
    print(f"{len(rows)} names read in code, {len(rows) - len(missing)} documented, {len(missing)} not documented", file=sys.stderr)
    return 1 if (missing and "--strict" in argv) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
