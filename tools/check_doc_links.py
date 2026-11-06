#!/usr/bin/env python3
"""Link check for the Markdown files (DOCS-restructure, v0.3.8): every relative link `[text](path)` and `href="path"` in README.md, CONTRIBUTING.md, docs/**.md,
benchmarks/**.md, patches/**/README.md and archive/**.md must point at a file or folder that exists (an #anchor is not checked, http(s) and mailto are skipped,
links inside code fences are skipped). Exit 1 and one line per broken link otherwise.
    python3 tools/check_doc_links.py [FILE ...]"""
import os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r'\]\(([^)\s]+)(?:\s+"[^"]*")?\)|href="([^"]+)"')


def md_files():
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    return [f for f in out if f and not f.startswith(("third_party/", "desktop/src-tauri/", "benchmarks/runs/"))]


def broken(path):
    bad, fence = [], False
    base = os.path.dirname(os.path.join(ROOT, path))
    for n, line in enumerate(open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read().split("\n"), 1):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        for m in LINK.finditer(line):
            t = (m.group(1) or m.group(2)).split("#")[0].split("?")[0]
            if not t or re.match(r"^[a-z][a-z0-9+.-]*:", t, re.I) or t.startswith("//"):
                continue
            p = os.path.normpath(os.path.join(ROOT, t.lstrip("/")) if t.startswith("/") else os.path.join(base, t))
            if not os.path.exists(p):
                bad.append((n, t))
    return bad


def main(argv):
    files = argv or md_files()
    n = 0
    for f in files:
        for line, t in broken(f):
            print(f"{f}:{line}: broken link {t}")
            n += 1
    print(f"link check: {len(files)} files, {n} broken link(s)")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
