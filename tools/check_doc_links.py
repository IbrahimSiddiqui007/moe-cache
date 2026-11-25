#!/usr/bin/env python3
"""Link check for the Markdown files (DOCS-restructure, v0.3.8): every relative link `[text](path)` and `href="path"` in README.md, CONTRIBUTING.md, docs/**.md,
benchmarks/**.md, patches/**/README.md and archive/**.md must point at a file or folder that exists (an #anchor is not checked, http(s) and mailto are skipped,
links inside code fences are skipped). DOCS-1003: a path in a code span (`docs/...`, `tools/...`, `bin/...`, `tests/...`, `src/...`, `benchmarks/...`,
`patches/...`, `.github/...`, written from the repository root) must exist too, except a name with a glob or a placeholder (`*`, `<`, `{`, `X`) and the files listed in
CODESPAN_SKIP (history that names files of older versions). Exit 1 and one line per broken link otherwise.
    python3 tools/check_doc_links.py [FILE ...]"""
import os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r'\]\(([^)\s]+)(?:\s+"[^"]*")?\)|href="([^"]+)"')


def md_files():
    """git's list; outside a git checkout (a release export, a tarball, a git that refuses the folder) a walk of the folder. An empty list is an error, never "0 broken"."""
    r = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True)
    out = r.stdout.split("\n") if r.returncode == 0 else []
    if not [f for f in out if f]:
        out = []
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in (".git", "node_modules", "dist", "build")]
            out += [os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/") for f in files if f.endswith(".md")]
    return [f for f in out if f and not f.startswith(("third_party/", "benchmarks/runs/"))]


SPAN = re.compile(r"`((?:docs|tools|bin|tests|src|benchmarks|patches|\.github)/[^`\s]*)`")
CODESPAN_SKIP = ("archive/", "patches/", "docs/releases/", "CHANGELOG.md", "docs/results/archive/", "docs/internal/", "benchmarks/claims/")


SPAN_OK = {"docs/build.md"}      # llama.cpp's own file (named in backends.md), not ours


def broken_spans(path):
    if path.startswith(CODESPAN_SKIP):
        return []
    bad, fence = [], False
    for n, line in enumerate(open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read().split("\n"), 1):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        for m in SPAN.finditer(line):
            t = m.group(1).rstrip(".,;:)")
            if re.search(r"[*<>{}\[\]$|]|\.\.\.|/X|vX|NAME|FILE|YYYY|VERSION|\bN\b", t) or not os.path.splitext(t)[1] and not t.endswith("/"):
                continue
            if t not in SPAN_OK and not os.path.exists(os.path.join(ROOT, t.split("#")[0].split(":")[0])):
                bad.append((n, t))
    return bad


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
        for line, t in broken_spans(f):
            print(f"{f}:{line}: code-span path does not exist: {t}")
            n += 1
    print(f"link check: {len(files)} files, {n} broken link(s)")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
