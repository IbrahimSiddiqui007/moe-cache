#!/usr/bin/env python3
"""tests/claims_lint_wide_test.py: `lint --wide` number forms and file set, no model needed (standard library only)."""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "bin"))
import moe_cache_claims as m

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok: fails.append(what)

def wide(line):
    return m.perf_numbers(line, wide=True)

# -- new number forms (wide=True): each finds its number (lookahead context is not part of the match)
check(wide("12.5 GB/s of host RAM read") == ["12.5 GB/s"], "throughput GB/s")
check(wide("about 4.2 MB/s") == ["4.2 MB/s"], "throughput MB/s")
check(wide("reads 3.1 GiB/s here") == ["3.1 GiB/s"], "throughput GiB/s")
check(wide("the next turn is 0.4 s faster") == ["0.4 s"], "duration s followed by faster")
check(wide("it saves 250 ms per token") == ["250 ms"], "ms followed by per token")
check(wide("startup is 1.2 minutes slower") == ["1.2 minutes"], "minutes followed by slower")
check(wide("3.9 sec to first token") == ["3.9 sec"], "sec followed by to first token")
check(wide("TTFT 3.9 s on the laptop") == ["TTFT 3.9 s"], "duration preceded by TTFT")
check(wide("latency 250 ms measured") == ["latency 250 ms"], "duration preceded by latency")
check(wide("time to first token 4.4 s") == ["time to first token 4.4 s"], "duration preceded by time to first token")
check(wide("30% faster than stock") == ["30%"], "unsigned percent followed by faster")
check(wide("20 % lower memory use") == ["20 %"], "unsigned percent with space followed by lower")
check(wide("a 45% reduction in prompt time") == ["45%"], "unsigned percent followed by reduction")
check(wide("30% less RAM for the cache") == ["30%"], "unsigned percent followed by less")
check(wide("a 10% improvement") == ["10%"], "unsigned percent followed by improvement")
check(wide("a 3-fold speedup") == ["3-fold"], "N-fold")
check(wide("decode is 2 times faster") == ["2 times faster"], "N times faster")
# -- existing PERF forms still found in wide mode
check(wide("<td>12 tok/s</td><!-- 5 tok/s --> +3.5 % 2x") == ["12 tok/s", "+3.5 %", "2x"], "wide keeps the default forms (tok/s, signed percent, Nx) and skips comments")

# -- negatives: wide still does not match plain sizes, versions, dates, ports, context lengths, hex
check(wide("24 GB VRAM") == [], "no plain size 24 GB VRAM")
check(wide("8 GB RAM and a 12.5 GB file") == [], "no plain sizes in RAM or file text")
check(wide("64k context") == [], "no context length 64k")
check(wide("moe-cache v0.3.9") == [], "no version v0.3.9")
check(wide("the mask 0x1f") == [], "no hex 0x1f")
check(wide("port 8080") == [], "no port 8080")
check(wide("it took 5 minutes") == [], "no bare duration without a comparison word")
check(wide("30% of users") == [], "no unsigned percent without a comparison word")
check(wide("2026-11-03, 4 slots") == [], "no dates or bare counts")

# -- default behaviour unchanged
check(m.perf_numbers("12.5 GB/s") == [], "default perf_numbers ignores 12.5 GB/s (no hit without wide)")
check(m.perf_numbers("<td>12 tok/s</td><!-- 5 tok/s --> +3.5 % 2x 16 GB") == ["12 tok/s", "+3.5 %", "2x"], "default perf_numbers unchanged on the bench_record_test line")

# -- markers, tags and fences still clear/skip
check(wide("<td>12.5 GB/s</td>") == ["12.5 GB/s"], "HTML tags are stripped, the number inside counts")
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "sample.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("12.5 GB/s faster\n```\n12.5 GB/s faster\n```\n<!-- claim:x --> 30% faster\n")
    probs, shown, ok = m.lint_file(p, {"x": {"id": "x"}}, {"x": ("current", "")}, wide=True)
    check(probs == [(1, "unclaimed", "12.5 GB/s")], "lint_file wide: fenced code skipped, marker clears, unclaimed line reported")
    check(shown == 2 and ok == 1, "lint_file wide: 2 lines show numbers, 1 with a current claim")

# -- --wide file set: sorted, public only, current release notes only
import tempfile
with tempfile.TemporaryDirectory() as d:
    cl = os.path.join(d, "CHANGELOG.md")
    open(cl, "w").write("# Changelog\n## 0.3.9\n- 12 tok/s\n## 0.3.8\n- 99 tok/s\n")
    pr, sh, _ = m.lint_file(cl, {}, {}, wide=True)
    check(pr == [(3, "unclaimed", "12 tok/s")], "wide reads only the newest CHANGELOG section")
    check(len(m.lint_file(cl, {}, {})[0]) == 2, "default lint_file reads the whole file")

files = [os.path.relpath(f, m.ROOT).replace(os.sep, "/") for f in m.wide_files()]
check(files == sorted(files), "wide file list is sorted")
check(m.wide_files() == m.wide_files(), "wide file list is deterministic")
check("README.md" in files, "wide includes README.md")
check("CHANGELOG.md" in files and "CONTRIBUTING.md" in files and "benchmarks/README.md" in files, "wide includes CHANGELOG, CONTRIBUTING, benchmarks/README")
check("docs/mockup/chat-3pane.html" in files and "docs/benchmark_card.html" in files, "wide includes the docs html files")
check(not any(f.startswith(("docs/internal/", "docs/results-archive/", "archive/", "third_party/", "benchmarks/claims/", "tests/")) for f in files), "wide excludes internal, results-archive, archive, third_party, claims, tests")
check("benchmarks/BOARD.md" not in files and "benchmarks/JOURNAL.md" not in files, "wide excludes the generated board and journal")
check("docs/releases/RELEASE_NOTES_v0.3.8.md" not in files, "wide excludes older release notes")
check("docs/releases/RELEASE_NOTES_v" + m.current_version() + ".md" in files, "wide keeps the release notes of the current version")

# -- command line: --warn exits 0; default lint reads exactly the default three files; --wide reads the wide set
def lint(args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "bin", "moe_cache_claims.py"), "lint", "--warn"] + args,
                       capture_output=True, text=True, cwd=ROOT)
    return r.returncode, r.stdout

rc, out = lint([])
check(rc == 0, "default lint --warn exits 0")
default_files = sorted(l.split(": ", 1)[0] for l in out.splitlines() if " lines show performance numbers" in l)
check(default_files == ["README.md", "docs/RESULTS_CURRENT.md", "docs/benchmark_card.html"], "default lint reads exactly the default three files")
rc, out = lint(["--wide"])
check(rc == 0, "wide lint --warn exits 0")
wide_out = sorted(l.split(": ", 1)[0] for l in out.splitlines() if " lines show performance numbers" in l)
check("README.md" in wide_out and "benchmarks/README.md" in wide_out, "wide lint reports on README.md and benchmarks/README.md")
check(not any(f.startswith("docs/internal/") or f.startswith("docs/results-archive/") for f in wide_out), "wide lint never reads internal or archived files")
check("docs/releases/RELEASE_NOTES_v0.3.8.md" not in wide_out, "wide lint never reads older release notes")

print("claims lint wide test: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)
