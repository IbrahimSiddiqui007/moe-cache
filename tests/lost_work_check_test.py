#!/usr/bin/env python3
"""lost_work_check_test (TOOLS-1309 a): a failed git call must fail the check, never read as an empty, clean result. Temp repo only.
   python3 tests/lost_work_check_test.py"""
import os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
import lost_work_check as lw

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok: fails.append(what)

T = tempfile.mkdtemp(prefix="moe-lostwork-")
def git(*a): subprocess.run(["git", "-C", T, *a], check=True, capture_output=True, env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))
git("init", "-q", "-b", "base"); open(os.path.join(T, "a"), "w").write("x\n"); git("add", "a"); git("commit", "-q", "-m", "one")
args = ["--repo", T, "--base", "base", "--no-claims", "--accept", os.path.join(T, "none.tsv")]

check(lw.main(args) == 0, "a healthy repo with nothing unmerged: OK (exit 0)")
g = lw.Git(T)
try:
    g.out("rev-parse", "--verify", "no-such-ref"); raised = False
except lw.GitError:
    raised = True
check(raised, "Git.out raises on a failed git call")
check(lw.Git(T).out("rev-parse", "--short", "base").strip() != "", "Git.out still returns the output of a good call")
# git fails on the ref listing: the check must not call that clean
real = lw.Git.run
def failing(self, *a):
    return (128, "") if a and a[0] == "for-each-ref" else real(self, *a)
lw.Git.run = failing
try:
    rc = lw.main(args)
finally:
    lw.Git.run = real
check(rc == 2, f"git for-each-ref failing: exit 2, not a clean 0 (got {rc})")
print("FAIL (%d failed)" % len(fails) if fails else "OK"); sys.exit(1 if fails else 0)
