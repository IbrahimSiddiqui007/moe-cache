#!/usr/bin/env python3
"""build_label_test (LAUNCHER-1281): `moe_cache.py --version` and the check header name the version AND the commit; a release kit reads BUILD_COMMIT, a checkout asks git,
no source of a commit = the plain version. build.sh passes the commit to the plugin's start-up line (the line keeps its "plugin vX (ggml backend API" prefix).
   python3 tests/build_label_test.py"""
import os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok: fails.append(what)

T = tempfile.mkdtemp(prefix="moe-buildlabel-")
def kit(text=None):
    d = tempfile.mkdtemp(dir=T)
    if text is not None: open(os.path.join(d, "BUILD_COMMIT"), "w").write(text)
    return d

check(mc.build_commit(kit("abc1234def\n")) == "abc1234def", "kit with BUILD_COMMIT: the commit")
check(mc.build_commit(kit("abc1234def-dirty\n")) == "abc1234def-dirty", "...a -dirty mark survives")
check(mc.build_commit(kit("not a commit; rm -rf\n")) is None, "garbage in BUILD_COMMIT: ignored (no .git either -> None)")
check(mc.build_commit(kit()) is None, "no BUILD_COMMIT and no .git: None")
g = kit()
run = lambda *a: subprocess.run(["git", "-C", g, *a], capture_output=True, text=True, env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"})
run("init", "-q"); open(os.path.join(g, "f"), "w").write("1"); run("add", "f"); run("commit", "-q", "-m", "x")
c = mc.build_commit(g); head = run("rev-parse", "HEAD").stdout.strip()
check(bool(c) and head.startswith(c), f"checkout: git's short commit ({c})")
open(os.path.join(g, "f"), "w").write("2")
check(mc.build_commit(g) == c + "-dirty", "checkout with a local change: -dirty")
out = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "--version"], capture_output=True, text=True).stdout.strip()
check(re.fullmatch(r"moe-cache " + re.escape(mc.VERSION) + r"( \(commit [0-9a-f]{7,40}(-dirty)?\))?", out) is not None, f"--version prints the version, plus the commit when known ({out})")
src = open(os.path.join(HERE, "src", "mc_state.inl")).read()
check("plugin v" + mc.VERSION + " (ggml backend API" in src and "MC_BUILD_COMMIT_STR" in src, "plugin banner keeps the set_version.sh prefix and names the build")
bs = open(os.path.join(HERE, "build.sh")).read()
check("-DMC_BUILD_COMMIT=" in bs, "build.sh passes the commit to the compiler")
# build.sh in a kit unpacked INSIDE another git repo: BUILD_COMMIT, not the outer repo's commit (critic 2026-11-23). Only the commit part of build.sh is run.
outer = kit(); subprocess.run(["git", "-C", outer, "init", "-q"], capture_output=True)
inner = os.path.join(outer, "kit"); os.makedirs(inner); open(os.path.join(inner, "BUILD_COMMIT"), "w").write("abc123456\n")
snip = "".join(l for l in bs.splitlines(True) if l.startswith(("MC_COMMIT", "[ -n \"$MC_COMMIT", "if [ -n \"$MC_COMMIT", "case \"$MC_COMMIT")))
got = subprocess.run(["bash", "-c", snip + 'echo "$MC_COMMIT"'], cwd=inner, capture_output=True, text=True).stdout.strip()
check(got == "abc123456", f"build.sh in a kit inside another repo uses BUILD_COMMIT ({got})")
calls = []
_orig = mc.build_commit; mc.build_commit = lambda *a, **k: calls.append(1) or "abc1234"; mc._VERSION_LABEL = None
l1, l2 = mc.version_label(), mc.version_label(); mc.build_commit = _orig; mc._VERSION_LABEL = None
check(l1 == l2 == mc.VERSION + " (commit abc1234)" and len(calls) == 1, "version_label asks git once per process (the control page polls every second)")
print("FAILED: " + "; ".join(fails) if fails else "all ok"); sys.exit(1 if fails else 0)
