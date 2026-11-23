#!/usr/bin/env python3
"""Every relative link in the repository's Markdown files points at a file or folder that exists (tools/check_doc_links.py), and the README stays short.
   python3 tests/doc_links_test.py"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = subprocess.run([sys.executable, os.path.join(HERE, "tools", "check_doc_links.py")], capture_output=True, text=True)
print(r.stdout.strip())
m = __import__("re").search(r"link check: (\d+) files", r.stdout)
if not m or int(m.group(1)) < 10:      # audit row 16: an empty file list used to print "0 files, 0 broken" and pass
    print("FAIL: the link check saw fewer than 10 Markdown files"); sys.exit(1)
n = len(open(os.path.join(HERE, "README.md"), encoding="utf-8").read().split("\n"))
print(f"README.md: {n} lines (limit: under 200)")
# audit row 16: outside a git checkout the check walks the folder (a copy of the tool in a temp tree with one good and one broken link), and an empty tree is an error
import shutil, tempfile
T = tempfile.mkdtemp(prefix="moe-doclinks-")
os.makedirs(os.path.join(T, "tools")); shutil.copy(os.path.join(HERE, "tools", "check_doc_links.py"), os.path.join(T, "tools"))
open(os.path.join(T, "A.md"), "w").write("[ok](B.md) [bad](nope.md)\n"); open(os.path.join(T, "B.md"), "w").write("x\n")
r2 = subprocess.run([sys.executable, os.path.join(T, "tools", "check_doc_links.py")], capture_output=True, text=True, env=dict(os.environ, GIT_DIR=os.path.join(T, "no-such")))
ok2 = r2.returncode == 1 and "nope.md" in r2.stdout and "2 files" in r2.stdout
print(("  ok   " if ok2 else "  FAIL ") + f"no git: the folder is walked and the broken link is found ({r2.returncode}, {r2.stdout.strip()[:80]})")
shutil.rmtree(T, ignore_errors=True)
sys.exit(0 if r.returncode == 0 and n < 200 and ok2 else 1)
