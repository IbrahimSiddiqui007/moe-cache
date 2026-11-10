#!/usr/bin/env python3
"""tools/rn_blanks_check.py: fails on any `{{RN: ...}}` blank in CHANGELOG.md, README.md or docs/releases/RELEASE_NOTES_v*.md (a made-up tree), passes without; and release.yml runs it.
   python3 tests/rn_blanks_test.py"""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = 0


def chk(name, cond):
    global fails
    print(("  ok   " if cond else "  FAIL ") + name)
    fails += 0 if cond else 1


def tree(files):
    d = tempfile.mkdtemp()
    for rel, text in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8").write(text)
    return d


def run(d):
    return subprocess.run([sys.executable, os.path.join(HERE, "tools", "rn_blanks_check.py"), "--root", d], capture_output=True, text=True)


clean = {"CHANGELOG.md": "# Changelog\nspeed 1.2x\n", "README.md": "hello\n", "docs/releases/RELEASE_NOTES_v9.9.9.md": "notes\n"}
r = run(tree(clean)); chk("no blanks: exit 0", r.returncode == 0)
for rel in clean:
    f = dict(clean); f[rel] += "decode {{RN: IQ3_XXS 65k decode}} tok/s\n"
    r = run(tree(f)); chk(f"a blank in {rel}: exit 1, file named", r.returncode == 1 and rel.split("/")[-1] in r.stdout)
f = dict(clean); f["README.md"] = "a {{RN: x}} and {{RN: y}}\n"
r = run(tree(f)); chk("two blanks on one line are counted", r.returncode == 1 and "2 blank(s)" in r.stdout)
wf = open(os.path.join(HERE, ".github", "workflows", "release.yml"), encoding="utf-8").read()
chk("release.yml runs tools/rn_blanks_check.py", "tools/rn_blanks_check.py" in wf)
print("rn_blanks_test: " + ("FAIL" if fails else "PASS"))
sys.exit(1 if fails else 0)
