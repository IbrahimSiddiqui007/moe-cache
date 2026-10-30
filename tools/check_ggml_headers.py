#!/usr/bin/env python3
"""Compare the ggml headers this repo vendors (third_party/ggml, third_party/ggml-v3) with a llama.cpp commit: a pin-move check that needs no build.
   python3 tools/check_ggml_headers.py --tree PATH/TO/llama.cpp --commit ff0d396e3 [--vendored third_party/ggml-v3]
For every vendored header it prints identical / additions only / changed. Additions (new declarations) cannot break a plugin built with the old header; a vendored line that
is gone or changed upstream is a declaration or struct layout the plugin may rely on: exit 1, and the lines are printed. ggml-backend-impl.h (the buffer/backend structs
the plugin fills in) is the file to read twice. Reads the commit with `git show`; nothing is downloaded (a partial clone without the blob says "not available")."""
import argparse, difflib, os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def show(tree, commit, path):
    r = subprocess.run(["git", "-C", tree, "show", f"{commit}:{path}"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.splitlines() if r.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--vendored", default=os.path.join(HERE, "third_party", "ggml-v3"))
    a = ap.parse_args()
    bad = missing = 0
    for sub, up in (("include", "ggml/include"), ("src", "ggml/src")):
        d = os.path.join(a.vendored, sub)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not f.endswith(".h"):
                continue
            mine = open(os.path.join(d, f), encoding="utf-8", errors="replace").read().splitlines()
            theirs = show(a.tree, a.commit, f"{up}/{f}")
            if theirs is None:
                print(f"  ??   {sub}/{f}: not available at {a.commit}"); missing += 1; continue
            if mine == theirs:
                print(f"  same {sub}/{f}"); continue
            sm = difflib.SequenceMatcher(None, mine, theirs, autojunk=False)
            removed = [(i1, mine[i1:i2], theirs[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag in ("delete", "replace")]
            added = sum(j2 - j1 for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag in ("insert", "replace"))
            if not removed:
                print(f"  add  {sub}/{f}: {added} new lines upstream, nothing of ours changed")
                continue
            bad += 1
            print(f"  DIFF {sub}/{f}: {len(removed)} place(s) where our lines are gone or changed upstream ({added} new lines)")
            for at, old, new in removed[:6]:
                print(f"       line {at + 1}: ours {old[:3]} -> theirs {new[:3]}")
    print(("FAIL" if bad else "ok") + f": {bad} vendored header(s) changed, {missing} not available (commit {a.commit})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
