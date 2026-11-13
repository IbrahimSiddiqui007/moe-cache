#!/usr/bin/env python3
"""Release gate for the text of a release: fails when a `{{RN: ...}}` blank (a number the release night has not filled in) is still in the CHANGELOG, the README or the
release notes. A blank that ships is a missing number on the public page.
   python3 tools/rn_blanks_check.py [--root DIR]     exit 0 when there are no blanks, 1 when there are (each one printed with file and line)"""
import glob, os, sys

FILES = ["CHANGELOG.md", "README.md", "docs/RESULTS_CURRENT.md", "docs/releases/RELEASE_NOTES_v*.md"]
DOC_FORM = "{{RN: ...}}"      # the form written out in an explanation (not a blank); a real blank always names its number


def blanks(root):
    out = []
    for pat in FILES:
        for path in sorted(glob.glob(os.path.join(root, pat))):
            with open(path, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    c = line.replace(DOC_FORM, "").count("{{RN")
                    if c:
                        out.append((os.path.relpath(path, root), n, c))
    return out


if __name__ == "__main__":
    root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    found = blanks(root)
    for path, n, c in found:
        print(f"{path}:{n}: {c} blank(s) '{{{{RN'")
    if found:
        print(f"FAIL: {sum(c for _, _, c in found)} release-night blank(s) left in {len({p for p, _, _ in found})} file(s): fill them from the release night before the tag")
        sys.exit(1)
    print("no {{RN blanks left")
