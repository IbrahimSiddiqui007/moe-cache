#!/usr/bin/env python3
"""Build a zip that is the same bytes every time (PACKAGING-828): reproducible_zip.py ROOT NAME OUT.zip
Zips ROOT/NAME with sorted entries, the date of SOURCE_DATE_EPOCH (default: the last commit, else 1980-01-01), fixed permissions and no user or
group names, so two builds of the same commit give the same sha256."""
import os, sys, time, zipfile


def stamp():
    e = os.environ.get("SOURCE_DATE_EPOCH", "")
    t = int(e) if e.isdigit() else 315532800          # 1980-01-01, the zip minimum
    return time.gmtime(max(t, 315532800))[:6]


def build(root, name, out):
    dt = stamp()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d, dirs, files in os.walk(os.path.join(root, name)):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(d, f)
                zi = zipfile.ZipInfo(os.path.relpath(p, root).replace(os.sep, "/"), dt)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.create_system = 3
                zi.external_attr = (0o755 if os.access(p, os.X_OK) else 0o644) << 16
                with open(p, "rb") as fh:
                    z.writestr(zi, fh.read())


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    build(*sys.argv[1:])
