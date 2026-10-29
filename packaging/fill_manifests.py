#!/usr/bin/env python3
"""Fill the scoop and winget manifests from a built Windows kit: python3 packaging/fill_manifests.py dist/moe-cache-vVERSION-windows-x64.zip [--out DIR]

Reads the version from the zip's name and its SHA256 from the file, checks the zip's top folder and moe-cache.bat are where the manifests
say, and writes (default DIR: dist/packaging):
  DIR/scoop/moe-cache.json                                           (for a scoop bucket)
  DIR/winget/manifests/i/IbrahimSiddiqui007/MoeCache/VERSION/*.yaml  (the layout of microsoft/winget-pkgs)
The templates next to this script stay as they are. Nothing is submitted or published. Standard library only (Linux and Windows).
"""
import argparse, datetime, hashlib, json, os, re, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "dist", "packaging"))
    ap.add_argument("--date", default=datetime.date.today().isoformat(), help="release date for winget (default today)")
    a = ap.parse_args(argv)
    m = re.fullmatch(r"moe-cache-v(.+)-windows-x64\.zip", os.path.basename(a.zip))
    if not m:
        sys.exit(f"not a kit zip name (moe-cache-vVERSION-windows-x64.zip): {a.zip}")
    ver = m.group(1)
    top = f"moe-cache-v{ver}-windows-x64/"
    with zipfile.ZipFile(a.zip) as z:
        names = z.namelist()
    if not all(n.startswith(top) for n in names) or top + "moe-cache.bat" not in names or top + "moe-cache.exe" not in names:
        sys.exit(f"{a.zip}: expected every file under {top}, and {top}moe-cache.bat / moe-cache.exe (the manifests' extract_dir, bin, RelativeFilePath)")
    h = hashlib.sha256()
    with open(a.zip, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    sha = h.hexdigest()
    subst = {"@VERSION@": ver, "@SHA256@": sha, "@SHA256_UPPER@": sha.upper(), "@DATE@": a.date}

    def fill(src, dst):
        with open(src, encoding="utf-8") as f:
            t = f.read()
        for k, v in subst.items():
            t = t.replace(k, v)
        left = re.findall(r"@[A-Z_0-9]+@", t)
        if left:
            sys.exit(f"{src}: unfilled {left}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(t)
        return t

    out = os.path.abspath(a.out)
    s = fill(os.path.join(HERE, "scoop", "moe-cache.json"), os.path.join(out, "scoop", "moe-cache.json"))
    j = json.loads(s)                                                        # must stay valid JSON
    assert j["version"] == ver and j["architecture"]["64bit"]["hash"] == sha
    wdir = os.path.join(out, "winget", "manifests", "i", "IbrahimSiddiqui007", "MoeCache", ver)
    for n in ("IbrahimSiddiqui007.MoeCache.yaml", "IbrahimSiddiqui007.MoeCache.installer.yaml", "IbrahimSiddiqui007.MoeCache.locale.en-US.yaml"):
        fill(os.path.join(HERE, "winget", n), os.path.join(wdir, n))
    print(f"version {ver}, sha256 {sha}")
    print(f"  {os.path.join(out, 'scoop', 'moe-cache.json')}")
    print(f"  {wdir}{os.sep}*.yaml (3 files)")
    print("Nothing submitted: the scoop bucket and the winget-pkgs pull request need the owner's OK.")


if __name__ == "__main__":
    main()
