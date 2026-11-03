#!/usr/bin/env python3
"""packaging/fill_manifests.py (AUD-092): on a tiny fake kit zip, the scoop and winget manifests come out filled (version, sha256, upper-case sha256 for
winget, date), the scoop JSON stays valid, no @PLACEHOLDER@ is left, and a wrongly named / wrongly laid out zip is refused. Also guards AUD-059: the
winget installer manifest must not pull in a second Python."""
import contextlib, hashlib, io, json, os, re, sys, tempfile, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "packaging"))
import fill_manifests as fm  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

T = tempfile.mkdtemp(prefix="fillm-test-")
def make_zip(name, files):
    p = os.path.join(T, name)
    with zipfile.ZipFile(p, "w") as z:
        for f in files:
            z.writestr(f, "x")
    return p
def run(argv):
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            fm.main(argv)
        return 0, out.getvalue()
    except SystemExit as e:
        return e.code, out.getvalue()

VER = "9.8.7"
top = f"moe-cache-v{VER}-windows-x64/"
good = make_zip(f"moe-cache-v{VER}-windows-x64.zip", [top + "moe-cache.bat", top + "moe-cache.exe", top + "bin/moe_cache.py"])
sha = hashlib.sha256(open(good, "rb").read()).hexdigest()
O = os.path.join(T, "out")
rc, out = run([good, "--out", O, "--date", "2026-11-03"])
check(rc == 0 and f"sha256 {sha}" in out, f"a good kit zip fills the manifests (rc {rc})")
j = json.load(open(os.path.join(O, "scoop", "moe-cache.json")))
check(j["version"] == VER and j["architecture"]["64bit"]["hash"] == sha and VER in j["architecture"]["64bit"]["url"], "scoop JSON: version, hash and url filled, valid JSON")
wdir = os.path.join(O, "winget", "manifests", "i", "IbrahimSiddiqui007", "MoeCache", VER)
files = sorted(os.listdir(wdir)) if os.path.isdir(wdir) else []
check(len(files) == 3, f"winget: three manifests in the winget-pkgs layout ({files})")
text = {n: open(os.path.join(wdir, n), encoding="utf-8").read() for n in files}
inst = next((t for n, t in text.items() if n.endswith(".installer.yaml")), "")
check(f'InstallerSha256: "{sha.upper()}"' in inst and f'PackageVersion: "{VER}"' in inst and 'ReleaseDate: "2026-11-03"' in inst, "winget installer: upper-case sha256, version, date")
check(f"moe-cache-v{VER}-windows-x64\\moe-cache.exe" in inst and f"download/v{VER}/moe-cache-v{VER}-windows-x64.zip" in inst, "winget installer: nested file path and url follow the version")
check(all(not re.search(r"@[A-Z_0-9]+@", t) for t in list(text.values()) + [open(os.path.join(O, "scoop", "moe-cache.json")).read()]), "no @PLACEHOLDER@ left in any output")
_code = "\n".join(l for l in inst.splitlines() if not l.lstrip().startswith("#"))
check("PackageDependencies" not in _code and "Python.Python" not in _code, "winget installer: no hard Python dependency (AUD-059)")
check(all(not t.endswith("\r\n") and "\r" not in t for t in text.values()), "outputs use LF line endings")
# the templates next to the script are not modified
check("@VERSION@" in open(os.path.join(ROOT, "packaging", "winget", "IbrahimSiddiqui007.MoeCache.installer.yaml")).read(), "the templates keep their placeholders")

rc, out = run([make_zip("kit.zip", [top + "moe-cache.bat"]), "--out", O])
check(isinstance(rc, str) and "not a kit zip name" in rc, "a zip with the wrong name is refused")
rc, out = run([make_zip(f"moe-cache-v1.0.0-windows-x64.zip", ["moe-cache-v1.0.0-windows-x64/moe-cache.bat"]), "--out", O])
check(isinstance(rc, str) and "expected every file under" in rc, "a kit without moe-cache.exe is refused")
rc, out = run([make_zip(f"moe-cache-v1.0.1-windows-x64.zip", ["moe-cache-v1.0.1-windows-x64/moe-cache.bat", "moe-cache-v1.0.1-windows-x64/moe-cache.exe", "stray.txt"]), "--out", O])
check(isinstance(rc, str) and "expected every file under" in rc, "a file outside the top folder is refused")

print("fill_manifests_test: " + ("PASS" if not fails else "FAILED: " + "; ".join(fails)))
sys.exit(1 if fails else 0)
