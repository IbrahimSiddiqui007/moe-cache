#!/usr/bin/env python3
"""install.sh as the release's stand-alone installer: a fake release (tarball + SHA256SUMS) served from a folder through file://, no network.
Checks: a wrong hash is refused and installs nothing, a good one installs, a second run is idempotent, an update replaces the old version,
--uninstall removes exactly what was added (a file of the user's in the same folders stays). Linux only.
   python3 tests/install_sh_test.py"""
import hashlib, io, os, subprocess, sys, tarfile, tempfile

if not sys.platform.startswith("linux"):
    print("install.sh test: Linux only, skipped"); sys.exit(0)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = tempfile.mkdtemp(prefix="moe-install-sh-")
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def release(ver, tamper=False):
    """A release folder: moe-cache-v<ver>-ggml0.26-linux-x86_64.tar.gz with bin/moe_cache.py, and SHA256SUMS (wrong hash when tamper)."""
    d = os.path.join(T, "rel-" + ver + ("-bad" if tamper else "")); os.makedirs(d, exist_ok=True)
    n = f"moe-cache-v{ver}-ggml0.26-linux-x86_64"
    py = f'import sys\nprint("moe-cache {ver}" if sys.argv[1:2] == ["--version"] else "args: " + " ".join(sys.argv[1:]))\n'.encode()
    with tarfile.open(os.path.join(d, n + ".tar.gz"), "w:gz") as tf:
        ti = tarfile.TarInfo(n + "/bin/moe_cache.py"); ti.size = len(py); ti.mode = 0o755; tf.addfile(ti, io.BytesIO(py))
    h = hashlib.sha256(open(os.path.join(d, n + ".tar.gz"), "rb").read()).hexdigest()
    open(os.path.join(d, "SHA256SUMS"), "w").write(("0" * 64 if tamper else h) + f"  {n}.tar.gz\n" + "1" * 64 + "  other-file.tar.gz\n")
    return d

inst = os.path.join(T, "dl", "install.sh"); os.makedirs(os.path.dirname(inst))
open(inst, "w").write(open(os.path.join(HERE, "install.sh")).read())      # alone in its folder, as downloaded
prefix, bindir = os.path.join(T, "data", "moe-cache"), os.path.join(T, "home", "bin")
os.makedirs(bindir); open(os.path.join(bindir, "mine.sh"), "w").write("keep me\n")
def run(rel, ver, *args, **extra):
    env = dict(os.environ, MOE_CACHE_BASE_URL="file://" + rel, MOE_CACHE_REF="v" + ver, HOME=os.path.join(T, "home"), **extra)
    r = subprocess.run(["sh", inst, "--prefix", prefix, "--bin-dir", bindir, "--no-apt", *args], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL, timeout=120)
    return r.returncode, r.stdout + r.stderr
shim = os.path.join(bindir, "moe-cache")

rc, out = run(release("9.9.1", tamper=True), "9.9.1")
check(rc != 0 and "REFUSED" in out and "sha256" in out, "a wrong hash is refused")
check(not os.path.exists(prefix) and not os.path.exists(shim), "...and nothing was installed")

rc, out = run(release("9.9.1"), "9.9.1", "--dry-run")
check(rc == 0 and "would download" in out and not os.path.exists(prefix), "--dry-run prints the plan and installs nothing")

r1 = release("9.9.1")
rc, out = run(r1, "9.9.1")
check(rc == 0 and "sha256 matches" in out and "installed: moe-cache 9.9.1" in out, f"install: version printed ({out.strip().splitlines()[-9:-6]})")
check(os.access(shim, os.X_OK) and subprocess.run([shim, "launch", "--x"], capture_output=True, text=True).stdout.strip() == "args: launch --x", "the moe-cache command runs the kit's moe_cache.py with its arguments")
check("undo:" in out and "--uninstall" in out, "it says how to undo")
rc, out = run(r1, "9.9.1")
check(rc == 0 and "already unpacked" in out and "installed: moe-cache 9.9.1" in out, "second run: idempotent, same version")
check(os.listdir(os.path.join(prefix, "versions")) == ["moe-cache-v9.9.1-ggml0.26-linux-x86_64"], "...one version folder")
rc, out = run(release("9.9.2"), "9.9.2")
check(rc == 0 and "installed: moe-cache 9.9.2 (was 9.9.1)" in out, "update in place says what it replaced")
check(os.listdir(os.path.join(prefix, "versions")) == ["moe-cache-v9.9.2-ggml0.26-linux-x86_64"], "...the old version folder is gone")

# ---- failure midway (AUD-049): a failed update keeps the old install, a failed first install leaves nothing
def state():
    return sorted(os.listdir(prefix)) if os.path.isdir(prefix) else None
def snapshot():
    return (state(), sorted(os.listdir(os.path.join(prefix, "versions"))) if os.path.isdir(os.path.join(prefix, "versions")) else None,
            os.readlink(os.path.join(prefix, "current")) if os.path.islink(os.path.join(prefix, "current")) else None,
            open(os.path.join(prefix, ".installed")).read() if os.path.exists(os.path.join(prefix, ".installed")) else None,
            open(shim).read() if os.path.exists(shim) else None, sorted(os.listdir(bindir)))
r3 = release("9.9.3")
before = snapshot()
for fp in ("unpack", "swap", "link", "shim", "record"):
    rc, out = run(r3, "9.9.3", MOE_CACHE_INSTALL_FAIL_AT=fp)
    check(rc != 0 and "injected at " + fp in out and snapshot() == before, f"update fails at '{fp}': the 9.9.2 install is exactly as before (no .part / .old / temp files)")
    check(subprocess.run([shim, "--version"], capture_output=True, text=True).stdout.strip() == "moe-cache 9.9.2", "...and the command still runs 9.9.2")
# a tarball whose hash matches but whose content is not a kit: refused, old install untouched
bad = os.path.join(T, "rel-9.9.4"); os.makedirs(bad)
nm = "moe-cache-v9.9.4-ggml0.26-linux-x86_64"
with tarfile.open(os.path.join(bad, nm + ".tar.gz"), "w:gz") as tf:
    ti = tarfile.TarInfo(nm + "/README"); ti.size = 2; tf.addfile(ti, io.BytesIO(b"x\n"))
open(os.path.join(bad, "SHA256SUMS"), "w").write(hashlib.sha256(open(os.path.join(bad, nm + ".tar.gz"), "rb").read()).hexdigest() + f"  {nm}.tar.gz\n")
rc, out = run(bad, "9.9.4")
check(rc != 0 and "no bin/moe_cache.py" in out and snapshot() == before, "a tarball without bin/moe_cache.py is refused, old install untouched")
# a truncated tarball with a matching hash (tar fails midway)
tr = os.path.join(T, "rel-9.9.5"); os.makedirs(tr)
nm5 = "moe-cache-v9.9.5-ggml0.26-linux-x86_64"
data = open(os.path.join(release("9.9.5"), nm5 + ".tar.gz"), "rb").read()[:-40]
open(os.path.join(tr, nm5 + ".tar.gz"), "wb").write(data)
open(os.path.join(tr, "SHA256SUMS"), "w").write(hashlib.sha256(data).hexdigest() + f"  {nm5}.tar.gz\n")
rc, out = run(tr, "9.9.5")
check(rc != 0 and snapshot() == before, "a truncated tarball (hash matches) fails with the old install untouched")
# a command of the user's: refused before anything is downloaded or unpacked
os.rename(shim, shim + ".keep"); open(shim, "w").write("#!/bin/sh\necho user\n")
rc, out = run(r3, "9.9.3")
check(rc != 0 and "not ours" in out and sorted(os.listdir(os.path.join(prefix, "versions"))) == before[1] and before[3] == open(os.path.join(prefix, ".installed")).read(), "a foreign command is refused before anything is unpacked (update keeps 9.9.2)")
os.remove(shim); os.rename(shim + ".keep", shim)
# successful update after all those failures still works
rc, out = run(r3, "9.9.3")
check(rc == 0 and "installed: moe-cache 9.9.3 (was 9.9.2)" in out and os.listdir(os.path.join(prefix, "versions")) == ["moe-cache-v9.9.3-ggml0.26-linux-x86_64"], "the update after the failures works")
# first install failing at every step leaves nothing behind
import shutil
rc, out = run(r3, "9.9.3", "--uninstall"); check(rc == 0 and not os.path.exists(prefix), "uninstall leaves no empty prefix")
for fp in ("unpack", "swap", "link", "shim", "record"):
    rc, out = run(r3, "9.9.3", MOE_CACHE_INSTALL_FAIL_AT=fp)
    check(rc != 0 and not os.path.exists(prefix) and sorted(os.listdir(bindir)) == ["mine.sh"], f"first install failing at '{fp}' leaves nothing (no prefix, no command, no temp files)")
rc, out = run(r3, "9.9.3"); check(rc == 0, "...and the next run installs normally")

open(os.path.join(prefix, "versions", "..", "user-note.txt"), "w").write("not ours\n")
rc, out = run(r1, "9.9.3", "--uninstall")
check(rc == 0 and not os.path.exists(shim) and not os.path.exists(os.path.join(prefix, "versions")) and not os.path.exists(os.path.join(prefix, "current")), "--uninstall removes the command, versions and current")
check(os.path.isfile(os.path.join(bindir, "mine.sh")) and os.path.isfile(os.path.join(prefix, "user-note.txt")), "...and only those: the user's files next to them stay")
rc, out = run(r1, "9.9.2", "--uninstall")
check(rc == 0 and "nothing to remove" in out, "uninstall twice is harmless")

open(shim, "w").write("#!/bin/sh\necho someone else's\n")
rc, out = run(r1, "9.9.1")
check(rc != 0 and "not ours" in out and "someone else's" in open(shim).read(), "a moe-cache command that is not ours is never overwritten")

check(subprocess.run(["bash", "-n", os.path.join(HERE, "install.sh")]).returncode == 0, "install.sh parses")
print("FAIL" if fails else "PASS", f"({len(fails)} failed)"); sys.exit(1 if fails else 0)
