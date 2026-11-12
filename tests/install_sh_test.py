#!/usr/bin/env python3
"""install.sh as the release's stand-alone installer: a fake release (tarball + SHA256SUMS) served from a folder through file://, no network.
Checks: a wrong hash is refused and installs nothing, a good one installs, a second run is idempotent, an update replaces the old version,
--uninstall removes exactly what was added (a file of the user's in the same folders stays). Linux only.
   python3 tests/install_sh_test.py"""
import hashlib, io, os, shutil, subprocess, sys, tarfile, tempfile

if not sys.platform.startswith("linux"):
    print("install.sh test: Linux only, skipped"); sys.exit(0)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = tempfile.mkdtemp(prefix="moe-install-sh-")
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# RELEASE-686: a fake `gh` (FAKE_GH_AUTH / FAKE_GH_VERIFY = exit codes; calls are logged in FAKE_GH_LOG) first on PATH, and a PATH without any gh
FAKEBIN = os.path.join(T, "fakebin"); os.makedirs(FAKEBIN)
GH_LOG = os.path.join(T, "gh.log")
open(os.path.join(FAKEBIN, "gh"), "w").write('''#!/bin/sh
case "$1 $2" in
  "auth status") exit ${FAKE_GH_AUTH_STATUS:-0};;   # exits 1 when ANY saved account is bad, whatever the active one can do: never the gate
  "attestation verify") echo "$@" >> "${FAKE_GH_LOG:-/dev/null}"
    [ "${FAKE_GH_AUTH:-0}" = 0 ] || { echo "To get started with GitHub CLI, please run:  gh auth login" >&2; exit 4; }   # no working login
    echo "Loaded digest sha256:abc for file (progress text on stderr, as the real gh prints)" >&2
    [ -z "${FAKE_GH_MSG:-}" ] || echo "$FAKE_GH_MSG" >&2; exit ${FAKE_GH_VERIFY:-0};;
esac
exit 1
'''); os.chmod(os.path.join(FAKEBIN, "gh"), 0o755)
NOGH = os.path.join(T, "nogh"); os.makedirs(NOGH)
for tool in "bash sh python3 tar gzip sha256sum awk sed grep cut head tail tr cat ls mkdir mktemp rm mv cp ln chmod readlink dirname basename uname id sort curl wget ldconfig rmdir nproc env printf test find xargs date".split():
    w = shutil.which(tool)
    if w and not os.path.exists(os.path.join(NOGH, tool)):
        os.symlink(w, os.path.join(NOGH, tool))
assert not os.path.exists(os.path.join(NOGH, "gh"))
GOOD_ENV = dict(PATH=FAKEBIN + ":" + os.environ["PATH"], FAKE_GH_LOG=GH_LOG)

def release(ver, tamper=False):
    """A release folder: moe-cache-v<ver>-ggml0.26-linux-x86_64.tar.gz with bin/moe_cache.py, and SHA256SUMS (wrong hash when tamper)."""
    d = os.path.join(T, "rel-" + ver + ("-bad" if tamper else "")); os.makedirs(d, exist_ok=True)
    n = f"moe-cache-v{ver}-ggml0.26-linux-x86_64"
    py = f'import sys\nprint("moe-cache {ver}" if sys.argv[1:2] == ["--version"] else "args: " + " ".join(sys.argv[1:]))\n'.encode()
    with tarfile.open(os.path.join(d, n + ".tar.gz"), "w:gz") as tf:
        ti = tarfile.TarInfo(n + "/bin/moe_cache.py"); ti.size = len(py); ti.mode = 0o755; tf.addfile(ti, io.BytesIO(py))
        so = tarfile.TarInfo(n + "/llama/libllama.so.0.8.0"); so.size = 3; tf.addfile(so, io.BytesIO(b"elf"))      # RELEASE-686: the real tarball ships llama.cpp's libraries with symlinks
        for ln_, tg in (("libllama.so.0", "libllama.so.0.8.0"), ("libllama.so", "libllama.so.0")):
            sl = tarfile.TarInfo(n + "/llama/" + ln_); sl.type = tarfile.SYMTYPE; sl.linkname = tg; tf.addfile(sl)
    h = hashlib.sha256(open(os.path.join(d, n + ".tar.gz"), "rb").read()).hexdigest()
    open(os.path.join(d, "SHA256SUMS"), "w").write(("0" * 64 if tamper else h) + f"  {n}.tar.gz\n" + "1" * 64 + "  other-file.tar.gz\n")
    return d

inst = os.path.join(T, "dl", "install.sh"); os.makedirs(os.path.dirname(inst))
open(inst, "w").write(open(os.path.join(HERE, "install.sh")).read())      # alone in its folder, as downloaded
prefix, bindir = os.path.join(T, "data", "moe-cache"), os.path.join(T, "home", "bin")
os.makedirs(bindir); open(os.path.join(bindir, "mine.sh"), "w").write("keep me\n")
def run(rel, ver, *args, **extra):
    env = dict(os.environ, MOE_CACHE_BASE_URL="file://" + rel, MOE_CACHE_REF="v" + ver, HOME=os.path.join(T, "home"), **GOOD_ENV); env.update(extra)
    r = subprocess.run(["sh", inst, "--prefix", prefix, "--bin-dir", bindir, "--no-apt", *args], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL, timeout=120)
    return r.returncode, r.stdout + r.stderr
shim = os.path.join(bindir, "moe-cache")

rc, out = run(release("9.9.1", tamper=True), "9.9.1")
check(rc != 0 and "REFUSED" in out and "sha256" in out, "a wrong hash is refused")
check(not os.path.exists(prefix) and not os.path.exists(shim), "...and nothing was installed")

rc, out = run(release("9.9.1"), "9.9.1", "--dry-run")
check(rc == 0 and "would download" in out and not os.path.exists(prefix), "--dry-run prints the plan and installs nothing")

# ---- RELEASE-686 (2026-11-12): hardening: the tag shape, a plain-http mirror, a link inside the tarball
rc, out = run(release("9.9.1"), "9.9.1", MOE_CACHE_REF="v9.9.1/../../x")
check(rc != 0 and "does not look like a release tag" in out and not os.path.exists(prefix), "MOE_CACHE_REF with a path in it is refused")
rc, out = run(release("9.9.1"), "9.9.1", MOE_CACHE_BASE_URL="http://example.invalid/mirror")
check(rc != 0 and "not https" in out and not os.path.exists(prefix), "a plain-http mirror is refused")
rc, out = run(release("9.9.1"), "9.9.1", MOE_CACHE_BASE_URL="http://localhost:80@example.invalid/mirror")
check(rc != 0 and "user part" in out and not os.path.exists(prefix), "a plain-http address with a user part (localhost:80@evil) is refused")
dl = release("9.9.3"); nl = "moe-cache-v9.9.3-ggml0.26-linux-x86_64"
with tarfile.open(os.path.join(dl, nl + ".tar.gz"), "a:") if False else tarfile.open(os.path.join(dl, nl + ".tar.gz"), "w:gz") as tf:
    py = b'print("x")\n'; ti = tarfile.TarInfo(nl + "/bin/moe_cache.py"); ti.size = len(py); ti.mode = 0o755; tf.addfile(ti, io.BytesIO(py))
    ln = tarfile.TarInfo(nl + "/bin/evil"); ln.type = tarfile.SYMTYPE; ln.linkname = "/etc/passwd"; tf.addfile(ln)
open(os.path.join(dl, "SHA256SUMS"), "w").write(hashlib.sha256(open(os.path.join(dl, nl + ".tar.gz"), "rb").read()).hexdigest() + f"  {nl}.tar.gz\n")
rc, out = run(dl, "9.9.3", MOE_CACHE_ALLOW_UNVERIFIED="1")
check(rc != 0 and "link to a file name beside it" in out and not os.path.exists(prefix), "a tarball with a symlink member is refused before it is unpacked")

# ---- RELEASE-686: the build attestation. Without it the install stops unless the user asked for it; the sha256 is checked either way.
rA = release("9.9.1")
def refused(label, **extra):
    rc, out = run(rA, "9.9.1", **extra)
    check(rc != 0 and "REFUSED" in out and "NOT checked: who built the file" in out and "checked:     sha256" in out and "--yes-unverified" in out
          and not os.path.exists(prefix) and not os.path.exists(shim), f"{label}: refused, says what was and was not checked, nothing installed")
    return out
out = refused("gh not installed", PATH=NOGH)
check("is not installed" in out, "...reason named: gh not installed")
out = refused("gh not logged in", FAKE_GH_AUTH="1")
check("not logged in" in out, "...reason named: not logged in")
out = refused("gh verify fails", FAKE_GH_VERIFY="1")
check("verify FAILED" in out, "...reason named: verify failed")
open(GH_LOG, "w").close()
rc, out = run(rA, "9.9.1", "--dry-run")
check(rc == 0 and open(GH_LOG).read() == "", "--dry-run asks gh nothing and installs nothing")
rc, out = run(rA, "9.9.1")          # a mirror (MOE_CACHE_BASE_URL is set) is NOT a reason to skip: gh is asked, verifies the downloaded file
check(rc == 0 and "build attestation verified" in out and "tar.gz" in open(GH_LOG).read() and "-R IbrahimSiddiqui007/moe-cache" in open(GH_LOG).read(), "a mirror folder with a working gh: the file is verified (not skipped)")
rc, out = run(rA, "9.9.1", "--uninstall"); check(rc == 0 and not os.path.exists(prefix), "...(uninstalled again)")
# the PC bug (2026-11-06): `gh auth status` exits 1 because a SECOND saved account has a bad token, while the active login works and verify passes
rc, out = run(rA, "9.9.1", FAKE_GH_AUTH_STATUS="1")
check(rc == 0 and "build attestation verified" in out, f"gh auth status failing (a bad second account) does not stop an install whose verify passes ({out.strip()[-120:]})")
rc, out = run(rA, "9.9.1", "--uninstall"); check(rc == 0, "...(uninstalled again)")
rc, out = run(rA, "9.9.1", "--yes-unverified", PATH=NOGH)
check(rc == 0 and "WARNING: continuing without the build attestation" in out and "NOT checked" in out and os.access(shim, os.X_OK), "--yes-unverified with no gh: installs, with a warning that says what was not checked")
rc, out = run(rA, "9.9.1", "--uninstall"); check(rc == 0, "...(uninstalled again)")
rc, out = run(rA, "9.9.1", MOE_CACHE_ALLOW_UNVERIFIED="1", FAKE_GH_VERIFY="1")
check(rc != 0 and "REFUSED" in out and "FAILED" in out and "never override" in out and not os.path.exists(prefix), "MOE_CACHE_ALLOW_UNVERIFIED=1 does NOT override a verification that ran and FAILED")
rc, out = run(rA, "9.9.1", "--yes-unverified", FAKE_GH_VERIFY="1")
check(rc != 0 and "never override" in out and not os.path.exists(prefix) and not os.path.exists(shim), "--yes-unverified does NOT override a failed verification either")
rc, out = run(rA, "9.9.1", FAKE_GH_VERIFY="1", FAKE_GH_MSG="dial tcp: lookup api.github.com: no such host")
check(rc != 0 and "REFUSED" in out and "could not reach GitHub" in out and not os.path.exists(prefix), "GitHub unreachable (the check could not run), no flag: refused")
rc, out = run(rA, "9.9.1", "--yes-unverified", FAKE_GH_VERIFY="1", FAKE_GH_MSG="dial tcp: lookup api.github.com: no such host")
check(rc == 0 and "WARNING" in out and "could not reach GitHub" in out and os.access(shim, os.X_OK), "a verify that could not run (GitHub unreachable) + --yes-unverified: installs, with a warning")
rc, out = run(rA, "9.9.1", "--uninstall"); check(rc == 0, "...(uninstalled again)")
open(GH_LOG, "w").close()
rc, out = run(rA, "9.9.1")
check(rc == 0 and "--signer-workflow IbrahimSiddiqui007/moe-cache/.github/workflows/release.yml" in open(GH_LOG).read(), "gh attestation verify is called with the pinned --signer-workflow")
rc, out = run(rA, "9.9.1", "--uninstall"); check(rc == 0, "...(uninstalled again)")
rc, out = run(release("9.9.1", tamper=True), "9.9.1", "--yes-unverified", PATH=NOGH)
check(rc != 0 and "REFUSED" in out and "sha256" in out and not os.path.exists(prefix), "--yes-unverified never skips the sha256: a wrong hash is still refused")

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

# ---- shared prefix, other --bin-dir, non-ASCII / quote in the path (AUD-053, 054, 055)
os.environ["LC_ALL"] = "C"
sp = os.path.join(T, "data2", "Jos\u00e9's apps"); sb = os.path.join(T, "other bin")
os.makedirs(os.path.join(sp, "versions", "my-own-tool")); os.makedirs(sb)
def run2(rel, ver, *args):
    env = dict(os.environ, MOE_CACHE_BASE_URL="file://" + rel, MOE_CACHE_REF="v" + ver, HOME=os.path.join(T, "home"), **GOOD_ENV)
    r = subprocess.run(["sh", inst, "--prefix", sp, "--bin-dir", sb, "--no-apt", *args], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL, timeout=120)
    return r.returncode, r.stdout + r.stderr
rc, out = run2(r1, "9.9.1")
shim2 = os.path.join(sb, "moe-cache")
check(rc == 0 and subprocess.run(["/bin/sh", shim2, "x"], capture_output=True, text=True).stdout.strip() == "args: x", f"a prefix with a non-ASCII letter and an apostrophe: the command runs under /bin/sh ({out.strip().splitlines()[-4:-3]})")
rc, out = run2(r3, "9.9.3")
check(rc == 0 and os.path.isdir(os.path.join(sp, "versions", "my-own-tool")), "an update does not delete folders in versions/ that are not ours")
rc, out = subprocess.run(["sh", inst, "--prefix", sp, "--uninstall"], capture_output=True, text=True, env=dict(os.environ, HOME=os.path.join(T, "home"), MOE_CACHE_REF="v9.9.3", **GOOD_ENV), stdin=subprocess.DEVNULL).returncode, ""
check(rc == 0 and not os.path.exists(shim2) and os.path.isdir(os.path.join(sp, "versions", "my-own-tool")), "uninstall without --bin-dir removes the command the install recorded, and still leaves folders that are not ours")

check(subprocess.run(["bash", "-n", os.path.join(HERE, "install.sh")]).returncode == 0, "install.sh parses")
print("FAIL" if fails else "PASS", f"({len(fails)} failed)"); sys.exit(1 if fails else 0)
