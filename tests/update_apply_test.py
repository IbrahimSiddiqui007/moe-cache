#!/usr/bin/env python3
"""moe_cache.py update --apply (bin/moe_cache_update.py): a fake release folder served through file://, the real install.sh. Checks: a copy the
installer did not put there is refused, same version is a no-op, --dry-run downloads nothing, a good update installs the new version and keeps
the previous one in versions/, a damaged package makes the installer refuse and leaves the old install working. Linux only.
   python3 tests/update_apply_test.py"""
import hashlib, io, os, shutil, subprocess, sys, tarfile, tempfile
if not sys.platform.startswith("linux"):
    print("update --apply test: Linux only, skipped"); sys.exit(0)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_update as up
T = tempfile.mkdtemp(prefix="moe-update-apply-")
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def release(ver, tamper=False, modules=True):
    d = os.path.join(T, "rel-" + ver + ("-bad" if tamper else "") + ("" if modules else "-nomod")); os.makedirs(d, exist_ok=True)
    n = f"moe-cache-v{ver}-ggml0.26-linux-x86_64"
    py = f'import sys\nprint("moe-cache {ver}" if sys.argv[1:2] == ["--version"] else "args")\n'.encode()
    with tarfile.open(os.path.join(d, n + ".tar.gz"), "w:gz") as tf:
        ti = tarfile.TarInfo(n + "/bin/moe_cache.py"); ti.size = len(py); ti.mode = 0o755; tf.addfile(ti, io.BytesIO(py))
        for m in (up.KIT_MODULES.split(", ")[1:] if modules else []):      # stand-ins for the launcher modules the check imports
            ti = tarfile.TarInfo(f"{n}/bin/{m}.py"); ti.size = 5; tf.addfile(ti, io.BytesIO(b"pass\n"))
    h = hashlib.sha256(open(os.path.join(d, n + ".tar.gz"), "rb").read()).hexdigest()
    open(os.path.join(d, "SHA256SUMS"), "w").write(("0" * 64 if tamper else h) + f"  {n}.tar.gz\n")
    open(os.path.join(d, "install.sh"), "w").write(open(os.path.join(HERE, "install.sh")).read())
    return d

prefix, bindir, home = os.path.join(T, "data", "moe-cache"), os.path.join(T, "home", "bin"), os.path.join(T, "home")
os.makedirs(bindir)
# RELEASE-686: a fake `gh` first on PATH (FAKE_GH_AUTH / FAKE_GH_VERIFY = exit codes), and a PATH without any gh
FAKEBIN = os.path.join(T, "fakebin"); os.makedirs(FAKEBIN)
open(os.path.join(FAKEBIN, "gh"), "w").write('''#!/bin/sh
case "$1 $2" in
  "auth status") exit ${FAKE_GH_AUTH:-0};;
  "attestation verify") echo "$@" >> "${FAKE_GH_LOG:-/dev/null}"; exit ${FAKE_GH_VERIFY:-0};;
esac
exit 1
'''); os.chmod(os.path.join(FAKEBIN, "gh"), 0o755)
NOGH = os.path.join(T, "nogh"); os.makedirs(NOGH)
for tool in "bash sh python3 tar gzip sha256sum awk sed grep cut head tail tr cat ls mkdir mktemp rm mv cp ln chmod readlink dirname basename uname id sort curl wget ldconfig rmdir nproc env printf test find xargs date".split():
    w = shutil.which(tool)
    if w and not os.path.exists(os.path.join(NOGH, tool)):
        os.symlink(w, os.path.join(NOGH, tool))
GH_LOG = os.path.join(T, "gh.log")
env = dict(os.environ, HOME=home, PATH=FAKEBIN + ":" + os.environ["PATH"], FAKE_GH_LOG=GH_LOG)
def say_to_str(fn):
    buf, old = io.StringIO(), up.say
    up.say = lambda m: buf.write(m + "\n")
    try:
        rc = fn()
    finally:
        up.say = old
    return rc, buf.getvalue()

# a checkout is not an installed copy
rc, out = say_to_str(lambda: up.apply("9.9.1", HERE, tag="v9.9.2", yes=True, env=env))
check(rc == 1 and "not put here by the moe-cache installer" in out, "a checkout / unpacked copy is refused")

# install 9.9.1 with the installer, as a user would
r1 = release("9.9.1")
subprocess.run(["bash", os.path.join(r1, "install.sh"), "--prefix", prefix, "--bin-dir", bindir, "--no-apt"], env=dict(env, MOE_CACHE_BASE_URL="file://" + r1, MOE_CACHE_REF="v9.9.1"),
               capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120, check=True)
here1 = os.path.join(prefix, "versions", "moe-cache-v9.9.1-ggml0.26-linux-x86_64")
lay = up.installed_layout(here1, False, env)
check(lay and lay["kind"] == "linux" and lay["prefix"] == prefix and lay["command"] == os.path.join(bindir, "moe-cache"), f"the installed layout is read from .installed ({lay})")

rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="v9.9.1", yes=True, env=env))
check(rc == 0 and "already the newest" in out, "same version: nothing to do")
rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="v9.9.2", dry=True, env=env, base_url="file://" + release("9.9.2")))
check(rc == 0 and "nothing was downloaded" in out and os.listdir(os.path.join(prefix, "versions")) == [os.path.basename(here1)], "--dry-run changes nothing")
rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="v9.9.2", yes=False, env=env, tty=False, base_url="file://" + release("9.9.2")))
check(rc == 1 and "--yes" in out, "without a terminal and without --yes it stops")
rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="1.2.3; rm -rf", yes=True, env=env))
check(rc == 1 and "does not look like a release tag" in out, "a strange tag is refused")

# a damaged package: the installer refuses, the old install keeps working
rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="v9.9.2", yes=True, env=env, base_url="file://" + release("9.9.2", tamper=True)))
check(rc != 0 and subprocess.run([os.path.join(bindir, "moe-cache"), "--version"], capture_output=True, text=True).stdout.strip() == "moe-cache 9.9.1", "a damaged package is refused and 9.9.1 still runs")

# RELEASE-686: no silent skip of the attestation: gh missing / not logged in / verify failing stop the update before the installer runs,
# unless --yes-unverified (unverified=True) or MOE_CACHE_ALLOW_UNVERIFIED=1; the installer's sha256 check is never skipped
r2 = release("9.9.2"); vers_before = sorted(os.listdir(os.path.join(prefix, "versions")))
def still_9_9_1():
    return sorted(os.listdir(os.path.join(prefix, "versions"))) == vers_before and subprocess.run([os.path.join(bindir, "moe-cache"), "--version"], capture_output=True, text=True).stdout.strip() == "moe-cache 9.9.1"
for label, extra, why in (("gh not installed", dict(PATH=NOGH), "is not installed"), ("gh not logged in", dict(FAKE_GH_AUTH="1"), "not logged in"), ("gh verify fails", dict(FAKE_GH_VERIFY="1"), "verify FAILED")):
    rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="v9.9.2", yes=True, env=dict(env, **extra), base_url="file://" + r2))
    check(rc == 1 and "REFUSED" in out and "NOT checked" in out and why in out and "--yes-unverified" in out and still_9_9_1(), f"{label}: update refused, says what was not checked, nothing changed")
open(GH_LOG, "w").close()
rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="v9.9.2", dry=True, env=dict(env, PATH=NOGH), base_url="file://" + r2))
check(rc == 0 and "nothing was downloaded" in out, "--dry-run does not need gh")
rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="v9.9.2", yes=True, env=dict(env, PATH=NOGH), base_url="file://" + release("9.9.2", tamper=True), unverified=True))
check(rc != 0 and "the installer stopped" in out and "WARNING: going on without" in out and still_9_9_1(), "--yes-unverified never skips sha256: a damaged package is still refused by the installer")
rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="v9.9.2", yes=True, env=dict(env, FAKE_GH_VERIFY="1", MOE_CACHE_ALLOW_UNVERIFIED="1"), base_url="file://" + r2))
check(rc == 0 and "WARNING" in out and subprocess.run([os.path.join(bindir, "moe-cache"), "--version"], capture_output=True, text=True).stdout.strip() == "moe-cache 9.9.2", "MOE_CACHE_ALLOW_UNVERIFIED=1 with a failing verify: updates, with a warning")
# put 9.9.1 back for the rest of the test
subprocess.run(["ln", "-sfn", "versions/moe-cache-v9.9.1-ggml0.26-linux-x86_64", os.path.join(prefix, "current")], check=True)
shutil.rmtree(os.path.join(prefix, "versions", "moe-cache-v9.9.2-ggml0.26-linux-x86_64"), ignore_errors=True)
vers_before = sorted(os.listdir(os.path.join(prefix, "versions")))
# a mirror folder with a working gh: the installer script is verified (not skipped because of the mirror)
rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="v9.9.2", yes=True, env=env, base_url="file://" + r2, dry=False))
check(rc == 0 and "build attestation of the installer verified" in out and "install.sh" in open(GH_LOG).read(), "a mirror with a working gh: verified, not skipped")
subprocess.run(["ln", "-sfn", "versions/moe-cache-v9.9.1-ggml0.26-linux-x86_64", os.path.join(prefix, "current")], check=True)
shutil.rmtree(os.path.join(prefix, "versions", "moe-cache-v9.9.2-ggml0.26-linux-x86_64"), ignore_errors=True)

# a package that unpacks but whose launcher modules do not import: the check after the swap puts the old version back
rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="v9.9.3", yes=True, env=env, base_url="file://" + release("9.9.3", modules=False)))
check(rc == 1 and "failed its check" in out and "went back to 9.9.1" in out and subprocess.run([os.path.join(bindir, "moe-cache"), "--version"], capture_output=True, text=True).stdout.strip() == "moe-cache 9.9.1", f"a version that fails the check is rolled back ({out.strip()[-160:]})")

# the real update
rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="v9.9.2", yes=True, env=env, base_url="file://" + release("9.9.2")))
check(rc == 0 and subprocess.run([os.path.join(bindir, "moe-cache"), "--version"], capture_output=True, text=True).stdout.strip() == "moe-cache 9.9.2", f"update to 9.9.2 ({out.strip()[-120:]})")
check(sorted(os.listdir(os.path.join(prefix, "versions"))) == ["moe-cache-v9.9.1-ggml0.26-linux-x86_64", "moe-cache-v9.9.2-ggml0.26-linux-x86_64"], "...the previous version is kept in versions/")
check("ln -sfn versions/moe-cache-v9.9.1-ggml0.26-linux-x86_64" in out, "...and the roll-back command is printed")
subprocess.run(["ln", "-sfn", "versions/moe-cache-v9.9.1-ggml0.26-linux-x86_64", os.path.join(prefix, "current")], check=True)
check(subprocess.run([os.path.join(bindir, "moe-cache"), "--version"], capture_output=True, text=True).stdout.strip() == "moe-cache 9.9.1", "...and that command really rolls back")

print("FAIL: " + "; ".join(fails) if fails else "update --apply: all checks passed")
sys.exit(1 if fails else 0)
