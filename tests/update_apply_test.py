#!/usr/bin/env python3
"""moe_cache.py update --apply (bin/moe_cache_update.py): a fake release folder served through file://, the real install.sh. Checks: a copy the
installer did not put there is refused, same version is a no-op, --dry-run downloads nothing, a good update installs the new version and keeps
the previous one in versions/, a damaged package makes the installer refuse and leaves the old install working. Linux only.
   python3 tests/update_apply_test.py"""
import hashlib, io, os, subprocess, sys, tarfile, tempfile
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

def release(ver, tamper=False):
    d = os.path.join(T, "rel-" + ver + ("-bad" if tamper else "")); os.makedirs(d, exist_ok=True)
    n = f"moe-cache-v{ver}-ggml0.26-linux-x86_64"
    py = f'import sys\nprint("moe-cache {ver}" if sys.argv[1:2] == ["--version"] else "args")\n'.encode()
    with tarfile.open(os.path.join(d, n + ".tar.gz"), "w:gz") as tf:
        ti = tarfile.TarInfo(n + "/bin/moe_cache.py"); ti.size = len(py); ti.mode = 0o755; tf.addfile(ti, io.BytesIO(py))
    h = hashlib.sha256(open(os.path.join(d, n + ".tar.gz"), "rb").read()).hexdigest()
    open(os.path.join(d, "SHA256SUMS"), "w").write(("0" * 64 if tamper else h) + f"  {n}.tar.gz\n")
    open(os.path.join(d, "install.sh"), "w").write(open(os.path.join(HERE, "install.sh")).read())
    return d

prefix, bindir, home = os.path.join(T, "data", "moe-cache"), os.path.join(T, "home", "bin"), os.path.join(T, "home")
os.makedirs(bindir)
env = dict(os.environ, HOME=home)
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

# the real update
rc, out = say_to_str(lambda: up.apply("9.9.1", here1, tag="v9.9.2", yes=True, env=env, base_url="file://" + release("9.9.2")))
check(rc == 0 and subprocess.run([os.path.join(bindir, "moe-cache"), "--version"], capture_output=True, text=True).stdout.strip() == "moe-cache 9.9.2", f"update to 9.9.2 ({out.strip()[-120:]})")
check(sorted(os.listdir(os.path.join(prefix, "versions"))) == ["moe-cache-v9.9.1-ggml0.26-linux-x86_64", "moe-cache-v9.9.2-ggml0.26-linux-x86_64"], "...the previous version is kept in versions/")
check("ln -sfn versions/moe-cache-v9.9.1-ggml0.26-linux-x86_64" in out, "...and the roll-back command is printed")
subprocess.run(["ln", "-sfn", "versions/moe-cache-v9.9.1-ggml0.26-linux-x86_64", os.path.join(prefix, "current")], check=True)
check(subprocess.run([os.path.join(bindir, "moe-cache"), "--version"], capture_output=True, text=True).stdout.strip() == "moe-cache 9.9.1", "...and that command really rolls back")

print("FAIL: " + "; ".join(fails) if fails else "update --apply: all checks passed")
sys.exit(1 if fails else 0)
