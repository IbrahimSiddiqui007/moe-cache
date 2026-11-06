#!/usr/bin/env python3
"""moe_cache.py update --apply on Windows (bin/moe_cache_update.py + the real install.ps1): a fake release folder (zip + SHA256SUMS-windows + install.ps1),
an install into a temporary -Root with -NoPath (the user's PATH is not touched), then update --apply. Checks: the new kit runs, the previous kit stays as
kit.prev, a version that fails the check after the swap is rolled back (kit.failed is kept, nothing is deleted by the test's own steps). Windows only.
   python tests\\update_apply_win_test.py [WORKDIR]      (WORKDIR default: a temp folder, left in place)"""
import io, os, subprocess, sys, tempfile, zipfile, hashlib
if sys.platform != "win32":
    print("update --apply Windows test: Windows only, skipped"); sys.exit(0)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_update as up
T = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="moe-update-win-")
os.makedirs(T, exist_ok=True)
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def release(ver, modules=True):
    d = os.path.join(T, "rel-" + ver + ("" if modules else "-nomod")); os.makedirs(d, exist_ok=True)
    n = f"moe-cache-v{ver}-windows-x64"
    py = f'import sys\nprint("moe-cache {ver}" if sys.argv[1:2] == ["--version"] else "args")\n'
    with zipfile.ZipFile(os.path.join(d, n + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(n + "/bin/moe_cache.py", py)
        for extra in ("moe-cache.exe", "ggml-moe-cache-api3.dll", "llama/llama-server.exe"):      # the files install.ps1 requires after the unpack
            z.writestr(f"{n}/{extra}", "x")
        for m in (up.KIT_MODULES.split(", ")[1:] if modules else []):
            z.writestr(f"{n}/bin/{m}.py", "pass\n")
    h = hashlib.sha256(open(os.path.join(d, n + ".zip"), "rb").read()).hexdigest()
    open(os.path.join(d, "SHA256SUMS-windows"), "w").write(f"{h}  {n}.zip\n")
    open(os.path.join(d, "install.ps1"), "w", encoding="utf-8").write(open(os.path.join(HERE, "install.ps1"), encoding="utf-8").read().replace("@VERSION@", ver))
    return d

# RELEASE-686: a fake gh.cmd first on PATH (FAKE_GH_AUTH / FAKE_GH_VERIFY = exit codes): the installer and the updater refuse without a verified build attestation
FAKEGH = os.path.join(T, "fakegh"); os.makedirs(FAKEGH, exist_ok=True)
open(os.path.join(FAKEGH, "gh.cmd"), "w").write("@echo off\r\nif \"%1 %2\"==\"auth status\" exit /b %FAKE_GH_AUTH_STATUS%\r\nif \"%1 %2\"==\"attestation verify\" if not \"%FAKE_GH_AUTH%\"==\"0\" (echo To get started with GitHub CLI, please run:  gh auth login 1>&2 & exit /b 4)\r\nif \"%1 %2\"==\"attestation verify\" echo Loaded digest sha256:abc, progress text on stderr as the real gh prints 1>&2\r\nif \"%1 %2\"==\"attestation verify\" exit /b %FAKE_GH_VERIFY%\r\nexit /b 1\r\n")
genv = dict(os.environ, PATH=FAKEGH + os.pathsep + os.environ["PATH"], FAKE_GH_AUTH="0", FAKE_GH_AUTH_STATUS="0", FAKE_GH_VERIFY="0")

def ps(*args, env=None):
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", *args], capture_output=True, text=True, env=env, timeout=300)
    return r.returncode, r.stdout + r.stderr

root = os.path.join(T, "root")
r1 = release("9.9.1")
rc, out = ps(os.path.join(r1, "install.ps1"), "-Root", root, "-NoPath", "-BaseUrl", r1, "-Version", "9.9.1", env=genv)
check(rc == 0 and "installed: moe-cache 9.9.1" in out, f"install 9.9.1 ({out.strip()[-150:]})")
kit = os.path.join(root, "kit")
lay = up.installed_layout(kit, True, os.environ)
check(lay and lay["kind"] == "windows" and lay["root"] == os.path.realpath(root), f"layout read from install.json ({lay})")

def say_to_str(fn):
    buf, old = io.StringIO(), up.say
    up.say = lambda m: buf.write(m + "\n")
    try:
        rc = fn()
    finally:
        up.say = old
    return rc, buf.getvalue()
def url(d):
    return "file:///" + d.replace("\\", "/")
def shim_version():
    return subprocess.run([os.path.join(root, "bin", "moe-cache.cmd"), "--version"], capture_output=True, text=True).stdout.strip()

rc, out = say_to_str(lambda: up.apply("9.9.1", kit, tag="v9.9.3", yes=True, windows=True, env=genv, base_url=url(release("9.9.3", modules=False))))
check(rc == 1 and "failed its check" in out and shim_version() == "moe-cache 9.9.1", f"a version that fails the check is rolled back ({out.strip()[-200:]})")
check(os.path.isdir(os.path.join(root, "kit.failed")) and not os.path.isdir(os.path.join(root, "kit.prev")), "...the failed kit is kept as kit.failed, the old kit is the kit again")
failed = os.path.join(root, "kit.failed"); os.rename(failed, os.path.join(T, "failed-kept"))      # moved away, not deleted: the next update needs the name free

for label, extra in (("gh not logged in", dict(FAKE_GH_AUTH="1")), ("gh verify fails", dict(FAKE_GH_VERIFY="1"))):
    rc, out = say_to_str(lambda: up.apply("9.9.1", kit, tag="v9.9.2", yes=True, windows=True, env=dict(genv, **extra), base_url=url(release("9.9.2"))))
    check(rc == 1 and "REFUSED" in out and "NOT checked" in out and "--yes-unverified" in out and shim_version() == "moe-cache 9.9.1" and not os.path.isdir(os.path.join(root, "kit.prev")), f"{label}: update refused, nothing changed")
# the real gh prints its login report on stderr: install.ps1 itself must still say why it stopped (it once ended with only "github.com ... Stopped")
r9 = release("9.9.4")
rc, out = ps(os.path.join(r9, "install.ps1"), "-Root", os.path.join(T, "root_nogh"), "-NoPath", "-BaseUrl", r9, "-Version", "9.9.4", env=dict(genv, FAKE_GH_AUTH="1"))
check(rc != 0 and "NOT checked" in out and "gh auth login" in out and "-YesUnverified" in out, f"install.ps1 with a gh that has no working login names the reason and the way out ({out.strip()[-300:]})")
rc, out = ps(os.path.join(r9, "install.ps1"), "-Root", os.path.join(T, "root_badsecond"), "-NoPath", "-BaseUrl", r9, "-Version", "9.9.4", env=dict(genv, FAKE_GH_AUTH_STATUS="1"))
check(rc == 0 and "build attestation verified" in out, f"install.ps1: gh auth status failing (a bad second account) does not stop an install whose verify passes ({out.strip()[-300:]})")
rc, out = say_to_str(lambda: up.apply("9.9.1", kit, tag="v9.9.2", yes=True, windows=True, env=genv, base_url=url(release("9.9.2"))))
check(rc == 0 and shim_version() == "moe-cache 9.9.2", f"update to 9.9.2 ({out.strip()[-200:]})")
check(os.path.isdir(os.path.join(root, "kit.prev")) and not os.path.isdir(os.path.join(root, "kit.old")), "...the previous kit stays as kit.prev")
prev_ver = subprocess.run([sys.executable, os.path.join(root, "kit.prev", "bin", "moe_cache.py"), "--version"], capture_output=True, text=True).stdout.strip()
check(prev_ver == "moe-cache 9.9.1", f"...and kit.prev is 9.9.1 ({prev_ver})")
err = up.roll_back(up.installed_layout(os.path.join(root, "kit"), True, os.environ))
check(err == "" and shim_version() == "moe-cache 9.9.1", f"roll_back makes 9.9.1 current again ({err})")

print("FAIL: " + "; ".join(fails) if fails else "update --apply (Windows): all checks passed")
print("work folder (not deleted):", T)
sys.exit(1 if fails else 0)
