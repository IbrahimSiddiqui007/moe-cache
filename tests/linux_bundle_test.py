#!/usr/bin/env python3
"""The Linux tarball's bundled official llama.cpp (tools/bundle_llama_linux.sh, launcher search dirs, install.sh): a fake llama.cpp tarball, no network,
no model, no compiler. Linux only (the bundle script is bash).
   python3 tests/linux_bundle_test.py"""
import hashlib, io, os, shutil, subprocess, sys, tarfile, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-linbundle-")
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

if not sys.platform.startswith("linux"):
    print("linux bundle test: skipped (Linux only)"); sys.exit(0)

# ---- a fake "official" tarball: llama-b99999/ with llama-server, a versioned library and its symlinks, no licence (the script fetches / copies it) ----
def make_tgz(path, with_server=True):
    with tarfile.open(path, "w:gz") as t:
        def add(name, data=b"", mode=0o644, link=None):
            ti = tarfile.TarInfo(name)
            if link:
                ti.type, ti.linkname = tarfile.SYMTYPE, link
                t.addfile(ti)
            else:
                ti.size, ti.mode = len(data), mode
                t.addfile(ti, io.BytesIO(data))
        if with_server:
            add("llama-b99999/llama-server", b"#!/bin/sh\necho 'version: 99999 (abc1234)'\n", 0o755)
        add("llama-b99999/libggml-base.so.0.28.0", b"\0ggml\0")
        add("llama-b99999/libggml-base.so.0", link="libggml-base.so.0.28.0")
        add("llama-b99999/libggml-base.so", link="libggml-base.so.0")
        add("llama-b99999/libggml-cpu-x64.so", b"\0cpu\0")
cache = os.path.join(TMP, "cache"); os.makedirs(cache)
asset = "llama-b99999-bin-ubuntu-vulkan-x64.tar.gz"
tgz = os.path.join(cache, asset); make_tgz(tgz)
sha = hashlib.sha256(open(tgz, "rb").read()).hexdigest()
script = os.path.join(HERE, "tools", "bundle_llama_linux.sh")
lic = os.path.join(HERE, "LICENSE")
env = dict(os.environ, LLAMA_LICENSE_FILE=lic)

def run(dest, tag, name, digest, cache_dir=cache):
    return subprocess.run(["bash", script, dest, tag, name, digest, cache_dir], capture_output=True, text=True, env=env, timeout=120)

dest = os.path.join(TMP, "dest-ok")
r = run(dest, "b99999", asset, sha)
llama = os.path.join(dest, "llama")
check(r.returncode == 0 and os.access(os.path.join(llama, "llama-server"), os.X_OK), f"bundle: llama-server lands in DEST/llama and is executable ({r.stderr.strip()[:80]})")
check(os.path.islink(os.path.join(llama, "libggml-base.so")) and os.path.isfile(os.path.join(llama, "libggml-base.so.0.28.0")), "bundle: the libraries and their symlinks are kept")
check(os.path.isfile(os.path.join(llama, "LICENSE-llama.cpp.txt")), "bundle: llama.cpp's licence file is copied")
note = open(os.path.join(llama, "README-moe-cache.txt")).read()
check("b99999" in note and sha in note and "unmodified" in note and "libvulkan1" in note, "bundle: README-moe-cache.txt names the tag, the digest and the Vulkan loader")
check(subprocess.run([os.path.join(llama, "llama-server"), "--version"], capture_output=True, text=True).stdout.startswith("version: 99999"), "the bundled llama-server runs from there")

dest2 = os.path.join(TMP, "dest-badsha")
r = run(dest2, "b99999", asset, "0" * 64)
check(r.returncode != 0 and "sha256 does not match" in r.stdout and not os.path.exists(os.path.join(dest2, "llama")), "a download that does not match the pinned sha256 is refused, nothing is copied")
bad = os.path.join(cache, "llama-b99999-bin-ubuntu-x64.tar.gz"); make_tgz(bad, with_server=False)
r = run(os.path.join(TMP, "dest-noserver"), "b99999", os.path.basename(bad), hashlib.sha256(open(bad, "rb").read()).hexdigest())
check(r.returncode != 0 and "no llama-server" in r.stdout, "a tarball without llama-server is refused")
check(subprocess.run(["bash", script], capture_output=True, text=True).returncode == 1, "no arguments: usage and exit 1")

# ---- the launcher finds <root>/llama: a copy of bin/ + planner/ with the bundle next to it ----
root = os.path.join(TMP, "root")
shutil.copytree(os.path.join(HERE, "bin"), os.path.join(root, "bin"), ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(os.path.join(HERE, "planner"), os.path.join(root, "planner"), ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(llama, os.path.join(root, "llama"), symlinks=True)
probe = r"""
import os, sys
sys.path.insert(0, os.path.join(%r, "bin"))
import moe_cache as mc
mc.shutil.which = lambda *a, **k: None          # a llama-server on this machine's PATH would win over the folders
mc.saved_llama_server = lambda: None
dirs = mc.llama_search_dirs()
r = mc.find_llama_server()
print(os.path.join(%r, "llama") in dirs)
print(r["path"])
print(r["source"])
""" % (root, root)
env2 = dict(os.environ, XDG_CACHE_HOME=os.path.join(TMP, "xdg"), HOME=os.path.join(TMP, "home"))
for k in ("LLAMA_SERVER",):
    env2.pop(k, None)
o = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True, env=env2, timeout=60)
lines = o.stdout.split("\n")
check(lines[0] == "True", f"launcher: llama_search_dirs lists <root>/llama on Linux ({o.stderr.strip()[:100]})")
check(lines[1] == os.path.join(root, "llama", "llama-server") and lines[2] == "a default folder", f"launcher: find_llama_server picks the bundled llama-server ({lines[1:3]})")
# the real tree (no llama/ in the checkout) still lists the folder: nothing breaks when it is absent
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc
check(os.path.join(HERE, "llama") in mc.llama_search_dirs(), "the checkout's search dirs include <root>/llama too (absent folder is skipped)")

# the Windows kit uses the same rule (the folder is <kit>/llama, not the kit's parent: a mistake in the first version of the kit commit)
_w, _up = mc.WINDOWS, os.environ.get("USERPROFILE")
mc.WINDOWS = True; os.environ["USERPROFILE"] = os.path.join(TMP, "nouser")
try:
    wd = mc.llama_search_dirs()
finally:
    mc.WINDOWS = _w
    if _up is None: os.environ.pop("USERPROFILE", None)
    else: os.environ["USERPROFILE"] = _up
check(os.path.join(HERE, "llama") in wd and os.path.join(os.path.dirname(HERE), "llama") not in wd, "Windows search dirs: the kit's own llama folder (<kit>/llama), not its parent's")

# ---- install.sh: bundled mode prints its plan; LLAMA_SHA and --own-llama build instead ----
inst = os.path.join(root, "install.sh"); shutil.copy(os.path.join(HERE, "install.sh"), inst)
def plan(*args, **envx):
    return subprocess.run(["bash", inst, "--dry-run", *args], capture_output=True, text=True, env=dict(os.environ, **envx), timeout=30).stdout.strip()
check("bundled llama.cpp" in plan() and "no build" in plan(), f"install.sh: uses the bundled llama.cpp ({plan()[:70]})")
check("would build llama.cpp" in plan("--own-llama"), "install.sh --own-llama: builds")
check("would build llama.cpp 23b0202a1" in plan(LLAMA_SHA="23b0202a1abc"), "install.sh LLAMA_SHA=...: builds that commit")
check("would build llama.cpp" in subprocess.run(["bash", os.path.join(HERE, "install.sh"), "--dry-run"], capture_output=True, text=True).stdout, "install.sh in a checkout without llama/: builds, as before")
check(subprocess.run(["bash", "-n", os.path.join(HERE, "install.sh")]).returncode == 0 and subprocess.run(["bash", "-n", script]).returncode == 0, "the scripts parse")

shutil.rmtree(TMP, ignore_errors=True)
print("linux bundle test:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
