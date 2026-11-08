#!/usr/bin/env python3
"""The kits' PATCHED llama.cpp (tools/bundle_llama_patched.sh, the PATCHES-moe-cache.txt note, the launcher's start-up line): fake CI builds, no network, no model,
no compiler. Linux only (the bundle script is bash).
   python3 tests/llama_patched_bundle_test.py"""
import glob, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-patchbundle-")
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

if not sys.platform.startswith("linux"):
    print("::warning::patched bundle test SKIPPED (Linux only): nothing was checked"); sys.exit(0)

script = os.path.join(HERE, "tools", "bundle_llama_patched.sh")
lic = os.path.join(HERE, "LICENSE")
env = dict(os.environ, LLAMA_LICENSE_FILE=lic)
nums = ",".join(re.match(r"\d+", os.path.basename(p)).group(0) for p in sorted(glob.glob(os.path.join(HERE, "patches", "llama.cpp", "[0-9]*.patch"))))
want = f"llama.cpp b12064 + moe-cache patches {nums} built in, all off"
check(nums.startswith("0004,0020,0030,0040"), f"the patch set is 0004,0020,0030,0040 (got {nums}); update the docs and this test when it changes")

def write(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(data)

# ---- Linux: llama-server + a versioned library with its symlinks ----
src = os.path.join(TMP, "lin_src")
write(os.path.join(src, "llama-server"), b"#!/bin/sh\necho 'version: 12064 (51fe42c5d)'\n"); os.chmod(os.path.join(src, "llama-server"), 0o644)
write(os.path.join(src, "libggml-base.so.0.28.0"))
os.symlink("libggml-base.so.0.28.0", os.path.join(src, "libggml-base.so.0"))
dest = os.path.join(TMP, "lin_dest")
r = subprocess.run(["bash", script, dest, src, "linux"], capture_output=True, text=True, env=env)
check(r.returncode == 0, "linux bundle succeeds: " + (r.stderr or r.stdout)[-200:])
L = os.path.join(dest, "llama")
check(os.access(os.path.join(L, "llama-server"), os.X_OK), "llama-server is executable")
check(os.path.islink(os.path.join(L, "libggml-base.so.0")), "library symlinks are kept")
check(open(os.path.join(L, "PATCHES-moe-cache.txt")).read() == want + "\n", "PATCHES-moe-cache.txt holds the one-line note")
readme = open(os.path.join(L, "README-moe-cache.txt")).read()
check("OFF" in readme and "LLAMA_SERVER" in readme and "official" in readme.lower(), "README says patches are off and how to use the official build")
check(os.path.isfile(os.path.join(L, "LICENSE-llama.cpp.txt")), "licence is next to the server")

# ---- Windows: needs ggml-vulkan.dll ----
wsrc = os.path.join(TMP, "win_src"); write(os.path.join(wsrc, "llama-server.exe")); write(os.path.join(wsrc, "ggml-base.dll"))
r = subprocess.run(["bash", script, os.path.join(TMP, "win_dest"), wsrc, "win"], capture_output=True, text=True, env=env)
check(r.returncode != 0 and "ggml-vulkan.dll" in r.stdout, "windows bundle without ggml-vulkan.dll is refused")
write(os.path.join(wsrc, "ggml-vulkan.dll"))
wdest = os.path.join(TMP, "win_dest")
r = subprocess.run(["bash", script, wdest, wsrc, "win"], capture_output=True, text=True, env=env)
check(r.returncode == 0, "windows bundle succeeds: " + (r.stderr or r.stdout)[-200:])
check(open(os.path.join(wdest, "llama", "PATCHES-moe-cache.txt"), "rb").read() == (want + "\r\n").encode(), "windows note ends in CRLF")
check(os.path.isfile(os.path.join(wdest, "llama", "ggml-vulkan.dll")), "ggml-vulkan.dll is in llama\\")
r = subprocess.run(["bash", script, os.path.join(TMP, "x"), os.path.join(TMP, "nothing"), "linux"], capture_output=True, text=True, env=env)
check(r.returncode != 0, "a missing source folder is refused")

# ---- the launcher reads the note ----
sys.path.insert(0, os.path.join(HERE, "bin"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(TMP, "xdg"))
import moe_cache
check(moe_cache.llama_patch_line(os.path.join(L, "llama-server")) == want, "launcher: llama_patch_line reads the note")
off = os.path.join(TMP, "official"); write(os.path.join(off, "llama-server"))
check(moe_cache.llama_patch_line(os.path.join(off, "llama-server")) is None, "launcher: an official build (no note) prints nothing")
write(os.path.join(off, "PATCHES-moe-cache.txt"), b"not a note\n")
check(moe_cache.llama_patch_line(os.path.join(off, "llama-server")) is None, "launcher: a file that is not a note is ignored")

# ---- the release scripts use it, and keep the official path ----
mr = open(os.path.join(HERE, "tools", "make_release.sh")).read()
mk = open(os.path.join(HERE, "tools", "make_windows_kit.sh")).read()
check("LLAMA_PATCHED_LINUX_DIR" in mr and "bundle_llama_linux.sh" in mr, "make_release.sh: patched build when LLAMA_PATCHED_LINUX_DIR is set, official otherwise")
check("LLAMA_PATCHED_WIN_DIR" in mk and "llama-$LLAMA_TAG-bin-win-vulkan-x64.zip" in mk, "make_windows_kit.sh: patched build when LLAMA_PATCHED_WIN_DIR is set, official otherwise")
shutil.rmtree(TMP, ignore_errors=True)
print("llama_patched_bundle_test:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
