#!/usr/bin/env python3
"""own_cache_test (LAUNCHER-1222 item 5): bin/moe_cache.own_cache_base / CACHE_DIR. An install in its own -Root / --prefix keeps its launcher cache beside it; XDG_CACHE_HOME
the user set still wins; an install made before the flag existed (or a plain checkout) keeps the user cache. Real function, temp folders only.
   python3 tests/own_cache_test.py"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok: fails.append(what)

T = tempfile.mkdtemp(prefix="moe-owncache-")
def mk(*parts):
    p = os.path.join(T, *parts); os.makedirs(p, exist_ok=True); return p
def write(path, text):
    with open(path, "w", encoding="utf-8") as f: f.write(text)

# ---- Windows layout: <root>\kit + <root>\install.json
r = mk("w1"); kit = mk("w1", "kit")
write(os.path.join(r, "install.json"), json.dumps({"version": "9", "own_cache": True}))
check(mc.own_cache_base(kit, windows=True) == os.path.join(os.path.realpath(r), "cache"), "windows: own_cache true -> <root>/cache")
r = mk("w2"); kit = mk("w2", "kit")
write(os.path.join(r, "install.json"), json.dumps({"version": "9", "own_cache": False}))
check(mc.own_cache_base(kit, windows=True) is None, "windows: own_cache false -> user cache")
r = mk("w3"); kit = mk("w3", "kit")
write(os.path.join(r, "install.json"), json.dumps({"version": "9"}))
check(mc.own_cache_base(kit, windows=True) is None, "windows: a record from before the flag -> user cache (nothing moves)")
r = mk("w4"); kit = mk("w4", "kit")
write(os.path.join(r, "install.json"), "{not json")
check(mc.own_cache_base(kit, windows=True) is None, "windows: a damaged record -> user cache, no crash")
write(os.path.join(r, "install.json"), "﻿" + json.dumps({"own_cache": True}))
check(mc.own_cache_base(kit, windows=True) is not None, "windows: a record with a BOM (Windows PowerShell 5.1 writes none, other tools do) is read")
r = mk("w5", "notkit")
write(os.path.join(os.path.dirname(r), "install.json"), json.dumps({"own_cache": True}))
check(mc.own_cache_base(r, windows=True) is None, "windows: a copy not named kit beside a record is not an install")
check(mc.own_cache_base(mk("w6", "kit"), windows=True) is None, "windows: no record -> user cache")

# ---- Linux layout: <prefix>/versions/moe-cache-v*/ + <prefix>/.installed (own_cache=1)
p = mk("l1"); v = mk("l1", "versions", "moe-cache-v9-ggml0.26-linux-x86_64")
write(os.path.join(p, ".installed"), "version=9\nprefix=%s\ncommand=x\nown_cache=1\n" % p)
check(mc.own_cache_base(v, windows=False) == os.path.join(os.path.realpath(p), "cache"), "linux: own_cache=1 -> <prefix>/cache")
os.symlink(os.path.join("versions", "moe-cache-v9-ggml0.26-linux-x86_64"), os.path.join(p, "current"))
check(mc.own_cache_base(os.path.join(p, "current"), windows=False) == os.path.join(os.path.realpath(p), "cache"), "linux: reached through the `current` link (how the command starts it)")
p = mk("l2"); v = mk("l2", "versions", "moe-cache-v9-ggml0.26-linux-x86_64")
write(os.path.join(p, ".installed"), "version=9\nprefix=%s\ncommand=x\nown_cache=0\n" % p)
check(mc.own_cache_base(v, windows=False) is None, "linux: own_cache=0 -> user cache")
p = mk("l3"); v = mk("l3", "versions", "moe-cache-v9-ggml0.26-linux-x86_64")
write(os.path.join(p, ".installed"), "version=9\nprefix=%s\ncommand=x\n" % p)
check(mc.own_cache_base(v, windows=False) is None, "linux: a record without the line (older install) -> user cache (nothing moves)")
check(mc.own_cache_base(HERE, windows=False) is None, "a plain checkout -> user cache")

# ---- CACHE_DIR in a fresh process: XDG wins, then the install's own cache, then ~/.cache
def cache_dir(kit_copy, xdg=None):
    env = dict(os.environ); env.pop("XDG_CACHE_HOME", None)
    if xdg: env["XDG_CACHE_HOME"] = xdg
    code = "import sys; sys.path.insert(0, %r); import moe_cache as m; print(m.CACHE_DIR)" % os.path.join(kit_copy, "bin")
    return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env).stdout.strip()
p = mk("e1"); v = mk("e1", "versions", "moe-cache-v9-ggml0.26-linux-x86_64"); os.makedirs(os.path.join(v, "bin"))
for f in os.listdir(os.path.join(HERE, "bin")):
    if f.endswith(".py"):
        os.symlink(os.path.join(HERE, "bin", f), os.path.join(v, "bin", f))
write(os.path.join(p, ".installed"), "version=9\nprefix=%s\ncommand=x\nown_cache=1\n" % p)
if os.name != "nt":
    os.symlink("versions/moe-cache-v9-ggml0.26-linux-x86_64", os.path.join(p, "current"))
    # symlinked .py files make abspath(__file__) the install's folder (realpath of HERE is the checkout): copy moe_cache.py so HERE is the install
    import shutil
    for f in os.listdir(os.path.join(v, "bin")): os.unlink(os.path.join(v, "bin", f))
    for f in os.listdir(os.path.join(HERE, "bin")):
        if f.endswith(".py"): shutil.copy(os.path.join(HERE, "bin", f), os.path.join(v, "bin", f))
    check(cache_dir(os.path.join(p, "current")) == os.path.join(os.path.realpath(p), "cache", "moe-cache"), "CACHE_DIR of an own-cache install = <prefix>/cache/moe-cache")
    check(cache_dir(os.path.join(p, "current"), xdg=mk("xdg")) == os.path.join(T, "xdg", "moe-cache"), "...XDG_CACHE_HOME the user set wins")
    check(cache_dir(HERE) == os.path.join(os.path.expanduser("~"), ".cache", "moe-cache"), "CACHE_DIR of a plain checkout = ~/.cache/moe-cache (default unchanged)")
print("own_cache_test: %s" % ("FAIL (%d)" % len(fails) if fails else "ok"))
sys.exit(1 if fails else 0)
