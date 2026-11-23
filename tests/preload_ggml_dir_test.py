#!/usr/bin/env python3
"""preload_ggml_dir_test: LD_PRELOAD loads put the directory of the server's own libggml-base first on LD_LIBRARY_PATH, so the plugin's libggml-base resolves to the
server's copy, not /usr/lib's (two ggml versions in one process: SIGSEGV exit 139 with a distro llama-server in /usr/lib/llama.cpp-cuda, 2026-11-23). Fake ldd output, no model."""
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
import moe_cache as mc  # noqa: E402
FAILS = []


def check(c, w):
    print(("  ok   " if c else "  FAIL ") + w)
    if not c:
        FAILS.append(w)


with tempfile.TemporaryDirectory() as t:
    real = os.path.join(t, "libggml-base.so.0.25.1")
    open(real, "w").close()
    link = os.path.join(t, "libggml-base.so.0")
    os.symlink(real, link)
    ldd = f"\tlibggml.so.0 => {t}/libggml.so.0 (0x1)\n\tlibggml-base.so.0 => {link} (0x2)\n\tlibc.so.6 => /usr/lib/libc.so.6 (0x3)\n"
    check(mc.server_ggml_dir(ldd) == os.path.realpath(t), "directory of the server's libggml-base (symlink resolved)")
    check(mc.server_ggml_dir("\tlibc.so.6 => /usr/lib/libc.so.6 (0x3)\n") is None, "no libggml-base in ldd: None")
    check(mc.server_ggml_dir("\tlibggml-base.so.0 => not found\n") is None, "'not found' gives None")
    if not mc.WINDOWS:
        mc.ldd = lambda p: ldd
        mc.which = lambda p: p
        old = os.environ.pop("LD_LIBRARY_PATH", None)
        env, method = mc.load_env("llama-server", "/x/lib.so", "preload")
        d = dict(env)
        check(method == "preload" and d.get("LD_PRELOAD") == "/x/lib.so", "preload keeps LD_PRELOAD")
        check(d.get("LD_LIBRARY_PATH") == os.path.realpath(t), "LD_LIBRARY_PATH = server's ggml dir")
        os.environ["LD_LIBRARY_PATH"] = "/opt/a"
        d = dict(mc.load_env("llama-server", "/x/lib.so", "preload")[0])
        check(d.get("LD_LIBRARY_PATH") == os.path.realpath(t) + ":/opt/a", "user's LD_LIBRARY_PATH kept after it")
        if old is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = old
sys.exit(1 if FAILS else 0)
