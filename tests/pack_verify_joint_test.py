#!/usr/bin/env python3
"""Joint test: `moe_cache.py pack --hash` output is accepted by the plugin's verify code (tests/pack_verify_joint.cpp replicating MOE_CACHE_PACK_VERIFY=1), a corrupted payload is refused.
   python3 tests/pack_verify_joint_test.py   (needs g++; skips with a message when it is missing and CI is unset)"""
import os, shutil, struct, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bin"))
from pack_test import make_gguf          # noqa: E402
import moe_cache_pack as mp              # noqa: E402

fails = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)


def flip_byte(path, off):
    with open(path, "r+b") as f:
        f.seek(off)
        b = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([b[0] ^ 1]))


d = tempfile.mkdtemp(prefix="pack-verify-joint-")
cpp = os.path.join(HERE, "pack_verify_joint.cpp")
exe = os.path.join(d, "pack_verify_joint")
gpp = shutil.which("g++")
if not gpp:
    msg = "g++ not found: skipping pack_verify_joint_test (install g++ to run it)"
    print(msg)
    sys.exit(1 if os.environ.get("CI") else 0)

r = subprocess.run([gpp, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-Isrc", cpp, "-o", exe],
                   cwd=ROOT, capture_output=True, text=True)
check(r.returncode == 0, f"the helper compiles clean with -Wall -Wextra -Werror ({r.stderr[-200:] if r.returncode else 'ok'})")


def helper(pack):
    return subprocess.run([exe, pack], capture_output=True, text=True, timeout=120)


def pack_for(name, *args):
    m = os.path.join(d, name + ".gguf")
    make_gguf(m)
    check(mp.pack_main([m, *args]) == 0, f"{name}: pack_main {list(args) or ''} ok")
    return m.replace(".gguf", ".moepack"), m


# a good --hash pack is accepted: exit 0, "ok: 3 layers"
pack, _ = pack_for("good", "--hash")
r = helper(pack)
check(r.returncode == 0 and "ok: 3 layers" in r.stdout, f"intact --hash pack: exit 0 and 'ok: 3 layers' (exit {r.returncode}, out {r.stdout.strip()!r})")

# one flipped byte in the last layer's payload -> exit 3 naming that layer (same trick as pack_test.test_hash_catches_a_flipped_byte)
pack_last, m_last = pack_for("last", "--hash")
flip_byte(pack_last, os.path.getsize(pack_last) - 4096 + 3)
files, layers = mp.plan(m_last)
last = max(layers)
r = helper(pack_last)
check(r.returncode == 3 and f"REFUSED: layer {last}:" in r.stdout,
      f"corrupt last layer: exit 3 naming layer {last} (exit {r.returncode}, out {r.stdout.strip()!r})")

# one flipped byte in layer 0's payload of a fresh pack -> exit 3 naming layer 0
pack_l0, m_l0 = pack_for("l0", "--hash")
lay, _ = mp.layout(mp.plan(m_l0)[1], mp.round_up(256 + 512 * len(mp.plan(m_l0)[1])))
flip_byte(pack_l0, lay[0][0] + 3)
r = helper(pack_l0)
check(r.returncode == 3 and "REFUSED: layer 0:" in r.stdout,
      f"corrupt layer 0: exit 3 naming layer 0 (exit {r.returncode}, out {r.stdout.strip()!r})")

# a pack made without --hash has no payload hash -> exit 2
pack_nh, _ = pack_for("nohash")
r = helper(pack_nh)
check(r.returncode == 2 and "REFUSED: no payload hash" in r.stdout,
      f"pack without --hash: exit 2 (exit {r.returncode}, out {r.stdout.strip()!r})")

# a --hash pack whose header hash-flag bytes 28..31 are zeroed -> exit 2
pack_fz, _ = pack_for("flagzero", "--hash")
with open(pack_fz, "r+b") as f:
    f.seek(28)
    f.write(b"\0\0\0\0")
check(struct.unpack_from("<I", open(pack_fz, "rb").read(32), 28)[0] == 0, "flagzero: header bytes 28..31 are zero")
r = helper(pack_fz)
check(r.returncode == 2 and "REFUSED: no payload hash" in r.stdout,
      f"zeroed hash flag: exit 2 (exit {r.returncode}, out {r.stdout.strip()!r})")

import shutil; shutil.rmtree(d, ignore_errors=True)
print(f"pack_verify_joint_test: {len(fails)} failed")
sys.exit(1 if fails else 0)
