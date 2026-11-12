#!/usr/bin/env python3
"""avail_reserve_test: on Windows the plugin's cache plans with the RAM available now minus what the whole-server cap keeps free (max(4 GiB, 12.5 % of RAM)); --ram, stock, Linux and an
environment MOE_CACHE_RAM_GIB of your own are untouched. Fake machine, no model (chain's Flash-Next probe 2026-11-13: default cache left 1.3 GB available on a 32 GB PC with apps open)."""
import contextlib, io, os, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
sys.path.insert(0, os.path.join(HERE, "..", "planner"))
import moe_cache as mc  # noqa: E402

GIB = 2 ** 30
FAILS = []


def check(c, w):
    print(("  ok   " if c else "  FAIL ") + w)
    if not c:
        FAILS.append(w)


class M(mc.Machine):
    def __init__(self, avail, total=31.1 * GIB, windows=True):
        self.avail, self.total, self._win = avail, total, windows
    @property
    def windows(self): return self._win
    def load_env(self, ls, lib, method="auto"): return [("LD_PRELOAD", lib)], "preload"
    def mem_for_server(self, ram_gb): return self.avail
    def mem_total(self): return self.total
    def mem_cap(self): return self.total - max(4 * GIB, self.total / 8)


def env_of(avail, args=(), stock=False, windows=True, total=31.1 * GIB):
    a, _, _ = mc.parse_server(["m.gguf", "--no-pin", *args])
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        env = mc.memory_env_stage(a, "llama-server", "lib.so", "m.gguf", "p", stock, "off", M(avail, total, windows))[3]
    return dict(env).get("MOE_CACHE_RAM_GIB"), out.getvalue()


saved = os.environ.pop("MOE_CACHE_RAM_GIB", None)
try:
    v, say = env_of(16.4 * GIB)
    check(v is not None and abs(float(v) - 12.4) < 0.01 and "kept free for your programs" in say, f"16.4 GiB available on a 31.1 GiB PC: the cache plans with 16.4 - 4 = 12.4 GiB ({v})")
    check("about 6.4 GiB of expert cache" in say or "about 5.4 GiB" in say or "GiB of expert cache" in say, f"the line gives the expected cache ({say.strip()[:140]})")
    v, say = env_of(30 * GIB, total=64 * GIB)
    check(abs(float(v) - 22.0) < 0.01 or abs(float(v) - (30 - 8)) < 0.01, f"64 GiB machine: the reserve is 12.5 % = 8 GiB ({v})")
    v, say = env_of(5 * GIB)
    check(v is not None and float(v) == 4.0 and "very little RAM" in say, f"little RAM available: never below 4 GiB, and a warning that the cache will be tiny ({v})")
    v, say = env_of(12 * GIB)
    check("very little RAM" in say, "12 GiB available: 12 - 4 - 3 - 3.5 = 1.5 GiB of cache: warned (critic V1)")
    v, say = env_of(20 * GIB)
    check("very little RAM" not in say, "20 GiB available: no warning")
    a2, _, _ = mc.parse_server(["m.gguf", "--no-pin"])
    check(abs(mc.avail_reserve_gib(a2, M(16.4 * GIB))[0] - 12.4) < 0.01 and mc.avail_reserve_gib(a2, M(16.4 * GIB, windows=False)) is None and mc.avail_reserve_gib(a2, M(16.4 * GIB), stock=True) is None,
          "avail_reserve_gib (the figure the MoE planner also gets, critic V5): 12.4 on Windows, None on Linux / stock")
    v, say = env_of(16.4 * GIB, args=["--ram", "10"])
    check(abs(float(v) - 10 * 1e9 / GIB) < 0.01 and "kept free" not in say, f"--ram wins ({v})")
    v, say = env_of(16.4 * GIB, stock=True)
    check(v is None, "stock: no plugin, nothing set")
    v, say = env_of(16.4 * GIB, windows=False)
    check(v is None, "Linux: unchanged")
    os.environ["MOE_CACHE_RAM_GIB"] = "9"
    v, say = env_of(16.4 * GIB)
    check(v is None and "kept free" not in say, "an MOE_CACHE_RAM_GIB of your own in the environment: untouched")
finally:
    os.environ.pop("MOE_CACHE_RAM_GIB", None)
    if saved is not None:
        os.environ["MOE_CACHE_RAM_GIB"] = saved
print("avail reserve test: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
sys.exit(1 if FAILS else 0)
