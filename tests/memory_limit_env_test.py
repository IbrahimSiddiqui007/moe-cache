#!/usr/bin/env python3
"""LAUNCHER-1249: the plugin's whole-server memory cap can never exceed the launcher's own limit, and a user's MOE_CACHE_RAM_GIB reaches every plugin run.
(a) launcher start: memory_env_stage passes the limit it computed (available RAM at the start minus the reserve) as MOE_CACHE_MAX_RAM_GB when that is below the
    plugin's default cap; the user's --max-ram / MOE_CACHE_MAX_RAM_GB / --ram / MOE_CACHE_RAM_GIB win. (b) quality gate: every plugin run gets the cap and the
    memory settings given with --env (candidate-only before: the floor arm lost them). Subprocess environments are mocked; no model, no PC. Same code on Linux and Windows."""
import contextlib, io, os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin")); sys.path.insert(0, os.path.join(HERE, "..", "planner")); sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import moe_cache as mc   # noqa: E402
import quality_gate as qg   # noqa: E402

GIB = 2 ** 30
TOT = 31.1 * GIB
FAILS = []
def check(c, w):
    print(("  ok   " if c else "  FAIL ") + w)
    if not c: FAILS.append(w)

class M(mc.Machine):
    def __init__(self, avail, windows): self.avail, self._w = avail, windows
    @property
    def windows(self): return self._w
    def mem_for_server(self, r): return self.avail
    def mem_available(self): return self.avail
    def mem_total(self): return TOT
    def mem_cap(self): return TOT - 4 * GIB
    def cgroup_limit(self): return None
    def load_env(self, ls, lib, method="auto"): return [("LD_PRELOAD", lib)], "preload"

def stage(args, avail, win):
    a, _, _ = mc.parse_server(["m.gguf", "--no-pin"] + args)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return mc.memory_env_stage(a, "llama-server", "lib.so", "m.gguf", "p", False, "off", M(avail, win))[3]

for k in ("MOE_CACHE_MAX_RAM_GB", "MOE_CACHE_RAM_GIB", "MOE_CACHE_AVAIL_KEEP_GIB"): os.environ.pop(k, None)
# (a) launcher
for win in (True, False):
    e = stage([], 17.4 * GIB, win); n = "Windows" if win else "Linux"
    got = [v for k, v in e if k == "MOE_CACHE_MAX_RAM_GB"]
    check(len(got) == 1 and abs(float(got[0]) - 13.4 * GIB / 1e9) < 0.01, f"{n}: 17.4 GiB available -> the plugin gets MOE_CACHE_MAX_RAM_GB {got} (13.4 GiB = available - 4 GiB reserve)")
    check(float(got[0]) * 1e9 < mc.Machine.mem_cap(M(0, win)) if got else False, f"{n}: below the plugin's own default cap (physical RAM - 12.5 %)")
    check(not any(k == "MOE_CACHE_MAX_RAM_GB" for k, _ in stage([], 31.1 * GIB, win)), f"{n}: limit equal to the default cap -> nothing added, a normal start is unchanged")
    check([v for k, v in stage(["--max-ram", "20"], 17.4 * GIB, win) if k == "MOE_CACHE_MAX_RAM_GB"] == ["20"], f"{n}: --max-ram 20 wins (passed once, as given)")
    check(not any(k == "MOE_CACHE_MAX_RAM_GB" for k, _ in stage(["--ram", "12"], 17.4 * GIB, win)), f"{n}: --ram 12: no launcher limit added (the user's size)")
os.environ["MOE_CACHE_MAX_RAM_GB"] = "22"
check(not any(k == "MOE_CACHE_MAX_RAM_GB" for k, _ in stage([], 17.4 * GIB, True)), "user's MOE_CACHE_MAX_RAM_GB in the environment wins (inherited unchanged)")
os.environ.pop("MOE_CACHE_MAX_RAM_GB")
os.environ["MOE_CACHE_RAM_GIB"] = "12"
check(not any(k in ("MOE_CACHE_MAX_RAM_GB", "MOE_CACHE_RAM_GIB") for k, _ in stage([], 17.4 * GIB, True)), "user's MOE_CACHE_RAM_GIB=12: launcher sets neither (the plugin reads the user's value from the inherited environment)")
os.environ.pop("MOE_CACHE_RAM_GIB")
D = mc.ram_cap_decision(17.4 * GIB, TOT, windows=True)
check(mc.avail_limit_gb(None, 1e9) is None and mc.avail_limit_gb(D, None) and abs(mc.avail_limit_gb(D, None) - 13.4 * GIB / 1e9) < 0.01,
      "avail_limit_gb: no decision / no cap")
check(abs(mc.avail_limit_gb({"avail": 3 * GIB, "keep": 4 * GIB, "gib": 0.0}, None) - 4 * GIB / 1e9) < 1e-6, "avail_limit_gb: nothing left -> the 4 GiB floor, not None")
check(abs(mc.avail_limit_gb({"avail": 8 * GIB, "keep": 4 * GIB, "gib": 6.5}, None) - 6.5 * GIB / 1e9) < 1e-6, "avail_limit_gb: under the announced cache size -> that size (6.5 GiB)")
# a realistic busy 31.1 GiB machine: the limit applies and the floor holds (limit never under max(4 GiB, MOE_CACHE_RAM_GIB))
for win in (True, False):
    n = "Windows" if win else "Linux"
    for av in (17.4, 11.0, 7.5, 5.0):
        e = stage([], av * GIB, win); ed = dict(e)
        got = ed.get("MOE_CACHE_MAX_RAM_GB"); rg = float(ed.get("MOE_CACHE_RAM_GIB", 0))
        check(got is not None and float(got) * 1e9 >= max(4.0, rg) * GIB - 1e6 and float(got) * 1e9 < M(0, win).mem_cap(),
              f"{n}: {av} GiB available -> MOE_CACHE_MAX_RAM_GB {got} GB >= max(4 GiB, cache {rg:.2f} GiB) and below the default cap")
# ONE consistent cap line: the whole-server cap shown is the limit in force
for win in (True, False):
    a, _, _ = mc.parse_server(["m.gguf", "--no-pin"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = mc.memory_env_stage(a, "llama-server", "lib.so", "m.gguf", "p", False, "off", M(17.4 * GIB, win))
    txt = buf.getvalue(); lim = float(dict(r[3])["MOE_CACHE_MAX_RAM_GB"])
    capln = [l for l in txt.splitlines() if "for the whole server" in l]
    check(len(capln) == 1 and f"{lim:.1f} GB" in capln[0] and "plugin's own cap is higher" not in txt and "whole-server memory limit" not in txt and abs(r[4] - lim * 1e9) < 1e6,
          f"one cap line showing the limit in force ({capln[:1]})")

# (b) quality gate: which environment each perplexity / server run gets
class R:
    returncode = 0
    stdout = "moe-cache: cache mode\nMean KLD: 0.001 ± 0.0001\nSame top p: 99.0 ± 0.1 %\nMaximum KLD: 0.1\nMean PPL(Q): 5.0\nMean PPL(base): 5.0\nMean PPL(Q)/PPL(base): 1.0\n"
def gate(extra, env_in=None, limit=20.0):
    runs, out = [], io.StringIO()
    def fake(cmd, env=None, **kw):
        runs.append((cmd, dict(env or {}))); return R()
    saved = (subprocess_run := qg.subprocess.run, qg.machine_limit_gb, sys.argv)
    qg.subprocess.run, qg.machine_limit_gb = fake, (lambda *_: limit)
    old_env = dict(os.environ); os.environ.update(env_in or {})
    sys.argv = ["quality_gate.py", "--llama-server", "srv", "--perplexity", "ppl", "--model", "m.gguf", "--lib", "/x/libggml-moe-cache.so", "--level", "kv",
                "--no-tasks", "--text", os.devnull, "--out", os.path.join(os.environ.get("TMPDIR", "/tmp"), "qg1249")] + extra
    try:
        with contextlib.redirect_stdout(out):
            try: qg.main()
            except SystemExit: pass
    finally:
        qg.subprocess.run, qg.machine_limit_gb, sys.argv = saved
        os.environ.clear(); os.environ.update(old_env)
    return runs
for k in ("MOE_CACHE_MAX_RAM_GB", "MOE_CACHE_RAM_GIB"): os.environ.pop(k, None)
import tempfile
os.makedirs(os.path.join(os.environ.get("TMPDIR", "/tmp"), "qg1249"), exist_ok=True)
for f in ("base.kld",):
    open(os.path.join(os.environ.get("TMPDIR", "/tmp"), "qg1249", f), "w").close()
runs = gate(["--env", "MOE_CACHE_RAM_GIB=12", "--cache-dir", os.path.join(os.environ.get("TMPDIR", "/tmp"), "qg1249")])
plug_runs = [e for c, e in runs if "MOE_CACHE_GGUF" in e]
check(len(plug_runs) == 2, f"gate: candidate and floor arm both run the plugin ({len(plug_runs)} plugin runs)")
check(all(e.get("MOE_CACHE_RAM_GIB") == "12" for e in plug_runs), "--env MOE_CACHE_RAM_GIB=12 reaches the candidate AND the floor arm")
check(all(e.get("MOE_CACHE_MAX_RAM_GB") == "20.000" for e in plug_runs), "the launcher's machine limit reaches every plugin run as MOE_CACHE_MAX_RAM_GB")
check(all("MOE_CACHE_GGUF" not in e for c, e in runs if e not in plug_runs), "the stock run gets no plugin environment")
runs = gate(["--ram-env", "MOE_CACHE_MAX_RAM_GB=15", "--cache-dir", os.path.join(os.environ.get("TMPDIR", "/tmp"), "qg1249")])
check(all(e.get("MOE_CACHE_MAX_RAM_GB") == "15" for c, e in runs if "MOE_CACHE_GGUF" in e), "--ram-env MOE_CACHE_MAX_RAM_GB=15 wins over the machine limit")
runs = gate(["--cache-dir", os.path.join(os.environ.get("TMPDIR", "/tmp"), "qg1249")], env_in={"MOE_CACHE_MAX_RAM_GB": "18", "MOE_CACHE_RAM_GIB": "9"})
check(all(e.get("MOE_CACHE_MAX_RAM_GB") == "18" and e.get("MOE_CACHE_RAM_GIB") == "9" for c, e in runs if "MOE_CACHE_GGUF" in e), "the user's own process environment reaches every plugin run unchanged")
runs = gate(["--cache-dir", os.path.join(os.environ.get("TMPDIR", "/tmp"), "qg1249")], limit=None)
check(all("MOE_CACHE_MAX_RAM_GB" not in e for c, e in runs if "MOE_CACHE_GGUF" in e), "no known machine limit: nothing set (the plugin keeps its default cap)")
import json
_gj = lambda: json.load(open(os.path.join(os.environ.get("TMPDIR", "/tmp"), "qg1249", "gate.json")))
runs = gate(["--cache-dir", os.path.join(os.environ.get("TMPDIR", "/tmp"), "qg1249")])
check(_gj().get("max_ram_gb") == "20.000", f"gate.json records the MOE_CACHE_MAX_RAM_GB the plugin runs had ({_gj().get('max_ram_gb')})")
runs = gate(["--ram-env", "MOE_CACHE_MAX_RAM_GB=15", "--cache-dir", os.path.join(os.environ.get("TMPDIR", "/tmp"), "qg1249")])
check(_gj().get("max_ram_gb") == "15", "gate.json records a --ram-env cap too")
# the 4 GiB floor in machine_limit_gb itself (a tiny available figure): never under 4 GiB
_m = types.SimpleNamespace(mem_total=lambda: TOT, _mem_for_server=lambda r: 5 * GIB, mem_cap=lambda: TOT - 4 * GIB, ram_keep_bytes=lambda t: 4 * GIB)
_old = qg.sys.modules.get("moe_cache"); qg.sys.modules["moe_cache"] = _m
try:
    check(abs(qg.machine_limit_gb() - 4 * GIB / 1e9) < 1e-6 and abs(qg.machine_limit_gb(6.0) - 6 * GIB / 1e9) < 1e-6, "machine_limit_gb: 5 GiB available -> the 4 GiB floor; a 6 GiB announced cache -> 6 GiB")
finally:
    qg.sys.modules["moe_cache"] = _old
lim = qg.machine_limit_gb()
check(lim is None or lim > 0, f"machine_limit_gb() on this machine: {lim and round(lim, 1)} GB")
print("PASS (0 failed)" if not FAILS else f"FAIL ({len(FAILS)} failed)")
sys.exit(1 if FAILS else 0)
