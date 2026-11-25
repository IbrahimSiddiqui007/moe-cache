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
check(mc.avail_limit_gb(None, 1e9) is None and mc.avail_limit_gb(D, None) and abs(mc.avail_limit_gb(D, None) - 13.4 * GIB / 1e9) < 0.01 and mc.avail_limit_gb({"avail": 3 * GIB, "keep": 4 * GIB}, None) is None,
      "avail_limit_gb: no decision / no cap / nothing left")

# (b) quality gate: which environment each perplexity / server run gets
class R:
    returncode = 0
    stdout = "moe-cache: cache mode\nMean KLD: 0.001 ± 0.0001\nSame top p: 99.0 ± 0.1 %\nMaximum KLD: 0.1\nMean PPL(Q): 5.0\nMean PPL(base): 5.0\nMean PPL(Q)/PPL(base): 1.0\n"
def gate(extra, env_in=None, limit=20.0):
    runs, out = [], io.StringIO()
    def fake(cmd, env=None, **kw):
        runs.append((cmd, dict(env or {}))); return R()
    saved = (subprocess_run := qg.subprocess.run, qg.machine_limit_gb, sys.argv)
    qg.subprocess.run, qg.machine_limit_gb = fake, (lambda: limit)
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
lim = qg.machine_limit_gb()
check(lim is None or lim > 0, f"machine_limit_gb() on this machine: {lim and round(lim, 1)} GB")
print("PASS (0 failed)" if not FAILS else f"FAIL ({len(FAILS)} failed)")
sys.exit(1 if FAILS else 0)
