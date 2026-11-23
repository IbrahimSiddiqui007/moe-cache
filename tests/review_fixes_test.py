#!/usr/bin/env python3
"""Fixes from the outside review of 2026-11-18 (handover/sessions/main4_2026-11-17/files/glm_review/REVIEW.md): findings 1-2 (`--flag=value` after `--`), 8 (KV undo),
21 (--max-ram edge cases), 22 (--port range), 23 (--override-tensor). No model, no llama-server.   python3 tests/review_fixes_test.py"""
import argparse, os, sys, types
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# findings 1-2: one reading of the flags after `--`
sp = mc.split_equals_flags
check(sp(["--host=0.0.0.0", "--cache-ram=512", "-ngl=1", "--alias", "a=b", "-ot", "x=CPU"]) == ["--host", "0.0.0.0", "--cache-ram", "512", "-ngl=1", "--alias", "a=b", "-ot", "x=CPU"],
      "split_equals_flags: long flags split, short flags and plain values (with =) untouched")
check(sp(["--chat-template-kwargs={\"a\":1}", "--override-kv=x=int:3"]) == ["--chat-template-kwargs", "{\"a\":1}", "--override-kv", "x=int:3"], "split_equals_flags: only the first = splits; JSON values survive")
check(sp(sp(["--host=1"])) == ["--host", "1"] and sp([]) == [], "split_equals_flags: idempotent, empty ok")
ex = sp(["--cache-ram=512", "--api-key=KEY", "-ngl", "0", "--n-gpu-layers=7"])
check(mc.arg_after(ex, "--cache-ram", "-cram") == "512" and mc.arg_after(ex, "--n-gpu-layers") == "7", "arg_after sees the equals form after the split (cache-ram, n-gpu-layers)")
check(mc.take_api_key_from_args(ex) == "KEY" and "KEY" not in ex and "--api-key" not in ex, "take_api_key_from_args: the key leaves the argument list")
a, extra, argv = mc.parse_server(["m.gguf", "--", "--host=0.0.0.0", "-c", "4096"])
check(extra == ["--host", "0.0.0.0", "-c", "4096"], "parse_server hands the flags after -- over in the split form")

# finding 21: --max-ram
p = lambda *v: mc.parse_server(["m.gguf", *v])[0].max_ram
check(p("--max-ram", "0") == "0" and p("--max-ram", "0.0") == "0" and p("--max-ram", "0.00") == "0" and p("--max-ram", "20") == "20", "--max-ram: 0, 0.0, 0.00 all mean no cap")
for bad in ("inf", "nan", "-1"):
    try:
        p("--max-ram", bad); check(False, f"--max-ram {bad} refused")
    except SystemExit:
        check(True, f"--max-ram {bad} refused")
mc._TL.max_ram = "inf"
os.environ.pop("MOE_CACHE_MAX_RAM_GB", None)
os.environ.pop("LLAMA_ARG_HOST", None)       # SECURITY-1231: the launcher reads it now
mc.mem_cap.warned = True
m_inf = mc.mem_cap()
mc._TL.max_ram = None
check(m_inf is None or m_inf != float("inf"), "mem_cap: inf never becomes an infinite cap (default cap used)")

# finding 22: --port
for ok in ("1", "8080", "65535"):
    check(mc.parse_server(["m.gguf", "--port", ok])[0].port == int(ok), f"--port {ok} accepted")
for bad in ("0", "70000", "-5", "http"):
    try:
        mc.parse_server(["m.gguf", "--port", bad]); check(False, f"--port {bad} refused")
    except SystemExit:
        check(True, f"--port {bad} refused")

# finding 23: --override-tensor is a custom RAM layout like -ot
class M:
    windows = False
    def mem_for_server(self, r): return 64e9
    def mem_total(self): return 64e9
    def mem_cap(self): return None
a_ = types.SimpleNamespace(exact=False, stock=False, ram=None)
import tempfile
mp = os.path.join(tempfile.mkdtemp(prefix="moe-rf-"), "m.gguf"); open(mp, "wb").write(b"x")
check(mc.load_mode_cache_none(a_, ["ls", "-ot", "x=CPU"], [], M(), mp, 0) is None and mc.load_mode_cache_none(a_, ["ls", "--override-tensor", "x=CPU"], [], M(), mp, 0) is None,
      "load_mode_cache_none: -ot and --override-tensor both keep the user's layout (None)")

# finding 8: the long-prompt fallback removes the KV pair the launcher added, whatever its types
def lp(types_):
    a = types.SimpleNamespace(long_prompt=True, lp_explicit=False, lp_kv=True, ub="auto", approx_kv=False, lp_with_fast=False, fast="kv", dense_kv_types=types_)
    extra = ["-t", "8", "-ctk", types_[0] if types_ else "q8_0", "-ctv", types_[1] if types_ else "q8_0"]
    old = mc.tune_launch
    mc.tune_launch = lambda *args, **kw: ({"ub": 2048}, {}, None)
    try:
        mc.long_prompt_stage(a, extra, "ls", "m", 65536, None, None, None, True, 0, {"ub": 2048}, {}, None)
    finally:
        mc.tune_launch = old
    return extra
check(lp(None) == ["-t", "8"], "long-prompt fallback: q8_0/q8_0 pair undone")
check(lp(("q8_0", "q4_0")) == ["-t", "8"], "long-prompt fallback: a dense K q8_0 / V q4_0 pair is undone too (finding 8)")

import moe_cache_launch as mcl  # noqa: E402
# launch --port gets the same range check as server --port
for ok in ("1", "8080", "65535"):
    check(mcl.parse(["--port", ok]).port == int(ok), f"launch --port {ok} accepted")
for bad in ("0", "70000", "-5", "http"):
    try:
        mcl.parse(["--port", bad]); check(False, f"launch --port {bad} refused")
    except SystemExit:
        check(True, f"launch --port {bad} refused")

import moe_cache_on_demand as od  # noqa: E402
check(od.model_alias("/m/big.gguf", ["x", "--", "--alias=foo"]) == "foo" and od.model_alias("/m/big.gguf", ["x", "--", "--alias", "foo"]) == "foo", "on-demand routes by --alias=foo like --alias foo")

# --- further fixes ---
import time  # noqa: E402
import moe_cache_bench as _bench  # noqa: E402
import moe_cache_ui as ui  # noqa: E402

# a server that dies right after /health answers is reported, not waved through
class _DeadP:
    def poll(self): return 0
old_healthy = mc.healthy
mc.healthy = lambda port: True
try:
    err = _bench._wait_ready(_DeadP(), ["llama-server listening"], 59999, time.time(), 30)
finally:
    mc.healthy = old_healthy
check(bool(err) and "stopped" in err, f"a server that dies right after /health is an error, not a green light ({err!r})")

# stop() on a lock that stays held raises, so the HTTP caller answers 503 and the swap can restart
class _BusyLock:
    def acquire(self, timeout=None, blocking=True): return False
    def release(self): pass
b = od.Backend.__new__(od.Backend)
b.lock, b.state, b.p, b.stop_asked = _BusyLock(), "stopping", None, False
b.log = lambda m: None
try:
    b.stop("test")
    check(False, "stop() returns while the lock is held — the raise is missing")
except RuntimeError:
    check(True, "stop() raises when the lock stays held past 30 s (the HTTP caller answers 503; the swap does not hang)")

# ttl=0 re-spawns --list-devices — an OOM-retry site must not read the plan-time memo
_calls = []
class _FakeOut:
    stdout = "Vulkan0: Foo (8192 MiB, 4096 MiB free)\n"
    stderr = ""
_old_run = mc.subprocess.run
mc.subprocess.run = lambda cmd, **kw: _calls.append(cmd) or _FakeOut()
try:
    d1, d2 = mc.list_devices("ls-fake"), mc.list_devices("ls-fake")
    check(len(_calls) == 1 and d1 == d2, "list_devices memo: the second call inside the ttl does not spawn again")
    d3 = mc.list_devices("ls-fake", ttl=0)
    check(len(_calls) == 2 and d3 == d1, "list_devices ttl=0: the OOM-retry sites re-spawn --list-devices instead of reading the stale memo")
finally:
    mc.subprocess.run = _old_run

# the chat-lock dict does not grow past its cap
for i in range(ui._CHAT_LOCKS_MAX + 50):
    ui._chat_lock(f"/chat/{i}.json")
check(len(ui._CHAT_LOCKS) <= ui._CHAT_LOCKS_MAX, f"the chat-lock dict is capped ({len(ui._CHAT_LOCKS)} <= {ui._CHAT_LOCKS_MAX}); a deleted chat does not leak a lock")

# a failed stock probe leaves stock_line empty — the probe's error text must not become the record's stock plan
tmp4 = tempfile.mkdtemp(prefix="moe-rf3-")
mdl = os.path.join(tmp4, "m.gguf"); open(mdl, "wb").write(b"x")
_captured = {}
_orig_br = _bench.build_record
def _cap_build_record(*a, **k):
    _captured["stock_line"] = a[7]
    return _orig_br(*a, **k)
_saved4 = (_bench.launcher_plan, _bench.doctor_facts, _bench.server_info, _bench.order, mc.port_busy, _bench.rec.conditions_now, _bench.build_record)
_bench.launcher_plan = lambda *a, **k: ("plugin", "llama-server -m m -ngl 99") if "--stock" not in (a[4] or []) else (None, "the launch probe failed: BoomError: 1")
_bench.doctor_facts = lambda *a, **k: (False, {}, [])
_bench.server_info = lambda *a, **k: ({}, {})
_bench.order = lambda n: []
mc.port_busy = lambda port: False
_bench.rec.conditions_now = lambda **k: {"cold": "warm"}
_bench.build_record = _cap_build_record
try:
    rc = _bench.bench_report([mdl, "--llama-server", "/bin/true", "--out", tmp4])
finally:
    _bench.launcher_plan, _bench.doctor_facts, _bench.server_info, _bench.order, mc.port_busy, _bench.rec.conditions_now, _bench.build_record = _saved4
check(rc == 0 and _captured.get("stock_line") == "", f"a failed stock probe gives stock_line \"\" in the record, not the error text (rc={rc}, stock_line={_captured.get('stock_line')!r})")

# Pool.stop: one backend that cannot stop does not leave the others running
import threading  # noqa: E402
import moe_cache_on_demand as _od  # noqa: E402
_stopped = []
class _FakeBackend:
    def __init__(self, name, fail=False): self.name, self.fail, self.key = name, fail, "k"
    def stop(self, reason):
        if self.fail:
            raise RuntimeError(f"could not stop ({reason})")
        _stopped.append(self.name)
_pool = _od.Pool([_FakeBackend("a", fail=True), _FakeBackend("b")])
try:
    _pool.stop("test")
    _raised = None
except RuntimeError as e:
    _raised = str(e)
check(_stopped == ["b"] and _raised and "could not stop" in _raised, f"Pool.stop: the second backend is stopped although the first failed, and the failure is still reported ({_stopped}, {_raised!r})")
check(_od.STOP_WORST_S >= 65, f"Backend.stop waits at least a normal stop's worst case for the server lock ({_od.STOP_WORST_S} s)")

# the page's chat locks: eviction never drops a lock that is held
import moe_cache_ui as _ui  # noqa: E402
_ui._CHAT_LOCKS.clear()
_held = _ui._chat_lock("held")
_held.acquire()
for _i in range(_ui._CHAT_LOCKS_MAX + 10):
    _ui._chat_lock(f"c{_i}")
check(_ui._CHAT_LOCKS.get("held") is _held and _held.locked(), "chat locks: a held lock survives eviction")
check(len(_ui._CHAT_LOCKS) <= _ui._CHAT_LOCKS_MAX, f"chat locks: the table stays at its cap while locks are free ({len(_ui._CHAT_LOCKS)})")
_held.release()
_ui._CHAT_LOCKS.clear()

print("PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
