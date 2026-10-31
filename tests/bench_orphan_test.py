#!/usr/bin/env python3
"""bench-report starts `moe_cache.py server` for every run. When the control page cancels "Share my numbers" it sends SIGTERM to bench-report alone
(Python dies without running its finally), so on Linux each server it started must carry the parent-death signal, or the launcher and llama-server stay
running on the benchmark port. Fake process, no server.
   python3 tests/bench_orphan_test.py"""
import io, os, sys, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bin"))
import moe_cache_bench as mb  # noqa: E402

seen = []
class FakeProc:
    def __init__(self, cmd, **kw):
        seen.append(kw)
        self.stdout = io.BytesIO(b"")
        self.pid = 0
    def poll(self): return 1
    def wait(self, timeout=None): return 1
    def terminate(self): pass
    def kill(self): pass
mb.subprocess.Popen = FakeProc
mb.mc.healthy = lambda port, key=None: False
mb.mc.port_busy = lambda port: False
a = types.SimpleNamespace(llama_server="x", ctx=2048, ram=None, extra=[], load_timeout=5, prompt_tokens=100, tokens=8, scenario="chat")
res = mb.run_one("m.gguf", "stock", a, 9)
ok = bool(seen) and (not sys.platform.startswith("linux") or seen[0].get("preexec_fn") == mb.mc._die_with_parent_linux)
print(("  ok   " if ok else "  FAIL ") + "the server bench-report starts dies with it (parent-death signal on Linux)")
sys.exit(0 if ok else 1)
