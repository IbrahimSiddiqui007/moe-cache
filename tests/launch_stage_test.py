#!/usr/bin/env python3
"""ENGINE-029: refusals of the launch plan that no golden case reaches (the goldens are dry runs, which do not need llama-server). No model weights, no server.
   python3 tests/launch_stage_test.py"""
import contextlib, io, os, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, os.path.join(HERE, "planner"))
import moe_cache as mc  # noqa: E402
import launch_plan_test as lpt  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

tmp = tempfile.mkdtemp(prefix="moe-stage-")
model = os.path.join(tmp, "m.gguf"); lpt.make_synthetic_model(model)
saved = mc.WINDOWS, mc.CACHE_DIR, mc.KEY_FILE
mc.CACHE_DIR = os.path.join(tmp, "cache"); mc.KEY_FILE = os.path.join(mc.CACHE_DIR, "api.key")
try:
    for win in (False, True):
        class M(lpt.FakeMachine):
            windows = win
        mc.WINDOWS = win
        a, ex, argv = mc.parse_server([model, "--llama-server", os.path.join(tmp, "missing-llama-server"), "--ram", "12"])
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                mc.launch_plan(a, ex, argv, machine=M(tmp), quiet=True)
            msg = None
        except SystemExit as e:
            msg = str(e.code)
        name = "Windows" if win else "Linux"
        check(msg is not None and "llama-server not found" in msg and "--llama-server PATH" in msg and ("setx LLAMA_SERVER" in msg) == win,
              f"{name}: a launch with no llama-server is refused with the {'setx' if win else 'LLAMA_SERVER'} advice ({(msg or '')[:60]})")
    # the plugin library that does not exist
    mc.WINDOWS = False
    ls = os.path.join(tmp, "llama-server"); open(ls, "w").close()
    class M2(lpt.FakeMachine):
        def lib_exists(self, lib): return False
    a, ex, argv = mc.parse_server([model, "--llama-server", ls, "--ram", "12", "--always-plugin", "--dry-run"])
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            mc.launch_plan(a, ex, argv, machine=M2(tmp), quiet=True)
        msg = None
    except SystemExit as e:
        msg = str(e.code)
    check(msg is not None and "plugin not built" in msg, f"no plugin library: refused with 'plugin not built' ({(msg or '')[:60]})")
finally:
    mc.WINDOWS, mc.CACHE_DIR, mc.KEY_FILE = saved
sys.exit(1 if fails else 0)
