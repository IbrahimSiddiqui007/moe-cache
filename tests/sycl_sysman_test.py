#!/usr/bin/env python3
"""ENGINE-048: the launcher sets ZES_ENABLE_SYSMAN=1 for llama-server when it sees a SYCL device, and leaves a value the user set alone.
No model (a synthetic header), no llama-server, nothing started.   python3 tests/sycl_sysman_test.py"""
import contextlib, io, os, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "planner")); sys.path.insert(0, os.path.join(HERE, "tests"))
import moe_cache as mc  # noqa: E402
import launch_plan_test as lpt  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

SYS = [("ZES_ENABLE_SYSMAN", "1")]
os.environ.pop("ZES_ENABLE_SYSMAN", None)
with tempfile.TemporaryDirectory() as tmp:
    ls = os.path.join(tmp, "llama-server")
    open(ls, "w").close()
    check(mc.sycl_sysman_env(ls) == [], "no SYCL listing and no SYCL library: nothing set")
    check(mc.sycl_sysman_env(ls, "Available devices:\n  CUDA0: GPU (8188 MiB, 7600 MiB free)\n") == [], "a CUDA-only listing: nothing set")
    check(mc.sycl_sysman_env(ls, "Available devices:\n  SYCL0: Intel(R) Arc(TM) A770 (16225 MiB, 15000 MiB free)\n") == SYS, "a SYCL line in --list-devices: set")
    open(os.path.join(tmp, "libggml-sycl.so"), "w").close()
    check(mc.sycl_sysman_env(ls) == SYS, "libggml-sycl.so next to llama-server: set")
    os.remove(os.path.join(tmp, "libggml-sycl.so")); open(os.path.join(tmp, "ggml-sycl.dll"), "w").close()
    check(mc.sycl_sysman_env(ls) == SYS, "ggml-sycl.dll next to llama-server: set")
    check(mc.sycl_sysman_env(os.path.join(tmp, "missing", "llama-server")) == [], "a folder that does not exist: nothing set, no error")
    os.environ["ZES_ENABLE_SYSMAN"] = "0"
    check(mc.sycl_sysman_env(ls) == [], "the user already set ZES_ENABLE_SYSMAN: left alone")
    os.environ.pop("ZES_ENABLE_SYSMAN")

    # the whole plan: with the library present the plan's environment carries it, without it does not
    model = os.path.join(tmp, "synthetic-moe.gguf")
    lpt.make_synthetic_model(model)
    open(os.path.join(tmp, "libggml-moe-cache.so"), "w").close()
    saved = (mc.CACHE_DIR, mc.KEY_FILE, mc.default_llama_server, mc.WINDOWS)
    mc.CACHE_DIR = os.path.join(tmp, "cache"); mc.KEY_FILE = os.path.join(mc.CACHE_DIR, "api.key"); mc.WINDOWS = False
    mc.default_llama_server = lambda refresh=False: ls
    def plan_env():
        a, extra, argv = mc.parse_server([model, "--llama-server", ls, "--ram", "12", "--no-pin", "--always-plugin", "--plan-json"])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return mc.launch_plan(a, extra, argv, machine=lpt.FakeMachine(tmp), quiet=True).env
    try:
        os.remove(os.path.join(tmp, "ggml-sycl.dll"))
        check(("ZES_ENABLE_SYSMAN", "1") not in plan_env(), "plan without SYCL: no ZES_ENABLE_SYSMAN")
        open(os.path.join(tmp, "libggml-sycl.so"), "w").close()
        env = plan_env()
        check(env.count(("ZES_ENABLE_SYSMAN", "1")) == 1, "plan with the SYCL library: ZES_ENABLE_SYSMAN=1 once in the server's environment")
    finally:
        mc.CACHE_DIR, mc.KEY_FILE, mc.default_llama_server, mc.WINDOWS = saved
sys.exit(1 if fails else 0)
