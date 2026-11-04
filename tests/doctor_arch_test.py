#!/usr/bin/env python3
"""LAUNCHER-006: the doctor says that llama.cpp does not support a model architecture yet, not only that llama-server exited. No model, no real llama-server.
   python3 tests/doctor_arch_test.py"""
import contextlib, io, os, stat, struct, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "planner"))
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

TMP = tempfile.mkdtemp(prefix="moe-doctor-arch-")

# the log parser
check(mc.arch_in_log(["x", "llama_model_load: error loading model architecture: unknown model architecture: 'clover'"]) == "clover", "log line with quotes: clover")
check(mc.arch_in_log(["unknown model architecture: qwen99moe"]) == "qwen99moe", "log line without quotes")
check(mc.arch_in_log(["load_backend: loaded CPU", "model loaded"]) is None, "no such line: None")
msg = mc.unsupported_arch_message("clover", mc.KNOWN_UNSUPPORTED_ARCH["clover"])
check("does not support the model architecture 'clover' yet" in msg and "31306" in msg and "newer build only helps once" in msg, "the sentence names the architecture and the open upstream pull request")
check("newer llama.cpp" not in msg and "Nothing is wrong with your setup" in msg, "it does not ask for a newer build")

def gguf_with_arch(path, arch):
    def s(x): return struct.pack("<Q", len(x)) + x.encode()
    kv = s("general.architecture") + struct.pack("<I", 8) + s(arch)
    open(path, "wb").write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 1) + kv + b"\0" * 64)

# the doctor on a model of a known unsupported architecture
m = os.path.join(TMP, "clover.gguf"); gguf_with_arch(m, "clover")
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    rc = mc.check([m, "--llama-server", os.path.join(TMP, "none")])
out = buf.getvalue()
check(rc == 1 and any("[FAIL]" in l and "architecture" in l and "does not support the model architecture 'clover'" in l for l in out.splitlines()), "doctor: a clover model gets a FAIL line saying llama.cpp does not support it yet")
m2 = os.path.join(TMP, "ok.gguf"); gguf_with_arch(m2, "qwen3moe")
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    mc.check([m2, "--llama-server", os.path.join(TMP, "none")])
check("does not support the model architecture" not in buf.getvalue(), "doctor: a supported architecture gets no such line")

# run_check with a stand-in llama-server that refuses the model the way llama.cpp does
if os.name != "nt":
    ls = os.path.join(TMP, "fake-llama-server")
    open(ls, "w").write("#!/bin/sh\necho \"llama_model_load: error loading model architecture: unknown model architecture: 'newarch'\" >&2\nexit 1\n")
    os.chmod(ls, os.stat(ls).st_mode | stat.S_IEXEC)
    ok_, text = mc.run_check(ls, m2, [], port=18097)
    check(not ok_ and "does not support the model architecture 'newarch' yet" in text and "exited before it was ready" in text, f"cache run: an unknown architecture in llama-server's output is named ({text[:100]})")
    open(ls, "w").write("#!/bin/sh\necho 'something else broke' >&2\nexit 1\n")
    ok_, text = mc.run_check(ls, m2, [], port=18097)
    check(not ok_ and "does not support" not in text, "cache run: another failure keeps the old wording")

# WINDOWS-061: the llama.cpp version line is plain words (no API numbers, no commit lists)
if os.name != "nt":
    vls = os.path.join(TMP, "ver-llama-server")
    open(vls, "w").write("#!/bin/sh\necho 'version: 12345 (abcdef1)'\nexit 0\n")
    os.chmod(vls, os.stat(vls).st_mode | stat.S_IEXEC)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        mc.check(["--llama-server", vls])
    line = next((l for l in buf.getvalue().splitlines() if "ggml version" in l), "")
    check("this llama.cpp build is not one we tested" in line and "API 2" not in line and "API 3" not in line and "ggml 0.27" not in line, f"doctor: an untested llama.cpp gets a plain sentence ({line[:120]})")
    open(vls, "w").write("#!/bin/sh\necho 'version: 12064 (commit 51fe42c5d)'\nexit 0\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        mc.check(["--llama-server", vls])
    line = next((l for l in buf.getvalue().splitlines() if "ggml version" in l), "")
    check("not one we tested" not in line, f"LAUNCHER-674: doctor: a tested commit (b12064) gets no warning ({line[:100]})")
sys.exit(1 if fails else 0)
