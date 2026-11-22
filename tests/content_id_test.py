#!/usr/bin/env python3
"""LAUNCHER-1251: stored auto-tune picks and load records are keyed by file CONTENT, not modification time: a model or a llama.cpp build copied again
(a kit unpacked anew, a model copied to another drive) keeps its key; a changed byte does not. Model-free."""
import os, shutil, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "planner"), os.path.join(HERE, "..", "bin")]
import load_record as lr        # noqa: E402
import moe_cache as mc          # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok   " if ok else "  FAIL ") + what)
    fails += 0 if ok else 1


with tempfile.TemporaryDirectory() as tmp:
    a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
    os.makedirs(a); os.makedirs(b)
    small = os.urandom(3 * 2**20)
    for n in ("llama-server", "libggml-base.so", "libllama.so"):
        open(os.path.join(a, n), "wb").write(small + n.encode())
    big = os.path.join(a, "m.gguf")
    with open(big, "wb") as f:                     # just over the full-hash limit: the sampled path
        f.write(b"GGUF" + os.urandom(2**20))
        f.truncate(lr.FULL_HASH_MAX + 7 * 2**20)
    time.sleep(0.01)
    for n in os.listdir(a):
        shutil.copyfile(os.path.join(a, n), os.path.join(b, n))      # copyfile: new modification times
    os.utime(os.path.join(b, "llama-server"), (1, 1))
    check(os.stat(os.path.join(a, "llama-server")).st_mtime_ns != os.stat(os.path.join(b, "llama-server")).st_mtime_ns, "the copy has another modification time")
    check(lr.build_id(os.path.join(a, "llama-server")) == lr.build_id(os.path.join(b, "llama-server")) != "", "build_id: the same build copied again keeps its id")
    check(lr.model_id(os.path.join(a, "m.gguf")) == lr.model_id(os.path.join(b, "m.gguf")) != "", "model_id: a big (sampled) model copied again keeps its id")
    check(mc.autotune_model_id(os.path.join(a, "m.gguf")) == mc.autotune_model_id(os.path.join(b, "m.gguf")), "autotune_model_id: copied model, same key")
    with open(os.path.join(b, "libggml-base.so"), "r+b") as f:
        f.seek(2 * 2**20); f.write(b"X")
    check(lr.build_id(os.path.join(a, "llama-server")) != lr.build_id(os.path.join(b, "llama-server")), "build_id: one byte changed in a library (whole-file hash) -> new id")
    with open(os.path.join(b, "m.gguf"), "r+b") as f:
        f.seek(10); f.write(b"Y")
    check(lr.model_id(os.path.join(a, "m.gguf")) != lr.model_id(os.path.join(b, "m.gguf")), "model_id: a changed byte in the metadata (sampled part) -> new id")
    t0 = time.monotonic(); lr.content_id(big); lr.content_id(big)
    check(time.monotonic() - t0 < 2.0, "content_id of a sampled file is quick and memoised")
print("content_id_test: " + ("all passed" if not fails else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
