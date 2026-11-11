#!/usr/bin/env python3
"""LAUNCHER-1068: a dense model that fits fully on the GPU loads with `--load-mode none` on Windows (PC 2026-11-10: default mmap load spiked used RAM ~12 GB for Qwen3.8-27B,
`--load-mode none` added 2-6 GB, ready in < 1 min). Linux, spilling layouts, --exact and the user's own load flags keep mmap.  python3 tests/loadmode_fit_test.py"""
import os, struct, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, os.path.join(HERE, "planner")); sys.path.insert(0, os.path.join(HERE, "bin"))
import launch_plan_test as lp      # noqa: E402
import gguf_plan as gp             # noqa: E402

GIB = 2**30
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def make_dense(path, n_layers=24, d=2048):
    def string(x):
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    infos, off = [], 0
    def add(name, dims, ty, nbytes):
        nonlocal off
        infos.append((name, dims, ty, off)); off += (nbytes + 31) // 32 * 32
    add("token_embd.weight", [d, 32000], 1, d * 32000 * 2)
    for l in range(n_layers):
        add(f"blk.{l}.attn_q.weight", [d, d], 12, d // 256 * 144 * d)
        add(f"blk.{l}.ffn_up.weight", [d, 4 * d], 12, d // 256 * 144 * 4 * d)
    kv = [("general.architecture", 8, string("llama")), ("llama.block_count", 4, struct.pack("<I", n_layers)), ("llama.context_length", 4, struct.pack("<I", 131072)),
          ("llama.embedding_length", 4, struct.pack("<I", d)), ("llama.attention.head_count", 4, struct.pack("<I", 16)), ("llama.attention.head_count_kv", 4, struct.pack("<I", 4))]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(kv))
    for k, t, v in kv:
        head += string(k) + struct.pack("<I", t) + v
    for name, dims, ty, o in infos:
        head += string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", x) for x in dims) + struct.pack("<IQ", ty, o)
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head); f.truncate(len(head) + off)

with tempfile.TemporaryDirectory() as tmp:
    model = os.path.join(tmp, "dense-7b-Q4.gguf"); make_dense(model)
    ctx = 8192
    need = lambda q: gp.dense_gpu_fit(model, ctx, 64 * GIB, q, 0.0, 0.0, 0.20)["need_gib"]
    f16, q8 = need(False), need(True)
    def run(machine, vram, opts=(), extra=()):
        r = lp.run_case(machine, ["--ram", "12", "--vram", f"{vram:.3f}", "--ctx", str(ctx)] + list(opts), list(extra), model, tmp)
        c = next((l for l in r["stdout"].splitlines() if "llama-server" in l and " -m " in l), "")
        return c, r["stdout"] + "\n" + r.get("stderr", "")
    lm = lambda c: c.split("--load-mode ")[1].split()[0] if "--load-mode " in c else None
    c, o = run("win", f16 + 1)
    check(lm(c) == "none" and "fits fully on the GPU" in o, "Windows, f16 fits: --load-mode none, says why")
    c, o = run("win", (f16 + q8) / 2)
    check("-ctk q8_0" in c and lm(c) == "none", "Windows, q8_0 KV fits: --load-mode none")
    # LAUNCHER-1063 (critic): the KV ladder's smaller types make a model fit that q8_0 alone would spill; then it fits fully and loads with --load-mode none
    mx, q4 = need(("q8_0", "q4_0")), need("q4_0")
    c, o = run("win", (q8 + mx) / 2)
    check("-ctk q8_0 -ctv q4_0" in c and lm(c) == "none" and "fits fully on the GPU" in o, "Windows, only K q8_0 / V q4_0 fits: that KV, and --load-mode none (it was mmap: judged with the q8_0 fit)")
    c, o = run("win", (mx + q4) / 2)
    check("-ctk q4_0 -ctv q4_0" in c and lm(c) == "none", "Windows, only q4_0/q4_0 fits: that KV, and --load-mode none")
    c, o = run("win", q4 / 2)
    check("-ctk q4_0 -ctv q4_0" in c and lm(c) == "mmap", "Windows, not even q4_0/q4_0 fits (layers spill): mmap stays")
    c, o = run("win", q8 / 2)
    check(lm(c) == "mmap", "Windows, layers spill to RAM: mmap stays")
    c, o = run("linux", f16 + 1)
    check(lm(c) == "mmap", "Linux, fits: mmap stays")
    c, o = run("win", f16 + 1, ["--exact"])
    check(lm(c) == "mmap", "Windows, --exact: mmap stays")
    c, o = run("win", f16 + 1, extra=["--load-mode", "mmap"])
    check("--load-mode mmap" in c and "none" not in c.split("--load-mode")[0].split()[-1:], "Windows, your own --load-mode stands")
    c, o = run("win", f16 + 1, extra=["--no-mmap"])
    check(lm(c) == "mmap" and "--no-mmap" in c, "Windows, your own --no-mmap stands (not overridden)")
if fails:
    print("FAILED: " + "; ".join(fails)); sys.exit(1)
print("all ok")
