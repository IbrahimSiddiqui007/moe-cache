#!/usr/bin/env python3
"""MM6 + MM7 prototypes through the real launcher on FAKE machines (no model download: sparse GGUF headers).
   python3 tests/dense_share_test.py
MM6 --auto-fit: a dense model gets no -ngl / --n-cpu-moe (llama.cpp's --fit places the layers); a MoE model and a user's own -ngl are untouched.
MM6 --mmproj: auto-finds mmproj*.gguf next to the model, passes it, counts it in RAM and VRAM; a missing one is an error.
MM7 --share-ram: other llama-servers' RSS shrinks this server's share; --ram stays; nothing running = a line only. Everything off: same command line."""
import copy, os, re, struct, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests"))
sys.path.insert(0, os.path.join(HERE, "planner"))
import launch_plan_test as lp         # noqa: E402
import moe_cache as mc                # noqa: E402

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
    kv = [("general.architecture", 8, string("llama")), ("llama.block_count", 4, struct.pack("<I", n_layers)), ("llama.context_length", 4, struct.pack("<I", 8192)),
          ("llama.embedding_length", 4, struct.pack("<I", d)), ("llama.attention.head_count", 4, struct.pack("<I", 16)), ("llama.attention.head_count_kv", 4, struct.pack("<I", 4))]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(kv))
    for k, t, v in kv:
        head += string(k) + struct.pack("<I", t) + v
    for name, dims, ty, o in infos:
        head += string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", x) for x in dims) + struct.pack("<IQ", ty, o)
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head); f.truncate(len(head) + off)

def cmdline(r):
    m = re.search(r"^.*llama-server\S* -m .*$", r["stdout"], re.M)
    return m.group(0) if m else ""

with tempfile.TemporaryDirectory() as tmp:
    moe = os.path.join(tmp, "moe", "synthetic-moe.gguf"); os.makedirs(os.path.dirname(moe)); lp.make_synthetic_model(moe)
    dense = os.path.join(tmp, "dense", "vision-7b-Q4.gguf"); os.makedirs(os.path.dirname(dense)); make_dense(dense)
    proj = os.path.join(tmp, "dense", "mmproj-vision-7b-f16.gguf"); open(proj, "wb").write(b"GGUF"); os.truncate(proj, 600 * 2**20)
    other = {"pids": []}
    mc.other_llama_servers = lambda: other["pids"]
    run = lambda model, opts, extra=(), mach="linux": lp.run_case(mach, ["--ram", "12"] + opts, list(extra), model, tmp)

    base = run(moe, [])
    check(cmdline(base) == cmdline(run(moe, ["--auto-fit"])), "MoE model with --auto-fit: same command line (nothing to fit by layers)")
    check("-ngl 99" in cmdline(run(dense, [])), "dense, flags off: -ngl 99 as before")
    r = run(dense, ["--auto-fit"])
    check("-ngl" not in cmdline(r) and "--n-cpu-moe" not in cmdline(r) and "PROTOTYPE" in r["stdout"], "dense + --auto-fit: no -ngl, no --n-cpu-moe, says prototype")
    check("-ngl 40" in cmdline(run(dense, ["--auto-fit"], ["-ngl", "40"])), "dense + --auto-fit + user -ngl 40: the user's value stands")

    r = run(dense, ["--mmproj", "auto"])
    check(f"--mmproj {dense.replace('vision-7b-Q4.gguf', 'mmproj-vision-7b-f16.gguf')}".replace(tmp, "<TMP>") in cmdline(r), "--mmproj auto finds the projector next to the model and passes it")
    check("600" in r["stdout"] or "0.63 GB" in r["stdout"], "its size is reported")
    big = run(dense, ["--mmproj", "auto", "--ram", "12"])
    check(big["rc"] == 0, "runs")
    r2 = lp.run_case("linux", ["--ram", "12", "--mmproj", os.path.join(tmp, "nope.gguf")], [], dense, tmp)
    check(r2["rc"] not in (0, None) and "not found" in r2["stderr"], "a missing projector file is an error, not a silent run")
    r3 = lp.run_case("linux", ["--ram", "12", "--mmproj", "auto"], [], moe, tmp)
    check(r3["rc"] not in (0, None), "--mmproj auto with no projector next to the model: an error that says to pass the file")
    # RAM floor: a model that just fits stock without the projector no longer fits with it
    sz = os.path.getsize(dense)
    fit = str(round((sz + mc.STOCK_HEADROOM) / 1e9 + 0.05, 2))
    a1 = lp.run_case("linux", ["--ram", fit], [], dense, tmp); a2 = lp.run_case("linux", ["--ram", fit, "--mmproj", "auto"], [], dense, tmp)
    check("running stock" in a1["stdout"] and "running stock" not in a2["stdout"], "the projector's size is added to the RAM floor (stock without it, plugin decision with it)")

    # MM7
    nothing = run(moe, ["--share-ram"])
    check("no other llama-server" in nothing["stdout"] and cmdline(nothing) == cmdline(base), "--share-ram, nothing else running: one line, same command line")
    other["pids"] = [(111, 10 * 10**9), (222, 6 * 10**9)]
    r = lp.run_case("linux", ["--share-ram", "--always-plugin"], [], moe, tmp)
    check("2 other llama-server(s) hold 16.0 GB" in r["stdout"] and "may use" in r["stdout"], "--share-ram with 16 GB held elsewhere: reported and shrunk")
    m = re.search(r"this server may use ([\d.]+) GB", r["stdout"])
    check(bool(m) and float(m.group(1)) <= 28 * GIB / 1e9 - 16 + 0.1, f"the new share ({m.group(1) if m else '?'} GB) fits under the 28 GiB cap next to them")
    r = run(moe, ["--share-ram"])
    check("as given" in r["stdout"], "--ram stays as the user gave it")
    check(cmdline(run(moe, [])) == cmdline(base), "flag off while others run: untouched")
    other["pids"] = [(1, 30 * 10**9)]
    r = lp.run_case("linux", ["--share-ram"], [], moe, tmp)
    check("only" in r["stderr"] + r["stdout"] and "left" in r["stderr"] + r["stdout"], "others hold nearly the whole cap: a warning, not a silent tiny size")

print("FAILED: " + "; ".join(fails) if fails else "all ok")
sys.exit(1 if fails else 0)
