#!/usr/bin/env python3
"""LAUNCHER-982 / 983: the dense KV-cache choice, the 'all on GPU' line, the llama-server log scan and the integrated-GPU warning, with fake VRAM figures and fake log lines.
   python3 tests/dense_kv_fit_test.py
PC 2026-11-08: Qwen3.8-27B at --ctx 65536 on a 16 GB card ran 18.9 tok/s against 35 at 24k because the f16 KV cache did not fit beside the model and llama.cpp moved work to the CPU."""
import os, struct, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, os.path.join(HERE, "planner")); sys.path.insert(0, os.path.join(HERE, "bin"))
import launch_plan_test as lp      # noqa: E402
import moe_cache as mc             # noqa: E402
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
    # --- the planner function ---
    ctx = 8192
    a16 = gp.dense_gpu_fit(model, ctx, 64 * GIB, False); a8 = gp.dense_gpu_fit(model, ctx, 64 * GIB, True)
    check(a16 and a16["fits"] and a8["kv_gib"] < a16["kv_gib"], "dense_gpu_fit: plenty of memory -> fits, the 8-bit KV cache is smaller")
    mid = (a16["need_gib"] + a8["need_gib"]) / 2            # f16 does not fit, q8_0 does
    r16 = gp.dense_gpu_fit(model, ctx, mid * GIB, False); r8 = gp.dense_gpu_fit(model, ctx, mid * GIB, True)
    check(not r16["fits"] and r8["fits"] and r16["over_gib"] > 0, "dense_gpu_fit: between the two needs -> f16 does not fit, q8_0 does")
    tiny = gp.dense_gpu_fit(model, ctx, (a8["need_gib"] / 2) * GIB, True)
    check(tiny and not tiny["fits"] and tiny["ctx_fit"] < ctx, "dense_gpu_fit: far too small -> neither fits, a smaller context is named")
    cf = r16["ctx_fit"]
    ok_at = gp.dense_gpu_fit(model, cf, mid * GIB, False) if cf else None
    bad_at = gp.dense_gpu_fit(model, cf + 1024, mid * GIB, False)
    check(cf > 0 and ok_at["fits"] and not bad_at["fits"], "dense_gpu_fit: ctx_fit is the largest multiple of 1024 that fits (%d)" % cf)
    check(gp.dense_gpu_fit(model, ctx, None, False) is None, "dense_gpu_fit: no GPU memory figure -> None")
    # --- the launcher, dry run on a fake machine (--vram gives the free memory) ---
    run = lambda vram, extra_opts=(), extra=(): lp.run_case("linux", ["--ram", "12", "--vram", f"{vram:.3f}", "--ctx", str(ctx)] + list(extra_opts), list(extra), model, tmp)
    out = lambda r: r["stdout"] + "\n" + r.get("stderr", "")
    cmd = lambda r: next((l for l in r["stdout"].splitlines() if "llama-server" in l and " -m " in l), "")
    roomy = run(a16["need_gib"] + 1)
    check("-ctk" not in cmd(roomy) and "all on GPU: yes" in roomy["stdout"], "dry run, room for f16: f16 KV, 'all on GPU: yes'")
    tight = run(mid)
    check("-ctk q8_0" in cmd(tight) and "q8_0 KV cache, so the whole model stays on the GPU" in tight["stdout"] and "all on GPU: yes" in tight["stdout"],
          "dry run, f16 too big but q8_0 fits: q8_0 KV chosen, says why, 'all on GPU: yes'")
    exact = run(mid, ["--exact"])
    check("-ctk" not in cmd(exact) and "all on GPU: NO" in out(exact), "dry run, --exact keeps f16 and says all on GPU: NO")
    user = run(mid, extra=["-ctk", "f16", "-ctv", "f16"])
    check("-ctk q8_0" not in cmd(user) and "all on GPU: NO" in out(user), "dry run, your own -ctk f16 stands, all on GPU: NO")
    n4 = gp.dense_gpu_fit(model, 4096, 64 * GIB, True)
    none = run((a8["need_gib"] + n4["need_gib"]) / 2)          # q8_0 does not fit 8192 but fits 4096
    check("-ctk q8_0" in cmd(none) and "neither KV type fits" in out(none) and "all on GPU: NO" in out(none) and "--ctx" in out(none).split("all on GPU: NO")[1],
          "dry run, neither fits: q8_0 (spills least), says so, names a context that fits")
    # --- the log scan ---
    line = b"llama_context: layer 41 is assigned to device Vulkan1 but Flash Attention over K/V rows is assigned to device CPU\n"
    n = mc.cpu_placement_note(line)
    check(n and n[0] == "attn" and "WARNING" in n[1] and "layer 41" in n[1] and "Vulkan1" in n[1], "log scan: 'Flash Attention over K/V rows is assigned to device CPU' -> loud warning naming the layer")
    n2 = mc.cpu_placement_note("load_tensors: offloaded 63/65 layers to GPU")
    check(n2 and n2[0] == "offload" and "63 of 65" in n2[1], "log scan: partial offload -> warning")
    check(mc.cpu_placement_note("load_tensors: offloaded 65/65 layers to GPU") is None, "log scan: everything offloaded -> nothing")
    check(mc.cpu_placement_note("load_tensors: offloaded 0/65 layers to GPU", cpu_run=True) is None, "log scan: a CPU-only run asked for 0 layers -> nothing")
    check(mc.cpu_placement_note("srv  load_model: loading model 'x'") is None, "log scan: an ordinary line -> nothing")
    # --- the integrated GPU warning ---
    listing = ("Available devices:\n  Vulkan0: AMD Radeon(TM) Graphics (24000 MiB, 23000 MiB free)\n  Vulkan1: AMD Radeon RX 9070 XT (16304 MiB, 15000 MiB free)\n")
    w = mc.igpu_device_note(["--device", "Vulkan0"], listing)
    check(w and "integrated GPU" in w and "Vulkan1" in w, "iGPU: --device Vulkan0 (Radeon Graphics) while Vulkan1 is the 9070 XT -> warning naming the better device")
    check(mc.igpu_device_note(["--device", "Vulkan1"], listing) is None, "iGPU: --device on the discrete card -> nothing")
    check(mc.igpu_device_note([], listing) is None and mc.igpu_device_note(["--device", "none"], listing) is None, "iGPU: no --device / none -> nothing")
if fails:
    print("FAILED: " + "; ".join(fails)); sys.exit(1)
print("all ok")
