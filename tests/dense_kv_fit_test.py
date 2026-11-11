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
    # --- the launcher, dry run on a fake machine (--vram gives the free memory; with it compute + context 0.20 GiB and no margin, the PC calibration) ---
    run = lambda vram, extra_opts=(), extra=(): lp.run_case("linux", ["--ram", "12", "--vram", f"{vram:.3f}", "--ctx", str(ctx)] + list(extra_opts), list(extra), model, tmp)
    out = lambda r: r["stdout"] + "\n" + r.get("stderr", "")
    cmd = lambda r: next((l for l in r["stdout"].splitlines() if "llama-server" in l and " -m " in l), "")
    pc = lambda c, q: gp.dense_gpu_fit(model, ctx, 64 * GIB, q, 0.0, 0.0, 0.20)["need_gib"]
    f16_need, q8_need = pc(ctx, False), pc(ctx, True)
    roomy = run(f16_need + 1)
    check("-ctk" not in cmd(roomy) and "KV cache does not fit" not in out(roomy), "dry run, room for f16: f16 KV, nothing said")
    tight = run((f16_need + q8_need) / 2)
    check("-ctk q8_0" in cmd(tight) and "q8_0 KV cache, which does" in out(tight) and "same context" in out(tight) and "-c 8192" in cmd(tight), "dry run, f16 too big but q8_0 fits: q8_0 KV chosen, says why, the context stays")
    exact = run((f16_need + q8_need) / 2, ["--exact"])
    check("-ctk" not in cmd(exact), "dry run, --exact keeps f16")
    user = run((f16_need + q8_need) / 2, extra=["-ctk", "f16", "-ctv", "f16"])
    check("-ctk q8_0" not in cmd(user), "dry run, your own -ctk f16 stands")
    # --- LAUNCHER-1063: f16 -> q8_0/q8_0 -> K q8_0 / V q4_0 -> q4_0/q4_0 -> only then layers to RAM; each step only when the one before would spill; the context never shortens ---
    mx_need, q4_need = pc(ctx, ("q8_0", "q4_0")), pc(ctx, "q4_0")
    check(q4_need < mx_need < q8_need < f16_need, f"planner: q4_0/q4_0 < K q8_0 / V q4_0 < q8_0/q8_0 < f16 (needs {q4_need:.2f} < {mx_need:.2f} < {q8_need:.2f} < {f16_need:.2f} GiB)")
    kv16, kv8, kvx, kv4 = (gp.kv_bytes(model, ctx, k) for k in (False, True, ("q8_0", "q4_0"), "q4_0"))
    check(abs(kv8 / kv16 - 34 / 64) < 0.02 and abs(kv4 / kv16 - 18 / 64) < 0.02 and abs(kvx / kv16 - 52 / 128) < 0.02, "planner: bytes per value q8_0 34/32, q4_0 18/32, f16 2; the mix is K q8_0 + V q4_0 (ratios on the KV part)")
    check(gp.kv_bytes(model, ctx, "q8_0") == kv8 and gp.kv_bytes(model, ctx, "f16") == kv16 and gp.kv_bytes(model, ctx, ("q4_0", "q4_0")) == kv4, "planner: the type can also be given by name or as a pair")
    check(gp.kv_bytes(model, ctx, ("q4_0", "q8_0")) == gp.kv_bytes(model, ctx, ("q8_0", "q4_0")), "planner: K q4_0 / V q8_0 costs the same bytes as K q8_0 / V q4_0 (this model has equal K and V lengths)")
    between8 = run((q8_need + mx_need) / 2)
    check("-ctk q8_0 -ctv q4_0" in cmd(between8) and "-c 8192" in cmd(between8), "dry run, q8_0/q8_0 would spill but K q8_0 / V q4_0 fits: that mix, the context stays")
    t8 = out(between8)
    check("KV K q8_0 / V q4_0: q8_0 / q8_0 would spill by" in t8 and "this step saves" in t8 and "does not spill" in t8 and "94.6 %" in t8 and "measured on Qwen3.8-27B" in t8 and "--exact keeps f16" in t8, "...and the plan line names the step, the spill and the GiB it saves")
    import re as _re
    sv = float(_re.search(r"this step saves ([0-9.]+) GiB", t8).group(1))
    check(abs(sv - (kv8 - kvx) / GIB) < 0.06, f"...the GiB saved is the KV difference to q8_0/q8_0 ({sv:.1f} vs {(kv8 - kvx) / GIB:.2f})")
    between4 = run((mx_need + q4_need) / 2)
    check("-ctk q4_0 -ctv q4_0" in cmd(between4) and "-ctk q8_0 -ctv q4_0" not in cmd(between4) and "-c 8192" in cmd(between4), "dry run, the mix would spill too but q4_0/q4_0 fits: q4_0/q4_0, the context stays")
    t4 = out(between4)
    check("KV q4_0 / q4_0: q8_0 / q8_0 would spill by" in t4 and "does not spill" in t4 and "92.9 %" in t4 and "measured on Qwen3.8-27B" in t4 and "K q8_0 / V q4_0 would still spill by" in t4, "...and the plan line says why, by how much, and that the mix would still spill")
    check("runs from system RAM" not in t4 and "all on GPU" not in t4, "...with no 'runs from RAM' line")
    still = run(q4_need / 2)
    check("-ctk q4_0 -ctv q4_0" in cmd(still) and "-c 8192" in cmd(still) and "is still" in out(still) and "system RAM" in out(still) and "all on GPU" not in out(still),
          "dry run, not even q4_0/q4_0 fits: q4_0/q4_0, the context stays, the line says part runs from RAM (layers spill only after the last step)")
    q8ok = run((f16_need + q8_need) / 2)
    check("-ctk q8_0 -ctv q8_0" in cmd(q8ok) and "q4_0" not in cmd(q8ok), "dry run, q8_0/q8_0 fits: it stays the choice (q4_0 only when q8_0 would spill)")
    ex4 = run((q8_need + mx_need) / 2, ["--exact"])
    check("-ctk" not in cmd(ex4) and "q4_0" not in cmd(ex4), "dry run, --exact keeps f16 even where q8_0 would spill")
    us8 = run((q8_need + mx_need) / 2, extra=["-ctk", "q8_0", "-ctv", "q8_0"])
    check(cmd(us8).count("-ctk") == 1 and "-ctk q8_0" in cmd(us8) and "q4_0" not in cmd(us8), "dry run, your own -ctk q8_0 stands (not changed)")
    usf = run((q8_need + mx_need) / 2, extra=["-ctk", "f16", "-ctv", "f16"])
    check(cmd(usf).count("-ctk") == 1 and "-ctk f16" in cmd(usf) and "q4_0" not in cmd(usf), "dry run, your own -ctk f16 stands")
    us4 = run(q4_need / 2, extra=["-ctk", "q4_0", "-ctv", "q4_0"])
    check(cmd(us4).count("-ctk") == 1 and "KV q4_0" not in out(us4) and "KV K q8_0" not in out(us4), "dry run, your own -ctk q4_0 stands, the launcher adds nothing of its own")
    usk = run((q8_need + mx_need) / 2, extra=["-ctk", "q8_0"])
    check(cmd(usk).count("-ctk") == 1 and "-ctv" not in cmd(usk).replace("--ctv", "") and "KV K q8_0" not in out(usk), "dry run, your own -ctk alone stands (the launcher chooses no V type next to it)")
    s8 = gp.dense_ffn_split(model, ctx, q4_need * GIB * 0.97, kv_q8=True, margin_gib=0.0)
    s4 = gp.dense_ffn_split(model, ctx, q4_need * GIB * 0.97, kv_q8="q4_0", margin_gib=0.0)
    sx = gp.dense_ffn_split(model, ctx, q4_need * GIB * 0.97, kv_q8=("q8_0", "q4_0"), margin_gib=0.0)
    check(s8 and s4 and sx and s4["kv_gib"] < sx["kv_gib"] < s8["kv_gib"] and (s4["n_off"] or 0) <= (sx["n_off"] or 0) <= (s8["n_off"] or 0), "layers to RAM are sized with the KV type chosen (q4_0 <= mix <= q8_0 layers)")
    check(gp.kv_elems(("bf16", "q4_0"))[0] == 2.0 and gp.kv_elems(("q5_0", "q8_0"))[0] == 22 / 32 and gp.kv_elems(("weird", "q4_0"))[0] == 2.0, "planner: other -ctk / -ctv types are sized by their own bytes, an unknown one as f16 (errs high; bf16 used to count as q8_0)")
    check([mc.kv_fit_index(*t) for t in (("f16",), ("q8_0",), ("q8_0", "q8_0"), ("q8_0", "q4_0"), ("q4_0", "q4_0"), ("bf16", "q4_0"))] == [0, 1, 1, 2, 3, 0], "kv_fit_index: which ladder entry sizes the KV types in the command")
    kq = mc.kv_quantized
    check(kq(["-ctk", "q8_0", "-ctv", "q8_0"]) and kq(["-ctk", "q4_0", "-ctv", "q4_0"]) and kq(["-ctk", "q8_0", "-ctv", "q4_0"]) and not kq(["-ctk", "f16", "-ctv", "f16"]) and not kq(["-c", "1"]),
          "kv_quantized: q8_0, q4_0 and the mix count as a quantized KV cache, f16 and no flag do not (the auto-tune path compared -ctk with 'q8_0' only: q4_0 read as f16)")
    _src = open(os.path.join(HERE, "bin", "moe_cache.py"), encoding="utf-8").read()
    check('"--cache-type-k") == "q8_0"' not in _src, "no caller decides 'KV quantized' with == \"q8_0\" any more")
    kt = mc.kv_plan_type
    check(kt(["-ctk", "q4_0", "-ctv", "q4_0"], True) == "q4_0" and kt(["-ctk", "q8_0"], False) == ("q8_0", "f16") and kt([], True) is True and kt([], False) is False
          and kt(["-ctk", "q8_0", "-ctv", "q4_0"], False) == ("q8_0", "q4_0") and kt(["-ctk", "q8_0", "-ctv", "q8_0"], False) is True and kt(["-ctk", "f16", "-ctv", "f16"], True) is False,
          "kv_plan_type: both types from the command (a missing one is f16), a pair for a mix, the launcher's flag when the command has none")
    none = run(q8_need / 2)
    check("-ctk q4_0" in cmd(none) and "-c 8192" in cmd(none) and "system RAM" in out(none) and "all on GPU" not in out(none), "dry run, q8_0 does not fit either: q4_0 (was q8_0), the context stays, says part runs from RAM, no 'all on GPU' line")
    check("all on GPU" not in out(roomy) and "all on GPU" not in out(tight), "no 'all on GPU' line anywhere")
    # --- automatic best GPU (owner's choice: --device only to leave integrated GPUs out) ---
    listing = ("Available devices:\n  Vulkan0: AMD Radeon(TM) Graphics (24000 MiB, 23000 MiB free)\n  Vulkan1: AMD Radeon RX 9070 XT (16304 MiB, 15000 MiB free)\n")
    ag = mc.auto_gpu_device([], listing)
    check(ag and ag[0] == "Vulkan1", "auto GPU: integrated Vulkan0 (more 'free' shared memory) and the 9070 XT -> --device Vulkan1")
    two = listing + "  Vulkan2: AMD Radeon RX 7900 XT (20000 MiB, 19000 MiB free)\n"
    check(mc.auto_gpu_device([], two)[0] == "Vulkan1,Vulkan2", "auto GPU: iGPU + two real cards -> both real cards (they still split as llama.cpp does)")
    check(mc.auto_gpu_device(["--device", "Vulkan0"], listing) is None and mc.auto_gpu_device(["-ts", "1,1"], listing) is None, "auto GPU: your own --device / -ts stands")
    check(mc.auto_gpu_device([], listing, cpu_only=True) is None and mc.auto_gpu_device([], listing, multi_gpu=True) is None, "auto GPU: not for --cpu-only or --multi-gpu")
    check(mc.auto_gpu_device([], "Available devices:\n  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15000 MiB free)\n") is None, "auto GPU: one GPU -> nothing to choose")
    only_i = "Available devices:\n  Vulkan0: Intel(R) Arc(TM) Graphics (20000 MiB, 19000 MiB free)\n  Vulkan1: AMD Radeon(TM) Graphics (24000 MiB, 23000 MiB free)\n"
    check(mc.auto_gpu_device([], only_i) is None, "auto GPU: only integrated GPUs -> nothing is left out, used as before")
    cc = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 3090 (24000 MiB, 23000 MiB free)\n  CUDA1: NVIDIA GeForce RTX 3090 (24000 MiB, 23000 MiB free)\n"
    check(mc.auto_gpu_device([], cc) is None, "auto GPU: two real cards, no integrated GPU -> nothing added (llama.cpp splits)")
    check(mc.pick_gpu_full(listing)[0] == "Vulkan1", "pick_gpu_full (plan and control page): a card that is not integrated comes first even on Linux")
    # --- the log scan ---
    line = b"llama_context: layer 41 is assigned to device Vulkan1 but Flash Attention over K/V rows is assigned to device CPU\n"
    n = mc.cpu_placement_note(line)
    check(n and n[0] == "attn" and "WARNING" in n[1] and "layer 41" in n[1] and "Vulkan1" in n[1], "log scan: 'Flash Attention over K/V rows is assigned to device CPU' -> loud warning naming the layer")
    n2 = mc.cpu_placement_note("load_tensors: offloaded 63/65 layers to GPU")
    check(n2 and n2[0] == "offload" and "63 of 65" in n2[1], "log scan: partial offload -> warning")
    check(mc.cpu_placement_note("load_tensors: offloaded 65/65 layers to GPU") is None, "log scan: everything offloaded -> nothing")
    check(mc.cpu_placement_note("load_tensors: offloaded 0/65 layers to GPU", cpu_run=True) is None, "log scan: a CPU-only run asked for 0 layers -> nothing")
    check(mc.cpu_placement_note("srv  load_model: loading model 'x'") is None, "log scan: an ordinary line -> nothing")
    # --- planned CPU work stays quiet ---
    mc.EXPECT_CPU_WORK[0] = True
    check(mc.EXPECT_CPU_WORK[0] is True, "log scan flag exists (the pump skips the note while the plan itself puts work on the CPU)")
    mc.EXPECT_CPU_WORK[0] = False
if fails:
    print("FAILED: " + "; ".join(fails)); sys.exit(1)
print("all ok")
