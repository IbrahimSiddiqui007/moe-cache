#!/usr/bin/env python3
"""LAUNCHER-982 / 983: the dense KV-cache choice, the 'all on GPU' line, the llama-server log scan and the integrated-GPU warning, with fake VRAM figures and fake log lines.
   python3 tests/dense_kv_fit_test.py
PC 2026-11-08: Qwen3.8-27B at --ctx 65536 on a 16 GB card ran 18.9 tok/s against 35 at 24k because the f16 KV cache did not fit beside the model and llama.cpp moved work to the CPU."""
import os, struct, sys, tempfile, types
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

def make_dense(path, n_layers=24, d=2048, nextn=0, ple_gib=0):
    def string(x):
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    infos, off = [], 0
    def add(name, dims, ty, nbytes):
        nonlocal off
        infos.append((name, dims, ty, off)); off += (nbytes + 31) // 32 * 32
    add("token_embd.weight", [d, 32000], 1, d * 32000 * 2)
    if ple_gib:                                                  # LAUNCHER-1162: Gemma-style per-layer token-embedding table (f16, ple_gib GiB; a sparse file)
        add("per_layer_token_embd.weight", [256, ple_gib * 2**30 // 512], 1, ple_gib * 2**30)
    for l in range(n_layers):
        add(f"blk.{l}.attn_q.weight", [d, d], 12, d // 256 * 144 * d)
        add(f"blk.{l}.ffn_up.weight", [d, 4 * d], 12, d // 256 * 144 * 4 * d)
    for l in range(n_layers, n_layers + nextn):               # the NextN / MTP layer(s): a whole layer past the main stack (Qwen3.8-27B: blk.64, 0.327 GiB)
        add(f"blk.{l}.attn_q.weight", [d, d], 12, d // 256 * 144 * d)
        add(f"blk.{l}.ffn_up.weight", [d, 4 * d], 12, d // 256 * 144 * 4 * d)
        add(f"blk.{l}.nextn.eh_proj.weight", [2 * d, d], 12, 2 * d // 256 * 144 * d)
        add(f"blk.{l}.ffn_gate.weight", [d, 40 * d], 12, d // 256 * 144 * 40 * d)        # a layer that weighs something (real: 0.327 GiB), so the plan line has a figure
    kv = [("general.architecture", 8, string("llama")), ("llama.block_count", 4, struct.pack("<I", n_layers + nextn)), ("llama.context_length", 4, struct.pack("<I", 131072)),
          ("llama.embedding_length", 4, struct.pack("<I", d)), ("llama.attention.head_count", 4, struct.pack("<I", 16)), ("llama.attention.head_count_kv", 4, struct.pack("<I", 4))] + ([("llama.nextn_predict_layers", 4, struct.pack("<I", nextn))] if nextn else [])
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
    # --- LAUNCHER-1162: the per-layer token-embedding table is host memory, not GPU weights ---
    ple = os.path.join(tmp, "dense-ple.gguf"); make_dense(ple, ple_gib=3)
    b0, b1 = gp.dense_gpu_fit(model, 8192, 64 * GIB, False), gp.dense_gpu_fit(ple, 8192, 64 * GIB, False)
    check(b0 and b1 and abs(b1["model_gib"] - b0["model_gib"]) < 0.01, f"dense_gpu_fit: a 3 GiB per_layer_token_embd table adds nothing to the GPU weights ({b0 and b0['model_gib']:.2f} vs {b1 and b1['model_gib']:.2f} GiB)")
    f0, f1 = gp.dense_ffn_split(model, 8192, 2 * GIB, False), gp.dense_ffn_split(ple, 8192, 2 * GIB, False)
    check(f0 and f1 and abs(f1["need_gib"] - f0["need_gib"]) < 0.01, "dense_ffn_split: the same table adds nothing to the need")
    # --- GUI-1156: the control page's "does a second exact-mode server fit" need counts the KV cache of the context, not a flat 10 % + 1.5 GiB ---
    import moe_cache_ui as ui      # noqa: E402
    sz = os.path.getsize(model)
    old = int(sz * 1.10 + 1.5 * GIB)
    n_small, n_big = ui.audit_need(sz, model, 4096), ui.audit_need(sz, model, 65536)
    check(n_big > n_small and n_big > old + 1.5 * GIB, f"audit_need: the need grows with the context ({n_small / GIB:.2f} GiB at 4k, {n_big / GIB:.2f} at 64k; the old flat rule said {old / GIB:.2f})")
    junk = os.path.join(tmp, "junk.gguf"); open(junk, "wb").write(b"GGUF" + b"\0" * 64)
    check(ui.audit_need(sz, junk, 65536) == old and ui.audit_need(sz, None, 8192) == old and ui.audit_need(None, model, 8192) is None, "audit_need: unreadable header or no path keeps the old rule; no size -> None")
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
    between8 = run((q8_need + mx_need) / 2, ["--kv-4bit"])
    check("-ctk q8_0 -ctv q4_0" in cmd(between8) and "-c 8192" in cmd(between8), "dry run, q8_0/q8_0 would spill but K q8_0 / V q4_0 fits: that mix, the context stays")
    t8 = out(between8)
    check("KV K q8_0 / V q4_0: q8_0 / q8_0 would spill by" in t8 and "this step saves" in t8 and "does not spill" in t8 and "+0.003 to +0.006 KL" in t8 and "one run each" in t8 and "--exact keeps f16" in t8, "...and the plan line names the step, the spill, the GiB it saves and the quality cost")
    import re as _re
    sv = float(_re.search(r"this step saves ([0-9.]+) GiB", t8).group(1))
    check(abs(sv - (kv8 - kvx) / GIB) < 0.06, f"...the GiB saved is the KV difference to q8_0/q8_0 ({sv:.1f} vs {(kv8 - kvx) / GIB:.2f})")
    between4 = run((mx_need + q4_need) / 2, ["--kv-4bit"])
    check("-ctk q4_0 -ctv q4_0" in cmd(between4) and "-ctk q8_0 -ctv q4_0" not in cmd(between4) and "-c 8192" in cmd(between4), "dry run, the mix would spill too but q4_0/q4_0 fits: q4_0/q4_0, the context stays")
    t4 = out(between4)
    check("KV q4_0 / q4_0: q8_0 / q8_0 would spill by" in t4 and "does not spill" in t4 and "+0.022 to +0.040 KL" in t4 and "one run each" in t4 and "K q8_0 / V q4_0 would still spill by" in t4, "...and the plan line says why, by how much, and that the mix would still spill")
    check("runs from system RAM" not in t4 and "all on GPU" not in t4, "...with no 'runs from RAM' line")
    still = run(q4_need / 2, ["--kv-4bit"])
    check("-ctk q4_0 -ctv q4_0" in cmd(still) and "-c 8192" in cmd(still) and "is still" in out(still) and "system RAM" in out(still) and "all on GPU" not in out(still),
          "dry run with --kv-4bit, not even q4_0/q4_0 fits: q4_0/q4_0, the context stays, the line says part runs from RAM (layers spill only after the last step)")
    q8ok = run((f16_need + q8_need) / 2)
    check("-ctk q8_0 -ctv q8_0" in cmd(q8ok) and "q4_0" not in cmd(q8ok), "dry run, q8_0/q8_0 fits: it stays the choice (q4_0 only when q8_0 would spill)")
    # owner 2026-11-20: the two 4-bit steps are off by default and come only with --kv-4bit (or your own -ctk/-ctv)
    import moe_cache_approx as ax
    check(ax.token_mode("kv-k8v4")["gate"] == "kv" and ax.token_mode("kv-q4")["gate"] == "kv" and not ax.default_on("kv-k8v4") and not ax.default_on("kv-q4")
          and all("2026-11-20" in ax.token_mode(i)["reason"] and "--kv-4bit" in ax.token_mode(i)["switch"] for i in ("kv-k8v4", "kv-q4")),
          "the two steps have their own rows, gate level kv, off today (owner 2026-11-20), switch --kv-4bit")
    for need, what in (((q8_need + mx_need) / 2, "the mix would fit"), ((mx_need + q4_need) / 2, "only q4_0/q4_0 would fit"), (q4_need / 2, "nothing fits")):
        d = run(need)
        o = out(d)
        check("q4_0" not in cmd(d) and "-ctk q8_0 -ctv q8_0" in cmd(d) and "-c 8192" in cmd(d) and "still spills" in o and "system RAM" in o and "--kv-4bit would store" in o
              and "GiB less KV cache" in o and "quality cost" in o and "owner 2026-11-20" in o,
              f"default, {what}: q8_0/q8_0 stays, the context stays, the line says part runs from RAM and what --kv-4bit would cost and gain")
    check("-ctk q8_0 -ctv q4_0" in cmd(run((q8_need + mx_need) / 2, ["--kv-4bit"])) and "-ctk q4_0 -ctv q4_0" in cmd(run((mx_need + q4_need) / 2, ["--kv-4bit"])), "--kv-4bit brings both steps back")
    sv_on = run((q8_need + mx_need) / 2, ["--kv-4bit"]); check("-ctk q8_0 -ctv q8_0" in cmd(run(q8_need + 0.01)) and "q4_0" not in cmd(run(q8_need + 0.01, ["--kv-4bit"])), "--kv-4bit changes nothing where q8_0/q8_0 fits")
    ex4 = run((q8_need + mx_need) / 2, ["--exact", "--kv-4bit"])
    check("-ctk" not in cmd(ex4) and "q4_0" not in cmd(ex4), "dry run, --exact keeps f16 even with --kv-4bit where q8_0 would spill")
    us8 = run((q8_need + mx_need) / 2, extra=["--kv-4bit", "-ctk", "q8_0", "-ctv", "q8_0"])
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
    # --- the MTP layer is not on the GPU when MTP is off (the pinned llama.cpp skips its tensors without load_mtp: 0.327 GiB on Qwen3.8-27B) ---
    mtp_model = os.path.join(tmp, "dense-mtp-Q4.gguf"); make_dense(mtp_model, nextn=1)
    nb = gp.nextn_bytes(mtp_model)
    check(nb > 0 and gp.nextn_bytes(model) == 0, f"nextn_bytes: the whole NextN layer ({nb / GIB:.3f} GiB), 0 for a model without one")
    full = gp.dense_gpu_fit(mtp_model, ctx, 64 * GIB, False, 0.0, 0.0, 0.20)
    trim = gp.dense_gpu_fit(mtp_model, ctx, 64 * GIB, False, 0.0, 0.0, 0.20, skip_mtp=True)
    check(abs((full["model_gib"] - trim["model_gib"]) - nb / GIB) < 1e-9 and trim["mtp_skipped_gib"] == nb / GIB and full["mtp_skipped_gib"] == 0.0, "dense_gpu_fit(skip_mtp): the NextN layer's bytes come off the model size")
    check(abs(gp.dense_gpu_fit(model, ctx, 64 * GIB, False, 0.0, 0.0, 0.20, skip_mtp=True)["need_gib"] - gp.dense_gpu_fit(model, ctx, 64 * GIB, False, 0.0, 0.0, 0.20)["need_gib"]) < 1e-9, "a model without a NextN layer: skip_mtp changes nothing")
    class _Hm:
        def __init__(self, text): self.text = text
        def run(self, cmd, **k): return types.SimpleNamespace(stdout=self.text, stderr="", returncode=0)
    check(mc.mtp_layer_skipped("/nonexistent/llama-server", [], _Hm("  --spec-type draft-mtp\n")) is True, "mtp_layer_skipped: MTP off + a llama-server that knows draft-mtp: the layer is not loaded")
    check(mc.mtp_layer_skipped("/nonexistent/llama-server", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"], _Hm("  --spec-type draft-mtp\n")) is False, "...MTP on: loaded, counted")
    check(mc.mtp_layer_skipped("/nonexistent/llama-server", [], _Hm("  --cache-ram N\n")) is False, "...an older llama-server without draft-mtp loads it with the model: counted")
    # the plan: f16 fits only without the MTP layer
    m16 = lambda sk: gp.dense_gpu_fit(mtp_model, ctx, 64 * GIB, False, 0.0, 0.0, 0.20, skip_mtp=sk)["need_gib"]
    lim = (m16(False) + m16(True)) / 2
    run2 = lambda extra_opts=(): lp.run_case("linux", ["--ram", "12", "--vram", f"{lim:.3f}", "--ctx", str(ctx)] + list(extra_opts), [], mtp_model, tmp)
    r_off = run2(["--no-mtp"]); r_on = run2(["--mtp"]); r_def = run2()
    c_off = next((l for l in r_off["stdout"].splitlines() if "llama-server" in l and " -m " in l), ""); c_on = next((l for l in r_on["stdout"].splitlines() if "llama-server" in l and " -m " in l), "")
    o_off = r_off["stdout"] + r_off.get("stderr", ""); o_on = r_on["stdout"] + r_on.get("stderr", "")
    check("-ctk" not in c_off and "MTP is off, so llama-server does not load the model's MTP layer" in o_off and "KV cache does not fit" not in o_off,
          "dry run, MTP off, f16 fits only without the MTP layer: f16 stays, the line says the layer is not counted and how many GiB")
    c_def = next((l for l in r_def["stdout"].splitlines() if "llama-server" in l and " -m " in l), "")
    check("-ctk" not in c_def and "MTP is off, so llama-server does not load" in r_def["stdout"] + r_def.get("stderr", ""), "dry run, no MTP flag (the default for a dense model): the layer is not counted, f16 stays")
    check("-ctk q8_0" in c_on and "does not load the model's MTP layer" not in o_on, "dry run, --mtp: the layer is counted, so f16 does not fit: q8_0 KV (unchanged behaviour)")
    kt = mc.kv_plan_type
    check(kt(["-ctk", "q4_0", "-ctv", "q4_0"], True) == "q4_0" and kt(["-ctk", "q8_0"], False) == ("q8_0", "f16") and kt([], True) is True and kt([], False) is False
          and kt(["-ctk", "q8_0", "-ctv", "q4_0"], False) == ("q8_0", "q4_0") and kt(["-ctk", "q8_0", "-ctv", "q8_0"], False) is True and kt(["-ctk", "f16", "-ctv", "f16"], True) is False,
          "kv_plan_type: both types from the command (a missing one is f16), a pair for a mix, the launcher's flag when the command has none")
    none = run(q8_need / 2, ["--kv-4bit"])
    check("-ctk q4_0" in cmd(none) and "-c 8192" in cmd(none) and "system RAM" in out(none) and "all on GPU" not in out(none), "dry run with --kv-4bit, q8_0 does not fit either: q4_0, the context stays, says part runs from RAM, no 'all on GPU' line")
    none_d = run(q8_need / 2)
    check("-ctk q8_0 -ctv q8_0" in cmd(none_d) and "q4_0" not in cmd(none_d) and "-c 8192" in cmd(none_d) and "system RAM" in out(none_d) and "all on GPU" not in out(none_d), "dry run by default, q8_0 does not fit either: q8_0 stays, the context stays, says part runs from RAM")
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
    # --- LAUNCHER-1067 / TEST-1205 D, default since LAUNCHER-1225 (items 607 + 614): the dense spill rule is ON by default for a dense plan that spills at 128k+, OFF below;
    #     MOE_CACHE_DENSE_SPILL_RULE=1 forces it from 64k, =0 forces it off; --exact and the user's own -np / -ub win ---
    c64, c128 = 65536, 131072
    q4_64 = gp.dense_gpu_fit(model, c64, 64 * GIB, "q4_0", 0.0, 0.0, 0.20)["need_gib"]
    q4_128 = gp.dense_gpu_fit(model, c128, 64 * GIB, "q4_0", 0.0, 0.0, 0.20)["need_gib"]
    real_stage = mc.dense_spill_stage
    box = {"env": None}
    def with_switch(*a_, **k_):          # lp.run_case clears MOE_CACHE_* from the environment, so the switch goes in through a wrapper of the stage
        if box["env"] is not None:
            os.environ["MOE_CACHE_DENSE_SPILL_RULE"] = box["env"]
        try:
            return real_stage(*a_, **k_)
        finally:
            os.environ.pop("MOE_CACHE_DENSE_SPILL_RULE", None)
    mc.dense_spill_stage = with_switch
    def sp(ctx, vram, env=None, opts=(), extra=()):
        box["env"] = env
        return lp.run_case("linux", ["--ram", "12", "--vram", f"{vram:.3f}", "--ctx", str(ctx)] + list(opts), list(extra), model, tmp)
    spilled = lambda r: "-np 1" in cmd(r) and "-ub 256" in cmd(r) and cmd(r).count("-ub ") == 1
    try:
        d128 = sp(c128, q4_128 / 2)
        check(spilled(d128) and "default at 128k+" in out(d128) and "spills" in out(d128) and "LAUNCHER-1225" in out(d128) and "+41.7 % prompt" in out(d128),
              "spill rule default, dense 128k plan that spills: -np 1 and -ub 256 (one -ub), the source line names the default, the ledger row and the measured figures")
        d64 = sp(c64, q4_64 / 2)
        check(not spilled(d64) and "dense spill rule" not in out(d64), "spill rule default, dense 64k plan that spills: plan unchanged, no line (64k lost 49 % prompt, item 606_r)")
        f128 = sp(c128, q4_128 * 3)
        check("-np 1" not in cmd(f128) and "does not spill: not applied" in out(f128), "spill rule default, 128k plan that fits: not applied, said")
        off128 = sp(c128, q4_128 / 2, env="0")
        check(not spilled(off128) and "dense spill rule" not in out(off128), "MOE_CACHE_DENSE_SPILL_RULE=0 at 128k: off")
        on64 = sp(c64, q4_64 / 2, env="1")
        check(spilled(on64) and "forced on" in out(on64), "MOE_CACHE_DENSE_SPILL_RULE=1 at 64k: on (forced), said")
        short = sp(8192, q4_64 / 4, env="1")
        check("-np 1" not in cmd(short), "MOE_CACHE_DENSE_SPILL_RULE=1, 8k context: below the 64k floor, not applied")
        ex = sp(c128, q4_128 / 2, opts=["--exact"])
        check(not spilled(ex) and "dense spill rule" not in out(ex), "--exact at 128k: not applied")
        ex1 = sp(c128, q4_128 / 2, env="1", opts=["--exact"])
        check(not spilled(ex1), "--exact beats MOE_CACHE_DENSE_SPILL_RULE=1")
        mine = sp(c128, q4_128 / 2, extra=["-np", "2"])
        check("-np 1" not in cmd(mine) and "-ub 256" not in cmd(mine) and "your own -np / -ub win" in out(mine), "default at 128k, your own -np 2: not applied, said")
        # golden plan lines: the dry-run command at 128k with the rule (default) and with the rule off differ in exactly -np 1 / -ub 256
        check(spilled(d128) and not spilled(off128) and "-ub" in cmd(off128), "golden 64k/128k: 128k default = rule off + (-np 1, -ub 256); 64k default = rule off")
    finally:
        mc.dense_spill_stage = real_stage
if fails:
    print("FAILED: " + "; ".join(fails)); sys.exit(1)
print("all ok")
