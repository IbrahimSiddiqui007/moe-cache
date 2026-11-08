#!/usr/bin/env python3
"""tools/expert_mix.py on tiny synthetic GGUF files (MODELS-759/760): header listing, the cold-first choice from a usage profile, and a mix whose experts come from
file A or file B. Uses the numpy codec (Q8_0 and Q4_0 inputs, Q8_0 output); when a libggml is given in EXPERT_MIX_LIBGGML the ggml codec is compared with it too.
   python3 tests/expert_mix_test.py"""
import os, struct, sys, tempfile
try:
    import numpy as np
except ImportError:             # the tool needs numpy; CI installs it (quick.yml), a contributor machine without it skips
    if os.environ.get("CI"):
        raise
    print("expert_mix_test: skipped (numpy is not installed)")
    sys.exit(0)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
import expert_mix as em   # noqa: E402
import stitch_gguf as sg  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

N_L, N_E, NE0, NE1 = 2, 4, 64, 8       # 2 layers, 4 experts, each expert slice 8 rows x 64 columns
rng = np.random.default_rng(7)

def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
def kv_str(k, v): return s(k) + struct.pack("<I", 8) + s(v)
def kv_u32(k, v): return s(k) + struct.pack("<II", 4, v)

def q8(x):
    return em.PyCodec().quant(8, x, NE1, NE0)

def q4_0(x):                       # a simple Q4_0 quantiser for the test: d = max-abs/8 as fp16, values 0..15 around 8
    x = x.reshape(-1, 32)
    d = (np.abs(x).max(axis=1) / 8.0).astype(np.float16); dd = d.astype(np.float32)
    q = np.clip(np.rint(x / np.where(dd == 0, 1, dd)[:, None]) + 8, 0, 15).astype(np.uint8)
    out = np.empty((x.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.tobytes()

def write_gguf(path, ty, quant, values):
    """values[layer][expert] = float32 (NE1 x NE0); tensors: blk.L.ffn_up_exps.weight (type ty) and token_embd.weight (F32)."""
    tens = []
    for L in range(N_L):
        raw = b"".join(quant(values[L][e].reshape(-1)) for e in range(N_E))
        tens.append((f"blk.{L}.ffn_up_exps.weight", [NE0, NE1, N_E], ty, raw))
    emb = np.arange(32, dtype=np.float32)
    tens.append(("token_embd.weight", [32], 0, emb.tobytes()))
    kv = kv_str("general.architecture", "test") + kv_u32("test.block_count", N_L) + kv_u32("test.expert_count", N_E) + kv_u32("general.alignment", 32)
    info, off = b"", 0
    for name, dims, t, raw in tens:
        info += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", t, off)
        off += (len(raw) + 31) // 32 * 32
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tens), 4) + kv + info
    head += b"\0" * (-len(head) % 32)
    with open(path, "wb") as f:
        f.write(head)
        for _, _, _, raw in tens:
            f.write(raw + b"\0" * (-len(raw) % 32))

def read_experts(path):
    g = sg.Gguf(path)
    out = {}
    with open(path, "rb") as f:
        for L in range(N_L):
            t = g.by_name[f"blk.{L}.ffn_up_exps.weight"]
            row = em.PyCodec().row_size(t["type"], NE0)
            for e in range(N_E):
                f.seek(g.data_start + t["offset"] + e * NE1 * row)
                out[(L, e)] = em.PyCodec().dequant(t["type"], f.read(NE1 * row), NE0 * NE1).reshape(NE1, NE0)
    return g, out

tmp = tempfile.mkdtemp(prefix="moe-mix-test-")
valsA = [[rng.normal(0, 1, (NE1, NE0)).astype(np.float32) for _ in range(N_E)] for _ in range(N_L)]
valsB = [[rng.normal(0, 1, (NE1, NE0)).astype(np.float32) for _ in range(N_E)] for _ in range(N_L)]
A, B = os.path.join(tmp, "a.gguf"), os.path.join(tmp, "b.gguf")
write_gguf(A, 8, lambda x: q8(x), valsA)
write_gguf(B, 2, q4_0, valsB)
_, deqA = read_experts(A)
_, deqB = read_experts(B)

# usage: expert (L, e) has count 10*L + e, so the coldest are (0,0), (0,1), (0,2), (0,3), (1,0) ...
usage = os.path.join(tmp, "u.usage")
open(usage, "wb").write(b"XUSE" + struct.pack("<ii", N_L, N_E) + np.array([[10 * L + e for e in range(N_E)] for L in range(N_L)], dtype="<f4").tobytes())
n_l, n_e, u = em.load_usage(usage)
check((n_l, n_e) == (N_L, N_E) and u[1, 2] == 12, "the usage profile is read (layers, experts, counts)")
pick = em.pick_from_b(u, 0.5)
check(pick == {(0, 0), (0, 1), (0, 2), (0, 3)}, f"fraction 0.5 takes the 4 coldest of 8 experts ({sorted(pick)})")
check(em.pick_from_b(u, 0.0) == set() and len(em.pick_from_b(u, 1.0)) == 8 and em.pick_from_b(u, 0.25) == {(0, 0), (0, 1)}, "fractions 0, 1 and 0.25")
ties = em.pick_from_b(np.zeros((N_L, N_E), dtype=np.float32), 0.25)
check(ties == {(0, 0), (0, 1)}, "ties go by (layer, expert), so a run can be repeated")

import contextlib, io
def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = em.main(list(argv))
    return rc, out.getvalue(), err.getvalue()

rc, out, _ = run("types", A, B)
check(rc == 0 and "Q8_0" in out and "Q4_0" in out and "expert tensors with another type in B: 2" in out, "types: both headers listed, 2 expert tensors differ in type")

for f in (0.0, 0.5, 1.0):
    M = os.path.join(tmp, f"mix{int(f * 100)}.gguf")
    rc, out, err = run("mix", "--a", A, "--b", B, "--fraction", str(f), "--usage", usage, "--fine", "q8_0", "--out", M)
    check(rc == 0, f"mix fraction {f}: written ({err.strip()})")
    g, deq = read_experts(M)
    types = {t["type"] for t in g.tensors if t["name"].startswith("blk.")}
    check(types == {8}, f"mix {f}: the expert tensors are Q8_0")
    want_b = em.pick_from_b(u, f)
    exact = all(np.abs(deq[k] - (deqB[k] if k in want_b else deqA[k])).max() < 0.02 * max(np.abs(deqA[k]).max(), np.abs(deqB[k]).max()) for k in deq)
    check(exact, f"mix {f}: every expert equals its source (B for {len(want_b)}, A for {len(deq) - len(want_b)}) within the Q8_0 error")
    emb = np.frombuffer(open(M, "rb").read()[g.data_start + g.by_name["token_embd.weight"]["offset"]:][:128], dtype=np.float32)
    check((emb == np.arange(32, dtype=np.float32)).all(), f"mix {f}: the other tensors are copied byte for byte")
    import json
    man = json.load(open(M + ".mix.json"))
    check(man["experts_from_b"] == len(want_b) and man["experts_from_a"] == 8 - len(want_b), f"mix {f}: the manifest counts {man['experts_from_a']} from A, {man['experts_from_b']} from B")

rc, _, err = run("mix", "--a", A, "--b", B, "--fraction", "0.5", "--usage", usage, "--fine", "q8_0", "--out", os.path.join(tmp, "mix50.gguf"))
check(rc == 2 and "exists" in err, "an existing output is never overwritten")
rc, _, err = run("mix", "--a", A, "--b", B, "--fraction", "0.5", "--usage", usage, "--out", os.path.join(tmp, "q6.gguf"))
check(rc == 0 and {t["type"] for t in sg.Gguf(os.path.join(tmp, "q6.gguf")).tensors if t["name"].startswith("blk.")} == {8}, f"default q6_k: 64-column tensors fall back to Q8_0 ({err.strip()[:60]})")

# A model in two files (NAME-00001-of-00002.gguf, as Hugging Face ships the big ones): the expert tensors in part 1, token_embd in part 2.
# 2026-11-08 the PC mix of Flash-Next read part 1 only (1,223 of 1,224 tensors) and copied split.count = 2 into a one-file output, which llama.cpp refuses.
def kv_u16(k, v): return s(k) + struct.pack("<IH", 2, v)
def split_copy(src, prefix):
    """Rewrite one test file as two parts with split.* metadata; -> path of part 1."""
    g = sg.Gguf(src)
    data = open(src, "rb").read()
    parts = [[t for t in g.tensors if t["name"].startswith("blk.")], [t for t in g.tensors if not t["name"].startswith("blk.")]]
    base = kv_str("general.architecture", "test") + kv_u32("test.block_count", N_L) + kv_u32("test.expert_count", N_E) + kv_u32("general.alignment", 32)
    for no, tens in enumerate(parts):
        kv = base + kv_u16("split.no", no) + kv_u16("split.count", 2) + struct.pack("<Q", len("split.tensors.count")) + b"split.tensors.count" + struct.pack("<Ii", 5, len(g.tensors))
        info, off, blobs = b"", 0, []
        for t in tens:
            raw = data[g.data_start + t["offset"]:][:t["bytes"]]
            info += s(t["name"]) + struct.pack("<I", len(t["dims"])) + b"".join(struct.pack("<Q", d) for d in t["dims"]) + struct.pack("<IQ", t["type"], off)
            off += (len(raw) + 31) // 32 * 32
            blobs.append(raw + b"\0" * (-len(raw) % 32))
        head = b"GGUF" + struct.pack("<IQQ", 3, len(tens), 7) + kv + info
        head += b"\0" * (-len(head) % 32)
        open(f"{prefix}-{no + 1:05d}-of-00002.gguf", "wb").write(head + b"".join(blobs))
    return f"{prefix}-00001-of-00002.gguf"

As, Bs = split_copy(A, os.path.join(tmp, "as")), split_copy(B, os.path.join(tmp, "bs"))
check(len(sg.Gguf(As).tensors) == 2 and em.Model(As).shards[1].path.endswith("-00002-of-00002.gguf") and len(em.Model(As).tensors) == 3, "a two-file model is read as one table (2 + 1 tensors)")
MS = os.path.join(tmp, "mix_split.gguf")
rc, out, err = run("mix", "--a", As, "--b", Bs, "--fraction", "0.5", "--usage", usage, "--fine", "q8_0", "--out", MS)
check(rc == 0, f"mix of two-file models: written ({err.strip()})")
gs, deqs = read_experts(MS)
check(len(gs.tensors) == 3 and "token_embd.weight" in gs.by_name, f"the one-file output holds all 3 tensors, also the one from part 2 ({len(gs.tensors)})")
check(not any(k.startswith("split.") for k, _, _ in gs.kv_spans) and gs.meta.get("general.architecture") == "test", "the output has no split.* metadata (llama.cpp would look for a part 2) and keeps the rest")
want_b = em.pick_from_b(u, 0.5)
check(all(np.abs(deqs[k] - (deqB[k] if k in want_b else deqA[k])).max() < 0.02 * max(np.abs(deqA[k]).max(), np.abs(deqB[k]).max()) for k in deqs), "two-file mix: every expert equals its source")
emb = np.frombuffer(open(MS, "rb").read()[gs.data_start + gs.by_name["token_embd.weight"]["offset"]:][:128], dtype=np.float32)
check((emb == np.arange(32, dtype=np.float32)).all(), "two-file mix: the tensor from part 2 is copied byte for byte")
os.rename(os.path.join(tmp, "as-00002-of-00002.gguf"), os.path.join(tmp, "as-gone.gguf"))
rc, _, err = run("mix", "--a", As, "--b", Bs, "--fraction", "0.5", "--usage", usage, "--fine", "q8_0", "--out", os.path.join(tmp, "x.gguf"))
check(rc == 2 and "missing" in err, "a missing part is refused, nothing written")
rc, _, err = run("mix", "--a", os.path.join(tmp, "bs-00002-of-00002.gguf"), "--b", Bs, "--fraction", "0.5", "--usage", usage, "--fine", "q8_0", "--out", os.path.join(tmp, "y.gguf"))
check(rc == 2 and "first file" in err, "part 2 given as --a is refused")

lib = os.environ.get("EXPERT_MIX_LIBGGML")
if lib and os.path.isfile(lib):          # Q6_K through libggml: 64 columns fall back to Q8_0 for that tensor, threads give the same bytes as one thread
    outs = []
    for th in (1, 3):
        M = os.path.join(tmp, f"q6_t{th}.gguf")
        rc, out, err = run("mix", "--a", A, "--b", B, "--fraction", "0.5", "--usage", usage, "--libggml", lib, "--threads", str(th), "--out", M)
        outs.append(open(M, "rb").read())
        check(rc == 0 and {t["type"] for t in sg.Gguf(M).tensors if t["name"].startswith("blk.")} == {8}, f"q6_k with libggml, {th} thread(s): 64-column tensors fall back to Q8_0 ({err.strip()[:60]})")
    check(outs[0] == outs[1], "one thread and three threads write the same bytes")
if lib and os.path.isfile(lib):
    c = em.GgmlCodec(lib)
    raw = q8(valsA[0][0].reshape(-1))
    check(np.allclose(c.dequant(8, raw, NE0 * NE1), em.PyCodec().dequant(8, raw, NE0 * NE1)), "libggml dequantises Q8_0 like the numpy codec")
    x6 = rng.normal(0, 1, (NE1, 256)).astype(np.float32)
    q6 = c.quant(14, x6, NE1, 256)
    check(len(q6) == c.row_size(14, 256) * NE1 and np.abs(c.dequant(14, q6, NE1 * 256) - x6.reshape(-1)).max() < 0.1, "libggml writes Q6_K and reads it back within its error")
else:
    print("  skip libggml checks (set EXPERT_MIX_LIBGGML to a libggml-base.so)")

import shutil; shutil.rmtree(tmp, ignore_errors=True)
print("FAILED: " + "; ".join(fails) if fails else "expert_mix_test: all passed")
sys.exit(1 if fails else 0)
