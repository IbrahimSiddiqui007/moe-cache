#!/usr/bin/env python3
"""MODELS-640: tools/stitch_gguf.py builds a GGUF from two quantisations of one model. Model-free: two synthetic GGUF files with the same tensor names and
shapes, different tensor types and different bytes (tiny, random), written here with the standard library.
   python3 tests/stitch_gguf_test.py"""
import json, os, random, struct, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools")); sys.path.insert(0, os.path.join(HERE, "planner"))
import stitch_gguf as S

bad = 0
def check(ok, what):
    global bad
    print(("  ok   " if ok else "  FAIL ") + what)
    bad += 0 if ok else 1

def s_(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
def kv(k, t, v): return s_(k) + struct.pack("<I", t) + v

def make(path, ty_attn, ty_exps, seed, n_layers=4, arch="qwen35moe", n_expert=8, drop=None, reshape=None, align=32, extra_tensor=None):
    """Tensors: token_embd, output_norm, output, and per layer attn_q, ffn_gate_exps, ffn_up_exps, ffn_down_exps (random payload; sizes depend on the type)."""
    rnd = random.Random(seed)
    size = {ty_attn: 40, ty_exps: 52}
    exact = lambda dm, ty, sz: ({0: 4, 1: 2}[ty] * __import__("math").prod(dm)) if ty in (0, 1) else sz
    ts = [("token_embd.weight", [16, 8], 1, 70), ("output_norm.weight", [16], 0, 64), ("output.weight", [16, 8], ty_attn, 40)]
    for L in range(n_layers):
        ts.append((f"blk.{L}.attn_q.weight", [16, 16], ty_attn, size[ty_attn] + 7 * L))
        for nm in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            ts.append((f"blk.{L}.{nm}.weight", [16, 4, n_expert], ty_exps, size[ty_exps] * 3 + 5 * L))
    if drop:
        ts = [t for t in ts if t[0] != drop]
    if reshape:
        ts = [(n, [d + 1 for d in dm] if n == reshape else dm, ty, sz) for n, dm, ty, sz in ts]
    if extra_tensor:
        ts.append(extra_tensor)
    kvs = [kv("general.architecture", 8, s_(arch)), kv("general.name", 8, s_("tiny")),
           kv(f"{arch}.block_count", 4, struct.pack("<I", n_layers)), kv(f"{arch}.expert_count", 4, struct.pack("<I", n_expert)),
           kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 3) + s_("a") + s_("bb") + s_("ccc")),
           kv("tokenizer.ggml.scores", 9, struct.pack("<IQ", 6, 3) + struct.pack("<3f", 0, 1, 2)), kv("general.file_type", 4, struct.pack("<I", ty_exps))]
    if align != 32:
        kvs.append(kv("general.alignment", 4, struct.pack("<I", align)))
    infos, data, off = b"", b"", 0
    for n, dm, ty, sz in ts:
        infos += s_(n) + struct.pack("<I", len(dm)) + b"".join(struct.pack("<Q", d) for d in dm) + struct.pack("<IQ", ty, off)
        sz = exact(dm, ty, sz)
        blob = bytes(rnd.randrange(256) for _ in range(sz))
        pad = (-sz) % align
        data += blob + b"\0" * pad
        off += sz + pad
    head = MAGIC = b"GGUF" + struct.pack("<IQQ", 3, len(ts), len(kvs)) + b"".join(kvs) + infos
    head += b"\0" * ((-len(head)) % align)
    with open(path, "wb") as f:
        f.write(head + data)
    return path

def run(*a):
    r = subprocess.run([sys.executable, os.path.join(HERE, "tools", "stitch_gguf.py"), *a], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr

with tempfile.TemporaryDirectory() as tmp:
    P = lambda n: os.path.join(tmp, n)
    A = make(P("iq3.gguf"), 91, 91, 1)              # type 91 stands in for IQ3_XXS (an unknown id: copied without a size check)
    B = make(P("q2.gguf"), 91, 92, 2)               # stream file: experts in another type (92 stands in for Q2_0)
    # ---- layer scope, --n-cpu-moe 2: layers 0-1 from B, 2-3 from A
    rc, out = run("--resident", A, "--stream", B, "--n-cpu-moe", "2", "--out", P("st.gguf"), "--verify")
    check(rc == 0 and "verify: every tensor equals its source" in out, f"stitch + verify ok ({out.strip().splitlines()[0][:100]})")
    m = json.load(open(P("st.gguf.stitch.json")))
    check(m["layers"] == {"0": "stream", "1": "stream", "2": "resident", "3": "resident"}, f"manifest: layers 0-1 stream, 2-3 resident ({m['layers']})")
    src = {t["name"]: t["from"] for t in m["tensors"]}
    check(src["token_embd.weight"] == "resident" and src["output.weight"] == "resident" and src["blk.1.attn_q.weight"] == "stream" and src["blk.3.ffn_up_exps.weight"] == "resident",
          "manifest: non-block tensors resident, streamed layers' tensors from the stream file")
    a, b, o = S.Gguf(A), S.Gguf(B), S.Gguf(P("st.gguf"))
    check(o.by_name["blk.0.ffn_gate_exps.weight"]["type"] == 92 and o.by_name["blk.2.ffn_gate_exps.weight"]["type"] == 91, "types in the output: Q2_0-like for streamed experts, IQ3-like for resident")
    check(all(o.by_name[n]["dims"] == a.by_name[n]["dims"] for n in a.by_name) and len(o.tensors) == len(a.tensors), "shapes and tensor count unchanged")
    check(all(t["offset"] % o.alignment == 0 for t in o.tensors), "every tensor offset is aligned")
    check(o.meta == a.meta and open(P("st.gguf"), "rb").read(o.kv_end)[a.kv_start:a.kv_end] == open(A, "rb").read(a.kv_end)[a.kv_start:a.kv_end], "metadata bytes are the resident file's, unchanged (token arrays included)")
    check(m["bytes_from"]["stream"] > 0 and m["bytes_from"]["resident"] > 0, "manifest: bytes per source")
    try:
        import gguf_plan
        lay = gguf_plan.model_layout(P("st.gguf"))
        check(lay is not None, "the project's own GGUF reader (planner) reads the stitched file")
    except Exception as e:
        check(False, f"planner reads the stitched file ({e})")
    # ---- experts scope, explicit layers
    rc, out = run("--resident", A, "--stream", B, "--stream-layers", "1,3", "--scope", "experts", "--out", P("ex.gguf"), "--verify")
    m = json.load(open(P("ex.gguf.stitch.json"))); src = {t["name"]: t["from"] for t in m["tensors"]}
    check(rc == 0 and src["blk.1.ffn_down_exps.weight"] == "stream" and src["blk.1.attn_q.weight"] == "resident" and src["blk.0.ffn_down_exps.weight"] == "resident"
          and m["layers"]["1"] == "mixed", "scope experts: only ffn_*_exps of layers 1 and 3 come from the stream file (layer 1 is 'mixed')")
    # ---- dry run, overwrite, bad arguments
    rc, out = run("--resident", A, "--stream", B, "--n-cpu-moe", "1", "--dry-run")
    check(rc == 0 and not os.path.exists(P("None")) and "stitch:" in out, "--dry-run prints the plan and writes nothing")
    rc, out = run("--resident", A, "--stream", B, "--n-cpu-moe", "1", "--out", P("st.gguf"))
    check(rc == 2 and "exists" in out, "an existing output is never overwritten")
    rc, out = run("--resident", A, "--stream", B, "--n-cpu-moe", "1", "--out", A)
    check(rc == 2 and "new file" in out, "--out cannot be one of the inputs")
    rc, out = run("--resident", A, "--stream", B, "--n-cpu-moe", "9", "--out", P("x1.gguf"))
    check(rc == 2 and "does not exist" in out and not os.path.exists(P("x1.gguf")), "a layer beyond the model's block count is refused")
    rc, out = run("--resident", A, "--stream", B, "--out", P("x2.gguf"))
    check(rc == 2 and "exactly one" in out, "streamed layers must be given")
    rc, out = run("--resident", A, "--stream", B, "--stream-layers", "3-1", "--out", P("x3.gguf"))
    check(rc == 2 and "backwards" in out, "a backwards range is refused")
    # ---- mismatches
    C = make(P("c.gguf"), 91, 92, 3, drop="blk.2.attn_q.weight")
    rc, out = run("--resident", A, "--stream", C, "--n-cpu-moe", "1", "--out", P("y1.gguf"))
    check(rc == 2 and "tensor names differ" in out and "blk.2.attn_q.weight" in out and not os.path.exists(P("y1.gguf")), "a missing tensor is named and nothing is written")
    C = make(P("d.gguf"), 91, 92, 3, reshape="blk.1.attn_q.weight")
    rc, out = run("--resident", A, "--stream", C, "--n-cpu-moe", "1", "--out", P("y2.gguf"))
    check(rc == 2 and "shape" in out and "blk.1.attn_q.weight" in out, "a shape mismatch is named")
    C = make(P("e.gguf"), 91, 92, 3, arch="llama")
    rc, out = run("--resident", A, "--stream", C, "--n-cpu-moe", "1", "--out", P("y3.gguf"))
    check(rc == 2 and "architecture" in out, "a different architecture is refused")
    C = make(P("f.gguf"), 91, 92, 3, n_layers=5)
    rc, out = run("--resident", A, "--stream", C, "--n-cpu-moe", "1", "--out", P("y4.gguf"))
    check(rc == 2 and "block_count" in out, "a different block count is refused")
    C = make(P("g.gguf"), 91, 92, 3, align=64)
    rc, out = run("--resident", A, "--stream", C, "--n-cpu-moe", "1", "--out", P("y5.gguf"))
    check(rc == 2 and "alignment" in out, "a different alignment is refused")
    # ---- truncated input, not a GGUF
    F = make(P("f32.gguf"), 0, 0, 4)               # F32 is a known type: its size is checked against the shape
    data = open(F, "rb").read(); open(P("t.gguf"), "wb").write(data[:-30])
    rc, out = run("--resident", F, "--stream", P("t.gguf"), "--n-cpu-moe", "1", "--out", P("y6.gguf"))
    check(rc == 2 and "truncated" in out and not os.path.exists(P("y6.gguf")), f"a truncated file of a known type is refused ({out.strip()[:90]})")
    rc, out = run("--resident", F, "--stream", F, "--n-cpu-moe", "1", "--out", P("y6b.gguf"), "--verify")
    check(rc == 0, "the same F32 file intact stitches fine (exact sizes pass the check)")
    open(P("n.gguf"), "wb").write(b"hello world")
    rc, out = run("--resident", P("n.gguf"), "--stream", B, "--n-cpu-moe", "1", "--out", P("y7.gguf"))
    check(rc == 2 and "not a GGUF" in out, "a file that is not a GGUF is refused")
    # ---- verify catches a flipped byte
    a, b = S.Gguf(A), S.Gguf(B)
    picks = S.plan(a, b, {0, 1}, "layer")
    raw = bytearray(open(P("st.gguf"), "rb").read()); o = S.Gguf(P("st.gguf"))
    raw[o.data_start + o.by_name["blk.0.ffn_up_exps.weight"]["offset"] + 3] ^= 0xFF
    open(P("bad.gguf"), "wb").write(bytes(raw))
    probs = S.verify(P("bad.gguf"), a, b, picks)
    check(len(probs) == 1 and "blk.0.ffn_up_exps.weight" in probs[0], f"verify finds one flipped byte and names the tensor ({probs})")
print("stitch_gguf_test:", "all passed" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
