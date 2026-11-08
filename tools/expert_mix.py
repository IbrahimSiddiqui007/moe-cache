#!/usr/bin/env python3
"""Per-expert quality simulation without plugin work (MODELS-759, MODELS-760): mix two quantisations of the SAME model expert by expert and write the
result in a type fine enough that it adds almost no error of its own (Q6_K, or Q8_0). Any llama.cpp then runs the mix with ordinary single-type tensors,
and the repo's KL test (tests/kl_batch.sh) says what the mix costs. Nothing is re-trained and no plugin code is needed.

  python3 tools/expert_mix.py types A.gguf [B.gguf]                    header only: the type of every tensor group (and A against B)
  python3 tools/expert_mix.py mix --a IQ3.gguf --b Q2_0.gguf --fraction 0.5 --usage profiles/qwen4exp-48x512.usage \\
        --libggml /path/to/libggml-base.so --out mix50.gguf            the coldest 50 % of experts come from B, the rest from A
  --fraction 0 writes A through the same pipeline and --fraction 1 writes B (the two controls: their KL must stay within 0.005 of the real files)

Experts are ranked by the usage profile (profiles/*.usage: 'XUSE', layers, experts, then float32 counts), coldest first, ties by (layer, expert). Only the
ffn_*_exps tensors change; every other tensor is copied byte for byte from A. Each expert is dequantised with libggml's own to_float and written with
ggml_quantize_chunk, one expert at a time, so memory stays at a few MB (still run big files under: systemd-run --user --scope -p MemoryMax=20G).
Header reading and the tensor table come from tools/stitch_gguf.py. Output is a NEW file; an existing file is never overwritten."""
import argparse, concurrent.futures, contextlib, ctypes, json, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stitch_gguf as sg  # noqa: E402

NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS",
         18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 30: "BF16", 42: "Q2_0"}
Q6_K, Q8_0 = 14, 8
FINE = {"q6_k": Q6_K, "q8_0": Q8_0}
EXPERT_RE = __import__("re").compile(r"^blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight$")


class MixError(Exception):
    pass


def type_name(t):
    return NAMES.get(t, f"type{t}")


# ---- codecs: dequantise a slice to float32, quantise float32 rows to the fine type ----

class GgmlCodec:
    """libggml through ctypes (the build that knows both input types; Q2_0 needs ggml 0.28 / llama.cpp b12064 or newer; the kit now bundles b12207)."""
    class Traits(ctypes.Structure):
        _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_interleave", ctypes.c_int64), ("type_size", ctypes.c_size_t),
                    ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p), ("from_float_ref", ctypes.c_void_p)]

    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        self.lib.ggml_get_type_traits.restype = ctypes.POINTER(self.Traits)
        self.lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self.lib.ggml_row_size.restype = ctypes.c_size_t
        self.lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.to_float_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)

    def row_size(self, ty, ne0):
        return int(self.lib.ggml_row_size(ty, ne0))

    def dequant(self, ty, raw, n):
        tr = self.lib.ggml_get_type_traits(ty)
        if not tr or not tr.contents.to_float:
            raise MixError(f"this libggml cannot dequantise type {type_name(ty)} ({ty}): use a build that knows it")
        out = np.empty(n, dtype=np.float32)
        buf = ctypes.create_string_buffer(bytes(raw), len(raw))
        self.to_float_t(tr.contents.to_float)(ctypes.cast(buf, ctypes.c_void_p), out.ctypes.data, n)
        return out

    def quant(self, ty, floats, nrows, ne0):
        out = ctypes.create_string_buffer(self.row_size(ty, ne0) * nrows)
        arr = np.ascontiguousarray(floats, dtype=np.float32)
        n = self.lib.ggml_quantize_chunk(ty, arr.ctypes.data, out, 0, nrows, ne0, None)
        if n != len(out):
            raise MixError(f"ggml_quantize_chunk wrote {n} bytes, expected {len(out)}")
        return out.raw


class PyCodec:
    """Pure numpy for F32, F16 and Q8_0 (dequantise and quantise) and Q4_0 (dequantise only): what the unit test uses, no libggml needed."""
    SIZE = {0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34)}

    def row_size(self, ty, ne0):
        if ty not in self.SIZE:
            raise MixError(f"PyCodec does not know type {type_name(ty)}")
        blk, by = self.SIZE[ty]
        return ne0 // blk * by

    def dequant(self, ty, raw, n):
        raw = bytes(raw)
        if ty == 0:
            return np.frombuffer(raw, dtype=np.float32, count=n).copy()
        if ty == 1:
            return np.frombuffer(raw, dtype=np.float16, count=n).astype(np.float32)
        if ty == 8:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
            d = b[:, :2].copy().view(np.float16).astype(np.float32)
            return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)[:n]
        if ty == 2:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
            d = b[:, :2].copy().view(np.float16).astype(np.float32)
            q = b[:, 2:]
            lo, hi = (q & 15).astype(np.float32) - 8, (q >> 4).astype(np.float32) - 8
            return (np.concatenate([lo, hi], axis=1) * d).reshape(-1)[:n]
        raise MixError(f"PyCodec cannot dequantise {type_name(ty)}")

    def quant(self, ty, floats, nrows, ne0):
        if ty != 8:
            raise MixError("PyCodec quantises only Q8_0")
        x = np.asarray(floats, dtype=np.float32).reshape(-1, 32)
        amax = np.abs(x).max(axis=1)
        d = (amax / 127.0).astype(np.float16)
        dd = d.astype(np.float32)
        q = np.where(dd[:, None] == 0, 0, np.rint(x / np.where(dd == 0, 1, dd)[:, None])).astype(np.int8)
        out = np.empty((x.shape[0], 34), dtype=np.uint8)
        out[:, :2] = d.view(np.uint8).reshape(-1, 2)
        out[:, 2:] = q.view(np.uint8)
        return out.tobytes()


# ---- usage ranking ----

def load_usage(path):
    data = open(path, "rb").read()
    if data[:4] != b"XUSE":
        raise MixError(f"{path}: not an XUSE usage profile")
    n_l, n_e = struct.unpack("<ii", data[4:12])
    if len(data) < 12 + 4 * n_l * n_e:
        raise MixError(f"{path}: shorter than {n_l} x {n_e} counts")
    return n_l, n_e, np.frombuffer(data, dtype="<f4", count=n_l * n_e, offset=12).reshape(n_l, n_e)


def pick_from_b(usage, fraction):
    """The coldest round(fraction x N) experts as a set of (layer, expert); ties go by (layer, expert) so a run can be repeated."""
    n_l, n_e = usage.shape
    order = sorted(((float(usage[l, e]), l, e) for l in range(n_l) for e in range(n_e)))
    k = int(round(fraction * len(order)))
    return {(l, e) for _, l, e in order[:k]}


class Model:
    """A model as one tensor table, also when it is split into several files (NAME-00001-of-0000N.gguf): give the first file, the others are found next to it.
    Each tensor dict carries the file it lives in ("shard"). The output of a mix is ONE file, so the split.* metadata is left out when writing."""
    SPLIT = __import__("re").compile(r"-(\d{5})-of-(\d{5})\.gguf$")

    def __init__(self, path):
        first = sg.Gguf(path)
        n = int(first.meta.get("split.count", 1))
        paths = [path]
        if n > 1:
            m = self.SPLIT.search(path)
            if not m or int(m.group(1)) != 1 or int(m.group(2)) != n:
                raise MixError(f"{path}: the file says it is part 1 of {n}; give the first file, named NAME-00001-of-{n:05d}.gguf")
            paths = [path[:m.start()] + f"-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
            missing = [q for q in paths if not os.path.isfile(q)]
            if missing:
                raise MixError(f"{path}: part(s) missing: {', '.join(missing)}")
        self.shards = [first] + [sg.Gguf(q) for q in paths[1:]]
        self.path, self.meta, self.version, self.alignment = path, first.meta, first.version, first.alignment
        self.size = sum(g.size for g in self.shards)
        self.tensors = [dict(t, shard=g) for g in self.shards for t in g.tensors]
        self.by_name = {t["name"]: t for t in self.tensors}
        if len(self.by_name) != len(self.tensors):
            raise MixError(f"{path}: duplicate tensor names across the parts")
        want = first.meta.get("split.tensors.count")
        if want is not None and int(want) != len(self.tensors):
            raise MixError(f"{path}: the parts hold {len(self.tensors)} tensors but the header says {want}")

    def kv_without_split(self, f):
        """-> (number of entries, raw bytes) of the first file's metadata without split.* entries."""
        g, out, n = self.shards[0], b"", 0
        for key, start, end in g.kv_spans:
            if not key.startswith("split."):
                f.seek(start)
                out += f.read(end - start)
                n += 1
        return n, out


# ---- the two commands ----

def expert_tensors(g):
    return {t["name"]: (int(m.group(1)),) for t in g.tensors for m in [EXPERT_RE.match(t["name"])] if m and len(t["dims"]) == 3}


def cmd_types(args):
    files = [Model(p) for p in args.files]
    for g in files:
        groups = {}
        for t in g.tensors:
            kind = "experts" if EXPERT_RE.match(t["name"]) else "other"
            groups.setdefault((kind, type_name(t["type"])), [0, 0])
            groups[(kind, type_name(t["type"]))][0] += 1
            groups[(kind, type_name(t["type"]))][1] += t["bytes"]
        print(f"{g.path}: {len(g.tensors)} tensors in {len(g.shards)} file(s), {g.size / 1e9:.2f} GB, architecture {g.meta.get('general.architecture')}")
        for (kind, ty), (n, by) in sorted(groups.items()):
            print(f"  {kind:8s} {ty:8s} {n:5d} tensors {by / 1e9:8.2f} GB")
    if len(files) == 2:
        sg.check_pair(files[0], files[1])
        diff = sum(1 for n in files[0].by_name if EXPERT_RE.match(n) and files[0].by_name[n]["type"] != files[1].by_name[n]["type"])
        same = sum(1 for n in files[0].by_name if EXPERT_RE.match(n)) - diff
        print(f"the two files have the same tensor names and shapes; expert tensors with another type in B: {diff}, with the same type: {same}")
    return 0


def mix_plan(a, b, fine):
    """-> list of dicts per tensor of A in order: name, dims, out type, out bytes (experts are rewritten as `fine`)."""
    out = []
    for t in a.tensors:
        m = EXPERT_RE.match(t["name"])
        if m and len(t["dims"]) == 3:
            ne0, ne1, ne = t["dims"]
            out.append({"name": t["name"], "dims": t["dims"], "layer": int(m.group(1)), "expert": True, "type": fine, "src": t})
        else:
            out.append({"name": t["name"], "dims": t["dims"], "expert": False, "type": t["type"], "src": t, "bytes": t["bytes"]})
    return out


def cmd_mix(args):
    fine = FINE[args.fine]
    codec = GgmlCodec(args.libggml) if args.libggml else PyCodec()
    a, b = Model(args.a), Model(args.b)
    sg.check_pair(a, b)
    if not 0.0 <= args.fraction <= 1.0:
        raise MixError("--fraction is between 0 and 1")
    if os.path.exists(args.out) and not args.dry_run:
        raise MixError(f"{args.out} exists; move it away first (nothing is overwritten)")
    n_l, n_e, usage = load_usage(args.usage)
    chosen = pick_from_b(usage, args.fraction)
    plan = mix_plan(a, b, fine)
    small_tensors = []
    for p in plan:
        if p["expert"]:
            ne0, ne1, ne = p["dims"]
            if ne != n_e:
                raise MixError(f"{p['name']}: {ne} experts but the profile has {n_e}")
            if ne0 % (256 if fine == Q6_K else 32):
                if ne0 % 32:
                    raise MixError(f"{p['name']}: {ne0} columns do not fit the {args.fine} block size (nor Q8_0's 32)")
                p["type"] = Q8_0       # Q6_K needs 256 columns (the down tensors of the Qwen3.8 family have 640): Q8_0 is finer still
                small_tensors.append(p["name"])
            p["row"] = codec.row_size(p["type"], ne0)
            p["bytes"] = ne * ne1 * p["row"]
    offs, cur = [], 0
    for p in plan:
        offs.append(cur)
        cur += (p["bytes"] + a.alignment - 1) // a.alignment * a.alignment
    taken = {"a": 0, "b": 0}
    if args.dry_run:
        print(f"expert_mix dry run: {len(plan)} tensors, output {cur / 1e9:.2f} GB, {sum(1 for p in plan if p['expert'])} expert tensors "
              f"({len(small_tensors)} would use Q8_0 because their rows are not a multiple of 256 columns), {len(chosen)} of {n_l * n_e} experts from B; nothing written")
        return 0
    with contextlib.ExitStack() as stack, open(args.out, "wb") as o:
        handles = {}
        def fh(g):                           # one open handle per part file
            if g.path not in handles:
                handles[g.path] = stack.enter_context(open(g.path, "rb"))
            return handles[g.path]
        n_kv, kv = a.kv_without_split(fh(a.shards[0]))
        o.write(sg.MAGIC + struct.pack("<IQQ", a.version, len(plan), n_kv))
        o.write(kv)
        for p, off in zip(plan, offs):
            nb = p["name"].encode()
            o.write(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(p["dims"])) + b"".join(struct.pack("<Q", d) for d in p["dims"]) + struct.pack("<IQ", p["type"], off))
        sg._pad(o, a.alignment)
        for p in plan:
            if not p["expert"]:
                sh = p["src"]["shard"]
                sg._copy(fh(sh), sh.data_start + p["src"]["offset"], p["bytes"], o)
            else:
                ne0, ne1, ne = p["dims"]
                raws = []
                for e in range(ne):                  # one tensor at a time: its slices are read in order, converted on several threads, written in order
                    from_b = (p["layer"], e) in chosen
                    g = b if from_b else a
                    t = g.by_name[p["name"]]
                    f = fh(t["shard"])
                    row_in = codec.row_size(t["type"], ne0)
                    f.seek(t["shard"].data_start + t["offset"] + e * ne1 * row_in)
                    raw = f.read(ne1 * row_in)
                    if len(raw) != ne1 * row_in:
                        raise MixError(f"{g.path}: short read in {p['name']} expert {e}")
                    raws.append((t["type"], raw))
                    taken["b" if from_b else "a"] += 1
                conv = lambda tr: codec.quant(p["type"], codec.dequant(tr[0], tr[1], ne0 * ne1), ne1, ne0)
                if args.threads > 1:
                    with concurrent.futures.ThreadPoolExecutor(args.threads) as ex:
                        for blob in ex.map(conv, raws):
                            o.write(blob)
                else:
                    for tr in raws:
                        o.write(conv(tr))
            sg._pad(o, a.alignment)
        o.truncate(o.tell())
    man = {"a": os.path.abspath(a.path), "b": os.path.abspath(b.path), "usage": os.path.abspath(args.usage), "fraction": args.fraction, "fine_type": args.fine,
           "experts_from_a": taken["a"], "experts_from_b": taken["b"], "q8_0_fallback_tensors": len(small_tensors), "out": os.path.abspath(args.out), "bytes": os.path.getsize(args.out)}
    with open(args.out + ".mix.json", "w") as f:
        json.dump(man, f, indent=1)
    print(f"wrote {args.out} ({man['bytes'] / 1e9:.2f} GB): {taken['a']} expert slices from A, {taken['b']} from B (fraction {args.fraction}), type {args.fine}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="mix two quantisations of one model expert by expert (quality simulation, no plugin)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("types", help="header-only listing of tensor types")
    t.add_argument("files", nargs="+")
    m = sub.add_parser("mix", help="write the dequantised mix")
    m.add_argument("--a", required=True, help="the file for the experts that stay (for example IQ3_XXS)")
    m.add_argument("--b", required=True, help="the file for the coldest experts (for example Q2_0)")
    m.add_argument("--fraction", type=float, required=True)
    m.add_argument("--usage", required=True, help="profiles/*.usage of the model family (counts per layer and expert)")
    m.add_argument("--libggml", help="libggml-base.so / ggml-base.dll that knows both types (default: numpy codec, Q8_0/F16/F32/Q4_0 only)")
    m.add_argument("--fine", choices=sorted(FINE), default="q6_k", help="q6_k falls back to q8_0 for tensors whose rows are not a multiple of 256 columns")
    m.add_argument("--threads", type=int, default=4, help="threads converting the slices of one tensor (libggml releases the GIL; default 4)")
    m.add_argument("--dry-run", action="store_true", help="check both headers and print the size of the output, write nothing")
    m.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        if len(args.files if args.cmd == "types" else []) > 2:
            raise MixError("types takes one or two files")
        return cmd_types(args) if args.cmd == "types" else cmd_mix(args)
    except (MixError, sg.StitchError) as e:
        print(f"expert_mix: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
