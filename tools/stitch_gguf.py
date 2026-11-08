#!/usr/bin/env python3
"""Stitch two quantisations of the SAME model into one GGUF (MODELS-640): the layers that stay resident from one file (for example IQ3_XXS), the layers that
stream from disk from the other (for example Q2_0). Tensor bytes are copied as they are, nothing is re-quantised, so the stitched file is exactly as good as
its parts and a layer is never half one type and half the other (with --scope layer).

  python3 tools/stitch_gguf.py --resident M-IQ3_XXS.gguf --stream M-Q2_0.gguf --n-cpu-moe 24 --out M-stitched.gguf
  python3 tools/stitch_gguf.py --resident A.gguf --stream B.gguf --stream-layers 0-9,20 --scope experts --out C.gguf --verify

  --resident FILE    metadata and every tensor that is not in a streamed layer come from here (token_embd, output, norms, resident layers)
  --stream FILE      the tensors of the streamed layers come from here
  --stream-layers L  which block indices stream: "0-23" or "0-9,20,22-24"
  --n-cpu-moe N      the same as --stream-layers 0-(N-1): llama.cpp's --n-cpu-moe N keeps the experts of the FIRST N layers on the CPU, so those are the ones read from disk
  --scope layer|experts   layer (default): every blk.L.* tensor of a streamed layer; experts: only the ffn_*_exps tensors (attention and norms stay resident)
  --verify           after writing, read the output back and compare every tensor with its source byte for byte
  --dry-run          only check and print the manifest, write nothing

Checks before anything is written: both files have the same tensor names and the same shapes, the same architecture / block / expert counts and the same alignment.
The manifest OUT.stitch.json lists, per tensor, where it came from (file, type, bytes) and, per layer, the source. general.file_type of the output is the
resident file's (llama.cpp uses it only for the printed name); the real types are per tensor. Standard library only; reads and writes in chunks (a model is tens of GB).
Not exact against either source: run the quality gate (docs/QUALITY_GATE.md) before using a stitched file for anything but speed."""
import argparse, hashlib, json, os, re, struct, sys

MAGIC = b"GGUF"
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}     # uint8 int8 uint16 int16 uint32 int32 float32 bool uint64 int64 float64
STRING, ARRAY = 8, 9
CHUNK = 8 << 20


# (elements per block, bytes per block) of the tensor types whose size is checked against the file (a truncated download shows up as a tensor shorter than its shape needs);
# any other type id is copied without that check
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
         14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 30: (1, 2)}


class StitchError(Exception):
    pass


def _u32(f): return struct.unpack("<I", f.read(4))[0]
def _u64(f): return struct.unpack("<Q", f.read(8))[0]


def _skip_value(f, t):
    """Advance past a metadata value of type t."""
    if t in SCALAR:
        f.seek(SCALAR[t], 1)
    elif t == STRING:
        f.seek(_u64(f), 1)
    elif t == ARRAY:
        et, n = _u32(f), _u64(f)
        if et in SCALAR:
            f.seek(SCALAR[et] * n, 1)
        else:
            for _ in range(n):
                _skip_value(f, et)
    else:
        raise StitchError(f"unknown metadata value type {t}")


def _read_value(f, t):
    if t == STRING:
        return f.read(_u64(f)).decode("utf-8", "replace")
    if t in (2, 3):
        return struct.unpack("<H" if t == 2 else "<h", f.read(2))[0]
    if t in (4, 5):
        return struct.unpack("<I" if t == 4 else "<i", f.read(4))[0]
    if t in (10, 11):
        return struct.unpack("<Q" if t == 10 else "<q", f.read(8))[0]
    _skip_value(f, t)
    return None


class Gguf:
    """Header of a GGUF file: raw metadata bytes (copied unchanged), the few values the checks need, and the tensor table."""
    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        with open(path, "rb") as f:
            if f.read(4) != MAGIC:
                raise StitchError(f"{path}: not a GGUF file")
            self.version = _u32(f)
            if self.version not in (2, 3):
                raise StitchError(f"{path}: GGUF version {self.version} is not supported (2 or 3)")
            n_tensors, n_kv = _u64(f), _u64(f)
            self.kv_start = f.tell()
            self.meta = {}
            self.kv_spans = []         # (key, start, end) of each metadata entry, so a writer can drop some (split.* of a multi-file model)
            for _ in range(n_kv):
                start = f.tell()
                key = f.read(_u64(f)).decode("utf-8", "replace")
                t = _u32(f)
                v = _read_value(f, t) if key in ("general.architecture", "general.alignment", "general.name") or key.endswith((".block_count", ".expert_count")) or key.startswith("split.") else _skip_value(f, t)
                if v is not None:
                    self.meta[key] = v
                self.kv_spans.append((key, start, f.tell()))
            self.kv_end = f.tell()
            self.n_kv = n_kv
            self.tensors = []          # dicts in file order
            for _ in range(n_tensors):
                name = f.read(_u64(f)).decode("utf-8", "replace")
                nd = _u32(f)
                dims = [_u64(f) for _ in range(nd)]
                ty, off = _u32(f), _u64(f)
                self.tensors.append({"name": name, "dims": dims, "type": ty, "offset": off})
            info_end = f.tell()
        self.alignment = int(self.meta.get("general.alignment", 32))
        self.data_start = (info_end + self.alignment - 1) // self.alignment * self.alignment
        order = sorted(range(len(self.tensors)), key=lambda i: self.tensors[i]["offset"])
        for k, i in enumerate(order):
            t = self.tensors[i]
            end = self.tensors[order[k + 1]]["offset"] if k + 1 < len(order) else self.size - self.data_start
            t["bytes"] = end - t["offset"]         # includes the padding up to the next tensor
            blk = BLOCK.get(t["type"])
            if blk:
                n_el = 1
                for d in t["dims"]:
                    n_el *= d
                need = (n_el + blk[0] - 1) // blk[0] * blk[1]
                if t["bytes"] < need:
                    raise StitchError(f"{path}: tensor {t['name']} has {t['bytes']} bytes but its shape and type need {need} (truncated file?)")
            if t["bytes"] <= 0 or self.data_start + t["offset"] + t["bytes"] > self.size:
                raise StitchError(f"{path}: tensor {t['name']} lies outside the file (truncated download?)")
        self.by_name = {t["name"]: t for t in self.tensors}
        if len(self.by_name) != len(self.tensors):
            raise StitchError(f"{path}: duplicate tensor names")


def parse_layers(spec):
    out = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise StitchError(f"--stream-layers: cannot read '{part}' (use 0-9,20)")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        if b < a:
            raise StitchError(f"--stream-layers: '{part}' is backwards")
        out.update(range(a, b + 1))
    return out


def layer_of(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def check_pair(a, b):
    """Raise StitchError when the two files are not two quantisations of the same model."""
    for k in ("general.architecture",):
        if a.meta.get(k) != b.meta.get(k):
            raise StitchError(f"{k} differs: {a.meta.get(k)!r} vs {b.meta.get(k)!r}")
    for k in sorted(set(a.meta) | set(b.meta)):
        if k.endswith((".block_count", ".expert_count")) and a.meta.get(k) != b.meta.get(k):
            raise StitchError(f"{k} differs: {a.meta.get(k)} vs {b.meta.get(k)}")
    if a.alignment != b.alignment:
        raise StitchError(f"alignment differs: {a.alignment} vs {b.alignment}")
    na, nb = set(a.by_name), set(b.by_name)
    if na != nb:
        only_a, only_b = sorted(na - nb)[:5], sorted(nb - na)[:5]
        raise StitchError(f"the tensor names differ ({len(na - nb)} only in the resident file, e.g. {only_a}; {len(nb - na)} only in the stream file, e.g. {only_b})")
    bad = [n for n in sorted(na) if a.by_name[n]["dims"] != b.by_name[n]["dims"]]
    if bad:
        raise StitchError(f"{len(bad)} tensor shape(s) differ, e.g. {bad[0]}: {a.by_name[bad[0]]['dims']} vs {b.by_name[bad[0]]['dims']}")


def plan(a, b, layers, scope):
    """-> list of (tensor name, 'resident' | 'stream') in the resident file's order."""
    out = []
    for t in a.tensors:
        L = layer_of(t["name"])
        pick = L is not None and L in layers and (scope == "layer" or "_exps" in t["name"])
        out.append((t["name"], "stream" if pick else "resident"))
    return out


def manifest(a, b, picks, layers, scope, out_path):
    srcs = {"resident": a, "stream": b}
    tensors, per_layer = [], {}
    for name, which in picks:
        t = srcs[which].by_name[name]
        tensors.append({"name": name, "from": which, "type": t["type"], "bytes": t["bytes"]})
        L = layer_of(name)
        if L is not None:
            per_layer.setdefault(L, set()).add(which)
    return {"resident_file": os.path.abspath(a.path), "stream_file": os.path.abspath(b.path), "output": os.path.abspath(out_path), "scope": scope,
            "stream_layers": sorted(layers),
            "layers": {str(L): ("mixed" if len(v) > 1 else next(iter(v))) for L, v in sorted(per_layer.items())},
            "bytes_from": {w: sum(t["bytes"] for t in tensors if t["from"] == w) for w in ("resident", "stream")},
            "tensors": tensors}


def _pad(f, align):
    f.write(b"\0" * ((-f.tell()) % align))


def _copy(src, off, n, dst):
    src.seek(off)
    while n > 0:
        b = src.read(min(CHUNK, n))
        if not b:
            raise StitchError("unexpected end of a source file")
        dst.write(b)
        n -= len(b)


def write(a, b, picks, out_path):
    srcs = {"resident": a, "stream": b}
    offs, cur = {}, 0
    for name, which in picks:
        offs[name] = cur
        n = srcs[which].by_name[name]["bytes"]
        cur += (n + a.alignment - 1) // a.alignment * a.alignment
    with open(a.path, "rb") as fa, open(b.path, "rb") as fb, open(out_path, "wb") as o:
        files = {"resident": fa, "stream": fb}
        o.write(MAGIC + struct.pack("<IQQ", a.version, len(picks), a.n_kv))
        fa.seek(a.kv_start)
        o.write(fa.read(a.kv_end - a.kv_start))
        for name, which in picks:
            t = srcs[which].by_name[name]
            nb = name.encode()
            o.write(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(t["dims"])) + b"".join(struct.pack("<Q", d) for d in t["dims"])
                    + struct.pack("<IQ", t["type"], offs[name]))
        _pad(o, a.alignment)
        for name, which in picks:
            t = srcs[which].by_name[name]
            _copy(files[which], srcs[which].data_start + t["offset"], t["bytes"], o)
            _pad(o, a.alignment)
        o.truncate(o.tell())


def _digest(path, off, n):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        while n > 0:
            c = f.read(min(CHUNK, n))
            h.update(c)
            n -= len(c)
    return h.hexdigest()


def verify(out_path, a, b, picks):
    """Read the output back: same tensor table (name, dims, type) and, per tensor, the same bytes as its source. Returns a list of problems."""
    o = Gguf(out_path)
    srcs = {"resident": a, "stream": b}
    bad = []
    if [t["name"] for t in o.tensors] != [n for n, _ in picks]:
        return ["the output's tensor list differs from the plan"]
    for name, which in picks:
        s, t = srcs[which].by_name[name], o.by_name[name]
        if t["dims"] != s["dims"] or t["type"] != s["type"]:
            bad.append(f"{name}: shape or type differs from its source")
        elif _digest(out_path, o.data_start + t["offset"], s["bytes"]) != _digest(srcs[which].path, srcs[which].data_start + s["offset"], s["bytes"]):
            bad.append(f"{name}: bytes differ from the {which} file")
    if o.meta != {k: v for k, v in a.meta.items() if k in o.meta}:
        bad.append("metadata differs from the resident file")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description="stitch two quantisations of one model into one GGUF (copies tensors, no re-quantising)")
    ap.add_argument("--resident", required=True); ap.add_argument("--stream", required=True)
    ap.add_argument("--stream-layers"); ap.add_argument("--n-cpu-moe", type=int)
    ap.add_argument("--scope", choices=("layer", "experts"), default="layer")
    ap.add_argument("--out"); ap.add_argument("--verify", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        if (args.stream_layers is None) == (args.n_cpu_moe is None):
            raise StitchError("give exactly one of --stream-layers and --n-cpu-moe")
        a, b = Gguf(args.resident), Gguf(args.stream)
        check_pair(a, b)
        layers = set(range(args.n_cpu_moe)) if args.n_cpu_moe is not None else parse_layers(args.stream_layers)
        nl = a.meta.get(next((k for k in a.meta if k.endswith(".block_count")), ""), None)
        if nl is not None and any(L >= nl for L in layers):
            raise StitchError(f"layer {max(layers)} does not exist (the model has {nl} blocks)")
        if not layers:
            raise StitchError("no layer to stream")
        if not args.dry_run and not args.out:
            raise StitchError("--out is required (or use --dry-run)")
        if args.out and os.path.abspath(args.out) in (os.path.abspath(args.resident), os.path.abspath(args.stream)):
            raise StitchError("--out must be a new file")
        picks = plan(a, b, layers, args.scope)
        m = manifest(a, b, picks, layers, args.scope, args.out or "(dry run)")
        n_s = sum(1 for _, w in picks if w == "stream")
        print(f"stitch: {len(picks)} tensors, {n_s} from the stream file ({m['bytes_from']['stream'] / 1e6:.1f} MB), {len(picks) - n_s} from the resident file "
              f"({m['bytes_from']['resident'] / 1e6:.1f} MB); streamed layers: {len(layers)} ({args.scope})")
        if args.dry_run:
            return 0
        if os.path.exists(args.out):
            raise StitchError(f"{args.out} exists; move it away first (nothing is overwritten)")
        write(a, b, picks, args.out)
        with open(args.out + ".stitch.json", "w") as f:
            json.dump(m, f, indent=1)
        print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) and {args.out}.stitch.json")
        if args.verify:
            bad = verify(args.out, a, b, picks)
            if bad:
                print("VERIFY FAILED:\n  " + "\n  ".join(bad[:10]), file=sys.stderr)
                return 1
            print("verify: every tensor equals its source, byte for byte")
        return 0
    except StitchError as e:
        print(f"stitch_gguf: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
