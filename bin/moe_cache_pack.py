"""moe_cache.py pack MODEL.gguf [--out PATH] [--verify] [--force]

Writes MODEL.moepack next to the model: for every (layer, expert) ONE contiguous, 4 KiB-aligned block that holds that expert's gate, up and down slices
(every 3-D `blk.N.*` tensor of the layer, in GGUF order). The plugin then reads a missing expert with one large unbuffered read instead of three
0.9 MB ones (PC SSD, RESULTS 73: 896 KiB reads 1.90 GB/s, 2688 KiB reads 3.47 GB/s). The GGUF is never touched, so plain llama.cpp keeps working, and the
bytes in the pack are copies of the GGUF's bytes, so the tokens are the same.

File layout (little endian; format version 1; src/mc_state_pack.inl reads it):
  0     char[8]  "MOEPACK1"
  8     u32      version (1)
  12    u32      number of layer records
  16    u32      number of model files (shards)
  20    u32      header_bytes (header + layer records, rounded up to 4096; the first block starts at or after it)
  24    u32      CRC-32 of the sampled source bytes (see below)
  28    u32      reserved
  32    u64[8]   size of each model file (0 for unused)
  96    char[160] base name of the first model file (informational)
  256   layer records, 512 bytes each, sorted by layer:
          u32 layer, u32 n_expert, u32 n_tensors (1..4), u32 reserved, u64 base (4096-aligned), u64 stride (4096-aligned, one block),
          4 x { char name[96]; u64 nb2 (bytes of one expert slice); u64 in_off (offset of the slice inside the block); u64 src_off (offset in its model file) }
  block e of a layer starts at base + e * stride; the slice of tensor k is at block + in_off[k], nb2[k] bytes, the same bytes as the GGUF's expert e of that tensor.
Source check: the plugin compares every tensor's name, file offset and size with the model it loaded, the file sizes, and a CRC-32 over 16 samples of 4096
bytes per file (offsets round_down(size * k / 17, 4096), k = 1..16, the last page clamped to the file) - a pack of another quantisation of the same model is refused.
"""
import argparse, json, os, shutil, struct, sys, time, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "planner"), os.path.join(HERE, "..", "planner")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
from gguf_plan import _FMT, _read_str, _skip_value, shards          # noqa: E402

MAGIC = b"MOEPACK1"
VERSION = 1
PAGE = 4096
REC = 512
MAX_T = 4
NAME_LEN = 96
SAMPLES = 16

# type id -> (values per block, bytes per block): ggml 0.25 - 0.28, checked against ggml-common.h static asserts
TYPES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
         13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
         22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54),
         35: (256, 66), 39: (32, 17), 40: (64, 36), 41: (128, 18), 42: (64, 18)}


def round_up(x, a=PAGE):
    return (x + a - 1) // a * a


def read_header(path):
    """-> (align, data_start, [(name, dims, type, offset)])"""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        ver = struct.unpack("<I", f.read(4))[0]
        if ver not in (2, 3):
            raise SystemExit(f"{path}: GGUF version {ver} is not supported")
        n_t, n_kv = struct.unpack("<QQ", f.read(16))
        align = 32
        for _ in range(n_kv):
            key = _read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            if key == "general.alignment" and t in (4, 5):
                align = struct.unpack(_FMT[t], f.read(4))[0]
            else:
                _skip_value(f, t)
        infos = []
        for _ in range(n_t):
            name = _read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            if nd > 8:
                raise SystemExit(f"{path}: tensor {name!r} has {nd} dimensions")
            dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd))) if nd else []
            ty, off = struct.unpack("<IQ", f.read(12))
            infos.append((name, dims, ty, off))
        data_start = round_up(f.tell(), align)
    return align, data_start, infos


def nbytes(dims, ty):
    if ty not in TYPES:
        raise SystemExit(f"unknown tensor type {ty}: update TYPES in moe_cache_pack.py")
    blck, size = TYPES[ty]
    if not dims or dims[0] % blck:
        raise SystemExit(f"row of {dims[:1]} values does not fit blocks of {blck} (type {ty})")
    n = dims[0] // blck * size
    for d in dims[1:]:
        n *= d
    return n


def sample_offsets(size):
    last = max(0, (size - PAGE) // PAGE * PAGE)
    return [min(size * k // (SAMPLES + 1) // PAGE * PAGE, last) for k in range(1, SAMPLES + 1)]


def source_crc(paths):
    crc = 0
    for p in paths:
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            for off in sample_offsets(size):
                f.seek(off)
                crc = zlib.crc32(f.read(PAGE), crc)
    return crc & 0xffffffff


def plan(model):
    """-> (files, layers) with layers = {layer: dict(n_expert, tensors=[(name, fi, src_off, nb2)])} for every 3-D blk.N.* tensor with more than one expert"""
    files = shards(model)
    if not files or not all(os.path.isfile(p) for p in files):
        raise SystemExit(f"model file(s) not found: {files or model}")
    layers = {}
    for fi, p in enumerate(files):
        align, data_start, infos = read_header(p)
        order = sorted(infos, key=lambda x: x[3])
        size = os.path.getsize(p)
        for i, (name, dims, ty, off) in enumerate(order):
            n = nbytes(dims, ty)
            end = order[i + 1][3] if i + 1 < len(order) else size - data_start
            if not (end - off - align < n <= end - off):     # the gap to the next tensor is the size rounded up to the alignment
                raise SystemExit(f"{p}: size of {name} ({n}) does not match its place in the file ({end - off}); type table out of date?")
            parts = name.split(".")
            if len(dims) != 3 or dims[2] <= 1 or len(parts) < 3 or parts[0] != "blk" or not parts[1].isdigit():
                continue
            if len(name.encode()) >= NAME_LEN:
                raise SystemExit(f"tensor name too long for the pack format: {name}")
            L = layers.setdefault(int(parts[1]), {"n_expert": dims[2], "tensors": []})
            if L["n_expert"] != dims[2]:
                raise SystemExit(f"layer {parts[1]}: tensors disagree on the expert count")
            if n % dims[2]:
                raise SystemExit(f"{name}: size {n} is not a multiple of the expert count {dims[2]}")
            L["tensors"].append((name, fi, data_start + off, n // dims[2], off))
    for l, L in list(layers.items()):
        L["tensors"].sort(key=lambda t: (t[1], t[4]))
        if len(L["tensors"]) < 2:
            del layers[l]               # one tensor per expert: nothing to join
        elif len(L["tensors"]) > MAX_T:
            raise SystemExit(f"layer {l}: {len(L['tensors'])} expert tensors, the format holds {MAX_T}")
    return files, layers


def layout(layers, header_bytes):
    """block offsets: -> {layer: (base, stride, [in_off...])}"""
    out, pos = {}, header_bytes
    for l in sorted(layers):
        L = layers[l]
        offs, o = [], 0
        for t in L["tensors"]:
            offs.append(o)
            o += t[3]
        stride = round_up(o)
        out[l] = (pos, stride, offs)
        pos += stride * L["n_expert"]
    return out, pos


def build_header(files, layers, lay, total, crc, model):
    hb = round_up(256 + REC * len(layers))
    h = bytearray(hb)
    sizes = [os.path.getsize(p) for p in files] + [0] * (8 - len(files))
    struct.pack_into("<8sIIIIII8Q", h, 0, MAGIC, VERSION, len(layers), len(files), hb, crc, 0, *sizes)
    h[96:96 + 160] = os.path.basename(model).encode()[:159].ljust(160, b"\0")
    for i, l in enumerate(sorted(layers)):
        L = layers[l]
        base, stride, offs = lay[l]
        o = 256 + i * REC
        struct.pack_into("<IIIIQQ", h, o, l, L["n_expert"], len(L["tensors"]), 0, base, stride)
        for k, t in enumerate(L["tensors"]):
            struct.pack_into("<96sQQQ", h, o + 32 + k * 120, t[0].encode().ljust(NAME_LEN, b"\0"), t[3], offs[k], t[2])
    return bytes(h)


def default_out(model):
    return (model[:-5] if model.endswith(".gguf") else model) + ".moepack"


def pack_main(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py pack", description="write MODEL.moepack: one contiguous block per (layer, expert) so the plugin reads a missing expert with one large unbuffered read")
    ap.add_argument("model")
    ap.add_argument("--out", help="pack file (default: next to the model, MODEL.moepack)")
    ap.add_argument("--verify", action="store_true", help="do not write; compare an existing pack with the model byte by byte")
    ap.add_argument("--force", action="store_true", help="rewrite a finished pack, ignore a partial one")
    ap.add_argument("--chunk-mib", type=int, default=256, help="read this much of one tensor at a time (default 256)")
    a = ap.parse_args(argv)
    model = a.model
    out = a.out or default_out(model)
    files, layers = plan(model)
    if not layers:
        raise SystemExit("moe-cache pack: this model has no layer with two or more 3-D expert tensors: nothing to pack")
    header_bytes = round_up(256 + REC * len(layers))
    lay, total = layout(layers, header_bytes)
    crc = source_crc(files)
    header = build_header(files, layers, lay, total, crc, model)
    n_exp = sum(L["n_expert"] for L in layers.values())
    gib = total / 2 ** 30
    if a.verify:
        return verify(files, layers, lay, header, out)
    if os.path.exists(out) and not a.force:
        with open(out, "rb") as f:
            same = f.read(len(header)) == header and os.path.getsize(out) == total
        if same:
            print(f"moe-cache pack: {out} is already complete for this model ({gib:.1f} GiB); --verify checks every byte, --force rewrites")
            return 0
        raise SystemExit(f"moe-cache pack: {out} exists but belongs to another file or is damaged; delete it or use --force")
    out_dir = os.path.dirname(os.path.abspath(out))
    free = shutil.disk_usage(out_dir).free
    part, prog = out + ".partial", out + ".partial.progress"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if free + have < total + (1 << 30):
        raise SystemExit(f"moe-cache pack: needs {gib:.1f} GiB (+1 GiB spare) in {out_dir}, only {free / 2 ** 30:.1f} GiB free")
    done = set()
    sig = f"{VERSION} {crc} {total} {len(layers)}"
    if os.path.exists(prog) and os.path.exists(part) and not a.force:
        lines = open(prog).read().split("\n")
        if lines and lines[0] == sig:
            done = {int(x) for x in lines[1:] if x.strip().isdigit()}
            print(f"moe-cache pack: resuming {part}: {len(done)} of {len(layers)} layers already written")
    if not done:
        with open(prog, "w") as f:
            f.write(sig + "\n")
        open(part, "wb").close()
    srcs = [open(p, "rb", buffering=0) for p in files]
    chunk = max(1, a.chunk_mib)
    t0, written = time.time(), 0
    try:
        with open(part, "r+b") as dst:
            for n_done, l in enumerate(sorted(layers)):
                L = layers[l]
                base, stride, offs = lay[l]
                if l in done:
                    continue
                per = max(1, (chunk << 20) // max(stride, 1))
                for e0 in range(0, L["n_expert"], per):
                    ne = min(per, L["n_expert"] - e0)
                    buf = bytearray(stride * ne)
                    mv = memoryview(buf)
                    for k, (name, fi, src_off, nb2, _) in enumerate(L["tensors"]):
                        srcs[fi].seek(src_off + e0 * nb2)
                        slab = bytearray(nb2 * ne)
                        got = srcs[fi].readinto(slab)
                        if got != len(slab):
                            raise SystemExit(f"moe-cache pack: short read of {name} in {files[fi]}")
                        for j in range(ne):
                            mv[j * stride + offs[k]:j * stride + offs[k] + nb2] = slab[j * nb2:(j + 1) * nb2]
                    dst.seek(base + e0 * stride)
                    dst.write(buf)
                    written += len(buf)
                dst.flush()
                os.fsync(dst.fileno())
                with open(prog, "a") as f:
                    f.write(f"{l}\n")
                el = time.time() - t0
                frac = (n_done + 1) / len(layers)
                print(f"moe-cache pack: layer {l} done ({100 * frac:.0f} %), {written / 2 ** 30:.1f} GiB written, {written / max(el, 1e-9) / 1e6:.0f} MB/s, "
                      f"about {el * (1 - frac) / max(frac, 1e-9):.0f} s left", flush=True)
            dst.seek(0)
            dst.write(header)
            dst.truncate(total)
            dst.flush()
            os.fsync(dst.fileno())
    finally:
        for s in srcs:
            s.close()
    os.replace(part, out)
    try:
        os.remove(prog)
    except OSError:
        pass
    print(f"moe-cache pack: wrote {out}: {n_exp} experts, {gib:.1f} GiB, {time.time() - t0:.0f} s. Experimental: the plugin uses it only when asked: "
          f"MOE_CACHE_PACK=1 (this file next to the model) or MOE_CACHE_PACK=PATH.")
    return 0


def verify(files, layers, lay, header, out):
    if not os.path.exists(out):
        raise SystemExit(f"{out} does not exist")
    srcs = [open(p, "rb") for p in files]
    bad = 0
    with open(out, "rb") as f:
        if f.read(len(header)) != header:
            raise SystemExit(f"{out}: header differs from the one this model gives")
        for l in sorted(layers):
            L = layers[l]
            base, stride, offs = lay[l]
            for e in range(L["n_expert"]):
                f.seek(base + e * stride)
                blk = f.read(stride)
                for k, (name, fi, src_off, nb2, _) in enumerate(L["tensors"]):
                    srcs[fi].seek(src_off + e * nb2)
                    if blk[offs[k]:offs[k] + nb2] != srcs[fi].read(nb2):
                        bad += 1
                        if bad <= 5:
                            print(f"DIFFERENT: layer {l} expert {e} tensor {name}")
            print(f"moe-cache pack: verified layer {l}", flush=True)
    if bad:
        raise SystemExit(f"moe-cache pack: {bad} slices differ")
    print("moe-cache pack: every slice equals the model's bytes")
    return 0


if __name__ == "__main__":
    sys.exit(pack_main(sys.argv[1:]))
