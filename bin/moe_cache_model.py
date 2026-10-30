"""One model identity for the launcher, the control page, bench-report, saved sessions and usage profiles (XA-1; Linux and Windows; standard library only).

  identity(path)  -> ModelId: canonical first-shard path, every shard and its size, the architecture, layers, experts, the experts' quant and size, and two short
                     hashes: header_fp (the GGUF header and tensor table of every shard: names, shapes, types, offsets, metadata; NOT the weights) and sample_fp
                     (the first 1 MiB and three 64 KiB samples inside the first file: a fine-tune with the same header still differs).
  slot_id(path, ctx, ctk, ctv)  the key of a saved prompt state (warm sessions, --project slots): file name, size, sample_fp, context and KV types. Byte for byte the
                     string the launcher used before this module existed, so states saved earlier are still found.
  profile_path(model, name, cache_dir, dry, say)  where the usage profile of this model lives: the old name (<file>.usage) unless a note next to it
                     (<profile>.id) says the profile was made for a model with another header; then a new file named with that header's hash.
The plugin keeps its own check inside the profile file (a hash of the tensor names and sizes): a profile of another model is ignored with a message there too.
"""
import glob, hashlib, json, os, re, struct, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(HERE, "planner"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "planner")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from gguf_plan import _FMT, _read_str, _skip_value        # noqa: E402

SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
MAX_HEADER = 256 << 20          # a header bigger than this is not a model header

# ggml type ids -> names (ggml.h; 0.25 - 0.28)
TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
              15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
              26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0"}
# (values per block, bytes per block) of those types, to turn a tensor's shape into bytes
TYPE_BLOCKS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
               13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
               22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54),
               35: (256, 66), 39: (32, 17), 40: (64, 36), 41: (128, 18), 42: (64, 18)}
# general.file_type (llama_ftype) -> the usual quant names
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
              14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
              23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
              36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE"}


def shard_paths(path):
    """[path of every file of this model]: the shards of name-00001-of-0000N.gguf in order, else [path]. Missing shards are left out (the caller sees fewer files)."""
    m = SHARD_RE.search(path)
    if not m:
        return [path]
    n = int(m.group(2))
    base = path[:m.start()]
    return [p for p in (f"{base}-{i:05d}-of-{m.group(2)}.gguf" for i in range(1, n + 1)) if os.path.isfile(p)]


def parse_header(path):
    """-> dict(version, align, data_start, kv={key: value for the keys we use}, tensors=[(name, dims, type, offset)]). Raises ValueError / OSError."""
    want = {"general.architecture", "general.name", "general.file_type", "general.alignment"}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"{path}: GGUF version {version}")
        n_t, n_kv = struct.unpack("<QQ", f.read(16))
        if n_t > 1 << 20 or n_kv > 1 << 20:
            raise ValueError(f"{path}: implausible header")
        kv = {}
        for _ in range(n_kv):
            key = _read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            if (key in want or key.endswith(".block_count") or key.endswith(".expert_count")) and (t == 8 or t in _FMT):
                kv[key] = _read_str(f) if t == 8 else struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]
            else:
                _skip_value(f, t)
        align = kv.get("general.alignment") if isinstance(kv.get("general.alignment"), int) else 32
        if not (1 <= align <= 1 << 16 and align & (align - 1) == 0):
            raise ValueError(f"{path}: alignment {align}")
        tensors = []
        for _ in range(n_t):
            name = _read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            if nd > 8:
                raise ValueError(f"{path}: tensor {name!r} has {nd} dimensions")
            dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd))) if nd else []
            ty, off = struct.unpack("<IQ", f.read(12))
            tensors.append((name, dims, ty, off))
        data_start = (f.tell() + align - 1) // align * align
    if data_start > MAX_HEADER:
        raise ValueError(f"{path}: header of {data_start} bytes")
    return {"version": version, "align": align, "data_start": data_start, "kv": kv, "tensors": tensors}


def tensor_bytes(dims, ty):
    blck, size = TYPE_BLOCKS.get(ty, (1, 0))
    if not dims or dims[0] % blck:
        return 0
    n = dims[0] // blck * size
    for d in dims[1:]:
        n *= d
    return n


def _hash_prefix(path, n):
    with open(path, "rb") as f:
        return f.read(n)


def sample_hash(path, size=None):
    """sha256 (hex, 16 chars) of the first 1 MiB and of three 64 KiB samples at 50 %, 75 % and the end of the file: the content part of the old slot key."""
    size = os.path.getsize(path) if size is None else size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        for frac in (0.5, 0.75, 1.0):
            f.seek(max(0, min(int(size * frac), size - (64 << 10)) - (32 << 10)))
            h.update(f.read(64 << 10))
    return h.hexdigest()[:16]


def slot_id(model, ctx=None, ctk=None, ctv=None):
    """The key of a saved prompt state: file name (first 40 safe characters), size, a hash of the header region and three samples inside the file, context size and
    KV cache types. Same on Linux and Windows; reads about 1.2 MiB. "" when the file cannot be read (then nothing is saved under it or restored). The same string
    as before the model identity existed: states saved by earlier versions are still found."""
    try:
        size = os.path.getsize(model)
        fp = sample_hash(model, size)
    except OSError:
        return ""
    return "-".join([re.sub(r"[^A-Za-z0-9_]", "_", os.path.basename(model))[:40], str(size), fp, f"c{ctx}", ctk or "f16", ctv or "f16"])


class ModelId:
    """What identifies a model file set. Fields: path (canonical first shard), name, shards [(path, size)], size, arch, title, n_layers, n_expert, expert_bytes,
    expert_quant (the experts' dominant tensor type), file_quant (general.file_type), header_fp, sample_fp, id."""
    def to_dict(self, paths=True):
        d = {"id": self.id, "name": self.name, "size": self.size, "shards": len(self.shards), "arch": self.arch, "layers": self.n_layers,
             "experts": self.n_expert, "expert_bytes": self.expert_bytes, "expert_quant": self.expert_quant, "file_quant": self.file_quant,
             "header_fp": self.header_fp, "sample_fp": self.sample_fp}
        if paths:
            d["path"] = self.path
        return d

    def summary(self):
        """One plain line for reports (no path)."""
        bits = [self.arch or "?"]
        if self.n_layers:
            bits.append(f"{self.n_layers} layers")
        if self.n_expert:
            bits.append(f"{self.n_expert} experts of {self.expert_quant or '?'}")
        bits.append(f"header {self.header_fp[:8]}")
        return ", ".join(bits)


_CACHE = {}


def identity(path):
    """The ModelId of the model whose first (or only) file is `path`. Cached per file (size and modification time). Raises OSError / ValueError for a file that
    is not a readable GGUF."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key in _CACHE:
        return _CACHE[key]
    files = shard_paths(path)
    first = parse_header(files[0])
    h = hashlib.sha256()
    shards = []
    for i, p in enumerate(files):
        hd = first if i == 0 else parse_header(p)
        size = os.path.getsize(p)
        shards.append((p, size))
        h.update(struct.pack("<QQ", size, hd["data_start"]))
        h.update(_hash_prefix(p, hd["data_start"]))        # metadata, tensor names, shapes, types and offsets of this shard
    m = ModelId()
    m.path = os.path.realpath(files[0])
    m.name = re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", os.path.basename(path))
    m.shards = shards
    m.size = sum(s for _, s in shards)
    m.header_fp = h.hexdigest()[:16]
    m.sample_fp = sample_hash(files[0], shards[0][1])
    kv = first["kv"]
    m.arch = kv.get("general.architecture") or ""
    m.title = kv.get("general.name") or ""
    m.file_quant = FILE_TYPES.get(kv.get("general.file_type"), "") if isinstance(kv.get("general.file_type"), int) else ""
    m.n_layers = next((v for k, v in kv.items() if k.endswith(".block_count") and isinstance(v, int)), 0)
    m.n_expert = next((v for k, v in kv.items() if k.endswith(".expert_count") and isinstance(v, int)), 0)
    by_type = {}
    ebytes = 0
    for i, p in enumerate(files):
        for name, dims, ty, _ in (first if i == 0 else parse_header(p))["tensors"]:
            if len(dims) == 3 and dims[2] > 1 and name.startswith("blk.") and "_exps" in name:
                nb = tensor_bytes(dims, ty)
                ebytes += nb
                by_type[ty] = by_type.get(ty, 0) + nb
    m.expert_bytes = ebytes
    m.expert_quant = TYPE_NAMES.get(max(by_type, key=by_type.get), "") if by_type else ""
    m.id = "-".join([re.sub(r"[^A-Za-z0-9_]", "_", m.name)[:40], str(m.size), m.header_fp])
    _CACHE[key] = m
    return m


def file_quant(path):
    """The quant a GGUF header names (general.file_type), "" when it names none or the file cannot be read. For files whose name says nothing (Ollama's blobs)."""
    try:
        return identity(path).file_quant
    except (OSError, ValueError, struct.error):
        return ""


# ---- usage profiles ----

def _note_path(profile):
    return profile + ".id"


def profile_path(model, name, cache_dir, dry=True, say=None):
    """The usage profile file of `model` (and of the profile `name`, "" = the shared one) in cache_dir. The name is the one the launcher always used,
    <model file name>[.<name>].usage, so profiles learned so far keep their warm start. Next to it a note <profile>.id records the header hash of the model
    the profile was learned on (written at the first start, adopting the profile that is there). If the note says another header (a different file with the
    same name: another quant, another revision), that profile is not used: a new one named with the header's hash is started and `say` tells why.
    dry=True writes nothing."""
    base = os.path.join(cache_dir, os.path.basename(model) + "." + (name + "." if name else "") + "usage")
    try:
        ident = identity(model)
    except (OSError, ValueError, struct.error):
        return base
    for path in (base, base[:-len(".usage")] + "." + ident.header_fp[:8] + ".usage"):
        note = _note_path(path)
        try:
            rec = json.load(open(note, encoding="utf-8"))
        except (OSError, ValueError):
            rec = None
        if rec is None:
            if not dry:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    json.dump({"header_fp": ident.header_fp, "id": ident.id, "arch": ident.arch, "size": ident.size, "experts": ident.n_expert}, open(note, "w", encoding="utf-8"))
                except OSError:
                    pass
            return path
        if rec.get("header_fp") == ident.header_fp:
            return path
        if path != base:
            break
        if say:
            say(f"moe-cache: the usage profile {os.path.basename(base)} was learned on another model file with the same name "
                f"({rec.get('size', 0) / 1e9:.1f} GB, header {str(rec.get('header_fp'))[:8]}; this one is {ident.size / 1e9:.1f} GB, header {ident.header_fp[:8]}): "
                f"not used; a new profile is started ({os.path.basename(base[:-len('.usage')] + '.' + ident.header_fp[:8] + '.usage')})", err=True)
    return base[:-len(".usage")] + "." + ident.header_fp[:8] + ".usage"
