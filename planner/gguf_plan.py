"""Minimal GGUF header reader (standard library only) and GPU placement planner.
   fit_n_cpu_moe(path, ctx) -> how many layers keep their experts on the CPU so that the rest fill the free VRAM."""
import glob, json, os, re, struct, subprocess, threading

def _win_ps():
    """powershell by its System32 path on Windows (a bare name searches the current folder first, AUD-034); the bare name elsewhere."""
    if os.name != "nt":
        return "powershell"
    p = os.path.join(os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    return p if os.path.isfile(p) else "powershell"


_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


MAX_STR = 1 << 24      # longest key or string value taken from a header (tokenizer entries are tiny; a hostile length must not allocate the file)


def _left(f):
    """Bytes from the current position to the end of the file."""
    return os.fstat(f.fileno()).st_size - f.tell()


def _skip(f, n):
    """Skip n bytes of the file without reading them (a hostile length field must not make us load a model-sized buffer)."""
    if n < 0 or n > _left(f):
        raise ValueError("GGUF header: length past the end of the file")
    f.seek(n, 1)


def _read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    if n > MAX_STR or n > _left(f):
        raise ValueError(f"GGUF header: string of {n} bytes")
    return f.read(n).decode("utf-8", "replace")


def _skip_str_array(f, n):
    """Skip n length-prefixed strings (the tokenizer: ~150k entries) by walking the lengths in 1 MiB blocks: one read per block, not one read and seek per entry
    (LAUNCHER-1186: about 4 s per start on the laptop 35B per the 2026-11-16 laptop measurement, one machine). The file size is taken once (one fstat per entry made a dry run take 16 s on Windows)."""
    size = os.fstat(f.fileno()).st_size
    pos = f.tell()
    if n * 8 > size - pos:
        raise ValueError("GGUF header: array past the end of the file")
    buf, base = b"", pos
    for _ in range(n):
        if pos + 8 - base > len(buf):
            f.seek(pos)
            buf, base = f.read(1 << 20), pos
            if len(buf) < 8:
                raise ValueError("GGUF header: array past the end of the file")
        pos += 8 + struct.unpack_from("<Q", buf, pos - base)[0]
        if pos > size:
            raise ValueError("GGUF header: length past the end of the file")
    f.seek(pos)


def _skip_value(f, t):
    if t in _FMT:
        _skip(f, struct.calcsize(_FMT[t]))
    elif t == 8:
        n = struct.unpack("<Q", f.read(8))[0]
        _skip(f, n)
    elif t == 9:
        it = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if it in _FMT:
            _skip(f, n * struct.calcsize(_FMT[it]))
        elif it == 8:
            _skip_str_array(f, n)
        else:          # arrays of arrays: no GGUF writer makes them, and they would recurse as deep as the file says
            raise ValueError(f"GGUF header: array of type {it}")
    else:
        raise ValueError(f"unknown GGUF value type {t}")


_KV = {}


def _kv_index(path):
    """-> (n_tensors, [(key, type, value offset)], offset after the key/value block) or None when the file is not GGUF. The header is walked ONCE per file
    (kept by path, size and modification time; a file rewritten in place to the same size with the same mtime would be served stale) and the readers below seek to the values they want: model_key, sampling_meta, mtp_layers, read_meta and
    read_tensors used to walk the tokenizer arrays one pass each (LAUNCHER-1186). Hostile headers raise ValueError exactly as before (nothing is cached then)."""
    st = os.stat(path)
    k = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if k not in _KV:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                _KV[k] = None
            else:
                f.read(4)
                n_t, n_kv = struct.unpack("<QQ", f.read(16))
                ents = []
                for _ in range(n_kv):
                    key = _read_str(f)
                    t = struct.unpack("<I", f.read(4))[0]
                    ents.append((key, t, f.tell()))
                    _skip_value(f, t)
                _KV[k] = (n_t, ents, f.tell())
    return _KV[k]


def model_key(path):
    """'<architecture>-<layers>x<experts>' from the GGUF header (e.g. 'qwen4exp-48x512'): names the shipped starter usage profile for a model family,
    shared by all quants of it; None when the header lacks these keys."""
    meta = {}
    idx = _kv_index(path)
    if idx is None:
        return None
    with open(path, "rb") as f:
        for key, t, off in idx[1]:
            f.seek(off)
            if key == "general.architecture" and t == 8:
                meta["arch"] = _read_str(f)
                if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", meta["arch"]) or ".." in meta["arch"]:
                    return None        # it becomes part of a file name (profiles/<key>.usage): no / \\ or ..
            elif (key.endswith(".block_count") or key.endswith(".expert_count")) and t in (4, 5, 10, 11):
                meta[key.rsplit(".", 1)[1]] = struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]
    if not all(k in meta for k in ("arch", "block_count", "expert_count")):
        return None
    return f"{meta['arch']}-{meta['block_count']}x{meta['expert_count']}"


_TENSORS = {}


def read_tensors(path):
    """-> [(name, size_bytes)] for the tensors stored in this one file (kept per file, size and modification time: a launch reads it several times)."""
    st = os.stat(path)
    k = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if k not in _TENSORS:
        _TENSORS[k] = _read_tensors(path)
    return list(_TENSORS[k])


def _read_tensors(path):
    idx = _kv_index(path)
    if idx is None:
        raise ValueError("not a GGUF file")
    n_t, ents, kv_end = idx
    with open(path, "rb") as f:
        align = 32
        for key, t, off in ents:
            if key == "general.alignment" and t in (4, 5):
                f.seek(off)
                align = struct.unpack(_FMT[t], f.read(4))[0]
                if not (1 <= align <= 1 << 16 and align & (align - 1) == 0):
                    raise ValueError(f"GGUF header: alignment {align}")
        f.seek(kv_end)
        infos = []
        for _ in range(n_t):
            name = _read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            if nd > 8:
                raise ValueError(f"GGUF header: tensor {name!r} has {nd} dimensions")
            _skip(f, 8 * nd + 4)
            off = struct.unpack("<Q", f.read(8))[0]
            infos.append((name, off))
        data_start = (f.tell() + align - 1) // align * align
        total = os.path.getsize(path) - data_start
    infos.sort(key=lambda x: x[1])
    out = []
    for i, (name, off) in enumerate(infos):
        end = infos[i + 1][1] if i + 1 < len(infos) else total
        out.append((name, max(0, end - off)))
    return out


def sampling_meta(path):
    """-> {"arch", "name", "basename", "file_type", "sampling": {key: value}} from the GGUF header: what picks the model card's sampling defaults
    (general.sampling.* are the publisher's own, which llama.cpp already applies unless a flag is given)."""
    out = {"arch": "", "name": "", "basename": "", "file_type": None, "sampling": {}}
    first = shards(path)[0]
    idx = _kv_index(first)
    if idx is None:
        return out
    with open(first, "rb") as f:
        for key, t, off in idx[1]:
            f.seek(off)
            if key in ("general.architecture", "general.name", "general.basename") and t == 8:
                out[{"general.architecture": "arch"}.get(key, key.split(".")[1])] = _read_str(f)
            elif (key == "general.file_type" or key.startswith("general.sampling.")) and t in _FMT:
                v = struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]
                if key == "general.file_type":
                    out["file_type"] = v
                else:
                    out["sampling"][key[len("general.sampling."):]] = v
    return out


def mtp_layers(path):
    """-> (head, declared): head = number of layers with multi-token-prediction tensors (blk.N.nextn.*) in the files, declared =
    <arch>.nextn_predict_layers from the header (0 when absent). A quant can keep the key but drop the tensors; only `head` says the head is there."""
    declared = 0
    files = shards(path)
    idx = _kv_index(files[0])
    if idx is None:
        return 0, 0
    with open(files[0], "rb") as f:
        for key, t, off in idx[1]:
            if key.endswith(".nextn_predict_layers") and t in (4, 5, 10, 11):
                f.seek(off)
                declared = struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]
    head = {m.group(1) for p in files for name, _ in read_tensors(p) for m in [re.match(r"blk\.(\d+)\.nextn\.", name)] if m}
    return len(head), declared


def shards(path):
    m = re.match(r"(.*)-(\d+)-of-(\d+)\.gguf$", path)
    return sorted(glob.glob(glob.escape(m.group(1)) + f"-*-of-{m.group(3)}.gguf")) if m else [path]      # the folder and name are the user's text


def is_host_side(name):
    """LAUNCHER-1162: tensors llama.cpp keeps in host memory (the input layer) and never puts on the GPU: the token embedding and the per-layer token-embedding
    table (Gemma-3n style per_layer_token_embd, Flash-Next's n-gram table). The dense planners (dense_ffn_split, dense_gpu_fit) subtract them from the weights
    they count; model_layout() and planner/commit_guard.py already did, by the same name rule."""
    return name == "token_embd.weight" or "per_layer_token_embd" in name


def model_layout(path):
    """-> dict(n_layers, expert_bytes_per_layer, other_bytes, total_bytes, mtp_blocks). n_layers counts the MTP head's block (blk.N.nextn.*) when it
    has experts (Qwen3.6: 41 = 40 + 1); mtp_blocks says how many of the highest blocks are that head."""
    layers, other, table, mtp, emb, has_output = {}, 0, 0, set(), 0, False
    for p in shards(path):
        for name, size in read_tensors(p):
            mm = re.match(r"blk\.(\d+)\.nextn\.", name)
            if mm:
                mtp.add(int(mm.group(1)))
            m = re.match(r"blk\.(\d+)\..*_exps\.(weight|scale)", name)
            if m:
                layers[int(m.group(1))] = layers.get(int(m.group(1)), 0) + size
            elif "per_layer_token_embd" in name:
                table += size              # n-gram lookup table (qwen4exp): read from the file when needed, never on the GPU (same rule as planner/plan.py)
            else:
                other += size
                if name == "token_embd.weight":
                    emb += size
                elif name == "output.weight":
                    has_output = True
    n = (max(layers) + 1) if layers else 0
    per = sum(layers.values()) / len(layers) if layers else 0
    # token_embd stays on the CPU (llama.cpp keeps the input layer there; laptop 35B log: CUDA0 model buffer 5116.81 MiB = the other weights minus token_embd plus the
    # GPU expert layers, to the MiB). A model without output.weight reuses token_embd as its output layer, which then IS on the GPU.
    return {"n_layers": n, "expert_bytes_per_layer": per, "other_bytes": other, "table_bytes": table, "total_bytes": other + table + sum(layers.values()),
            "mtp_blocks": len(mtp & set(layers)), "expert_bytes_by_layer": dict(sorted(layers.items())), "cpu_only_bytes": emb if has_output else 0}


def gpu_expert_bytes(lay, n_cpu_moe):
    """Bytes of expert weights on the GPU with --n-cpu-moe n_cpu_moe (llama.cpp keeps the experts of layers 0 .. n-1 on the CPU): the real sizes of the layers
    above n (quantisation mixes make some layers bigger: the 35B UD-IQ3_XXS has 262 and 300 MiB layers)."""
    by = lay.get("expert_bytes_by_layer") or {}
    if by:
        return float(sum(v for k, v in by.items() if k >= n_cpu_moe))
    return float(max(0, lay["n_layers"] - n_cpu_moe) * lay["expert_bytes_per_layer"])


def gpu_layers_that_fit(lay, room):
    """The most expert layers (counted from the top, as --n-cpu-moe places them) whose real sizes fit `room` bytes."""
    by = lay.get("expert_bytes_by_layer") or {}
    if not lay["n_layers"] or not lay["expert_bytes_per_layer"] or room <= 0:
        return 0
    if not by:
        return max(0, min(lay["n_layers"], int(room // lay["expert_bytes_per_layer"])))
    used, g = 0.0, 0
    for k in range(lay["n_layers"] - 1, -1, -1):
        used += by.get(k, 0)
        if used > room:
            break
        g += 1
    return g


SMI_GAP_GIB = 0.40           # nvidia-smi (total - used) is 0.4 GiB more than CUDA can allocate on the RTX 4060 8 GB laptop (8146 vs 7745 MiB at idle); the first live run ran out of memory without it
DENSE_FIXED_GIB = 0.30        # CUDA context + compute buffer (-ub 512) besides the weights and the KV cache; the token embedding stays on the CPU. RTX 4060 8 GB: Qwen3-8B at 16k f16 peaked at 6.88 GiB (4.33 weights + 2.25 KV), Qwen3-14B q8_0 fits at 32k with 25 FFN layers off and not with 22


def dense_ffn_split(path, ctx, free_bytes, kv_q8=False, margin_gib=0.25):
    """LAUNCHER-732 / owner lever (a): a dense, pure-attention model bigger than the GPU. -> dict(first_layer, n_layers, n_off, ffn_gib, need_gib, free_gib, kv_gib, fits_all)
    with the FFN weights of the last n_off layers sent to the CPU while attention, the KV cache and the rest of the weights stay on the GPU, or None when this is not the
    right plan (MoE, hybrid recurrent model, no GPU memory known, the model plus its KV already fits, or even all FFN weights on the CPU leave it too big).
    Measured (Qwen3-14B Q4_K_M, RTX 4060 8 GB, q8_0 KV): at 32k tokens a plain layer split decodes 0.9 tok/s, FFN in RAM 3.4-4.0; at 16k 1.4 vs 4.4-5.5; at 4k 3.3 vs 4.4-5.4.
    The layer split wins only below about 2k tokens. A hybrid model (Qwen3.5/3.6/3.8: recurrent layers, little KV) did not gain: its layer split is faster (27B: 1.88 vs 1.2 tok/s)."""
    GiB = 2**30
    try:
        m = read_meta(shards(path)[0])
        arch = m.get("general.architecture")
        if m.get(f"{arch}.expert_count") or m.get(f"{arch}.full_attention_interval") or m.get(f"{arch}.ssm.inner_size") is not None or not free_bytes:
            return None
        ffn, total, emb, n_blocks = {}, 0, 0, int(m.get(f"{arch}.block_count", 0))
        for pth in shards(path):
            for name, size in read_tensors(pth):
                total += size
                emb += size if is_host_side(name) else 0
                mm = re.match(r"blk\.(\d+)\.ffn_(up|down|gate|gate_up)\.weight$", name)
                if mm:
                    ffn[int(mm.group(1))] = ffn.get(int(mm.group(1)), 0) + size
        kv = kv_bytes(path, ctx, kv_q8=kv_q8)
        if not ffn or kv is None or len(ffn) != n_blocks:
            return None
    except Exception:
        return None
    per = sum(ffn.values()) / len(ffn)
    free = free_bytes / GiB - SMI_GAP_GIB
    need = (total - emb) / GiB + kv / GiB + DENSE_FIXED_GIB + margin_gib
    out = {"n_layers": n_blocks, "ffn_gib": per / GiB, "need_gib": need, "free_gib": free, "kv_gib": kv / GiB, "fits_all": need <= free, "n_off": 0, "first_layer": n_blocks}
    if need <= free:
        return out
    n_off = int(-(-(need - free) * GiB // per))
    if n_off > n_blocks:
        out["n_off"] = None            # even every FFN on the CPU does not fit: the caller keeps llama.cpp's --fit
        return out
    out["n_off"], out["first_layer"] = n_off, n_blocks - n_off
    return out


def nextn_bytes(path):
    """Bytes of the NextN / MTP layers of a model file (the blocks past the main stack: blk.N with N >= block_count - nextn_predict_layers, ALL their tensors, not only the
    blk.N.nextn.* ones: Qwen3.8-27B's blk.64 is a whole layer, 0.327 GiB while its nextn.* tensors are 0.04 GiB); 0 when the model has none. llama.cpp builds that know
    `load_mtp` do not load them unless MTP drafting is on (LAUNCHER-1063 follow-up: the memory plan then must not count them)."""
    try:
        m = read_meta(shards(path)[0])
        arch = m.get("general.architecture")
        n_next, blocks = int(m.get(f"{arch}.nextn_predict_layers", 0) or 0), int(m.get(f"{arch}.block_count", 0) or 0)
        if n_next <= 0 or blocks <= n_next:
            return 0
        first, total, has_nextn = blocks - n_next, 0, False
        for pth in shards(path):
            for name, size in read_tensors(pth):
                mm = re.match(r"blk\.(\d+)\.", name)
                if mm and int(mm.group(1)) >= first:
                    total += size
                    has_nextn = has_nextn or ".nextn." in name
        return total if has_nextn else 0      # block_count must really include the NextN layers: no nextn.* tensor there = count nothing
    except Exception:
        return 0


def dense_gpu_fit(path, ctx, free_bytes, kv_q8=False, margin_gib=0.25, gap_gib=0.0, fixed_gib=None, skip_mtp=False):
    """LAUNCHER-982: does a model WITHOUT experts (dense, including the hybrid recurrent Qwen3.5/3.6/3.8 layouts) fit the GPU together with its KV cache?
    -> dict(model_gib, kv_gib, fixed_gib, margin_gib, need_gib, free_gib, fits, over_gib, ctx_fit) or None (unknown KV layout: sliding window, MLA; no GPU memory figure).
    model = file minus the token embedding (it stays on the CPU, laptop-measured); kv = kv_bytes() as llama-server sizes it (attention layers + recurrent state); fixed = CUDA/Vulkan
    context + compute buffer (DENSE_FIXED_GIB, measured on an 8 GB CUDA card at -ub 512; the PC's Vulkan figure is checked by the 27B 64k item); gap_gib = what the free-memory source
    overstates (nvidia-smi: SMI_GAP_GIB; llama-server's own --list-devices: 0). ctx_fit = the largest context (multiple of 1024, at most ctx) that fits all on the GPU with this KV type, 0 if none."""
    GiB = 2**30
    fixed = DENSE_FIXED_GIB if fixed_gib is None else fixed_gib
    if not free_bytes:
        return None
    try:
        total, emb = 0, 0
        for pth in shards(path):
            for name, size in read_tensors(pth):
                total += size
                emb += size if is_host_side(name) else 0
        kv = kv_bytes(path, ctx, kv_q8=kv_q8)
        if kv is None:
            return None
        skipped = nextn_bytes(path) / GiB if skip_mtp else 0.0          # MTP is off and this llama-server does not load the MTP layer: it takes no GPU memory
        model = (total - emb) / GiB - skipped
        free = free_bytes / GiB - gap_gib
        need = model + kv / GiB + fixed + margin_gib
        kv1 = kv_bytes(path, ctx, kv_q8=kv_q8, n_seq=1)
        out = {"np1_saves_gib": ((kv - kv1) / GiB if kv1 is not None else 0.0), "model_gib": model, "mtp_skipped_gib": skipped, "kv_gib": kv / GiB, "fixed_gib": fixed, "margin_gib": margin_gib, "need_gib": need, "free_gib": free,
               "fits": need <= free, "over_gib": max(need - free, 0.0), "ctx_fit": ctx if need <= free else 0}
        if need > free:
            lo, hi = 0, ctx // 1024
            while lo < hi:                      # largest multiple of 1024 that fits (the KV bytes grow with the context)
                mid = (lo + hi + 1) // 2
                k = kv_bytes(path, mid * 1024, kv_q8=kv_q8)
                if k is not None and model + k / GiB + fixed + margin_gib <= free:
                    lo = mid
                else:
                    hi = mid - 1
            out["ctx_fit"] = lo * 1024
        return out
    except Exception:
        return None


def read_meta(path):
    """-> {key: value} of the GGUF header: numbers and strings, arrays of numbers up to 4096 entries as lists (longer arrays are skipped)."""
    out = {}
    idx = _kv_index(path)
    if idx is None:
        raise ValueError("not a GGUF file")
    with open(path, "rb") as f:
        for key, t, off in idx[1]:
            f.seek(off)
            if t in _FMT:
                out[key] = struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]
            elif t == 8:
                out[key] = _read_str(f)
            elif t == 9:
                it, n = struct.unpack("<I", f.read(4))[0], struct.unpack("<Q", f.read(8))[0]
                if it in _FMT and n <= 4096:
                    size = struct.calcsize(_FMT[it])
                    if n * size > _left(f):
                        raise ValueError("GGUF header: array past the end of the file")
                    out[key] = [struct.unpack(_FMT[it], f.read(size))[0] for _ in range(n)]
    return out


KV_ELEM = {False: 2.0, True: 34 / 32, "q4_0": 18 / 32}       # bytes per K/V value: f16, q8_0 (--fast), q4_0 (a block of 32 values is 18 bytes)


KV_ELEM_BY_NAME = {"f16": 2.0, "bf16": 2.0, "f32": 4.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}      # bytes per value of llama.cpp's -ctk / -ctv types (checked at the source: ggml's block_q4_1 is d(2) + m(2) + qs[16] = 20 bytes per 32 values; 24 bytes is q5_1)


def kv_elems(kv):
    """(bytes per K value, bytes per V value) for a KV type (LAUNCHER-1063): False / "f16" = f16, True / "q8_0" = q8_0 for both, "q4_0" = q4_0 for both, or a pair
    ("q8_0", "q4_0") = K type, V type (llama.cpp's -ctk / -ctv; K matters more for quality than V)."""
    one = lambda t: KV_ELEM[t] if t == "q4_0" else (KV_ELEM_BY_NAME.get(t, 2.0) if isinstance(t, str) else KV_ELEM[bool(t)])      # a type not in the table counts as f16: errs high
    if isinstance(kv, (tuple, list)):
        return one(kv[0]), one(kv[1])
    e = one(kv)
    return e, e


def kv_bytes(path, ctx, kv_q8=False, n_seq=4):
    """GPU memory of llama-server's KV cache for `ctx` tokens plus the recurrent state of n_seq slots (llama-server's default 4), from the header
    as llama.cpp sizes them: attention layers (every full_attention_interval-th layer on hybrid models) x KV heads x (key + value length), one
    indexer key per token on models with a sparse-attention indexer (Flash-Next), and per slot the SSM conv + state tensors (f32) of the other layers.
    Flash-Next at 256k, as llama-server reports it on the PC (pc-bench, 620d2225c): KV 6144 + indexer 1536 MiB, RS 450 MiB. None for layouts this does not model (MLA, per-layer KV head lists:
    GLM-5.3): the callers then keep the flat 40 KB per token (idea N26: the flat number over-reserved 3-5 GiB at 256k)."""
    try:
        m = read_meta(shards(path)[0])
    except Exception:
        return None
    arch = m.get("general.architecture")
    g = lambda k, d=None: m.get(f"{arch}.{k}", d)
    blocks, heads, kvh = g("block_count"), g("attention.head_count"), g("attention.head_count_kv")
    if isinstance(heads, list):         # LAUNCHER-741 (b): per-layer head counts (zero on layers without attention): the widest layer gives the head size
        heads = max((h for h in heads if isinstance(h, int)), default=0)
    kv_list = kvh if isinstance(kvh, list) and all(isinstance(x, int) and x >= 0 for x in kvh) else None
    if not isinstance(blocks, int) or not isinstance(heads, int) or (isinstance(kvh, list) and kv_list is None) or \
            g("attention.key_length_mla") is not None or g("attention.kv_lora_rank") is not None or g("attention.sliding_window") is not None:
        return None
    if heads <= 0:
        return None
    kvh = (max(kv_list) if kv_list else heads) if kv_list is not None else (kvh or heads)
    n_main = blocks - int(g("nextn_predict_layers", 0) or 0)
    k_len = g("attention.key_length") or (g("embedding_length", 0) // heads)
    v_len = g("attention.value_length") or k_len
    interval = int(g("full_attention_interval", 0) or 0)
    n_attn = n_main // interval if interval else n_main
    ek, ev = kv_elems(kv_q8)
    elem = (ek + ev) / 2
    if kv_list is not None:                                   # llama.cpp sizes each layer by its own n_head_kv
        n_attn = sum(1 for x in kv_list[:n_main] if x > 0)
        per_tok = sum(kv_list[:n_main]) * (k_len * ek + v_len * ev)
    else:
        per_tok = n_attn * kvh * (k_len * ek + v_len * ev)
    if g("attention.indexer.key_length"):
        per_tok += 2 * n_attn * g("attention.indexer.key_length") * elem    # the indexer's cache: 2 x key_length per token and attention layer (PC, 256k: 1536 MiB f16)
    state = 0
    if g("ssm.inner_size"):
        inner, n_state, groups, rank, conv = (g("ssm.inner_size"), g("ssm.state_size", 0), g("ssm.group_count", 1), g("ssm.time_step_rank") or 1,
                                              g("ssm.conv_kernel", 4))
        per_layer = rank * (inner // rank) * n_state * 4 + (conv - 1) * (inner + 2 * groups * n_state) * 4
        state = (n_main - n_attn) * per_layer * n_seq
    return ctx * per_tok + state


def _kv_or_flat(path, ctx, kv_q8=False):
    kv = kv_bytes(path, ctx, kv_q8)
    return kv if kv is not None else ctx * 40e3 * (0.53 if kv_q8 else 1.0)


def gpu_total_free_bytes():
    """(total, free) bytes of the GPU the planner uses (NVIDIA first, else the AMD GPU with the most VRAM); None when there is none."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, errors="replace", timeout=5).stdout
        out = out.strip().splitlines()[0].split(",")
        return float(out[0]) * 1048576, (float(out[0]) - float(out[1])) * 1048576
    except Exception:
        pass
    best = None
    for d in glob.glob("/sys/class/drm/card[0-9]*/device"):
        try:
            if open(d + "/vendor").read().strip() != "0x1002":
                continue
            tot = int(open(d + "/mem_info_vram_total").read()); used = int(open(d + "/mem_info_vram_used").read())
        except Exception:
            continue
        if tot >= 2**30 and (best is None or tot > best[0]):
            best = (tot, tot - used)
    return (float(best[0]), float(best[1])) if best else None


# R2-06 (MOE_CACHE_GPU_MEM_LOG=1, default off): the launcher samples how much GPU memory llama-server holds, once a second, and logs it. On Windows the
# counters "GPU Process Memory" Dedicated Usage and Shared Usage of the process: Shared growing while Dedicated sits at the card's size means the driver
# moved memory to system RAM (the over-commit that ggml's own free-memory reading cannot see). Linux: the AMD card's VRAM and GTT use (device-wide, sysfs)
# or, for NVIDIA, nvidia-smi's use by the process. Log only.
def parse_gpu_process_counters(text, pid=None):
    """(dedicated MiB, shared MiB) from lines of 'counter path|value;...' (Get-Counter output flattened by the launcher); instances of other processes are
    skipped when pid is given; (None, None) when no counter matched."""
    ded = sha = None
    for item in text.replace("\n", ";").split(";"):
        path, _, val = item.rpartition("|")
        path = path.lower()
        if "gpu process memory" not in path or (pid is not None and f"pid_{pid}_" not in path):
            continue
        try:
            v = float(val) / 1048576
        except ValueError:
            continue
        if path.endswith("dedicated usage"):
            ded = (ded or 0) + v
        elif path.endswith("shared usage"):
            sha = (sha or 0) + v
    return ded, sha


def sample_gpu_mem_linux(pid=None):
    """(dedicated MiB, shared MiB, what) now: AMD sysfs (whole device: VRAM and GTT use), else nvidia-smi's use by pid (shared None); None when neither."""
    best = None
    for d in glob.glob("/sys/class/drm/card[0-9]*/device"):
        try:
            if open(d + "/vendor").read().strip() != "0x1002":
                continue
            tot = int(open(d + "/mem_info_vram_total").read()); used = int(open(d + "/mem_info_vram_used").read()); gtt = int(open(d + "/mem_info_gtt_used").read())
        except Exception:
            continue
        if best is None or tot > best[0]:
            best = (tot, used / 1048576, gtt / 1048576)
    if best:
        return best[1], best[2], "amdgpu, whole device"
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, errors="replace", timeout=5).stdout
        tot = [float(m) for p, m in (l.split(",")[:2] for l in out.splitlines() if "," in l) if pid is None or p.strip() == str(pid)]
        return (sum(tot), None, "nvidia-smi, this process") if tot else None
    except Exception:
        return None


def start_gpu_mem_log(pid, say, windows, env=None, alive=None):
    """Start the sampler thread when MOE_CACHE_GPU_MEM_LOG is set (1 or more); returns the thread or None. A line goes out whenever dedicated or shared use
    moved 64 MiB since the last one, and one summary (peak values) when the process is gone. say(text) prints a launcher line;
    alive() says whether the process runs (the launcher passes its Popen's poll: an unwaited child is a zombie that os.kill still finds)."""
    import threading, time
    v = (env if env is not None else os.environ).get("MOE_CACHE_GPU_MEM_LOG", "")
    if v in ("", "0"):
        return None
    def alive_os():
        if windows:
            return True
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    alive = alive or alive_os
    def run():
        peak = [0.0, 0.0]; last = [-1e9, -1e9]; n = 0; shown_what = ""
        ps = None
        if windows:
            cmd = (f"Get-Counter -Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage','\\GPU Process Memory(pid_{pid}_*)\\Shared Usage' "
                   "-SampleInterval 1 -Continuous | ForEach-Object { ($_.CounterSamples | ForEach-Object { $_.Path + '|' + $_.CookedValue }) -join ';'; [Console]::Out.Flush() }")
            try:
                ps = subprocess.Popen([_win_ps(), "-NoProfile", "-NonInteractive", "-Command", cmd], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, errors="replace")
            except OSError:
                say("moe-cache: GPU memory log: PowerShell not found, no log")
                return
        while True:
            if windows:
                line = ps.stdout.readline()
                if not line:
                    break
                d, s_ = parse_gpu_process_counters(line, pid); what = "Windows GPU Process Memory counters"
            else:
                time.sleep(1)
                if not alive():
                    break
                r = sample_gpu_mem_linux(pid)
                if not r:
                    continue
                d, s_, what = r
            if d is None and s_ is None:
                continue
            n += 1
            shown_what = what
            peak = [max(peak[0], d or 0), max(peak[1], s_ or 0)]
            if abs((d or 0) - last[0]) >= 64 or abs((s_ or 0) - last[1]) >= 64:
                last = [d or 0, s_ or 0]
                say(f"moe-cache: GPU memory ({what}): dedicated {d or 0:.0f} MiB" + (f", shared {s_:.0f} MiB" if s_ is not None else ""))
        if ps:
            ps.kill()
        if n:
            say(f"moe-cache: GPU memory peak ({shown_what}, {n} samples): dedicated {peak[0]:.0f} MiB, shared {peak[1]:.0f} MiB")
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


# LAUNCHER-1133 (audit 2026-11-12, rank 1; build's LAUNCHER-1131 WIP, pre-load half only): Windows AMD / Intel: llama-server's --list-devices "free" is the Vulkan budget (what the process may
# allocate). With the desktop on the card the process keeps only its dedicated share (PC, RX 9070 XT 16 GB: 13.1-13.5 GiB of 15.4 GiB "free") and the rest is paged to shared system memory without a log line.
ROOM_RESERVE_MIB = 300      # kept free below the card's size minus the other programs' use (their use grows while the server runs)


def counters_text(pid=None, timeout=8):
    """One PowerShell read of '\\GPU Process Memory(*)' Dedicated Usage and, in the same read, '\\GPU Adapter Memory(*)' Dedicated Usage ('path|value;...', the format others_dedicated_mib
    reads; the adapter counter is the card's own total, the figure that caps the sum of the processes). '' when the process read fails."""
    pat = f"pid_{pid}_*" if pid else "*"
    cmd = ("$e = 'SilentlyContinue'; "         # two separate reads: if one counter set is missing the other still answers (critic R5)
           "foreach ($c in @('\\GPU Process Memory(P)\\Dedicated Usage', '\\GPU Adapter Memory(*)\\Dedicated Usage')) { "
           "Get-Counter -Counter $c -ErrorAction $e | "
           "ForEach-Object { ($_.CounterSamples | ForEach-Object { $_.Path + '|' + $_.CookedValue }) -join ';' } }").replace("(P)", f"({pat})")
    try:
        return subprocess.run([_win_ps(), "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True, text=True, errors="replace", timeout=timeout).stdout
    except Exception:
        return ""


def _luid(path):
    m = re.search(r"luid_(0x[0-9a-f]+_0x[0-9a-f]+)", path)
    return m.group(1) if m else None


def others_dedicated_mib(text, exclude_pid=None):
    """MiB of dedicated GPU memory the OTHER programs hold on the card; None when no counter matched.
    2026-11-12 (PC, RX 9070 XT; real capture in tests/data/gpu_counters_pc_20261112_*.csv): the sum of '\\GPU Process Memory(*)\\Dedicated Usage' over the card's own instances was 3.89 GiB while the
    card's '\\GPU Adapter Memory' Dedicated Usage read 1.92 GiB (gputaskman 1.93). It is NOT a cross-adapter or multi-phys double count (the other luids held ~0): the per-process figures
    themselves add up ~2 GiB above what the adapter holds (cause not established). So:
      - with adapter counters in the text: only the card's instances (the adapter with the most dedicated use, as bin/moe_cache_record.py picks it), and never more than that adapter's own
        figure minus our own share (exclude_pid);
      - without them (older text, a failed second counter): the plain sum over all adapters, as before (the safe side)."""
    proc, adapters = [], {}
    for item in text.replace("\n", ";").split(";"):
        path, _, val = item.rpartition("|")
        path = path.lower()
        try:
            v = float(val) / 1048576
        except ValueError:
            continue
        if "gpu adapter memory" in path and path.endswith("dedicated usage"):
            lu = _luid(path)
            if lu:
                adapters[lu] = max(adapters.get(lu, 0.0), v)
        elif "gpu process memory" in path and path.endswith("dedicated usage"):
            proc.append((path, v))
    target = max(adapters, key=adapters.get) if adapters else None
    tot, own = None, 0.0
    for path, v in proc:
        mine = exclude_pid is not None and f"pid_{exclude_pid}_" in path
        if target and _luid(path) not in (None, target):
            continue                                      # another adapter (the iGPU)
        if mine:
            own += v
            continue
        tot = (tot or 0.0) + v
    if tot is not None and target:
        tot = min(tot, max(adapters[target] - own, 0.0))
    return tot


def real_room_mib(total_mib, others_mib, reserve_mib=ROOM_RESERVE_MIB):
    """What a new process keeps in dedicated VRAM: the card's size minus the other programs' dedicated use minus a growth reserve; None when a figure is missing or the result is not plausible."""
    if not total_mib or others_mib is None or others_mib >= total_mib:
        return None
    r = total_mib - others_mib - reserve_mib
    return r if r >= 1024 else None


def desk_reserve_gb(total, free, base=0.5):
    """VRAM (GB) to keep free next to everything the tuner places, scaled with what other programs already hold (the desktop compositor, a browser).
    Their use grows while the server runs (a browser tab, a video): the more the desktop already takes, the more room is left so the long-prompt GPU
    block still finds its room (PC 2026-10-29: dwm 1.2 GB + browser 0.9 GB held 2.8 GB of 16; the block found 697 of 791 MiB and ran on the CPU,
    73 instead of ~290 tok/s). Returns (reserve_gb, used_by_others_gb). A headless card (under ~0.5 GB used) keeps the default `base`; at most 2.5 GB."""
    if not total or free is None:
        return base, 0.0
    used = max(0.0, (total - free) / 2**30)
    return min(2.5, max(base, 0.25 + 0.5 * used)), used


LAST_FREE_SOURCE = [None]       # where free_vram_bytes' last figure came from (the plan counts the CUDA gap only for nvidia-smi's figure)


def free_vram_bytes():
    """Free memory of the GPU: NVIDIA (nvidia-smi), else the AMD GPU with the most VRAM (amdgpu sysfs, as plan.py); None when there is none."""
    LAST_FREE_SOURCE[0] = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, errors="replace", timeout=5).stdout
        out = out.strip().splitlines()[0].split(",")    # one line per GPU: the first, as plan.py (with two GPUs the whole text failed to parse: no GPU)
        v = (float(out[0]) - float(out[1])) * 1048576
        LAST_FREE_SOURCE[0] = "nvidia-smi"
        return v
    except Exception:
        pass
    best = None
    for d in glob.glob("/sys/class/drm/card[0-9]*/device"):
        try:
            if open(d + "/vendor").read().strip() != "0x1002":
                continue
            tot = int(open(d + "/mem_info_vram_total").read()); used = int(open(d + "/mem_info_vram_used").read())
        except Exception:
            continue
        # mem_info_vram_total is dedicated VRAM: on an integrated GPU (APU) the BIOS carve-out, which the OS cannot use, so it is counted. GTT (system
        # RAM the GPU may borrow) is never counted. Under 1 GiB (e.g. a 512 MiB carve-out) no expert layer fits next to the buffers: skipped
        if tot >= 2**30 and (best is None or tot > best[0]):
            best = (tot, tot - used)
    if best:
        LAST_FREE_SOURCE[0] = "amd-sysfs"
    return best[1] if best else None


def fit_n_cpu_moe(path, ctx=8192, reserve_gb=1.3, free=None):
    """Layers whose experts stay on the CPU. The GPU gets: every non-expert weight, KV cache and buffers (estimated), and as many whole layers of experts as fit."""
    lay = model_layout(path)
    free = free if free is not None else free_vram_bytes()
    if not lay["n_layers"] or free is None:
        return 999, lay, None
    # 0.55e9 is a legacy "compute buffers" estimate with no measurement behind it (the launcher's own path uses the _COMPUTE_MIB table instead,
    # which shows 0.7-4.6 GiB at 64k+). Not reachable from the launcher: fit_n_cpu_moe runs only from this file's __main__ (a debugging CLI) and
    # gpu_cache_mib() has no caller — the gpu_cache_mib value in tune()'s dict is computed by the ladder below. Left at the old value on purpose.
    fixed = lay["other_bytes"] + 0.55e9 + _kv_or_flat(path, ctx)          # weights + compute buffers (legacy estimate) + KV (from the header when it can)
    room = free - fixed - reserve_gb * 1e9
    gpu_layers = max(0, min(lay["n_layers"], int(room // lay["expert_bytes_per_layer"]))) if lay["expert_bytes_per_layer"] else 0
    return lay["n_layers"] - gpu_layers, lay, free


def gpu_cache_mib(path, ctx=8192, reserve_gb=1.3, free=None):
    """VRAM (MiB) llama.cpp's GPU expert cache can use when no whole layer of experts is placed on the GPU: free VRAM minus the non-expert weights,
    compute buffers, KV cache and a reserve. 0 when there is too little (or the free VRAM is unknown)."""
    lay = model_layout(path)
    free = free if free is not None else free_vram_bytes()
    if not lay["n_layers"] or free is None:
        return 0
    fixed = lay["other_bytes"] + 0.55e9 + _kv_or_flat(path, ctx)      # same legacy compute-buffer estimate as fit_n_cpu_moe above; this function has no caller (dead) — value left as is
    room = free - fixed - reserve_gb * 1e9
    return int(room // (1024 * 1024)) if room > 0.5e9 else 0


def gpu_cache_cap_mib(path, max_tensor_bytes, margin=0.97):
    """Largest llama.cpp --moe-cache-mib whose cache banks each fit in one device tensor of max_tensor_bytes; None when every size fits.

    llama.cpp (src/llama-moe-cache.cpp) groups the layers by expert layout (types and shapes), makes one bank tensor per expert tensor per group
    (all of the group's slots), and shares the budget among groups by their expert bytes. Vulkan rejects a tensor above the device's maximum
    buffer size or storage buffer range (ggml_backend_vk_device_supports_op), also a bank, and the scheduler then aborts the load (seen on the PC:
    5900 MiB loads, 8909 aborts; T5). A group of n layers whose largest expert tensor is m bytes per layer gets budget * host_g / host_all, and its
    biggest bank is about that * m / (bytes of one layer's experts), so the budget may be at most max_tensor_bytes * host_all / (n * m).
    margin covers the bank's extra slot and layers llama.cpp leaves out (an MTP head that is not loaded)."""
    groups = {}
    for p in shards(path):
        per = {}
        for name, size in read_tensors(p):
            m = re.match(r"blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight$", name)
            if m:
                per.setdefault(int(m.group(1)), {})[m.group(2)] = size
        for il, t in per.items():
            groups.setdefault(tuple(sorted(t.items())), []).append(il)
    host_all = sum(len(ls) * sum(sz for _, sz in key) for key, ls in groups.items())
    cap = None
    for key, ls in groups.items():
        big = len(ls) * max(sz for _, sz in key)            # the group's largest bank when it holds all of its experts
        if big > max_tensor_bytes:
            c = max_tensor_bytes * host_all / big
            cap = c if cap is None else min(cap, c)
    return int(cap * margin // 1048576) if cap is not None else None


# Measured GPU compute buffers (MiB) of Qwen3.8-Flash-Next with a q8_0 KV cache, by context (rounded up between points) and micro-batch size.
# Other architectures differ somewhat; the numbers are the best available and the reserve below covers the usual spread.
# Chunk 2048 at long context, 2026-11-02, PC, GPU_SUM on, Flash-Next Q2_0: 34k 607 tok/s at 39 CPU layers and 567-582 at 40, against 394-411 at -ub 1024 with 37; decode 27.4 against 28.5.
# 65536 (measured at ctx 40960 and 65536): 39 CPU layers; at 64k 455-458 tok/s, decode 26.1-26.8, against 41 layers 454 / 22.3 and -ub 1024 415 / 25.6. 131072 keeps 3150 (41 CPU layers): 128k 415-438. A fuller card has
# blocks wait far longer, so these rows are room to leave, not only the buffer's size. Earlier "1024 is faster on long prompts" predates HOST_OUT and GPU_SUM.
_COMPUTE_MIB = {32768: {512: 770, 1024: 770, 2048: 990}, 65536: {512: 690, 1024: 920, 2048: 1750}, 131072: {512: 960, 1024: 1700, 2048: 3150},
                # 256k (N29, PC llama-server 620d2225c, same with f16 or q8_0 KV): 512 1403, 1024 2983, 2048 3979 MiB. 2048 picked since 2026-11-03: PC, Flash-Next Q2_0,
                # 222k-token prompt, official b11966 server, A B B A: 334.8 / 333.2 tok/s at 46 CPU layers against 263.4 / 265.8 at -ub 1024 with 45 (+26 %)
                262144: {512: 1403, 1024: 2983, 2048: 3979}}
# Measured but never auto-picked (None above): still the size llama-server takes when the user asks for that micro-batch (audit 2026-10-29 L1:
# without these, -ub 2048 at 128k/256k was guessed at 990 MiB, about 2-3 GiB short, so the long-prompt block found no room)
_COMPUTE_MIB_NOT_PICKED = {131072: {2048: 3150},
                           # SINGLE MEASUREMENT: every -ub 4096 number the planner uses comes from ONE model, Flash-Next IQ3_XXS at 64k (one run, no other model, card or context).
                           # compute_buffer_guess_mib extrapolates it: <= 64k reuses 4660, 128k and 256k scale the 2048 rows (x2: 7958 MiB), above 256k it grows with the context.
                           # The numbers err high on purpose; a second model or context at -ub 4096 should replace the extrapolation (tests/planner_test.py pins the current values).
                           # -ub 4096 at 64k, IQ3_XXS, f16 KV, PC night chain 2026-11-07 (item 15): 4,660 MiB (2,021 MiB at -ub 2048 on the same file); used only by the long-prompt mode
                           65536: {4096: 4660}}
_PROMPT_GAIN = {512: 1.0, 1024: 1.26, 2048: 1.58}      # measured prefill speed-up of a bigger micro-batch (6k-token prompt)


def compute_buffer_mib(ctx, ub):
    key = next((k for k in sorted(_COMPUTE_MIB) if ctx <= k), None)
    return _COMPUTE_MIB[key].get(ub) if key else None


def compute_buffer_guess_mib(ctx, ub):
    """compute_buffer_mib, or for a pair that was not measured an estimate that errs high: the same micro-batch at the next larger measured
    context, else the largest 2048 value scaled to this micro-batch (a micro-batch the user chose must still leave the long-prompt block its room)."""
    b = compute_buffer_mib(ctx, ub)
    if b is None:
        def known(k):
            return _COMPUTE_MIB[k].get(ub) or _COMPUTE_MIB_NOT_PICKED.get(k, {}).get(ub)
        b = next((known(k) for k in sorted(_COMPUTE_MIB) if k >= ctx and known(k)), None)
    if b is None and ctx > max(_COMPUTE_MIB):
        # past the largest measured context the buffer keeps growing with it: scale that row (T0a: a flat 0.55 GB guess at 256k, where llama-server
        # takes 1403 MiB, put 3 expert layers on the GPU and left the long-prompt block 0 MiB, so prompts ran on the CPU at half speed)
        top = max(_COMPUTE_MIB); row = {u: v for u, v in _COMPUTE_MIB[top].items() if v}
        near = min(row, key=lambda u: abs(u - ub))
        b = row[near] * max(ub, 512) / near * ctx / top
    if b is None:
        b = max(v.get(2048) or 0 for v in _COMPUTE_MIB.values()) * max(ub, 512) / 2048
        # a micro-batch between two measured ones never needs more than the larger measured one (the -ub 4096 row at 64k, 2026-11-07)
        larger = [v for k in set(_COMPUTE_MIB) | set(_COMPUTE_MIB_NOT_PICKED) if k >= ctx
                  for u2, v in {**_COMPUTE_MIB.get(k, {}), **_COMPUTE_MIB_NOT_PICKED.get(k, {})}.items() if u2 > ub and v]
        if larger:
            b = min(b, min(larger))
    return b


# MOE_CACHE_PLAN_RULE=1 (default off): one reserve rule for every model and card, replacing the table above for the numbers (the table still decides which
# micro-batches are candidates). 1) The compute buffer is measured at start with llama-fit-params (graphs built with no_alloc at this context and -ub, about
# 3 s each, cached for the process); 2) without that tool, or when it fails, a formula fitted to the Flash-Next table, scaled up 1.3x so it errs high;
# 3) either way the reserve never falls when -ub grows (running maximum over the candidate micro-batches). The table's reserve was not monotone: 920 / 3150 /
# 1485 / 1980 / 2970 MiB for -ub 1024 / 2048 / 3072 / 4096 / 6144 at 64k, so each -ub got a different number of layers (scout round 2, section 2).
PLAN_UBS = (512, 1024, 2048, 3072, 4096, 6144)
_FIT_CACHE = {}


def plan_rule_on():
    return os.environ.get("MOE_CACHE_PLAN_RULE") not in (None, "", "0")


PLAN_RULE_DEFAULT_ON = False   # when True the plan rule (measured compute buffer via llama-fit-params) is used by default when llama-fit-params is found; MOE_CACHE_PLAN_RULE=0 forces the old table; PC pair decides, LAUNCHER-1236


def plan_rule_active(fit_params=None):
    """LAUNCHER-1236: whether the plan rule decides the reserve: MOE_CACHE_PLAN_RULE=0 forces the old table, =1 turns it on, else the default (and llama-fit-params must be found)."""
    if os.environ.get("MOE_CACHE_PLAN_RULE") == "0":
        return False
    return plan_rule_on() or bool(PLAN_RULE_DEFAULT_ON and fit_params)


def formula_buffer_mib(ctx, ub):
    """Compute buffer (MiB) by formula, non-decreasing in ctx and ub: least squares through the 11 measured Flash-Next points (a + b ub + c ub ctx + d ctx,
    ctx in units of 64k) times 1.3, which is the largest ratio of a measured value to the fit. Not model specific: llama-fit-params is the real answer."""
    c = ctx / 65536
    return 1.3 * (318 + 0.249 * ub + 0.379 * ub * c + 109.7 * c)


def parse_fit_params_compute_mib(text):
    """Largest compute figure of a GPU in llama-fit-params --fit-print output ('DEVICE model context compute' in MiB per line; 'Host' skipped); None if none."""
    best = None
    for line in text.splitlines():
        f = line.split()
        if len(f) == 4 and f[0].lower() != "host" and all(x.isdigit() for x in f[1:]):
            best = max(best or 0, int(f[3]))
    return float(best) if best is not None else None


def find_fit_params(llama_server=None):
    """Path of llama-fit-params next to llama-server or on PATH (MOE_CACHE_FIT_PARAMS overrides); None when there is none."""
    import shutil
    cands = [os.environ.get("MOE_CACHE_FIT_PARAMS")]
    if llama_server:
        d = os.path.dirname(os.path.abspath(llama_server))
        cands += [os.path.join(d, "llama-fit-params"), os.path.join(d, "llama-fit-params.exe")]
    cands.append(shutil.which("llama-fit-params"))
    return next((c for c in cands if c and os.path.isfile(c)), None)


FIT_DISK_NAME = "fit-params-cache.json"
FIT_DISK_KEEP = 400
_FIT_LOCK = threading.Lock()
_GPU_ID = []


def fit_disk_file():
    return os.environ.get("MOE_CACHE_FIT_CACHE") or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "moe-cache", FIT_DISK_NAME)


def drm_driver_version():
    """The driver an AMD / Intel card on Linux runs with, for the record key (follow-up 6; nvidia-smi's identity already carries its driver version): the kernel
    driver's name and its module version when it has one, else the kernel release (the amdgpu / i915 driver ships with the kernel). "" when nothing is readable."""
    names = set()
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        try:
            names.add(os.path.basename(os.path.realpath(card + "/driver")))
        except OSError:
            pass
    parts, versioned = [], False
    for n in sorted(x for x in names if x and x != "driver"):
        v = ""
        try:
            v = open(f"/sys/module/{n}/version").read().strip()
        except OSError:
            pass
        versioned = versioned or bool(v)
        parts.append(n + (f" {v}" if v else ""))
    if not versioned:                      # the amdgpu / i915 driver ships with the kernel and has no version of its own: the kernel release stands for it
        try:
            parts.append("kernel " + os.uname().release)
        except AttributeError:
            pass
    return ", ".join(parts)


def gpu_identity():
    """Which GPU and driver the measured buffers belong to (PREFILL-180): nvidia-smi's name and driver, else the vendor/device ids of the DRM cards (Linux sysfs);
    "" when neither answers, and then nothing is cached on disk. Asked once per process."""
    if _GPU_ID:
        return _GPU_ID[0]
    ident = ""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, errors="replace", timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            ident = "nvidia:" + ";".join(out.stdout.strip().splitlines())
    except Exception:
        pass
    if not ident:
        ids = []
        for card in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
            try:
                ids.append(open(card + "/vendor").read().strip() + ":" + open(card + "/device").read().strip())
            except OSError:
                pass
        ident = "drm:" + ",".join(ids) if ids else ""
        if ident:
            ident += ";" + drm_driver_version()
    _GPU_ID.append(ident)
    return ident


def _file_id(p):
    st = os.stat(p)
    return [os.path.basename(p), st.st_size, st.st_mtime_ns]


def fit_disk_key(fit_params, path, ctx, ub, kv_q8):
    """The key of a measured buffer: the tool, the model files, context, -ub, the KV type and the GPU; None when any part is unknown (nothing cached)."""
    try:
        gpu = gpu_identity()
        if not gpu:
            return None
        return json.dumps([_file_id(os.path.realpath(fit_params)), [_file_id(x) for x in shards(path)], ctx, ub, bool(kv_q8), gpu])
    except OSError:
        return None


def _fit_disk_read():
    try:
        d = json.load(open(fit_disk_file(), encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _fit_disk_store(key, v):
    with _FIT_LOCK:
        d = _fit_disk_read()
        d.pop(key, None)
        d[key] = v
        for old in list(d)[:-FIT_DISK_KEEP]:
            del d[old]
        f = fit_disk_file()
        try:
            os.makedirs(os.path.dirname(f), exist_ok=True)
            tmp = f"{f}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(d, fh)
            os.replace(tmp, f)
        except OSError:
            pass


def fit_params_buffer_mib(fit_params, path, ctx, ub, kv_q8=False, timeout=90):
    """Compute buffer (MiB) llama.cpp itself would reserve for this model, context and -ub (-b raised to match, or the tool clamps -ub to 2048); None on any failure.
    Remembered for the process and on disk (PREFILL-180, ~3 s a measurement): the same model, context, -ub, KV type, GPU and tool files give the same figure."""
    key = (fit_params, path, ctx, ub, bool(kv_q8))
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]
    dkey = fit_disk_key(fit_params, path, ctx, ub, kv_q8)
    v = None
    if dkey:
        got = _fit_disk_read().get(dkey)
        if isinstance(got, (int, float)) and not isinstance(got, bool) and got > 0:
            v = float(got)
    if v is None:
        try:
            cmd = [fit_params, "-m", path, "-c", str(ctx), "-ub", str(ub), "-b", str(max(ub, 2048)), "-ngl", "99", "--n-cpu-moe", "999", "--fit-print", "on"]
            if kv_q8:
                cmd += ["-ctk", "q8_0", "-ctv", "q8_0"]
            v = parse_fit_params_compute_mib(subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout).stdout)
        except Exception:
            pass
        if v is not None and dkey:
            _fit_disk_store(dkey, v)           # a failure is never written: the next launch tries again
    _FIT_CACHE[key] = v
    return v


def prefetch_fit_params(fit_params, path, ctx, ubs, kv_q8=False, workers=4):
    """PREFILL-181: measure the micro-batch candidates that are not remembered yet at the same time (the tool builds graphs with no_alloc, so the runs
    do not compete for GPU memory); the figures are the same as one by one, only the waiting is shorter. Returns the number of runs started."""
    todo = [u for u in dict.fromkeys(ubs) if fit_params and (fit_params, path, ctx, u, bool(kv_q8)) not in _FIT_CACHE]
    if len(todo) < 2:
        return 0
    ts = [threading.Thread(target=fit_params_buffer_mib, args=(fit_params, path, ctx, u, kv_q8), daemon=True) for u in todo[:max(1, workers)]]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return len(ts)


def rule_buffer_mib(path, lay, ctx, ub, kv_q8=False, fit_params=None, ubs=PLAN_UBS):
    """-> (MiB, source) the planner reserves for the GPU compute buffer under the rule: non-decreasing in ub (running maximum over `ubs` up to this one)."""
    best, src = 0.0, "formula"
    for u in sorted(set(ubs) | {ub}):
        if u > ub:
            break
        m = fit_params_buffer_mib(fit_params, path, ctx, u, kv_q8) if fit_params else None
        if m is None:
            m, s = formula_buffer_mib(ctx, u), "formula"
        else:
            s = "llama-fit-params"
        if m >= best:
            best, src = m, s
    return best, src


def rule_block_mib(path, lay, ub):
    """Room the long-prompt GPU block needs (MiB): one layer of experts + 35 MiB + the batch's intermediate results (per token n_used x (2 n_ff_exp + n_embd) x 4 B x 1.2;
    Flash-Next 0.18 MiB, measured 923 MiB at -ub 1024 and 1107 at 2048) + the 256 MiB the plugin wants free before it allocates a block."""
    per_tok = 0.18 * 1048576
    try:
        m = read_meta(shards(path)[0]); a = m.get("general.architecture")
        used, ff, emb = (m.get(f"{a}.{k}") for k in ("expert_used_count", "expert_feed_forward_length", "embedding_length"))
        if all(isinstance(x, int) and x > 0 for x in (used, ff, emb)):
            per_tok = used * (2 * ff + emb) * 4 * 1.2
    except Exception:
        pass
    return (lay["expert_bytes_per_layer"] + (35 + 256) * 1048576 + per_tok * ub) / 1048576 if lay["expert_bytes_per_layer"] else 0.0


DECODE_PCT_PER_LAYER = 2.5    # decode speed lost per expert layer moved to the CPU; display-only, read by plan_table(); a guess until PC pair d1 (ncmoe 3 vs 7) decides, PC records suggest ~4


def plan_table(path, ctx=8192, free=None, kv_q8=False, reserve_gb=0.5, fit_params=None, ubs=PLAN_UBS):
    """One row per micro-batch candidate under the rule, for the launcher's --dry-run: reserve, block room, layers on the GPU and what that costs.
    decode_pct: decode speed against the candidate with the most GPU layers, DECODE_PCT_PER_LAYER % per layer (scout round 2: 1.5-4 %, a GUESS until PC pair d1 (ncmoe 3 vs 7) decides).
    prompt_gain: the table's measured prompt speed-up against 512 (6k-token prompt), None above 2048 where nothing is measured."""
    lay = model_layout(path)
    free = free if free is not None else free_vram_bytes()
    if not lay["n_layers"] or free is None or not lay["expert_bytes_per_layer"]:
        return []
    fixed = lay["other_bytes"] + _kv_or_flat(path, ctx, kv_q8)
    rows = []
    prefetch_fit_params(fit_params, path, ctx, ubs, kv_q8)
    for u in ubs:
        buf, src = rule_buffer_mib(path, lay, ctx, u, kv_q8, fit_params, ubs)
        blk = rule_block_mib(path, lay, u)
        room = free - fixed - buf * 1048576 - reserve_gb * 1e9 - blk * 1048576
        gl = max(0, min(lay["n_layers"], int(max(room, 0) // lay["expert_bytes_per_layer"])))
        old = compute_buffer_guess_mib(ctx, u)
        rows.append({"ub": u, "reserve_mib": int(buf), "source": src, "old_reserve_mib": int(old), "block_mib": int(blk), "gpu_layers": gl,
                     "n_cpu_moe": lay["n_layers"] - gl, "prompt_gain": _PROMPT_GAIN.get(u)})
    top = max(r["gpu_layers"] for r in rows)
    for r in rows:
        r["decode_pct"] = round(100 - DECODE_PCT_PER_LAYER * (top - r["gpu_layers"]), 1)
    return rows


def plan_table_lines(rows, ctx):
    out = [f"moe-cache: plan rule (MOE_CACHE_PLAN_RULE=1), context {ctx}: reserve for the GPU compute buffer by micro-batch (old = the table's value)"]
    out.append("moe-cache:   -ub  reserve MiB  (old)  source            block MiB  GPU layers  --n-cpu-moe  decode vs most layers  prompt vs 512 (table)")
    for r in rows:
        pg = f"{r['prompt_gain']:.2f}x" if r["prompt_gain"] else "not measured"
        out.append(f"moe-cache:   {r['ub']:>4}  {r['reserve_mib']:>11}  {r['old_reserve_mib']:>5}  {r['source']:<16}  {r['block_mib']:>9}  {r['gpu_layers']:>10}  {r['n_cpu_moe']:>11}  "
                   f"{r['decode_pct']:>19.1f} %  {pg}")
    out.append(f"moe-cache:   decode column: ~{DECODE_PCT_PER_LAYER} % per layer moved to the CPU (a guess, 1.5-4 %); the prompt column is the table's 6k-token figure, long prompts differ")
    return out


MTP_DRAFT_GIB = 0.85          # MTP's draft context on the GPU (planner/plan.py's figure, GLM on the PC, RESULTS 62); used until a load with MTP on was measured
GAP_SOURCES = ("nvidia-smi",)  # free figures that overstate what llama.cpp can fill: nvidia-smi's (total - used) is ~0.4 GiB above CUDA's free on the RTX 4060 laptop


def gpu_fixed_weights_bytes(lay):
    """The model's weights that live on the GPU besides the expert layers: everything but the experts and the tensors that stay on the CPU (token_embd). The one
    figure tune() and planner/multigpu.py count (follow-up 8)."""
    return lay["other_bytes"] - lay.get("cpu_only_bytes", 0)


def long_prompt_block_bytes(lay, ub):
    """Room kept free for the plugin's long-prompt GPU block at micro-batch ub (one expert layer plus the batch's results, rule of ours; see tune()). 0 for a dense model."""
    return ((1.6 if ub > 1024 else 1.35) * lay["expert_bytes_per_layer"] + (160 + 256) * 1048576) if lay["expert_bytes_per_layer"] else 0


def room_bytes(free, compute_mib, reserve_gb, block=0):
    """What is left of one card's free memory (bytes) after the compute buffer, the reserve and the long-prompt block: the subtraction tune() and multigpu.plan() both make."""
    return free - compute_mib * 1048576 - reserve_gb * 1e9 - block


def tune(path, ctx=8192, free=None, kv_q8=False, want_cache=True, reserve_gb=0.5, ub=None, fit_params=None, long_prompt=False, plugin=True, measured=None,
         free_source=None, mtp=False, keep_ub=False, loop_n_head=0):
    """One decision for everything that competes for VRAM: micro-batch, GPU expert cache or static expert layers, and how much is left. The only placement
    function of the launcher (planner map fix 1; planner/plan.py's formula no longer decides).

    Returns a dict: n_cpu_moe, ub, gpu_cache_mib (0 = use static layers), expected VRAM use, a list of reasons and "inputs": every size the plan used with its
    value, source (measured / probe / constant) and why, for the plan file. Rules come from measurements
    (docs/internal/PERFORMANCE_ANALYSIS.md): a GPU cache below about 1.5 GiB is slower than static layers; bigger micro-batches speed up prompts but need
    compute buffer that grows with the context; the disk lookup table never counts as GPU weight. ub: a micro-batch the user chose (the rest is
    planned around it; without this a chosen -ub left no room for the long-prompt block and long prompts ran on the CPU, RESULTS 61). long_prompt: the opt-in
    long-prompt mode (--long-prompt): -ub 4096 is tried first and taken only when the memory plan leaves room for it (no GPU cache wanted, no -ub chosen); else the usual ladder.
    plugin: the plugin runs (only then is the room for its long-prompt GPU block kept free: stock llama-server has no such block). measured: what the last real
    loads on this machine measured (planner/load_record.py lookup): {"non_expert_mib", "gap_mib", "draft_mib", "buffers": {ub: {"compute_mib", "context_mib"}}}; a missing part
    falls back to the header (weights, KV) or our tables (compute buffer, gap). free_source: where `free` came from ("nvidia-smi" overstates it by the CUDA
    reserve and context, SMI_GAP_GIB until measured). mtp: MTP drafting is on (its draft context needs GPU memory, MTP_DRAFT_GIB until a load with MTP was recorded).
    keep_ub: a stock cache on a CUDA llama-server keeps -ub 2048 (CACHE-1187; measured on one NVIDIA laptop only), see the cache branch below.
    """
    lay = model_layout(path)
    free = free if free is not None else free_vram_bytes()
    why = []
    if not lay["n_layers"] or free is None:
        return {"n_cpu_moe": 999, "ub": 512, "gpu_cache_mib": 0, "why": ["no GPU memory information: all experts stay on the CPU, default micro-batch"], "inputs": []}
    MiB = 1048576
    measured = measured or {}
    mbuf = {int(k): v for k, v in (measured.get("buffers") or {}).items()}
    inputs = []

    def note(name, value_mib, source, text):
        inputs.append({"setting": name, "value": round(value_mib), "source": source, "why": text})
    if measured.get("non_expert_mib"):
        weights = measured["non_expert_mib"] * MiB
        note("gpu_weights_mib", weights / MiB, "measured", "the model's weights on the GPU besides the expert layers, from llama-server's buffer lines at the last load on this machine and build")
    else:
        weights = gpu_fixed_weights_bytes(lay)
        note("gpu_weights_mib", weights / MiB, "probe", "the model's weights besides the expert layers, from the file's header (token_embd stays on the CPU); no load of this model was recorded here")
    ctx_m = next((v["context_mib"] for v in mbuf.values() if v.get("context_mib")), None)
    if ctx_m:
        kv = ctx_m * MiB
        note("gpu_kv_mib", ctx_m, "measured", "KV cache and recurrent state at this context and KV type, from llama-server's buffer lines at the last load")
    else:
        kv = _kv_or_flat(path, ctx, kv_q8)
        note("gpu_kv_mib", kv / MiB, "probe", "KV cache and recurrent state at this context and KV type, from the file's header")
    fixed = weights + kv
    if measured.get("gap_mib") is not None:
        gap = measured["gap_mib"] * MiB
        note("gpu_overhead_mib", gap / MiB, "measured", "what the free figure overstates (driver vs CUDA total, CUDA context, buffers made after load"
             + "), from the memory tables of the last load")
    elif free_source in GAP_SOURCES:
        gap = SMI_GAP_GIB * 2**30
        note("gpu_overhead_mib", gap / MiB, "constant", "nvidia-smi's free figure is about this much above what CUDA can fill (RTX 4060 laptop, 8146 vs 7745 MiB at idle); no load recorded here")
    else:
        gap = 0.0
        note("gpu_overhead_mib", 0, "constant", f"none counted for this free figure ({free_source or 'given'}); no load recorded here")
    draft = 0.0
    if mtp:
        if measured.get("draft_mib"):
            draft = measured["draft_mib"] * MiB
            note("mtp_draft_mib", draft / MiB, "measured", "MTP's draft context on the GPU (KV, recurrent state and compute buffer), from llama-server's buffer lines at the last load with MTP on")
        else:
            draft = MTP_DRAFT_GIB * 2**30
            note("mtp_draft_mib", draft / MiB, "constant", "MTP's draft context on the GPU (one model on our PC); no load with MTP on was recorded here")
    free = free - gap - draft

    rule = plan_rule_active(fit_params)
    if rule and fit_params:
        prefetch_fit_params(fit_params, path, ctx, [u for u in PLAN_UBS if u <= max(ub or 0, 2048)] + ([ub] if ub else []), kv_q8)

    def loop_bias_mib(u):
        # LAUNCHER-1284: a looped model's relative-position bias is a dense n_kv x n_tokens x n_head F32 tensor, two alive at once (launcher loop_compute_mib; 2.8 GiB measured at 32k -ub 512
        # on the laptop vs our 3 GiB estimate). The table and fit-params numbers are for a normal model; a measured load already contains it. loop_n_head 0 = not looped / not asked: 0.
        return 2 * ctx * u * loop_n_head * 4 / MiB if loop_n_head else 0

    def cbm(u, guess):
        v = cbm0(u, guess)
        return v if v is None or mbuf.get(u, {}).get("compute_mib") else v + loop_bias_mib(u)

    def cbm0(u, guess):
        if rule:              # MOE_CACHE_PLAN_RULE=1: the table only says whether u is a candidate (as before); the number is the rule's
            if not guess and compute_buffer_mib(ctx, u) is None:
                return None
            return rule_buffer_mib(path, lay, ctx, u, kv_q8, fit_params)[0]
        if mbuf.get(u, {}).get("compute_mib"):
            return mbuf[u]["compute_mib"]           # measured at the last load with this context, micro-batch and KV type
        return compute_buffer_guess_mib(ctx, u) if guess else compute_buffer_mib(ctx, u)
    # long prompts run their expert block on the GPU and upload one whole layer of experts for it: the micro-batch, the cache and static layers must
    # all leave that room (measured on Flash-Next: without it the block failed to allocate and prompt processing fell back to the CPU, 18-24 tok/s instead
    # of 50-60). The block needs 923 MiB for a 704 MiB layer at -ub 1024 (weights plus the batch's intermediate results), so 1.35x plus headroom
    # (measured at -ub 2048: 1107 MiB for the same layer, 1.57x: the batch's intermediate results grow with the micro-batch)
    # plus the 256 MiB the plugin itself requires free before it allocates a block (gpu_segment's headroom check): without it the reserve had 24 MiB of
    # slack on Flash-Next at -ub 2048, and at ctx 8192 the block found 423 MiB free where it needed 455, so prompts fell back to the CPU (~20 tok/s)
    def block_for(u):
        if not plugin:
            return 0              # stock llama-server: no plugin, no long-prompt GPU block (LAUNCHER-1177: 839 MiB were kept free for nothing on the laptop 35B)
        if rule:
            return rule_block_mib(path, lay, u) * 1048576
        return long_prompt_block_bytes(lay, u)
    best = None
    fixed_ub = ub
    # gpt-oss (2026-11-02): chunk 2048 at ctx 40960 gave prompts +73 % but decode 14.9 -> 9.9 with one layer's difference, cause not known: it keeps the old choice above 32k
    keep_1024 = ctx > 32768 and "gpt-oss" in (sampling_meta(path).get("arch") or "")
    try_4096 = bool(long_prompt) and not fixed_ub and not want_cache and not keep_1024
    for ub in ((fixed_ub,) if fixed_ub else (((4096,) if try_4096 else ()) + ((1024, 512) if keep_1024 else (2048, 1024, 512)))):
        buf = cbm(ub, bool(fixed_ub) or ub == 4096)
        if buf is None:
            continue
        block = block_for(ub)
        room = room_bytes(free, buf, reserve_gb, block) - fixed
        if room > 0 or ub == 512 or fixed_ub:
            best = (ub, room)
            break
    ub, room = best if best else (512, room_bytes(free, cbm(512, True), reserve_gb, block_for(512)) - fixed)
    block = block_for(ub)
    if fixed_ub:
        why.append(f"micro-batch {ub} as asked")
    elif want_cache and ub > 1024 and room > 0 and (plugin or not keep_ub or room < 1536 * 1048576):
        # plugin path and every non-CUDA stock path (keep_ub False): unchanged. Stock CUDA llama-server (CACHE-1187/LAUNCHER-1188, laptop 4060, 35B, b12207, one machine): -ub 1024 cost 14-34 % of the prompt
        # speed for no decode gain, so a stock cache keeps -ub 2048 as long as the cache still gets 1.5 GiB at that batch; below that the smaller batch buys the room
        ub = 1024
        buf = cbm(1024, False) or 0
        block = block_for(1024)
        room = room_bytes(free, buf, reserve_gb, block) - fixed
        why.append("GPU cache requested: micro-batch limited to 1024 (the cache gains more than a bigger batch)")
    cache_room = room
    cache_mib = int(cache_room // 1048576) if cache_room > 0 else 0
    expert_total = lay["expert_bytes_per_layer"] * lay["n_layers"]
    static_n = lay["n_layers"] - gpu_layers_that_fit(lay, room)       # the static answer for the same room (the launcher falls back to it when no cache size works out)
    if want_cache and expert_total > 0 and cache_mib >= 1536 and cache_mib * 1048576 >= 0.04 * expert_total:
        n_cpu_moe = 999
        why.append(f"GPU expert cache of {cache_mib} MiB ({100 * cache_mib * 1048576 / expert_total:.0f} % of the experts; "
                   f"{int(block // 1048576)} MiB kept free for the long-prompt GPU block)")
    else:
        cache_mib = 0
        gpu_layers = gpu_layers_that_fit(lay, room)
        n_cpu_moe = lay["n_layers"] - gpu_layers
        # --n-cpu-moe counts from layer 0, so the MTP head's block (the last one) is the first to go to the GPU: say the model's own layers apart
        mtpb = min(lay.get("mtp_blocks", 0), gpu_layers)
        where = (f"{gpu_layers - mtpb} of {lay['n_layers'] - lay['mtp_blocks']} layers of experts, plus the MTP head's block (used only with --mtp), stay on the GPU"
                 if mtpb else f"{gpu_layers} layers of experts stay on the GPU")
        why.append(f"{where} ({int(block // 1048576)} MiB kept free for the long-prompt GPU block"
                   + (")" if not want_cache else "; a cache would be under 1.5 GiB or under 4 % of the experts)"))
    why.append(f"micro-batch {ub} (long-prompt mode: the memory plan leaves room for it)" if ub == 4096 else f"micro-batch {ub} (about {_PROMPT_GAIN.get(ub, 1.0):.2f}x the prompt speed of 512)")
    comp = cbm(ub, True) if fixed_ub or not best or ub == 4096 else cbm(ub, False)
    if mbuf.get(ub, {}).get("compute_mib"):
        note("gpu_compute_mib", comp or 0, "measured", f"llama-server's compute buffer at -ub {ub} and this context, from the last load (or the size a failed load asked for)")
    elif rule:
        note("gpu_compute_mib", comp or 0, "measured" if fit_params else "constant", "compute buffer by MOE_CACHE_PLAN_RULE (" + ("llama-fit-params" if fit_params else "formula") + ")")
    else:
        note("gpu_compute_mib", comp or 0, "constant", f"compute buffer at -ub {ub}: our table (measured on Flash-Next on the PC); no load of this model, context and micro-batch recorded here")
    note("long_prompt_block_mib", block / MiB, "constant",
         "room for the plugin's long-prompt GPU block (one expert layer plus the batch's results, rule of ours)" if plugin else "0: stock llama-server runs no plugin, so no long-prompt block")
    note("gpu_reserve_mib", reserve_gb * 1e9 / MiB, "constant", "kept free for other programs and growth (0.5 GB rule of ours, more when the desktop holds GPU memory, --gpu-margin)")
    return {"n_cpu_moe": n_cpu_moe, "ub": ub, "gpu_cache_mib": cache_mib, "kv_mib": int(kv // 1048576),
            "compute_mib": comp, "loop_bias_mib": round(loop_bias_mib(ub)) if (comp and not mbuf.get(ub, {}).get("compute_mib")) else 0, "why": why, "inputs": inputs, "gap_mib": round(gap / MiB), "plugin": bool(plugin), "static_n_cpu_moe": static_n}


def fast_kv_hint(path, ctx, free=None, reserve_gb=0.5, ub=None, min_ctx=131072):
    """Exact mode keeps llama-server's f16 KV cache (identity with stock). At long contexts (>= 128k) q8_0 KV, which --fast uses, can free enough
    VRAM for more expert layers on the GPU: one plain line saying so when it gains 2+ layers or lets the context fit at all; None otherwise
    (idea N28; the owner keeps exact mode exact, so this only advises)."""
    if ctx < min_ctx:
        return None
    lay = model_layout(path)
    free = free if free is not None else free_vram_bytes()
    if not lay["n_layers"] or free is None:
        return None
    kw = {"reserve_gb": reserve_gb, "want_cache": False}
    if ub:
        kw["ub"] = ub
    a, b = tune(path, ctx, free=free, kv_q8=False, **kw), tune(path, ctx, free=free, kv_q8=True, **kw)
    kv16, kv8 = _kv_or_flat(path, ctx), _kv_or_flat(path, ctx, True)
    gained = a["n_cpu_moe"] - b["n_cpu_moe"]
    bigger_ub = b["ub"] > a["ub"]          # since 2026-11-03 the room q8_0 frees can go to -ub 2048 (256k: prompts +26 %) instead of more layers
    no_fit = lay["other_bytes"] + kv16 > free and lay["other_bytes"] + kv8 <= free
    if gained < 2 and not no_fit and not bigger_ub:
        return None
    what = ("the context does not fit the card's memory with an f16 KV cache, with q8_0 it does" if no_fit else
            f"{gained} more layers of experts on the GPU ({b['n_cpu_moe']} instead of {a['n_cpu_moe']} on the CPU)" if gained >= 2 else
            f"a prompt chunk of {b['ub']} instead of {a['ub']} tokens (faster long prompts; {b['n_cpu_moe']} instead of {a['n_cpu_moe']} layers of experts on the CPU)")
    return (f"at {ctx // 1024}k context, --fast would store the KV cache as q8_0 ({kv8 / 2**30:.1f} instead of {kv16 / 2**30:.1f} GiB): {what}. "
            "It changes some tokens (exact mode stays identical to stock)")


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        n, lay, free = fit_n_cpu_moe(p)
        t = tune(p)
        print("  tuned:", {k: v for k, v in t.items() if k != "why"}, "; ".join(t["why"]))
        print(f"{os.path.basename(p)[:44]:<44} layers {lay['n_layers']:>3}  experts/layer {lay['expert_bytes_per_layer']/1e9:5.2f} GB  other {lay['other_bytes']/1e9:5.2f} GB  free VRAM {free/1e9 if free else 0:4.1f} GB -> --n-cpu-moe {n} ({lay['n_layers']-n} layers on GPU)")
