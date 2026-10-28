"""Minimal GGUF header reader (standard library only) and GPU placement planner.
   fit_n_cpu_moe(path, ctx) -> how many layers keep their experts on the CPU so that the rest fill the free VRAM."""
import glob, os, re, struct, subprocess

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
            # string arrays (the tokenizer: ~150k entries): the file size once, not one fstat per entry (that made a dry run take 16 s on Windows)
            size = os.fstat(f.fileno()).st_size
            if n * 8 > size - f.tell():
                raise ValueError("GGUF header: array past the end of the file")
            for _ in range(n):
                k = struct.unpack("<Q", f.read(8))[0]
                if k > size - f.tell():
                    raise ValueError("GGUF header: length past the end of the file")
                f.seek(k, 1)
        else:          # arrays of arrays: no GGUF writer makes them, and they would recurse as deep as the file says
            raise ValueError(f"GGUF header: array of type {it}")
    else:
        raise ValueError(f"unknown GGUF value type {t}")


def model_key(path):
    """'<architecture>-<layers>x<experts>' from the GGUF header (e.g. 'qwen4exp-48x512'): names the shipped starter usage profile for a model family,
    shared by all quants of it; None when the header lacks these keys."""
    meta = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return None
        f.read(4)
        _, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            key = _read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            if key == "general.architecture" and t == 8:
                meta["arch"] = _read_str(f)
                if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", meta["arch"]) or ".." in meta["arch"]:
                    return None        # it becomes part of a file name (profiles/<key>.usage): no / \\ or ..
            elif (key.endswith(".block_count") or key.endswith(".expert_count")) and t in (4, 5, 10, 11):
                meta[key.rsplit(".", 1)[1]] = struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]
            else:
                _skip_value(f, t)
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
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        struct.unpack("<I", f.read(4))
        n_t, n_kv = struct.unpack("<QQ", f.read(16))
        align = 32
        for _ in range(n_kv):
            key = _read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            if key == "general.alignment" and t in (4, 5):
                align = struct.unpack(_FMT[t], f.read(4))[0]
                if not (1 <= align <= 1 << 16 and align & (align - 1) == 0):
                    raise ValueError(f"GGUF header: alignment {align}")
            else:
                _skip_value(f, t)
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
    with open(shards(path)[0], "rb") as f:
        if f.read(4) != b"GGUF":
            return out
        f.read(4)
        _, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            key = _read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            if key in ("general.architecture", "general.name", "general.basename") and t == 8:
                out[{"general.architecture": "arch"}.get(key, key.split(".")[1])] = _read_str(f)
            elif (key == "general.file_type" or key.startswith("general.sampling.")) and t in _FMT:
                v = struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]
                if key == "general.file_type":
                    out["file_type"] = v
                else:
                    out["sampling"][key[len("general.sampling."):]] = v
            else:
                _skip_value(f, t)
    return out


def mtp_layers(path):
    """-> (head, declared): head = number of layers with multi-token-prediction tensors (blk.N.nextn.*) in the files, declared =
    <arch>.nextn_predict_layers from the header (0 when absent). A quant can keep the key but drop the tensors; only `head` says the head is there."""
    declared = 0
    files = shards(path)
    with open(files[0], "rb") as f:
        if f.read(4) != b"GGUF":
            return 0, 0
        f.read(4)
        _, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            key = _read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            if key.endswith(".nextn_predict_layers") and t in (4, 5, 10, 11):
                declared = struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]
            else:
                _skip_value(f, t)
    head = {m.group(1) for p in files for name, _ in read_tensors(p) for m in [re.match(r"blk\.(\d+)\.nextn\.", name)] if m}
    return len(head), declared


def shards(path):
    m = re.match(r"(.*)-(\d+)-of-(\d+)\.gguf$", path)
    return sorted(glob.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf")) if m else [path]


def model_layout(path):
    """-> dict(n_layers, expert_bytes_per_layer, other_bytes, total_bytes, mtp_blocks). n_layers counts the MTP head's block (blk.N.nextn.*) when it
    has experts (Qwen3.6: 41 = 40 + 1); mtp_blocks says how many of the highest blocks are that head."""
    layers, other, table, mtp = {}, 0, 0, set()
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
    n = (max(layers) + 1) if layers else 0
    per = sum(layers.values()) / len(layers) if layers else 0
    return {"n_layers": n, "expert_bytes_per_layer": per, "other_bytes": other, "table_bytes": table, "total_bytes": other + table + sum(layers.values()),
            "mtp_blocks": len(mtp & set(layers))}


def free_vram_bytes():
    """Free memory of the GPU: NVIDIA (nvidia-smi), else the AMD GPU with the most VRAM (amdgpu sysfs, as plan.py); None when there is none."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.split(",")
        return (float(out[0]) - float(out[1])) * 1048576
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
    return best[1] if best else None


def fit_n_cpu_moe(path, ctx=8192, reserve_gb=1.3, free=None):
    """Layers whose experts stay on the CPU. The GPU gets: every non-expert weight, KV cache and buffers (estimated), and as many whole layers of experts as fit."""
    lay = model_layout(path)
    free = free if free is not None else free_vram_bytes()
    if not lay["n_layers"] or free is None:
        return 999, lay, None
    fixed = lay["other_bytes"] + 0.55e9 + ctx * 40e3          # weights + compute buffers + rough KV for an 8k context
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
    fixed = lay["other_bytes"] + 0.55e9 + ctx * 40e3
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
_COMPUTE_MIB = {32768: {512: 770, 1024: 770, 2048: 990}, 65536: {512: 690, 1024: 920, 2048: None}, 131072: {512: 960, 1024: 1700, 2048: 3150}}
_PROMPT_GAIN = {512: 1.0, 1024: 1.26, 2048: 1.58}      # measured prefill speed-up of a bigger micro-batch (6k-token prompt)


def compute_buffer_mib(ctx, ub):
    key = next((k for k in sorted(_COMPUTE_MIB) if ctx <= k), None)
    return _COMPUTE_MIB[key].get(ub) if key else None


def compute_buffer_guess_mib(ctx, ub):
    """compute_buffer_mib, or for a pair that was not measured an estimate that errs high: the same micro-batch at the next larger measured
    context, else the largest 2048 value scaled to this micro-batch (a micro-batch the user chose must still leave the long-prompt block its room)."""
    b = compute_buffer_mib(ctx, ub)
    if b is None:
        b = next((_COMPUTE_MIB[k][ub] for k in sorted(_COMPUTE_MIB) if k >= ctx and _COMPUTE_MIB[k].get(ub)), None)
    if b is None:
        b = max(v.get(2048) or 0 for v in _COMPUTE_MIB.values()) * max(ub, 512) / 2048
    return b


def tune(path, ctx=8192, free=None, kv_q8=False, want_cache=True, reserve_gb=0.5, ub=None):
    """One decision for everything that competes for VRAM: micro-batch, GPU expert cache or static expert layers, and how much is left.

    Returns a dict: n_cpu_moe, ub, gpu_cache_mib (0 = use static layers), expected VRAM use and a list of reasons. Rules come from measurements
    (docs/PERFORMANCE_ANALYSIS.md): a GPU cache below about 1.5 GiB is slower than static layers; bigger micro-batches speed up prompts but need
    compute buffer that grows with the context; the disk lookup table never counts as GPU weight. ub: a micro-batch the user chose (the rest is
    planned around it; without this a chosen -ub left no room for the long-prompt block and long prompts ran on the CPU, RESULTS 61).
    """
    lay = model_layout(path)
    free = free if free is not None else free_vram_bytes()
    why = []
    if not lay["n_layers"] or free is None:
        return {"n_cpu_moe": 999, "ub": 512, "gpu_cache_mib": 0, "why": ["no GPU memory information: all experts stay on the CPU, default micro-batch"]}
    kv = ctx * 40e3 * (0.53 if kv_q8 else 1.0)
    fixed = lay["other_bytes"] + kv
    # long prompts run their expert block on the GPU and upload one whole layer of experts for it: the micro-batch, the cache and static layers must
    # all leave that room (measured on Flash-Next: without it the block failed to allocate and prompt processing fell back to the CPU, 18-24 tok/s instead
    # of 50-60). The block needs 923 MiB for a 704 MiB layer at -ub 1024 (weights plus the batch's intermediate results), so 1.35x plus headroom
    # (measured at -ub 2048: 1107 MiB for the same layer, 1.57x: the batch's intermediate results grow with the micro-batch)
    # plus the 256 MiB the plugin itself requires free before it allocates a block (gpu_segment's headroom check): without it the reserve had 24 MiB of
    # slack on Flash-Next at -ub 2048, and at ctx 8192 the block found 423 MiB free where it needed 455, so prompts fell back to the CPU (~20 tok/s)
    def block_for(u):
        return ((1.6 if u > 1024 else 1.35) * lay["expert_bytes_per_layer"] + (160 + 256) * 1048576) if lay["expert_bytes_per_layer"] else 0
    best = None
    fixed_ub = ub
    for ub in ((fixed_ub,) if fixed_ub else (2048, 1024, 512)):
        buf = compute_buffer_guess_mib(ctx, ub) if fixed_ub else compute_buffer_mib(ctx, ub)
        if buf is None:
            continue
        block = block_for(ub)
        room = free - fixed - buf * 1048576 - reserve_gb * 1e9 - block
        if room > 0 or ub == 512 or fixed_ub:
            best = (ub, room)
            break
    ub, room = best if best else (512, free - fixed - 0.55e9 - reserve_gb * 1e9 - block_for(512))
    block = block_for(ub)
    if fixed_ub:
        why.append(f"micro-batch {ub} as asked")
    elif want_cache and ub > 1024 and room > 0:
        ub = 1024
        buf = compute_buffer_mib(ctx, 1024) or 0
        block = block_for(1024)
        room = free - fixed - buf * 1048576 - reserve_gb * 1e9 - block
        why.append("GPU cache requested: micro-batch limited to 1024 (the cache gains more than a bigger batch)")
    cache_room = room
    cache_mib = int(cache_room // 1048576) if cache_room > 0 else 0
    expert_total = lay["expert_bytes_per_layer"] * lay["n_layers"]
    if want_cache and expert_total > 0 and cache_mib >= 1536 and cache_mib * 1048576 >= 0.04 * expert_total:
        n_cpu_moe = 999
        why.append(f"GPU expert cache of {cache_mib} MiB ({100 * cache_mib * 1048576 / expert_total:.0f} % of the experts; "
                   f"{int(block // 1048576)} MiB kept free for the long-prompt GPU block)")
    else:
        cache_mib = 0
        gpu_layers = max(0, min(lay["n_layers"], int(max(room, 0) // lay["expert_bytes_per_layer"]))) if lay["expert_bytes_per_layer"] else 0
        n_cpu_moe = lay["n_layers"] - gpu_layers
        # --n-cpu-moe counts from layer 0, so the MTP head's block (the last one) is the first to go to the GPU: say the model's own layers apart
        mtpb = min(lay.get("mtp_blocks", 0), gpu_layers)
        where = (f"{gpu_layers - mtpb} of {lay['n_layers'] - lay['mtp_blocks']} layers of experts, plus the MTP head's block (used only with --mtp), stay on the GPU"
                 if mtpb else f"{gpu_layers} layers of experts stay on the GPU")
        why.append(f"{where} ({int(block // 1048576)} MiB kept free for the long-prompt GPU block"
                   + (")" if not want_cache else "; a cache would be under 1.5 GiB or under 4 % of the experts)"))
    why.append(f"micro-batch {ub} (about {_PROMPT_GAIN.get(ub, 1.0):.2f}x the prompt speed of 512)")
    return {"n_cpu_moe": n_cpu_moe, "ub": ub, "gpu_cache_mib": cache_mib, "kv_mib": int(kv // 1048576),
            "compute_mib": compute_buffer_guess_mib(ctx, ub) if fixed_ub else compute_buffer_mib(ctx, ub), "why": why}


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        n, lay, free = fit_n_cpu_moe(p)
        t = tune(p)
        print("  tuned:", {k: v for k, v in t.items() if k != "why"}, "; ".join(t["why"]))
        print(f"{os.path.basename(p)[:44]:<44} layers {lay['n_layers']:>3}  experts/layer {lay['expert_bytes_per_layer']/1e9:5.2f} GB  other {lay['other_bytes']/1e9:5.2f} GB  free VRAM {free/1e9 if free else 0:4.1f} GB -> --n-cpu-moe {n} ({lay['n_layers']-n} layers on GPU)")
