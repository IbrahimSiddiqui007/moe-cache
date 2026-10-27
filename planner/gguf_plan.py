"""Minimal GGUF header reader (standard library only) and GPU placement planner.
   fit_n_cpu_moe(path, ctx) -> how many layers keep their experts on the CPU so that the rest fill the free VRAM."""
import glob, os, re, struct, subprocess

_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def _skip_value(f, t):
    if t in _FMT:
        f.read(struct.calcsize(_FMT[t]))
    elif t == 8:
        _read_str(f)
    elif t == 9:
        it = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if it in _FMT:
            f.read(n * struct.calcsize(_FMT[it]))
        else:
            for _ in range(n):
                _skip_value(f, it)
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
            elif (key.endswith(".block_count") or key.endswith(".expert_count")) and t in (4, 5, 10, 11):
                meta[key.rsplit(".", 1)[1]] = struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]
            else:
                _skip_value(f, t)
    if not all(k in meta for k in ("arch", "block_count", "expert_count")):
        return None
    return f"{meta['arch']}-{meta['block_count']}x{meta['expert_count']}"


def read_tensors(path):
    """-> [(name, size_bytes)] for the tensors stored in this one file."""
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
            else:
                _skip_value(f, t)
        infos = []
        for _ in range(n_t):
            name = _read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            f.read(8 * nd)
            f.read(4)
            off = struct.unpack("<Q", f.read(8))[0]
            infos.append((name, off))
        data_start = (f.tell() + align - 1) // align * align
        total = os.path.getsize(path) - data_start
    infos.sort(key=lambda x: x[1])
    out = []
    for i, (name, off) in enumerate(infos):
        end = infos[i + 1][1] if i + 1 < len(infos) else total
        out.append((name, end - off))
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


def header_info(path):
    """-> dict(arch, block_count, embd, nextn) from the GGUF header of the first file (missing numbers are 0, a missing arch is None)."""
    meta, arch = {}, None
    with open(shards(path)[0], "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        f.read(4)
        _, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            key = _read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            if key == "general.architecture" and t == 8:
                arch = _read_str(f)
            elif key.rsplit(".", 1)[-1] in ("block_count", "embedding_length", "nextn_predict_layers") and t in (4, 5, 10, 11):
                meta[key] = struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]
            else:
                _skip_value(f, t)
    get = lambda k: meta.get(f"{arch}.{k}", 0)  # noqa: E731
    return {"arch": arch, "block_count": get("block_count"), "embd": get("embedding_length"), "nextn": get("nextn_predict_layers")}


def mtp_head_check(model, head_file):
    """Does head_file (an MTP-only GGUF, mtp-*.gguf, used with -md) belong to model? -> (ok, reason, info). The head must have the model's
    architecture, width and trunk depth (block_count minus nextn layers, so a trunk that keeps the nextn key but dropped the tensors still matches)
    and carry blk.N.nextn.* tensors. info: bytes (file size, what the GPU holds), shared (no token_embd/output of its own: a "shared-" head that
    borrows them from the model, which needs a llama.cpp with tensor borrowing)."""
    m, h = header_info(model), header_info(head_file)
    names = [n for n, _ in read_tensors(head_file)]
    info = {"bytes": os.path.getsize(head_file), "shared": "token_embd.weight" not in names, "arch": h["arch"]}
    if not any(re.match(r"blk\.\d+\.nextn\.", n) for n in names):
        return False, "it has no blk.N.nextn.* tensors (not an MTP head)", info
    if h["arch"] != m["arch"]:
        return False, f"it is for architecture '{h['arch']}', the model is '{m['arch']}'", info
    if h["embd"] != m["embd"] or h["block_count"] - h["nextn"] != m["block_count"] - m["nextn"]:
        return False, (f"its shape ({h['block_count'] - h['nextn']} layers x {h['embd']}) differs from the model's "
                       f"({m['block_count'] - m['nextn']} layers x {m['embd']}): a head for another model of the same family"), info
    return True, "", info


def shards(path):
    m = re.match(r"(.*)-(\d+)-of-(\d+)\.gguf$", path)
    return sorted(glob.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf")) if m else [path]


def model_layout(path):
    """-> dict(n_layers, expert_bytes_per_layer, other_bytes, total_bytes)"""
    layers, other, table = {}, 0, 0
    for p in shards(path):
        for name, size in read_tensors(p):
            m = re.match(r"blk\.(\d+)\..*_exps\.(weight|scale)", name)
            if m:
                layers[int(m.group(1))] = layers.get(int(m.group(1)), 0) + size
            elif "per_layer_token_embd" in name:
                table += size              # n-gram lookup table (qwen4exp): read from the file when needed, never on the GPU (same rule as planner/plan.py)
            else:
                other += size
    n = (max(layers) + 1) if layers else 0
    per = sum(layers.values()) / len(layers) if layers else 0
    return {"n_layers": n, "expert_bytes_per_layer": per, "other_bytes": other, "table_bytes": table, "total_bytes": other + table + sum(layers.values())}


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


# Measured GPU compute buffers (MiB) of Qwen3.8-Flash-Next with a q8_0 KV cache, by context (rounded up between points) and micro-batch size.
# Other architectures differ somewhat; the numbers are the best available and the reserve below covers the usual spread.
_COMPUTE_MIB = {32768: {512: 770, 1024: 770, 2048: 990}, 65536: {512: 690, 1024: 920, 2048: None}, 131072: {512: 960, 1024: 1700, 2048: 3150}}
_PROMPT_GAIN = {512: 1.0, 1024: 1.26, 2048: 1.58}      # measured prefill speed-up of a bigger micro-batch (6k-token prompt)


def compute_buffer_mib(ctx, ub):
    key = next((k for k in sorted(_COMPUTE_MIB) if ctx <= k), None)
    return _COMPUTE_MIB[key].get(ub) if key else None


def tune(path, ctx=8192, free=None, kv_q8=False, want_cache=True, reserve_gb=0.5):
    """One decision for everything that competes for VRAM: micro-batch, GPU expert cache or static expert layers, and how much is left.

    Returns a dict: n_cpu_moe, ub, gpu_cache_mib (0 = use static layers), expected VRAM use and a list of reasons. Rules come from measurements
    (docs/PERFORMANCE_ANALYSIS.md): a GPU cache below about 1.5 GiB is slower than static layers; bigger micro-batches speed up prompts but need
    compute buffer that grows with the context; the disk lookup table never counts as GPU weight.
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
    for ub in (2048, 1024, 512):
        buf = compute_buffer_mib(ctx, ub)
        if buf is None:
            continue
        block = block_for(ub)
        room = free - fixed - buf * 1048576 - reserve_gb * 1e9 - block
        if room > 0 or ub == 512:
            best = (ub, room)
            break
    ub, room = best if best else (512, free - fixed - 0.55e9 - reserve_gb * 1e9 - block_for(512))
    block = block_for(ub)
    if want_cache and ub > 1024 and room > 0:
        ub = 1024
        buf = compute_buffer_mib(ctx, 1024) or 0
        block = block_for(1024)
        room = free - fixed - buf * 1048576 - reserve_gb * 1e9 - block
        why.append("GPU cache requested: micro-batch limited to 1024 (the cache gains more than a bigger batch)")
    cache_room = room
    cache_mib = int(cache_room // 1048576) if cache_room > 0 else 0
    expert_total = lay["expert_bytes_per_layer"] * lay["n_layers"]
    if want_cache and cache_mib >= 1536 and cache_mib * 1048576 >= 0.04 * expert_total:
        n_cpu_moe = 999
        why.append(f"GPU expert cache of {cache_mib} MiB ({100 * cache_mib * 1048576 / expert_total:.0f} % of the experts; "
                   f"{int(block // 1048576)} MiB kept free for the long-prompt GPU block)")
    else:
        cache_mib = 0
        gpu_layers = max(0, min(lay["n_layers"], int(max(room, 0) // lay["expert_bytes_per_layer"]))) if lay["expert_bytes_per_layer"] else 0
        n_cpu_moe = lay["n_layers"] - gpu_layers
        why.append(f"{gpu_layers} layers of experts stay on the GPU ({int(block // 1048576)} MiB kept free for the long-prompt GPU block"
                   + (")" if not want_cache else "; a cache would be under 1.5 GiB or under 4 % of the experts)"))
    why.append(f"micro-batch {ub} (about {_PROMPT_GAIN.get(ub, 1.0):.2f}x the prompt speed of 512)")
    return {"n_cpu_moe": n_cpu_moe, "ub": ub, "gpu_cache_mib": cache_mib, "kv_mib": int(kv // 1048576), "compute_mib": compute_buffer_mib(ctx, ub), "why": why}


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        n, lay, free = fit_n_cpu_moe(p)
        t = tune(p)
        print("  tuned:", {k: v for k, v in t.items() if k != "why"}, "; ".join(t["why"]))
        print(f"{os.path.basename(p)[:44]:<44} layers {lay['n_layers']:>3}  experts/layer {lay['expert_bytes_per_layer']/1e9:5.2f} GB  other {lay['other_bytes']/1e9:5.2f} GB  free VRAM {free/1e9 if free else 0:4.1f} GB -> --n-cpu-moe {n} ({lay['n_layers']-n} layers on GPU)")
