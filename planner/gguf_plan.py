"""Minimal GGUF header reader (standard library only) and GPU placement planner.
   fit_n_cpu_moe(path, ctx) -> how many layers keep their experts on the CPU so that the rest fill the free VRAM."""
import glob, json, os, re, struct, subprocess

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
    return sorted(glob.glob(glob.escape(m.group(1)) + f"-*-of-{m.group(3)}.gguf")) if m else [path]      # the folder and name are the user's text


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


def read_meta(path):
    """-> {key: value} of the GGUF header: numbers and strings, arrays of numbers up to 4096 entries as lists (longer arrays are skipped)."""
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        f.read(4)
        _, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            key = _read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            if t in _FMT:
                out[key] = struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]
            elif t == 8:
                out[key] = _read_str(f)
            elif t == 9:
                pos = f.tell()
                it, n = struct.unpack("<I", f.read(4))[0], struct.unpack("<Q", f.read(8))[0]
                if it in _FMT and n <= 4096:
                    size = struct.calcsize(_FMT[it])
                    if n * size > _left(f):
                        raise ValueError("GGUF header: array past the end of the file")
                    out[key] = [struct.unpack(_FMT[it], f.read(size))[0] for _ in range(n)]
                else:
                    f.seek(pos)
                    _skip_value(f, t)
            else:
                _skip_value(f, t)
    return out


KV_ELEM = {False: 2.0, True: 34 / 32}       # bytes per K/V value: f16, q8_0 (--fast)


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
    if not isinstance(blocks, int) or not isinstance(heads, int) or isinstance(kvh, list) or isinstance(heads, list) or \
            g("attention.key_length_mla") is not None or g("attention.kv_lora_rank") is not None or g("attention.sliding_window") is not None:
        return None
    if heads <= 0:
        return None
    kvh = kvh or heads
    n_main = blocks - int(g("nextn_predict_layers", 0) or 0)
    k_len = g("attention.key_length") or (g("embedding_length", 0) // heads)
    v_len = g("attention.value_length") or k_len
    interval = int(g("full_attention_interval", 0) or 0)
    n_attn = n_main // interval if interval else n_main
    elem = KV_ELEM[bool(kv_q8)]
    per_tok = n_attn * kvh * (k_len + v_len) * elem
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


def desk_reserve_gb(total, free, base=0.5):
    """VRAM (GB) to keep free next to everything the tuner places, scaled with what other programs already hold (the desktop compositor, a browser).
    Their use grows while the server runs (a browser tab, a video): the more the desktop already takes, the more room is left so the long-prompt GPU
    block still finds its room (PC 2026-10-29: dwm 1.2 GB + browser 0.9 GB held 2.8 GB of 16; the block found 697 of 791 MiB and ran on the CPU,
    73 instead of ~290 tok/s). Returns (reserve_gb, used_by_others_gb). A headless card (under ~0.5 GB used) keeps the default `base`; at most 2.5 GB."""
    if not total or free is None:
        return base, 0.0
    used = max(0.0, (total - free) / 2**30)
    return min(2.5, max(base, 0.25 + 0.5 * used)), used


def free_vram_bytes():
    """Free memory of the GPU: NVIDIA (nvidia-smi), else the AMD GPU with the most VRAM (amdgpu sysfs, as plan.py); None when there is none."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, errors="replace", timeout=5).stdout
        out = out.strip().splitlines()[0].split(",")    # one line per GPU: the first, as plan.py (with two GPUs the whole text failed to parse: no GPU)
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
    fixed = lay["other_bytes"] + 0.55e9 + _kv_or_flat(path, ctx)          # weights + compute buffers + KV (from the header when it can)
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
    fixed = lay["other_bytes"] + 0.55e9 + _kv_or_flat(path, ctx)
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
_COMPUTE_MIB_NOT_PICKED = {131072: {2048: 3150}}
_PROMPT_GAIN = {512: 1.0, 1024: 1.26, 2048: 1.58}      # measured prefill speed-up of a bigger micro-batch (6k-token prompt)


# idea L01 (MOE_CACHE_COMPUTE_DB=FILE, unset = off): GPU compute buffers measured on THIS model (the "compute buffer size" lines of llama-server's log),
# kept by model layout, context and micro-batch. tune() takes the larger of the table above and a measured value; `python3 gguf_plan.py --record-buffer
# MODEL CTX UB LOGFILE` adds a measurement. A conservative store: it only ever raises a reserve, and a missing or damaged file means the table above.
def compute_db_key(lay):
    return f"{lay['n_layers']}:{int(lay['expert_bytes_per_layer'])}:{int(lay['other_bytes'])}"


def _compute_db():
    f = os.environ.get("MOE_CACHE_COMPUTE_DB")
    if not f:
        return None, {}
    try:
        with open(f) as fh:
            d = json.load(fh)
        return f, d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return f, {}


def measured_buffer_mib(lay, ctx, ub):
    """Largest measured buffer (MiB) of this model at the smallest measured context >= ctx with this micro-batch; None if there is none."""
    f, d = _compute_db()
    rows = [e for e in d.get(compute_db_key(lay), []) if isinstance(e, dict) and e.get("ub") == ub and isinstance(e.get("ctx"), int) and e["ctx"] >= ctx]
    if not rows:
        return None
    c0 = min(e["ctx"] for e in rows)
    return max(float(e["mib"]) for e in rows if e["ctx"] == c0)


def parse_compute_buffer_mib(log_text):
    """Largest GPU 'compute buffer size = X MiB' in a llama-server log (CPU lines are skipped), or None."""
    vals = [float(m.group(2)) for m in re.finditer(r"(\S+) compute buffer size =\s*([\d.]+) MiB", log_text) if not m.group(1).upper().startswith("CPU")]
    return max(vals) if vals else None


def record_compute_buffer(path, ctx, ub, mib):
    f, d = _compute_db()
    if not f:
        return False
    key = compute_db_key(model_layout(path))
    rows = [e for e in d.get(key, []) if isinstance(e, dict)]
    for e in rows:
        if e.get("ctx") == ctx and e.get("ub") == ub:
            e["mib"] = max(float(e["mib"]), float(mib))
            break
    else:
        rows.append({"ctx": int(ctx), "ub": int(ub), "mib": float(mib)})
    d[key] = rows
    tmp = f"{f}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh)
    os.replace(tmp, f)
    return True


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


def fit_params_buffer_mib(fit_params, path, ctx, ub, kv_q8=False, timeout=90):
    """Compute buffer (MiB) llama.cpp itself would reserve for this model, context and -ub (-b raised to match, or the tool clamps -ub to 2048); None on any failure."""
    key = (fit_params, path, ctx, ub, bool(kv_q8))
    if key not in _FIT_CACHE:
        v = None
        try:
            cmd = [fit_params, "-m", path, "-c", str(ctx), "-ub", str(ub), "-b", str(max(ub, 2048)), "-ngl", "99", "--n-cpu-moe", "999", "--fit-print", "on"]
            if kv_q8:
                cmd += ["-ctk", "q8_0", "-ctv", "q8_0"]
            v = parse_fit_params_compute_mib(subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout).stdout)
        except Exception:
            pass
        _FIT_CACHE[key] = v
    return _FIT_CACHE[key]


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
        d = measured_buffer_mib(lay, ctx, u)       # a buffer logged on this model (MOE_CACHE_COMPUTE_DB) can only raise it
        if d is not None and d > m:
            m, s = d, "measured log"
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


def plan_table(path, ctx=8192, free=None, kv_q8=False, reserve_gb=0.5, fit_params=None, ubs=PLAN_UBS):
    """One row per micro-batch candidate under the rule, for the launcher's --dry-run: reserve, block room, layers on the GPU and what that costs.
    decode_pct: decode speed against the candidate with the most GPU layers at ~2.5 % per layer (scout round 2: 1.5-4 %, a GUESS until the 2 x 2 test).
    prompt_gain: the table's measured prompt speed-up against 512 (6k-token prompt), None above 2048 where nothing is measured."""
    lay = model_layout(path)
    free = free if free is not None else free_vram_bytes()
    if not lay["n_layers"] or free is None or not lay["expert_bytes_per_layer"]:
        return []
    fixed = lay["other_bytes"] + _kv_or_flat(path, ctx, kv_q8)
    rows = []
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
        r["decode_pct"] = round(100 - 2.5 * (top - r["gpu_layers"]), 1)
    return rows


def plan_table_lines(rows, ctx):
    out = [f"moe-cache: plan rule (MOE_CACHE_PLAN_RULE=1), context {ctx}: reserve for the GPU compute buffer by micro-batch (old = the table's value)"]
    out.append("moe-cache:   -ub  reserve MiB  (old)  source            block MiB  GPU layers  --n-cpu-moe  decode vs most layers  prompt vs 512 (table)")
    for r in rows:
        pg = f"{r['prompt_gain']:.2f}x" if r["prompt_gain"] else "not measured"
        out.append(f"moe-cache:   {r['ub']:>4}  {r['reserve_mib']:>11}  {r['old_reserve_mib']:>5}  {r['source']:<16}  {r['block_mib']:>9}  {r['gpu_layers']:>10}  {r['n_cpu_moe']:>11}  "
                   f"{r['decode_pct']:>19.1f} %  {pg}")
    out.append("moe-cache:   decode column: ~2.5 % per layer moved to the CPU (a guess, 1.5-4 %); the prompt column is the table's 6k-token figure, long prompts differ")
    return out


def tune(path, ctx=8192, free=None, kv_q8=False, want_cache=True, reserve_gb=0.5, ub=None, fit_params=None):
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
    kv = _kv_or_flat(path, ctx, kv_q8)
    fixed = lay["other_bytes"] + kv

    rule = plan_rule_on()

    def cbm(u, guess):        # idea L01: the table, raised by a buffer measured on this model when MOE_CACHE_COMPUTE_DB has one
        if rule:              # MOE_CACHE_PLAN_RULE=1: the table only says whether u is a candidate (as before); the number is the rule's
            if not guess and compute_buffer_mib(ctx, u) is None:
                return None
            return rule_buffer_mib(path, lay, ctx, u, kv_q8, fit_params)[0]
        st = compute_buffer_guess_mib(ctx, u) if guess else compute_buffer_mib(ctx, u)
        m = measured_buffer_mib(lay, ctx, u)
        if m is None:
            return st
        return m if st is None and guess else (None if st is None else max(st, m))
    # long prompts run their expert block on the GPU and upload one whole layer of experts for it: the micro-batch, the cache and static layers must
    # all leave that room (measured on Flash-Next: without it the block failed to allocate and prompt processing fell back to the CPU, 18-24 tok/s instead
    # of 50-60). The block needs 923 MiB for a 704 MiB layer at -ub 1024 (weights plus the batch's intermediate results), so 1.35x plus headroom
    # (measured at -ub 2048: 1107 MiB for the same layer, 1.57x: the batch's intermediate results grow with the micro-batch)
    # plus the 256 MiB the plugin itself requires free before it allocates a block (gpu_segment's headroom check): without it the reserve had 24 MiB of
    # slack on Flash-Next at -ub 2048, and at ctx 8192 the block found 423 MiB free where it needed 455, so prompts fell back to the CPU (~20 tok/s)
    def block_for(u):
        if rule:
            return rule_block_mib(path, lay, u) * 1048576
        return ((1.6 if u > 1024 else 1.35) * lay["expert_bytes_per_layer"] + (160 + 256) * 1048576) if lay["expert_bytes_per_layer"] else 0
    best = None
    fixed_ub = ub
    plan_cost = os.environ.get("MOE_CACHE_PLAN_COST") not in (None, "", "0")
    # gpt-oss (2026-11-02): chunk 2048 at ctx 40960 gave prompts +73 % but decode 14.9 -> 9.9 with one layer's difference, cause not known: it keeps the old choice above 32k
    keep_1024 = ctx > 32768 and "gpt-oss" in (sampling_meta(path).get("arch") or "")
    for ub in ((fixed_ub,) if fixed_ub else ((1024, 512) if keep_1024 else (2048, 1024, 512))):
        # idea L02 (MOE_CACHE_PLAN_COST=1, default off): from 64k context on a long prompt dominates the request and -ub 2048 is the fastest measured
        # (RESULTS 61: 128k and 256k), so 2048 is a candidate there with its measured or estimated (high) buffer, not skipped for lack of a table row
        buf = cbm(ub, bool(fixed_ub) or (plan_cost and ctx >= 65536))
        if buf is None:
            continue
        block = block_for(ub)
        room = free - fixed - buf * 1048576 - reserve_gb * 1e9 - block
        if room > 0 or ub == 512 or fixed_ub:
            best = (ub, room)
            break
    ub, room = best if best else (512, free - fixed - cbm(512, True) * 1048576 - reserve_gb * 1e9 - block_for(512))
    block = block_for(ub)
    if fixed_ub:
        why.append(f"micro-batch {ub} as asked")
    elif want_cache and ub > 1024 and room > 0:
        ub = 1024
        buf = cbm(1024, False) or 0
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
            "compute_mib": cbm(ub, True) if fixed_ub or not best else cbm(ub, False), "why": why}


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
    if len(sys.argv) == 6 and sys.argv[1] == "--record-buffer":
        mib = parse_compute_buffer_mib(open(sys.argv[5], errors="replace").read())
        if mib is None or not record_compute_buffer(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), mib):
            raise SystemExit("nothing recorded (no GPU compute buffer line in the log, or MOE_CACHE_COMPUTE_DB is not set)")
        print(f"recorded {mib:.0f} MiB")
        raise SystemExit(0)
    for p in sys.argv[1:]:
        n, lay, free = fit_n_cpu_moe(p)
        t = tune(p)
        print("  tuned:", {k: v for k, v in t.items() if k != "why"}, "; ".join(t["why"]))
        print(f"{os.path.basename(p)[:44]:<44} layers {lay['n_layers']:>3}  experts/layer {lay['expert_bytes_per_layer']/1e9:5.2f} GB  other {lay['other_bytes']/1e9:5.2f} GB  free VRAM {free/1e9 if free else 0:4.1f} GB -> --n-cpu-moe {n} ({lay['n_layers']-n} layers on GPU)")
