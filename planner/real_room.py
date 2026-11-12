"""LAUNCHER-1131 (audit 2026-11-12 ranks 4 and 5): the second half of the real-room planner and "every load is a measurement".

1. Paging detector + learned ceiling: after the load the launcher reads its own server's dedicated and shared GPU memory (Windows counters); shared use above GPU_PAGED_MIB while the
   dedicated memory sits near the card's room is paging. The dedicated figure is then this GPU's ceiling for 30 days, the next start plans against it (the KV ladder steps down) and one
   'moe-cache: warning:' line goes out (the Monitor lifts those).
2. Every load is a measurement: the buffer lines llama-server prints while loading (model, KV, recurrent state, compute per device) plus the counters are stored per machine + model + context
   + KV types; the dense fit prefers the stored values over its tables for that exact plan.
No model, no GPU needed to test: the readers take injected text. Nothing here raises into the launcher (callers wrap it; the functions return None / '' on bad input)."""
import hashlib, json, os, re, subprocess, time

GPU_PAGED_MIB = 300         # shared use of our process above this, with dedicated memory near the room: paging
ROOM_KEEP_S = 30 * 86400    # a learned ceiling is trusted for 30 days
MEASURE_KEEP_S = 90 * 86400
NEAR_ROOM_MIB = 1024        # dedicated memory this close to the room (or closer) means the card is full
MEASURE_VERSION = 1


def _win_ps():
    from gguf_plan import _win_ps as w
    return w()


def process_counters_text(pid, timeout=8):
    """One PowerShell read of this process's '\\GPU Process Memory' Dedicated and Shared Usage ('path|value;...', what gguf_plan.parse_gpu_process_counters reads). '' when it fails."""
    pat = f"pid_{pid}_*"
    cmd = ("Get-Counter -Counter '\\GPU Process Memory(P)\\Dedicated Usage','\\GPU Process Memory(P)\\Shared Usage' -ErrorAction SilentlyContinue | "
           "ForEach-Object { ($_.CounterSamples | ForEach-Object { $_.Path + '|' + $_.CookedValue }) -join ';' }").replace("(P)", f"({pat})")
    try:
        return subprocess.run([_win_ps(), "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True, text=True, errors="replace", timeout=timeout).stdout
    except Exception:
        return ""


# ---- the learned ceiling ----

def _store_file(cache_dir):
    return os.path.join(cache_dir, "gpu_room.json")


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _write_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def room_ceiling(cache_dir, device, now=None):
    """The learned dedicated ceiling (MiB) of this GPU, or None (never paged, older than 30 days, unreadable)."""
    d = _read_json(_store_file(cache_dir)).get(device) or {}
    try:
        if d.get("ceiling_mib") and (now or time.time()) - float(d.get("t", 0)) <= ROOM_KEEP_S:
            return int(d["ceiling_mib"])
    except (TypeError, ValueError):
        pass
    return None


def room_learn(cache_dir, device, ded, sha, now=None, room_mib=None):
    """Record one reading of our own process. Paging = shared above GPU_PAGED_MIB while dedicated memory is within NEAR_ROOM_MIB of room_mib (the pre-load room; unknown: any shared use counts):
    the dedicated figure becomes the ceiling. Shared use with dedicated far below the room is staging, not paging. No paging with more dedicated memory than a stored ceiling, or a pre-load
    room clearly above it: the desktop gave room back, the ceiling goes. Returns True when the process is paging."""
    if ded is None:
        return False
    paging = sha is not None and sha > GPU_PAGED_MIB
    if paging and room_mib and ded < room_mib - NEAR_ROOM_MIB:
        paging = False
    st = _read_json(_store_file(cache_dir))
    d = st.get(device) or {}
    before = json.dumps(d, sort_keys=True)
    if paging:
        d.update(ceiling_mib=int(ded), paged_mib=int(sha), t=now or time.time())
    elif d.get("ceiling_mib") and (ded > d["ceiling_mib"] + 256 or (room_mib and room_mib > d["ceiling_mib"] + NEAR_ROOM_MIB)):
        d.pop("ceiling_mib", None); d.pop("paged_mib", None)
    if json.dumps(d, sort_keys=True) != before:
        st[device] = d
        _write_json(_store_file(cache_dir), st)
    return paging


def watch_room(pid, device, cache_dir, say, alive, ready, sampler=None, sleep=None, delays=(5, 25, 60), every=300, room_mib=None, on_reading=None):
    """After the load (ready() true) read our own process's dedicated and shared GPU memory at `delays`, then every `every` seconds while alive(): learn the ceiling and say one
    'moe-cache: warning:' line when paging starts. on_reading(ded, sha) gets every reading (the measurement store uses the first). Stops after 3 readings in a row without a counter.
    sampler(pid) -> text and sleep(s) are injectable for tests."""
    from gguf_plan import parse_gpu_process_counters
    sampler = sampler or process_counters_text
    sleep = sleep or time.sleep
    while alive() and not ready():
        sleep(1)
    warned = False; miss = 0; i = 0
    while alive():
        d = delays[i] if i < len(delays) else every
        i += 1
        sleep(d)
        if not alive():
            return
        ded, sha = parse_gpu_process_counters(sampler(pid), pid)
        if ded is None:
            miss += 1
            if miss >= 3:
                return
            continue
        miss = 0
        if on_reading:
            try:
                on_reading(ded, sha)
            except Exception:
                pass
        if room_learn(cache_dir, device, ded, sha, room_mib=room_mib) and not warned:
            warned = True
            say(f"moe-cache: warning: GPU memory is paging: {sha:.0f} MiB of this server sit in shared system memory (the card keeps {ded / 1024:.1f} GiB of dedicated memory for it, "
                "the desktop and other programs hold the rest), so decode is slower. The next start plans with that room and takes a smaller KV cache type (q8_0, then K q8_0 / V q4_0, "
                "then q4_0) or moves layers to RAM; closing programs that use the card gives it room back")


# ---- every load is a measurement ----

BUF_RE = re.compile(r"(\S+?)\s+(model|KV|compute|output|RS|Compute)\s+buffer size\s*=\s*([\d.]+)\s*MiB")
OFFLOAD_RE = re.compile(r"offloaded (\d+)/(\d+) layers")


def parse_load_buffers(lines):
    """The buffer lines llama-server prints while loading -> {"gpu": {kind: MiB summed over the GPU devices}, "cpu": {kind: MiB}, "offloaded": (n, total) or None}. A device whose name starts with
    CPU or HOST, or holds 'host' (CPU_Mapped, CUDA_Host, Vulkan_Host), counts as CPU. The LAST line of a device and kind wins: a recent llama.cpp first sizes the plan without allocating (its
    buffer lines read 0.00 MiB or a projection) and prints the real ones after. None when no buffer line was seen."""
    last, off = {}, None
    for l in lines:
        m = BUF_RE.search(l)
        if m:
            last[(m.group(1).strip(":"), m.group(2).lower())] = float(m.group(3))
            continue
        m = OFFLOAD_RE.search(l)
        if m:
            off = (int(m.group(1)), int(m.group(2)))
    if not last:
        return None
    gpu, cpu = {}, {}
    for (dev, kind), mib in last.items():
        tgt = cpu if re.match(r"(CPU|HOST)", dev, re.I) or "host" in dev.lower() else gpu
        tgt[kind] = tgt.get(kind, 0.0) + mib
    return {"gpu": gpu, "cpu": cpu, "offloaded": off}


def _flag(cmd, *names):
    for i, x in enumerate(cmd):
        for n in names:
            if x == n and i + 1 < len(cmd):
                return cmd[i + 1]
            if x.startswith(n + "="):
                return x.split("=", 1)[1]
    return None


def cmd_facts(cmd):
    """(model path, ctx, ctk, ctv, ub, np) of a llama-server command line (list of strings); None for a missing model. KV types default to f16, ub to 512, np to 1 (llama.cpp's defaults)."""
    model = _flag(cmd, "-m", "--model")
    if not model:
        return None
    ctk = _flag(cmd, "-ctk", "--cache-type-k"); ctv = _flag(cmd, "-ctv", "--cache-type-v")
    def num(x, d):
        try:
            return int(x)
        except (TypeError, ValueError):
            return d
    return model, num(_flag(cmd, "-c", "--ctx-size"), 0), ctk or "f16", ctv or ctk or "f16", num(_flag(cmd, "-ub", "--ubatch-size"), 512), num(_flag(cmd, "-np", "--parallel"), 1)


def measure_key(model, ctx, ctk, ctv, device):
    """The key of one measurement: the model file (name, size, modification second), the context, the KV types and the GPU. ub / np are stored inside, not in the key."""
    try:
        st = os.stat(model)
        ident = (os.path.basename(model), st.st_size, int(st.st_mtime))
    except OSError:
        ident = (os.path.basename(model), 0, 0)
    return hashlib.sha256(json.dumps([ident, ctx, ctk, ctv, device, MEASURE_VERSION]).encode()).hexdigest()[:20]


def _measure_file(cache_dir, key):
    return os.path.join(cache_dir, "measured", key + ".json")


def measure_save(cache_dir, cmd, device, buffers, ded=None, sha=None, now=None):
    """Store one load. Only a load that put every layer on the GPU is kept (a partial offload says nothing about the all-on-GPU need). Returns the key or None."""
    f = cmd_facts(cmd)
    if not f or not buffers or not buffers.get("gpu") or not buffers["gpu"].get("model") or not f[1]:
        return None
    off = buffers.get("offloaded")
    if not off or off[0] < off[1]:
        return None
    model, ctx, ctk, ctv, ub, npar = f
    key = measure_key(model, ctx, ctk, ctv, device)
    g = buffers["gpu"]
    rec = {"version": MEASURE_VERSION, "t": now or time.time(), "device": device, "ctx": ctx, "ctk": ctk, "ctv": ctv, "ub": ub, "np": npar,
           "model_mib": g.get("model", 0.0), "kv_mib": g.get("kv", 0.0), "rs_mib": g.get("rs", 0.0), "compute_mib": g.get("compute", 0.0) + g.get("output", 0.0),
           "cpu_mib": sum((buffers.get("cpu") or {}).values())}
    if ded is not None:
        rec["dedicated_mib"] = round(ded)
    if sha is not None:
        rec["shared_mib"] = round(sha)
    return key if _write_json(_measure_file(cache_dir, key), rec) else None


def measure_load(cache_dir, model, ctx, ctk, ctv, device, now=None):
    """The stored measurement of this exact plan (model file, context, KV types, GPU), or None (missing, older than 90 days, other version)."""
    d = _read_json(_measure_file(cache_dir, measure_key(model, ctx, ctk, ctv, device)))
    try:
        if d.get("version") == MEASURE_VERSION and (now or time.time()) - float(d.get("t", 0)) <= MEASURE_KEEP_S and d.get("model_mib"):
            return d
    except (TypeError, ValueError):
        pass
    return None


def measured_need_gib(m, ub=None, table_fixed_gib=0.2):
    """GiB the stored load needed on the GPU: model + KV + recurrent state + compute buffer. The compute buffer follows the ubatch size: with another ub than the one measured, the table's
    fixed figure stands in for it (and the result is marked approximate by the caller through the second value). -> (need GiB without margin, compute taken from the measurement?)"""
    same_ub = ub is None or int(ub) == int(m.get("ub", 0))
    comp = (m.get("compute_mib", 0.0) / 1024) if same_ub else table_fixed_gib
    return (m["model_mib"] + m.get("kv_mib", 0.0) + m.get("rs_mib", 0.0)) / 1024 + comp, same_ub
