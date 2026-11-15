"""Every load is a measurement (planner map fix 1, LAUNCHER-1177): llama-server's own buffer lines from the last real load, parsed from its log and
stored per machine + llama.cpp build + model (+ context, micro-batch and KV type for the buffers that depend on them). The GPU plan (gguf_plan.tune)
uses them instead of our tables; with no load recorded it falls back to the tables and says so (source "constant" in the plan file).

What llama-server prints (checked on the laptop logs of 2026-11-13 / 2026-11-15, a llama.cpp build of 2026-11):
  - at the default verbosity (3): NO buffer lines; only when a load fails, the size it could not allocate
    ("allocating 788.02 MiB on device 0: cudaMalloc failed" inside graph_reserve = the compute buffer);
  - at verbosity 4 (-lv 4, or LLAMA_ARG_LOG_VERBOSITY=4): "CUDA0 model buffer size = 5116.81 MiB", "CUDA0 KV buffer size", "CUDA0 RS buffer size",
    "CUDA0 compute buffer size", and a memory breakdown table at load (llama's own projection, with the device's total and free memory) and at exit
    ("| - CUDA0 (RTX 4060 Laptop GPU) | 7857 = 983 + (6668 = 5116 + 891 + 660) + 205 |": total = free + (self = model + context + compute) + unaccounted).
The launcher raises the verbosity to 4 through the environment for its own loads (the command stays the same) and hides the extra lines from the
terminal (bin/moe_cache.py: extra_log_line); MOE_CACHE_LOAD_RECORD=0 turns all of this off.

Stored (one JSON file, <cache dir>/loads.json, newest 400 entries per kind):
  weights  [machine, build, model]                       -> non_expert_mib: the model's GPU weights minus the expert layers placed there (does not depend on the context)
  buffers  [machine, build, model, ctx, ub, kv]          -> compute_mib (also from a failed load's size), context_mib (KV + recurrent state)
  overhead [machine, build, engine, mtp]                 -> gap_mib: what the free figure the plan used overstates the memory llama.cpp can fill
                                                            (driver total vs CUDA total, CUDA context, growth after load, MTP's draft context)
"""
import hashlib
import json
import os
import re
import threading
import time

KEEP = 400
GROWTH_CAP_MIB = 512        # growth after load counted at most (laptop 35B: 79 MiB of CUDA work buffers); more is other programs growing over a long run, not this server
FILE_NAME = "loads.json"
SCHEMA = 1
_LOCK = threading.Lock()

_NUM = r"([0-9]+(?:\.[0-9]+)?)"
_DEV = r"((?:CUDA|ROCm|Vulkan|SYCL|Metal|MUSA|CANN|OpenCL)[0-9]*)"
MODEL_RE = re.compile(r"load_tensors:\s+" + _DEV + r" model buffer size\s*=\s*" + _NUM + r" MiB")
KV_RE = re.compile(r"llama_kv_cache:\s+" + _DEV + r" KV buffer size\s*=\s*" + _NUM + r" MiB")
RS_RE = re.compile(r"llama_memory_recurrent:\s+" + _DEV + r" RS buffer size\s*=\s*" + _NUM + r" MiB")
COMPUTE_RE = re.compile(r"sched_reserve:\s+" + _DEV + r" compute buffer size\s*=\s*" + _NUM + r" MiB")
# | - CUDA0 (RTX 4060 Laptop GPU) |  7857 = 983 + (6668 =  5116 +     891 +     660) +         205 |
BREAKDOWN_RE = re.compile(r"memory_breakdown_print: \|\s+- " + _DEV + r"[^|]*\|\s*([0-9]+)\s*=\s*([0-9]+)\s*\+\s*\(\s*([0-9]+)\s*=\s*([0-9]+)\s*\+\s*([0-9]+)\s*\+\s*([0-9]+)\s*\)\s*\+\s*(-?[0-9]+)\s*\|")
OOM_ALLOC_RE = re.compile(r"allocating " + _NUM + r" MiB on device ([0-9]+): \w+ failed")
COMPUTE_FAIL_RE = re.compile(r"failed to allocate compute|graph_reserve: failed to allocate")
NCTX_RE = re.compile(r"llama_context: n_ctx\s*=\s*([0-9]+)")
NUB_RE = re.compile(r"llama_context: n_ubatch\s*=\s*([0-9]+)")
KVTYPE_RE = re.compile(r"llama_kv_cache: size =.*K \((\w+)\).*V \((\w+)\)")
LOG_LINE_RE = re.compile(r"^\s*[0-9.]+ ([DIWE]) (\S+)")


# tagged info lines that llama-server prints only at verbosity 4 ((tag, function) as the log shows them; laptop logs of 2026-11-13 / 2026-11-15, verbosity 3 vs 4)
LV4_ONLY = {("cmn", "common_conte"), ("cmn", "common_fit_a"), ("cmn", "common_init_"), ("slot", "create_check"), ("slot", "init_sampler"), ("slot", "load_model"),
            ("slot", "operator()"), ("slot", "process_sing"), ("slot", "prompt_clear"), ("spec", "common_specu"), ("srv", "get_availabl"), ("srv", "init"),
            ("srv", "init_listene"), ("srv", "load"), ("srv", "prompt_save"), ("srv", "server_prepa"), ("srv", "server_strea"), ("srv", "update"), ("srv", "update_slots")}
LV4_ONLY_TEXT = ("- skipping, slot is empty", "- checking sim", "sampler chain:", "sampler params:", "common_param: device_info", "common_param:   - ",
                 "common_param: system_info", "load_model: local path")
TAGGED_RE = re.compile(r"^\s*[0-9.]+ ([DIWE]) (\S+)\s+(\S+)")


def extra_log_line(line):
    """True for a line llama-server prints only at verbosity 4: an info or debug line of the library itself ("I load_tensors: ..."), or one of the tagged
    server lines in LV4_ONLY. The tagged lines verbosity 3 shows ("I srv  load_model: ...", "I slot print_timing ...") stay. Warnings and errors are never extra."""
    m = LOG_LINE_RE.match(line)
    if not m:
        return False
    if m.group(1) == "D" or (m.group(1) == "I" and m.group(2).endswith(":")):
        return True
    t = TAGGED_RE.match(line)
    return bool(t) and t.group(1) == "I" and ((t.group(2), t.group(3).rstrip(":")) in LV4_ONLY or any(x in line for x in LV4_ONLY_TEXT))


class LineFilter:
    """Which lines to show when the launcher raised the verbosity: extra_log_line, and lines without the log prefix that continue a hidden line (the chat
    template dump after "srv init: ... example_format:"). Plugin lines ("moe-cache: ...") and continuations of shown lines always show."""
    def __init__(self):
        self.last_hidden = False

    def show(self, line):
        if LOG_LINE_RE.match(line):
            self.last_hidden = extra_log_line(line)
            return not self.last_hidden
        if line.lstrip().startswith("moe-cache"):
            return True
        return not self.last_hidden


def parse_log(lines):
    """The GPU figures of one load (MiB) from llama-server's log lines. Returns a dict with any of:
    device, model_mib, kv_mib, rs_mib, compute_mib, context_mib, total_mib, free_start_mib, free_end_mib, unaccounted_mib, failed_compute_mib, n_ctx, n_ubatch, kv_type;
    {} when nothing was found. With more than one GPU only the one holding the most model weights counts."""
    models, kv, rs, comp, bds = {}, {}, {}, {}, []
    out = {}
    pending_oom = None
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        m = MODEL_RE.search(line)
        if m:
            models[m.group(1)] = models.get(m.group(1), 0.0) + float(m.group(2))
            continue
        m = KV_RE.search(line)
        if m:
            kv[m.group(1)] = kv.get(m.group(1), 0.0) + float(m.group(2))
            continue
        m = RS_RE.search(line)
        if m:
            rs[m.group(1)] = rs.get(m.group(1), 0.0) + float(m.group(2))
            continue
        m = COMPUTE_RE.search(line)
        if m:
            comp[m.group(1)] = max(comp.get(m.group(1), 0.0), float(m.group(2)))
            continue
        m = BREAKDOWN_RE.search(line)
        if m:
            bds.append({"device": m.group(1), "total": int(m.group(2)), "free": int(m.group(3)), "self": int(m.group(4)), "model": int(m.group(5)),
                        "context": int(m.group(6)), "compute": int(m.group(7)), "unaccounted": int(m.group(8))})
            continue
        m = OOM_ALLOC_RE.search(line)
        if m:
            pending_oom = float(m.group(1))
            continue
        if pending_oom is not None and COMPUTE_FAIL_RE.search(line):
            out["failed_compute_mib"] = pending_oom          # the failed allocation was the compute buffer (a model buffer fails in load_tensors)
            pending_oom = None
            continue
        for rx, key in ((NCTX_RE, "n_ctx"), (NUB_RE, "n_ubatch")):
            m = rx.search(line)
            if m:
                out[key] = int(m.group(1))
        m = KVTYPE_RE.search(line)
        if m:
            out["kv_type"] = f"{m.group(1)}/{m.group(2)}"
    devs = set(models) | {b["device"] for b in bds} | set(comp)
    if not devs:
        return out
    dev = max(devs, key=lambda d: (models.get(d, 0.0), max([b["model"] for b in bds if b["device"] == d] or [0])))
    out["device"] = dev
    if dev in models:
        out["model_mib"] = round(models[dev], 2)
    if dev in kv:
        out["kv_mib"] = round(kv[dev], 2)
    if dev in rs:
        out["rs_mib"] = round(rs[dev], 2)
    if dev in comp:
        out["compute_mib"] = round(comp[dev], 2)
    if dev in kv or dev in rs:
        out["context_mib"] = round(kv.get(dev, 0.0) + rs.get(dev, 0.0), 2)
    mine = [b for b in bds if b["device"] == dev]
    if mine:
        first, last = mine[0], mine[-1]
        out["total_mib"] = last["total"]
        out["free_start_mib"] = first["free"]            # printed while loading: the device's free memory before the weights (this process's context is already made)
        if len(mine) > 1:                                # the second table is printed at exit: what was really held
            out["free_end_mib"] = last["free"]
            out["unaccounted_mib"] = last["unaccounted"]
        out.setdefault("model_mib", float(last["model"]))
        out.setdefault("context_mib", float(last["context"]))
        out.setdefault("compute_mib", float(last["compute"]))
    return out


def build_id(llama_server):
    """Which llama.cpp build: the llama-server program and the llama / ggml libraries next to it (names, sizes, modification times), hashed; "" when the
    program cannot be found. A rebuilt or updated llama.cpp gets a new id, so its loads are measured anew."""
    import shutil
    try:
        p = llama_server if os.path.isfile(llama_server or "") else shutil.which(llama_server or "")
        if not p:
            return ""
        p = os.path.realpath(p)
        d = os.path.dirname(p)
        parts = [os.path.basename(p), os.path.getsize(p), os.stat(p).st_mtime_ns]
        for n in sorted(os.listdir(d)):
            low = n.lower()
            if (low.startswith(("libllama", "libggml", "llama", "ggml")) and (".so" in low or low.endswith((".dll", ".dylib")))):
                st = os.stat(os.path.join(d, n))
                parts += [n, st.st_size, st.st_mtime_ns]
        return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]
    except OSError:
        return ""


def model_id(path):
    """The model file (name, size, modification time); "" when unreadable."""
    try:
        st = os.stat(path)
        return f"{os.path.basename(path)}:{st.st_size}:{st.st_mtime_ns}"
    except (OSError, TypeError):
        return ""


def store_path(cache_dir):
    return os.path.join(cache_dir, FILE_NAME)


def _read(cache_dir):
    try:
        with open(store_path(cache_dir), encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and d.get("schema") == SCHEMA:
            return d
    except (OSError, ValueError):
        pass
    return {"schema": SCHEMA, "weights": {}, "buffers": {}, "overhead": {}}


def _key(*parts):
    return json.dumps(list(parts))


def kv_name(kv):
    """One text for a KV type as the planner holds it: False / True (q8_0) / "q4_0" / a (K, V) pair."""
    if isinstance(kv, (tuple, list)):
        return f"{kv[0]}/{kv[1]}"
    if kv is True:
        return "q8_0/q8_0"
    if not kv:
        return "f16/f16"
    return f"{kv}/{kv}"


def lookup(cache_dir, machine, build, model, ctx, ub, kv, engine, mtp):
    """The measured parts for this plan: {"non_expert_mib", "compute_mib", "context_mib", "gap_mib"} (any may be missing), each with its record time under
    "<name>_at". {} when nothing was recorded, or the machine, build or model is unknown."""
    if not (machine and build and model):
        return {}
    d = _read(cache_dir)
    out = {}
    w = d["weights"].get(_key(machine, build, model))
    if w and isinstance(w.get("non_expert_mib"), (int, float)):
        out["non_expert_mib"], out["non_expert_mib_at"] = float(w["non_expert_mib"]), w.get("at")
    b = d["buffers"].get(_key(machine, build, model, int(ctx), int(ub or 0), kv_name(kv)))
    for k in ("compute_mib", "context_mib"):
        if b and isinstance(b.get(k), (int, float)) and b[k] > 0:
            out[k], out[k + "_at"] = float(b[k]), b.get("at")
    o = d["overhead"].get(_key(machine, build, engine, bool(mtp)))
    if o and isinstance(o.get("gap_mib"), (int, float)):
        out["gap_mib"], out["gap_mib_at"] = float(o["gap_mib"]), o.get("at")
    return out


def entries(rec, ctx, ub, kv, engine, mtp, gpu_expert_mib, free_plan_mib, free_source):
    """The store entries one parsed load gives: {"weights": {...}, "buffers": {...}, "overhead": {...}} (each may be missing).
    gpu_expert_mib: the expert bytes this load placed on the GPU (from the model header and the --n-cpu-moe it ran with); free_plan_mib: the free GPU figure the
    plan used and free_source where it came from (the gap is only learned against the same kind of figure)."""
    now = time.strftime("%Y-%m-%d %H:%M")
    out = {}
    if rec.get("model_mib") and gpu_expert_mib is not None and rec["model_mib"] - gpu_expert_mib > 0:
        out["weights"] = {"non_expert_mib": round(rec["model_mib"] - gpu_expert_mib, 1), "at": now}
    comp = rec.get("compute_mib") or rec.get("failed_compute_mib")
    if comp or rec.get("context_mib"):
        if (rec.get("n_ctx") in (None, int(ctx))) and (rec.get("n_ubatch") in (None, int(ub or 0))):     # the log must be the plan's own load
            b = {"at": now}
            if comp:
                b["compute_mib"] = round(float(comp), 1)
            if rec.get("context_mib"):
                b["context_mib"] = round(float(rec["context_mib"]), 1)
            out["buffers"] = b
    if free_plan_mib and rec.get("free_start_mib") is not None and rec.get("total_mib"):
        # what the plan's free figure overstates: the plan's figure minus what the device really had free when this process started loading, plus what was
        # held at exit beyond the context made at start and llama's own buffers (CUDA work buffers, MTP's draft context, the plugin's block)
        held_at_start = rec["total_mib"] - rec["free_start_mib"]
        growth = min(GROWTH_CAP_MIB, max(0.0, float(rec["unaccounted_mib"]) - held_at_start)) if rec.get("unaccounted_mib") is not None else None
        if growth is not None:
            gap = float(free_plan_mib) - rec["free_start_mib"] + growth
            if 0 <= gap < 4096:                                # a figure outside this is a misread (another program freed or took memory meanwhile)
                out["overhead"] = {"gap_mib": round(gap, 1), "free_source": free_source, "at": now}
    return out


def save(cache_dir, machine, build, model, ctx, ub, kv, engine, mtp, ent):
    """Write the entries (atomic, private); returns True when something was written. Never raises."""
    if not (machine and build and model) or not ent:
        return False
    try:
        with _LOCK:
            d = _read(cache_dir)
            for kind, key in (("weights", _key(machine, build, model)), ("buffers", _key(machine, build, model, int(ctx), int(ub or 0), kv_name(kv))),
                              ("overhead", _key(machine, build, engine, bool(mtp)))):
                if kind in ent:
                    d[kind].pop(key, None)
                    d[kind][key] = ent[kind]
                    for old in list(d[kind])[:-KEEP]:
                        del d[kind][old]
            os.makedirs(cache_dir, exist_ok=True)
            p = store_path(cache_dir)
            tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1)
            os.replace(tmp, p)
        return True
    except (OSError, TypeError, ValueError):
        return False


class Recorder:
    """Collects one llama-server's log lines (feed) and stores what they measured (finish). ctx: the plan's dict with cache_dir, machine, build, model, ctx, ub,
    kv, engine, mtp, gpu_expert_mib, free_plan_mib, free_source. Writes at most twice: when the server is ready (load figures) and at exit (the exit table)."""
    def __init__(self, ctx):
        self.ctx, self.lines, self.saved = ctx, [], 0

    def feed(self, line):
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        if ("buffer size" in line or "memory_breakdown" in line or "llama_context: n_" in line or "llama_kv_cache: size" in line
                or "MiB on device" in line or "failed to allocate" in line):
            self.lines.append(line)

    def finish(self):
        c = self.ctx
        rec = parse_log(self.lines)
        if not rec:
            return {}
        ent = entries(rec, c["ctx"], c.get("ub"), c.get("kv"), c.get("engine"), c.get("mtp"), c.get("gpu_expert_mib"), c.get("free_plan_mib"), c.get("free_source"))
        if save(c["cache_dir"], c["machine"], c["build"], c["model"], c["ctx"], c.get("ub"), c.get("kv"), c.get("engine"), c.get("mtp"), ent):
            self.saved += 1
        return ent
