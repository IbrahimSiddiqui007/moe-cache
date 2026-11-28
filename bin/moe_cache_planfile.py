"""moe_cache_planfile: the launch plan as a data file (planner map fix 3, owner decision 7, rule R19).

For every setting the launcher decides: the value, where it came from, and one plain line of why.
Source is one of: probe (read from this machine at start), measured (a real load or the auto-tune
on this machine), constant (a rule or number of ours, not read from the machine), user (your flag),
llama (llama.cpp's own default; the launcher passes nothing).

Read-only on the plan: it describes what the launcher decided, it changes no decision and no command.
File: <cache dir>/plans/port-<PORT>.json (next to the auto-tune picks in <cache dir>/autotune/), or --plan-file PATH.
"""
import json
import os
import re
import sys

SCHEMA = 1
SOURCES = ("probe", "measured", "constant", "user", "llama")


def _after(args, *names):
    v = None
    for i, x in enumerate(args):
        if x in names and i + 1 < len(args):
            v = args[i + 1]
    return v


def _has(args, *names):
    return any(x in names or any(x.startswith(n + "=") for n in names if n.startswith("--")) for x in args)


_OT_CPU_RE = re.compile(r"blk\\\.\(([0-9|]+)\)\\\.ffn_[^=]*_exps\\\.weight=CPU")


def cpu_expert_layers(cmd):
    """LAUNCHER-1312: the expert layers the command puts on the CPU with the launcher's own -ot rule (expert placement by use count), [] when it has none.
    The same on Linux and Windows (it only reads the command)."""
    out = []
    for i, x in enumerate(cmd):
        v = x.split("=", 1)[1] if x.startswith("--override-tensor=") else (cmd[i + 1] if x in ("-ot", "--override-tensor") and i + 1 < len(cmd) else None)
        m = _OT_CPU_RE.search(v or "")
        if m:
            out += [int(t) for t in m.group(1).split("|") if t.isdigit()]
    return sorted(set(out))


def _row(name, value, source, why):
    if source not in SOURCES:
        raise ValueError(source)
    return {"setting": name, "value": value, "source": source, "why": why, "class": _class(name)}


def _class(name):
    """LAUNCHER-1225 step 2: the setting's answer class (exact / placement / quality / input), planner/never_list.json; "unknown" when the table cannot be read."""
    try:
        here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner")
        if here not in sys.path:
            sys.path.insert(0, here)
        import never_list
        return never_list.answer_class(name)[0]
    except Exception:
        return "unknown"


def build(plan):
    """The plan file's content for a finished LaunchPlan (a plain dict, JSON-ready). Describes the plan at start: a load retry may change it later."""
    a, cmd = plan.a, list(plan.cmd)
    uflags = getattr(plan, "user_flags", None) or {}
    uextra = list(getattr(plan, "user_extra", None) or [])
    own = list(getattr(plan, "argv_in", None) or [])          # the launcher's own flags as the user typed them (a.fast / a.stock are also set by defaults)
    cc = getattr(plan, "commit_changed", None) or {}           # what the Windows commit check changed after the plan was made
    tune = plan.autotune or {}
    tuned = plan.tuned or {}
    pl = plan.planner_info or {}
    pick = str(tune.get("picked") or "")
    picked = bool(tune.get("changed")) and tune.get("state") in ("cached", "tuned")
    free = pl.get("free_vram_GiB")
    env = dict(plan.env)
    rows = []

    def commit_row(name, value):
        b, f = cc[name]
        return _row(name, value, "probe", f"changed by the Windows memory (commit) check after the plan was made: {b} -> {f} (the commit figure was read at start)")

    # engine
    if _has(own, "--stock"):
        rows.append(_row("engine", plan.engine, "user", "--stock"))
    elif _has(own, "--always-plugin", "--fast", "--hybrid", "--approx"):
        rows.append(_row("engine", plan.engine, "user", "the plugin was asked for (--always-plugin, --fast, --hybrid or --approx)"))
    elif plan.engine == "stock":
        rows.append(_row("engine", "stock", "probe", "the model file plus 1.5 GB fits the RAM this server may use, so plain llama-server (rule of ours, RAM read at start)"))
    else:
        rows.append(_row("engine", "plugin", "probe", "the model file does not fit the RAM this server may use (or a GPU plan needs it): the expert-cache plugin runs (rule of ours, RAM read at start)"))

    # memory figures the plan used
    if free is not None:
        uv = getattr(a, "vram_user", getattr(a, "vram", None))
        rows.append(_row("free_gpu_gib", free, "user" if uv else "probe",
                         "your --vram" if uv else "free GPU memory read from the driver at start (before the reserves of the plan)"))
    if plan.avail:
        rows.append(_row("ram_for_server_gib", round(plan.avail / 2**30, 1), "user" if getattr(a, "ram", None) else "probe",
                         "your --ram" if getattr(a, "ram", None) else "RAM available at start, capped by our rule (total minus max(4 GiB, 12.5 %))"))

    # n-cpu-moe
    n = _after(cmd, "--n-cpu-moe", "-ncmoe")
    ot_cpu = cpu_expert_layers(cmd) if plan.ncmoe_user is None and n in (None, "0") and not _has(uextra, "-ot", "--override-tensor") and not _has(own, "-ot", "--override-tensor") else []
    if ot_cpu:                          # LAUNCHER-1312: --n-cpu-moe says 0 because the planner's layers go in by -ot; the table shows what the command really puts on the CPU
        srcs_ot = {r["source"] for r in tuned.get("inputs", []) if r["setting"] in ("gpu_weights_mib", "gpu_kv_mib", "gpu_compute_mib", "gpu_overhead_mib")}
        rows.append(_row("n_cpu_moe", len(ot_cpu), "measured" if (picked and pick.startswith("ncmoe")) or srcs_ot == {"measured"} else "probe",   # R19: counted from the free GPU memory read at run time, never a constant
                         "expert layers on the CPU, placed by use count with -ot (layers %s; the command's --n-cpu-moe is %s): the planner's count for the free GPU memory (the rows gpu_* below)"
                         % (",".join(map(str, ot_cpu)), n if n is not None else "absent")))
    elif n is None:
        rows.append(_row("n_cpu_moe", None, "llama", "no --n-cpu-moe passed (dense model, or llama.cpp places the layers)"))
    elif plan.ncmoe_user is not None or "ncmoe" in uflags or _has(uextra, "--n-cpu-moe", "-ncmoe"):
        rows.append(_row("n_cpu_moe", n, "user", "your --n-cpu-moe"))
    elif "n_cpu_moe" in cc:
        rows.append(commit_row("n_cpu_moe", n))
    elif picked and pick.startswith("ncmoe"):
        rows.append(_row("n_cpu_moe", n, "measured", f"the auto-tune pick '{pick}' for this model, machine and context (a real load was timed)"))
    elif n == "999":
        rows.append(_row("n_cpu_moe", n, "constant", "all experts on the CPU side: the GPU expert cache or --hybrid holds the hot ones"))
    elif tuned and not any("no GPU memory information" in w for w in tuned.get("why", [])):
        srcs = {r["source"] for r in tuned.get("inputs", []) if r["setting"] in ("gpu_weights_mib", "gpu_kv_mib", "gpu_compute_mib", "gpu_overhead_mib")}
        sizes = ("measured at the last load on this machine" if srcs == {"measured"} else "partly measured at the last load, partly from the header and our tables" if "measured" in srcs
                 else "from the model header and our tables: no load of this model was recorded here yet")
        rows.append(_row("n_cpu_moe", n, "measured" if srcs == {"measured"} else "constant" if "constant" in srcs else "probe", "expert layers that fit the free GPU memory" + (f" ({free} GiB)" if free is not None else "")
                         + f" after the other weights, KV cache, compute buffer, overhead and a reserve; sizes {sizes} (the rows gpu_* below)"))
    else:
        rows.append(_row("n_cpu_moe", n, "constant", "all experts on the CPU: no GPU memory figure was read, or the GPU plan could not be made"))
    # what the placement counted (planner map fix 1): each size with where it came from
    if tuned and n is not None and plan.ncmoe_user is None:
        for r in tuned.get("inputs", []):
            rows.append(_row(r["setting"], r["value"], r["source"], r["why"]))
    if n is not None:
        rec_on = os.environ.get("MOE_CACHE_LOAD_RECORD", "1") != "0"
        uv = getattr(a, "vram_user", getattr(a, "vram", None))
        rows.append(_row("load_record", rec_on and not uv, "user" if not rec_on or uv else "constant",
                         "MOE_CACHE_LOAD_RECORD=0" if not rec_on else "your --vram: the plan does not learn against a figure you set" if uv else
                         "every load (a real start or an auto-tune probe, with MTP too) stores llama-server's buffer sizes (its log at verbosity 4, extra lines hidden) and a failed GPU allocation for the next plan on this machine, driver and llama.cpp build"))

    # micro-batch
    ub, b = _after(cmd, "-ub", "--ubatch-size"), _after(cmd, "-b", "--batch-size")
    user_b = _has(uextra, "-b", "--batch-size")
    if ub is None:
        rows.append(_row("ub", None, "llama", "no -ub passed: llama.cpp's own micro-batch"))
    else:
        if str(getattr(a, "ub", "auto")).isdigit() or _has(uextra, "-ub", "--ubatch-size"):
            rows.append(_row("ub", ub, "user", "your --ub / -ub"))
        elif "ub" in cc:
            rows.append(commit_row("ub", ub))
        elif picked and pick.startswith("ub"):
            rows.append(_row("ub", ub, "measured", f"the auto-tune pick '{pick}'"))
        elif tuned:
            rows.append(_row("ub", ub, "probe", "the largest micro-batch from the ladder 4096 / 2048 / 1024 / 512 that fits the free GPU memory (buffer sizes are tables of ours)"))
        else:
            rows.append(_row("ub", ub, "constant", "the default micro-batch of ours"))
    if b is None:
        rows.append(_row("b", None, "llama", "no -b passed: llama.cpp's own batch"))
    elif user_b:
        rows.append(_row("b", b, "user", "your -b"))
    else:
        rows.append(_row("b", b, "constant", "the larger of -ub and 2048 (rule of ours)"))

    # KV cache type
    ctk, ctv = _after(cmd, "-ctk", "--cache-type-k"), _after(cmd, "-ctv", "--cache-type-v")
    if ctk is None and ctv is None:
        rows.append(_row("kv_type", "f16", "llama", "no KV type passed: llama.cpp's f16 (--exact, a short context or no GPU)"))
    elif _has(uextra, "-ctk", "--cache-type-k", "-ctv", "--cache-type-v"):
        rows.append(_row("kv_type", f"{ctk or 'f16'}/{ctv or 'f16'}", "user", "your -ctk / -ctv"))
    elif "kv_type" in cc:
        rows.append(commit_row("kv_type", f"{ctk}/{ctv}"))
    elif getattr(a, "dense_kv", False):
        rows.append(_row("kv_type", f"{ctk}/{ctv}", "probe", "f16 KV did not fit the free GPU memory next to the model: the next smaller KV type that fits"))
    elif getattr(a, "lp_kv", False):
        rows.append(_row("kv_type", f"{ctk}/{ctv}", "constant", "long-prompt mode (-ub 4096 with an 8-bit KV cache), on by default for a big model at agent-sized context (rule of ours, claim v038-longprompt-default)"))
    elif _has(own, "--fast", "--approx"):
        rows.append(_row("kv_type", f"{ctk}/{ctv}", "user", "8-bit KV cache from your --fast / --approx"))
    else:
        rows.append(_row("kv_type", f"{ctk}/{ctv}", "constant", "8-bit KV cache by default from 128k context (rule of ours)"))

    # threads and pinning
    cpumask, thr = _after(cmd, "-C", "--cpu-mask"), _after(cmd, "-t", "--threads")
    if _has(uextra, "-C", "--cpu-mask", "-Cr", "--cpu-range", "-t", "--threads"):
        rows.append(_row("threads", {"cpu_mask": cpumask, "threads": thr}, "user", "your -C / -t after --"))
    elif getattr(a, "leave_room", False) and thr is not None and cpumask is None:
        rows.append(_row("threads", {"cpu_mask": None, "threads": thr}, "constant", "--leave-room: all physical cores but two (read at start) for your own programs"))
    elif cpumask is not None:
        rows.append(_row("threads", {"cpu_mask": cpumask, "threads": thr, "strict": _after(cmd, "--cpu-strict")}, "probe",
                         "hybrid CPU: threads pinned to the performance cores found at start (core layout read from this machine; thread rule of ours)"))
    elif thr is not None:
        rows.append(_row("threads", {"cpu_mask": None, "threads": thr}, "constant", "thread count set by the launcher"))
    else:
        rows.append(_row("threads", None, "llama", "no pinning or thread count passed: llama.cpp's default"))

    # context
    c = _after(cmd, "-c", "--ctx-size")
    if _has(own, "--ctx") or "ctx" in uflags:
        rows.append(_row("context", c, "user", "your --ctx / -c"))
    elif getattr(a, "ctx", None):
        rows.append(_row("context", c, "constant", "the context stored with the model picked by `moe_cache.py pick`"))
    else:
        rows.append(_row("context", c, "constant", "default context: 65536 with --agent, 8192 otherwise (rule of ours)"))

    # load mode
    lm = _after(cmd, "--load-mode")
    if lm is None:
        rows.append(_row("load_mode", None, "llama", "no --load-mode passed"))
    elif _has(uextra, "--load-mode"):
        rows.append(_row("load_mode", lm, "user", "your --load-mode"))
    elif "load_mode" in cc:
        rows.append(commit_row("load_mode", lm))
    else:
        rows.append(_row("load_mode", lm, "constant", "mmap for mixture-of-experts models; dense models switch to none when the weights fit the GPU or RAM is short (rules of ours)"))

    # MTP drafting
    mtp = "draft-mtp" in cmd
    if getattr(a, "mtp", None) is not None:
        asked = "--mtp" if a.mtp else "--no-mtp"
        rows.append(_row("mtp", mtp, "user", asked if mtp == bool(a.mtp) else f"{asked} was asked but not applied (the model has no MTP head, llama-server lacks it, or --exact): see the notes"))
    elif _has(uextra, "--spec-type"):
        rows.append(_row("mtp", mtp, "user", "your --spec-type after --"))
    elif mtp and getattr(a, "approx_spec", False):
        rows.append(_row("mtp", True, "user", "from your --approx tier"))
    elif mtp:
        rows.append(_row("mtp", True, "constant", "on in --agent mode when the model has an MTP head and llama-server supports it (rule of ours, measured on one model)"))
    else:
        rows.append(_row("mtp", False, "constant", "off by default (on only in --agent mode for a model with an MTP head; never for dense models)"))

    # prompt cache
    cr = _after(cmd, "--cache-ram")
    if cr is None:
        rows.append(_row("cache_ram_mib", None, "llama", "no --cache-ram passed: llama.cpp's default"))
    elif _has(uextra, "--cache-ram", "-cram"):
        rows.append(_row("cache_ram_mib", cr, "user", "your --cache-ram"))
    elif "cache_ram_mib" in cc:
        rows.append(commit_row("cache_ram_mib", cr))
    else:
        rows.append(_row("cache_ram_mib", cr, "constant", "a cap of ours (2048 MiB; 0 on machines with under 16 GB RAM or when the commit check or a MOE_CACHE_ setting says so)"))

    # slots
    np_ = _after(cmd, "-np", "--parallel")
    if np_ is None:
        rows.append(_row("slots", None, "llama", "no -np passed: llama.cpp's default"))
    elif _has(uextra, "-np", "--parallel"):
        rows.append(_row("slots", np_, "user", "your -np"))
    elif picked and pick == "np1":
        rows.append(_row("slots", np_, "measured", "the auto-tune pick 'np1'"))
    else:
        rows.append(_row("slots", np_, "constant", "set by the launcher (agent mode)"))

    # environment settings that decide speed
    if "GGML_OP_OFFLOAD_MIN_BATCH" in env:
        rows.append(_row("offload_min_batch", env["GGML_OP_OFFLOAD_MIN_BATCH"], "constant", "stock run with CPU expert layers: batches below this stay on the CPU (value of ours, from one small PC)"))
    if "MOE_CACHE_RAM_LOCK" in env:
        rows.append(_row("ram_lock", env["MOE_CACHE_RAM_LOCK"], "measured" if picked and pick == "ramlock" else "user", "keep the model in RAM (the auto-tune pick 'ramlock')" if picked and pick == "ramlock" else "your MOE_CACHE_RAM_LOCK setting"))

    return {"schema": SCHEMA, "model": os.path.basename(plan.model), "port": plan.port, "engine": plan.engine, "context": plan.ctx,
            "note": "the plan at start; a load retry (out of memory, bad pick) may change it later",
            "autotune": {"state": tune.get("state"), "picked": tune.get("picked")} if tune else None,
            "settings": rows}


def summary(rec):
    """A short block of lines for the terminal: one line per decided setting."""
    out = ["moe-cache: launch plan (value, where it came from: probe / measured / constant / user / llama):"]
    for r in rec["settings"]:
        if r["value"] is None and r["source"] == "llama":
            continue
        v = json.dumps(r["value"]) if isinstance(r["value"], (dict, bool)) else str(r["value"])
        out.append(f"  {r['setting']:<19} {v:<28} {r['source']:<9} {r['why']}")
    return out


def default_path(cache_dir, port):
    return os.path.join(cache_dir, "plans", f"port-{port}.json")


def write(path, rec):
    """Write the file (private, atomic). Returns True when written; a failure never stops a launch."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")
        os.replace(tmp, path)
        return True
    except (OSError, TypeError, ValueError):
        return False


def remove(path, rec):
    """Delete the file when it is still the one this launcher wrote (a newer server on the same port keeps its own)."""
    try:
        with open(path, encoding="utf-8") as f:
            if json.load(f) == json.loads(json.dumps(rec)):
                os.unlink(path)
    except (OSError, ValueError):
        pass
