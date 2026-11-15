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


def _row(name, value, source, why):
    assert source in SOURCES, source
    return {"setting": name, "value": value, "source": source, "why": why}


def build(plan):
    """The plan file's content for a finished LaunchPlan (a plain dict, JSON-ready)."""
    a, cmd = plan.a, list(plan.cmd)
    uflags = getattr(plan, "user_flags", None) or {}
    uextra = list(getattr(plan, "user_extra", None) or [])
    tune = plan.autotune or {}
    tuned = plan.tuned or {}
    pl = plan.planner_info or {}
    picked = bool(tune.get("changed")) and tune.get("state") in ("cached", "tuned")
    free = pl.get("free_vram_GiB")
    rows = []

    # engine
    own = list(getattr(plan, "argv_in", None) or [])          # the launcher's own flags as the user typed them (a.fast / a.stock are also set by defaults)
    if _has(own, "--stock"):
        rows.append(_row("engine", plan.engine, "user", "--stock"))
    elif _has(own, "--always-plugin", "--fast", "--hybrid"):
        rows.append(_row("engine", plan.engine, "user", "the plugin was asked for (--always-plugin, --fast or --hybrid)"))
    elif plan.engine == "stock":
        rows.append(_row("engine", "stock", "probe", "the model file plus 1.5 GB fits the RAM this server may use, so plain llama-server (rule of ours, RAM read at start)"))
    else:
        rows.append(_row("engine", "plugin", "probe", "the model file does not fit the RAM this server may use (or a GPU plan needs it): the expert-cache plugin runs (rule of ours, RAM read at start)"))

    # n-cpu-moe
    n = _after(cmd, "--n-cpu-moe", "-ncmoe")
    if n is None:
        rows.append(_row("n_cpu_moe", None, "llama", "no --n-cpu-moe passed (dense model, GPU cache or llama.cpp places the layers)"))
    elif plan.ncmoe_user is not None or "ncmoe" in uflags or _has(uextra, "--n-cpu-moe", "-ncmoe"):
        rows.append(_row("n_cpu_moe", n, "user", "your --n-cpu-moe"))
    elif picked and (tune.get("picked") or "") not in ("", "C0"):
        rows.append(_row("n_cpu_moe", n, "measured", f"the auto-tune pick '{tune.get('picked')}' for this model, machine and context (measured on a real load)"))
    elif tuned and not any("no GPU memory information" in w for w in tuned.get("why", [])):
        rows.append(_row("n_cpu_moe", n, "probe", "layers that fit the free GPU memory" + (f" ({free} GiB)" if free is not None else "") + " after model parts, KV cache, compute buffer and a reserve (the sizes are tables of ours)"))
    else:
        rows.append(_row("n_cpu_moe", n, "constant", "the planner's answer from fixed sizes (no GPU memory figure was read)"))

    # micro-batch
    ub, b = _after(cmd, "-ub", "--ubatch-size"), _after(cmd, "-b", "--batch-size")
    if ub is None:
        rows.append(_row("ub", None, "llama", "no -ub passed: llama.cpp's own micro-batch"))
        rows.append(_row("b", b, "llama" if b is None else "constant", "llama.cpp's own batch" if b is None else "the batch size set with -ub"))
    else:
        if (str(getattr(a, "ub", "auto")).isdigit()) or _has(uextra, "-ub", "--ubatch-size"):
            rows.append(_row("ub", ub, "user", "your --ub / -ub"))
        elif picked and tune.get("picked") and "ub" in str(tune.get("picked")):
            rows.append(_row("ub", ub, "measured", f"the auto-tune pick '{tune.get('picked')}'"))
        elif tuned:
            rows.append(_row("ub", ub, "probe", "the largest micro-batch from the ladder 4096 / 2048 / 1024 / 512 that fits the free GPU memory (buffer sizes are tables of ours)"))
        else:
            rows.append(_row("ub", ub, "constant", "the default micro-batch of ours"))
        rows.append(_row("b", b, "user" if _has(uextra, "-b", "--batch-size") else "constant", "-b is the larger of -ub and 2048 (rule of ours)" if not _has(uextra, "-b", "--batch-size") else "your -b"))

    # KV cache type
    ctk, ctv = _after(cmd, "-ctk", "--cache-type-k"), _after(cmd, "-ctv", "--cache-type-v")
    if ctk is None and ctv is None:
        rows.append(_row("kv_type", "f16", "llama", "no KV type passed: llama.cpp's f16 (--exact, a short context or no GPU)"))
    elif _has(uextra, "-ctk", "--cache-type-k", "-ctv", "--cache-type-v"):
        rows.append(_row("kv_type", f"{ctk or 'f16'}/{ctv or 'f16'}", "user", "your -ctk / -ctv"))
    else:
        if getattr(a, "dense_kv", False):
            src, why = "probe", "f16 KV did not fit the free GPU memory next to the model: the next smaller KV type that fits"
        elif getattr(a, "lp_kv", False):
            src, why = "constant", "long-prompt mode (-ub 4096 with an 8-bit KV cache), on by default for a big model at agent-sized context (rule of ours, claim v038-longprompt-default)"
        elif _has(own, "--fast"):
            src, why = "constant", "8-bit KV cache from --fast or the 128k-and-up default (rule of ours)"
        else:
            src, why = "constant", "8-bit KV cache (rule of ours)"
        rows.append(_row("kv_type", f"{ctk}/{ctv}", src, why))

    # threads and pinning
    cpumask, thr = _after(cmd, "-C", "--cpu-mask"), _after(cmd, "-t", "--threads")
    if _has(uextra, "-C", "--cpu-mask", "-Cr", "--cpu-range", "-t", "--threads"):
        rows.append(_row("threads", {"cpu_mask": cpumask, "threads": thr}, "user", "your -C / -t after --"))
    elif cpumask is not None or thr is not None:
        rows.append(_row("threads", {"cpu_mask": cpumask, "threads": thr, "strict": _after(cmd, "--cpu-strict")}, "probe",
                         "hybrid CPU: threads pinned to the performance cores found at start (core layout read from this machine; thread rule of ours)"))
    else:
        rows.append(_row("threads", None, "llama", "no pinning or thread count passed: llama.cpp's default"))

    # context
    c = _after(cmd, "-c", "--ctx-size")
    if getattr(a, "ctx", None) or "ctx" in uflags:
        rows.append(_row("context", c, "user", "your --ctx / -c"))
    else:
        rows.append(_row("context", c, "constant", "default context: 65536 with --agent, 8192 otherwise (rule of ours)"))

    # load mode
    lm = _after(cmd, "--load-mode")
    if lm is None:
        rows.append(_row("load_mode", None, "llama", "no --load-mode passed"))
    elif _has(uextra, "--load-mode"):
        rows.append(_row("load_mode", lm, "user", "your --load-mode"))
    else:
        rows.append(_row("load_mode", lm, "constant", "mmap for mixture-of-experts models; dense models switch to none when the weights fit the GPU or RAM is short (rules of ours)"))

    # MTP drafting
    mtp = "draft-mtp" in cmd
    if getattr(a, "mtp", None) is not None and mtp == bool(getattr(a, "mtp", None)):
        rows.append(_row("mtp", mtp, "user", "--mtp" if mtp else "--no-mtp"))
    elif _has(uextra, "--spec-type"):
        rows.append(_row("mtp", mtp, "user", "your --spec-type after --"))
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
    else:
        rows.append(_row("cache_ram_mib", cr, "constant", "2048 MiB cap for the prompt cache, 0 on machines with under 16 GB RAM (rule of ours)"))

    # slots
    np_ = _after(cmd, "-np", "--parallel")
    if np_ is None:
        rows.append(_row("slots", None, "llama", "no -np passed: llama.cpp's default"))
    else:
        rows.append(_row("slots", np_, "measured" if picked and "np" in str(tune.get("picked")) else ("user" if _has(uextra, "-np", "--parallel") else "constant"),
                         "set with -np"))

    rec = {"schema": SCHEMA, "model": os.path.basename(plan.model), "port": plan.port, "engine": plan.engine, "context": plan.ctx,
           "autotune": {"state": tune.get("state"), "picked": tune.get("picked")} if tune else None,
           "settings": rows}
    return rec


def summary(rec):
    """A short block of lines for the terminal: one line per decided setting."""
    out = ["moe-cache: launch plan (value, where it came from: probe / measured / constant / user / llama):"]
    for r in rec["settings"]:
        if r["value"] is None and r["source"] == "llama":
            continue
        v = json.dumps(r["value"]) if isinstance(r["value"], (dict, bool)) else str(r["value"])
        out.append(f"  {r['setting']:<14} {v:<28} {r['source']:<9} {r['why']}")
    return out


def default_path(cache_dir, port):
    return os.path.join(cache_dir, "plans", f"port-{port}.json")


def write(path, rec):
    """Write the file (private, atomic). Returns True when written; a failure never stops a launch."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")
        os.replace(tmp, path)
        return True
    except OSError:
        return False
