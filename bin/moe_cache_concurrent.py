"""on-demand --concurrent (MM1, PROTOTYPE): the launch options that let several llama-servers run at the same time under ONE memory cap.

group_plan() is pure (the sizes, the cap, the free VRAM and the core list are arguments, so a test fakes the machine): it asks moe_cache_budget for the
RAM and VRAM split, cuts the CPU cores into disjoint slices (one thread per physical core, weighted like the memory), and returns for every model the
launcher options to put in front of its own (`--ram`, `--vram`) and the llama-server options after `--` (`-C mask --cpu-strict 1 -t N`). A model whose
share holds its whole file runs as plain stock llama-server (the launcher decides that from --ram, as for any fitting model). When the floors do not
fit under the cap there is no split: ok is False and on-demand keeps today's behaviour (one model at a time, swap).
Nothing here is speed-tested; the split is static (a quiet model keeps its share until it is released after 30 s idle)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_cache_budget as budget  # noqa: E402

GIB = 2**30
RAM_PER_SERVER = 3 * GIB        # prompt cache (--cache-ram 2048 MiB) + 1 GiB agent margin the launcher reserves per server (multi_model_gpu report, table 1)


def core_slices(cores, weights):
    """Split a list of CPU ids (one per physical core) into disjoint contiguous slices, at least one core each, sizes in proportion to `weights`
    (largest remainder). None when there are fewer cores than models (no pinning then)."""
    n = len(weights)
    if n == 0 or len(cores) < n:
        return None
    ws = [max(float(w), 1e-9) for w in weights]
    spare = len(cores) - n
    raw = [spare * w / sum(ws) for w in ws]
    take = [1 + int(r) for r in raw]
    order = sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    for i in order[: len(cores) - sum(take)]:
        take[i] += 1
    out, k = [], 0
    for t in take:
        out.append(cores[k:k + t]); k += t
    return out


def mask_of(cpus):
    return hex(sum(1 << c for c in cpus))


def group_plan(models, cap, vram_free=None, cores=None):
    """models: dicts with name, size (GGUF bytes), kind ("moe" | "dense"), optional floor, weight, extra (see moe_cache_budget.split_ram).
    Returns {"ok": bool, "models": {name: {"launcher": [..], "llama": [..], "engine": "stock"|"plugin", "ram_bytes": int}}, "why": [str]}."""
    ms = [dict(m, floor=float(m.get("floor", 3.5 * GIB)) + RAM_PER_SERVER) if m.get("kind", "moe") != "dense" else dict(m, extra=float(m.get("extra", 0)) + RAM_PER_SERVER) for m in models]
    ram = budget.split_ram(cap, ms)
    if not ram["ok"]:
        return {"ok": False, "models": {}, "why": ram["why"]}
    vram = budget.split_vram(vram_free, models) if vram_free else {}
    sl = core_slices(cores, [m.get("weight", 1.0) for m in models]) if cores else None
    out = {}
    for i, (m, r) in enumerate(zip(models, ram["models"])):
        launcher = ["--ram", r["ram_gb_flag"]]
        if m["name"] in vram:
            launcher += ["--vram", f"{vram[m['name']] / GIB:.2f}"]
        llama = ["-C", mask_of(sl[i]), "--cpu-strict", "1", "-t", str(len(sl[i]))] if sl else []
        out[m["name"]] = {"launcher": launcher, "llama": llama, "engine": r["engine"], "ram_bytes": r["ram_bytes"]}
    why = list(ram["why"])
    if sl is None:
        why.append("fewer physical cores than models: no CPU pinning")
    return {"ok": True, "models": out, "why": why}


def physical_core_ids():
    """One CPU id per physical core (the first thread), performance cores only on a hybrid CPU; the same result shape on Linux and Windows."""
    import moe_cache as mc
    if mc.WINDOWS:
        cores = mc.win_cores()                       # [(efficiency class, [logical CPUs])]
        if not cores:
            return []
        top = max(e for e, _ in cores)
        return sorted(c[0] for e, c in cores if e == top and c)
    try:
        base = "/sys/devices/cpu_core/cpus" if os.access("/sys/devices/cpu_core/cpus", os.R_OK) else None
        allowed = mc.parse_cpus(open(base).read()) if base else sorted(os.sched_getaffinity(0))
        seen, prim = set(), []
        for c in allowed:
            sib = tuple(mc.parse_cpus(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read()))
            if sib not in seen:
                seen.add(sib); prim.append(min(sib))
        return sorted(prim)
    except (OSError, ValueError):
        return []
