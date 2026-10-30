"""Split one machine's memory between several llama-servers that run at the same time (PROTOTYPE, avenue 1 of
handover/parts/multi_model_gpu_20261030.md; nothing calls it yet; on-demand would use it to set `--ram` / `--vram` per model).

Why: the launcher sizes each server from what the machine has free when that server starts, and the cap (physical RAM minus max(4 GiB, 12.5 %), 28 GiB
on a 32 GiB PC) is per server. Two servers started one after the other each plan against the same cap, and a cache that fills later is not seen by the
other server's plan. A fixed split makes the sum of the parts stay under ONE cap.

Rules (plain water-filling):
  * every model first gets its floor: a dense model needs its whole file (nothing to cache), a MoE model needs its non-expert weights + KV + buffers
    (`floor`) plus a minimum expert cache (`min_cache`, the plugin's own minimum is 3 x its largest expert tensor);
  * the rest is shared in proportion to `weight` (how much the model is used, default 1), nobody gets more than it can use (the whole file plus
    `headroom`; a MoE model that gets that much runs as stock llama-server, as the launcher decides for a fitting model today);
  * if the floors already exceed the cap, there is no split: the answer says so and the caller should swap (today's on-demand) instead.
"""

GIB = 2**30


def split_ram(cap, models, headroom=1 * GIB, min_cache=1 * GIB):
    """cap: bytes the whole group may use. models: dicts with name, size (GGUF bytes), kind ("moe" | "dense"), optional floor (bytes, MoE only;
    default 3.5 GiB = the plugin's guess for everything that is not expert cache), weight (default 1.0), extra (bytes outside the GGUF, e.g. an
    mmproj projector, default 0). Returns {"ok": bool, "models": [{name, ram_bytes, engine, ...}], "why": [str]}."""
    rows = []
    for m in models:
        size = float(m["size"]) + float(m.get("extra", 0))
        full = size + headroom                                   # enough to hold everything: stock llama-server, nothing to manage
        if m.get("kind", "moe") == "dense":
            floor, want = full, full                             # no experts to cache: it needs all of it or it thrashes the file cache
        else:
            floor = min(full, float(m.get("floor", 3.5 * GIB)) + min_cache)
            want = full
        rows.append({"name": m["name"], "kind": m.get("kind", "moe"), "floor": floor, "want": want, "weight": max(float(m.get("weight", 1.0)), 1e-9), "share": floor})
    why = []
    need = sum(r["floor"] for r in rows)
    if need > cap:
        why.append(f"the floors alone ({need / GIB:.1f} GiB) are above the cap ({cap / GIB:.1f} GiB): run the models one at a time (swap)")
        return {"ok": False, "models": [{"name": r["name"], "ram_bytes": 0, "engine": "swap"} for r in rows], "why": why}
    left = cap - need
    open_ = [r for r in rows if r["share"] < r["want"] - 1]
    while left > 1 and open_:
        wsum = sum(r["weight"] for r in open_)
        given = 0.0
        for r in open_:
            g = min(left * r["weight"] / wsum, r["want"] - r["share"])
            r["share"] += g
            given += g
        left -= given
        open_ = [r for r in open_ if r["share"] < r["want"] - 1]
        if given < 1:
            break
    out = []
    for r in rows:
        fits = r["share"] >= r["want"] - 1
        out.append({"name": r["name"], "ram_bytes": int(r["share"]), "ram_gb_flag": f"{r['share'] / 1e9:.2f}",
                    "engine": "stock" if fits else "plugin", "kind": r["kind"]})
        if not fits and r["kind"] == "dense":
            why.append(f"{r['name']}: dense model, share {r['share'] / GIB:.1f} GiB is below its file: it cannot be cached partly")
    return {"ok": not any(w.endswith("cannot be cached partly") for w in why), "models": out, "why": why}


def split_vram(total, models, reserve=1 * GIB):
    """Static VRAM split by weight for the models that use a GPU: each gets total*weight/sum(weights) after one shared reserve. Returns {name: bytes}.
    (The planner then places as many expert layers as that share allows; a coordinator that moves VRAM at run time is a later step.)"""
    use = [m for m in models if m.get("gpu", True)]
    if not use:
        return {}
    free = max(total - reserve, 0)
    ws = sum(max(float(m.get("weight", 1.0)), 1e-9) for m in use)
    return {m["name"]: int(free * max(float(m.get("weight", 1.0)), 1e-9) / ws) for m in use}
