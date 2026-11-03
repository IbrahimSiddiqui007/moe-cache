#!/usr/bin/env python3
"""moe_cache.py models: the catalogue (models.json) for THIS PC. One line per model: fits / tight / no (the one fit rule, moe_cache_pick.fit), size, whether
it is already downloaded (and where), the measured speed with its source ("not measured" when nothing was measured on a PC like this one).
Models that do not fit are hidden unless --all. Reads the machine like `pick`; downloads nothing. Standard library only."""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_cache as mc  # noqa: E402
import moe_cache_pick as pk  # noqa: E402

DEFAULT_CTX = {"agents": 65536, "chat": 32768, "docs": 65536}
ORDER = {"fits": 0, "tight": 1, "no": 2}


def headline(k, spec):
    """The speed line of model k for this PC: the first decode number (shortest context) measured on a machine this PC is at least as strong as, with where it
    came from; None when nothing was measured on a PC like this. {text, value, ctx, machine, date, source, note}."""
    got = pk.measured_for(k, (spec.get("vram") or 0) / pk.GIB, (spec.get("ram_cap") or 0) / 1e9)
    dec = [(r, m) for r, m in got if r["metric"] == "decode_tps"]
    if not dec:
        return None
    r, m = dec[0]
    text = f"{r['value']:g} tok/s decode" + (f" at context {r['ctx']}" if r.get("ctx") else "") + f" ({m['name'].split(':')[0]}, {r['source']}, {r['date']})" + (f"; {r['note']}" if r.get("covers_fit") is False and r.get("note") else "")
    return {"text": text, "value": r["value"], "ctx": r.get("ctx"), "machine": r["machine"], "date": r["date"], "source": r["source"], "note": r.get("note")}


def rows(spec, use, ctx, with_all):
    out = []
    for k, e in pk.CAT.items():
        f = pk.fit(spec, k, ctx, use)
        if f["verdict"] == "no" and not with_all:
            continue
        loc = (spec.get("local") or {}).get(k)
        free = spec.get("disk_free")
        need = pk.need_free(k)
        out.append({"id": k, "name": e["name"], "fit": f, "bytes": pk.size_of(k, spec.get("local") or {}), "downloaded": bool(loc), "path": loc,
                    "disk_ok": bool(loc) or free is None or free >= need, "speed": headline(k, spec), "license": e.get("license"),
                    "mtp": e.get("mtp"), "uses": e["uses"], "warnings": e["warnings"], "source": f"https://huggingface.co/{e['hf']['repo']}"})
    out.sort(key=lambda r: (ORDER[r["fit"]["verdict"]], r["bytes"]))
    return out


def lines(spec, use, ctx, rs, hidden):
    gb = pk.gb
    out = [f"moe-cache models for this PC: RAM {gb(spec['ram_total'])} (the server may use {gb(spec['ram_cap'])}), "
           + (f"GPU {spec['gpu'] or '?'} {spec['vram'] / pk.GIB:.0f} GiB" if spec.get("vram") else "no GPU with its own memory")
           + f", disk {gb(spec['disk_free'])} free ({spec['disk_kind'] or 'type unknown'})",
           f"verdicts are for {pk.USE_SHORT[use]} at context {ctx} (--use, --ctx change that); fits = measured on a PC like this one, tight = runs but outside what we measured, no = will not run"]
    for r in rs:
        here = f"downloaded: {r['path']}" if r["downloaded"] else ("not downloaded" + ("" if r["disk_ok"] else f" (disk too small: needs {gb(pk.need_free(r['id']))} free)"))
        out.append(f"{r['fit']['verdict'].upper():5} {r['id']:20} {gb(r['bytes']):>9}   {here}")
        out.append(f"      {r['name']}: " + (r["speed"]["text"] if r["speed"] else "speed: not measured on a PC like this one"))
        out.append(f"      {r['fit']['reason']}")
    if hidden:
        out.append(f"({hidden} more do not fit this PC: --all lists them)")
    out.append("pick one for a use: moe_cache.py pick --use agents|chat|docs [--model ID]; the whole catalogue: models.json, docs/MODELS.md")
    return out


def models(argv, spec=None):
    ap = argparse.ArgumentParser(prog="moe_cache.py models", description="the model catalogue for this PC: fits / tight / no, size, downloaded or not, measured speed")
    ap.add_argument("--use", choices=list(pk.USES), default="agents", help="what the verdicts are for (default agents: coding agents)")
    ap.add_argument("--ctx", type=int, help="context in tokens (default: 65536 for agents and docs, 32768 for chat)")
    ap.add_argument("--all", action="store_true", help="also the models that do not fit this PC")
    ap.add_argument("--json", action="store_true", help="print the machine and the models as JSON")
    ap.add_argument("--models-dir", action="append", default=[], help="another folder with .gguf files (repeatable)")
    ap.add_argument("--llama-server", default=mc.default_llama_server())
    a = ap.parse_args(argv)
    spec = spec or pk.machine_specs(a.models_dir, a.llama_server)
    ctx = a.ctx or DEFAULT_CTX[a.use]
    every = rows(spec, a.use, ctx, True)
    rs = every if a.all else [r for r in every if r["fit"]["verdict"] != "no"]
    if a.json:
        print(json.dumps({"machine": spec, "use": a.use, "ctx": ctx, "models": rs}, indent=1))
    else:
        mc.say("\n".join(lines(spec, a.use, ctx, rs, len(every) - len(rs))))
    return 0


if __name__ == "__main__":
    sys.exit(models(sys.argv[1:]))
