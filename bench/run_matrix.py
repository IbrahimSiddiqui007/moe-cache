#!/usr/bin/env python3
"""Run the standard benchmark over many models and many memory limits, then print a table you can publish.

  bench/run_matrix.py --ollama --dir ~/models                 # every Ollama model and every .gguf in a folder (plan only: add --run)
  bench/run_matrix.py --ollama --dir ~/models --run           # really run it
  bench/run_matrix.py --model A.gguf --model B.gguf --limits 8,12,16,24 --profiles cpu,gpu --run

What it does for every MoE model it finds (dense models are skipped, moe-cache has nothing to manage there):
  - two placements ("profiles"): cpu = all experts on the CPU, gpu = as many expert layers on the GPU as fit
  - several memory limits (hard cgroup limits, swap off). Default limits for a model of S GB: 0.4 S, 0.6 S, 0.8 S and 1.1 S (rounded, at least 1 GB)
  - stock llama.cpp and moe-cache, both orders, token identity checked, runs that are not clean are flagged
It needs the benchmark server: bin/moe-cache-bench --no-browser   (this script starts one for you if none answers).
Output: a markdown table and a CSV next to this script's working directory (results-matrix-YYYYMMDD-HHMM.md/.csv).
Nothing in the output contains prompts, outputs or local paths (only the model file name).
"""
import argparse, csv, glob, json, os, re, shutil, struct, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from standard import api  # noqa: E402  (same HTTP helper and token handling as standard.py)


def gguf_is_moe(path):
    """True if the GGUF header has an expert count > 1. Reads only the metadata (no tensors)."""
    SC = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            def rd(fmt):
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            def rstr():
                return f.read(rd("<Q")).decode("utf-8", "replace")
            def rval(t):
                if t in SC:
                    return rd(SC[t])
                if t == 8:
                    return rstr()
                if t == 9:
                    et, n = rd("<I"), rd("<Q")
                    if et in SC and et != 7:
                        f.seek(n * struct.calcsize(SC[et]), 1)
                        return None
                    for _ in range(n):
                        rval(et)
                    return None
                raise ValueError(t)
            rd("<I"); rd("<Q"); n_kv = rd("<Q")
            for _ in range(n_kv):
                k = rstr(); v = rval(rd("<I"))
                if k.endswith(".expert_count"):
                    return int(v) > 1
    except Exception:
        return None
    return False


def split_ok(path):
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path)
    return m is None or m.group(1) == "00001"


def model_size_gb(path):
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return os.path.getsize(path) / 1e9
    tot = 0
    for i in range(1, int(m.group(2)) + 1):
        part = path[:m.start()] + f"-{i:05d}-of-{m.group(2)}.gguf"
        tot += os.path.getsize(part) if os.path.exists(part) else 0
    return tot / 1e9


def ollama_models():
    out = []
    try:
        names = [l.split()[0] for l in subprocess.check_output(["ollama", "list"], text=True, timeout=20).splitlines()[1:] if l.strip()]
    except Exception:
        return out
    for n in names:
        try:
            mf = subprocess.check_output(["ollama", "show", n, "--modelfile"], text=True, timeout=20)
        except Exception:
            continue
        for l in mf.splitlines():
            if l.startswith("FROM ") and os.path.isfile(l[5:].strip()):
                p = l[5:].strip()
                if os.path.getsize(p) > 100_000_000:     # skip projector files
                    out.append((n, p))
                break
    return out


def default_limits(size_gb, ram_gb):
    cands = [0.4 * size_gb, 0.6 * size_gb, 0.8 * size_gb, 1.1 * size_gb]
    cap = max(2, int(ram_gb * 0.8))
    lims = sorted({min(cap, max(1, int(round(c)))) for c in cands})
    return lims


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ollama", action="store_true", help="include every model of `ollama list`")
    ap.add_argument("--dir", action="append", default=[], help="folder with .gguf files (searched recursively; repeatable)")
    ap.add_argument("--model", action="append", default=[], help="a model file (repeatable)")
    ap.add_argument("--exclude", action="append", default=[], help="skip paths or names containing this text (repeatable)")
    ap.add_argument("--limits", help="memory limits in GB for every model, for example 8,12,16,24 (default: derived from each model's size)")
    ap.add_argument("--profiles", default="cpu,gpu")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER") or shutil.which("llama-server") or "/usr/bin/llama-server")
    ap.add_argument("--plugin-lib", default=os.path.join(HERE, "..", "build", "libggml-moe-cache.so"))
    ap.add_argument("--bg-max", type=float, default=25)
    ap.add_argument("--run", action="store_true", help="run it (without this flag only the plan is printed)")
    a = ap.parse_args()

    ram_gb = int(open("/proc/meminfo").readline().split()[1]) / 1048576
    cands = []
    for n, p in (ollama_models() if a.ollama else []):
        cands.append((n, p))
    for d in a.dir:
        for p in sorted(glob.glob(os.path.join(os.path.expanduser(d), "**", "*.gguf"), recursive=True)):
            if split_ok(p) and "mmproj" not in os.path.basename(p) and "vocab" not in os.path.basename(p):
                cands.append((os.path.basename(p), p))
    for p in a.model:
        cands.append((os.path.basename(p), os.path.abspath(p)))
    seen, models = set(), []
    for n, p in cands:
        rp = os.path.realpath(p)
        if rp in seen or any(x in p or x in n for x in a.exclude):
            continue
        seen.add(rp)
        moe = gguf_is_moe(p)
        if moe is None:
            print(f"skip (cannot read header): {n}")
            continue
        if not moe:
            print(f"skip (dense model, nothing to manage): {n}")
            continue
        models.append((n, p, model_size_gb(p)))

    profiles = a.profiles.split(",")
    plan = []
    for n, p, gb in models:
        lims = [float(x) for x in a.limits.split(",")] if a.limits else default_limits(gb, ram_gb)
        for prof in profiles:
            for lim in lims:
                plan.append((n, p, gb, prof, lim))
    print(f"\n{len(models)} MoE models, {len(plan)} cells (each cell = stock + moe-cache, both orders, about 6 to 15 minutes)")
    print(f"{'model':44s} {'GB':>6s}  profile  limits")
    for n, p, gb in models:
        lims = [float(x) for x in a.limits.split(",")] if a.limits else default_limits(gb, ram_gb)
        print(f"{n[:44]:44s} {gb:6.1f}  {','.join(profiles):7s}  {','.join(f'{x:g}' for x in lims)}")
    if not a.run:
        print("\nPlan only. Add --run to start. Start by closing browsers and other heavy programs: the tool waits for a cool, quiet CPU and flags runs that were not.")
        return 0

    base = f"http://127.0.0.1:{a.port}"
    try:
        api(base, "/api/config")
    except Exception:
        print("starting the benchmark server ...")
        subprocess.Popen([sys.executable, os.path.join(HERE, "moe_bench.py"), "--port", str(a.port), "--no-browser"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(40):
            time.sleep(0.5)
            try:
                api(base, "/api/config")
                break
            except Exception:
                pass
    sysinfo = api(base, "/api/system?ls=" + urllib.request.quote(a.llama_server))
    stamp = time.strftime("%Y%m%d-%H%M")
    rows = []
    for k, (n, p, gb, prof, lim) in enumerate(plan, 1):
        cfg = {"llama_server": a.llama_server, "plugin_lib": os.path.abspath(a.plugin_lib), "model": p, "modes": ["stock", "plugin"],
               "limits": [int(lim) if lim == int(lim) else lim], "ctx": a.ctx, "n_cpu_moe": "all" if prof == "cpu" else "fit", "pin": True,
               "speed": True, "speed_tokens": a.tokens, "speed_prompts": 4, "long": False, "sweep": "", "custom": "", "both_orders": True,
               "gate_temp": 58, "gate_wait_s": 900, "bg_max": a.bg_max}
        print(f"[{k}/{len(plan)}] {n}  {prof}  {lim:g} GB ...", flush=True)
        try:
            jid = api(base, "/api/run", cfg)["id"]
            while api(base, f"/api/job/{jid}")["state"] not in ("done", "failed", "cancelled"):
                time.sleep(5)
            job = api(base, f"/api/job/{jid}")
            rid = job.get("result_id")
            r = json.load(open(os.path.expanduser(f"~/.local/share/moe-cache-bench/results/{rid}.json")))
            sp, ident = r["summary"]["speed"], r["summary"]["identity"]
            key = lambda m: f"{int(lim) if lim == int(lim) else lim}|{m}"
            st = (sp.get(key("stock")) or {}).get("steady")
            pl = (sp.get(key("plugin")) or {}).get("steady")
            clean = all(x.get("clean") for x in r["runs"])
            ncm = r["runs"][0].get("n_cpu_moe")
            idn = ident.get(str(int(lim)))
            rows.append({"model": n, "size_gb": round(gb, 1), "profile": prof, "limit_gb": lim, "n_cpu_moe": ncm, "stock_tok_s": st, "moe_cache_tok_s": pl,
                         "ratio": round(pl / st, 2) if st and pl else None, "identity": f"{idn[0]}/{idn[1]}" if idn else "", "clean": clean})
        except Exception as e:
            rows.append({"model": n, "size_gb": round(gb, 1), "profile": prof, "limit_gb": lim, "n_cpu_moe": None, "stock_tok_s": None,
                         "moe_cache_tok_s": None, "ratio": None, "identity": f"error: {e}", "clean": False})
        print("   ", {k2: rows[-1][k2] for k2 in ("stock_tok_s", "moe_cache_tok_s", "ratio", "identity", "clean")}, flush=True)
        with open(f"results-matrix-{stamp}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    fmt = lambda v: "killed/none" if v in (None, 0) else f"{v:.1f}"
    lines = ["| Model | File GB | Placement | RAM limit | Stock tok/s | moe-cache tok/s | Ratio | Identity | Clean |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        place = "all experts on CPU" if r["profile"] == "cpu" else f"GPU takes layers ({r['n_cpu_moe']} on CPU)"
        lines.append(f"| {r['model']} | {r['size_gb']} | {place} | {r['limit_gb']:g} GB | {fmt(r['stock_tok_s'])} | {fmt(r['moe_cache_tok_s'])} | "
                     f"{('%.1fx' % r['ratio']) if r['ratio'] else '-'} | {r['identity']} | {'yes' if r['clean'] else 'NO'} |")
    head = (f"Hardware: {sysinfo.get('cpu')}, {sysinfo.get('ram_gb')} GB RAM, {'; '.join(sysinfo.get('devices') or [])}. "
            f"Steady generation speed = mean of requests 2 to 4 (100 tokens each), context {a.ctx}, hard memory limit with swap off, "
            f"start at or below 58 C, stock and moe-cache run in both orders. Rows marked 'Clean: NO' were not clean and should not be quoted.")
    out = f"results-matrix-{stamp}.md"
    open(out, "w").write(head + "\n\n" + "\n".join(lines) + "\n")
    print("\n" + head + "\n\n" + "\n".join(lines))
    print(f"\nwritten: {out}, results-matrix-{stamp}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
