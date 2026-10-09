#!/usr/bin/env python3
"""Standard test protocol runner (see docs/TEST_PROTOCOL.md): every model, two placements, the same settings, stock and moe-cache, both orders.
   usage: bench/standard.py MODEL.gguf [MODEL2.gguf ...] [--limits 12,24] [--profiles cpu,gpu] [--tokens 100] [--port 8765]
   Needs the benchmark server (bin/moe-cache-bench --no-browser) to be running. Prints one row per cell; full results are saved like any other run."""
import argparse, json, os, sys, time, urllib.request

def api(base, path, body=None):
    r = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=30))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--limits", default="12,24")
    ap.add_argument("--profiles", default="cpu,gpu")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER", "/usr/bin/llama-server"))
    ap.add_argument("--plugin-lib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libggml-moe-cache.so"))
    ap.add_argument("--bg-max", type=float, default=25)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    rows = []
    for model in a.models:
        for prof in a.profiles.split(","):
            for lim in [float(x) for x in a.limits.split(",")]:
                cfg = {"llama_server": a.llama_server, "plugin_lib": os.path.abspath(a.plugin_lib), "model": model, "modes": ["stock", "plugin"], "limits": [int(lim) if lim == int(lim) else lim],
                       "ctx": a.ctx, "n_cpu_moe": "all" if prof == "cpu" else "fit", "pin": True, "speed": True, "speed_tokens": a.tokens, "speed_prompts": 4, "long": False,
                       "sweep": "", "custom": "", "both_orders": True, "gate_temp": 58, "gate_wait_s": 900, "bg_max": a.bg_max}
                jid = api(base, "/api/run", cfg)["id"]
                while api(base, f"/api/job/{jid}")["state"] not in ("done", "failed", "cancelled"):
                    time.sleep(5)
                rid = api(base, f"/api/job/{jid}")["result_id"]
                r = json.load(open(os.path.expanduser(f"~/.local/share/moe-cache-bench/results/{rid}.json")))
                sp, ident = r["summary"]["speed"], r["summary"]["identity"]
                key = lambda m: f"{int(lim) if lim == int(lim) else lim}|{m}"
                st, pl = (sp.get(key("stock")) or {}).get("steady"), (sp.get(key("plugin")) or {}).get("steady")
                ncm = r["runs"][0].get("n_cpu_moe")
                clean = all(x.get("clean") for x in r["runs"])
                rows.append((os.path.basename(model)[:34], prof, lim, ncm, st, pl, ident.get(str(int(lim))), clean, rid))
                print(f"{rows[-1][0]:<34} {prof:<4} {lim:>4g} GB  n-cpu-moe {ncm!s:>4}  stock {st or 0:6.2f}  moe-cache {pl or 0:6.2f}  ratio {(pl/st) if st and pl else 0:5.1f}x  identity {rows[-1][6]}  clean {clean}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
