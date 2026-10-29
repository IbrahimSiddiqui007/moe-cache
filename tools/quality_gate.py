#!/usr/bin/env python3
"""Quality gate for the approximate mode (docs/QUALITY_GATE.md): KL divergence and top-1 agreement against stock, plus the 202-task set.

Replaces token identity for the approximate levels. Standard library only (Linux and Windows).
  quality_gate.py --llama-server BIN --perplexity BIN --model GGUF --lib PLUGIN --level kv|fast|fewer|exact [--smoke]
                  [--n-cpu-moe N] [--ngl 99] [--ctx 4096] [--chunks 3] [--text FILE] [--tasks-file FILE] [--no-tasks] [--no-kl] [--out DIR]
                  [--env K=V ...] [--args "..."] [--common-args "..."] [--ram-env K=V ...]     (extra plugin environment / llama.cpp arguments for the candidate only, on top of the level)
Exit status: 0 = pass, 1 = a limit missed, 2 = the plugin was not active or a run failed.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KV = ["-ctk", "q8_0", "-ctv", "q8_0"]
# level -> (candidate plugin env, candidate llama.cpp args, max mean KLD, min same-top %, tasks tolerance)   (docs/QUALITY_GATE.md)
LEVELS = {
    "exact": ({}, [], 1e-5, 100.0, 0),
    "kv": ({}, KV, 0.015, 95.0, 2),
    "fast": ({"MOE_CACHE_ROUTE": "skip"}, KV, 0.040, 90.0, 4),
    "fewer": ({"MOE_CACHE_ROUTE": "skip", "MOE_CACHE_FEWER": "2"}, KV, 0.060, 88.0, 5),
}
SMOKE = {"exact": (1e-5, 100.0, 0), "kv": (0.25, 70.0, 10), "fast": (0.25, 70.0, 10), "fewer": (0.25, 70.0, 10)}
LOOP_EXTRA = 2


def plugin_env(server, lib):
    """Environment that makes this llama-server load the plugin (bin/_moe_cache_env.sh in Python)."""
    if os.name == "nt":
        return {"GGML_BACKEND_PATH": lib}
    try:
        ldd = subprocess.run(["ldd", shutil.which(server) or server], capture_output=True, text=True).stdout
    except OSError:
        ldd = ""
    return {"LD_PRELOAD": lib, "MOE_CACHE_PRELOAD": "1"} if "libggml-cpu" in ldd else {"GGML_BACKEND_PATH": lib}


def run_ppl(a, base_file, test, env, extra, log):
    cmd = [a.perplexity, "-m", a.model, "-f", a.text, "-c", "512", "--chunks", str(a.chunks), "-b", "8", "-ub", "8", "-ngl", str(a.ngl),
           "--n-cpu-moe", str(a.n_cpu_moe), "--kl-divergence-base", base_file] + a.common_args.split() + extra + (["--kl-divergence"] if test else [])
    e = dict(os.environ, **env)
    print("  $", " ".join(cmd), flush=True)
    t = time.time()
    p = subprocess.run(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    open(log, "w", encoding="utf-8").write(p.stdout)
    print(f"  done in {time.time() - t:.0f} s, exit {p.returncode}", flush=True)
    if p.returncode != 0:
        print("FAIL: llama-perplexity failed; log", log); sys.exit(2)
    return p.stdout


def num(text, pat):
    m = re.search(pat + r"\s*:?\s*(-?[0-9.]+)(?:\s*±\s*([0-9.]+))?", text)
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else None) if m else (None, None)


def serve_and_run_tasks(a, env, extra, tag, port):
    out = os.path.join(a.out, f"tasks_{tag}.json")
    cmd = [a.llama_server, "-m", a.model, "-ngl", str(a.ngl), "--n-cpu-moe", str(a.n_cpu_moe), "-c", str(a.ctx), "-np", "1", "-fa", "on",
           "--no-warmup", "--port", str(port), "--seed", "42"] + a.common_args.split() + extra
    logf = open(os.path.join(a.out, f"server_{tag}.log"), "w")
    e = dict(os.environ, MOE_CACHE_STATS="1", **env)
    srv = subprocess.Popen(cmd, env=e, stdout=logf, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            if srv.poll() is not None:
                print("FAIL: server exited early; log", logf.name); sys.exit(2)
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).status == 200:
                    break
            except Exception:
                time.sleep(2)
        else:
            print("FAIL: server did not start"); sys.exit(2)
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "quality_yardstick.py"), "run", "--url", f"http://127.0.0.1:{port}", "--out", out,
                            "--label", tag, "--tasks", a.tasks_file, "--tasks-only", "--task-n-predict", "64"] + (["--no-think"] if a.no_think else []),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        print("  " + (r.stdout.strip().splitlines() or [""])[-1])
        if r.returncode != 0:
            print("FAIL: yardstick run failed:", r.stdout[-500:]); sys.exit(2)
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
        logf.close()
    log = open(logf.name, errors="replace").read()
    return out, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-server", required=True); ap.add_argument("--perplexity", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--lib", required=True, help="the plugin library (libggml-moe-cache.so / ggml-moe-cache.dll)")
    ap.add_argument("--level", required=True, choices=sorted(LEVELS)); ap.add_argument("--smoke", action="store_true", help="CI smoke limits (small model)")
    ap.add_argument("--n-cpu-moe", default="999"); ap.add_argument("--ngl", default="99"); ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--chunks", type=int, default=3); ap.add_argument("--text", help="text for the KL run (default: README + docs of this repo)")
    ap.add_argument("--tasks-file", default=os.path.join(HERE, "quality_tasks.json")); ap.add_argument("--no-tasks", action="store_true"); ap.add_argument("--no-kl", action="store_true")
    ap.add_argument("--no-think", action="store_true"); ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--cache-dir", default="", help="keeps the stock base logits (base.kld) and the stock task run (tasks_stock.json) for the next level; one dir per model/settings")
    ap.add_argument("--out", default=""); ap.add_argument("--env", action="append", default=[]); ap.add_argument("--args", default="")
    ap.add_argument("--common-args", default="", help="llama.cpp arguments for stock AND candidate (e.g. --no-repack)")
    ap.add_argument("--ram-env", action="append", default=[], help="K=V for BOTH plugin runs (e.g. MOE_CACHE_SIZE_GIB=0.2)")
    a = ap.parse_args()
    a.out = a.out or tempfile.mkdtemp(prefix="qgate_")
    os.makedirs(a.out, exist_ok=True)
    if a.cache_dir:
        os.makedirs(a.cache_dir, exist_ok=True)
    lenv, largs, kld_max, top_min, tol = LEVELS[a.level]
    if a.smoke:
        kld_max, top_min, tol = SMOKE[a.level]
    cand_env = dict(lenv); cand_env.update(kv.split("=", 1) for kv in a.env)
    cand_args = list(largs) + a.args.split()
    both_env = dict(kv.split("=", 1) for kv in a.ram_env)
    load = plugin_env(a.llama_server, a.lib)
    plug = dict(load, MOE_CACHE_GGUF=a.model, MOE_CACHE_SIZE_GIB=both_env.pop("MOE_CACHE_SIZE_GIB", "auto"), MOE_CACHE_STATS="1", **both_env)
    print(f"quality gate, level {a.level}{' (smoke limits)' if a.smoke else ''}: KLD <= {kld_max}, top-1 >= {top_min} %, tasks >= stock - {tol}; out {a.out}")
    print(f"candidate: env {cand_env}, args {cand_args}")
    res, fails = {"level": a.level, "smoke": a.smoke, "model": os.path.basename(a.model), "env": cand_env, "args": cand_args}, []
    if not a.no_kl:
        if not a.text:
            a.text = os.path.join(a.out, "text.txt")
            with open(a.text, "w", encoding="utf-8") as f:
                for p in [os.path.join(ROOT, "README.md")] + sorted(os.path.join(ROOT, "docs", x) for x in os.listdir(os.path.join(ROOT, "docs")) if x.endswith(".md")):
                    f.write(open(p, encoding="utf-8", errors="replace").read())
        base = os.path.join(a.cache_dir or a.out, "base.kld")
        if os.path.isfile(base):
            print("1. KL: stock base logits reused from", base)
        else:
            print("1. KL: stock (f16 KV) writes the base logits")
            run_ppl(a, base, False, {}, [], os.path.join(a.out, "kl_stock.log"))
        print("2. KL: candidate")
        txt = run_ppl(a, base, True, dict(plug, **cand_env), cand_args, os.path.join(a.out, "kl_candidate.log"))
        if not re.search(r"moe-cache: (cache mode|pass-through mode)", txt):
            print("FAIL: the plugin was not active in the candidate run"); sys.exit(2)
        kld, kerr = num(txt, r"Mean +KLD"); top, terr = num(txt, r"Same top p"); mx, _ = num(txt, r"Maximum KLD")
        res.update(kld=kld, kld_err=kerr, top1=top, top1_err=terr, kld_max=mx)
        print(f"   mean KLD {kld} +- {kerr}, same top token {top} % +- {terr}, max KLD {mx}")
        if kld is None or top is None:
            print("FAIL: could not read the KL result"); sys.exit(2)
        if kld > kld_max: fails.append(f"mean KLD {kld} > {kld_max}")
        if top < top_min: fails.append(f"same top token {top} % < {top_min} %")
    if not a.no_tasks:
        s_out = os.path.join(a.cache_dir, "tasks_stock.json") if a.cache_dir else ""
        if s_out and os.path.isfile(s_out):
            print("3. tasks: stock run reused from", s_out)
        else:
            print("3. tasks: stock server")
            s_out, _ = serve_and_run_tasks(a, {}, [], "stock", a.port)
            if a.cache_dir:
                shutil.copy(s_out, os.path.join(a.cache_dir, "tasks_stock.json"))
        print("4. tasks: candidate server")
        c_out, slog = serve_and_run_tasks(a, dict(plug, **cand_env), cand_args, "candidate", a.port)
        if not re.search(r"moe-cache: (cache mode|pass-through mode)", slog):
            print("FAIL: the plugin was not active in the candidate server"); sys.exit(2)
        sys.path.insert(0, HERE)
        import quality_yardstick as qy
        rs, rc = qy.summary(json.load(open(s_out, encoding="utf-8"))), qy.summary(json.load(open(c_out, encoding="utf-8")))
        res.update(tasks_stock=rs["task_ok"], tasks_candidate=rc["task_ok"], tasks_n=rs["task_n"], loops_stock=rs["loops"], loops_candidate=rc["loops"])
        print(f"   tasks right: stock {rs['task_ok']}/{rs['task_n']}, candidate {rc['task_ok']}/{rc['task_n']}; looping answers: stock {rs['loops']}, candidate {rc['loops']}")
        if rc["task_ok"] < rs["task_ok"] - tol: fails.append(f"tasks {rc['task_ok']} < stock {rs['task_ok']} - {tol}")
        if rc["loops"] > rs["loops"] + LOOP_EXTRA: fails.append(f"loops {rc['loops']} > stock {rs['loops']} + {LOOP_EXTRA}")
    res["pass"] = not fails
    json.dump(res, open(os.path.join(a.out, "gate.json"), "w"), indent=1)
    print(json.dumps(res))
    if fails:
        print("GATE FAIL (" + a.level + "): " + "; ".join(fails)); sys.exit(1)
    print("GATE PASS (" + a.level + ")")


if __name__ == "__main__":
    main()
