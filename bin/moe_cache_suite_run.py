#!/usr/bin/env python3
"""TEST-1142: `moe_cache.py bench-suite run`: measure one or more models with the same suite and write one private suite record each (bin/moe_cache_suite.py), then print the comparison.
  moe_cache.py bench-suite run MODEL.gguf [MODEL2.gguf ...] [--profile quick|standard|night] [--ref-kv f16] [--out DIR] [--launch-args "--ctx 70000 ..."] [--port 8140]
  moe_cache.py bench-suite run --url http://127.0.0.1:8080 --name NAME [--quant Q] [--profile ...]      (measure a server that is already running; no launch, no quality section)
  moe_cache.py bench-suite compare R1.json R2.json ... | validate | show | new | publish      (see bin/moe_cache_suite.py)
Sections by profile: quick = speed probe at 8k + 20 tasks (thinking off); standard = real 32k conversation (speed + memory) + 202 tasks (off and on) + KL against the same model with f16 KV (8 windows);
night = standard with a 65k conversation + the hard set (no cap). The model is started through the launcher (the plan a user gets), one model at a time; every section of every model uses the same
protocol, which is hashed into the record. Records go to benchmarks/suites/ (git-ignored: private until `publish`)."""
import argparse, json, os, re, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, HERE)
import moe_cache_suite as S      # noqa: E402

PROFILES = {"quick": {"speed": "probe", "depth": 8192, "tasks_n": 20, "think": ["off"], "kl_chunks": 0, "hard": False},
            "standard": {"speed": "conv", "depth": 32768, "tasks_n": 202, "think": ["off", "on"], "kl_chunks": 8, "hard": False},
            "night": {"speed": "conv", "depth": 65536, "tasks_n": 202, "think": ["off", "on"], "kl_chunks": 8, "hard": True}}
HARD_CAP = 0


def http(url, path, body=None, timeout=3600):
    req = urllib.request.Request(url.rstrip("/") + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def filler_text():
    """Real text that ships with the repo (the CHANGELOG and the docs), in a fixed order: the same prompt for every model."""
    parts = []
    for p in ["CHANGELOG.md", "README.md"] + sorted(os.path.join("docs", f) for f in os.listdir(os.path.join(ROOT, "docs")) if f.endswith(".md")):
        try:
            parts.append(open(os.path.join(ROOT, p), encoding="utf-8").read())
        except OSError:
            pass
    return "\n\n".join(parts)


def run_speed_probe(url, depth, gen=128):
    """One request with a prompt of about `depth` tokens (cache off): prompt and decode tok/s. A probe, not a conversation: the record says so in its protocol."""
    txt = filler_text(); n = len(http(url, "/tokenize", {"content": txt[:200000]})["tokens"]); cpt = 200000 / max(1, n)
    body = txt[:int(depth * cpt)]; r = http(url, "/completion", {"prompt": body, "n_predict": gen, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False})
    t = r.get("timings", {}); r1 = http(url, "/completion", {"prompt": txt[:int(1000 * cpt)], "n_predict": 1, "temperature": 0, "cache_prompt": False}); t1 = r1.get("timings", {})
    ttft = (t1.get("prompt_ms") or 0) + (t1.get("predicted_ms") or 0)
    return {"protocol": S.protocol_hash(kind="speed-probe", depth=depth, gen=gen, filler=S.file_hash(os.path.join(ROOT, "CHANGELOG.md"))), "bands": [{"depth": depth, "prompt_tps": round(t.get("prompt_per_second", 0), 1), "decode_tps": round(t.get("predicted_per_second", 0), 2)}],
            "ttft_ms": round(ttft), "decode_b1": round(t.get("predicted_per_second", 0), 2)}


def sections_from_conversation(run, ctx):
    """speed + memory sections from a tools/conversation_bench.py run dict (a real multi-turn conversation filled to ctx)."""
    bands = [{"depth": b["hi"], "prompt_tps": b.get("prompt_tps"), "decode_tps": b.get("decode_tps")} for b in run.get("bands", []) if b.get("decode_tps") is not None]
    sp = {"protocol": S.protocol_hash(kind="speed-conv", ctx=ctx, seed=run.get("settings", {}).get("seed")), "bands": bands}
    t = [b.get("ttft_median_s") for b in run.get("bands", []) if b.get("ttft_median_s") is not None]
    if t:
        sp["ttft_ms"] = round(t[0] * 1000)
    mem = {"protocol": S.protocol_hash(kind="memory", ctx=ctx)}
    if run.get("process_ram_peak_mib"):
        mem["rss_gb"] = round(run["process_ram_peak_mib"] / 1024, 2)
    if run.get("vram_used_peak_mib"):
        mem["vram_gb"] = round(run["vram_used_peak_mib"] / 1024, 2)
    return sp, mem


def _py(script, *args, timeout=None):
    r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, script), *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed: {(r.stdout + r.stderr)[-400:]}")
    return r.stdout


def run_tasks(url, think, n, tmp):
    tf = os.path.join(TOOLS, "quality_tasks.json"); tasks = json.load(open(tf, encoding="utf-8"))
    step = max(1, len(tasks) // n) if n < len(tasks) else 1; sub = tasks[::step][:n]
    sf = os.path.join(tmp, f"tasks_{think}.json"); json.dump(sub, open(sf, "w"))
    out = os.path.join(tmp, f"tasks_{think}_run.json"); cap = 2048 if think == "on" else 64
    _py("quality_yardstick.py", "run", "--url", url, "--out", out, "--label", f"suite-{think}", "--tasks", sf, "--tasks-only", "--task-n-predict", str(cap), *(["--no-think"] if think == "off" else []))
    return S.section_tasks(json.load(open(out)), think, S.file_hash(sf), cap)


def run_hard(url, cap, workers, tmp):
    out = os.path.join(tmp, "hard_run.json")
    _py("hard_tasks_run.py", "run", "--url", url, "--out", out, "--label", "suite", "--cap", str(cap), "--workers", str(workers))
    return S.section_hard(json.load(open(out)), S.file_hash(os.path.join(TOOLS, "hard_tasks.json")))


def run_quality(model, kv_flags, chunks, tmp, perplexity="llama-perplexity", windows=None, ngl=99, ref_label="same model f16 KV"):
    """KL and same-top-token of the candidate settings (kv_flags) against the same model with f16 KV, on the real-chat windows file (text of the repo's docs when no file is given)."""
    windows = windows or os.path.join(tmp, "windows.txt")
    if not os.path.exists(windows):
        open(windows, "w", encoding="utf-8").write(filler_text()[:2_000_000])
    base = os.path.join(tmp, "base.kld"); common = ["-m", model, "-f", windows, "-c", "8192", "-ngl", str(ngl), "-fa", "on", "--chunks", str(chunks)]
    subprocess.run([perplexity, *common, "--save-all-logits", base], capture_output=True, check=True)
    r = subprocess.run([perplexity, *common, *kv_flags, "--kl-divergence-base", base, "--kl-divergence"], capture_output=True, text=True, check=True)
    return S.section_quality(r.stdout + r.stderr, ref_label, S.file_hash(windows), chunks, chunks * 4096, " ".join(kv_flags) or "f16")


def launch(model, port, extra):
    """Start the model through the launcher (the plan a user gets) and wait for /health; on failure the error carries the launcher's last lines."""
    cmd = [sys.executable, os.path.join(HERE, "moe_cache.py"), "server", model, "--port", str(port)] + extra
    log = tempfile.TemporaryFile("w+"); p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).status == 200:
                return p
        except Exception:
            time.sleep(3)
        if p.poll() is not None:
            log.seek(0); raise RuntimeError("the server exited while starting: " + " | ".join(log.read().strip().splitlines()[-4:])[-500:])
    p.terminate(); raise RuntimeError("the server did not become ready")


def machine_facts():
    import platform
    m = {"platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        m["ram_gb"] = round(int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1)) / 1048576, 1)
    except Exception:
        pass
    return m


def measure(url, name, quant, prof, model_path=None, kv_flags=(), tmp=None, workers=1, perplexity="llama-perplexity", progress=print):
    """All sections of the profile against a running server; returns the suite record."""
    P = PROFILES[prof]; sections = {}; notes = []
    progress(f"[{name}] speed ({P['speed']})")
    if P["speed"] == "probe":
        sections["speed"] = run_speed_probe(url, P["depth"])
    else:
        out = os.path.join(tmp, "conv.json")
        _py("conversation_bench.py", "run", "--url", url, "--arm", "suite", "--ctx", str(P["depth"]), "--out", out)
        run = json.load(open(out)); run = run["runs"][0] if "runs" in run else run
        sections["speed"], sections["memory"] = sections_from_conversation(run, P["depth"])
    for think in P["think"]:
        progress(f"[{name}] tasks, thinking {think}")
        sections["tasks" if think == "off" else "tasks_on"] = run_tasks(url, think, P["tasks_n"], tmp)
    if P["hard"]:
        progress(f"[{name}] hard set (no cap)"); sections["hard"] = run_hard(url, HARD_CAP, workers, tmp)
    if P["kl_chunks"] and model_path:
        progress(f"[{name}] KL against f16 KV"); sections["quality"] = run_quality(model_path, list(kv_flags), P["kl_chunks"], tmp, perplexity)
    elif P["kl_chunks"]:
        notes.append("quality section skipped: no model file given (a running server was measured)")
    model = {"name": name, "quant": quant, "file": os.path.basename(model_path or "")}
    return S.new_record(model, machine=machine_facts(), settings={"kv_flags": list(kv_flags), "profile": prof}, sections=sections, profile=prof, notes=notes)


def cmd_run(a):
    out = a.out or os.path.join(ROOT, "benchmarks", "suites"); os.makedirs(out, exist_ok=True); recs = []
    targets = [(None, a.name or "server", a.quant, a.url)] if a.url else [(m, os.path.splitext(os.path.basename(m))[0], a.quant, None) for m in a.models]
    for model, name, quant, url in targets:
        proc = None
        try:
            if url is None:
                proc = launch(model, a.port, a.launch_args.split()); url = f"http://127.0.0.1:{a.port}"
            with tempfile.TemporaryDirectory() as tmp:
                rec = measure(url, name, quant, a.profile, model, tuple(a.kv_flags.split()), tmp, a.workers, a.perplexity)
            pr = S.validate_record(rec)
            if pr:
                print("invalid record:", pr, file=sys.stderr); return 1
            path = os.path.join(out, rec["id"] + ".json"); json.dump(rec, open(path, "w"), indent=1); recs.append(rec); print("wrote", path)
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
            url = None if a.url is None else url
    if len(recs) > 1:
        print(S.compare_text(S.compare(recs)))
    return 0


def main(argv):
    if argv[:1] == ["card"]:
        import moe_cache_suite_card as C
        return C.main(argv[1:])
    if argv[:1] and argv[0] != "run":
        return S.main(argv)
    ap = argparse.ArgumentParser(prog="bench-suite run"); ap.add_argument("run"); ap.add_argument("models", nargs="*"); ap.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    ap.add_argument("--url", default=""); ap.add_argument("--name", default=""); ap.add_argument("--quant", default=""); ap.add_argument("--out", default=""); ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--launch-args", default=""); ap.add_argument("--kv-flags", default="", help="llama-perplexity flags of the candidate settings, e.g. '-ctk q8_0 -ctv q4_0'")
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--perplexity", default="llama-perplexity")
    a = ap.parse_args(argv)
    if not a.url and not a.models:
        ap.error("give model files or --url")
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
