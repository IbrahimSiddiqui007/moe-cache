#!/usr/bin/env python3
"""tools/conversation_bench.py (TEST-699) against a fake llama-server (no model): the history grows turn by turn, the conversation stops before the context
limit, the server's timings are read, the per-depth-band numbers add up, stock and plugin servers are run in A B B A order, the plugin's counters are
read per turn, and the record validates against docs/bench_record.schema.json.
   python3 tests/conversation_bench_test.py"""
import hashlib, http.server, json, os, signal, socketserver, sys, tempfile, threading

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
sys.path.insert(0, os.path.join(HERE, "bin"))
import conversation_bench as cb  # noqa: E402
import moe_cache_record as rr  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

ORDER = []           # which server answered, in order


def toks(m, div=4):
    return len(m["content"]) // div + 8


class Fake(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == "/props":
            b = json.dumps({"default_generation_settings": {"n_ctx": self.server.n_ctx}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        s = self.server
        div = getattr(s, "tok_div", 4)
        if self.path == "/tokenize":                      # llama-server's tokenizer endpoint (a server without it answers 404)
            if getattr(s, "no_tokenize", False):
                self.send_response(404); self.end_headers(); return
            b = json.dumps({"tokens": list(range(len(body.get("content", "")) // div))}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            return
        s.requests.append(body)
        ORDER.append(s.name)
        msgs = body["messages"]
        total = sum(toks(m, div) for m in msgs)
        if getattr(s, "llama_rule", False):               # llama.cpp: refuses only a prompt over the context, with this message
            if total > s.n_ctx:
                b = json.dumps({"error": {"code": 400, "message": f"the request ({total} tokens) exceeds the available context size ({s.n_ctx} tokens), try increasing it", "type": "exceed_context_size_error"}}).encode()
                self.send_response(400); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        elif total + body["max_tokens"] > s.n_ctx:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"exceeds the context"}'); return
        common = 0
        for a, b in zip(s.last, msgs):
            if a != b:
                break
            common += toks(b, div)
        pn = total - common
        answer = f"Answer {len(msgs)}: " + "word " * 30
        pred = body["max_tokens"] // 2
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for piece in (answer[:20], answer[20:]):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n\n"); self.wfile.flush()
        tm = {"prompt_n": pn, "cache_n": common, "prompt_ms": pn / s.p_tps * 1000, "prompt_per_second": s.p_tps, "predicted_n": pred,
              "predicted_ms": pred / s.d_tps * 1000, "predicted_per_second": s.d_tps}
        self.wfile.write(b"data: " + json.dumps({"choices": [], "timings": tm, "usage": {"prompt_tokens": total, "completion_tokens": pred}}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        s.last = msgs + [{"role": "assistant", "content": answer}]


class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def start(name, n_ctx, p_tps, d_tps):
    s = Srv(("127.0.0.1", 0), Fake)
    s.name, s.n_ctx, s.p_tps, s.d_tps, s.requests, s.last = name, n_ctx, p_tps, d_tps, [], []
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}"


quiet = lambda *a, **k: None

# ---- the script is fixed by its seed
t1, t2, t3 = cb.make_turn(42, 5), cb.make_turn(42, 5), cb.make_turn(43, 5)
check(t1 == t2 and t1["text"] != t3["text"], "the same seed and turn give the same user message; another seed another one")
kinds = {cb.make_turn(7, i)["kind"] for i in range(60)}
check({"followup", "code", "tool_output"} <= kinds, f"the script mixes small follow-ups and big pastes ({sorted(kinds)})")

# ---- one conversation up to the context limit
A, ua = start("A", 4096, 100.0, 10.0)
run = cb.run_conversation(ua, "stock", 4096, 32, 42, "", "m", 30, None, None, None, False, False, quiet)
turns = run["turns"]
check(len(turns) >= 4 and run["stop_reason"] == "context full", f"it stops because the context is full ({len(turns)} turns, {run['stop_reason']})")
check(run["final_depth"] <= 4096 and all(t["error"] is None for t in turns), f"it never passes the limit (final depth {run['final_depth']} of 4096) and no request failed")
check([len(r["messages"]) for r in A.requests] == [2 + 2 * i for i in range(len(turns))], "the history grows by one user and one assistant message per turn")
check(all(turns[i + 1]["depth_before"] > turns[i]["depth_after"] for i in range(len(turns) - 1)), "each turn starts deeper than the last one ended")
check(all(turns[i + 1]["cache_n"] >= turns[i]["depth_after"] - 8 for i in range(len(turns) - 1)), "the earlier history comes from the cache (cache_n of the fake server)")
check(all(t["prompt_tps"] == 100.0 and t["decode_tps"] == 10.0 for t in turns), "the server's timings are read (prompt 100, decode 10 tok/s)")
check(sum(b["turns"] for b in run["bands"]) == len(turns) and len(run["bands"]) == 4, "every turn is in exactly one of the four depth bands")
big = [b for b in run["bands"] if b["prompt_turns"]]
check(big and all(abs(b["prompt_tps"] - 100.0) < 0.05 and abs(b["decode_tps"] - 10.0) < 0.05 for b in big if b["decode_tps"]), "band speeds are the weighted means of the turns (100 and 10 here)")
check(all(b["cache_reuse_pct"] is None or 0 <= b["cache_reuse_pct"] <= 100 for b in run["bands"]), "cache reuse per band is a percentage")

# ---- the server's own context caps --ctx
B, ub = start("B", 2048, 100.0, 10.0)
rc = cb.main(["run", "--url", ub, "--arm", "moe-cache", "--ctx", "100000", "--max-tokens", "16", "--out", os.path.join(tempfile.mkdtemp(), "r.json")])
check(rc == 0 and max(r["max_tokens"] for r in B.requests) == 16 and B.last, "--ctx above the server's n_ctx is cut to it (the conversation still runs)")

# ---- stock and plugin servers, A B B A, then the record
ORDER.clear()
S, us = start("stock", 4096, 100.0, 10.0)
P, up = start("plugin", 4096, 200.0, 20.0)
out = tempfile.mkdtemp()
rc = cb.main(["ab", "--stock-url", us, "--plugin-url", up, "--out-dir", out, "--ctx", "4096", "--max-tokens", "32", "--command-line", "llama-server -c 4096 -ngl 99"])
k = len(ORDER) // 4
seq = [ORDER[i * k] for i in range(4)]
check(rc == 0 and len(ORDER) == 4 * k and all(len(set(ORDER[i * k:(i + 1) * k])) == 1 for i in range(4)) and seq == ["stock", "plugin", "plugin", "stock"], f"the order is stock, plugin, plugin, stock ({seq})")
rec = json.load(open(os.path.join(out, "record.json")))
check(rr.validate(rec, rr.load_schema()) == [], "the record validates against docs/bench_record.schema.json")
check([r["arm"] for r in rec["runs"]] == ["stock", "moe-cache", "moe-cache", "stock"], "the record keeps the runs in the order they were played")
check(len(rec["runs"]) == 4 and all("turns" in r and "bands" in r for r in rec["runs"]), "four runs, each with its per-turn and per-band numbers")
check(set(rec["summary"]) == {"stock", "moe-cache"} and rec["protocol"]["order"] == "ABBA" and rec["protocol"]["prompt_tokens"] > 3000, "summary per arm, ABBA and the depth reached are in the record")
check(abs(rec["summary"]["stock"]["decode_tps"] - 10.0) < 0.05 and abs(rec["summary"]["moe-cache"]["decode_tps"] - 20.0) < 0.05, "whole-run decode speeds per arm (10 and 20 here)")
runs = [json.load(open(os.path.join(out, f))) for f in sorted(os.listdir(out)) if f[0].isdigit()]
tab = cb.band_table(runs)
check("2.00x" in tab and "stock prompt tok/s" in tab and "moe-cache decode tok/s" in tab, "the band table puts the arms side by side with their ratio (2.00x)")
rc2 = cb.main(["assemble"] + [os.path.join(out, f) for f in sorted(os.listdir(out)) if f[0].isdigit()] + ["--out", os.path.join(out, "again.json")])
check(rc2 == 0 and os.path.exists(os.path.join(out, "again.json")), "assemble builds the same kind of record from the run files")

# ---- the plugin's counters per turn (a SIGUSR1 handler plays the plugin)
if os.name != "nt":
    log = os.path.join(tempfile.mkdtemp(), "server.err.log")
    state = {"n": 0}
    def on_usr1(sig, frm):
        state["n"] += 1
        with open(log, "a") as f:
            f.write("moe-cache-stats " + json.dumps({"hits_dec": 10 * state["n"], "misses_dec": 2 * state["n"], "hits": 0, "misses": 0}) + "\n")
    open(log, "w").close()
    old = signal.signal(signal.SIGUSR1, on_usr1)
    try:
        C, uc = start("C", 4096, 100.0, 10.0)
        st = cb.Stats(os.getpid(), log)
        r3 = cb.run_conversation(uc, "moe-cache", 4096, 32, 42, "", "m", 30, st, 5, None, False, False, quiet)
        check(all(t["hits_dec"] == 10 and t["misses_dec"] == 2 for t in r3["turns"]), "hits_dec and misses_dec of each turn are the difference of two snapshots (10 and 2 here)")
        check(any(b["hit_rate_dec_pct"] == 83.3 for b in r3["bands"]), "the band hit rate is hits / (hits + misses) (83.3 %)")
    finally:
        signal.signal(signal.SIGUSR1, old)
else:
    print("  skip the plugin counter test (needs SIGUSR1)")
no = cb.run_conversation(ua, "stock", 4096, 32, 42, "", "m", 30, None, 3, None, False, False, quiet)
check(all(t["hits_dec"] is None for t in no["turns"]) and all(b["hit_rate_dec_pct"] is None for b in no["bands"]), "without the plugin's counters the hit rate is empty, not guessed")

# ---- a server that fails ends the conversation with the reason, it does not hang
D, ud = start("D", 4096, 100.0, 10.0)
D.n_ctx = 100            # every request is refused with HTTP 400
bad = cb.run_conversation(ud, "stock", 4096, 32, 42, "", "m", 30, None, None, None, False, False, quiet)
check(bad["stop_reason"].startswith("error: HTTP 400") and len(bad["turns"]) == 1, "an HTTP error ends the run with its reason")

# ---- full metrics (owner's rule): resource peaks, load time, disk reads, per-run settings; stock hit rate is n/a, not a dash
E_srv, ue = start("E", 4096, 100.0, 10.0)
slog = os.path.join(tempfile.mkdtemp(), "server.log")
open(slog, "w").write("llm_load_print_meta: n_layer          = 48\nload_tensors: offloaded 49/49 layers to GPU\nVulkan0 model buffer size = 5000.00 MiB\nVulkan0 KV buffer size = 600.00 MiB\n")
out2 = tempfile.mkdtemp()
cmdfile = os.path.join(out2, "cmd.txt")
open(cmdfile, "w", encoding="utf-8-sig").write("env MOE_CACHE_STATS_SIGNAL=1 llama-server -m \"C:\\my models\\x.gguf\" -c 4096 -ub 2048 --n-cpu-moe 30 -ngl 99\n")
rc = cb.main(["run", "--url", ue, "--arm", "stock", "--ctx", "4096", "--max-tokens", "32", "--max-turns", "6", "--monitor-pid", str(os.getpid()),
              "--command-line", "@" + cmdfile, "--server-log", slog, "--load-s", "42.5",
              "--out", os.path.join(out2, "1-stock.json")])
rs = json.load(open(os.path.join(out2, "1-stock.json")))
check(rc == 0 and rs["load_s"] == 42.5 and rs["settings"]["ctx"] == 4096 and rs["settings"]["ubatch"] == 2048 and rs["settings"]["n_cpu_moe"] == 30, "load time and the run's own settings (ctx, ub, n-cpu-moe) are stored")
check(rs["settings"]["gpu_expert_layers"] == 18 and rs["settings"]["environment"].get("MOE_CACHE_STATS_SIGNAL") == "1", "GPU expert layers come from n_layer in the server log (48 - 30) and the env prefix is kept")
check(isinstance(rs["monitor"]["process_ram_peak_mib"], (int, float)) and rs["monitor"]["samples"] >= 1, "resource peaks are sampled with the bench-report sampler (process RAM here)")
if os.name != "nt":
    log2 = os.path.join(tempfile.mkdtemp(), "server.err.log"); n2 = {"n": 0}
    def on2(sig, frm):
        n2["n"] += 1
        open(log2, "a").write("moe-cache-stats " + json.dumps({"hits_dec": 10 * n2["n"], "misses_dec": 2 * n2["n"], "read_bytes": 2_000_000_000 * n2["n"]}) + "\n")
    open(log2, "w").close(); old2 = signal.signal(signal.SIGUSR1, on2)
    try:
        F, uf = start("F", 4096, 200.0, 20.0)
        rp = cb.run_conversation(uf, "moe-cache", 4096, 32, 42, "", "m", 30, cb.Stats(os.getpid(), log2), 4, None, False, False, quiet)
        check(all(t["read_gb"] == 2.0 for t in rp["turns"]) and rp["disk_read_gb"] == 8.0, "disk reads per turn and per run come from the plugin's read_bytes counter (2 GB per turn here)")
        runs3 = [rs, rp, dict(rp), dict(rs)]
        for r in runs3: r.setdefault("settings", None)
        ns = type("A", (), dict(model=None, llama_server=None, backend=None, model_url=None, command_line=None, server_cmd=None, baseline=None, other_programs=None, cache=None))()
        rec3 = cb.build_record(runs3, ns)
        check(rr.validate(rec3, rr.load_schema()) == [], "a record with monitor, settings and disk reads still validates")
        e0, e1 = rec3["runs"][0], rec3["runs"][1]
        check(isinstance(e0["process_ram_peak_mib"], (int, float)) and e0["load_s"] == 42.5, "the stock run carries its peaks and load time")
        check(e1["disk_read_gb"] == 8.0 and e1["hit_rate_dec_pct"] == 83.3, "the plugin run carries disk reads and the decode hit rate")
        check(e0["hit_rate_dec_pct"] == "n/a (no cache)", "the stock run's hit rate is n/a (no cache) in the record")
        tab3 = cb.band_table(runs3)
        check("n/a (no cache)" in tab3 and "moe-cache hit rate %" in tab3, "the stock hit-rate cells in the band table say n/a (no cache), not a dash")
        check("n/a (no cache)" in cb.turn_table(rs), "the stock per-turn table says n/a (no cache)")
        check(rec3["runs"][0]["settings"]["ctx"] == 4096 and "settings in this record" in " ".join(rec3["notes"]), "per-run settings and the note on where the record's settings come from are in the record")
    finally:
        signal.signal(signal.SIGUSR1, old2)

# ---- the record names the plugin the moe-cache server ran (its MOE_CACHE_LIB), not a DLL found by a file search; --version is read from stderr too
mc_lib = dict(rs, arm="moe-cache", settings=dict(rs["settings"], environment=dict(rs["settings"]["environment"], MOE_CACHE_LIB="C:\\kit\\ggml-moe-cache-api3.dll")))
ns4 = type("A", (), dict(model=None, llama_server=None, backend=None, model_url=None, command_line=None, server_cmd=None, baseline=None, other_programs=None, cache=None))()
rec4 = cb.build_record([dict(rs), mc_lib, dict(mc_lib), dict(rs)], ns4)
check(rec4["software"]["plugin_file"] == "ggml-moe-cache-api3.dll" and rec4["software"]["plugin_sha256_prefix"] == rr.NR,
      f"plugin_file comes from the moe-cache run's MOE_CACHE_LIB, hash not recorded when the file is not on this machine ({rec4['software']['plugin_file']})")
real_lib = os.path.join(tempfile.mkdtemp(), "ggml-moe-cache-api3.dll"); open(real_lib, "wb").write(b"plugin bytes")
mc_real = dict(mc_lib, settings=dict(mc_lib["settings"], environment=dict(mc_lib["settings"]["environment"], MOE_CACHE_LIB=real_lib)))
rec5 = cb.build_record([dict(rs), mc_real, dict(mc_real), dict(rs)], ns4)
check(rec5["software"]["plugin_file"] == "ggml-moe-cache-api3.dll" and rec5["software"]["plugin_sha256_prefix"] == hashlib.sha256(b"plugin bytes").hexdigest()[:12],
      "plugin_file and its hash come from the DLL the server ran when it exists here")
check(rr.validate(rec4, rr.load_schema()) == [] and rr.validate(rec5, rr.load_schema()) == [], "both records still validate")
rec6 = cb.build_record([dict(rs), dict(rs)], ns4)
check(rec6["software"]["plugin_file"] == rr.NR, "a record without a MOE_CACHE_LIB says the plugin is not recorded (no file search)")
check("version: 12064 (51fe42c5d)" in rr._run([sys.executable, "-c", "import sys; sys.stderr.write('version: 12064 (51fe42c5d)\\n')"], both=True)
      and rr._run([sys.executable, "-c", "import sys; sys.stderr.write('x')"]) == "", "_run(both=True) keeps stderr, where llama-server prints --version")

check(rr.llama_build("version: 0.8.0-dev (build 12064, commit 51fe42c5d)\nbuilt with Clang 20.1.8") == (12064, "51fe42c5d")
      and rr.llama_build("version: 11966 (ff0d396e3)") == (11966, "ff0d396e3"), "llama_build reads the new '(build N, commit C)' and the old 'version: N (C)' lines")

# ---- Stage 0: gpu_blocks, upload bytes avoided and SSD wait of a run are the difference between the first and the last snapshot; extra --server-log files are read
if os.name != "nt":
    log4 = os.path.join(tempfile.mkdtemp(), "server.err.log"); n4 = {"n": 0}
    def on4(sig, frm):
        n4["n"] += 1
        open(log4, "a").write("moe-cache-stats " + json.dumps({"hits_dec": 10 * n4["n"], "misses_dec": 2 * n4["n"], "read_bytes": 1_000_000_000 * n4["n"], "gpu_segments": 100 * n4["n"],
                                                               "gpu_up_saved": 3_000_000_000 * n4["n"], "ns_ssd_wait": 1_500_000_000 * n4["n"]}) + "\n")
    open(log4, "w").close(); old4 = signal.signal(signal.SIGUSR1, on4)
    try:
        G, ug = start("G", 4096, 100.0, 10.0)
        r4 = cb.run_conversation(ug, "moe-cache", 4096, 32, 42, "", "m", 30, cb.Stats(os.getpid(), log4), 4, None, False, False, quiet)
        check(all(t["gpu_blocks"] == 100 and t["gpu_up_saved_gb"] == 3.0 and t["ssd_wait_s"] == 1.5 for t in r4["turns"]), "per turn: GPU blocks, upload GB avoided and SSD wait seconds are snapshot differences (100, 3.0, 1.5)")
        check(r4["gpu_blocks"] == 400 and r4["gpu_up_saved_gb"] == 12.0 and r4["ssd_wait_s"] == 6.0, "per run: 4 turns give 400 GPU blocks, 12 GB avoided, 6 s SSD wait")
        ns6 = type("A", (), dict(model=None, llama_server=None, backend=None, model_url=None, command_line=None, server_cmd=None, baseline=None, other_programs=None, cache=None))()
        rec7 = cb.build_record([dict(r4), dict(r4)], ns6)
        e7 = rec7["runs"][0]
        check(e7["gpu_blocks"] == 400 and e7["gpu_up_saved_gb"] == 12.0 and e7["ssd_wait_s"] == 6.0 and rr.validate(rec7, rr.load_schema()) == [], "the record carries gpu_blocks, gpu_up_saved_gb and ssd_wait_s and still validates")
    finally:
        signal.signal(signal.SIGUSR1, old4)
no2 = cb.run_conversation(ua, "stock", 4096, 32, 42, "", "m", 30, None, 2, None, False, False, quiet)
check("gpu_blocks" not in no2 and all(t["gpu_blocks"] is None for t in no2["turns"]), "without the plugin's counters gpu_blocks is absent, so the record says not recorded")
out7 = tempfile.mkdtemp()
so_log, se_log = os.path.join(out7, "o.log"), os.path.join(out7, "e.log")
open(so_log, "w").write("moe-cache: profile x loaded (48 layers x 512 experts)\nmoe-cache: experts of 42 layers stay on the CPU and are managed by the cache\n")
open(se_log, "w").write("0.12.691.866 I cmn          init: llama threadpool init, n_threads = 16\n")
H, uh = start("H", 4096, 100.0, 10.0)
rc = cb.main(["run", "--url", uh, "--arm", "stock", "--ctx", "4096", "--max-tokens", "16", "--max-turns", "2", "--command-line", "llama-server -m x.gguf -c 4096 --n-cpu-moe 42 -ngl 99",
              "--server-log", se_log, "--server-log", so_log, "--out", os.path.join(out7, "r.json")])
rh = json.load(open(os.path.join(out7, "r.json")))["settings"]
check(rc == 0 and rh["threads"] == 16 and rh["n_layer"] == 48 and rh["gpu_expert_layers"] == 6 and str(rh["kv_type_k"]).startswith("f16"),
      f"two --server-log files (stderr and stdout) are read together: threads {rh['threads']}, n_layer {rh['n_layer']}, GPU expert layers {rh['gpu_expert_layers']}, KV {rh['kv_type_k']}")

print("conversation bench test: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
# TEST-754: the cost-model fit recovers known A, B, C from synthetic turns
def syn(ub, A=3.0, B=1.2, C=0.06):
    out = []
    cache = 0
    for n in (900, 2500, 300, 4000, 1500, 700, 3200, 2048, 5000):
        d = cache + n / 2
        out.append({"error": None, "prompt_n": n, "cache_n": cache, "prompt_ms": 1000 * (A * -(-n // ub) + B * n / 1000 + C * (n / 1000) * (d / 1000))})
        cache += n
    return out
f = cb.cost_fit_runs([(syn(2048), 2048), (syn(4096), 4096)])
check(f is not None and abs(f["a_s_per_pass"] - 3.0) < 0.01 and abs(f["b_s_per_1k"] - 1.2) < 0.01 and abs(f["c_s_per_1k_x_1k_depth"] - 0.06) < 0.001, f"fit over two -ub arms recovers A, B, C ({f})")
check(cb.cost_fit([], 2048) is None, "fit: too few turns gives None, no crash")

# ---- 2026-11-07: visible answers (thinking off, 256 tokens, answer hint) and the short-answer warning; --legacy keeps the v0.3.7 behaviour
import argparse as _ap
def _args(*argv):
    ap = _ap.ArgumentParser(); cb._run_args(ap); return ap.parse_args(list(argv))
check(cb._settings(_args()) == (256, True, True), "defaults: 256 tokens, thinking off, answer hint on")
check(cb._settings(_args("--legacy")) == (128, False, False), "--legacy: 128 tokens, thinking on, no hint (the v0.3.7 settings)")
check(cb._settings(_args("--think")) == (256, False, True) and cb._settings(_args("--no-answer-hint")) == (256, True, False) and cb._settings(_args("--max-tokens", "64")) == (64, True, True), "--think, --no-answer-hint and --max-tokens each change one thing")
check(cb._settings(_args("--no-think")) == (256, True, True), "--no-think from old command lines is accepted and changes nothing")
C1, uc1 = start("C1", 8192, 100.0, 10.0)
rc = cb.main(["run", "--url", uc1, "--arm", "moe-cache", "--ctx", "8192", "--max-turns", "6", "--out", os.path.join(tempfile.mkdtemp(), "r.json")])
rq = C1.requests
check(rc == 0 and all(r["max_tokens"] == 256 and r.get("chat_template_kwargs") == {"enable_thinking": False} for r in rq), "a default run asks for 256 tokens with thinking off")
check(all(r["messages"][-1]["content"].endswith(cb.ANSWER_HINT) for r in rq) and all(not m["content"].endswith(cb.ANSWER_HINT) or m["role"] == "user" for r in rq for m in r["messages"]), "every user message ends with the answer hint")
C2, uc2 = start("C2", 8192, 100.0, 10.0)
rc = cb.main(["run", "--url", uc2, "--arm", "moe-cache", "--ctx", "8192", "--max-turns", "4", "--legacy", "--out", os.path.join(tempfile.mkdtemp(), "r.json")])
rq = C2.requests
check(rc == 0 and all(r["max_tokens"] == 128 and "chat_template_kwargs" not in r and not r["messages"][-1]["content"].endswith(cb.ANSWER_HINT) for r in rq), "--legacy asks for 128 tokens, leaves thinking to the server and adds no hint")
msgs = []
runS = cb.run_conversation(start("C3", 8192, 100.0, 10.0)[1], "stock", 8192, 8, 42, "", "m", 30, None, 4, None, True, False, msgs.append, None, None, True, 24)
check(runS["thinking"] is False and runS["answer_hint"] is True and runS["answer_stats"]["median_tokens"] == 4 and runS["answer_stats"]["warnings"] and any("WARNING" in m for m in msgs),
      f"answers of 4 tokens: the run carries a warning and the log says WARNING ({runS['answer_stats']['median_tokens']})")
runL = cb.run_conversation(start("C4", 8192, 100.0, 10.0)[1], "stock", 8192, 256, 42, "", "m", 30, None, 4, None, True, False, lambda *a: None, None, None, True, 24)
check(runL["answer_stats"]["median_tokens"] == 128 and runL["answer_stats"]["warnings"] == [], "answers of 128 tokens: no warning")
st = cb.answer_stats([{"predicted_n": 40, "answer_chars": 0}] * 3 + [{"predicted_n": 40, "answer_chars": 90}], 24)
check(st["empty_turns"] == 3 and any("no visible answer" in w for w in st["warnings"]), "more than a quarter of the turns without a visible answer: a warning")
check(cb.answer_stats([], 24)["warnings"] == [], "no turns: no warning, no crash")

# ---- cause finding (2026-11-07): answers are kept, the system message / hint / temperature / empty marker can be changed, the stats say what follows a short answer
C5, uc5 = start("C5", 8192, 100.0, 10.0)
rc = cb.main(["run", "--url", uc5, "--arm", "stock", "--ctx", "8192", "--max-turns", "3", "--system", "neutral", "--hint", "Explain in detail.", "--temperature", "0.7",
              "--out", os.path.join(tempfile.mkdtemp(), "r5.json")])
rq = C5.requests
check(rc == 0 and all(r["messages"][0]["content"] == cb.NEUTRAL_SYSTEM and r["temperature"] == 0.7 and r["messages"][-1]["content"].endswith("Explain in detail.") for r in rq),
      "--system neutral, --temperature and --hint reach the request")
C6, uc6 = start("C6", 8192, 100.0, 10.0)
r6 = cb.run_conversation(uc6, "stock", 8192, 32, 42, "", "m", 30, None, 3, None, True, False, lambda *a: None)
check(all("answer_head" in t and t["answer_head"].startswith("Answer ") and "finish_reason" in t for t in r6["turns"]), "every turn keeps the first characters of the answer and its finish reason")
r7 = cb.run_conversation(uc6, "stock", 8192, 32, 42, "", "m", 30, None, 2, None, True, False, lambda *a: None, None, None, False, 24, None, 0.0, "(no visible answer)", None, False)
check(all("answer_head" not in t for t in r7["turns"]), "keep_answers off: no answer text in the run")
seq = [40, 40, 2, 2, 2, 40, 2, 40]
turns = [{"turn": i, "kind": "followup" if i % 2 else "code", "predicted_n": n, "answer_chars": 5, "finish_reason": "stop"} for i, n in enumerate(seq)]
st = cb.answer_stats(turns, 24)
check(st["first_short_turn"] == 2 and st["p_short_after_short"] == 0.5 and st["p_short_after_long"] == 0.667 and st["finish_reasons"] == {"stop": 8} and set(st["median_by_kind"]) == {"code", "followup"},
      f"answer_stats: first short turn, chance of a short answer after a short / a long one, median by kind ({st['first_short_turn']}, {st['p_short_after_short']}, {st['p_short_after_long']})")

# ---- 2026-11-07: turns are sized by the server's own token count; a context-size refusal ends the run as "context full", not as an error
def heavy(name, n_ctx, **kw):
    s, u = start(name, n_ctx, 100.0, 10.0)
    s.tok_div, s.llama_rule = 1, True               # 1 character per token (far denser than any real text: the old guess of 3 undershoots by a lot), the refusal rule and message of llama.cpp
    for k, v in kw.items():
        setattr(s, k, v)
    return s, u
check(cb.count_tokens(heavy("T0", 4096)[1], "", "x" * 100) == 100, "count_tokens asks /tokenize (100 characters = 100 tokens on the dense fake)")
check(cb.count_tokens(heavy("T0b", 4096, no_tokenize=True)[1], "", "x" * 100) is None, "a server without /tokenize gives None, not a crash")
check(cb.is_context_error("HTTP 400: {\"error\":{\"message\":\"the request (66515 tokens) exceeds the available context size (65536 tokens), try increasing it\"}}")
      and not cb.is_context_error("HTTP 500: boom") and not cb.is_context_error(None), "the context-size refusal is recognised, other errors are not")
T1, ut1 = heavy("T1", 8192)
q1 = cb.run_conversation(ut1, "stock", 8192, 256, 42, "", "m", 30, None, None, None, True, False, quiet, None, None, True, 24)
check(q1["stop_reason"] == "context full" and q1["final_depth"] is not None and q1["final_depth"] <= 8192 and not any(t["error"] for t in q1["turns"]),
      f"dense tokens, /tokenize: the run ends with context full, no error, final depth {q1['final_depth']} within 8192 ({len(q1['turns'])} turns)")
T2, ut2 = heavy("T2", 16384, no_tokenize=True)
q2 = cb.run_conversation(ut2, "stock", 16384, 256, 42, "", "m", 30, None, None, None, True, False, quiet, None, None, True, 24)
check(q2["stop_reason"] == "context full" and q2["final_depth"] is not None and not any(t["error"] for t in q2["turns"]),
      f"dense tokens, no /tokenize: the guess may be low, a 400 context refusal still ends as context full ({q2['stop_reason']}, final depth {q2['final_depth']})")
check(T2.requests and True, "the no-/tokenize run did send requests")
sys.exit(1 if fails else 0)
