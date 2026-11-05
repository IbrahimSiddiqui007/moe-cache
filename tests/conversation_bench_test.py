#!/usr/bin/env python3
"""tools/conversation_bench.py (TEST-699) against a fake llama-server (no model): the history grows turn by turn, the conversation stops before the context
limit, the server's timings are read, the per-depth-band numbers add up, stock and plugin servers are run in A B B A order, the plugin's counters are
read per turn, and the record validates against docs/bench_record.schema.json.
   python3 tests/conversation_bench_test.py"""
import http.server, json, os, signal, socketserver, sys, tempfile, threading

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


def toks(m):
    return len(m["content"]) // 4 + 8


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
        s.requests.append(body)
        ORDER.append(s.name)
        msgs = body["messages"]
        total = sum(toks(m) for m in msgs)
        if total + body["max_tokens"] > s.n_ctx:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"exceeds the context"}'); return
        common = 0
        for a, b in zip(s.last, msgs):
            if a != b:
                break
            common += toks(b)
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

print("conversation bench test: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)
