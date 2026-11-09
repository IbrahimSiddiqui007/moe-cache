#!/usr/bin/env python3
"""GUI-043: the audit's scoring (bin/moe_cache_audit.py) against a scripted fake llama-server: positions, same-top share, probabilities, unknown probabilities,
cancel, errors, and the character ranges of the changed words. Standard library only.
   python3 tests/audit_test.py"""
import http.server, json, math, os, sys, threading

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_audit as au  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# ---- a fake server: words are tokens; the "exact model" wants the sequence EXACT after the prompt ----
FAKE = r"""
import http.server, json, threading
VOCAB = {w: i + 100 for i, w in enumerate("<s> <u> the cat sat on a mat dog ran fast hat ate fish café !".split())}
INV = {v: k for k, v in VOCAB.items()}
EXACT = [VOCAB[w] for w in "the cat sat on a mat".split()]
calls = []
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.reply({"status": "ok"} if self.path == "/health" else {"tokens": []} if self.path == "/slots" else {}, 200)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        calls.append((self.path, body))
        if self.headers.get("Authorization") == "Bearer wrong":
            return self.reply({"error": "no"}, 401)
        if self.path == "/apply-template":
            return self.reply({"prompt": "<u> " + " ".join(m["content"] for m in body["messages"])})
        if self.path == "/tokenize":
            words = body["content"].split()
            toks = [{"id": VOCAB[w], "piece": ("" if (i == 0 and not body["add_special"]) else " ") + w} for i, w in enumerate(words)] if not body["add_special"] else \
                   [{"id": VOCAB["<s>"], "piece": "<s>"}] + [{"id": VOCAB[w], "piece": " " + w} for w in words]
            return self.reply({"tokens": toks})
        if self.path == "/completion":
            if body.get("n_probs", 0) < 1: return self.reply({"content": ""})
            ids = body["prompt"]
            k = len(ids) - 2                                            # the prompt "<s> <u>" is two tokens
            ex = EXACT[k]
            others = [(100 + (ex + j) % 15, -2.0 - j) for j in range(1, body["n_probs"])]
            tops = [{"id": ex, "token": INV[ex], "logprob": -0.1}] + [{"id": i, "token": INV[i], "logprob": lp} for i, lp in others]
            return self.reply({"completion_probabilities": [{"id": ex, "token": INV[ex], "logprob": -0.1, "top_logprobs": tops}], "tokens_cached": len(ids) - 1})
        self.reply({}, 404)
    def reply(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
def serve(port):
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
"""
ns = {}
exec(FAKE, ns)
VOCAB, calls = ns["VOCAB"], ns["calls"]
srv = ns["serve"](0)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}"

# ---- pieces of the protocol ----
check(au.render_prompt(url, [{"role": "user", "content": "hi"}]) == "<u> hi", "apply-template: the prompt the chat template makes")
check(au.tokenize(url, "the cat", special=False) == [{"id": VOCAB["the"], "piece": "the"}, {"id": VOCAB["cat"], "piece": " cat"}], "tokenize: ids and pieces")

# ---- an answer that differs from the exact one at positions 2 and 5 ----
answer = "the cat ran on a hat"                                        # exact: the cat sat on a mat
res = au.score(url, "<u>", answer, top=3)
check(res["n"] == 6 and res["same"] == 4 and abs(res["same_pct"] - 100 * 4 / 6) < 1e-9 and res["mismatches"] == [2, 5], f"same top token 4 of 6, positions 2 and 5 differ ({res['same']}/{res['n']}, {res['mismatches']})")
t2 = res["tokens"][2]
check(t2["piece"] == " ran" and t2["exact_piece"] == "sat" and not t2["same"] and abs(t2["exact_p"] - math.exp(-0.1)) < 1e-9, f"position 2: answer 'ran', exact 'sat' with its probability ({t2['piece']}, {t2['exact_piece']})")
check(res["tokens"][0]["same"] and abs(res["tokens"][0]["p"] - math.exp(-0.1)) < 1e-9, "a position that agrees: the exact probability of the answer's token is the top probability")
known = [r for r in res["tokens"] if r["p"] is not None]
unknown = [r for r in res["tokens"] if r["p"] is None]
check(len(known) + len(unknown) == 6 and res["unknown_p"] == len(unknown), f"positions whose token is outside the top 3 have no probability ({len(unknown)} unknown)")
check(all(r["p_below"] is not None and r["p_below"] <= math.exp(-2.0) + 1e-12 for r in unknown), "...but a bound: less likely than the last of the top list")
check(res["mean_logloss"] is not None and res["mean_logloss"] > 0 and res["prompt_tokens"] == 2 and res["truncated"] is False, "summary: mean log-loss over the known positions, prompt length, not truncated")
cs = [b for p, b in calls if p == "/completion"]
check(len(cs) == 6 and all(b["n_predict"] == 1 and b["temperature"] == 0 and b["cache_prompt"] is True for b in cs) and cs[0]["prompt"] == [VOCAB["<s>"], VOCAB["<u>"]] and cs[3]["prompt"][-3:] == [VOCAB["the"], VOCAB["cat"], VOCAB["ran"]],
      "one /completion per position, the answer so far as token ids, the prompt cache on, greedy")

# ---- progress, cancel, truncation, errors ----
seen = []
au.score(url, "<u>", "the cat", progress=lambda d, n: seen.append((d, n)))
check(seen == [(1, 2), (2, 2)], f"progress after each position ({seen})")
state = {"n": 0}
def cancel():
    state["n"] += 1
    return state["n"] > 2
try:
    au.score(url, "<u>", answer, cancel=cancel); ok = False
except au.AuditError as e:
    ok = "cancelled" in str(e)
check(ok, "cancel stops the audit")
check(au.score(url, "<u>", answer, max_tokens=3)["truncated"] is True and au.score(url, "<u>", answer, max_tokens=3)["n"] == 3, "a long answer is cut at max_tokens and says so")
for bad, what in ((lambda: au.score(url, "<u>", "", top=3), "an empty answer"), (lambda: au.score(url, "<u>", "the cat", key="wrong"), "a wrong API key"), (lambda: au.score("http://127.0.0.1:1", "<u>", "the"), "no server")):
    try:
        bad(); ok = False
    except au.AuditError:
        ok = True
    check(ok, f"{what} is an AuditError, not a crash")

# ---- character ranges ----
rs = au.char_ranges(res["tokens"], answer)
check(rs == [[7, 11, 2], [16, 20, 5]] and [answer[a:b] for a, b, _ in rs] == [" ran", " hat"], f"the changed words as character ranges ({rs})")
ans2 = "the café ran"
toks2 = [{"i": 0, "piece": "the", "same": True}, {"i": 1, "piece": " café", "same": False}, {"i": 2, "piece": " ran", "same": True}]
r2 = au.char_ranges(toks2, ans2)
check(r2 == [[3, 8, 1]] and ans2[3:8] == " café", f"ranges count characters, not bytes ({r2})")
check(au.char_ranges([{"i": 0, "piece": "x", "same": False}], "different text") == [], "pieces that do not add up to the text give no ranges (nothing invented)")

# ---- the job: a second exact-mode server beside the running one, or pause-and-restore (the launcher is faked by the same server) ----
import tempfile, time, socket as _s
TMP = tempfile.mkdtemp(prefix="moe-audit-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache"); os.environ["XDG_CONFIG_HOME"] = os.path.join(TMP, "config")
import moe_cache_ui as ui  # noqa: E402
LAUNCH = FAKE + r"""
import sys, time
port = int(sys.argv[sys.argv.index("--port") + 1])
srv = serve(port)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print("moe-cache: web UI and API will be at http://127.0.0.1:%d" % port, flush=True)
print("ARGS " + " ".join(sys.argv[1:]), flush=True)
time.sleep(600)
"""
ui.LAUNCHER = [sys.executable, "-c", LAUNCH]
gg = os.path.join(TMP, "fake.gguf"); open(gg, "wb").write(b"GGUF" + b"\0" * 64)
def free():
    k = _s.socket(); k.bind(("127.0.0.1", 0)); p = k.getsockname()[1]; k.close(); return p
main_port = free()
main = ui.Runner()
check(not ui.audit_plan(main, 1024)["ok"], "the plan says 'start the model first' when no server runs")
main.start({"model": gg, "mode": "fast", "engine": "auto", "mtp": True, "agent": True, "no_warm": False, "leave_room": False, "no_long_prompt": False, "approx": "safe", "port": main_port, "ctx": 4096, "ram": 1.5,
            "llama_server": "llama-server"})
t0 = time.time()
while main.state != "ready" and time.time() - t0 < 60: time.sleep(0.3)
check(main.state == "ready", "the fake 'server' is ready")
pl = ui.audit_plan(main, 1024)
check(pl["ok"] and pl["mode"] == "beside" and "next to the running one" in pl["why"], f"small model: beside ({pl['why'][:70]})")
big = ui.audit_plan(main, 10 ** 15)
check(big["ok"] and big["mode"] == "pause" and "stopped for the audit" in big["why"] and "started again afterwards" in big["why"], "model too big for the free RAM: pause-and-audit, with the consequences said")

job = ui.AuditJob()
msgs = [{"role": "user", "content": ""}]          # the fake template makes "<u> " of it: the prompt is exactly two tokens
job.start(main, 1024, msgs, "the cat ran on a hat")
t0 = time.time()
while job.state not in ("done", "error", "cancelled") and time.time() - t0 < 120: time.sleep(0.3)
check(job.state == "done" and job.mode == "beside", f"beside: the audit finishes ({job.state} {job.message})")
r = job.result
check(r and r["n"] == 6 and r["same"] == 4 and r["n_mismatches"] == 2 and r["mismatches"] == [2, 5] and r["flags"] == "001001" and r["ranges"] == [[7, 11, 2], [16, 20, 5]], f"beside: the summary the page gets ({r and (r['same'], r['mismatches'], r['ranges'])})")
check(r["detail"]["2"]["exact"] == "sat" and r["detail"]["2"]["piece"] == " ran", "beside: both tokens of a changed position")
log = [l for _, l in main.log]
check(main.state == "ready", "beside: the running server is untouched")
time.sleep(1.5)
check(all(not ui.mc.port_busy(p) for p in range(8190, 8200)), "beside: the exact-mode server is stopped afterwards")

job2 = ui.AuditJob()
job2.start(main, 1024, msgs, "the cat sat")
t0 = time.time()
while job2.state not in ("done", "error", "cancelled") and time.time() - t0 < 120: time.sleep(0.2)
check(job2.state == "done" and job2.result["same"] == 3, "a second audit after the first works")

args2 = ui.server_args(dict(main.settings, mode="exact", mtp=False, agent=False, approx="", no_warm=True, port=8190, api_key="", no_long_prompt=True))
check("--exact" in args2 and "--agent" not in args2 and "--fast" not in args2 and "--approx" not in args2, f"the audit's server is started in exact mode, no agent, no approximate tier ({[a for a in args2 if a.startswith('--')]})")

# pause-and-restore
job3 = ui.AuditJob()
old_pid = main.pid
job3.start(main, 10 ** 15, msgs, "the cat ran")
t0 = time.time()
while job3.state not in ("done", "error", "cancelled") and time.time() - t0 < 120: time.sleep(0.3)
check(job3.state == "done" and job3.mode == "pause" and job3.result["n"] == 3, f"pause: the audit finishes ({job3.state} {job3.message})")
t0 = time.time()
while main.state != "ready" and time.time() - t0 < 60: time.sleep(0.3)
check(main.state == "ready" and main.settings.get("port") == main_port and main.settings.get("mode") == "fast" and main.settings.get("approx") == "safe", f"pause: the server is started again with its own settings ({main.state}, {main.settings.get('mode')})")

# cancel
job4 = ui.AuditJob()
job4.start(main, 1024, msgs, "the cat sat on a mat " * 1)
job4.cancel()
t0 = time.time()
while job4.state not in ("done", "error", "cancelled") and time.time() - t0 < 60: time.sleep(0.2)
check(job4.state in ("cancelled", "done"), f"cancel ends the job ({job4.state})")
try:
    ui.AuditJob().start(main, 1024, [{"role": "user", "content": "x", "images": ["data:..."]}], "the")
    ok = False
except ValueError as e:
    ok = "text only" in str(e)
check(ok, "a conversation with an image is refused: the audit reads text only")
main.stop()

# ---- the page ----
page = open(os.path.join(HERE, "ui", "index.html"), encoding="utf-8").read()
check("data-audit=" in page and "/api/audit/start" in page and "/api/audit/plan" in page and "function buildMsgs(c, upto)" in page and "const msgs = buildMsgs(c, c.messages.length);" in page,
      "the page: the audit chip, the plan and start routes, one function builds the messages for a new answer and for an audit")
check("confirm(" in page.split("async function startAudit")[1].split("async function pollAudit")[0], "the page asks before it starts anything (plan text and a confirm)")

print("audit_test:", "FAILED" if fails else "ok", f"({len(fails)} failed)" if fails else "")
sys.exit(1 if fails else 0)
