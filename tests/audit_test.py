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
# R15-01: a conversation with an image is no longer refused: the images become [image WxH] placeholders
import base64, struct, zlib
def _png(w, h):
    def ch(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(b"\0" * (h * (1 + 3 * w)))) + ch(b"IEND", b"")
PNG_URL = "data:image/png;base64," + base64.b64encode(_png(37, 19)).decode()
job5 = ui.AuditJob()
try:
    job5.start(main, 1024, [{"role": "user", "content": "x", "images": [PNG_URL]}], "the cat", max_tokens=100000)
    ok = True
except ValueError as e:
    ok = False
job5.cancel()
t0 = time.time()
while job5.state not in ("done", "error", "cancelled") and time.time() - t0 < 60: time.sleep(0.2)
check(ok and job5.notes and "placeholder" in job5.notes[0] and "1 image " in job5.notes[0], f"a conversation with an image is accepted, the result says the images were placeholders ({job5.notes})")
check(job5.max_tokens == ui.AUDIT_MAX_TOKENS_LIMIT, "a request may ask for more tokens, up to the limit")
main.stop()

# ---- the page ----
page = open(os.path.join(HERE, "ui", "index.html"), encoding="utf-8").read()
check("data-audit=" in page and "/api/audit/start" in page and "/api/audit/plan" in page and "function buildMsgs(c, upto)" in page and "const msgs = buildMsgs(c, c.messages.length);" in page,
      "the page: the audit chip, the plan and start routes, one function builds the messages for a new answer and for an audit")
check("confirm(" in page.split("async function startAudit")[1].split("async function pollAudit")[0], "the page asks before it starts anything (plan text and a confirm)")


# ---- R15-01: images, the thinking part, the larger cut ----
check(au.MAX_TOKENS == 16384 and ui.AUDIT_MAX_TOKENS == 16384, "the audit scores 16384 tokens (was 4096)")
check(au.image_size(PNG_URL) == (37, 19), "image size from a PNG header")
gif = "data:image/gif;base64," + base64.b64encode(b"GIF89a" + struct.pack("<HH", 640, 480) + b"\0" * 20).decode()
check(au.image_size(gif) == (640, 480), "image size from a GIF header")
jpeg = b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\0" + b"\0" * 9 + b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", 768, 1024) + b"\x03" + b"\0" * 9
check(au.image_size("data:image/jpeg;base64," + base64.b64encode(jpeg).decode()) == (1024, 768), "image size from a JPEG start-of-frame (after a JFIF segment)")
webp = b"RIFF\0\0\0\0WEBPVP8X" + b"\0" * 8 + (299).to_bytes(3, "little") + (199).to_bytes(3, "little")
check(au.image_size("data:image/webp;base64," + base64.b64encode(webp).decode()) == (300, 200), "image size from a WebP (VP8X) header")
check(au.image_size("data:image/png;base64,AAAA") is None and au.image_size("nonsense") is None, "an unreadable image has no size, no exception")
msgs_i, n_i = au.with_image_placeholders([{"role": "user", "content": "what is this", "images": [PNG_URL, "data:image/png;base64,AAAA"]}, {"role": "assistant", "content": "a cat"}])
check(n_i == 2 and msgs_i[0]["content"] == "what is this\n[image 37x19] [image]" and "images" not in msgs_i[0] and msgs_i[1] == {"role": "assistant", "content": "a cat"}, f"images become placeholders in the text ({msgs_i[0]['content']!r})")
check(au.with_image_placeholders([{"role": "user", "content": "", "images": [PNG_URL]}])[0][0]["content"] == "[image 37x19]", "an image-only message becomes just the placeholder")

TPL = r"""
import http.server, json
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = ""
        for i, m in enumerate(body["messages"]):
            out += "<|im_start|>" + m["role"] + "\n"
            if m["role"] == "assistant" and m.get("reasoning_content") and MODE == "think":
                out += "<think>\n" + m["reasoning_content"] + "\n</think>\n\n"
            out += m["content"] + "<|im_end|>\n"
        out += "<|im_start|>assistant\n"
        b = json.dumps({"prompt": out}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
"""
for mode in ("think", "nothink"):
    g = {"MODE": mode}
    exec(TPL, g)
    sv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), g["H"])
    threading.Thread(target=sv.serve_forever, daemon=True).start()
    u2 = f"http://127.0.0.1:{sv.server_address[1]}"
    conv = [{"role": "user", "content": "hi"}]
    pr = au.render_prompt(u2, conv)
    txt, how = au.answer_with_reasoning(u2, conv, pr, "Hello there", "The user says hi.", None)
    if mode == "think":
        check(txt == "<think>\nThe user says hi.\n</think>\n\nHello there" and how == "chat template", f"the thinking part is scored as the model's own template writes it ({txt!r})")
    else:
        check(txt.endswith("\n</think>\n\nHello there") and "The user says hi." in txt and how.startswith("reasoning +"), f"a template that drops the thinking part: the fall-back form says so ({how[:30]})")
    check(au.answer_with_reasoning(u2, conv, pr, "Hello there", "", None) == ("Hello there", "answer only"), "no thinking part: the answer alone")
    sv.shutdown()
summ = ui.audit_summary({"n": 3, "same": 3, "same_pct": 100.0, "mean_logloss": 0.1, "unknown_p": 0, "truncated": False, "prompt_tokens": 5, "seconds": 1.0, "mismatches": [], "tokens": [
    {"i": 0, "piece": "ab", "same": False, "exact_piece": "x", "exact_p": .5, "p": .1, "p_below": None}, {"i": 1, "piece": "cd", "same": True}, {"i": 2, "piece": "ef", "same": False, "exact_piece": "y", "exact_p": .5, "p": .1, "p_below": None}]},
    "abcdef", content_offset=4)
check(summ["ranges"] == [[0, 2, 2]] and summ["think_ranges"] == [[0, 2, 0]] and summ["content_offset"] == 4, f"ranges inside the answer are shifted to start at 0 ({summ['ranges']}, {summ['think_ranges']})")
summ2 = ui.audit_summary({"n": 1, "same": 0, "same_pct": 0.0, "mean_logloss": None, "unknown_p": 0, "truncated": False, "prompt_tokens": 5, "seconds": 1.0, "mismatches": [0], "tokens": [
    {"i": 0, "piece": "ab", "same": False, "exact_piece": "x", "exact_p": .5, "p": .1, "p_below": None}]}, "abcdef"[:2], content_offset=2)
check(summ2["think_ranges"] == [[0, 2, 0]] and summ2["ranges"] == [], "a changed token inside the thinking part goes to think_ranges")
check('reasoning: m.think || ""' in page and "else if (last0) out.push" in page and "data-audit=" in page, "the page sends the thinking part and offers the audit for answers that have one")

print("audit_test:", "FAILED" if fails else "ok", f"({len(fails)} failed)" if fails else "")
sys.exit(1 if fails else 0)
