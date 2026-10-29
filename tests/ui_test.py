#!/usr/bin/env python3
"""Control page (GUI1, bin/moe_cache_ui.py) without a model: fake GGUF files, a mocked planner and a fake launcher that answers /health and
prints the lines llama-server and the plugin print (timings, 'moe-cache-stats {json}'). Linux and Windows; standard library only.
   python3 tests/ui_test.py"""
import http.client, json, os, socket, sys, tempfile, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-ui-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")          # before the import: the launcher reads it once
os.environ["XDG_CONFIG_HOME"] = os.path.join(TMP, "config")        # 'connect opencode --write' writes here, not into the real config
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_ui as ui  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# ---- fake models ----
models = os.path.join(TMP, "models")
os.makedirs(os.path.join(models, "sub")); os.makedirs(os.path.join(models, ".hidden"))
def touch(p, n=1024):
    with open(os.path.join(models, p), "wb") as f:
        f.write(b"GGUF" + b"\0" * (n - 4))
touch("Small-MoE-Q4_K_M.gguf")
touch("sub/Big-UD-IQ2_XXS-00001-of-00002.gguf", 2000); touch("sub/Big-UD-IQ2_XXS-00002-of-00002.gguf", 3000)
touch("mmproj-Small-F16.gguf"); touch(".hidden/Secret-Q8_0.gguf"); touch("notes.txt")
ui.default_folders = lambda: [models]
ui.ollama_roots = lambda: []

found = ui.find_models([models])
names = [m["name"] for m in found]
check(names == ["Big-UD-IQ2_XXS", "Small-MoE-Q4_K_M"], f"model list: split model once, projector and hidden folders left out ({names})")
big = next(m for m in found if m["name"].startswith("Big"))
small = next(m for m in found if m["name"].startswith("Small"))
check(big["size"] == 5000 and big["quant"] == "UD-IQ2_XXS" and small["quant"] == "Q4_K_M", "sizes add up over the parts; quant from the name")

# ---- quant from the header (Ollama's blobs have no name) and measured speeds by model family ----
import struct
def gguf_header(path, kv):
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)))
        for k, v in kv:
            f.write(struct.pack("<Q", len(k)) + k.encode())
            if isinstance(v, str):
                f.write(struct.pack("<IQ", 8, len(v)) + v.encode())
            else:
                f.write(struct.pack("<II", 4, v))
blob = os.path.join(TMP, "sha256-0123")
gguf_header(blob, [("general.architecture", "qwen35moe"), ("qwen35moe.block_count", 41), ("qwen35moe.expert_count", 256), ("general.file_type", 15)])
check(ui.header_quant(blob) == "Q4_K_M", f"quant from general.file_type ({ui.header_quant(blob)})")
m = ui.measured(blob, 21.7e9)
check(len(m) == 4 and "22.6" in m[1] and "RESULTS 44" in m[1] and m[1].startswith("laptop,"), "measured speeds for a known model family and size (Qwen3.6-35B Q4_K_M)")
check(m[3].startswith("PC, 32 GB RAM") and "RESULTS 60" in m[3], f"measured: the PC's row too ({m[3]})")
fnx = os.path.join(TMP, "fnx-blob")
gguf_header(fnx, [("general.architecture", "qwen4exp"), ("qwen4exp.block_count", 48), ("qwen4exp.expert_count", 512), ("general.file_type", 15)])
mf = ui.measured(fnx, 68.0e9)
check(any("stock 14.89 tok/s, moe-cache 21.94" in r for r in mf) and any(r.startswith("PC, 16 GB RAM") and "stock" not in r for r in mf),
      f"measured: Flash-Next IQ2_XS on the PC, and the --ram 16 row without a stock number ({mf})")
mq = ui.measured(fnx, 66.4e9)
check(len(mq) == 1 and "stock 18.41 tok/s, moe-cache 22.16" in mq[0] and "Q2_0" in mq[0], f"measured: Flash-Next Q2_0 gets its own row, not IQ2_XS's (closest size; {mq})")
check(any("moe-cache 21.94" in r for r in ui.measured(fnx, 67.5e9)), "measured: a size nearer IQ2_XS still gets IQ2_XS")
m122 = os.path.join(TMP, "q122-blob")
gguf_header(m122, [("general.architecture", "qwen35moe"), ("qwen35moe.block_count", 48), ("qwen35moe.expert_count", 256)])
check(any("stock 15.95 tok/s, moe-cache 19.73" in r and "RESULTS 61" in r for r in ui.measured(m122, 36.6e9)), "measured: the 122B row is RESULTS 61's re-measure (+24 %)")
check(ui.measured(blob, 30e9) == [], "no measured speeds for another size of the same family")

# ---- the plan and its reasons (planner mocked) ----
PLAN = {"arch": "qwen3moe", "layers": 48, "experts": 128, "used": 8, "n_cpu_moe": 40, "cache_GiB": 6.0, "cpu_experts_GiB": 18.0,
        "free_vram_GiB": 7.5, "note": ""}
TUNE = {"n_cpu_moe": 38, "ub": 1024, "gpu_cache_mib": 0, "why": ["10 layers of experts stay on the GPU", "micro-batch 1024"]}
real_planner, ui.run_planner = ui.run_planner, lambda path, ctx, ram, vram, kv="f16": dict(PLAN)
ui.run_tuner = lambda path, ctx, vram, kv_q8: dict(TUNE)

p = ui.plan_model(small["path"], ram="2")
check(p["kind"] == "stock" and "fits in RAM" in p["verdict"], f"model + 1.5 GB under the RAM budget: stock ({p['verdict']})")
check(any("RESULTS 44" in r for r in p["reasons"]), "stock: says why stock is faster")
check(p["summary"].startswith("Stock llama-server: the model") and "fastest" in p["summary"], f"summary: why stock ({p['summary']})")
p = ui.plan_model(small["path"], ram="1")
check(p["kind"] == "cache" and p["verdict"] == "does not fit: moe-cache exact", f"over the budget: moe-cache exact ({p['verdict']})")
check(any("10 of 48 layers keep their experts on the GPU" in r for r in p["reasons"]), "reasons: GPU layers from the tuner (as the launcher)")
check(any("holds 33 %" in r for r in p["reasons"]), "reasons: share of the CPU experts the cache holds")
check(any(r.startswith("tuner: ") for r in p["reasons"]), "reasons: the tuner's own lines")
check(p["summary"].startswith("moe-cache: the model") and "keeps 33 %" in p["summary"] and "same as stock" in p["summary"], f"summary: why moe-cache ({p['summary']})")
p = ui.plan_model(small["path"], ram="1", engine="stock")
check(p["kind"] == "stock" and "Run as Stock" in p["summary"] and "compare" in p["summary"], f"summary: stock as asked, model bigger than RAM ({p['summary']})")
p = ui.plan_model(small["path"], ram="2", engine="plugin")
check(p["kind"] == "cache" and "as you chose" in p["summary"] and "stock would likely be faster" in p["summary"], f"summary: plugin on a model that fits ({p['summary']})")
p = ui.plan_model(small["path"], ram="2", mode="fast")
check(p["kind"] == "cache" and "needs the plugin" in p["verdict"], f"--fast keeps the plugin even when the model fits ({p['verdict']})")
TUNE["n_cpu_moe"] = 0
p = ui.plan_model(small["path"], ram="1")
check(p["kind"] == "gpu" and "nothing to do" in p["summary"], f"every expert layer on the GPU ({p['verdict']}; {p['summary']})")
TUNE["n_cpu_moe"] = 38
ui.run_planner = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no expert tensors (blk.N.ffn_{gate,up,down,gate_up}_exps) found: this does not look like a MoE GGUF"))
p = ui.plan_model(small["path"], ram="1")
check(p["kind"] == "error" and any("not a MoE model" in r for r in p["reasons"]), "planner error: shown with the planner's own words")
check(p["summary"].startswith("The planner could not read"), f"summary: planner error ({p['summary']})")
ui.run_planner = lambda path, ctx, ram, vram, kv="f16": dict(PLAN)

args = ui.server_args({"model": "m.gguf", "port": 8123, "ctx": 32768, "ram": 12, "mode": "fast", "agent": True})
check(args[:2] == ["m.gguf", "--port"] and "--fast" in args and "--agent" in args and args[-2:] == ["--", "--metrics"] and "12.0" in args,
      "launcher arguments: " + " ".join(args))
check("KEY" in ui.shown_command({"model": "m.gguf", "api_key": "secret"}) and "secret" not in ui.shown_command({"model": "m.gguf", "api_key": "secret"}),
      "the shown command hides the API key")

# --dry-run must reach the launcher, not llama-server (it went after the -- once)
seen = []
ui.subprocess.run, real_run = (lambda cmd, **kw: seen.append(cmd) or type("R", (), {"stdout": "", "stderr": ""})()), ui.subprocess.run
ui.dry_run({"model": "m.gguf"})
ui.subprocess.run = real_run
check(seen and seen[0].index("--dry-run") < seen[0].index("--"), "--dry-run goes to the launcher, before the --")

# ---- a fake launcher: serves /health and /slots, prints the plugin's lines, runs until stopped ----
fake = os.path.join(TMP, "fake_launcher.py")
with open(fake, "w") as f:
    f.write(r'''
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = {"/health": {"status": "ok"}, "/slots": [{"id": 0, "is_processing": False, "n_prompt_tokens": 4100, "n_ctx": 8192}], "/v1/models": {"data": [{"id": "fake-moe"}]},
                "/props": {"default_generation_settings": {"n_ctx": 65536}}}.get(self.path)
        self.send_response(200 if body is not None else 404); self.end_headers(); self.wfile.write(json.dumps(body).encode())
    def do_POST(self):
        q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/detokenize":
            body = {"content": "You are opencode, an interactive CLI tool" if q["tokens"][:3] == [11, 22, 33] else "Hello"}
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(body).encode()); return
        if self.path == "/v1/chat/completions":         # the page's own chat: streamed; the page server must have added the key
            ok = self.headers.get("Authorization") == "Bearer sekrit-123" and q["stream"] is True and q.get("temperature") == 0.5 and set(q["messages"][-1]) == {"role", "content"}
            self.send_response(200 if ok else 401); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            if not ok:
                self.wfile.write(json.dumps({"error": {"message": "Invalid API Key"}}).encode()); return
            for piece in ("Hel", "lo"):
                self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n\n"); self.wfile.flush()
            t = {"predicted_n": 2, "predicted_per_second": 25.0, "predicted_ms": 80.0, "prompt_n": 5, "prompt_per_second": 500.0, "cache_n": 0, "draft_n": 4, "draft_n_accepted": 3}
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": t}).encode() + b"\n\ndata: [DONE]\n\n"); return
        assert self.path == "/completion" and q["temperature"] == 0 and q["seed"] == 42 and q["cache_prompt"] is False
        t = {"predicted_n": q["n_predict"] if q.get("ignore_eos") else min(q["n_predict"], 149), "predicted_per_second": 20.0 + len(q["prompt"]) % 7 - (10.0 if "--stock" in sys.argv else 0), "prompt_n": len(q["prompt"]) // 4, "prompt_per_second": 300.0}
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({"content": "x", "timings": t}).encode())
srv = HTTPServer(("127.0.0.1", port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
import os
print("fake: argv has the key: %s, LLAMA_API_KEY set: %s" % (any("sekrit" in x for x in sys.argv), bool(os.environ.get("LLAMA_API_KEY"))), flush=True)
print("fake: agent snippet apiKey auto-key-ABCDEFGH", flush=True)
print("moe-cache: GPU AMD Radeon RX 9070 XT (Vulkan0): 15.00 GiB free (test)", flush=True)
print("moe-cache: AMD GPU on Windows, driver older than 32.0.32015.7029: --ctx-checkpoints 0 (test)", flush=True)
if "--stock" in sys.argv:
    print("moe-cache: --stock: running stock llama-server (test)", flush=True)
else:
    print("moe-cache: experts of 38 layers stay on the CPU and are managed by the cache; profile /tmp/x.usage; plugin loaded by preload", flush=True)
print("moe-cache: MTP on: the model has an MTP head (1 layer) and llama-server has draft-mtp: --spec-type draft-mtp --spec-draft-n-max 2 (test)", flush=True)
print(f"moe-cache: web UI and API will be at http://127.0.0.1:{port}", flush=True)
time.sleep(0.5)
print("prompt eval time =     100.00 ms /    50 tokens (    2.00 ms per token,   500.00 tokens per second)", flush=True)
print("       eval time =    1000.00 ms /    25 tokens (   40.00 ms per token,    25.00 tokens per second)", flush=True)
print("moe-cache: warm session: restored a saved agent conversation (4096 tokens) into slot 0 in 1.0 s", flush=True)
print("moe-cache: warm session: saved slot 0 (4096 tokens, 120 MB) in 0.5 s", flush=True)
for _ in range(2):
    print("0.01.002.003 W moe-cache: warning: the expert cache shrank from 10.0 to 8.0 GiB because the rest of llama-server now uses 3.0 GiB", flush=True)
print('moe-cache-stats {"hits":90,"misses":10,"read_bytes":0,"resident_bytes":6000000000,"budget":7000000000}', flush=True)
time.sleep(1.0)
print('moe-cache-stats {"hits":170,"misses":30,"read_bytes":50000000,"resident_bytes":6500000000,"budget":7000000000}', flush=True)
while True:
    time.sleep(1)
''')
ui.LAUNCHER = [sys.executable, fake]
AUTO_KEY = "auto-key-ABCDEFGH"
ui.KEY_CMD = [sys.executable, "-c", f"print('{AUTO_KEY}')"]           # the launcher's `key` command (option C), faked

TOKEN = "test-token-123"
srv = ui.make_server(0, "llama-server", token=TOKEN)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"

def req(path, body=None, headers=None):
    h = dict({"X-MoeCache-Token": TOKEN}, **(headers or {}))
    h = {k: v for k, v in h.items() if v is not None}
    if body is not None:
        h.setdefault("Content-Type", "application/json")
    r = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, h)
    try:
        with urllib.request.urlopen(r, timeout=30) as x:
            return x.status, x.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

st, page = req("/")
check(st == 200 and "<title>moe-cache</title>" in page and "/api/status" in page, "GET / serves the page")

# ---- security review (2026-10-26): frame protection, per-run token, Sec-Fetch-Site, body cap, ranges, CSV ----
def raw(method, path, headers=None, body=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse(); data = r.read(); c.close()
    return r.status, r.getheader, data.decode(errors="replace")
st, hdr, _ = raw("GET", "/?t=" + TOKEN)
cookie = hdr("Set-Cookie") or ""
check(st == 303 and hdr("Location") == "/" and cookie.startswith(f"moe_cache_ui_{srv.server_address[1]}={TOKEN}") and "HttpOnly" in cookie and "SameSite=Strict" in cookie,
      f"the launcher's link sets an HttpOnly, SameSite=Strict cookie and drops the token from the address ({st}, {cookie})")
st, hdr, page = raw("GET", "/", {"Cookie": cookie.split(";")[0]})
check(st == 200 and "<title>moe-cache</title>" in page, "the page loads with the cookie")
check(hdr("X-Frame-Options") == "DENY" and "frame-ancestors 'none'" in (hdr("Content-Security-Policy") or "") and "frame-src 'none'" in hdr("Content-Security-Policy"),
      "1: no framing (X-Frame-Options DENY, CSP frame-ancestors 'none'); the page has no frames either (own chat)")
st, hdr, _ = raw("GET", "/api/status")
check(st == 403 and hdr("X-Frame-Options") == "DENY", "1: the headers are on errors too")
st, _, page = raw("GET", "/")
check(st == 403 and "?t=" in page, "2: the page without the token: 403 and how to open it")
check(req("/api/status", headers={"X-MoeCache-Token": None})[0] == 403, "2: /api without the token: 403")
check(req("/api/status", headers={"X-MoeCache-Token": "wrong"})[0] == 403, "2: /api with a wrong token: 403")
check(req("/api/stop", {}, headers={"X-MoeCache-Token": None})[0] == 403, "2: POST without the token: 403")
check(raw("GET", "/?t=wrong")[0] == 403, "2: a wrong token in the link: 403")
check(req("/api/status", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403, "3: GET with Sec-Fetch-Site cross-site: 403")
check(req("/api/status", headers={"Sec-Fetch-Site": "same-site"})[0] == 403, "3: GET from another port of 127.0.0.1 (same-site): 403")
check(req("/api/status", headers={"Sec-Fetch-Site": "none"})[0] == 403, "3: an API call typed into the address bar (none): 403")
check(req("/api/status", headers={"Sec-Fetch-Site": "same-origin"})[0] == 200, "3: same-origin: allowed")
check(req("/api/stop", {}, headers={"Sec-Fetch-Site": "cross-site"})[0] == 403, "3: POST cross-site: 403")
check(req("/api/status", headers={"Origin": "http://attacker.example"})[0] == 403, "3: GET with a foreign Origin: 403")
check(raw("GET", "/", {"Sec-Fetch-Site": "none", "Cookie": cookie.split(";")[0]})[0] == 200, "3: opening the page itself (none) works")
# audit 2026-10-27: the browser gets a private file:// page that forwards to the link (no token in its command line, readable in /proc)
st, hdr, page = raw("GET", "/?t=" + TOKEN, {"Sec-Fetch-Site": "cross-site"})
check(st == 200 and (hdr("Set-Cookie") or "").startswith(f"moe_cache_ui_{srv.server_address[1]}={TOKEN}") and "SameSite=Strict" in hdr("Set-Cookie")
      and "url=/'" in page, f"S1: the link opened from the launcher's file page sets the cookie and moves on to / itself ({st})")
check(raw("GET", "/?t=wrong", {"Sec-Fetch-Site": "cross-site"})[0] == 403, "S1: a cross-site link with a wrong token: 403")
check(raw("GET", "/", {"Sec-Fetch-Site": "cross-site", "Cookie": cookie.split(";")[0]})[0] == 403, "S1: cross-site without the link token stays refused")
check(raw("GET", "/api/status?t=" + TOKEN, {"Sec-Fetch-Site": "cross-site"})[0] == 403, "S1: the cross-site exception is for the page link only")
check(raw("GET", "/?t=%C3%A9")[0] == 403 and req("/api/status", headers={"X-MoeCache-Token": "\u00e9"})[0] in (400, 403),
      "S1: a non-ASCII token is refused, not a dropped connection")
opened = []
import webbrowser as _wb
real_wb, _wb.open = _wb.open, (lambda u: opened.append(u))
try:
    ui.open_page("http://127.0.0.1:1/?t=secret-x")
finally:
    _wb.open = real_wb
fpath = urllib.request.url2pathname(urllib.parse.urlparse(opened[0]).path) if opened else ""
check(opened and opened[0].startswith("file:") and "secret-x" not in opened[0] and "secret-x" in open(fpath, encoding="utf-8").read()
      and (os.name == "nt" or (os.stat(fpath).st_mode & 0o077) == 0), f"S1: the browser is given a private file, not the token ({opened})")
st, _, _ = raw("POST", "/api/folders", {"Content-Type": "application/json", "X-MoeCache-Token": TOKEN, "Content-Length": str(70000)}, b"x" * 70000)
check(st == 413, f"5: a body over 64 KiB is refused ({st})")
check(ui.Handler.timeout == 30, "5: a handler timeout is set")
for bad in ({"ram": "nan"}, {"ram": "inf"}, {"vram": "-1"}, {"ctx": "2000000"}, {"ctx": "1e400"}, {"port": "70000"}):
    st, body = req("/api/start", dict({"model": small["path"]}, **bad))
    check(st == 400, f"7: {bad} refused ({body[:80]})")
st, _ = req("/api/plan?model=" + urllib.request.quote(small["path"]) + "&ram=nan")
check(st == 400, "7: the plan checks the same ranges")
check([ui.csv_safe(x) for x in ("=cmd|' /C calc'!A0", "+1", "-2", "@SUM(1)", "\uff20x", "Qwen.gguf", 12.5)] ==
      ["'=cmd|' /C calc'!A0", "'+1", "'-2", "'@SUM(1)", "'\uff20x", "Qwen.gguf", "12.5"], "9: CSV cells that would run as formulas get an apostrophe")
check(req("/api/command?model=x")[0] == 404, "4: /api/command is POST only")
st, body = req("/api/models")
check(st == 200 and len(json.loads(body)["models"]) == 2, "GET /api/models")
st, body = req("/api/plan?model=" + urllib.request.quote(small["path"]) + "&ram=1&ctx=8192&mode=exact")
check(st == 200 and json.loads(body)["kind"] == "cache", "GET /api/plan")
st, _ = req("/api/plan?model=" + urllib.request.quote(os.path.join(TMP, "elsewhere.gguf")))
check(st == 400, "a model that is not in the list is refused")
st, _ = req("/api/status", headers={"Host": "attacker.example:80"})
check(st == 403, "a foreign Host header is refused (DNS rebinding)")
st, _ = req("/api/stop", {}, headers={"Origin": "http://attacker.example"})
check(st == 403, "a POST from a foreign origin is refused")
st, _ = req("/api/stop", {}, headers={"Content-Type": "text/plain"})
check(st == 403, "a POST that is not JSON is refused (no simple cross-site form posts)")
check(srv.server_address[0] == "127.0.0.1", "listens on 127.0.0.1 only")

with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
st, _ = req("/api/bench", {})
check(st == 400, "no benchmark without a running model")
st, body = req("/api/chat", {"messages": [{"role": "user", "content": "hi"}]})
check(st == 400 and "no model is running" in body, f"chat: no answer without a running model ({st})")
st, body = req("/api/start", {"model": small["path"], "on_demand": True})
check(st == 400 and "first request" in body and json.loads(req("/api/system")[1])["on_demand"] is False, "on-demand hook: refused until the launcher has it; /api/system says so")
st, body = req("/api/start", {"model": small["path"], "port": port, "ram": "1", "ctx": 8192, "mode": "exact", "agent": True, "api_key": "sekrit-123"})
check(st == 200, f"POST /api/start ({body[:120]})")
st, _ = req("/api/start", {"model": small["path"], "port": port + 1})
check(st == 400, "a second start while one runs is refused")
deadline, status = time.time() + 30, {}
while time.time() < deadline:
    status = json.loads(req("/api/status")[1])
    if status["state"] == "ready" and status["cache"].get("hit_rate") is not None:
        break
    time.sleep(0.3)
check(status.get("state") == "ready", f"the server becomes ready ({status.get('state')})")
check(status.get("mode") == "moe-cache exact + MTP", f"mode (and MTP) from the launcher's log ({status.get('mode')})")
check(status.get("last", {}).get("decode_tps") == 25.0 and status["last"].get("prefill_tps") == 500.0, "decode and prefill tok/s from the timing lines")
c = status.get("cache", {})
check(abs((c.get("hit_rate") or 0) - 0.8) < 1e-9, f"recent hit rate from two snapshots: 80/100 ({c.get('hit_rate')})")
check(abs((c.get("hit_rate_total") or 0) - 0.85) < 1e-9 and c.get("budget") == 7e9, "total hit rate and the cache budget")
check((c.get("read_mbs") or 0) > 0, f"SSD reads MB/s from read_bytes ({c.get('read_mbs')})")
check(f"--port {port}" in status.get("command", ""), "the status shows the launcher command")
d = status.get("decisions", [])
check(len(d) == 4 and d[0].startswith("GPU AMD Radeon") and d[1].startswith("AMD GPU on Windows") and d[2].startswith("experts of 38 layers") and d[3].startswith("MTP on:"),
      f"what the launcher decided, up to its URL line, without the prefix ({d})")
w = status.get("warnings", [])
check(len(w) == 1 and w[0]["count"] == 2 and w[0]["line"].startswith("moe-cache: warning: the expert cache shrank"),
      f"the shrink warning is lifted out of the log, once, counted, without llama-server's prefix ({w})")
check("sekrit" not in json.dumps(status), "the API key is not in the status")
check(status.get("api_key") == {"in_use": True, "source": "typed on this page", "ends": "-123"}, f"API key in use, masked, typed on the page ({status.get('api_key')})")
st, body = req("/api/key")
check(st == 200 and json.loads(body)["key"] == "sekrit-123", "Copy key: /api/key gives the key in use")
check(req("/api/key", headers={"X-MoeCache-Token": None})[0] == 403, "Copy key: not without the page token")

# ---- the page's own chat (3-pane page): streamed through the page server, which adds llama-server's API key ----
HI = {"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi", "shown": "hi", "files": []}], "temperature": 0.5}
st, body = req("/api/chat", HI)
ev = [json.loads(l[6:]) for l in body.splitlines() if l.startswith("data: {")]
text = "".join((((e.get("choices") or [{}])[0]).get("delta") or {}).get("content", "") for e in ev)
check(st == 200 and text == "Hello" and body.rstrip().endswith("data: [DONE]") and any((e.get("timings") or {}).get("draft_n_accepted") == 3 for e in ev),
      f"chat: streamed answer, timings (MTP drafts) passed through, key added by the page server ({st}, {text!r})")
check("sekrit" not in body, "chat: the key is not in the stream")
moe = [e["moe"] for e in ev if "moe" in e]
check(len(moe) <= 1 and all(set(m) == {"hits", "misses", "hit_rate"} for m in moe), f"chat: at most one per-answer cache event ({moe})")
for bad, what in (({"messages": []}, "no messages"), ({"messages": [{"role": "tool", "content": "x"}]}, "an unknown role"), ({"messages": [{"role": "user", "content": 5}]}, "content not text"),
                  (dict(HI, temperature=99), "temperature out of range"), (dict(HI, max_tokens="nan"), "max_tokens not a number")):
    check(req("/api/chat", bad)[0] == 400, f"chat: {what} is refused")
check(req("/api/chat", HI, headers={"X-MoeCache-Token": None})[0] == 403 and req("/api/chat", HI, headers={"Sec-Fetch-Site": "cross-site"})[0] == 403,
      "chat: not without the page token, not from another site")
st, body = req("/api/chat", dict(HI, messages=[{"role": "user", "content": "x" * 200000}]))
check(st != 413, f"chat: up to 8 MB for messages with attached files ({st})")
# chats are files in the cache folder (browser storage is per port; the app picks a free port each run)
CH = {"id": "abc-123", "title": "Hello chat", "model": small["path"], "system": "", "messages": HI["messages"][1:] + [{"role": "assistant", "content": "Hello", "meta": {"predicted_n": 2}}],
      "tokens": 7, "files": [], "junk": "dropped"}
st, body = req("/api/chats/save", CH)
check(st == 200 and os.path.isfile(os.path.join(ui.CHAT_DIR, "abc-123.json")), f"chats: saved as a file ({st} {body[:80]})")
lst = json.loads(req("/api/chats")[1])["chats"]
check([(c["id"], c["title"], c["n"], c["tokens"]) for c in lst] == [("abc-123", "Hello chat", 2, 7)], f"chats: listed ({lst})")
got = json.loads(req("/api/chats?id=abc-123")[1])
check(got["messages"][1]["meta"] == {"predicted_n": 2} and "junk" not in got, "chats: read back with the answer's numbers; unknown keys dropped")
check(req("/api/chats/save", dict(CH, id="../x"))[0] == 400 and req("/api/chats?id=..%2Fui")[0] == 400 and req("/api/chats?id=nothere")[0] == 404, "chats: ids are checked")
check(req("/api/chats/save", dict(CH, messages=[{"role": "user"}]))[0] == 400, "chats: messages are checked")
check(req("/api/chats/save", dict(CH, messages=[{"role": "user", "content": "y" * 100000}]))[0] == 200, "chats: bigger than 64 KiB is fine")
check(req("/api/chats", headers={"X-MoeCache-Token": None})[0] == 403 and req("/api/chats/save", CH, headers={"X-MoeCache-Token": None})[0] == 403, "chats: not without the page token")
lg = [l for _, l in json.loads(req("/api/log?since=0")[1])["lines"]]
check("fake: argv has the key: False, LLAMA_API_KEY set: True" in lg, "4: the API key reaches the launcher in LLAMA_API_KEY, not in its argv")
check(not any("sekrit" in l for l in lg), "4: and is not in the log")
os.makedirs(os.path.join(ui.mc.CACHE_DIR, "slots", "agent"), exist_ok=True)
# agent sessions: the launcher's slot files (header 'ggsq', version, token count, tokens), restored ones, agent guess, live slots, Trash
import struct
sd = ui.slot_dir(); os.makedirs(sd, exist_ok=True)
def slot_file(name, n, first, pad=1000):
    with open(os.path.join(sd, name), "wb") as f:
        f.write(struct.pack("<III", 0x67677371, 4, n) + struct.pack(f"<{len(first)}i", *first) + b"\0" * pad)
tag = ui.session_tag({"model": small["path"], "ctx": 8192, "agent": True})
check(tag == "Small-MoE-Q4_K_M-c8192" and ui.session_tag({"model": "/m/Big-00001-of-00002.gguf", "agent": True, "mode": "fast"}) == "Big-c65536-q8_0-q8_0",
      f"the launcher's tag for saved sessions (alias, context, q8_0 KV with --fast) ({tag})")
slot_file(tag + "-s0.bin", 4096, [11, 22, 33]); slot_file(tag + "-s1.bin", 2500, [5, 6, 7])
slot_file("Other-Model-c65536-s0.bin", 9000, [1, 2]); slot_file(tag + "-s2.new.bin", 10, [1])
open(os.path.join(sd, "notes.bin"), "wb").write(b"x"); open(os.path.join(sd, "Weird-c8192-s3.bin"), "wb").write(b"not a slot file")
warm = json.loads(req("/api/status")[1])["warm"]
check([e["line"] for e in warm["events"]] == ["restored a saved agent conversation (4096 tokens) into slot 0 in 1.0 s", "saved slot 0 (4096 tokens, 120 MB) in 0.5 s"],
      f"warm-session events from the launcher's log ({warm['events']})")
# servers started elsewhere (status only): one port, and the scan that marks the one this page started
import socket
_s = socket.socket(); _s.bind(("127.0.0.1", 0)); closed = _s.getsockname()[1]; _s.close()
pk = json.loads(req(f"/api/peek?port={port}")[1])
check(pk.get("up") and pk.get("model") == "fake-moe" and pk.get("ctx") == 65536 and pk.get("slots") == 1 and pk.get("busy") == 0 and pk.get("health") == "ready",
      f"peek: model, context and slots of a llama-server on this machine ({pk})")
check(json.loads(req(f"/api/peek?port={closed}")[1]) == {"port": closed, "up": False}, "peek: a closed port is just 'not up'")
check(req("/api/peek?port=99999")[0] == 400 and req("/api/peek")[0] == 400, "peek: the port is range-checked")
ui.PEEK_PORTS = [port, closed]
sv = json.loads(req("/api/servers")[1])["servers"]
check(len(sv) == 1 and sv[0]["port"] == port and sv[0]["started_here"], f"servers: the one this page started is marked, closed ports left out ({sv})")
st, body = req("/api/sessions")
ss = json.loads(body)
byname = {f["name"]: f for f in ss["files"]}
check(st == 200 and set(byname) == {tag + "-s0.bin", tag + "-s1.bin", "Other-Model-c65536-s0.bin", "Weird-c8192-s3.bin"}, f"saved sessions listed, half-written and foreign names not ({sorted(byname)})")
f0 = byname[tag + "-s0.bin"]
check(f0["tokens"] == 4096 and f0["slot"] == 0 and f0["ctx"] == 8192 and f0["model"] == "Small-MoE-Q4_K_M" and byname["Weird-c8192-s3.bin"]["tokens"] is None,
      "token count from the file header; model, context and slot from the name")
check(f0["restored"] and not byname[tag + "-s1.bin"]["restored"] and not byname["Other-Model-c65536-s0.bin"]["this_server"], "restored at start: slot 0 of this server's tag only")
check(f0["agent"] == "OpenCode" and byname[tag + "-s1.bin"]["agent"] is None and byname["Other-Model-c65536-s0.bin"]["agent"] is None,
      "agent guessed from the conversation's start (detokenized by the running server), only for this model")
check(ss["live"] == [{"id": 0, "tokens": 4100, "processing": False, "n_ctx": 8192}], f"live slots through /slots with the key ({ss['live']})")
# Trash: the platform's own (Windows: Recycle Bin); on Linux and others a freedesktop Trash under a temporary XDG_DATA_HOME, never the real one
if os.name != "nt":
    os.environ["XDG_DATA_HOME"] = os.path.join(TMP, "data"); ui.shutil_which = lambda prog: None
st, body = req("/api/sessions/trash", {"name": tag + "-s1.bin"})
j = json.loads(body)
check(st == 200 and not os.path.exists(os.path.join(sd, tag + "-s1.bin")), f"Move to Trash: the file leaves the folder ({j.get('where') or j})")
if os.name != "nt":
    tf = os.path.join(TMP, "data", "Trash")
    info = open(os.path.join(tf, "info", tag + "-s1.bin.trashinfo")).read()
    check(os.path.isfile(os.path.join(tf, "files", tag + "-s1.bin")) and "Path=" in info and "DeletionDate=" in info, "in the Trash with its .trashinfo (restorable)")
    slot_file(tag + "-s1.bin", 10, [1]); req("/api/sessions/trash", {"name": tag + "-s1.bin"})
    check(os.path.isfile(os.path.join(tf, "files", tag + "-s1.bin.1")), "a second file of the same name gets its own Trash name")
check(req("/api/sessions/trash", {"name": "../ui.json"})[0] == 400 and req("/api/sessions/trash", {"name": "notes.bin"})[0] == 400, "only listed session files can be moved")
st, body = req("/api/chats/trash", {"id": "abc-123"})
check(st == 200 and not os.path.exists(os.path.join(ui.CHAT_DIR, "abc-123.json")) and json.loads(req("/api/chats")[1])["chats"] == [],
      f"chats: Trash moves the file to the Trash (after the test Trash is set up; {body[:100]})")
check(req("/api/sessions/trash", {"name": tag + "-s0.bin"}, headers={"X-MoeCache-Token": None})[0] == 403, "not without the page token")
check(ui.server_args({"model": "m.gguf", "agent": True})[3:5] == ["--ctx", "65536"] and ui.server_args({"model": "m.gguf"})[3:5] == ["--ctx", "8192"],
      "Agent mode without a chosen context: 65536, as the launcher's --agent (the page passed 8192 before)")
if sys.platform.startswith("linux"):
    # the launcher runs short helpers before llama-server (planner, `llama-server --help`): the stats go to the child that serves (--port)
    import subprocess as sp
    par = sp.Popen([sys.executable, "-c", "import subprocess as s, sys, time; a = s.Popen([sys.executable, '-c', 'import time; time.sleep(30)', '--help']); "
                    "b = s.Popen([sys.executable, '-c', 'import time; time.sleep(30)', '--port', '1']); time.sleep(30)"])
    found = None
    for _ in range(50):
        found = ui.child_pid(par.pid)
        if found:
            break
        time.sleep(0.1)
    kids = open(f"/proc/{par.pid}/task/{par.pid}/children").read().split()
    check(found is not None and b"--port" in open(f"/proc/{found}/cmdline", "rb").read() and len(kids) == 2, f"child_pid picks the serving child, not a helper ({found}, children {kids})")
    check(ui.alive(found) and not ui.alive(999999999), "alive() tells a running process from a gone one")
    for k in kids:
        os.kill(int(k), 15)
    par.kill(); par.wait()
check(ui.server_args({"model": "m.gguf", "agent": True, "no_warm": True})[:-2][-1] == "--no-warm" and "--no-warm" not in ui.server_args({"model": "m.gguf", "no_warm": True}),
      "Don't restore: --no-warm (agent mode only)")

# connect buttons: preview, then write (into the temporary XDG_CONFIG_HOME)
st, body = req("/api/connect", {"agent": "opencode", "write": False})
j = json.loads(body)
check(st == 200 and j["rc"] == 0 and "fake-moe" in j["output"] and not os.path.exists(os.path.join(TMP, "config", "opencode", "opencode.json")),
      "Preview OpenCode: shows the block, writes nothing")
st, body = req("/api/connect", {"agent": "opencode", "write": True})
j = json.loads(body)
cfg = os.path.join(TMP, "config", "opencode", "opencode.json")
check(st == 200 and j["rc"] == 0 and j["file"] == cfg and os.path.isfile(cfg), f"Connect OpenCode writes {cfg} ({j['output'][-120:]})")
check('"moe-cache/fake-moe"' in (j["written"] or "") and f"127.0.0.1:{port}/v1" in j["written"], "and shows the provider block now in the file")
check("sekrit" not in body and "sekrit-123" in open(cfg).read(), "the key is written to the file but never shown on the page")
st, body = req("/api/connect", {"agent": "hermes", "write": False})
check(st == 200 and "hermes config set" in json.loads(body)["output"], "Preview Hermes: the 'hermes config set' commands")
st, _ = req("/api/connect", {"agent": "bash"})
check(st == 400, "only opencode and hermes")

# quick benchmark: warm-up, 2 x 256 decode, one ~3.5k-token prefill (pc_speed.ps1's method), saved to the CSV in the profile dir
# an older bench.csv (before the start/load/flush columns) is upgraded, its rows kept
old_csv = os.path.join(ui.mc.CACHE_DIR, "bench.csv")
os.makedirs(os.path.dirname(old_csv), exist_ok=True)
with open(old_csv, "w", newline="", encoding="utf-8") as f:
    f.write("time,model,decode_mean,prefill\n2026-10-20 10:00,Old.gguf,9.50,100.0\n")
st, body = req("/api/bench", {})
check(st == 200 and json.loads(body)["state"] == "running", "POST /api/bench starts the benchmark")
deadline = time.time() + 30
while time.time() < deadline:
    b = json.loads(req("/api/status")[1])["bench"]
    if b["state"] != "running":
        break
    time.sleep(0.3)
r = b.get("result", {})
check(b["state"] == "done" and r.get("decode_tokens") == "256/256" and float(r["decode_mean"]) > 0, f"two decode runs of 256 tokens even when the model would stop early (ignore_eos) ({b.get('error') or r.get('decode_1')} / {r.get('decode_2')})")
check(3000 <= int(r.get("prefill_tokens") or 0) <= 4500 and r.get("prefill") == "300.0", f"one prefill of about 3.5k tokens ({r.get('prefill_tokens')})")
check(r.get("mode") == "moe-cache exact + MTP" and r.get("model") == "Small-MoE-Q4_K_M.gguf" and "sekrit" not in r.get("command", ""), "the row names model and mode, no key")
rows = json.loads(req("/api/bench")[1])["rows"]
check(len(rows) == 2 and rows[0]["model"] == "Old.gguf" and rows[0]["start"] == "warm" and os.path.isfile(old_csv + ".bak")
      and open(old_csv, encoding="utf-8").readline().strip() == ",".join(ui.BENCH_FIELDS), "an older bench.csv gets the new columns, its rows kept, a .bak copy")
check(rows[-1]["decode_mean"] == r["decode_mean"] and rows[-1]["start"] == "warm" and rows[-1]["load_s"] == "" and os.path.isfile(os.path.join(ui.mc.CACHE_DIR, "bench.csv")),
      "appended to bench.csv in the profile dir and read back")

lines = [l for _, l in json.loads(req("/api/log?since=0")[1])["lines"]]
check(any("eval time" in l for l in lines) and not any(l.startswith("moe-cache-stats") for l in lines), "the log has the server lines, not the stats lines")
# cold start (fair): stop, flush the model from the file cache (mocked here), start again, measure; recorded as "cold"
flushed = []
real_flush, ui.flush_model_cache = ui.flush_model_cache, (lambda model, other: flushed.append((model, list(other))) or (True, "fake flush"))
st, _ = req("/api/bench", {"cold": True, "other": [os.path.join(TMP, "nowhere")]})
check(st == 400, "cold start: a folder that does not exist is refused")
runner = srv.RequestHandlerClass.runner
pid_before = runner.p.pid
st, body = req("/api/bench", {"cold": True, "other": [models]})
check(st == 200, f"cold-start benchmark starts ({body[:100]})")
deadline = time.time() + 60
while time.time() < deadline:
    b = json.loads(req("/api/status")[1])["bench"]
    if b["state"] not in ("running",):
        break
    time.sleep(0.3)
r = b.get("result", {})
check(b["state"] == "done" and r.get("start") == "cold" and r.get("flush") == "fake flush" and r.get("load_s", "") != "", f"cold run recorded: start cold, flush line, load time ({b.get('error') or r.get('load_s')})")
check(flushed == [(small["path"], [models])] and runner.p.pid != pid_before and runner.state == "ready", "the server was stopped, the model flushed, the server started again")
check(json.loads(req("/api/bench")[1])["rows"][-1]["start"] == "cold", "cold / warm in the CSV")
# compare (I42): measure, start again with Run as flipped (moe-cache -> stock), measure, then back to the user's settings
flushed.clear()
st, body = req("/api/bench", {"compare": True})
check(st == 200 and json.loads(body).get("compare") is True, f"compare benchmark starts ({body[:100]})")
deadline, steps = time.time() + 90, set()
while time.time() < deadline:
    b = json.loads(req("/api/status")[1])["bench"]
    steps.add(b.get("step", ""))
    if b["state"] != "running":
        break
    time.sleep(0.2)
res = b.get("results") or []
check(b["state"] == "done" and len(res) == 2 and res[0]["mode"].startswith("moe-cache") and res[1]["mode"].startswith("stock llama-server (--stock"),
      f"compare: one moe-cache run, then one stock run ({b.get('error')}; {[r['mode'] for r in res]})")
check(float(res[1]["decode_mean"]) < float(res[0]["decode_mean"]) and flushed == [] and all(r["start"] == "warm" for r in res), "compare without Cold start: no flush, both warm")
check(any(x.startswith("run 2 of 2: ") for x in steps), f"compare: the steps say which run ({sorted(steps)[:4]})")
check(runner.state == "ready" and runner.settings.get("engine") in (None, "", "auto") and (runner.mode or "").startswith("moe-cache"),
      f"compare: back to the user's settings at the end ({runner.settings.get('engine')}, {runner.mode})")
cmp_rows = json.loads(req("/api/bench")[1])["compare"]
check(any(r["model"] == "Small-MoE-Q4_K_M.gguf" and r["decode_x"] and r["decode_x"] > 1 for r in cmp_rows), "compare: the pair shows in the Stock vs moe-cache table")
here = ui.measured_here(small["path"])
check(len(here) >= 1 and "stock " in here[0] and "x)" in here[0], f"plan: this computer's stock vs moe-cache pair next to the verdict ({here})")
check(ui.measured_here(os.path.join(TMP, "Other.gguf")) == [], "plan: no lines for a model never benchmarked here")
ui.flush_model_cache = real_flush
cs = ui.cold_start_support()
check(cs["available"] == (sys.platform.startswith("linux") or os.name == "nt"), f"cold start offered on Linux and Windows ({cs})")
if sys.platform.startswith("linux"):
    import mmap
    fp = os.path.join(TMP, "flush-me.gguf")
    with open(fp, "wb") as f:
        f.write(os.urandom(8 << 20)); f.flush(); os.fsync(f.fileno())
    ok, line = ui.flush_model_cache(fp, [])
    check((ok and "after 0 %" in line) or (not ok and "kept the model in memory" in line), f"Linux flush: dropped, or honestly not on tmpfs ({line})")
    with open(fp, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        ok, line = ui.flush_model_cache(fp, [])
        m.close()
    check(not ok and "another program uses this model file" in line, "Linux flush: refused while a process maps the model (it would slow that program down)")
if os.name == "nt":
    # the real tools\win_flush_cache.ps1: a small "other" folder cannot cover the RAM, so it reads it all and says the flush is partial
    od = os.path.join(TMP, "other-big"); os.makedirs(od, exist_ok=True)
    open(os.path.join(od, "filler.bin"), "wb").write(os.urandom(1 << 20))
    tp = os.path.join(TMP, "target.gguf"); open(tp, "wb").write(b"GGUF" + b"\0" * 1000)
    ok, line = ui.flush_model_cache(tp, [od])
    check(not ok and "flushed file cache" in line, f"Windows flush: runs win_flush_cache.ps1 and reports a partial flush honestly ({line[:160]})")

st, _ = req("/api/stop", {})
deadline = time.time() + 30
while time.time() < deadline and status["state"] not in ("stopped", "exited"):
    status = json.loads(req("/api/status")[1]); time.sleep(0.3)
check(status["state"] == "stopped", f"POST /api/stop stops it ({status['state']}, exit code {status['rc']})")
check(req("/api/key")[0] == 400, "Copy key: nothing once the server stopped")
# option C: agent mode without a typed key uses the launcher's stable key (KEY_CMD), for /slots, the benchmark and connect
check(ui.launcher_key() == AUTO_KEY, "launcher_key reads the key the launcher prints")
saved_cmd = ui.KEY_CMD
for bad in ([sys.executable, "-c", "import sys; sys.exit(3)"], [sys.executable, "-c", "print('two words')"], [os.path.join(TMP, "missing")]):
    ui.KEY_CMD = bad
    check(ui.launcher_key() is None, f"no key when the command fails or prints something else ({os.path.basename(bad[-1])[:30]})")
ui.KEY_CMD = saved_cmd
with socket.socket() as so:
    so.bind(("127.0.0.1", 0)); port2 = so.getsockname()[1]
st, body = req("/api/start", {"model": small["path"], "port": port2, "ram": "1", "agent": True})
deadline = time.time() + 30
while time.time() < deadline:
    status = json.loads(req("/api/status")[1])
    if status["state"] == "ready" and status["log_seq"] > 5:
        break
    time.sleep(0.3)
check(status.get("api_key", {}).get("source", "").startswith("the launcher's agent key") and status["api_key"]["ends"] == "EFGH", f"agent mode, no typed key: the launcher's key ({status.get('api_key')})")
lg = [l for _, l in json.loads(req("/api/log?since=0")[1])["lines"]]
check("fake: agent snippet apiKey KEY" in lg and not any(AUTO_KEY in l for l in lg), "the key is masked in the log")
check(AUTO_KEY not in json.dumps(status) and json.loads(req("/api/key")[1])["key"] == AUTO_KEY, "not in the status; Copy key gets it")
st, body = req("/api/connect", {"agent": "opencode", "write": True})
check(st == 200 and AUTO_KEY in open(cfg).read() and AUTO_KEY not in body, "Connect OpenCode writes the launcher's key into the config, never shows it")
req("/api/stop", {})
deadline = time.time() + 30
while time.time() < deadline and json.loads(req("/api/status")[1])["state"] not in ("stopped", "exited"):
    time.sleep(0.3)
# "Which model?" (bin/moe_cache_pick.py): /api/pick and /api/pick/use with a fake machine (32 GB RAM, 16 GB GPU, the Bolt file here)
bolt = os.path.join(TMP, "pickmodels", "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf")
os.makedirs(os.path.dirname(bolt)); open(bolt, "wb").write(b"GGUF" + b"\0" * 60)
fake_spec = {"os": "test", "ram_total": 32 * 2**30, "ram_cap": 28 * 2**30, "gpu": "GPU", "vram": 16 * 2**30, "disk_path": TMP, "disk_free": 500e9,
             "disk_kind": "SSD", "avx2": True, "avx512": False, "llama_build": 11966, "local": {"bolt": bolt}}
ui.picker.machine_specs = lambda folders=None, ls=None: dict(fake_spec)
pbase = base
def preq(path, body=None):
    r = urllib.request.Request(pbase + path, json.dumps(body).encode() if body is not None else None,
                               {"X-MoeCache-Token": TOKEN, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=30) as x:
            return x.status, json.loads(x.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())
st, j = preq("/api/pick?use=agents&fresh=1")
check(st == 200 and len(j["recommendations"]) == 1 and j["recommendations"][0]["id"] == "bolt" and j["recommendations"][0]["ctx"] == 131072
      and any("RAM" in l for l in j["lines"]), f"/api/pick: the machine and one recommendation (Bolt at 128k for agents) ({st})")
st, j = preq("/api/pick")
check(st == 200 and [r["use"] for r in j["recommendations"]] == ["agents", "chat", "docs"], "/api/pick without a use: all three")
check(preq("/api/pick/use", {"use": "gaming"})[0] == 400, "/api/pick/use: unknown use refused")
st, j = preq("/api/pick/use", {"use": "agents"})
check(st == 200 and j["last"]["model"] == bolt and j["last"]["ctx"] == 131072 and j["last"]["agent"] is True and j["pick"]["id"] == "bolt",
      f"/api/pick/use: Start now uses the picked file, its context and agent mode ({st})")
saved = json.load(open(ui.SETTINGS_FILE, encoding="utf-8"))
check(saved["last"]["model"] == bolt and saved["pick"]["use"] == "agents", "/api/pick/use: written to ui.json (the launcher's `server` without a model reads it)")
st, j = preq("/api/system")
check(st == 200 and (j.get("pick") or {}).get("id") == "bolt" and j["last"]["model"] == bolt, "/api/system: the page sees the pick at load")
check("pickuse" in open(ui.PAGE, encoding="utf-8").read(), "the page has the Which model? panel")

srv.shutdown()

# Windows without nvidia-smi: VRAM through the launcher's helpers (mocked: they need Windows and a llama-server)
mc = ui.mc
saved = mc.pick_gpu, mc.list_devices, mc.win_dedicated_vram
mc.list_devices = lambda ls: "listing"
mc.pick_gpu = lambda listing: ("Vulkan1", "AMD Radeon RX 7900 XT", 18000)
mc.win_dedicated_vram = lambda: {"AMD Radeon RX 7900 XT": 20 * 2**30, "AMD Radeon(TM) Graphics": 512 * 2**20}
g = ui.windows_gpu_once("llama-server")
check(g == {"name": "AMD Radeon RX 7900 XT", "free": 18000 * 2**20, "total": 20.0 * 2**30, "integrated": False, "device": "Vulkan1",
            "adapter": "AMD Radeon RX 7900 XT"}, f"discrete AMD GPU: free and total ({g})")
mc.pick_gpu = lambda listing: ("Vulkan0", "AMD Radeon(TM) Graphics", 16000)
g = ui.windows_gpu_once("llama-server")
check(g["integrated"] and g["total"] is None and "integrated" in g["name"], "integrated GPU: marked, no dedicated total")
mc.pick_gpu, mc.list_devices, mc.win_dedicated_vram = saved
ui.vram_now, ui._GPU["gpu"] = (lambda: None), {"name": "AMD Radeon RX 7900 XT", "free": 15 * 2**30, "total": 20.0 * 2**30, "integrated": False}
si = ui.system_info()
check(si["gpu"] == "AMD Radeon RX 7900 XT" and si["vram_free"] == 15 * 2**30 and si["vram_total"] == 20 * 2**30, "the status uses it when nvidia-smi and sysfs have nothing")

# warnings: llama-server's "no usable GPU found" only repeats the plan; real warnings still show
check(not ui.is_warning("warning: no usable GPU found, --gpu-layers option will be ignored") and ui.is_warning("moe-cache: WARNING: the plugin did not load: test")
      and ui.is_warning("error: invalid argument: --foo"), "warnings: 'no usable GPU found' left out, real warnings kept")

# benchmark comparison: stock and moe-cache runs of one model side by side
R0 = {"model": "Q.gguf", "ram_gb": "12", "ctx": "8192", "start": "warm", "gpu": "G", "os": "Linux", "hit_rate": ""}
rows = [dict(R0, mode="stock llama-server (--stock, to compare)", decode_mean="5.00", prefill="100.0", time="t1"),
        dict(R0, mode="moe-cache exact", decode_mean="20.00", prefill="150.0", hit_rate="0.950", time="t2"),
        dict(R0, mode="stock llama-server (--stock, to compare)", decode_mean="10.00", prefill="100.0", time="t3"),      # newer stock run wins
        dict(R0, mode="moe-cache --fast", decode_mean="22.00", prefill="", time="t4"),
        dict(R0, ram_gb="24", mode="moe-cache exact", decode_mean="30.00", prefill="200.0", time="t5"),                 # no stock at 24 GB: no pair
        dict(R0, mode="all experts on the GPU", decode_mean="50.00", prefill="300.0", time="t6")]
cmp_ = ui.bench_compare(rows)
check(len(cmp_) == 2 and cmp_[0]["mode"] == "moe-cache --fast" and cmp_[1]["mode"] == "moe-cache exact", f"compare: one row per moe-cache mode that has a stock run ({cmp_})")
check(cmp_[1]["decode_x"] == 2.0 and cmp_[1]["prefill_x"] == 1.5 and cmp_[1]["stock_time"] == "t3" and cmp_[0]["prefill_x"] is None,
      "compare: ratio against the newest stock run; a missing number gives no ratio")
mrows = [dict(R0, mode="stock llama-server (--stock, to compare)", decode_mean="10.00", prefill="100.0", time="t1"),
         dict(R0, mode="moe-cache exact + MTP", decode_mean="30.00", prefill="100.0", time="t2")]
check(ui.bench_compare(mrows) == [] and len(ui.bench_compare(mrows + [dict(mrows[0], mode="stock llama-server (--stock, to compare) + MTP")])) == 1,
      "compare: an MTP run pairs only with an MTP run")
check(ui.bench_compare([]) == [] and ui.bench_compare([{"mode": "stock", "decode_mean": "x"}]) == [], "compare: empty or bad rows give nothing")

# one stock-or-plugin rule: the page asks the launcher's decide(), so the two cannot disagree
calls = []
real_decide, ui.mc.decide = ui.mc.decide, (lambda path, ram=None, plugin_wanted=False, stock_wanted=False: calls.append((path, ram)) or (True, 1e9, 64e9))
p = ui.plan_model(small["path"], ram="1")
ui.mc.decide = real_decide
check(calls and all(c == (small["path"], "1") for c in calls) and p["kind"] == "stock", f"the plan follows moe_cache.decide() ({calls}, {p['kind']})")
st, sz, av = ui.mc.decide(small["path"], "64")
_cap = ui.mc.mem_cap()
check(st and av == (min(64e9, _cap) if _cap else 64e9) and sz == ui.size_of(small["path"]), f"decide(): a small model fits in 64 GB (or the memory cap): stock ({av})")
check(not ui.mc.decide(small["path"], "64", plugin_wanted=True)[0], "decide(): --fast/--hybrid/--always-plugin keep the plugin")
check(ui.mc.decide(small["path"], "0.1", stock_wanted=True)[0], "decide(): --stock runs stock even when the model does not fit")
p = ui.plan_model(small["path"], ram="0.1", engine="stock")
check(p["kind"] == "stock" and "as asked" in p["verdict"] and any("--stock" in r for r in p["reasons"]), f"Run as Stock, model bigger than RAM: said plainly ({p['verdict']})")
p = ui.plan_model(small["path"], ram="64", engine="plugin")
check(p["kind"] == "cache" and "--always-plugin" in p["verdict"], f"Run as moe-cache, model fits: the plugin, and why ({p['verdict']})")
base = {"model": "/m.gguf", "port": 8080}
check("--stock" in ui.server_args(dict(base, engine="stock", mode="fast")) and "--fast" not in ui.server_args(dict(base, engine="stock", mode="fast")),
      "server args: Stock wins over Fast (they cannot go together)")
check("--always-plugin" in ui.server_args(dict(base, engine="plugin")) and "--always-plugin" not in ui.server_args(dict(base, engine="plugin", mode="fast")),
      "server args: moe-cache adds --always-plugin (not needed with --fast)")
check("--mtp" in ui.server_args(dict(base, mtp=True)) and "--no-mtp" in ui.server_args(dict(base, mtp=False))
      and not any("mtp" in x for x in ui.server_args(dict(base, mtp=None))), "server args: MTP On / Off / Auto")
# MTP line in the plan: the launcher's own mtp_decide(); the head is faked here (the real reader is tested in planner_test)
import gguf_plan
real_ml, gguf_plan.mtp_layers = gguf_plan.mtp_layers, (lambda path: (1, 1) if "GLM" in path or "Small" in path else (0, 0))
ui._MTP_SERVER.update({"ls-new": True, "ls-old": False})
check(ui.mtp_reason(small["path"], None, True, "ls-new").startswith("MTP on: the model has an MTP head"), "MTP: Agent mode, head, new llama-server: on, and why")
check(ui.mtp_reason(small["path"], None, False, "ls-new").startswith("this model has an MTP head: --mtp"), "MTP: plain server with a head: the hint")
check("lacks --spec-type draft-mtp" in ui.mtp_reason(small["path"], None, True, "ls-old"), "MTP: old llama-server: off, and why")
check(ui.mtp_reason(small["path"], False, True, "ls-new") == "MTP off (--no-mtp)", "MTP: Off")
check(ui.mtp_reason(blob, True, True, "ls-new").startswith("MTP: Start would stop with: --mtp: the model has no MTP head"), "MTP: On for a model without a head: says Start would stop")
check(ui.mtp_reason(blob, None, False, "ls-new") is None, "MTP: no head, plain server: nothing to say")
check("fails" not in ui.mtp_reason("/m/GLM-5.3-Flash.gguf", None, True, "ls-new"), "MTP: no GLM caveat any more (GLM + MTP works, RESULTS 60)")
p = ui.plan_model(small["path"], ram="1", agent=True, mtp=None, ls="ls-new")
check(any(r.startswith("MTP on:") for r in p["reasons"]), "MTP: the plan's Why list has the line")
gguf_plan.mtp_layers = real_ml
check(not any(x in ui.server_args(dict(base, engine="auto")) for x in ("--stock", "--always-plugin")), "server args: Auto adds nothing")
check(not ui.mc.decide(small["path"], f"{(sz + ui.mc.STOCK_HEADROOM - 1) / 1e9}")[0], "decide(): model + 1.5 GB headroom must fit")

# Windows + AMD rules in the plan, in the launcher's words (the checkpoint rule is the launcher's own function)
fn = os.path.join(TMP, "Flash-Next-IQ2_XS.gguf")
gguf_header(fn, [("general.architecture", "qwen4exp"), ("qwen4exp.block_count", 48), ("qwen4exp.expert_count", 512)])
amd = {"device": "Vulkan0", "adapter": "AMD Radeon RX 9070 XT"}
rs = ui.windows_amd_rules(amd, fn, agent=True, windows=True)
check(len(rs) == 1 and "--ctx-checkpoints 0" in rs[0] and "agents:" in rs[0] and not any("system RAM" in r or "stay on the CPU" in r for r in rs),
      f"Flash-Next on an AMD GPU: checkpoints off; no 'all experts on the CPU' rule any more ({rs})")
check(ui.mc.WINDOWS == (os.name == "nt"), "the launcher's WINDOWS flag is put back")
check(ui.windows_amd_rules(amd, blob, windows=True) == [], "Qwen3.6: no AMD rule")
check(ui.windows_amd_rules({"device": "CUDA0", "adapter": "NVIDIA GeForce RTX 4060"}, fn, windows=True) == [], "NVIDIA: no AMD rule")
check(ui.windows_amd_rules(amd, fn, windows=False) == [], "Linux: no AMD rule")
ui.WINDOWS, ui._GPU["gpu"] = True, dict(amd, free=15 * 2**30, integrated=False)
ui.run_planner = lambda path, ctx, ram, vram, kv="f16": dict(PLAN, free_vram_GiB=float(vram or 0))
real_decide, ui.mc.decide = ui.mc.decide, (lambda path, ram=None, plugin_wanted=False, stock_wanted=False: (stock_wanted, 68e9, 24e9))
p = ui.plan_model(fn, ram="24")
ui.WINDOWS, ui.mc.decide = os.name == "nt", real_decide
check(p["numbers"]["free_vram_GiB"] == 15.0 and not any("system RAM" in r or "stay on the CPU" in r for r in p["reasons"]),
      f"plan on Windows + AMD, model bigger than RAM: the free VRAM found goes to the planner, no 'experts on the CPU' rule ({p['numbers']['gpu_expert_layers']} GPU layers)")

# --ram is GB; the planner's --ram-gib and the plugin's MOE_CACHE_RAM_GIB are GiB (B1: --ram 20 planned 21.5 GB under a 20 GB job limit)
check(mc.ram_gib("20") == "18.626" and abs(float(mc.ram_gib(12.5)) * 2**30 - 12.5e9) < 1e6, f"ram_gib: GB -> GiB ({mc.ram_gib('20')})")
seen = []
real_run, ui.subprocess.run = ui.subprocess.run, (lambda p, **k: seen.append(p) or type("R", (), {"stdout": "{}", "stderr": ""})())
try:
    real_planner(fn, 8192, "20", None)
finally:
    ui.subprocess.run = real_run
check(seen and seen[0][seen[0].index("--ram-gib") + 1] == "18.626", f"the page passes --ram 20 GB to the planner as 18.626 GiB ({seen})")

print("ui test:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
