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
check(len(m) == 3 and "22.6" in m[1] and "RESULTS 44" in m[1], "measured speeds for a known model family and size (Qwen3.6-35B Q4_K_M)")
check(ui.measured(blob, 30e9) == [], "no measured speeds for another size of the same family")

# ---- the plan and its reasons (planner mocked) ----
PLAN = {"arch": "qwen3moe", "layers": 48, "experts": 128, "used": 8, "n_cpu_moe": 40, "cache_GiB": 6.0, "cpu_experts_GiB": 18.0,
        "free_vram_GiB": 7.5, "note": ""}
TUNE = {"n_cpu_moe": 38, "ub": 1024, "gpu_cache_mib": 0, "why": ["10 layers of experts stay on the GPU", "micro-batch 1024"]}
ui.run_planner = lambda path, ctx, ram, vram, kv="f16": dict(PLAN)
ui.run_tuner = lambda path, ctx, vram, kv_q8: dict(TUNE)

p = ui.plan_model(small["path"], ram="2")
check(p["kind"] == "stock" and "fits in RAM" in p["verdict"], f"model + 1.5 GB under the RAM budget: stock ({p['verdict']})")
check(any("RESULTS 44" in r for r in p["reasons"]), "stock: says why stock is faster")
p = ui.plan_model(small["path"], ram="1")
check(p["kind"] == "cache" and p["verdict"] == "does not fit: moe-cache exact", f"over the budget: moe-cache exact ({p['verdict']})")
check(any("10 of 48 layers keep their experts on the GPU" in r for r in p["reasons"]), "reasons: GPU layers from the tuner (as the launcher)")
check(any("holds 33 %" in r for r in p["reasons"]), "reasons: share of the CPU experts the cache holds")
check(any(r.startswith("tuner: ") for r in p["reasons"]), "reasons: the tuner's own lines")
p = ui.plan_model(small["path"], ram="2", mode="fast")
check(p["kind"] == "cache" and "needs the plugin" in p["verdict"], f"--fast keeps the plugin even when the model fits ({p['verdict']})")
TUNE["n_cpu_moe"] = 0
p = ui.plan_model(small["path"], ram="1")
check(p["kind"] == "gpu", f"every expert layer on the GPU ({p['verdict']})")
TUNE["n_cpu_moe"] = 38
ui.run_planner = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no expert tensors (blk.N.ffn_{gate,up,down,gate_up}_exps) found: this does not look like a MoE GGUF"))
p = ui.plan_model(small["path"], ram="1")
check(p["kind"] == "error" and any("not a MoE model" in r for r in p["reasons"]), "planner error: shown with the planner's own words")
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
        assert self.path == "/completion" and q["temperature"] == 0 and q["seed"] == 42 and q["cache_prompt"] is False
        t = {"predicted_n": q["n_predict"] if q.get("ignore_eos") else min(q["n_predict"], 149), "predicted_per_second": 20.0 + len(q["prompt"]) % 7, "prompt_n": len(q["prompt"]) // 4, "prompt_per_second": 300.0}
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({"content": "x", "timings": t}).encode())
srv = HTTPServer(("127.0.0.1", port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
import os
print("fake: argv has the key: %s, LLAMA_API_KEY set: %s" % (any("sekrit" in x for x in sys.argv), bool(os.environ.get("LLAMA_API_KEY"))), flush=True)
print("fake: agent snippet apiKey auto-key-ABCDEFGH", flush=True)
print("moe-cache: AMD GPU on Windows and a model bigger than RAM: all experts stay on the CPU (test)", flush=True)
print("moe-cache: experts of 38 layers stay on the CPU and are managed by the cache; profile /tmp/x.usage; plugin loaded by preload", flush=True)
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
check(hdr("X-Frame-Options") == "DENY" and "frame-ancestors 'none'" in (hdr("Content-Security-Policy") or "") and "frame-src http://127.0.0.1:*" in hdr("Content-Security-Policy"),
      "1: no framing (X-Frame-Options DENY, CSP frame-ancestors 'none'); the chat frame may load 127.0.0.1")
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
check(status.get("mode") == "moe-cache exact", f"mode from the launcher's log ({status.get('mode')})")
check(status.get("last", {}).get("decode_tps") == 25.0 and status["last"].get("prefill_tps") == 500.0, "decode and prefill tok/s from the timing lines")
c = status.get("cache", {})
check(abs((c.get("hit_rate") or 0) - 0.8) < 1e-9, f"recent hit rate from two snapshots: 80/100 ({c.get('hit_rate')})")
check(abs((c.get("hit_rate_total") or 0) - 0.85) < 1e-9 and c.get("budget") == 7e9, "total hit rate and the cache budget")
check((c.get("read_mbs") or 0) > 0, f"SSD reads MB/s from read_bytes ({c.get('read_mbs')})")
check(f"--port {port}" in status.get("command", ""), "the status shows the launcher command")
d = status.get("decisions", [])
check(len(d) == 2 and d[0].startswith("AMD GPU on Windows") and d[1].startswith("experts of 38 layers"),
      f"what the launcher decided, up to its URL line, without the prefix ({d})")
w = status.get("warnings", [])
check(len(w) == 1 and w[0]["count"] == 2 and w[0]["line"].startswith("moe-cache: warning: the expert cache shrank"),
      f"the shrink warning is lifted out of the log, once, counted, without llama-server's prefix ({w})")
check("sekrit" not in json.dumps(status), "the API key is not in the status")
check(status.get("api_key") == {"in_use": True, "source": "typed on this page", "ends": "-123"}, f"API key in use, masked, typed on the page ({status.get('api_key')})")
st, body = req("/api/key")
check(st == 200 and json.loads(body)["key"] == "sekrit-123", "Copy key: /api/key gives the key in use")
check(req("/api/key", headers={"X-MoeCache-Token": None})[0] == 403, "Copy key: not without the page token")
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
check(req("/api/sessions/trash", {"name": tag + "-s0.bin"}, headers={"X-MoeCache-Token": None})[0] == 403, "not without the page token")
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
check(r.get("mode") == "moe-cache exact" and r.get("model") == "Small-MoE-Q4_K_M.gguf" and "sekrit" not in r.get("command", ""), "the row names model and mode, no key")
rows = json.loads(req("/api/bench")[1])["rows"]
check(len(rows) == 1 and rows[0]["decode_mean"] == r["decode_mean"] and os.path.isfile(os.path.join(ui.mc.CACHE_DIR, "bench.csv")),
      "appended to bench.csv in the profile dir and read back")

lines = [l for _, l in json.loads(req("/api/log?since=0")[1])["lines"]]
check(any("eval time" in l for l in lines) and not any(l.startswith("moe-cache-stats") for l in lines), "the log has the server lines, not the stats lines")
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

# Windows + AMD rules in the plan, in the launcher's words (the checkpoint rule is the launcher's own function)
fn = os.path.join(TMP, "Flash-Next-IQ2_XS.gguf")
gguf_header(fn, [("general.architecture", "qwen4exp"), ("qwen4exp.block_count", 48), ("qwen4exp.expert_count", 512)])
amd = {"device": "Vulkan0", "adapter": "AMD Radeon RX 9070 XT"}
cpu_only, rs = ui.windows_amd_rules(amd, fn, 68e9, 30e9, agent=True, windows=True)
check(cpu_only and len(rs) == 2 and "all experts stay on the CPU" in rs[0] and "--ctx-checkpoints 0" in rs[1] and "agents:" in rs[1],
      f"Flash-Next bigger than RAM on an AMD GPU: experts on the CPU, checkpoints off ({rs})")
check(ui.mc.WINDOWS == (os.name == "nt"), "the launcher's WINDOWS flag is put back")
cpu_only, rs = ui.windows_amd_rules(amd, blob, 21.7e9, 30e9, windows=True)
check(not cpu_only and rs == [], "Qwen3.6 that fits: no AMD rule")
check(ui.windows_amd_rules({"device": "CUDA0", "adapter": "NVIDIA GeForce RTX 4060"}, fn, 68e9, 30e9, windows=True) == (False, []), "NVIDIA: no AMD rule")
check(ui.windows_amd_rules(amd, fn, 68e9, 30e9, windows=False) == (False, []), "Linux: no AMD rule")
ui.WINDOWS, ui._GPU["gpu"] = True, dict(amd, free=15 * 2**30, integrated=False)
ui.run_planner = lambda path, ctx, ram, vram, kv="f16": dict(PLAN, free_vram_GiB=float(vram or 0))
real_size, ui.size_of = ui.size_of, (lambda path: 68e9)
p = ui.plan_model(fn, ram="24")
ui.WINDOWS, ui.size_of = os.name == "nt", real_size
check(p["numbers"]["gpu_expert_layers"] == 0 and p["numbers"]["free_vram_GiB"] == 15.0 and any(r.startswith("Windows + AMD: all experts") for r in p["reasons"]),
      "plan on Windows + AMD: the free VRAM found goes to the planner, the rule keeps the experts on the CPU, and says why")

print("ui test:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
