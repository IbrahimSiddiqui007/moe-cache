#!/usr/bin/env python3
"""Tests for `moe_cache.py on-demand` (bin/moe_cache_on_demand.py) with a fake model server: no model, no llama-server; Linux and Windows.

  1. without the key: 401 and the model is not started; /health and /v1/models, /props answer without starting it
  2. the first request starts the server; streaming (SSE) passes through piece by piece; local origins get CORS, other origins do not
  3. idle stop after the set time; the next request starts it again
  4. a server that dies by itself is started again by the next request; an agent that hangs up mid-stream does not break the endpoint
  5. option checks (on-demand owns --port), the systemd unit text, connect's default URL follows the on-demand port

   python3 tests/on_demand_test.py
"""
import http.client, json, os, socket, subprocess, sys, tempfile, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="od-test-")
os.environ["XDG_CACHE_HOME"] = TMP           # before the import: the key, choice and log go to a scratch folder
os.environ["XDG_CONFIG_HOME"] = TMP
sys.path.insert(0, os.path.join(ROOT, "bin"))
import moe_cache as mc  # noqa: E402
import moe_cache_on_demand as od  # noqa: E402

FAKE = os.path.join(TMP, "fake_server.py")
open(FAKE, "w").write(r'''
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
key = os.environ.get("LLAMA_API_KEY")
time.sleep(0.5)                                   # "loading the model"
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        if self.headers.get("Origin") == "http://localhost:3000": self.send_header("Access-Control-Allow-Origin", "http://localhost:3000")
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": return self.send(200, {"status": "ok"})
        if self.headers.get("Authorization") != "Bearer " + key: return self.send(401, {"error": "key"})
        if self.path == "/v1/models": return self.send(200, {"object": "list", "data": [{"id": "fake-real", "object": "model"}]})
        if self.path == "/props": return self.send(200, {"default_generation_settings": {"n_ctx": 12345}})
        if self.path == "/crash": os._exit(3)
        self.send(404, {})
    def do_POST(self):
        if self.headers.get("Authorization") != "Bearer " + key: return self.send(401, {"error": "key"})
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not req.get("stream"):
            return self.send(200, {"choices": [{"message": {"content": "hello " + req["messages"][0]["content"]}}]})
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Transfer-Encoding", "chunked")
        if self.headers.get("Origin") == "http://localhost:3000": self.send_header("Access-Control-Allow-Origin", "http://localhost:3000")
        self.end_headers()
        try:
            for i in range(req.get("n", 5)):
                d = b"data: " + json.dumps({"i": i, "t": time.time()}).encode() + b"\n\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(d), d)); self.wfile.flush(); time.sleep(0.3)
            d = b"data: [DONE]\n\n"; self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(d), d))
        except OSError:
            open(os.path.join(os.environ["XDG_CACHE_HOME"], "hangup"), "w").write("1")
print("fake server on", port, flush=True)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')

fails = []


def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""), flush=True)
    if not ok:
        fails.append(name)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = free_port()
lines = []
choice = {"model": os.path.join(TMP, "Fake-Model-00001-of-00002.gguf"), "args": ["--ctx", "70000"], "env": {}}
b = od.Backend(choice, launcher=[sys.executable, FAKE], idle_min=2 / 60, start_timeout=30, log=lambda m: (lines.append(m), print("     log:", m)))
srv = od.serve(b, PORT)
threading.Thread(target=srv.serve_forever, daemon=True).start()
KEY = mc.stored_key()


def req(method, path, body=None, key=KEY, origin=None):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
    h = {"Content-Type": "application/json"}
    if key: h["Authorization"] = "Bearer " + key
    if origin: h["Origin"] = origin
    c.request(method, path, json.dumps(body) if body is not None else None, h)
    return c, c.getresponse()


def get(path, **kw):
    c, r = req("GET", path, **kw)
    out = r.status, r.read().decode(), r
    c.close()
    return out


chat = {"messages": [{"role": "user", "content": "there"}]}

# 1
c, r = req("POST", "/v1/chat/completions", chat, key=None)
check("no key: 401", r.status == 401, r.status); c.close()
c, r = req("POST", "/v1/chat/completions", chat, key="wrong")
check("wrong key: 401", r.status == 401, r.status); c.close()
check("no key: the model was not started", b.p is None and b.state == "idle")
st, body, _ = get("/health", key=None)
check("/health without key, idle", st == 200 and json.loads(body)["on_demand"] == "idle", body)
st, body, _ = get("/v1/models")
check("/v1/models while idle: the alias, without starting", st == 200 and json.loads(body)["data"][0]["id"] == "Fake-Model" and b.p is None, body)
st, body, _ = get("/props")
check("/props while idle: the context from --ctx", st == 200 and json.loads(body)["default_generation_settings"]["n_ctx"] == 70000, body)
st, body, _ = get("/v1/models", key=None)
check("/v1/models without key: 401 (connect finds the stored key this way)", st == 401, st)
c, r = req("OPTIONS", "/v1/chat/completions", origin="http://127.0.0.1:5173")
check("preflight from a local page: allowed", r.getheader("Access-Control-Allow-Origin") == "http://127.0.0.1:5173", r.getheaders()); r.read(); c.close()
c, r = req("OPTIONS", "/v1/chat/completions", origin="https://evil.example")
check("preflight from another site: no CORS header", r.getheader("Access-Control-Allow-Origin") is None, r.getheaders()); r.read(); c.close()

# 2
t0 = time.time()
st, body, r = get("/v1/models", key=KEY)       # still idle: answered locally
c, r = req("POST", "/v1/chat/completions", dict(chat, stream=True), origin="http://localhost:3000")
check("first request starts the server", r.status == 200 and b.state == "ready" and b.p is not None, (r.status, b.state))
check("streaming answer is chunked", (r.getheader("Transfer-Encoding") or "").lower() == "chunked", r.getheaders())
check("CORS header from the server passes through", r.getheader("Access-Control-Allow-Origin") == "http://localhost:3000", r.getheaders())
got = []
while True:
    piece = r.read1(65536)
    if not piece:
        break
    for ln in piece.split(b"\n"):
        if ln.startswith(b"data: {"):
            got.append((time.time(), json.loads(ln[6:])["t"]))
c.close()
check("5 SSE events arrive", len(got) == 5, got)
check("each event arrives when it is sent (not all at the end)", got and max(a - s for a, s in got) < 0.25, [round(a - s, 2) for a, s in got])
st, body, _ = get("/v1/models")
check("/v1/models while ready: the real server's answer", json.loads(body)["data"][0]["id"] == "fake-real", body)
st, body, _ = get("/props")
check("/props while ready: proxied", json.loads(body)["default_generation_settings"]["n_ctx"] == 12345, body)
c, r = req("POST", "/v1/chat/completions", chat, origin="https://evil.example")
body = r.read().decode(); c.close()
check("non-streaming answer", r.status == 200 and "hello there" in body, body)
check("another site's origin gets no CORS header", r.getheader("Access-Control-Allow-Origin") is None, r.getheaders())
check("one start log line", sum("starting" in x for x in lines) == 1, lines)

# 3: a stream that lasts longer than the idle time keeps it running
c, r = req("POST", "/v1/chat/completions", dict(chat, stream=True, n=10))
r.read(); c.close()
check("a 3 s stream with a 2 s idle time: not stopped meanwhile", not any("stopped" in x for x in lines), lines)
pid1 = b.p.pid
t = time.time()
while b.state != "idle" and time.time() - t < 20:
    time.sleep(0.2)
check("idle stop", b.state == "idle" and b.p.poll() is not None, b.state)
check("one stop log line", sum("stopped" in x for x in lines) == 1, lines)
st, body, _ = get("/v1/models")
check("after the stop: the model list from the last run, without starting", json.loads(body)["data"][0]["id"] == "fake-real" and b.state == "idle", body)
c, r = req("POST", "/v1/chat/completions", chat); body = r.read().decode(); c.close()
check("second request starts it again", r.status == 200 and b.state == "ready" and b.p.pid != pid1, (r.status, b.state))

# 4
c, r = req("GET", "/crash")
try:
    r.read()
except Exception:
    pass
c.close()
t = time.time()
while b.p.poll() is None and time.time() - t < 10:
    time.sleep(0.1)
time.sleep(0.5)
check("a server that dies is noticed", b.state == "idle" and any("ended by itself" in x for x in lines), lines[-1:])
c, r = req("POST", "/v1/chat/completions", chat); body = r.read().decode(); c.close()
check("the next request starts it again", r.status == 200 and "hello" in body, body)
c, r = req("POST", "/v1/chat/completions", dict(chat, stream=True, n=20))
r.read1(100)
c.sock.shutdown(socket.SHUT_RDWR); c.close()          # the agent hangs up mid-answer
t = time.time()
while not os.path.exists(os.path.join(TMP, "hangup")) and time.time() - t < 10:
    time.sleep(0.2)
check("a hang-up reaches the server (it stops generating)", os.path.exists(os.path.join(TMP, "hangup")))
c, r = req("POST", "/v1/chat/completions", chat); body = r.read().decode(); c.close()
check("the endpoint still answers after a hang-up", r.status == 200, r.status)
check("no request is counted as running", b.active == 0, b.active)
b.stop("test end")
srv.shutdown(); srv.server_close()

# 5
env = dict(os.environ)
open(choice["model"], "wb").close()
r = subprocess.run([sys.executable, os.path.join(ROOT, "bin", "moe_cache.py"), "on-demand", choice["model"], "--agent"], capture_output=True, text=True, env=env)
check("on-demand refuses launcher options it sets itself", r.returncode != 0 and "--agent" in r.stderr, r.stderr[-300:])
check("model alias drops the shard suffix", od.model_alias("/m/Q-00001-of-00003.gguf", []) == "Q")
check("model alias follows --alias after --", od.model_alias("/m/Q.gguf", ["--ram", "4", "--", "--alias", "mine"]) == "mine")
check("local origins", all(od.local_origin(o) for o in ("http://localhost", "http://127.0.0.1:8070", "https://[::1]:3000")) and
      not any(od.local_origin(o) for o in ("http://localhost.evil.com", "http://evil.com", "null", "")))
u = od.unit_text()
check("systemd unit runs the listener", "ExecStart=" in u and "on-demand" in u and "WantedBy=default.target" in u, u)
check("connect default: 8080 without on-demand", mc.default_agent_url() == "http://127.0.0.1:8080")
od.save_choice({"model": "/m/x.gguf", "args": [], "port": 8123})
check("connect default follows the on-demand port", mc.default_agent_url() == "http://127.0.0.1:8123")

print("FAILED: " + ", ".join(fails) if fails else "all on-demand tests passed")
sys.exit(1 if fails else 0)
