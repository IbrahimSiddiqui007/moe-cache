#!/usr/bin/env python3
"""GUI-013 on the page server, without a model: a fake launcher plays llama-server. The page starts a main server and one more on another port, lists both
(and one started elsewhere), chats with each of them through /api/chat?port, refuses what it must (same port, too many, ports it may not reach, a server
that wants a key), and stops the extra one.
   python3 tests/servers_test.py"""
import json, os, shutil, socket, sys, tempfile, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-srv-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_ui as ui  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

models = os.path.join(TMP, "models"); os.makedirs(models)
for n in ("Alpha-Q4_K_M.gguf", "Beta-Q4_K_M.gguf"):
    open(os.path.join(models, n), "wb").write(b"GGUF" + b"\0" * 1000)
ui.default_folders = lambda: [models]
ui.ollama_roots = lambda: []
alpha, beta = os.path.join(models, "Alpha-Q4_K_M.gguf"), os.path.join(models, "Beta-Q4_K_M.gguf")

fake = os.path.join(TMP, "fake_launcher.py")
open(fake, "w").write(r'''
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1]); name = sys.argv[1].replace("\\", "/").split("/")[-1]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = {"/health": {"status": "ok"}, "/slots": [{"id": 0, "is_processing": False, "n_prompt_tokens": 1, "n_ctx": 8192}], "/v1/models": {"data": [{"id": name}]},
                "/props": {"default_generation_settings": {"n_ctx": 8192}, "modalities": {"vision": False}}}.get(self.path)
        self.send_response(200 if body is not None else 404); self.end_headers(); self.wfile.write(json.dumps(body).encode())
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": "from " + name + " on " + str(port)}}]}).encode() + b"\n\ndata: [DONE]\n\n")
srv = ThreadingHTTPServer(("127.0.0.1", port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"moe-cache: web UI and API will be at http://127.0.0.1:{port}", flush=True)
while True:
    time.sleep(1)
''')
ui.LAUNCHER = [sys.executable, fake]
ui.KEY_CMD = [sys.executable, "-c", "print('k')"]

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p
P1, P2, P3 = free_port(), free_port(), free_port()
ui.PEEK_PORTS = range(1, 65536)             # the test servers use random ports; the real page asks 8080-8089

TOKEN = "srv-token"
srv = ui.make_server(0, "llama-server", token=TOKEN)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"
def req(path, body=None):
    h = {"X-MoeCache-Token": TOKEN}
    if body is not None:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, h)
    try:
        with urllib.request.urlopen(r, timeout=30) as x:
            return x.status, x.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
def wait_ready(port, t=30):
    end = time.time() + t
    while time.time() < end:
        m = {x["port"]: x for x in json.loads(req("/api/servers")[1])["managed"]}
        if port in m and m[port]["state"] == "ready":
            return True
        time.sleep(0.3)
    return False

check(json.loads(req("/api/servers")[1])["managed"] == [], "no server: nothing managed")
st, body = req("/api/start", {"model": alpha, "port": P1})
check(st == 200 and wait_ready(P1), f"main server ready on {P1}")
st, body = req("/api/servers/start", {"model": beta, "port": P1})
check(st == 400 and "another server of this page" in body, "a second server on the main server's port: refused")
st, body = req("/api/servers/start", {"model": beta, "port": P2})
check(st == 200 and wait_ready(P2), f"a second server starts on {P2} and gets ready")
m = {x["port"]: x for x in json.loads(req("/api/servers")[1])["managed"]}
check(set(m) == {P1, P2} and m[P1]["main"] and not m[P2]["main"] and "Beta" in m[P2]["model"], f"managed list: main and extra ({[(p, x['main'], x['state']) for p, x in m.items()]})")
st, body = req("/api/servers/start", {"model": beta, "port": P3})
check(st == 200, "a third server (the second extra) is allowed")
st, body = req("/api/servers/start", {"model": beta, "port": free_port()})
check(st == 400 and "at most 2" in body, "a fourth server is refused")
check(req("/api/servers/start", {"model": "/etc/passwd", "port": free_port()})[0] == 400, "an unlisted model is refused (the same checks as Start)")
req("/api/servers/stop", {"port": P3}); time.sleep(1)

def chat(port=None):
    b = {"messages": [{"role": "user", "content": "hi"}]}
    if port is not None:
        b["port"] = port
    st, body = req("/api/chat", b)
    return st, body
st, body = chat()
check(st == 200 and f"on {P1}" in body, "chat without a port goes to the main server")
st, body = chat(P2)
check(st == 200 and f"from Beta-Q4_K_M.gguf on {P2}" in body, "chat with the extra server's port goes there")
st, body = chat(P1)
check(st == 200 and f"on {P1}" in body, "chat with the main server's own port goes to it")

# a server started elsewhere: listed as not started here, chat allowed when it wants no key; one that wants a key is refused
class Foreign(BaseHTTPRequestHandler):
    keyed = False
    def log_message(self, *a): pass
    def do_GET(self):
        code = 401 if self.keyed and self.path != "/health" else 200
        body = json.dumps({"status": "ok"} if self.path == "/health" else {"data": [{"id": "Foreign-model"}]}).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"])); self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"foreign says hi"}}]}\n\ndata: [DONE]\n\n')
fo = ThreadingHTTPServer(("127.0.0.1", 0), Foreign); threading.Thread(target=fo.serve_forever, daemon=True).start()
fp = fo.server_address[1]
ui.PEEK_PORTS = [P1, P2, P3, fp]
sv = json.loads(req("/api/servers")[1])["servers"]
by = {x["port"]: x for x in sv}
check(fp in by and not by[fp]["started_here"] and by[P2]["started_here"] and by[P1]["started_here"], "servers: the page's own are marked started here, the foreign one is not")
st, body = chat(fp)
check(st == 200 and "foreign says hi" in body, "chat with a server started elsewhere that wants no key")
Foreign.keyed = True
st, body = chat(fp)
check(st == 400 and "wants an API key" in body, "a foreign server that wants a key: refused, the page's key is not sent")
ui.PEEK_PORTS = range(8080, 8090)
st, body = chat(22)
check(st == 400 and "only servers on 127.0.0.1" in body, "a port outside 8080-8089 that is not the page's own: refused")
check(chat("x")[0] == 400, "a port that is not a number: refused")

# stop the extra one
check(req("/api/servers/stop", {"port": P1})[0] == 400, "the main server cannot be stopped through the extra-server route")
check(req("/api/servers/stop", {"port": 12345})[0] == 400, "stopping a server the page did not start: refused")
check(req("/api/servers/stop", {"port": P2})[0] == 200, "stop the extra server")
end = time.time() + 30
while time.time() < end and any(x["port"] == P2 for x in json.loads(req("/api/servers")[1])["managed"]):
    time.sleep(0.3)
check(not any(x["port"] == P2 for x in json.loads(req("/api/servers")[1])["managed"]), "the stopped server is gone from the list")
check(any(x["port"] == P1 for x in json.loads(req("/api/servers")[1])["managed"]), "the main server keeps running")
check(chat(P2)[0] == 400, "chat with the stopped server: refused")

# the page
page = open(os.path.join(HERE, "ui", "index.html"), encoding="utf-8").read()
check("/api/servers/start" in page and "/api/servers/stop" in page and "S.chatPort ? { port: S.chatPort }" in page and 'id="chattarget"' in page, "page: start another, stop, chat target")

for r in [srv.RequestHandlerClass.runner] + list(srv.RequestHandlerClass.extra.values()):
    r.stop()
shutil.rmtree(TMP, ignore_errors=True)
print("\nFAILED: " + "; ".join(fails) if fails else "\nserver tests passed")
sys.exit(1 if fails else 0)
