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
import os
port = int(sys.argv[sys.argv.index("--port") + 1]); name = sys.argv[1].replace("\\", "/").split("/")[-1]
open(os.path.join(os.environ["FAKE_ARGV_DIR"], f"argv-{port}.json"), "w").write(json.dumps(sys.argv[1:]))
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
os.environ["FAKE_ARGV_DIR"] = TMP
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


# ---- MULTIMODEL-010: the split for two models at once ----
GIB = 2**30
M = lambda n, gb: {"name": n, "path": "/nope/" + n + ".gguf", "size": int(gb * 1e9)}           # unreadable path: counts as MoE
inp = {"cap": 28 * GIB, "vram_free": 8 * GIB, "cores": [0, 1, 2, 3, 4, 5, 6, 7]}
v = ui.group_plan_view([M("A", 22), M("B", 16)], inp)
total = sum(r["ram_gb"] for r in v["rows"]) * 1e9
check(v["ok"] and all(r["engine"] == "plugin" for r in v["rows"]) and total <= 28 * GIB * 1.001, f"split: two MoE models under one cap, shares add up to {total / GIB:.1f} GiB of 28 ({[(r['name'], r['ram_gb'], r['engine']) for r in v['rows']]})")
masks = [r["llama"][1] for r in v["rows"]]
cores = [int(m, 16) for m in masks]
check(len(masks) == 2 and cores[0] & cores[1] == 0 and all(r["llama"][2:4] == ["--cpu-strict", "1"] for r in v["rows"]), f"split: disjoint core slices ({masks})")
check(all("--vram" in r["launcher"] for r in v["rows"]), "split: each model gets a VRAM share")
small = ui.group_plan_view([M("A", 3), M("B", 2)], inp)
check(small["ok"] and all(r["engine"] == "stock" for r in small["rows"]), "split: two models that both fit run as stock llama-server")
tight = ui.group_plan_view([M("A", 40), M("B", 40)], dict(inp, cap=12 * GIB))
check(not tight["ok"] and tight["why"] and all(r["engine"] == "swap" for r in tight["rows"]), f"split: floors above the cap: no split, says why ({tight['why']})")
weighted = ui.group_plan_view([dict(M("A", 22), weight=3), M("B", 22)], inp)
check(weighted["rows"][0]["ram_gb"] > weighted["rows"][1]["ram_gb"], "split: the model used more gets more memory")
check(ui.group_plan_view([M("A", 22), M("B", 16)], dict(inp, cores=[0]))["rows"][0]["llama"] == [], "split: fewer cores than models: no pinning")

for r in [srv.RequestHandlerClass.runner] + list(srv.RequestHandlerClass.extra.values()):
    r.stop()
time.sleep(1)
for r in list(srv.RequestHandlerClass.extra.values()):
    r.stop()
srv.RequestHandlerClass.extra.clear()
ui.group_inputs = lambda: dict(inp)
G1, G2 = free_port(), free_port()
st, body = req("/api/servers/plan", {"models": [{"model": alpha, "port": G1}, {"model": beta, "port": G2, "weight": 2}]})
pl = json.loads(body)
check(st == 200 and pl["ok"] and pl["ports"] == [G1, G2] and len(pl["rows"]) == 2 and not json.loads(req("/api/servers")[1])["managed"], "plan: answers without starting anything")
check(req("/api/servers/plan", {"models": [{"model": alpha}]})[0] == 400 and req("/api/servers/plan", {"models": [{"model": alpha}, {"model": "/etc/passwd"}]})[0] == 400, "plan: one model, or an unlisted one: 400")
check(req("/api/servers/plan", {"models": [{"model": alpha, "port": 9000}, {"model": beta, "port": 9000}]})[0] == 400, "plan: the same port twice: 400")
ui.group_inputs = lambda: dict(inp, cap=1 * GIB)
big = [{"name": "A", "path": alpha, "size": 40 * GIB}, {"name": "B", "path": beta, "size": 40 * GIB}]
check(not ui.group_plan_view(big)["ok"], "group: refused when the floors do not fit (view)")
st, body = req("/api/servers/group", {"models": [{"model": alpha, "port": G1}, {"model": beta, "port": G2}]})
check(st == 400 and "do not fit together" in body and not json.loads(req("/api/servers")[1])["managed"], f"group: under a 1 GiB cap nothing starts and the answer says why ({body[:100]})")
ui.group_inputs = lambda: dict(inp)
st, body = req("/api/servers/group", {"models": [{"model": alpha, "port": G1}, {"model": beta, "port": G2}]})
check(st == 200 and json.loads(body)["ports"] == [G1, G2], f"group: starts both ({body[:120]})")
check(wait_ready(G1) and wait_ready(G2), "group: both reach ready, one after the other")
a1 = json.load(open(os.path.join(TMP, f"argv-{G1}.json"))); a2 = json.load(open(os.path.join(TMP, f"argv-{G2}.json")))
check("--peers" in a1 and "--peers" in a2 and "--ram" in a1 and "--ram" in a2, f"group: both servers got --ram and --peers ({a1[-8:]})")
i1, i2 = a1.index("--"), a2.index("--")
check(a1[i1 + 1] == "-C" and a2[i2 + 1] == "-C" and a1[i1 + 2] != a2[i2 + 2] and a1[-1] == "--metrics", f"group: each got its own core mask after -- ({a1[i1:]}, {a2[i2:]})")
check(req("/api/servers/group", {"models": [{"model": alpha, "port": free_port()}, {"model": beta, "port": free_port()}]})[0] == 400, "group: refused while servers of the page run")
st, body = chat(G2)
check(st == 200 and f"on {G2}" in body, "group: chat reaches the second server")

# a second server next to a running one counts the RAM the first holds
req("/api/servers/stop", {"port": G2}); time.sleep(2)
st, body = req("/api/servers/start", {"model": beta, "port": G2})
check(st == 200 and json.loads(body)["share_ram"] is True and wait_ready(G2), "next to a running server: started with --share-ram")
check("--share-ram" in json.load(open(os.path.join(TMP, f"argv-{G2}.json"))), "next to a running server: the launcher got --share-ram")
page2 = open(os.path.join(HERE, "ui", "index.html"), encoding="utf-8").read()
check("/api/servers/plan" in page2 and "/api/servers/group" in page2, "page: the split panel calls plan and group")

# the page
page = open(os.path.join(HERE, "ui", "index.html"), encoding="utf-8").read()
check("/api/servers/start" in page and "/api/servers/stop" in page and "S.chatPort ? { port: S.chatPort }" in page and 'id="chattarget"' in page, "page: start another, stop, chat target")

for r in [srv.RequestHandlerClass.runner] + list(srv.RequestHandlerClass.extra.values()):
    r.stop()
shutil.rmtree(TMP, ignore_errors=True)
print("\nFAILED: " + "; ".join(fails) if fails else "\nserver tests passed")
sys.exit(1 if fails else 0)
