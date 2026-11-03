#!/usr/bin/env python3
"""The desktop shell's contract with the control page, without a model and without a window (Linux and Windows; standard library only).

It starts the control page the way desktop/src-tauri/src/main.rs does (`python -u bin/moe_cache.py ui --ui-port 0 --no-browser`, MOE_CACHE_APP=1,
stdin and stdout piped), then does what the shell and the window do: reads the "control page at URL" line, opens the address like a webview
(navigation, token in ?t=), calls the API with the X-MoeCache-Token header, starts a fake model, streams a chat answer, stops the model, ends the page.
The model is a tiny fake launcher (it answers /health and a streamed /v1/chat/completions); the page file list is a fake .gguf.
   python3 desktop/tests/shell_contract_test.py"""
import http.client, json, os, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMP = tempfile.mkdtemp(prefix="moe-desktop-test-")
WINDOWS = os.name == "nt"
MARKER = "control page at "          # main.rs: const MARKER

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)

# ---- a fake launcher (what Runner starts instead of `moe_cache.py server`) ----
fake = os.path.join(TMP, "fake_launcher.py")
with open(fake, "w") as f:
    f.write(r'''
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake.pid"), "w").write(str(os.getpid()))
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = {"/health": {"status": "ok"}, "/slots": [{"id": 0, "is_processing": False, "n_prompt_tokens": 0, "n_ctx": 8192}],
                "/v1/models": {"data": [{"id": "fake-moe"}]}, "/props": {"default_generation_settings": {"n_ctx": 8192}}}.get(self.path)
        self.send_response(200 if body is not None else 404); self.end_headers(); self.wfile.write(json.dumps(body).encode())
    def do_POST(self):
        q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for piece in ("Hel", "lo"):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n\n"); self.wfile.flush(); time.sleep(0.05)
        self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode() + b"\n\ndata: [DONE]\n\n")
srv = HTTPServer(("127.0.0.1", port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"moe-cache: web UI and API will be at http://127.0.0.1:{port}", flush=True)
while True:
    time.sleep(1)
''')

# ---- a fake model file and the entry point the shell runs, with the model parts replaced ----
models = os.path.join(TMP, "models"); os.makedirs(models)
with open(os.path.join(models, "Tiny-MoE-Q4_K_M.gguf"), "wb") as f:
    f.write(b"GGUF" + b"\0" * 1020)
entry = os.path.join(TMP, "entry.py")
with open(entry, "w") as f:
    f.write(f'''
import os, sys
sys.path.insert(0, {os.path.join(HERE, "bin")!r})
import moe_cache_ui as ui
ui.default_folders = lambda: [{models!r}]
ui.ollama_roots = lambda: []
ui.LAUNCHER = [sys.executable, {fake!r}]
ui.KEY_CMD = [sys.executable, "-c", "print('k')"]
sys.exit(ui.ui(["--ui-port", "0", "--no-browser"]))
''')

env = dict(os.environ, PYTHONIOENCODING="utf-8", MOE_CACHE_APP="1", XDG_CACHE_HOME=os.path.join(TMP, "cache"), XDG_CONFIG_HOME=os.path.join(TMP, "config"),
           LOCALAPPDATA=os.path.join(TMP, "local"), APPDATA=os.path.join(TMP, "roaming"), USERPROFILE=TMP, HOME=TMP)
proc = subprocess.Popen([sys.executable, "-u", entry], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=HERE,
                        text=True, encoding="utf-8")
lines, got = [], threading.Event()
url_holder = []
def reader():
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        i = line.find(MARKER)
        if i >= 0 and not url_holder:
            url_holder.append(line[i + len(MARKER):].split()[0])
            got.set()
threading.Thread(target=reader, daemon=True).start()

def finish(code):
    try:
        proc.stdin.close()
    except OSError:
        pass
    if proc.poll() is None:
        proc.kill()
    print("\n".join(lines[-15:]) if fails else "", flush=True)
    print(f"{len(fails)} failed" if fails else "all ok", flush=True)
    sys.exit(code)

check(got.wait(60) and bool(url_holder), "the page prints 'control page at URL' within 60 s")
if not url_holder:
    finish(1)
url = url_holder[0]
# the same parse as main.rs parse_url: http://127.0.0.1:PORT/?t=TOKEN
check(url.startswith("http://127.0.0.1:") and "/?t=" in url, f"the address has the shape the shell parses ({url.split('?t=')[0]}?t=…)")
rest = url[len("http://127.0.0.1:"):]
port = int(rest.split("/")[0]); token = rest.split("t=", 1)[1]
check(port > 0 and len(token) >= 16, f"port {port}, a token of {len(token)} characters")

def call(method, path, headers=None, body=None, stream=False):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    h = {"Host": f"127.0.0.1:{port}"}
    h.update(headers or {})
    if body is not None:
        body = json.dumps(body); h["Content-Type"] = "application/json"
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    data = r.read().decode(errors="replace")
    c.close()
    return r.status, r.getheader, data

TOK = {"X-MoeCache-Token": token}
# what a webview sends when the shell moves it to the address: a navigation from the splash page (another origin) or from nowhere
for site in ("cross-site", "none", None):
    h = {"Sec-Fetch-Site": site, "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"} if site else {}
    st, hdr, page = call("GET", "/?t=" + token, h)
    check(st == 200 and f'<meta name="moe-token" content="{token}">' in page and hdr("Set-Cookie") is None,
          f"the window can open the address (Sec-Fetch-Site: {site}): the page carries the token, no cookie")
st, hdr, page = call("GET", "/", {"Sec-Fetch-Site": "none"})
check(st == 403 and token not in page, "the page without the token is refused (a reload in the window needs the address again, by design)")
st, _, _ = call("GET", "/api/status")
check(st == 403, "the API without the token header is refused")
st, _, body = call("GET", "/api/status", TOK)
check(st == 200 and json.loads(body).get("state") == "idle", f"the shell's status call works with the header ({body[:60]})")

st, _, body = call("GET", "/api/models", TOK)
mods = json.loads(body).get("models", []) if st == 200 else []
check(len(mods) == 1, f"the page lists the fake model ({[m.get('name') for m in mods]})")
free = 0
import socket
with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); free = s.getsockname()[1]
st, _, body = call("POST", "/api/start", TOK, {"model": mods[0]["path"], "port": free, "ctx": 8192, "mode": "exact"}) if mods else (0, None, "")
check(st == 200, f"Start answers 200 ({st} {body[:100]})")
state = ""
for _ in range(100):
    state = json.loads(call("GET", "/api/status", TOK)[2]).get("state")
    if state in ("ready", "error", "ended"):
        break
    time.sleep(0.2)
check(state == "ready", f"the fake model reaches 'ready' (state: {state})")

st, hdr, body = call("POST", "/api/chat", TOK, {"messages": [{"role": "user", "content": "hi"}]})
text = "".join(json.loads(l[6:])["choices"][0]["delta"].get("content", "") for l in body.splitlines()
               if l.startswith("data: {") and '"choices"' in l)
check(st == 200 and "text/event-stream" in (hdr("Content-Type") or "") and text == "Hello" and "data: [DONE]" in body,
      f"chat streams server-sent events through the page ({text!r}, status {st})")

# Quit as main.rs does it: POST /api/stop, poll the status until not starting / ready / stopping, then end the page process
st, _, _ = call("POST", "/api/stop", TOK, {})
check(st == 200, "Stop answers 200")
for _ in range(100):
    state = json.loads(call("GET", "/api/status", TOK)[2]).get("state")
    if state not in ("starting", "ready", "stopping"):
        break
    time.sleep(0.3)
check(state not in ("starting", "ready", "stopping"), f"the model is stopped (state: {state})")
pidf = os.path.join(TMP, "fake.pid")
fake_pid = int(open(pidf).read()) if os.path.exists(pidf) else 0
if fake_pid and not WINDOWS:
    time.sleep(0.5)
    try:
        os.kill(fake_pid, 0); alive = True
    except OSError:
        alive = False
    check(not alive, "the fake model process is gone after Stop (no orphan)")
proc.kill()
check(proc.wait(timeout=10) is not None, "the page process ends when the shell kills it")
finish(1 if fails else 0)
