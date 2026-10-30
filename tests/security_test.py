#!/usr/bin/env python3
"""Regression tests for the six bugs an external audit confirmed in v0.3.1 (launcher, key, saved state). Fake servers, no model; Linux and Windows.

  1. first-launch key race        2. stored key never goes to a remote URL      3. concurrent chat saves
  4. on-demand custom key          5. CORS loopback check (127.attacker.example)  6. saved state of another model is not restored
   python3 tests/security_test.py"""
import json, os, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="security-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
os.environ["XDG_CONFIG_HOME"] = os.path.join(TMP, "config")
sys.path.insert(0, os.path.join(ROOT, "bin"))
import moe_cache as mc  # noqa: E402
import moe_cache_on_demand as od  # noqa: E402
import moe_cache_ui as ui  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# ---- 1. key race: many first launches at once, one key, the saved one ----
for rnd in range(5):
    if os.path.exists(mc.KEY_FILE):
        os.unlink(mc.KEY_FILE)
    got, bar = [], threading.Barrier(16)
    def first():
        bar.wait()
        got.append(mc.stored_key(create=True))
    ts = [threading.Thread(target=first) for _ in range(16)]
    [t.start() for t in ts]; [t.join() for t in ts]
    saved = open(mc.KEY_FILE).read().strip()
    if len(set(got)) != 1 or got[0] != saved:
        break
check(len(got) == 16 and len(set(got)) == 1 and got[0] == saved and len(saved) >= 20, "16 simultaneous first launches x 5 rounds: one key, and it is the saved one")
check(not [f for f in os.listdir(os.path.dirname(mc.KEY_FILE)) if f.endswith(".tmp")], "no temporary key file is left behind")
if os.name == "posix":
    check(oct(os.stat(mc.KEY_FILE).st_mode & 0o777) == "0o600", "the key file is private (0600)")
# a reader that meets a created-but-still-empty file waits for the writer instead of making its own key
os.unlink(mc.KEY_FILE); open(mc.KEY_FILE, "w").close()
threading.Timer(0.3, lambda: open(mc.KEY_FILE, "w").write("writer-key\n")).start()
check(mc.stored_key(create=True) == "writer-key", "a key being written (empty file for a moment) is waited for, not replaced")
os.unlink(mc.KEY_FILE); open(mc.KEY_FILE, "w").close()
k = mc.stored_key(create=True)
check(k and open(mc.KEY_FILE).read().strip() == k, "a stale empty key file is replaced by a complete one")

# ---- 2. the stored key goes only to this machine ----
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(401); self.send_header("Content-Length", "0"); self.end_headers()
    def log_message(self, *a): pass
srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
calls = []
real_http = mc.http
mc.http = lambda url, *a, **kw: (calls.append(url), (401, ""))[1]
for u in ("https://example.com/v1", "http://127.attacker.example:8080", "http://192.168.1.5:8080", "http://localhost.evil.test", "http://[::2]:8080", "http://0.0.0.0:8080"):
    calls.clear()
    check(mc.key_for(u) is None and not calls, f"remote URL {u}: no key, no probe")
for u in ("http://127.0.0.1:8080", "http://localhost:8080/v1", "http://[::1]:8080", "http://127.5.5.5:1"):
    check(mc.key_for(u) == mc.stored_key(), f"loopback URL {u} answering 401: the stored key")
mc.http = real_http
check(mc.key_for(f"http://127.0.0.1:{srv.server_address[1]}") == mc.stored_key(), "real local server answering 401: the stored key")
check(mc.key_for("http://example.com", "given") == "given", "a key the user gave is used as given")
srv.shutdown()

# ---- 3. concurrent chat saves ----
ui.CHAT_DIR = os.path.join(TMP, "chats")
errs, N = [], 40
def saver(i):
    try:
        for j in range(15):
            ui.save_chat({"id": "chat-1", "title": f"t{i}", "messages": [{"role": "user", "content": f"{i}-{j}" * 2000}]})
    except Exception as e:
        errs.append(repr(e))
ts = [threading.Thread(target=saver, args=(i,)) for i in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]
check(not errs, f"8 threads x 15 saves of one conversation: no exception ({errs[:1]})")
try:
    c = json.load(open(ui.chat_path("chat-1"), encoding="utf-8")); valid = c["messages"][0]["content"][:1].isdigit()
except Exception:
    valid = False
check(valid, "the saved conversation is valid JSON of one complete save")
check(os.listdir(ui.CHAT_DIR) == ["chat-1.json"], f"no temporary file is left in the chat folder ({os.listdir(ui.CHAT_DIR)})")

# ---- 4. on-demand custom key: listener and backend use the same key ----
choice = {"model": "/x/a.gguf", "args": ["--ctx", "32768", "--api-key", "my-custom-key"]}
b1 = od.Backend(choice, launcher=["x"], log=lambda m: None)
b2 = od.Backend({"model": "/x/b.gguf", "args": ["--api-key=other"]}, launcher=["x"], log=lambda m: None)
b3 = od.Backend({"model": "/x/c.gguf", "args": []}, launcher=["x"], log=lambda m: None)
check(b1.key == "my-custom-key" and "--api-key" not in b1.cmd(1234) and "my-custom-key" not in b1.cmd(1234) and "--ctx" in b1.cmd(1234),
      "custom key is the backend's key and is not put on its command line")
check(choice["args"][-1] == "my-custom-key", "the saved choice is not changed")
pool = od.Pool([b1, b2, b3])
check(pool.key == "my-custom-key" and b2.key == b3.key == "my-custom-key", "one key for the listener and every backend")
class Stub:
    def __init__(self, h): self.headers, self.server = h, type("S", (), {"pool": pool})()
check(od.Handler.authed(Stub({"Authorization": "Bearer my-custom-key"})), "listener accepts the custom key")
check(od.Handler.authed(Stub({"X-Api-Key": "my-custom-key"})), "listener accepts the custom key as x-api-key")
check(not od.Handler.authed(Stub({"Authorization": "Bearer " + mc.stored_key()})), "listener refuses the stored key when a custom one is configured")
p2 = od.Pool([od.Backend({"model": "/x/d.gguf", "args": []}, launcher=["x"], log=lambda m: None)])
check(p2.key == mc.stored_key(), "without a custom key the stored key is used")

# ---- 5. loopback origins ----
LO = od.local_origin
for o in ("http://localhost:3000", "https://localhost", "http://127.0.0.1:8080", "http://127.5.6.7", "http://[::1]:9"):
    check(LO(o), f"origin accepted: {o}")
for o in ("http://127.attacker.example", "http://127.0.0.1.evil.test:80", "http://localhost.evil.test", "http://evil.test", "http://127.0.0.256", "http://128.0.0.1",
          "http://127.1", "null", "", "http://[::2]", "ftp://localhost", "http://user@127.0.0.1", "http://127.0.0.1@evil.test"):
    check(not LO(o), f"origin refused: {o!r}")

# ---- 6. a saved state of another model is not restored ----
def fake_model(name, body=b"", size=3 << 20):
    p = os.path.join(TMP, name)
    with open(p, "wb") as f:
        f.write(b"GGUF" + b"\1" * 100)
        f.write(b"\0" * (size - 104))
    if body:
        with open(p, "r+b") as f:
            f.seek(size // 2); f.write(body)
    return p
A = fake_model("m.gguf"); B = fake_model("m2.gguf", b"different weights")
os.makedirs(os.path.join(TMP, "d")); C = os.path.join(TMP, "d", "m.gguf")
with open(C, "wb") as f, open(fake_model("tmp.gguf", b"other"), "rb") as g:
    f.write(g.read())
idA = mc.model_identity(A, 65536, "q8_0", "q8_0")
check(idA and idA == mc.model_identity(A, 65536, "q8_0", "q8_0"), "same model, ctx and KV types: same identity")
check(idA != mc.model_identity(B, 65536, "q8_0", "q8_0"), "another file of the same size and header: another identity")
check(idA != mc.model_identity(C, 65536, "q8_0", "q8_0"), "the same file name and size with other content: another identity")
check(idA != mc.model_identity(A, 32768, "q8_0", "q8_0") and idA != mc.model_identity(A, 65536, None, None), "another ctx or KV type: another identity")
check(mc.model_identity(os.path.join(TMP, "none.gguf"), 1) == "", "an unreadable model has no identity")
sd = os.path.join(mc.CACHE_DIR, "slots", "proj"); os.makedirs(sd)
open(os.path.join(sd, "proj.bin"), "wb").write(b"x")
check(not mc.project_id_ok(sd, "proj"), "an old saved file without identity is not restored")
open(os.path.join(sd, "server.id"), "w").write(idA + "\n")
check(not mc.project_id_ok(sd, "proj"), "still not restored: the file has no identity record")
open(os.path.join(sd, "proj.bin.id"), "w").write(idA + "\n")
check(mc.project_id_ok(sd, "proj"), "the file saved by this model is restored")
open(os.path.join(sd, "server.id"), "w").write(mc.model_identity(B, 65536, "q8_0", "q8_0") + "\n")
check(not mc.project_id_ok(sd, "proj"), "model B is refused model A's saved state")
# the CLI: restore refused without a request to the server; save stamps the identity
hits = []
class S(BaseHTTPRequestHandler):
    def _r(self):
        n = int(self.headers.get("Content-Length") or 0); self.rfile.read(n); hits.append(self.path)
        b = b'{"ok":1}'; self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    do_GET = do_POST = _r
    def log_message(self, *a): pass
fs = ThreadingHTTPServer(("127.0.0.1", 0), S); threading.Thread(target=fs.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{fs.server_address[1]}"
try:
    mc.project(["restore", "proj", "--url", url, "--api-key", "k"]); refused = False
except SystemExit:
    refused = True
check(refused and not [h for h in hits if "restore" in h], "project restore of model A's file on model B's server: refused before any request")
mc.project(["save", "proj", "--url", url, "--api-key", "k"])
check(open(os.path.join(sd, "proj.bin.id")).read().strip() == open(os.path.join(sd, "server.id")).read().strip(), "project save records the server's model identity")
mc.project(["restore", "proj", "--url", url, "--api-key", "k"])
check(any("restore" in h for h in hits), "after a save by this server, restore goes through")
fs.shutdown()

print("\nall security tests passed" if not fails else f"\n{len(fails)} security test(s) FAILED")
sys.exit(1 if fails else 0)
