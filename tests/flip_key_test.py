#!/usr/bin/env python3
"""SERVER-692: the stored API key goes only to a server this launcher started itself. A local program that answers /v1/models 200 and then 401 (the flip
server) must never get an Authorization header from any caller of key_for(): connect, connect --check, project save / restore, the launch ready check.
Also: a port the launcher really owns (record file with a live pid) still gets the key.
   python3 tests/flip_key_test.py"""
import json, os, shutil, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
TMP = tempfile.mkdtemp(prefix="moe-flip-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
os.environ["HOME"] = TMP
os.environ.pop("LLAMA_API_KEY", None)
import moe_cache as mc  # noqa: E402
import moe_cache_launch as L  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

SECRET = "STORED-SECRET-KEY-123"
os.makedirs(mc.CACHE_DIR, exist_ok=True)
open(mc.KEY_FILE, "w").write(SECRET + "\n")

seen = []
class Flip(BaseHTTPRequestHandler):
    """/v1/models: 200 the first time, 401 after (so a second probe, which key_for() used to make, sees "wants a key")."""
    n = 0
    def log_message(self, *a): pass
    def _send(self, code, body=b"{}"):
        self.send_response(code); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        seen.append(self.headers.get("Authorization"))
        if self.path == "/v1/models":
            Flip.n += 1
            return self._send(200 if Flip.n == 1 else 401, json.dumps({"data": [{"id": "m.gguf"}]}).encode() if Flip.n == 1 else b"{}")
        self._send(401)
    do_POST = do_GET
srv = ThreadingHTTPServer(("127.0.0.1", 0), Flip)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]
url = f"http://127.0.0.1:{port}"
def leaked():
    return any(a and SECRET in a for a in seen)
def reset():
    Flip.n = 0; seen.clear()

env = dict(os.environ)
def cli(*args):
    return subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), *args], capture_output=True, text=True, timeout=60, env=env)

check(not mc.launcher_owns(url), "a port nobody of ours started is not ours")
check(mc.key_for(url) is None and mc.key_for(url, None, False) is None, "key_for: untrusted by default")
reset(); mc.key_for(url); check(not leaked() and len(seen) == 0, "key_for(url) makes no request and sends nothing")

# every caller
reset(); cli("connect", "opencode", "--url", url, "--config", os.path.join(TMP, "oc.json"))
check(not leaked(), f"connect: the flip server got no stored key ({seen[:3]})")
reset(); cli("connect", "opencode", "--url", url, "--config", os.path.join(TMP, "oc2.json"), "--check")
check(not leaked(), f"connect --check: the flip server got no stored key ({seen[:3]})")
os.makedirs(os.path.join(mc.CACHE_DIR, "slots", "p1"), exist_ok=True)
for cmd in ("save", "restore"):
    reset(); cli("project", cmd, "p1", "--url", url)
    check(not leaked(), f"project {cmd}: the flip server got no stored key ({seen[:3]})")
reset()
d = L.Deps()
d.up(url, False)
if hasattr(d, "served_model"): d.served_model(url)
check(not leaked(), f"launch ready check (up, served_model): the flip server got no stored key ({seen[:3]})")

# a port the launcher really owns still gets it
os.makedirs(mc.SERVER_DIR, exist_ok=True)
rec = os.path.join(mc.SERVER_DIR, f"{port}.pid"); open(rec, "w").write(str(os.getpid()))
check(mc.launcher_owns(url), "a port with a live launcher record is ours")
Flip.n = 1                                                  # the server now answers 401 to a keyless probe
check(mc.key_for(url, None, mc.launcher_owns(url)) == SECRET, "our own server that wants a key gets the stored one")
open(rec, "w").write("999999999")
check(not mc.launcher_owns(url), "a record whose pid is gone is not ours")
os.remove(rec)
with mc.OwnsPort(port):
    check(mc.launcher_owns(url), "OwnsPort writes the record while the launcher runs")
check(not mc.launcher_owns(url), "...and removes it afterwards")
check(not mc.launcher_owns("http://10.0.0.5:%d" % port), "a non-loopback host is never ours")

srv.shutdown(); shutil.rmtree(TMP, ignore_errors=True)
print("\nFAILED: " + "; ".join(fails) if fails else "\nflip key tests passed")
sys.exit(1 if fails else 0)
