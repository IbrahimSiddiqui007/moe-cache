#!/usr/bin/env python3
"""moe_cache.http must not follow redirects: urllib would copy the Authorization header (the stored API key) to the redirect target."""
import http.server, os, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin"))
import moe_cache as mc

seen = []
class Target(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        seen.append(self.headers.get("Authorization"))
        self.send_response(200); self.end_headers(); self.wfile.write(b"target")
    def log_message(self, *a): pass
tsrv = http.server.HTTPServer(("127.0.0.1", 0), Target)
class Redir(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(302); self.send_header("Location", f"http://127.0.0.1:{tsrv.server_address[1]}/x"); self.end_headers()
    def log_message(self, *a): pass
rsrv = http.server.HTTPServer(("127.0.0.1", 0), Redir)
for s in (tsrv, rsrv):
    threading.Thread(target=s.serve_forever, daemon=True).start()
st, body = mc.http(f"http://127.0.0.1:{rsrv.server_address[1]}/v1/models", api_key="SECRET")
assert not seen, f"redirect was followed and the target saw Authorization={seen}"
assert st == 302, st
print("redirect_key_test: ok")
