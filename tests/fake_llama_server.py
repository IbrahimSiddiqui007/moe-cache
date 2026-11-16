#!/usr/bin/env python3
"""A fake llama-server for the driver tests (no model): /health, /props, /tokenize and a streaming /v1/chat/completions with llama.cpp-style timings.
   fake_llama_server.py PORT [N_CTX]    speeds from the environment: FAKE_P_TPS (prompt, default 100) and FAKE_D_TPS (decode, default 10); FAKE_ARM names it in /props."""
import http.server, json, os, socketserver, sys

PORT = int(sys.argv[1]); N_CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
P_TPS = float(os.environ.get("FAKE_P_TPS", "100")); D_TPS = float(os.environ.get("FAKE_D_TPS", "10"))
LAST = []
toks = lambda m: len(m["content"]) // 4 + 8


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    def log_message(self, *a): pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._json(200, {"status": "ok"})
        elif self.path == "/props":
            self._json(200, {"default_generation_settings": {"n_ctx": N_CTX}})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        global LAST
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self._json(200, {"tokens": list(range(len(body.get("content", "")) // 4))})
        msgs = body["messages"]; total = sum(toks(m) for m in msgs)
        if total > N_CTX:
            return self._json(400, {"error": {"code": 400, "message": f"the request ({total} tokens) exceeds the available context size ({N_CTX} tokens), try increasing it"}})
        common = 0
        for a, b in zip(LAST, msgs):
            if a != b: break
            common += toks(b)
        pn = total - common; pred = body.get("max_tokens", 32) // 2; answer = f"Answer {len(msgs)}: " + "word " * 30
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for piece in (answer[:20], answer[20:]):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n\n"); self.wfile.flush()
        tm = {"prompt_n": pn, "cache_n": common, "prompt_ms": pn / P_TPS * 1000, "prompt_per_second": P_TPS, "predicted_n": pred, "predicted_ms": pred / D_TPS * 1000, "predicted_per_second": D_TPS}
        self.wfile.write(b"data: " + json.dumps({"choices": [], "timings": tm, "usage": {"prompt_tokens": total, "completion_tokens": pred}}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        LAST = msgs + [{"role": "assistant", "content": answer}]


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


S(("127.0.0.1", PORT), H).serve_forever()
