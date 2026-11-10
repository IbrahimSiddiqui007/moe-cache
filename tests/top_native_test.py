#!/usr/bin/env python3
"""R15-03 (phase 0 spike): the native Monitor bin/moe-cache-top against a fake llama-server (/metrics, /slots) and a fake control page (/api/monitor with `schema`).
The binary is built by ./build_top.sh (FTXUI 7.0.3, vendored); the test builds it when MOE_CACHE_TOP_BUILD=1 or CI is set, runs it when it exists, and otherwise says SKIP.
   python3 tests/top_native_test.py"""
import json, os, re, subprocess, sys, threading, http.server, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(HERE, "bin", "moe-cache-top" + (".exe" if os.name == "nt" else ""))
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

if os.name == "nt":
    print("SKIP: the native Monitor is built for Windows by build_top.sh --windows (mingw cross compiler), not here")
    sys.exit(0)
if not os.path.isfile(EXE) and (os.environ.get("MOE_CACHE_TOP_BUILD") == "1" or os.environ.get("CI")):
    r = subprocess.run(["bash", os.path.join(HERE, "build_top.sh")], capture_output=True, text=True, timeout=1500, env=dict(os.environ, JOBS=os.environ.get("JOBS", "4")))
    print("build:", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])
if not os.path.isfile(EXE):
    print("SKIP: bin/moe-cache-top is not built (./build_top.sh, or MOE_CACHE_TOP_BUILD=1)")
    sys.exit(0)

state = {"n": 0, "metrics_calls": 0}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send_body(self, code, body, ctype="text/plain"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/metrics":
            state["n"] += 1
            n = state["n"]
            self.send_body(200, "# HELP llamacpp:prompt_tokens_total x\n# TYPE x counter\n"
                           f"llamacpp:prompt_tokens_total {500 * n}\nllamacpp:prompt_seconds_total {0.25 * n}\nllamacpp:tokens_predicted_total {100 * n}\nllamacpp:tokens_predicted_seconds_total {2.0 * n}\n"
                           "llamacpp:prompt_tokens_seconds 2000\nllamacpp:predicted_tokens_seconds 50\nllamacpp:kv_cache_usage_ratio 0.25\nllamacpp:requests_processing 1\nllamacpp:requests_deferred 0\n"
                           'llamacpp:labelled{slot="1"} 3\n')
        elif self.path == "/slots":
            self.send_body(200, json.dumps([{"id": 0, "n_ctx": 8192, "is_processing": True, "next_token": {"n_decoded": 42, "n_remain": -1, "has_next_token": True}},
                                            {"id": 1, "n_ctx": 8192, "is_processing": False, "params": {"x": [1, 2, {"y": "\\u00e9\"q"}]}, "next_token": {"n_decoded": 0, "n_remain": -1}}]), "application/json")
        elif self.path.startswith("/api/monitor"):
            if self.headers.get("X-MoeCache-Token") != "tok":
                return self.send_body(403, "{}")
            state["monitor_path"] = self.path
            self.send_body(200, json.dumps({"schema": 1, "reset": False, "requests": [], "layer_cols": []}), "application/json")
        else:
            self.send_body(404, "no")
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]

def run(*args, timeout=30):
    return subprocess.run([EXE, *args], capture_output=True, text=True, timeout=timeout)

r = run("--version")
check(r.returncode == 0 and r.stdout.startswith("moe-cache-top ") and "FTXUI 7.0.3" in r.stdout, f"--version ({r.stdout.strip()})")
r = run("--once", "--no-color", "--port", str(port), "--interval", "300", "--width", "100", "--height", "30")
out = r.stdout
check(r.returncode == 0 and "decode" in out and "prompt" in out and "slot" in out, "--once draws the Overview (decode, prompt, slots) and exits 0")
check("50.0 tok/s" in out, f"decode speed from the counters' change: 100 tokens / 2.0 s = 50.0 tok/s ({re.findall(r'[0-9.]+ tok/s', out)[:3]})")
check("2000 tok/s" in out or "2.00e+03" in out, "prompt speed: 500 tokens / 0.25 s = 2000 tok/s")
check("4096 / 16384" in out, "context: 25 % of the 16384 tokens of KV cache the two slots share")
check("working" in out and "idle" in out and "42" in out, "the slot table: one working slot (42 decoded) and one idle")
check("\x1b" not in out and 29 <= len(out.splitlines()) <= 31 and max(len(l) for l in out.splitlines()) == 100, f"--no-color and the frame has the size that was asked for ({len(out.splitlines())} lines, widest {max(len(l) for l in out.splitlines())})")
r = run("--json", "--port", str(port), "--interval", "200")
j = json.loads(r.stdout)
check(r.returncode == 0 and j["reachable"] is True and abs(j["decode_tps"] - 50.0) < 0.05 and j["slots"] == 2 and j["schema"] is None, f"--json: {r.stdout.strip()}")
r = run("--once", "--no-color", "--port", str(port), "--interval", "200", "--monitor-url", f"http://127.0.0.1:{port}", "--token", "tok")
check("schema 1" in r.stdout and "layers=0" in state.get("monitor_path", ""), f"the control page's schema number is read and shown (path {state.get('monitor_path')})")
r = run("--once", "--no-color", "--port", str(port), "--interval", "200", "--monitor-url", f"http://127.0.0.1:{port}", "--token", "wrong")
check("schema control page not answering" in r.stdout.replace("\n", " ") or "not answering" in r.stdout, "a refused token is said, not hidden")
r = run("--once", "--no-color", "--port", "1", "--width", "90", "--height", "10")
check(r.returncode == 1 and "not answering" in r.stdout, f"no server: exit 1 and a plain line ({r.stdout.strip().splitlines()[-2:-1]})")
r = run("--port")
check(r.returncode == 2, "an option without its value is refused")
r = run("--nope")
check(r.returncode == 2 and "unknown option" in r.stderr, "an unknown option is refused")
size = os.path.getsize(EXE)
print(f"       binary size {size / 1e6:.2f} MB")
check(size < 4 * 1024 * 1024, "the binary is under the 4 MB kill line")

print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
