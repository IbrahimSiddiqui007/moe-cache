#!/usr/bin/env python3
"""Runs the real OpenCode and Hermes (whatever is installed) once each against a fake OpenAI server in a temporary HOME and writes the first requests they
send to tests/agent_sessions/real_prompts.json (GUI-016). No model, no network, nothing outside the temp folder is touched. The page's agent guess
(bin/moe_cache_ui.py guess_agent) is tested against these real texts by tests/agent_guess_test.py.
   python3 tools/capture_agent_prompts.py [--out FILE]"""
import http.server, json, os, shutil, subprocess, sys, tempfile, threading

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = []


class Fake(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def send_json(self, d):
        b = json.dumps(d).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.send_json({"object": "list", "data": [{"id": "fake-model", "object": "model", "meta": {"n_ctx_train": 65536}}], "models": [{"name": "fake-model", "model": "fake-model"}]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        LOG.append(body)
        if not body.get("stream"):
            return self.send_json({"id": "1", "object": "chat.completion", "model": "fake-model", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        def chunk(d):
            x = ("data: " + d + "\n\n").encode(); self.wfile.write(b"%x\r\n" % len(x) + x + b"\r\n")
        base = {"id": "1", "object": "chat.completion.chunk", "model": "fake-model"}
        chunk(json.dumps(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": "Hello."}, "finish_reason": None}])))
        chunk(json.dumps(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}], usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})))
        chunk("[DONE]"); self.wfile.write(b"0\r\n\r\n")


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(HERE, "tests", "agent_sessions", "real_prompts.json")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    srv.handle_error = lambda *a: None          # an agent closing a connection early is fine
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    home = tempfile.mkdtemp(prefix="moe-agents-")
    env = dict(os.environ, HOME=home, USERPROFILE=home, XDG_CONFIG_HOME=home + "/.config", XDG_DATA_HOME=home + "/.local/share", XDG_CACHE_HOME=home + "/.cache", HERMES_HOME=home + "/.hermes")
    proj = os.path.join(home, "proj"); os.makedirs(proj)
    result = {"about": "Real first requests of the installed agents, captured by tools/capture_agent_prompts.py against a fake OpenAI server (no model). System prompts cut at 7000 characters. "
                       "'main' is the conversation, 'title' the agent's second, title-making request.", "versions": {}, "requests": []}
    cmds = {"OpenCode": (["opencode", "run", "-m", "moe-cache/fake-model", "Say hello in one word."], ["opencode", "--version"], "opencode"),
            "Hermes": (["hermes", "chat", "-q", "Say hello in one word."], ["hermes", "--version"], "hermes")}
    try:
        for agent, (cmd, ver, conn) in cmds.items():
            if not shutil.which(cmd[0]):
                print(f"{agent}: not installed, skipped"); continue
            subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "connect", conn, "--url", url, "--write"], env=env, capture_output=True, text=True, timeout=120, check=True)
            result["versions"][agent] = subprocess.run(ver, env=env, capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[0]
            LOG.clear()
            subprocess.run(["nice", "-n", "19"] + cmd if os.name != "nt" else cmd, cwd=proj, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=180)
            for body in LOG:
                ms = body.get("messages") or []
                if not ms or ms[0].get("role") != "system":
                    continue
                s = ms[0]["content"]; s = s if isinstance(s, str) else json.dumps(s)
                result["requests"].append({"agent": agent, "kind": "main" if body.get("tools") else "title", "system": s[:7000], "system_chars": len(s), "first_user": str(ms[1]["content"])[:200]})
            print(f"{agent}: {sum(1 for r in result['requests'] if r['agent'] == agent)} request(s)")
    finally:
        shutil.rmtree(home, ignore_errors=True)
    json.dump(result, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
