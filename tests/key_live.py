#!/usr/bin/env python3
"""API key and CORS check with a real server (DECISION_LLAMA_SERVER_KEY, option C; CI on Granite, Linux):
  1. `server MODEL --agent`: a request WITHOUT a key gets 401, a wrong key 401, `moe_cache.py key` as Bearer 200, /health stays public;
  2. the key never shows in the launcher's --dry-run output, its log, or any process's command line (it goes to llama-server by environment);
  3. a plain (non-agent) server has no key and answers a CORS preflight from http://evil.example without echoing that origin, while a localhost origin is
     allowed (skipped with a note when this llama-server has no --cors-origins).
   python3 tests/key_live.py MODEL.gguf --llama-server PATH [--port 8086]
The plugin is found the launcher's way (MOE_CACHE_LIB or the build folder). Linux (reads /proc). Standard library only."""
import argparse, glob, json, os, signal, subprocess, sys, tempfile, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--llama-server", required=True); ap.add_argument("--port", type=int, default=8086)
a = ap.parse_args()
model = os.path.abspath(a.model)
LAUNCHER = [sys.executable, os.path.join(HERE, "bin", "moe_cache.py")]
env = dict(os.environ, XDG_CACHE_HOME=tempfile.mkdtemp(prefix="moe-key-live-"))
env.pop("LLAMA_API_KEY", None)
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)

def call(method, path, body=None, headers=None, port=None):
    r = urllib.request.Request(f"http://127.0.0.1:{port or a.port}{path}", json.dumps(body).encode() if body is not None else None,
                               dict({"Content-Type": "application/json"}, **(headers or {})), method=method)
    try:
        with urllib.request.urlopen(r, timeout=120) as x:
            return x.status, dict(x.headers), x.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()

def wait_health(proc, port, limit=300):
    t0 = time.time()
    while time.time() - t0 < limit and proc.poll() is None:
        try:
            if call("GET", "/health", port=port)[0] == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def start(log, *extra):
    return subprocess.Popen(LAUNCHER + ["server", model, "--llama-server", a.llama_server, "--port", str(a.port), "--ctx", "4096", "--always-plugin",
                                        *extra], stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)

def stop(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
    time.sleep(2)

def cmdlines():
    out = []
    for f in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            out.append(open(f, "rb").read())
        except OSError:
            pass
    return out

chat = {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0}

# 1 + 2: agent mode needs the stable per-user key
dry = subprocess.run(LAUNCHER + ["server", model, "--llama-server", a.llama_server, "--ctx", "4096", "--always-plugin", "--agent", "--dry-run"],
                     capture_output=True, text=True, env=env)
key = subprocess.run(LAUNCHER + ["key"], capture_output=True, text=True, env=env).stdout.strip()
check(len(key) >= 30 and "\n" not in key, f"`moe_cache.py key` prints one key line ({len(key)} chars)")
check(oct(os.stat(os.path.join(env["XDG_CACHE_HOME"], "moe-cache", "api.key")).st_mode & 0o777) == "0o600", "the key file is mode 0600")
check(key not in dry.stdout and key not in dry.stderr and "LLAMA_API_KEY=(hidden)" in dry.stdout, "--dry-run does not print the key (shows LLAMA_API_KEY=(hidden))")
p = start("key_live_agent.log", "--agent")
try:
    ready = wait_health(p, a.port)
    check(ready, "the agent-mode server is ready (/health is public)")
    if ready:
        check(call("POST", "/v1/chat/completions", chat)[0] == 401, "no key -> 401")
        check(call("POST", "/v1/chat/completions", chat, {"Authorization": "Bearer wrong-key"})[0] == 401, "wrong key -> 401")
        st, _, body = call("POST", "/v1/chat/completions", chat, {"Authorization": "Bearer " + key})
        check(st == 200, f"stored key as Bearer -> 200 ({st})")
        check(call("GET", "/slots", headers={})[0] in (401, 403), "/slots needs the key")
        check(call("GET", "/slots", headers={"Authorization": "Bearer " + key})[0] in (200, 501), "/slots with the key answers")
        k = key.encode()
        check(not any(k in c for c in cmdlines()), "the key is on no process's command line")
        log = open("key_live_agent.log", encoding="utf-8", errors="replace").read()
        check(key not in log, "the launcher's log does not contain the key")
        check("API key" in log and "api.key" in log, "the log says a key is in use and where it is stored")
finally:
    stop(p)

# 3: a plain server: no key, CORS allow-list
helptext = subprocess.run([a.llama_server, "--help"], capture_output=True, text=True).stdout
if "--cors-origins" not in helptext:
    print("  skip  this llama-server has no --cors-origins: the CORS check does not apply", flush=True)
else:
    p = start("key_live_plain.log")
    try:
        ready = wait_health(p, a.port)
        check(ready, "the plain server is ready")
        if ready:
            check(call("POST", "/v1/chat/completions", chat)[0] == 200, "plain server: no key needed")
            pre = {"Origin": "http://evil.example", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type,authorization"}
            st, hdr, _ = call("OPTIONS", "/v1/chat/completions", headers=pre)
            allow = {k.lower(): v for k, v in hdr.items()}.get("access-control-allow-origin", "")
            check("evil.example" not in allow and allow != "*", f"preflight from http://evil.example is not allowed (Access-Control-Allow-Origin: '{allow}')")
            st, hdr, _ = call("OPTIONS", "/v1/chat/completions", headers=dict(pre, Origin="http://localhost:3000"))
            allow = {k.lower(): v for k, v in hdr.items()}.get("access-control-allow-origin", "")
            check("localhost" in allow, f"a localhost origin is still allowed (Access-Control-Allow-Origin: '{allow}')")
    finally:
        stop(p)

print(f"key checks: {len(fails)} failed", flush=True)
sys.exit(1 if fails else 0)
