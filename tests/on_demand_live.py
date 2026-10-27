#!/usr/bin/env python3
"""`moe_cache.py on-demand` with a real model (CI: Granite 1B): the first request starts the server, answers stream back, the idle stop frees it,
a second request starts it again, `connect opencode --write` points at the endpoint without starting the model.

   python3 tests/on_demand_live.py MODEL.gguf --llama-server PATH [--port 8087]     (MOE_CACHE_LIB as for the launcher)
"""
import argparse, json, os, subprocess, sys, tempfile, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--llama-server", required=True); ap.add_argument("--port", type=int, default=8087)
a = ap.parse_args()
TMP = tempfile.mkdtemp(prefix="od-live-")
env = dict(os.environ, XDG_CACHE_HOME=TMP, LLAMA_SERVER=a.llama_server)
env.pop("LLAMA_API_KEY", None)
MC = [sys.executable, os.path.join(ROOT, "bin", "moe_cache.py")]
U = f"http://127.0.0.1:{a.port}"
fails = []


def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""), flush=True)
    if not ok:
        fails.append(name)


def req(path, body=None, key=None, stream=False):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    r = urllib.request.Request(U + path, json.dumps(body).encode() if body else None, h)
    try:
        with urllib.request.urlopen(r, timeout=600) as f:
            return f.status, ([ln for ln in f if ln.startswith(b"data: ")] if stream else f.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def state():
    return json.loads(req("/health")[1])["on_demand"]


r = subprocess.run(MC + ["on-demand", a.model, "--ram", "3", "--ctx", "4096", "--always-plugin", "--port", str(a.port), "--idle", "0.25", "--status"],
                   env=env, capture_output=True, text=True, timeout=600)
print(r.stdout + r.stderr)
check("on-demand remembers the model (the launcher's dry run accepts the options)", r.returncode == 0)
key = subprocess.run(MC + ["key"], env=env, capture_output=True, text=True).stdout.strip()
out = open(os.path.join(TMP, "listener.out"), "w")
p = subprocess.Popen(MC + ["on-demand"], env=env, stdout=out, stderr=subprocess.STDOUT)
try:
    for _ in range(50):
        try:
            if state() == "idle":
                break
        except Exception:
            time.sleep(0.2)
    check("listener up, model not started", state() == "idle")
    check("no key: 401", req("/v1/chat/completions", {"messages": []})[0] == 401)
    cfg = os.path.join(TMP, "opencode.json")
    r = subprocess.run(MC + ["connect", "opencode", "--write", "--config", cfg], env=env, capture_output=True, text=True, timeout=60)
    check("connect opencode --write uses the on-demand port, model still stopped",
          r.returncode == 0 and f"127.0.0.1:{a.port}/v1" in open(cfg).read() and state() == "idle", r.stdout + r.stderr)
    chat = {"messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 8, "temperature": 0}
    t = time.time()
    st, ev = req("/v1/chat/completions", dict(chat, stream=True), key=key, stream=True)
    check(f"first request starts the model and streams ({time.time() - t:.1f} s, {len(ev)} events)", st == 200 and len(ev) > 2 and ev[-1].startswith(b"data: [DONE]"), (st, ev[-1:]))
    st, body = req("/v1/chat/completions", chat, key=key)
    check("non-streaming answer", st == 200 and json.loads(body)["choices"][0]["message"]["content"], body[:200])
    t = time.time()
    while state() != "idle" and time.time() - t < 90:
        time.sleep(1)
    check(f"idle stop ({time.time() - t:.0f} s after the last answer)", state() == "idle")
    st, body = req("/v1/chat/completions", chat, key=key)
    check("second request starts it again", st == 200 and state() == "ready", body[:200])
finally:
    p.terminate()
    p.wait(timeout=90)
    out.close()
log = open(os.path.join(TMP, "listener.out")).read()
print(log)
# (on Windows terminate() kills the listener, so its own last stop line is missing there)
check("one start line per start, one stop line per stop", log.count(" starting ") == 2 and log.count(" stopped (") in ((1, 2) if os.name == "nt" else (2,)), log)
lg = open(os.path.join(TMP, "moe-cache", "on-demand.log"), errors="replace").read()
check("the key never shows in the server log", key not in lg and key not in log)
print("plugin lines:", *[ln for ln in lg.splitlines() if "cache mode" in ln or "not loaded" in ln][:2], sep="\n  ")
print("FAILED: " + ", ".join(fails) if fails else "all on-demand live tests passed")
sys.exit(1 if fails else 0)
