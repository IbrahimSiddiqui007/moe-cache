#!/usr/bin/env python3
"""Control page with a real model (GUI1 draft, step 2; CI on Granite, Linux and Windows): starts `moe_cache.py ui`, adds the model's folder, starts the
model through the page with a RAM budget small enough that the launcher keeps the plugin, sends one request, checks that the page shows the plugin's
live numbers (hit rate from the stats snapshot: SIGUSR1 on Linux, a named event on Windows), and stops it through the page.
   python3 tests/ui_live.py MODEL.gguf --llama-server PATH [--ram GB] [--ui-port 8075] [--port 8076]
The plugin is found the launcher's way (MOE_CACHE_LIB or the build folder). Standard library only."""
import argparse, json, os, subprocess, sys, tempfile, time, urllib.error, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--llama-server", required=True); ap.add_argument("--ram", help="GB; default: the model + 1.2 GB, just under the launcher's stock rule (model + 1.5 GB), so it keeps the plugin")
ap.add_argument("--ui-port", type=int, default=8075); ap.add_argument("--port", type=int, default=8076)
a = ap.parse_args()
model = os.path.abspath(a.model)
a.ram = a.ram or f"{os.path.getsize(model) / 1e9 + 1.2:.1f}"
UI = f"http://127.0.0.1:{a.ui_port}"
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)

def req(path, body=None, timeout=120, base=UI):
    r = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as x:
            return x.status, json.loads(x.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

env = dict(os.environ, XDG_CACHE_HOME=tempfile.mkdtemp(prefix="moe-ui-live-"))    # the page's settings and the usage profile stay out of the real cache
log = open("ui_live_page.log", "w")
page = subprocess.Popen([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "ui", "--no-browser", "--ui-port", str(a.ui_port),
                         "--llama-server", a.llama_server], stdout=log, stderr=subprocess.STDOUT, env=env)
try:
    for _ in range(60):
        try:
            if req("/api/status", timeout=2)[0] == 200:
                break
        except Exception:
            time.sleep(0.5)
    st, j = req("/api/folders", {"add": os.path.dirname(model)})
    check(st == 200 and any(m["path"] == model for m in j["models"]), f"the page lists the model after adding its folder ({st})")
    st, plan = req("/api/plan?" + urllib.parse.urlencode({"model": model, "ram": a.ram, "ctx": 4096, "mode": "exact"}), timeout=300)
    check(st == 200 and plan.get("kind") in ("cache", "gpu"), f"plan at {a.ram} GB: {plan.get('verdict')} ({'; '.join(plan.get('reasons', [])[:2])})")
    st, j = req("/api/start", {"model": model, "port": a.port, "ram": a.ram, "ctx": 4096, "mode": "exact"})
    check(st == 200, f"start through the page ({j.get('error', j.get('state'))})")
    t0, s = time.time(), {}
    while time.time() - t0 < 600:
        s = req("/api/status")[1]
        if s["state"] != "starting":
            break
        time.sleep(1)
    check(s.get("state") == "ready", f"ready after {time.time() - t0:.0f} s ({s.get('state')}); mode: {s.get('mode')}")
    check(s.get("mode", "").startswith("moe-cache"), "the launcher kept the plugin")
    if s.get("state") == "ready":
        st, r = req("/v1/chat/completions", {"messages": [{"role": "user", "content": "Count from one to twenty in words."}], "max_tokens": 48, "temperature": 0},
                    timeout=600, base=f"http://127.0.0.1:{a.port}")
        check(st == 200 and r.get("choices"), "one request")
        t1 = time.time()
        while time.time() - t1 < 30:
            s = req("/api/status")[1]
            if s["cache"].get("hit_rate_total") is not None and s["last"].get("decode_tps"):
                break
            time.sleep(1)
        c = s.get("cache", {})
        print(f"  status: decode {s['last'].get('decode_tps')} tok/s, prefill {s['last'].get('prefill_tps')} tok/s, hit rate {c.get('hit_rate_total')}, "
              f"cache {c.get('resident')} / {c.get('budget')} bytes, llama-server RAM {s.get('server_rss')}, warnings {[w['line'] for w in s.get('warnings', [])]}")
        check(s["last"].get("decode_tps", 0) > 0, "decode tok/s from the timing lines")
        check(c.get("hit_rate_total") is not None and c.get("budget"), "hit rate and cache budget from the plugin's stats snapshot")
        check(s.get("server_rss"), "llama-server's RAM")
    req("/api/stop", {})
    t2 = time.time()
    while time.time() - t2 < 120:
        s = req("/api/status")[1]
        if s["state"] in ("stopped", "exited"):
            break
        time.sleep(1)
    check(s["state"] == "stopped", f"stopped through the page ({s['state']}, exit code {s.get('rc')}, {time.time() - t2:.0f} s)")
    st, lg = req("/api/log?since=0")
    if fails:
        print("\n".join(l for _, l in lg.get("lines", [])[-60:]))
finally:
    page.terminate()
    try:
        page.wait(timeout=30)
    except subprocess.TimeoutExpired:
        page.kill()
print("ui live test:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
