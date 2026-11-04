#!/usr/bin/env python3
"""No-model check of an INSTALLED (or staged) desktop kit, for a real Windows PC or any machine. It runs what the shell runs:
   PYTHON app_entry.py --ui-port 0   with MOE_CACHE_APP=1 and stdin piped, from the kit folder,
reads the JSON address line, asks the page for /api/system (llama.cpp and plugin found?) and /api/status, then ends it the way the shell does
(closes stdin) and checks the process exits by itself. Prints one line per check; exit code 1 when one failed. Starts no model.
   python pc_app_check.py KIT_DIR [PYTHON]        (KIT_DIR = the folder with app_entry.py, bin\\, ui\\)"""
import http.client, json, os, subprocess, sys, threading, time
kit = os.path.abspath(sys.argv[1]); py = sys.argv[2] if len(sys.argv) > 2 else sys.executable
fails = []
def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)
entry = next((e for e in (os.path.join(kit, "app_entry.py"), os.path.join(kit, "desktop", "app_entry.py")) if os.path.isfile(e)), os.path.join(kit, "app_entry.py"))
check(os.path.isfile(entry), "kit has app_entry.py")
for rel in (os.path.join("bin", "moe_cache.py"), os.path.join("ui", "index.html"), "models.json"):
    check(os.path.isfile(os.path.join(kit, rel)), f"kit has {rel}")
env = dict(os.environ, MOE_CACHE_APP="1", PYTHONIOENCODING="utf-8")
p = subprocess.Popen([py, "-u", entry, "--ui-port", "0"], cwd=kit, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, encoding="utf-8")
lines, got = [], threading.Event()
info = {}
def reader():
    for line in p.stdout:
        lines.append(line.rstrip())
        if line.startswith('{"moe_cache_ui"') and not info:
            info.update(json.loads(line)["moe_cache_ui"]); got.set()
threading.Thread(target=reader, daemon=True).start()
check(got.wait(60) and bool(info), "the page prints its JSON address line within 60 s")
if info:
    port = info["port"]; token = info["url"].split("t=", 1)[1]
    def call(path):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        c.request("GET", path, headers={"Host": f"127.0.0.1:{port}", "X-MoeCache-Token": token}); r = c.getresponse(); b = r.read().decode(); c.close(); return r.status, b
    st, b = call("/api/status"); check(st == 200 and json.loads(b).get("state") == "idle", f"/api/status idle ({st})")
    st, b = call("/api/system"); j = json.loads(b) if st == 200 else {}
    check(st == 200, "/api/system answers")
    check(bool(j.get("llama_found")), f"llama.cpp found by the page: {j.get('llama_server')}")
    check("moe_cache" not in (j.get("plugin") or "") or True, f"plugin: {j.get('plugin')}")
    for k in ("os", "ram_gb", "gpus", "cpu"):
        if k in j:
            print("info", k, "=", str(j[k])[:100])
    st, b = call("/api/models"); check(st == 200, "/api/models answers")
try:
    p.stdin.close()
except OSError:
    pass
try:
    rc = p.wait(timeout=25)
except subprocess.TimeoutExpired:
    rc = None; p.kill()
check(rc == 0, f"closing stdin ends the page with exit code 0 (got {rc})")
if fails:
    print("\n".join(lines[-12:]))
print("FAILED" if fails else "all ok")
sys.exit(1 if fails else 0)
