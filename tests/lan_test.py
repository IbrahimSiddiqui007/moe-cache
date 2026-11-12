#!/usr/bin/env python3
"""SERVER-016 (--lan) and GUI-012 (--ui-lan), without a second computer: the address rules (bin/moe_cache_lan.py), the launcher's --lan on a dry run with fake
addresses (MOE_CACHE_LAN_ADDRS), and the page server with a fake remote client. docs/LAN_ACCESS.md has the threat table these checks follow.
   python3 tests/lan_test.py"""
import json, os, shutil, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-lan-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_lan as lan  # noqa: E402
import moe_cache_ui as ui  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# ---- the address rules ----
for addr, kind in (("127.0.0.1", "loopback"), ("::1", "loopback"), ("192.168.1.20", "private"), ("10.0.0.5", "private"), ("172.16.4.4", "private"), ("172.32.0.1", "public"),
                   ("169.254.3.3", "private"), ("100.101.102.103", "private"), ("203.0.113.5", "public"), ("8.8.8.8", "public"), ("0.0.0.0", "any"), ("::", "any"), ("fd00::1", "private"),
                   ("2001:4860:4860::8888", "public"), ("localhost", "bad"), ("", "bad")):
    check(lan.classify(addr) == kind, f"classify {addr or '(empty)'} -> {kind}")
check(lan.pick_address(["203.0.113.5", "172.16.4.4", "192.168.1.20"]) == ("192.168.1.20", None), "pick: a home address first, never the public one")
check(lan.pick_address(["203.0.113.5"])[0] is None and "no private" in lan.pick_address(["203.0.113.5"])[1], "pick: only a public address: nothing to listen on, says so")
check(lan.pick_address(["192.168.1.20", "10.0.0.5"], "10.0.0.5") == ("10.0.0.5", None), "pick: the address the user names")
check(lan.pick_address(["192.168.1.20"], "8.8.8.8")[0] is None and lan.pick_address(["192.168.1.20"], "192.168.1.99")[0] is None, "pick: a public or foreign address is refused")
check(lan.check_bind("192.168.1.20", ["192.168.1.20"]) is None and lan.check_bind("127.0.0.1", []) is None, "bind: private and loopback are fine")
check("public" in (lan.check_bind("8.8.8.8", ["192.168.1.20"]) or "") and lan.check_bind("8.8.8.8", [], True) is None, "bind: a public address needs --allow-public")
check(lan.check_bind("0.0.0.0", ["192.168.1.20"]) is None, "bind: every card is fine when this PC has no public address")
check("public address (203.0.113.5)" in (lan.check_bind("0.0.0.0", ["192.168.1.20", "203.0.113.5"]) or "") and lan.check_bind("0.0.0.0", ["203.0.113.5"], True) is None, "bind: every card is refused when this PC has a public address")
check(lan.check_bind("nonsense", []) is not None, "bind: not an address: refused")
check(not any("KEYVALUE" in l for l in lan.risk_lines("http://192.168.1.20:8080", "/k/api.key")) and any("plain http" in l for l in lan.risk_lines("http://x:1")), "risk lines: plain words, no key")

# ---- the launcher's --lan on a dry run ----
model = os.path.join(TMP, "m.gguf"); open(model, "wb").write(b"GGUF" + b"\0" * 64)
def launcher(addrs, *args):
    env = dict(os.environ, MOE_CACHE_LAN_ADDRS=addrs, LLAMA_API_KEY="")
    env.pop("LLAMA_API_KEY")
    r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "server", model, "--dry-run", "--llama-server", sys.executable, *args],
                       capture_output=True, text=True, timeout=120, env=env, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr
rc, out = launcher("192.168.1.20", "--lan")
key = open(os.path.join(TMP, "cache", "moe-cache", "api.key")).read().strip() if os.path.exists(os.path.join(TMP, "cache", "moe-cache", "api.key")) else "?"
check(rc == 0 and "--host 0.0.0.0" in out and "LLAMA_API_KEY=(hidden)" in out, "launcher --lan: listens on every card, key on")
check("http://192.168.1.20:8080" in out and "plain http" in out and "api.key" in out, "launcher --lan: prints the exact address, the key file and the plain-words risk")
check(key != "?" and key not in out, "launcher --lan: the key itself is never printed")
rc, out = launcher("192.168.1.20,203.0.113.5", "--lan")
check(rc != 0 and "public address (203.0.113.5)" in out and "--allow-public" in out, "launcher --lan on a PC with a public address: refused")
rc, out = launcher("192.168.1.20,203.0.113.5", "--lan", "--allow-public")
check(rc == 0 and "--host 0.0.0.0" in out, "launcher --lan --allow-public: allowed")
rc, out = launcher("203.0.113.5", "--lan")
check(rc != 0 and "no private" in out, "launcher --lan with no home-network address: refused")
rc, out = launcher("192.168.1.20,10.0.0.5", "--lan", "--lan-address", "10.0.0.5")
check(rc == 0 and "http://10.0.0.5:8080" in out, "launcher --lan-address picks the address to tell")
rc, out = launcher("192.168.1.20", "--lan", "--", "--host", "192.168.1.20")
check(rc != 0 and "remove --host" in out, "launcher --lan plus --host: refused (one way to say it)")
rc, out = launcher("192.168.1.20", "--", "--host", "8.8.8.8")
check(rc != 0 and "public" in out, "launcher --host with a public address: refused without --allow-public")
rc, out = launcher("192.168.1.20", "--", "--host", "8.8.8.8", "--api-key", "k")
check(rc != 0, "launcher --host public with a key: still refused without --allow-public")
rc, out = launcher("192.168.1.20")
check(rc == 0 and "--host" not in out and "LAN" not in out, "launcher without --lan: no --host, nothing about LAN (default stays 127.0.0.1)")
rc, out = launcher("192.168.1.20", "--", "--host", "192.168.1.20")
check(rc == 0 and "LLAMA_API_KEY=(hidden)" in out and "other computers can reach it" in out and "WARNING" not in out, "launcher --host with a private address and no key: the stored key is forced (no warning needed)")
rc, out = launcher("192.168.1.20", "--", "--host", "0.0.0.0")
check(rc == 0 and "LLAMA_API_KEY=(hidden)" in out, "launcher --host 0.0.0.0 (no --lan): key forced too")
rc, out = launcher("192.168.1.20", "--", "--host", "127.0.0.1")
check(rc == 0 and "other computers can reach it" not in out and "LLAMA_API_KEY" not in out, "launcher --host 127.0.0.1: nothing forced")
rc, out = launcher("192.168.1.20", "--", "--host", "::")
check(rc != 0 and "IPv6" in out and "--allow-public" in out, "launcher --host :: is refused like a public address")
rc, out = launcher("192.168.1.20", "--allow-public", "--", "--host", "::")
check(rc == 0 and "LLAMA_API_KEY=(hidden)" in out, "launcher --host :: --allow-public: allowed, key forced")
check(lan.is_loopback_host("") and lan.is_loopback_host("127.0.0.1") and lan.is_loopback_host("localhost") and lan.is_loopback_host("::1") and not lan.is_loopback_host("0.0.0.0") and not lan.is_loopback_host("192.168.1.2") and not lan.is_loopback_host("::"), "is_loopback_host: only this PC counts")
check("IPv6" in (lan.check_bind("::", ["192.168.1.20"]) or "") and lan.check_bind("::", ["192.168.1.20"], True) is None, "bind: :: is refused without --allow-public even when no public IPv4 is known")

# ---- the page server with a fake remote client ----
TOKEN = "lan-token-1"
srv = ui.make_server(0, "llama-server", token=TOKEN, lan_hosts=("192.168.1.20:PORT",))
port = srv.server_address[1]
srv.RequestHandlerClass.lan_hosts = (f"192.168.1.20:{port}",)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{port}"
def req(path, body=None, headers=None, token=TOKEN):
    h = dict(headers or {})
    if token:
        h["X-MoeCache-Token"] = token
    if body is not None:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, h)
    try:
        with urllib.request.urlopen(r, timeout=30) as x:
            return x.status, x.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
check(req("/api/status", headers={"Host": f"192.168.1.20:{port}"})[0] == 200, "page: the LAN address with the right token is answered")
check(req("/api/status", headers={"Host": f"192.168.1.21:{port}"})[0] == 403 and req("/api/status", headers={"Host": f"evil.example:{port}"})[0] == 403, "page: any other Host (DNS rebinding, another address) is refused")
check(req("/api/status", headers={"Host": f"192.168.1.20:{port}"}, token=None)[0] == 403 and req("/api/status", headers={"Host": f"192.168.1.20:{port}"}, token="wrong")[0] == 403, "page: the LAN address without the token, or with a wrong one: 403")
check(ui.Handler.lan_hosts == () , "page: without --ui-lan no extra host is accepted (class default)")
H = srv.RequestHandlerClass
local = req("/api/llama", {"path": ""}, headers={"Host": f"127.0.0.1:{port}"})[0]
check(local == 400, f"page: from this PC the llama path route is reachable (400 for an empty path, not 403: {local})")
H.remote = lambda self: True                       # every request now looks like another computer's
HOST = {"Host": f"192.168.1.20:{port}"}
# every route the handler knows, read from its source: a new route that is not in the decision tables below fails this test until someone decides
import inspect, re
get_routes = set(re.findall(r'u\.path == "(/[^"]*)"', inspect.getsource(ui.Handler.do_GET))) | {"/", "/index.html"}
post_routes = set(re.findall(r'self\.path == "(/[^"]*)"', inspect.getsource(ui.Handler.do_POST)))
GET_ALLOWED = {"/", "/index.html", "/api/system", "/api/busy", "/api/pick", "/api/downloads", "/api/download", "/api/models", "/api/plan", "/api/status",
               "/api/bench", "/api/servers", "/api/chats", "/api/log"}
GET_DENIED = {"/api/key", "/api/agents", "/api/sessions", "/api/peek", "/api/share", "/api/ready", "/api/results", "/api/monitor", "/api/mini", "/api/risk", "/api/audit", "/api/audit/plan", "/api/suite/records", "/api/suite/compare", "/api/suite/status"}
POST_ALLOWED = {"/api/chat", "/api/chats/save"}
POST_DENIED = {"/api/chats/trash", "/api/folders", "/api/download/start", "/api/download/cancel", "/api/servers/start", "/api/servers/stop", "/api/servers/group", "/api/servers/plan", "/api/llama", "/api/start",
               "/api/pick/use", "/api/command", "/api/sessions/trash", "/api/bench", "/api/share", "/api/share/cancel", "/api/stop", "/api/connect", "/api/audit/start", "/api/audit/cancel", "/api/suite/import", "/api/suite/publish", "/api/suite/run", "/api/suite/cancel"}
check(get_routes == GET_ALLOWED | GET_DENIED, f"routes: every GET route has a remote decision (undecided: {sorted(get_routes ^ (GET_ALLOWED | GET_DENIED))})")
check(post_routes == POST_ALLOWED | POST_DENIED, f"routes: every POST route has a remote decision (undecided: {sorted(post_routes ^ (POST_ALLOWED | POST_DENIED))})")
check(set(ui.REMOTE_GET) == GET_ALLOWED and set(ui.REMOTE_POST) == POST_ALLOWED, "routes: the code's allow-lists are exactly the decided ones")
check(not (GET_ALLOWED & GET_DENIED) and not (POST_ALLOWED & POST_DENIED), "routes: no route is both allowed and denied")
H.remote = lambda self: True                       # every request now looks like another computer's
for path in sorted(GET_DENIED | {"/api/not-a-route", "//api/key", "/api/key/", "/api/key?x=1"}):
    st, out = req(path, headers=HOST)
    check(st == 403 and "not offered to other computers" in out, f"remote GET {path}: 403 ({st})")
for path in sorted(POST_DENIED | {"/api/not-a-route"}):
    st, out = req(path, {}, headers=HOST)
    check(st == 403 and "not offered to other computers" in out, f"remote POST {path}: 403 ({st})")
for path in sorted(GET_ALLOWED - {"/", "/index.html", "/api/plan"}):
    check(req(path, headers=HOST)[0] not in (403, 429), f"remote GET {path}: allowed")
for path in sorted(POST_ALLOWED):
    check(req(path, {}, headers=HOST)[0] not in (403, 429), f"remote POST {path}: allowed (400 for the empty body, not 403)")
st, out = req("/api/key", headers=HOST)
check(st == 403 and "error" in json.loads(out) and "key" not in json.loads(out), "remote: /api/key is 403 with the right token")
H.fails.clear()
codes = [req("/api/status", headers={"Host": f"192.168.1.20:{port}"}, token="guess%d" % i)[0] for i in range(25)]
check(codes[:20] == [403] * 20 and codes[20:] == [429] * 5, f"remote: 20 wrong tokens, then 429 ({set(codes[:20])}, {set(codes[20:])})")
check(req("/api/status", headers={"Host": f"192.168.1.20:{port}"})[0] == 429, "remote: even the right token waits out the minute")
H.fails.clear()
check(req("/api/status", headers={"Host": f"192.168.1.20:{port}"})[0] == 200, "remote: after the minute (cleared here) it works again")
del H.remote
check(req("/api/key", headers={"Host": f"127.0.0.1:{port}"})[0] in (200, 400), "this PC itself still reaches /api/key (the Copy button)")
H.fails.clear()
codes = [req("/api/status", token="guess%d" % i)[0] for i in range(25)]
check(set(codes) == {403}, "this PC: wrong tokens are never locked out (only remote addresses are)")

# ---- --ui-lan itself ----
def ui_lan(addrs, *args):
    os.environ["MOE_CACHE_LAN_ADDRS"] = addrs
    try:
        ui.ui(["--no-browser", "--ui-lan", *args])
        return None
    except SystemExit as e:
        return str(e)
    finally:
        os.environ.pop("MOE_CACHE_LAN_ADDRS", None)
check("public address" in (ui_lan("192.168.1.20,203.0.113.5") or ""), "--ui-lan on a PC with a public address: refused")
check("no private" in (ui_lan("203.0.113.5") or ""), "--ui-lan with no home-network address: refused")
page = open(os.path.join(HERE, "bin", "moe_cache_ui.py"), encoding="utf-8").read()
check('host = "0.0.0.0"' in page and 'host="127.0.0.1"' in page and "--ui-lan" in page, "page code: 127.0.0.1 is the default, 0.0.0.0 only inside the --ui-lan branch")

shutil.rmtree(TMP, ignore_errors=True)
print("\nFAILED: " + "; ".join(fails) if fails else "\nLAN tests passed")
sys.exit(1 if fails else 0)
