#!/usr/bin/env python3
"""on-demand --concurrent (MM1 prototype): planning (pure) and a live pool of two FAKE model servers (no model, no llama-server, a few seconds).
   python3 tests/concurrent_test.py
Checks: shares add up under ONE cap; core slices are disjoint; a model that fits runs stock; floors that do not fit fall back to swap; two servers are alive
at once and answer by the request's model name; the share options reach each command line, a user's own --ram / -C stay; default (off) still swaps."""
import http.client, json, os, socket, sys, tempfile, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="cc-test-")
os.environ["XDG_CACHE_HOME"] = TMP
os.environ["XDG_CONFIG_HOME"] = TMP
sys.path.insert(0, os.path.join(ROOT, "bin"))
import moe_cache as mc                 # noqa: E402
import moe_cache_on_demand as od       # noqa: E402
import moe_cache_concurrent as cc      # noqa: E402

GIB = 2**30
fails = []
def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)

# ---- planning
cores = list(range(0, 16))
sl = cc.core_slices(cores, [1, 1, 2])
check(sum(len(x) for x in sl) == 16 and not set(sl[0]) & set(sl[1]) and not set(sl[1]) & set(sl[2]) and len(sl[2]) > len(sl[0]), f"core slices cover the cores once, weights respected: {[len(x) for x in sl]}")
check(cc.core_slices([0], [1, 1]) is None, "fewer cores than models: no slices")
big = {"name": "big-moe", "size": 40 * GIB, "kind": "moe", "floor": 4 * GIB}
ocr = {"name": "ocr", "size": 3 * GIB, "kind": "dense"}
g = cc.group_plan([big, ocr], 28 * GIB, 16 * GIB, cores)
check(g["ok"] and g["models"]["ocr"]["engine"] == "stock" and g["models"]["big-moe"]["engine"] == "plugin", f"a dense model that fits runs stock, the big MoE keeps the plugin: {g['models']['ocr']['engine']}/{g['models']['big-moe']['engine']}")
tot = sum(float(m["launcher"][1]) * 1e9 for m in g["models"].values())
check(tot <= 28 * GIB * 1.001, f"RAM shares add up to <= the cap ({tot / GIB:.1f} of 28 GiB)")
vr = sum(float(m["launcher"][3]) for m in g["models"].values())
check(vr < 16, f"VRAM shares leave the reserve ({vr:.1f} of 16 GiB)")
masks = [int(m["llama"][1], 16) for m in g["models"].values()]
check(masks[0] & masks[1] == 0, "the two CPU masks do not overlap")
check(not cc.group_plan([dict(big, size=60 * GIB), dict(ocr, size=30 * GIB)], 28 * GIB, 16 * GIB, cores)["ok"], "dense model + floors above the cap: no split (swap)")
check(cc.group_plan([big, ocr], 28 * GIB, None, None)["models"]["ocr"]["llama"] == [], "unknown cores / VRAM: no pinning, no --vram")

# ---- live pool with fake servers
FAKE = os.path.join(TMP, "fake_server.py")
open(FAKE, "w").write(r'''
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1]); key = os.environ["LLAMA_API_KEY"]; me = os.environ.get("FAKE_ID", "x")
open(os.path.join(os.environ["XDG_CACHE_HOME"], "argv-" + me), "w").write(json.dumps(sys.argv[1:]))
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": return self.send(200, {"status": "ok"})
        self.send(200, {"object": "list", "data": [{"id": me, "object": "model"}]})
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"])); self.send(200, {"choices": [{"message": {"content": "from " + me}}]})
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]
lines = []
def mk(name, args=()):
    return od.Backend({"model": os.path.join(TMP, name + ".gguf"), "args": list(args), "env": {"FAKE_ID": name}}, launcher=[sys.executable, FAKE], idle_min=5, start_timeout=30,
                      log=lambda m: lines.append(m), release_min=5)
sizes = [{"name": "A", "size": 40 * GIB, "kind": "moe", "floor": 4 * GIB}, {"name": "B", "size": 3 * GIB, "kind": "dense"}]
def ask(port, model):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("POST", "/v1/chat/completions", json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]}), {"Content-Type": "application/json", "Authorization": "Bearer " + mc.stored_key()})
    r = c.getresponse(); body = r.read().decode(); c.close(); return r.status, body

for name in ("A", "B"):
    open(os.path.join(TMP, name + ".gguf"), "wb").close()
A, B = mk("A", ["--ctx", "8192"]), mk("B", ["--ram", "5"])
pool = od.Pool([A, B])
ok = pool.go_concurrent(cap=28 * GIB, vram=16 * GIB, cores=cores, sizes=sizes)
check(ok and pool.concurrent, "go_concurrent: split accepted")
check(A.release <= 30 and B.release <= 30, "idle release shortened to 30 s")
port = free_port()
srv = od.serve(pool, port)
threading.Thread(target=srv.serve_forever, daemon=True).start()
st1, b1 = ask(port, "A"); st2, b2 = ask(port, "B")
check(st1 == 200 and "from A" in b1 and st2 == 200 and "from B" in b2, "both models answer by name")
check(A.running() and B.running(), "BOTH servers are alive at the same time (no swap)")
argA = json.load(open(os.path.join(TMP, "argv-A"))); argB = json.load(open(os.path.join(TMP, "argv-B")))
check("--ram" in argA and "--vram" in argA and "-C" in argA and "--cpu-strict" in argA and argA.index("--") < argA.index("-C"), f"A's command has its share and core slice: {argA[2:]}")
check(argB.count("--ram") == 2 and argB[argB.index("--ram", argB.index("--ram") + 1) + 1] == "5", f"B: the user's own --ram 5 comes last, so it wins: {argB[2:]}")
mA, mB = int(argA[argA.index("-C") + 1], 16), int(argB[argB.index("-C") + 1], 16)
check(mA & mB == 0, "the two servers pin to disjoint cores")
check(pool.state == "ready", "pool state ready")
pool.stop("test"); srv.shutdown(); srv.server_close()
check(not A.running() and not B.running(), "pool stop ends both")

# user's own -C kept; swap fallback; default off
A2 = mk("A", ["--ram", "10", "--", "-C", "0x3"]); B2 = mk("B")
p2 = od.Pool([A2, B2]); p2.go_concurrent(cap=28 * GIB, vram=None, cores=cores, sizes=sizes)
cmd = A2.cmd(1234)
check(cmd.count("-C") == 1 and "0x3" in cmd, "a user's own -C is not overridden by the slice")
A3, B3 = mk("A"), mk("B")
p3 = od.Pool([A3, B3]); res = p3.go_concurrent(cap=20 * GIB, vram=None, cores=cores, sizes=[dict(sizes[0], size=50 * GIB), dict(sizes[1], size=30 * GIB)])
check(res is False and not p3.concurrent and A3.share_args == [], "floors above the cap: falls back to one at a time")
p4 = od.Pool([mk("A"), mk("B")])
check(not p4.concurrent and p4.bs[0].share_args == [] and "--ram" not in p4.bs[0].cmd(1), "default: not concurrent, command line without a share")
print("FAILED: " + "; ".join(fails) if fails else "all ok")
sys.exit(1 if fails else 0)
