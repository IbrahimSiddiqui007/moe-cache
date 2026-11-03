#!/usr/bin/env python3
"""server.id of a --project folder must record the KV types the server really runs with (--fast / 128k context add -ctk q8_0 -ctv q8_0 after the
identity used to be written), or a state saved with q8_0 KV is restored into an f16 server.
   python3 tests/project_id_test.py"""
import os, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests"))
import launch_plan_test as lp
mc = lp.mc

tmp = tempfile.mkdtemp(prefix="project-id-")
model = os.path.join(tmp, "synth.gguf")
lp.make_synthetic_model(model)
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def sid(opts, extra=()):
    shutil_dir = os.path.join(tmp, "cache", "slots", "p")
    if os.path.isdir(shutil_dir):
        for f in os.listdir(shutil_dir):
            os.unlink(os.path.join(shutil_dir, f))
    started = []
    def hold():
        mc.run_launch_plan = lambda plan: started.append(plan) or 0
    lp.run_case("linux", ["--ram", "12", "--project", "p", *opts], list(extra), model, tmp, flag="--no-starter", call=hold)
    return open(os.path.join(shutil_dir, "server.id")).read().strip(), started[0]

real = mc.run_launch_plan
i, plan = sid(["--fast"])
check(i.endswith("-q8_0-q8_0"), "--fast: server.id says q8_0 KV (got ...%s)" % i[-24:])
check(i == mc.model_identity(model, plan.ctx, "q8_0", "q8_0"), "--fast: server.id equals the identity of the command that runs")
i, plan = sid([])
check(i.endswith("-f16-f16"), "no --fast: server.id says f16 KV")
i, plan = sid([], ["-ctk", "q4_0", "-ctv", "q4_0"])
check(i.endswith("-q4_0-q4_0"), "own -ctk/-ctv: recorded")
mc.run_launch_plan = real

# AUD-039: `project save` against a server that runs another model must not stamp the folder's identity on the state
import http.server, threading, json
props = {"model_path": "/models/other-model.gguf"}
class Fake(http.server.BaseHTTPRequestHandler):
    def _send(self, body):
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(body).encode())
    def do_GET(self):
        self._send(props)
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0)); self._send({"ok": True})
    def log_message(self, *a): pass
srv = http.server.HTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = "http://127.0.0.1:%d" % srv.server_address[1]
mc.CACHE_DIR = os.path.join(tmp, "cache2")
sd = os.path.join(mc.CACHE_DIR, "slots", "pp"); os.makedirs(sd)
open(os.path.join(sd, "server.id"), "w").write("my_model_q4_gguf-1234-abcd-c4096-f16-f16\n")
mc.say = lambda *a, **k: None
mc.project(["save", "pp", "--url", url, "--api-key", "x"])
check(not os.path.exists(os.path.join(sd, "pp.bin.id")), "project save: a server running another model does not get the folder's identity")
props["model_path"] = "/models/my_model_q4.gguf"
mc.project(["save", "pp", "--url", url, "--api-key", "x"])
check(open(os.path.join(sd, "pp.bin.id")).read().strip().startswith("my_model_q4_gguf-1234"), "project save: the matching model gets it")
os.unlink(os.path.join(sd, "pp.bin.id")); props.clear()
mc.project(["save", "pp", "--url", url, "--api-key", "x"])
check(os.path.exists(os.path.join(sd, "pp.bin.id")), "project save: a server that does not report its model keeps the old behaviour")
print("\nall project id tests passed" if not fails else f"\n{len(fails)} FAILED")
sys.exit(1 if fails else 0)
