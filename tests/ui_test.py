#!/usr/bin/env python3
"""Control page (GUI1, bin/moe_cache_ui.py) without a model: fake GGUF files, a mocked planner and a fake launcher that answers /health and
prints the lines llama-server and the plugin print (timings, 'moe-cache-stats {json}'). Linux and Windows; standard library only.
   python3 tests/ui_test.py"""
import json, os, socket, sys, tempfile, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-ui-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")          # before the import: the launcher reads it once
os.environ["XDG_CONFIG_HOME"] = os.path.join(TMP, "config")        # 'connect opencode --write' writes here, not into the real config
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_ui as ui  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# ---- fake models ----
models = os.path.join(TMP, "models")
os.makedirs(os.path.join(models, "sub")); os.makedirs(os.path.join(models, ".hidden"))
def touch(p, n=1024):
    with open(os.path.join(models, p), "wb") as f:
        f.write(b"GGUF" + b"\0" * (n - 4))
touch("Small-MoE-Q4_K_M.gguf")
touch("sub/Big-UD-IQ2_XXS-00001-of-00002.gguf", 2000); touch("sub/Big-UD-IQ2_XXS-00002-of-00002.gguf", 3000)
touch("mmproj-Small-F16.gguf"); touch(".hidden/Secret-Q8_0.gguf"); touch("notes.txt")
ui.default_folders = lambda: [models]
ui.ollama_roots = lambda: []

found = ui.find_models([models])
names = [m["name"] for m in found]
check(names == ["Big-UD-IQ2_XXS", "Small-MoE-Q4_K_M"], f"model list: split model once, projector and hidden folders left out ({names})")
big = next(m for m in found if m["name"].startswith("Big"))
small = next(m for m in found if m["name"].startswith("Small"))
check(big["size"] == 5000 and big["quant"] == "UD-IQ2_XXS" and small["quant"] == "Q4_K_M", "sizes add up over the parts; quant from the name")

# ---- the plan and its reasons (planner mocked) ----
PLAN = {"arch": "qwen3moe", "layers": 48, "experts": 128, "used": 8, "n_cpu_moe": 40, "cache_GiB": 6.0, "cpu_experts_GiB": 18.0,
        "free_vram_GiB": 7.5, "note": ""}
TUNE = {"n_cpu_moe": 38, "ub": 1024, "gpu_cache_mib": 0, "why": ["10 layers of experts stay on the GPU", "micro-batch 1024"]}
ui.run_planner = lambda path, ctx, ram, vram, kv="f16": dict(PLAN)
ui.run_tuner = lambda path, ctx, vram, kv_q8: dict(TUNE)

p = ui.plan_model(small["path"], ram="2")
check(p["kind"] == "stock" and "fits in RAM" in p["verdict"], f"model + 1.5 GB under the RAM budget: stock ({p['verdict']})")
check(any("RESULTS 44" in r for r in p["reasons"]), "stock: says why stock is faster")
p = ui.plan_model(small["path"], ram="1")
check(p["kind"] == "cache" and p["verdict"] == "does not fit: moe-cache exact", f"over the budget: moe-cache exact ({p['verdict']})")
check(any("10 of 48 layers keep their experts on the GPU" in r for r in p["reasons"]), "reasons: GPU layers from the tuner (as the launcher)")
check(any("holds 33 %" in r for r in p["reasons"]), "reasons: share of the CPU experts the cache holds")
check(any(r.startswith("tuner: ") for r in p["reasons"]), "reasons: the tuner's own lines")
p = ui.plan_model(small["path"], ram="2", mode="fast")
check(p["kind"] == "cache" and "needs the plugin" in p["verdict"], f"--fast keeps the plugin even when the model fits ({p['verdict']})")
TUNE["n_cpu_moe"] = 0
p = ui.plan_model(small["path"], ram="1")
check(p["kind"] == "gpu", f"every expert layer on the GPU ({p['verdict']})")
TUNE["n_cpu_moe"] = 38
ui.run_planner = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no expert tensors (blk.N.ffn_{gate,up,down,gate_up}_exps) found: this does not look like a MoE GGUF"))
p = ui.plan_model(small["path"], ram="1")
check(p["kind"] == "error" and any("not a MoE model" in r for r in p["reasons"]), "planner error: shown with the planner's own words")
ui.run_planner = lambda path, ctx, ram, vram, kv="f16": dict(PLAN)

args = ui.server_args({"model": "m.gguf", "port": 8123, "ctx": 32768, "ram": 12, "mode": "fast", "agent": True})
check(args[:2] == ["m.gguf", "--port"] and "--fast" in args and "--agent" in args and args[-2:] == ["--", "--metrics"] and "12.0" in args,
      "launcher arguments: " + " ".join(args))
check("KEY" in ui.shown_command({"model": "m.gguf", "api_key": "secret"}) and "secret" not in ui.shown_command({"model": "m.gguf", "api_key": "secret"}),
      "the shown command hides the API key")

# --dry-run must reach the launcher, not llama-server (it went after the -- once)
seen = []
ui.subprocess.run, real_run = (lambda cmd, **kw: seen.append(cmd) or type("R", (), {"stdout": "", "stderr": ""})()), ui.subprocess.run
ui.dry_run({"model": "m.gguf"})
ui.subprocess.run = real_run
check(seen and seen[0].index("--dry-run") < seen[0].index("--"), "--dry-run goes to the launcher, before the --")

# ---- a fake launcher: serves /health and /slots, prints the plugin's lines, runs until stopped ----
fake = os.path.join(TMP, "fake_launcher.py")
with open(fake, "w") as f:
    f.write(r'''
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = {"/health": {"status": "ok"}, "/slots": [{"id": 0, "is_processing": False}], "/v1/models": {"data": [{"id": "fake-moe"}]},
                "/props": {"default_generation_settings": {"n_ctx": 65536}}}.get(self.path)
        self.send_response(200 if body is not None else 404); self.end_headers(); self.wfile.write(json.dumps(body).encode())
srv = HTTPServer(("127.0.0.1", port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print("moe-cache: experts of 38 layers stay on the CPU and are managed by the cache; profile /tmp/x.usage; plugin loaded by preload", flush=True)
time.sleep(0.5)
print("prompt eval time =     100.00 ms /    50 tokens (    2.00 ms per token,   500.00 tokens per second)", flush=True)
print("       eval time =    1000.00 ms /    25 tokens (   40.00 ms per token,    25.00 tokens per second)", flush=True)
for _ in range(2):
    print("0.01.002.003 W moe-cache: warning: the expert cache shrank from 10.0 to 8.0 GiB because the rest of llama-server now uses 3.0 GiB", flush=True)
print('moe-cache-stats {"hits":90,"misses":10,"read_bytes":0,"resident_bytes":6000000000,"budget":7000000000}', flush=True)
time.sleep(1.0)
print('moe-cache-stats {"hits":170,"misses":30,"read_bytes":50000000,"resident_bytes":6500000000,"budget":7000000000}', flush=True)
while True:
    time.sleep(1)
''')
ui.LAUNCHER = [sys.executable, fake]

srv = ui.make_server(0, "llama-server")
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"

def req(path, body=None, headers=None):
    h = dict(headers or {})
    if body is not None:
        h.setdefault("Content-Type", "application/json")
    r = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, h)
    try:
        with urllib.request.urlopen(r, timeout=30) as x:
            return x.status, x.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

st, page = req("/")
check(st == 200 and "<title>moe-cache</title>" in page and "/api/status" in page, "GET / serves the page")
st, body = req("/api/models")
check(st == 200 and len(json.loads(body)["models"]) == 2, "GET /api/models")
st, body = req("/api/plan?model=" + urllib.request.quote(small["path"]) + "&ram=1&ctx=8192&mode=exact")
check(st == 200 and json.loads(body)["kind"] == "cache", "GET /api/plan")
st, _ = req("/api/plan?model=" + urllib.request.quote(os.path.join(TMP, "elsewhere.gguf")))
check(st == 400, "a model that is not in the list is refused")
st, _ = req("/api/status", headers={"Host": "attacker.example:80"})
check(st == 403, "a foreign Host header is refused (DNS rebinding)")
st, _ = req("/api/stop", {}, headers={"Origin": "http://attacker.example"})
check(st == 403, "a POST from a foreign origin is refused")
st, _ = req("/api/stop", {}, headers={"Content-Type": "text/plain"})
check(st == 403, "a POST that is not JSON is refused (no simple cross-site form posts)")
check(srv.server_address[0] == "127.0.0.1", "listens on 127.0.0.1 only")

with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
st, body = req("/api/start", {"model": small["path"], "port": port, "ram": "1", "ctx": 8192, "mode": "exact", "agent": True, "api_key": "sekrit-123"})
check(st == 200, f"POST /api/start ({body[:120]})")
st, _ = req("/api/start", {"model": small["path"], "port": port + 1})
check(st == 400, "a second start while one runs is refused")
deadline, status = time.time() + 30, {}
while time.time() < deadline:
    status = json.loads(req("/api/status")[1])
    if status["state"] == "ready" and status["cache"].get("hit_rate") is not None:
        break
    time.sleep(0.3)
check(status.get("state") == "ready", f"the server becomes ready ({status.get('state')})")
check(status.get("mode") == "moe-cache exact", f"mode from the launcher's log ({status.get('mode')})")
check(status.get("last", {}).get("decode_tps") == 25.0 and status["last"].get("prefill_tps") == 500.0, "decode and prefill tok/s from the timing lines")
c = status.get("cache", {})
check(abs((c.get("hit_rate") or 0) - 0.8) < 1e-9, f"recent hit rate from two snapshots: 80/100 ({c.get('hit_rate')})")
check(abs((c.get("hit_rate_total") or 0) - 0.85) < 1e-9 and c.get("budget") == 7e9, "total hit rate and the cache budget")
check((c.get("read_mbs") or 0) > 0, f"SSD reads MB/s from read_bytes ({c.get('read_mbs')})")
check(f"--port {port}" in status.get("command", ""), "the status shows the launcher command")
w = status.get("warnings", [])
check(len(w) == 1 and w[0]["count"] == 2 and w[0]["line"].startswith("moe-cache: warning: the expert cache shrank"),
      f"the shrink warning is lifted out of the log, once, counted, without llama-server's prefix ({w})")
check("sekrit" not in json.dumps(status), "the API key is not in the status")

# connect buttons: preview, then write (into the temporary XDG_CONFIG_HOME)
st, body = req("/api/connect", {"agent": "opencode", "write": False})
j = json.loads(body)
check(st == 200 and j["rc"] == 0 and "fake-moe" in j["output"] and not os.path.exists(os.path.join(TMP, "config", "opencode", "opencode.json")),
      "Preview OpenCode: shows the block, writes nothing")
st, body = req("/api/connect", {"agent": "opencode", "write": True})
j = json.loads(body)
cfg = os.path.join(TMP, "config", "opencode", "opencode.json")
check(st == 200 and j["rc"] == 0 and j["file"] == cfg and os.path.isfile(cfg), f"Connect OpenCode writes {cfg} ({j['output'][-120:]})")
check('"moe-cache/fake-moe"' in (j["written"] or "") and f"127.0.0.1:{port}/v1" in j["written"], "and shows the provider block now in the file")
check("sekrit" not in body and "sekrit-123" in open(cfg).read(), "the key is written to the file but never shown on the page")
st, body = req("/api/connect", {"agent": "hermes", "write": False})
check(st == 200 and "hermes config set" in json.loads(body)["output"], "Preview Hermes: the 'hermes config set' commands")
st, _ = req("/api/connect", {"agent": "bash"})
check(st == 400, "only opencode and hermes")

lines = [l for _, l in json.loads(req("/api/log?since=0")[1])["lines"]]
check(any("eval time" in l for l in lines) and not any(l.startswith("moe-cache-stats") for l in lines), "the log has the server lines, not the stats lines")
st, _ = req("/api/stop", {})
deadline = time.time() + 30
while time.time() < deadline and status["state"] not in ("stopped", "exited"):
    status = json.loads(req("/api/status")[1]); time.sleep(0.3)
check(status["state"] == "stopped", f"POST /api/stop stops it ({status['state']}, exit code {status['rc']})")
srv.shutdown()

# Windows without nvidia-smi: VRAM through the launcher's helpers (mocked: they need Windows and a llama-server)
mc = ui.mc
saved = mc.pick_gpu, mc.list_devices, mc.win_dedicated_vram
mc.list_devices = lambda ls: "listing"
mc.pick_gpu = lambda listing: ("Vulkan1", "AMD Radeon RX 7900 XT", 18000)
mc.win_dedicated_vram = lambda: {"AMD Radeon RX 7900 XT": 20 * 2**30, "AMD Radeon(TM) Graphics": 512 * 2**20}
g = ui.windows_gpu_once("llama-server")
check(g == {"name": "AMD Radeon RX 7900 XT", "free": 18000 * 2**20, "total": 20.0 * 2**30, "integrated": False}, f"discrete AMD GPU: free and total ({g})")
mc.pick_gpu = lambda listing: ("Vulkan0", "AMD Radeon(TM) Graphics", 16000)
g = ui.windows_gpu_once("llama-server")
check(g["integrated"] and g["total"] is None and "integrated" in g["name"], "integrated GPU: marked, no dedicated total")
mc.pick_gpu, mc.list_devices, mc.win_dedicated_vram = saved
ui.vram_now, ui._GPU["gpu"] = (lambda: None), {"name": "AMD Radeon RX 7900 XT", "free": 15 * 2**30, "total": 20.0 * 2**30, "integrated": False}
si = ui.system_info()
check(si["gpu"] == "AMD Radeon RX 7900 XT" and si["vram_free"] == 15 * 2**30 and si["vram_total"] == 20 * 2**30, "the status uses it when nvidia-smi and sysfs have nothing")

print("ui test:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
