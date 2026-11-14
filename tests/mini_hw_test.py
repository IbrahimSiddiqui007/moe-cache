#!/usr/bin/env python3
"""GUI-044 / GUI-046: the control page's mini window (ui/mini.html, /api/mini, /api/monitor?tail) and the hardware readings (bin/moe_cache_hw.py) without a model: readings are parsed
from recorded tool output, the real control-page server answers over HTTP, the mini page's script runs under node against a stub DOM (a recorded payload: tests/data/top_payload.json).
(The terminal monitor that used to share this file was removed before v0.3.8.)
   python3 tests/mini_hw_test.py"""
import http.server, json, os, re, shutil, stat, subprocess, sys, tempfile, threading, time, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-mini-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
os.environ["XDG_CONFIG_HOME"] = os.path.join(TMP, "config")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_hw as hw  # noqa: E402
import moe_cache_monitor as mm  # noqa: E402
import moe_cache_ui as ui  # noqa: E402


fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# ---- hardware readings from recorded output ----
nv = "NVIDIA GeForce RTX 4060 Laptop GPU, 96, 59, 116.42, 140.00, 7012, 8188, 4, 8\n"
g = hw.parse_nvidia(nv)
check(g["name"].endswith("4060 Laptop GPU") and g["util"] == 96 and g["temp"] == 59 and abs(g["power"] - 116.42) < 1e-9 and g["power_limit"] == 140 and g["mem_used"] == 7012 * 2**20 and g["pcie_gen"] == 4 and g["pcie_width"] == 8, f"nvidia-smi line parsed ({g})")
g2 = hw.parse_nvidia("Quadro, [N/A], 40, [N/A], [N/A], 100, 2000, [N/A], [N/A]\nBig, 90, 70, 200, 250, 1, 2, 5, 16\n")
check(g2["name"] == "Big" and g2["cards"] == 2 and hw.parse_nvidia("")["cards"] if hw.parse_nvidia("") else hw.parse_nvidia("") is None, "nvidia-smi: the busiest of two cards; [N/A] becomes None; no output is None")
g3 = hw.parse_nvidia("X, [N/A], [N/A], [N/A], [N/A], [N/A], [N/A], [N/A], [N/A]")
check(g3["util"] is None and g3["mem_used"] is None and g3["pcie_gen"] is None, "nvidia-smi: every field [N/A] gives Nones, not a crash")
root = os.path.join(TMP, "drm"); dev = os.path.join(root, "card1", "device")
os.makedirs(os.path.join(dev, "hwmon", "hwmon3"))
for fn, val in (("vendor", "0x1002"), ("mem_info_vram_total", str(16 * 2**30)), ("mem_info_vram_used", str(4 * 2**30)), ("gpu_busy_percent", "37"), ("current_link_speed", "16.0 GT/s PCIe"), ("current_link_width", "16"),
                ("hwmon/hwmon3/temp1_input", "61000"), ("hwmon/hwmon3/power1_average", "95000000"), ("hwmon/hwmon3/power1_cap", "304000000")):
    open(os.path.join(dev, fn), "w").write(val + "\n")
a = hw.amd_sysfs(root)
check(a and a["util"] == 37 and a["temp"] == 61 and a["power"] == 95 and a["power_limit"] == 304 and a["mem_total"] == 16 * 2**30 and a["pcie_gen"] == 4 and a["pcie_width"] == 16, f"AMD sysfs card read without admin ({a})")
check(hw.amd_sysfs(os.path.join(TMP, "nothing")) is None, "AMD sysfs: no card is None")
check(hw.parse_proc_stat_total("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 1 1 1") == (150, 1000), "/proc/stat: busy and total jiffies")
check(hw.parse_pid_stat("123 (llama (server) x) S 1 2 3 4 5 6 7 8 9 10 400 60 0 0 20") == 460, "/proc/PID/stat: utime + stime, even with ')' and spaces in the name")
check(hw.parse_pid_io("rchar: 5\nread_bytes: 4096\nwrite_bytes: 512\n") == (4096, 512), "/proc/PID/io: read and written bytes")
check(hw.parse_netstat("  TCP    0.0.0.0:8080     0.0.0.0:0     LISTENING       4242\n  TCP    127.0.0.1:8081   0.0.0.0:0   LISTENING   99\n  TCP 127.0.0.1:8080 1.2.3.4:5 ESTABLISHED 7", 8080) == 4242, "netstat: the pid listening on the port")
check(hw.parse_netstat("", 8080) is None and hw.pid_of_port(1) is None, "no listener: None")
if os.name != "nt":
    m = hw.HostMeter(os.getpid()); m.sample(); x = 0
    for _ in range(300000): x += 1
    time.sleep(0.1); r = m.sample()
    check(r["cpu_sys"] is not None and r["cpu_proc"] is not None and r["io_r"] is not None and r["cpu_proc"] >= 0, f"HostMeter on this process gives CPU and disk numbers ({r})")
    srv_sock = __import__("socket").socket(); srv_sock.bind(("127.0.0.1", 0)); srv_sock.listen(1)
    check(hw.pid_of_port(srv_sock.getsockname()[1]) == os.getpid(), "pid_of_port finds this process listening on a port (Linux /proc)")
    srv_sock.close()
else:       # Windows: the kernel32 calls (CI runs this on a real Windows machine)
    m = hw.HostMeter(os.getpid()); m.sample(); x = 0
    for _ in range(2000000): x += 1
    time.sleep(0.2); r = m.sample()
    check(r["cpu_sys"] is not None and r["cpu_proc"] is not None and r["cpu_proc"] > 0 and r["io_r"] is not None, f"HostMeter on this process gives CPU and disk numbers on Windows ({r})")
check(hw.logical_cpus() >= 1, "logical CPUs counted")

payload = json.load(open(os.path.join(HERE, "tests", "data", "top_payload.json")))
ui.LAUNCHER = [sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "server"]
# ---- the real control page: /api/mini, /api/monitor?tail, --url ----
TOKEN = "top-test-token"
srv = ui.make_server(0, "llama-server", token=TOKEN)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"
def get(path, tok=TOKEN):
    r = urllib.request.Request(base + path, headers={"X-MoeCache-Token": tok} if tok else {})
    try:
        with urllib.request.urlopen(r, timeout=20) as x:
            return x.status, x.read().decode(), x.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), None
st, body, ct = get("/api/mini")
check(st == 200 and ct.startswith("text/html") and "__TOKEN__" in body and "/api/monitor" in body, "/api/mini serves the small window with a token placeholder")
check(get("/api/mini", None)[0] == 403 and get("/api/mini", "wrong")[0] == 403, "/api/mini needs the page token")
rn = srv.RequestHandlerClass.runner
for i in range(12):
    rn.add_line(f"line {i}")
j = json.loads(get("/api/monitor?since=0&log=0&tail=3")[1])
check([l[4] for l in j["log"]] == ["line 9", "line 10", "line 11"], f"/api/monitor?tail=3 returns the newest three lines ({[l[4] for l in j['log']]})")
check(len(json.loads(get("/api/monitor?since=0&log=0")[1])["log"]) == 12, "without tail, all of them")
jd = json.loads(get("/api/monitor?since=0&log=0&req=0&layers=0")[1])
check(jd["schema"] == 1 and jd["layers_included"] is False and jd["layer_cols"] == [] and jd["requests"] == [], f"/api/monitor?req=0&layers=0 over HTTP: schema {jd['schema']}, no layer columns, only newer requests")
check(json.loads(get("/api/monitor?req=abc&layers=1")[1])["layers_included"] is True, "a req that is not a number is ignored, layers=1 keeps the layers")

# ---- a browser that goes away before the answer (the user's WinError 10053 traceback) ----
import io, socket, contextlib
err = io.StringIO()
with contextlib.redirect_stderr(err):
    for i in range(40):
        c = socket.create_connection(("127.0.0.1", srv.server_address[1]))
        c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00" if os.name != "nt" else b"\x01\x00\x00\x00\x00\x00")
        c.sendall(f"GET /api/monitor?since=0&log=0 HTTP/1.1\r\nHost: 127.0.0.1\r\nX-MoeCache-Token: {TOKEN}\r\n\r\n".encode())
        c.close()                       # reset at once: the server's answer meets a dead connection
    time.sleep(1.0)
check("Traceback" not in err.getvalue(), "a connection dropped before the answer leaves no traceback on the console" + ("" if "Traceback" not in err.getvalue() else ": " + err.getvalue()[-200:]))

# ---- the main page and the mini page ----
page = open(os.path.join(HERE, "ui", "index.html"), encoding="utf-8").read()
check('id="monpop"' in page and "/api/mini" in page and "popup=yes,width=360,height=480" in page, "the page has the Mini button and opens a 360 x 480 pop-up")
mini = open(os.path.join(HERE, "ui", "mini.html"), encoding="utf-8").read()
check("__TOKEN__" in mini and "http://" not in mini.replace("http://www.w3.org", "") and "https://" not in mini, "mini.html: token placeholder, no outside address")
node = shutil.which("node")
if node:
    js = re.search(r"<script>(.*)</script>", mini, re.S).group(1)
    pj = os.path.join(TMP, "mini.js"); open(pj, "w", encoding="utf-8").write(js)
    o = subprocess.run([node, "--check", pj], capture_output=True, text=True, timeout=60)
    check(o.returncode == 0, "mini.html's script parses (node --check)" + ("" if o.returncode == 0 else ": " + o.stderr[:300]))
    sample_payload = dict(payload, samples=[dict(s_, t=s_["t"]) for s_ in payload["samples"]])
    harness = ("const els = {}; const mk = id => els[id] || (els[id] = {textContent: '', innerHTML: '', className: '', style: {}, parentNode: {className: ''}});\n"
               "const document = {getElementById: mk, title: ''}; const setInterval = () => 1; let seen = null;\n"
               "const fetch = async (u, o) => { seen = [u, o.headers['X-MoeCache-Token']]; return {ok: true, json: async () => (" + json.dumps(sample_payload) + ")}; };\n"
               + js.replace('const TOKEN = "__TOKEN__";', 'const TOKEN = "tok-1";') + "\n"
               "setTimeout(() => console.log(JSON.stringify({seen, dec: els.dec.textContent, model: els.model.textContent, state: els.state.textContent, ctx: els['v-ctx'].textContent, hit: els['v-hit'].textContent, ram: els['v-ram'].textContent,\n"
               "  ssd: els['v-ssd'].textContent, vram: els['v-vram'].textContent, spark: els.spark.innerHTML.includes('polyline'), log: els.log.innerHTML, title: document.title, ramw: els['b-ram'].style.width})), 50);\n")
    hf = os.path.join(TMP, "minirun.js"); open(hf, "w", encoding="utf-8").write(harness)
    o = subprocess.run([node, hf], capture_output=True, text=True, encoding="utf-8", timeout=30)
    try:
        r_ = json.loads(o.stdout.strip().splitlines()[-1])
    except Exception:
        r_ = None; print(o.stdout[-300:], o.stderr[-600:])
    check(r_ is not None and r_["seen"][0].startswith("/api/monitor?since=0&log=0&tail=5") and r_["seen"][1] == "tok-1", "mini: polls /api/monitor with the token header and tail=5")
    check(r_ is not None and r_["model"] == payload["totals"]["model"] and r_["state"].startswith("ready") and r_["ctx"] == "9830 / 32768" and r_["hit"] == "82.0 %" and r_["ssd"] == "2.00 GB/s", f"mini: model, state, context, hit rate, SSD ({r_ and (r_['state'], r_['ctx'], r_['hit'], r_['ssd'])})")
    check(r_ is not None and r_["ram"] == "12.0 / 28.0 GiB" and r_["vram"] == "6.0 / 8.0 GiB" and r_["spark"] and 'class="warn"' in r_["log"] and 'class="error"' in r_["log"], "mini: RAM against the cap, VRAM, the sparkline and the warning line")
    check(r_ is not None and r_["dec"].replace(".", "").isdigit() and "tok/s" in r_["title"], f"mini: decode tok/s in the big number and the window title ({r_ and (r_['dec'], r_['title'])})")
else:
    print("  skip mini page checks: node not found")

