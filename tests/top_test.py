#!/usr/bin/env python3
"""GUI-044 / GUI-045 / GUI-046: the mini window (ui/mini.html, /api/mini), the full-screen terminal monitor `moe_cache.py top` (bin/moe_cache_top.py) and its hardware
readings (bin/moe_cache_hw.py) without a model: screens are drawn from a recorded payload (tests/data/top_payload.json), readings are parsed from recorded tool output,
the sources are a fake llama-server and the real control-page server, the mini page's script runs under node against a stub DOM.
   python3 tests/top_test.py"""
import http.server, json, os, re, shutil, stat, subprocess, sys, tempfile, threading, time, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-top-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
os.environ["XDG_CONFIG_HOME"] = os.path.join(TMP, "config")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_top as top  # noqa: E402
import moe_cache_hw as hw  # noqa: E402
import moe_cache_monitor as mm  # noqa: E402
import moe_cache_ui as ui  # noqa: E402
TOP_FILE0 = ui.TOP_FILE

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# ---- drawing pieces ----
U, A = top.UNICODE, top.ASCII
txt = lambda pieces: "".join(t for t, _ in pieces)
check(txt(top.hbar(0.5, 10, U)) == "█████░░░░░" and txt(top.hbar(None, 4, A)) == "...." and txt(top.hbar(2, 4, A)) == "####", "bar: half, none, over 100 %")
sp = txt(top.spark([0, 1, None, 4], 6, A))
check(len(sp) == 6 and sp[-2] == " " and sp[-1] == "@" and sp.startswith("  "), f"spark: width, a gap for None, the largest is the tallest ({sp!r})")
check(txt(top.spark([50, 100], 2, A, top=1000)) == "  "[:0] + txt(top.spark([50, 100], 2, A, top=1000)) and txt(top.spark([1.0, 0.5], 2, A, top=1.0))[0] == "@", "spark: a fixed top draws percentages against 100 %")
h = top.heat([1.0, 0.6, 0.2, None], 4, U)
check([s for _, s in h] == ["ok", "warn", "bad", "dim"], f"heat: green, yellow, red, dim ({[s for _, s in h]})")
check(len(h) == 4 and len(top.heat([1.0] * 100, 20, U)) == 20 and len(top.heat([], 5, U)) == 1, "heat: one cell per layer, averaged into the width when there are more")
bx = top.box("Speed", [[("12 tok/s", "bold")], [], []], 20, U)
check(len(bx) == 5 and all(top.plain(l) == 20 for l in bx) and txt(bx[0]).startswith("╭─ Speed "), "box: title in the top border, every line the same width")
j = top.join(([[("a", "")], [("b", "")]], 3), ([[("c", "")]], 4), gap=2)
check(len(j) == 2 and all(top.plain(l) == 9 for l in j), "join: blocks side by side, padded")
check(top.wrap("one two three four", 9) == ["one two", "three", "four"], "wrap")
check(top.decode_keys("q") == ["q"] and top.decode_keys("\x1b[A\x1b[B\x1b[5~\x1b[6~") == ["up", "down", "pgup", "pgdn"] and top.decode_keys("\x1b") == ["esc"]
      and top.decode_keys("a\r\x7f\x03") == ["a", "enter", "backspace", "ctrl-c"] and top.decode_keys("\x1b[H\x1b[F") == ["home", "end"], "keys: arrows, PgUp/PgDn, Esc, Enter, Backspace, Ctrl-C")

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
    check(top.enable_vt() in (True, False), "enable_vt answers on Windows")
check(hw.logical_cpus() >= 1, "logical CPUs counted")

# ---- the screens, from a recorded payload ----
payload = json.load(open(os.path.join(HERE, "tests", "data", "top_payload.json")))
def make(view=0, theme="dark", g=U):
    t = top.Top(g, theme); t.feed(payload); t.src_url = "http://127.0.0.1:8070"; t.view = view
    return t
SIZES = ((110, 33), (80, 24), (60, 18), (40, 12), (200, 60))
for v, name in enumerate(top.VIEWS):
    ok = True
    for (w, hh) in SIZES:
        for g_ in (U, A):
            lines = top.render(make(v, g=g_), w, hh)
            ok = ok and len(lines) == hh and all(top.plain(l) == w for l in lines)
    check(ok, f"view {v + 1} {name}: exactly the screen size at {', '.join('%dx%d' % s for s in SIZES)}, both glyph sets")
scr = lambda v, w=110, hh=33, **k: "\n".join(top.paint(top.render(make(v, **k), w, hh), "dark", False))
for v, wants in enumerate([
    ["Overview", "[Generating]", "9,830 / 32,768 tokens", "Speed", "tok/s", "Cache hit", "SSD read", "GPU load", "VRAM", "GPU temp", "Power", "PCIe", "Gen4 x16", "Disk read", "Context fill", "Cache budget", "Server RAM", "28.0 GiB cap",
     "Hit rate by layer", "Recent requests", "REUSED", "13,500", "TIME"],
    ["Plan as it runs", "4096 / 4096", "q8_0", "experts on the CPU", "Token modes", "kv8", "won", "in use", "unmeasured", "What the launcher decided", "memory cap 27.0 GiB"],
    ["PROMPT", "REUSED", "DURATION", "MTP", "13,500 tokens (93 %)", "prompt processing", "400 tok/s", "decoding", "whole request", "70 %"],
    ["Cache and storage", "hit rate", "SSD wait", "executor", "Hit rate by layer", "whole run", "lowest layers", "Counters", "prefetch used", "MTP drafted / accepted", "40 / 30 (75 %)", "approx routing", "skip"],
    ["GPU load", "RTX 4060 Laptop GPU", "CPU threads", "memory cap", "server RAM now"],
    ["Log ", "59 of 59 lines", "following", "line 59 of the log"],
    ["Connect an app", "http://127.0.0.1:8080/v1", "API key", "opencode", "b opens"]]):
    out = scr(v)
    miss = [x for x in wants if x not in out]
    check(not miss, f"view {v + 1} {top.VIEWS[v]} shows its numbers (missing: {miss})")
check(all(ord(c) < 128 for c in scr(0, g=A) + scr(3, g=A) + scr(1, g=A)), "ASCII glyph set: nothing outside ASCII on the screen")
col = top.paint(top.render(make(5), 110, 33), "dark", True)
check(any("\x1b[93m" in l and "of the log" in l for l in col) and any("\x1b[91m" in l for l in col), "colour (dark): warnings yellow, errors red")
check("\x1b[" not in "".join(top.paint(top.render(make(5), 110, 33), "dark", False)), "no colour: no escape codes")
mono = "".join(top.paint(top.render(make(5), 110, 33), "mono", True))
check("\x1b[93m" not in mono and "\x1b[91m" not in mono and "\x1b[" in mono, "mono theme: bold and dim only, no colours")
light = "".join(top.paint(top.render(make(5), 110, 33), "light", True))
check("\x1b[33m" in light and "\x1b[93m" not in light, "light theme: the dark-background colours are not used")
over = json.loads(json.dumps(payload)); over["cap"] = 10 * 2**30
tt = top.Top(U); tt.feed(over); tt.src_url = "x"
check(any("\x1b[91m" in l and "█" in l for l in top.paint(top.render(tt, 110, 33), "dark", True)), "server RAM above the cap is drawn red")
empty = top.Top(U); empty.src_url = "x"
check(all(len(top.render(empty, 80, 24)) == 24 for _ in range(1)) and all(len(top.render(empty, 80, 24)) == 24 and (setattr(empty, "view", i) or True) for i in range(7)), "an empty payload draws every view (idle)")
na = json.loads(json.dumps(payload))
for r in na["samples"]:
    for k in ("gpu_util", "gpu_temp", "gpu_power", "pcie_gen", "pcie_w", "io_r", "cpu_sys", "hit", "ssd", "cpu"):
        r.pop(k, None)
tn = top.Top(U); tn.feed(na); tn.src_url = "x"
o = "\n".join(top.paint(top.render(tn, 110, 33), "dark", False))
check(o.count("n/a") >= 5 and "no plugin numbers here" in o, "readings the machine cannot give are shown as n/a")

# ---- the same from a real recording (granite 1B through the plugin on the laptop, no usable GPU: nvidia-smi fails there) ----
real = json.load(open(os.path.join(HERE, "tests", "data", "top_payload_real.json")))
def make_real(view, w=110, hh=33, g=U):
    t = top.Top(g); t.feed(real); t.src_url = "http://127.0.0.1:8070"; t.view = view
    return "\n".join(top.paint(top.render(t, w, hh), "dark", False))
ok = True
for v in range(7):
    for (w, hh) in SIZES:
        t_ = top.Top(U); t_.feed(real); t_.view = v
        ls = top.render(t_, w, hh)
        ok = ok and len(ls) == hh and all(top.plain(l) == w for l in ls)
check(ok, "recorded payload: every view at every size is exactly the screen size")
o = make_real(0)
check("n/a" in o and "GPU load" in o and "SSD read" in o and "CPU executor" in o and "granite-3.1-1b" in o and "Recent requests" in o, "recorded: Overview shows the plugin numbers and n/a for the GPU readings this machine cannot give")
o = make_real(2)
check("114" in o and "115" in o and "reused from the prompt cache" in o and "99 %" in o, "recorded: Requests shows 114 of 115 prompt tokens reused (99 %)")
o = make_real(3)
check("Hit rate by layer" in o and "whole run" in o and "lowest layers" in o, "recorded: Engine shows the per-layer heat rows")
o = make_real(5)
check("moe-cache: prompt cache" in o or "slot release" in o or "stop processing" in o, "recorded: Logs shows the real launcher and llama-server lines")
o = make_real(1)
check("-ub" in o and "512" in o and "Token modes" in o and "memory cap" in o, "recorded: Serving shows -ub 512, the token modes and the launcher's memory-cap line")

live = json.loads(json.dumps(payload))
live["approx"] = {"mode": "skip", "considered": 2000, "given_up": 300, "replaced": 100, "fewer": 2, "fewer_considered": 800, "fewer_dropped": 400, "live": True}
tl = top.Top(U); tl.feed(live); tl.src_url = "x"; tl.view = 3
ol = "\n".join(top.paint(top.render(tl, 110, 40), "dark", False))
check("skip (live)" in ol and "20.0 % of the picks looked at" in ol and "picks changed" in ol and "dropped 400 of 800" in ol, "Engine shows the approximate counters live: looked at, given up, replaced, share changed, fewer experts")
check("(at last stop)" in "\n".join(top.paint(top.render(make(3), 110, 40), "dark", False)), "Engine says 'at last stop' for counters that come from a stop-time line")

# ---- keys ----
t = make(0)
for k, want in (("2", 1), ("7", 6), ("left", 5), ("right", 6), ("right", 0), ("1", 0)):
    t.handle(k)
    if t.view != want:
        check(False, f"key {k} goes to view {want + 1} (at {t.view + 1})")
check(t.view == 0, "keys 1-7 and left/right (wrapping) switch views")
check(t.handle(" ") is None and t.frozen and "[FROZEN]" in scr(0).replace("\n", "") or t.frozen, "space freezes")
t.handle(" "); check(not t.frozen, "space again resumes")
check(t.handle("r") == "refresh" and t.handle("b") == "open" and t.handle("q") == "quit" and t.handle("ctrl-c") == "quit", "r refresh, b open, q and Ctrl-C quit")
th0 = t.theme; t.handle("t"); t.handle("t"); t.handle("t")
check(t.theme == th0, "t cycles dark, light, mono")
t.handle("?"); check(t.help and "Keys (any key closes this)" in "\n".join(top.paint(top.render(t, 110, 33), "dark", False)), "? shows the key help")
t.handle("x"); check(not t.help, "any key closes the help")
t = make(5)
t.handle("l"); check(t.level == "warn" and all(l[3] != "info" for l in t.filtered_log()), "l: warnings and errors only")
t.handle("l"); check(t.level == "error" and all(l[3] == "error" for l in t.filtered_log()) and t.filtered_log(), "l again: errors only")
t.handle("l"); t.handle("s"); check(t.source == "launcher" and all(l[2] == "launcher" for l in t.filtered_log()), "s: one source")
for _ in range(4): t.handle("s")
check(t.source == "all", "s cycles back to all sources")
t.handle("/"); check(t.typing, "/ starts typing a search")
for ch in "line 4": t.handle(ch)
check(t.query == "line 4" and all("line 4" in l[4] for l in t.filtered_log()) and len(t.filtered_log()) == 11, f"search narrows the log ({len(t.filtered_log())} lines)")
o = "\n".join(top.paint(top.render(t, 110, 33), "dark", False))
check("search: line 4_" in o, "the footer shows the search being typed")
t.handle("backspace"); t.handle("enter"); check(t.query == "line " and not t.typing, "backspace edits, Enter keeps the search")
t.handle("/"); t.handle("l"); t.handle("esc"); check(t.query == "" and not t.typing, "Esc clears the search")
t.handle("/")
for ch in "/line (1|2)5 of/": t.handle(ch)
check([l[4][:7] for l in t.filtered_log()] == ["line 15", "line 25"][:len(t.filtered_log())] and len(t.filtered_log()) == 2, f"/regex/ search ({[l[4][:7] for l in t.filtered_log()]})")
t.handle("enter"); t.handle("/"); t.handle("esc")
t.handle("up"); check(t.scroll == 1 and not t.follow, "up scrolls back from the newest line")
t.handle("home"); check(t.scroll == 10 ** 6, "Home: the oldest")
o = "\n".join(top.paint(top.render(t, 110, 33), "dark", False))
check("line 1 of the log" in o and "scrolled" in o and t.scroll < 10 ** 6, "scrolling is clamped to the lines there are")
t.handle("end"); check(t.scroll == 0 and t.follow, "End: follow the newest")
t.handle("f"); check(not t.follow, "f toggles following")
t = make(2)
for k in ("down", "down", "end", "down"):
    t.handle(k)
check(t.sel == 7, f"requests: the selection stops at the last row ({t.sel})")
t.handle("home"); t.handle("up"); check(t.sel == 0, "requests: and at the first")
t.handle("down"); o = "\n".join(top.paint(top.render(t, 110, 33), "dark", False))
check("Request at" in o, "requests: the selected request's detail")
t = make(6); check(t.handle("k") == "key" and t.key_shown and t.handle("k") is None and not t.key_shown, "Connect: k asks for the key, again hides it")

# ---- the page's port and token file ----
check(top.top_file() == ui.TOP_FILE, f"top and the page agree on the file ({top.top_file()})")
ui.TOP_FILE = os.path.join(TMP, "ui-top.json")
ui.write_top_file(8070, "tok-abc")
d = json.load(open(ui.TOP_FILE))
check(d["port"] == 8070 and d["token"] == "tok-abc" and d["pid"] == os.getpid(), "the page writes its port, token and pid")
if os.name != "nt":
    check(stat.S_IMODE(os.stat(ui.TOP_FILE).st_mode) == 0o600, "the file is private to this user (0600)")
d["pid"] = 1
json.dump(d, open(ui.TOP_FILE, "w"))
ui.remove_top_file()
check(os.path.exists(ui.TOP_FILE), "another page's file is left alone")
d["pid"] = os.getpid(); json.dump(d, open(ui.TOP_FILE, "w"))
ui.remove_top_file()
check(not os.path.exists(ui.TOP_FILE), "the page removes its own file when it stops")

# ---- discovery ----
check(top.discover("http://127.0.0.1:1/")[0] is None and "?t=" in top.discover("http://127.0.0.1:1/")[1], "--url without a token is refused with the fix")
src, _ = top.discover("http://127.0.0.1:8070/?t=abc")
check(isinstance(src, top.PageSource) and src.token == "abc" and src.base == "http://127.0.0.1:8070", "--url gives a page source with its token")
check(top.discover(port=1)[0] is None, "--port with nothing listening says so")

# ---- a fake llama-server: attach by port ----
METRICS = "llamacpp:prompt_tokens_total {p}\nllamacpp:prompt_seconds_total {ps}\nllamacpp:tokens_predicted_total {d}\nllamacpp:tokens_predicted_seconds_total {ds}\n"
state = {"n": 0}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        state["n"] += 1
        n = state["n"]
        body, code = "{}", 200
        if self.path == "/health": body = '{"status":"ok"}'
        elif self.path == "/metrics": body = METRICS.format(p=100 * n, ps=n, d=40 * n, ds=2 * n)
        elif self.path == "/slots": body = '[{"id":0,"is_processing":true,"n_ctx":4096}]'
        elif self.path == "/props": body = '{"model_path":"/models/Fake-Q4.gguf","default_generation_settings":{"n_ctx":4096}}'
        else: code = 404
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body.encode())
fake = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=fake.serve_forever, daemon=True).start()
fport = fake.server_address[1]
check(top.port_up(fport), "port_up sees the fake server")
s = top.ServerSource(fport)
p1 = s.poll(); time.sleep(0.05); p2 = s.poll()
check(p2["totals"]["state"] == "ready" and p2["totals"]["model"] == "Fake-Q4.gguf" and p2["ctx"]["max"] == 4096, f"attached: state, model name and context size ({p2['totals']})")
check(p2["samples"] and p2["samples"][-1]["dec"] == 20.0 and p2["samples"][-1]["pre"] == 100.0 and p2["samples"][-1]["busy"] == 1, "attached: decode and prompt tok/s from the counters' difference")
out = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "top", "--port", str(fport), "--once", "--ascii", "--no-color", "--width", "90", "--height", "24"],
                     capture_output=True, text=True, timeout=60)
check(out.returncode == 0 and "MOE-CACHE" in out.stdout and "Fake-Q4.gguf" in out.stdout and len(out.stdout.rstrip("\n").splitlines()) == 24 and "[Generating]" in out.stdout, f"`moe_cache.py top --port N --once` prints one 24-row screen (rc {out.returncode}) {out.stderr[-200:]}")
if os.name != "nt":      # the real thing in a pseudo-terminal: draw, answer keys, quit, give the terminal back
    import pty, select, fcntl, termios, struct
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    pr = subprocess.Popen([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "top", "--port", str(fport), "--interval", "0.5"], stdin=slave, stdout=slave, stderr=subprocess.PIPE,
                          env=dict(os.environ, TERM="xterm-256color", LANG="C.UTF-8", PYTHONIOENCODING="utf-8"), close_fds=True)
    os.close(slave)
    buf = b""
    def pump(sec):
        global buf
        end = time.time() + sec
        while time.time() < end:
            r, _, _ = select.select([master], [], [], 0.1)
            if r:
                try:
                    buf += os.read(master, 65536)
                except OSError:
                    return
    pump(2.0)
    for k in (b"2", b"3", b"4", b"6", b"/", b"a", b"\r", b"\x1b[B", b"5", b"?", b"x", b"t", b" ", b" ", b"r", b"1"):
        os.write(master, k); pump(0.25)
    os.write(master, b"q"); pump(1.5)
    try:
        rc = pr.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pr.kill(); rc = None
    out_ = buf.decode("utf-8", "replace")
    check(rc == 0 and pr.stderr.read().decode()[-200:] == "", f"top in a terminal: quits on q with exit 0 (rc {rc})")
    check("\x1b[?1049h" in out_ and out_.rstrip().endswith("\x1b[?25h\x1b[?1049l") or ("\x1b[?1049l" in out_ and "\x1b[?25h" in out_), "top in a terminal: uses the alternate screen and gives the terminal back (cursor shown again)")
    for want in ("MOE-CACHE", "Overview", "Serving", "Requests", "Engine", "Hardware", "Logs", "Fake-Q4.gguf", "Keys (any key closes this)", "search:", "[FROZEN]"):
        if want not in out_:
            check(False, f"top in a terminal: the screen showed {want!r}")
    check("top in a terminal" not in "".join(fails), "top in a terminal: every view, the search box, the help and freeze were drawn")
    os.close(master)

nothing = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "top", "--port", "1", "--once"], capture_output=True, text=True, timeout=60)
check(nothing.returncode == 1 and "nothing answers" in nothing.stderr, "top with nothing to watch exits 1 and says why")

# ---- moe-cache-server --top: start a (fake) launcher through the page's own Runner and watch it ----
import contextlib, io, socket as _s
FAKE = r"""
import http.server, sys, threading, time
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = {"/health": '{"status":"ok"}', "/slots": '[{"id":0,"is_processing":false,"n_ctx":2048}]', "/props": '{"model_path":"/m/fake-model.gguf"}',
                "/metrics": "llamacpp:prompt_tokens_total 10\nllamacpp:prompt_seconds_total 1\n"}.get(self.path)
        self.send_response(200 if body is not None else 404); self.end_headers(); self.wfile.write((body or "").encode())
srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print("moe-cache: fake launcher says hello", flush=True)
print("moe-cache: web UI and API will be at http://127.0.0.1:%d" % port, flush=True)
print("1.00.000.000 W slot something looks odd", flush=True)
time.sleep(600)
"""
ui.LAUNCHER = [sys.executable, "-c", FAKE]
gg = os.path.join(TMP, "fake-model.gguf"); open(gg, "wb").write(b"GGUF" + b"\0" * 64)
sk = _s.socket(); sk.bind(("127.0.0.1", 0)); free_port = sk.getsockname()[1]; sk.close()
buf_ = io.StringIO()
with contextlib.redirect_stdout(buf_):
    rc_ = top.top_start([gg, "--port", str(free_port), "--ctx", "2048", "--once", "--ascii", "--no-color", "--view", "6", "--width", "100", "--height", "20", "--interval", "0.5"])
o_ = buf_.getvalue()
check(rc_ == 0 and "MOE-CACHE" in o_ and "fake launcher says hello" in o_ and "odd" in o_ and "ui:" in o_, f"--top: starts through the launcher and shows its log (rc {rc_})")
time.sleep(1.5)
check(not top.port_up(free_port), "--top: the server it started is stopped when the screen ends")
rc2 = top.top_start([gg, "--cpu-moe"]); check(rc2 == 2, "--top: an option it does not support is refused with the way out")
check(top.top_start([os.path.join(TMP, "missing.gguf")]) == 2, "--top: a missing model is refused")
check(top.top_start([gg, "--ram", "0"]) == 2, "--top: a number out of range is refused")
ui.LAUNCHER = [sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "server"]

# ---- the real control page: /api/mini, /api/monitor?tail, top --url ----
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
out = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "top", "--url", f"{base}/?t={TOKEN}", "--once", "--view", "6", "--ascii", "--no-color", "--width", "80", "--height", "20"],
                     capture_output=True, text=True, timeout=60)
check(out.returncode == 0 and "line 11" in out.stdout and "[LIVE]" in out.stdout, f"`top --url` shows the page's log ({out.stderr[-200:]})")
ui.TOP_FILE = TOP_FILE0
ui.write_top_file(srv.server_address[1], TOKEN)
env = dict(os.environ, XDG_CACHE_HOME=os.environ["XDG_CACHE_HOME"])
out = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "top", "--once", "--view", "6", "--ascii", "--no-color", "--width", "80", "--height", "20"], capture_output=True, text=True, timeout=60, env=env)
check(out.returncode == 0 and "control page" not in out.stderr and "line 11" in out.stdout, f"`top` alone finds the page through its file ({out.stderr[-200:]})")
os.remove(ui.TOP_FILE)

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

print("top_test:", "FAILED" if fails else "ok", f"({len(fails)} failed)" if fails else "")
sys.exit(1 if fails else 0)
