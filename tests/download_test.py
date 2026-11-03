#!/usr/bin/env python3
"""Model downloads in the control page (GUI-008, bin/moe_cache_download.py) and the MTP label (GUI-011), without a model: a local fake HTTP
server plays Hugging Face and serves small files (a few hundred KB). Nothing real is ever fetched.
   python3 tests/download_test.py"""
import hashlib, http.server, json, os, shutil, struct, sys, tempfile, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-dl-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
os.environ["XDG_CONFIG_HOME"] = os.path.join(TMP, "config")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_ui as ui  # noqa: E402
import moe_cache_download as dl  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# ---- the fake Hugging Face ----
DATA = {"fake/repo/resolve/rev1/sub/A.gguf": os.urandom(700_000), "fake/repo/resolve/rev1/B.gguf": os.urandom(300_000)}
MODE = {"cut_first": False, "ignore_range": False, "status": None, "slow": 0.0}
LOG = []

class Fake(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_GET(self):
        body = DATA.get(self.path.lstrip("/"))
        LOG.append((self.path, self.headers.get("Range")))
        if MODE["status"]:
            self.send_response(MODE["status"]); self.send_header("Content-Length", "0"); self.end_headers(); return
        if body is None:
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
        start, code = 0, 200
        rg = self.headers.get("Range")
        if rg and not MODE["ignore_range"]:
            start, code = int(rg.split("=")[1].rstrip("-")), 206
        chunk = body[start:]
        self.send_response(code)
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        if MODE["cut_first"] and start == 0:
            MODE["cut_first"] = False
            self.wfile.write(chunk[:200_000]); self.wfile.flush(); self.close_connection = True; return      # the connection drops half way
        for i in range(0, len(chunk), 50_000):
            try:
                self.wfile.write(chunk[i:i + 50_000]); self.wfile.flush()
            except OSError:
                return
            if MODE["slow"]:
                time.sleep(MODE["slow"])

fake = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
fake.daemon_threads = True
threading.Thread(target=fake.serve_forever, daemon=True).start()
dl.BASE = f"http://127.0.0.1:{fake.server_address[1]}"
dl.BACKOFF = 0.1
dl.SPARE = 1000                  # the real spare is 2 GB; the temp disk is not asked for that much

def sha(b): return hashlib.sha256(b).hexdigest()
A, B = DATA["fake/repo/resolve/rev1/sub/A.gguf"], DATA["fake/repo/resolve/rev1/B.gguf"]
ENTRY = {"name": "Fake model", "license": "mit", "mtp": "in-file", "hf": {"repo": "fake/repo", "revision": "rev1"},
         "files": [{"name": "A.gguf", "path": "sub/A.gguf", "bytes": len(A), "sha256": sha(A)}, {"name": "B.gguf", "path": "B.gguf", "bytes": len(B), "sha256": sha(B)}]}
BAD = json.loads(json.dumps(ENTRY)); BAD["files"][1]["sha256"] = "0" * 64
_real = dict(dl.catalogue())
dl.catalogue = lambda: dict(_real, fake=ENTRY, badsum=BAD)
ui.picker.CAT = dict(ui.picker.CAT, fake=ENTRY, badsum=BAD)       # verify_download only reads files= here, but keep the lookups honest

def wait(d, states=("done", "failed", "cancelled"), t=30):
    end = time.time() + t
    while time.time() < end:
        s = d.status()
        if s["status"] in states:
            return s
        time.sleep(0.05)
    return d.status()

def fresh(name):
    p = os.path.join(TMP, name); shutil.rmtree(p, ignore_errors=True); os.makedirs(p); return p

# ---- module: state, confirm rule, disk check ----
f1 = fresh("m1")
st = dl.state_of(ENTRY, f1)
check(st["bytes"] == 1_000_000 and st["missing"] == 1_000_000 and not st["complete"], "state: nothing here, 1,000,000 bytes to fetch")
d = dl.Downloads()
for bad, what in (([], "no file ticked"), (["B.gguf"], "only one of two files ticked"), (["B.gguf", "sub/A.gguf", "x"], "a file the page did not show")):
    try:
        d.start("fake", f1, bad); check(False, "owner OK: refused when " + what)
    except ValueError as e:
        check("not the files still to fetch" in str(e), "owner OK: refused when " + what)
try:
    d.start("nope", f1, []); check(False, "unknown model refused")
except ValueError as e:
    check("unknown model" in str(e), "unknown model refused")
real_free = dl.free_disk
dl.free_disk = lambda folder: 500_000
try:
    d.start("fake", f1, ["sub/A.gguf", "B.gguf"]); check(False, "disk check refuses")
except ValueError as e:
    check("not enough free disk" in str(e) and "500.0 KB" not in str(e) and "free disk" in str(e) and not LOG, f"disk check: refused before any request ({e})")
dl.free_disk = real_free
check(not os.listdir(f1), "nothing was written by a refused start")
for evil in ("../x.gguf", "/etc/x", "a/../../x", "C:/x"):
    try:
        dl.safe_rel(evil); check(False, "path refused: " + evil)
    except ValueError:
        check(True, "path refused: " + evil)

# ---- a clean download, with sha256 ----
d.start("fake", f1, ["sub/A.gguf", "B.gguf"])
s = wait(d)
check(s["status"] == "done" and s["done"] == 1_000_000 and "2 file(s) checked" in s["message"], f"download: done, both files checked ({s['status']}: {s['message']})")
check(open(os.path.join(f1, "sub", "A.gguf"), "rb").read() == A and open(os.path.join(f1, "B.gguf"), "rb").read() == B, "download: the files are byte-identical")
check(not [x for r, _, fs in os.walk(f1) for x in fs if x.endswith(".part")], "download: no .part left")
check(dl.state_of(ENTRY, f1)["complete"], "state: complete afterwards")
try:
    d.start("fake", f1, []); check(False, "nothing to fetch refused")
except ValueError as e:
    check("already here" in str(e), "a second start for a complete model is refused")

# ---- resume after cancel ----
f2 = fresh("m2"); LOG.clear(); MODE["slow"] = 0.1
d = dl.Downloads()
d.start("fake", f2, ["sub/A.gguf", "B.gguf"])
end = time.time() + 20
while time.time() < end and d.status()["done"] < 100_000:
    time.sleep(0.02)
d.cancel()
s = wait(d)
check(s["status"] == "cancelled", f"cancel: status cancelled ({s['status']})")
part = os.path.getsize(os.path.join(f2, "sub", "A.gguf.part"))
check(0 < part < len(A), f"cancel: the part is kept ({part} bytes)")
check(dl.state_of(ENTRY, f2)["have"] == part, "state: the partial bytes count as already here")
MODE["slow"] = 0; LOG.clear()
d.start("fake", f2, ["sub/A.gguf", "B.gguf"])
s = wait(d)
check(s["status"] == "done", f"resume: finished and verified ({s['status']}: {s['message']})")
check(LOG[0][1] == f"bytes={part}-", f"resume: the request asked for bytes={part}- ({LOG[0]})")
check(open(os.path.join(f2, "sub", "A.gguf"), "rb").read() == A, "resume: the joined file is byte-identical")

# ---- lost connection: retried from the bytes on disk ----
f3 = fresh("m3"); LOG.clear(); MODE["cut_first"] = True
d = dl.Downloads(); d.start("fake", f3, ["sub/A.gguf", "B.gguf"]); s = wait(d)
check(s["status"] == "done" and any(r == "bytes=200000-" for _, r in LOG), f"connection cut at 200,000 bytes: retried with Range and finished ({s['status']}, {[r for _, r in LOG][:3]})")

# ---- a server that ignores Range: the file starts again, result still right ----
f4 = fresh("m4"); os.makedirs(os.path.join(f4, "sub")); open(os.path.join(f4, "sub", "A.gguf.part"), "wb").write(A[:1000]); MODE["ignore_range"] = True
d = dl.Downloads(); d.start("fake", f4, ["sub/A.gguf", "B.gguf"]); s = wait(d); MODE["ignore_range"] = False
check(s["status"] == "done" and open(os.path.join(f4, "sub", "A.gguf"), "rb").read() == A, f"no Range support: restarts that file, right bytes ({s['status']})")

# ---- wrong checksum: failed, bad file set aside ----
f5 = fresh("m5"); d = dl.Downloads(); d.start("badsum", f5, ["sub/A.gguf", "B.gguf"]); s = wait(d)
check(s["status"] == "failed" and "sha256" in s["message"] and "B.gguf" in s["message"], f"sha256 mismatch: failed with a plain sentence ({s['message'][:90]})")
check(os.path.isfile(os.path.join(f5, "B.gguf.bad")) and not os.path.exists(os.path.join(f5, "B.gguf")) and os.path.isfile(os.path.join(f5, "sub", "A.gguf")),
      "sha256 mismatch: the bad file is kept as .bad, the good one stays")

# ---- server errors ----
f6 = fresh("m6"); MODE["status"] = 404
d = dl.Downloads(); d.start("fake", f6, ["sub/A.gguf", "B.gguf"]); s = wait(d); MODE["status"] = None
check(s["status"] == "failed" and "404" in s["message"], f"HTTP 404: failed at once ({s['message'][:70]})")
f7 = fresh("m7"); MODE["status"] = 503; dl.RETRIES = 2
d = dl.Downloads(); d.start("fake", f7, ["sub/A.gguf", "B.gguf"]); s = wait(d); MODE["status"] = None; dl.RETRIES = 5
check(s["status"] == "failed" and "kept failing" in s["message"], f"HTTP 503: retried, then failed with advice ({s['message'][:70]})")

# ---- one at a time ----
f8 = fresh("m8"); MODE["slow"] = 0.05
d = dl.Downloads(); d.start("fake", f8, ["sub/A.gguf", "B.gguf"])
try:
    d.start("fake", fresh("m8b"), ["sub/A.gguf", "B.gguf"]); check(False, "second download refused")
except ValueError as e:
    check("already running" in str(e), "a second download while one runs is refused")
d.cancel(); wait(d); MODE["slow"] = 0

# ---- through the page's server ----
folder = fresh("folder")
ui.default_folders = lambda: [folder]
ui.ollama_roots = lambda: []
TOKEN = "dl-token"
srv = ui.make_server(0, "llama-server", token=TOKEN)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"
def req(path, body=None, token=TOKEN):
    h = {"X-MoeCache-Token": token} if token else {}
    if body is not None:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, h)
    try:
        with urllib.request.urlopen(r, timeout=30) as x:
            return x.status, json.loads(x.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")
check(req("/api/downloads", token=None)[0] == 403 and req("/api/download/start", {}, token=None)[0] == 403, "page: downloads need the page token")
st, j = req("/api/downloads")
fk = next((m for m in j["models"] if m["id"] == "fake"), None)
check(st == 200 and fk and fk["missing"] == 1_000_000 and fk["fits_disk"] and fk["free"] > 0 and fk["folder"] == os.path.abspath(folder) and fk["has_checksums"],
      "page: /api/downloads lists size, bytes missing, free disk, folder, checksums")
check(len(j["models"]) >= 9 and any(m["id"] == "bolt" for m in j["models"]), "page: the real catalogue entries are listed too (not fetched)")
check(req("/api/download/start", {"id": "fake", "files": ["B.gguf"]})[0] == 400, "page: start without every file ticked: 400")
check(req("/api/download/start", {"id": "fake", "folder": "/etc", "files": ["sub/A.gguf", "B.gguf"]})[0] == 400, "page: a folder that is not a model folder: 400")
check(req("/api/download/cancel", {})[0] == 400, "page: cancel with nothing running: 400")
MODE["slow"] = 0.1
st, j = req("/api/download/start", {"id": "fake", "files": ["sub/A.gguf", "B.gguf"]})
check(st == 200 and j["running"], "page: start with every file ticked: running")
time.sleep(0.5)
st, j = req("/api/download")
check(st == 200 and 0 < j["done"] < j["total"] == 1_000_000 and j["status"] == "downloading", f"page: progress while running ({j['done']} of {j['total']})")
for _ in range(200):
    st, j = req("/api/download")
    if not j["running"]:
        break
    time.sleep(0.1)
check(j["status"] == "done", f"page: finished ({j['status']}: {j.get('message')})")
check(os.path.isfile(os.path.join(folder, "B.gguf")), "page: the file is in the model folder")
MODE["slow"] = 0

# ---- MTP label (GUI-011) ----
def gguf(path, names, kv=()):
    """A tiny GGUF: header, string metadata, and tensor names (no tensor data)."""
    b = b"GGUF" + struct.pack("<IQQ", 3, len(names), len(kv))
    for k, v in kv:
        b += struct.pack("<Q", len(k)) + k.encode() + struct.pack("<II", 4, v)
    for n in names:
        b += struct.pack("<Q", len(n)) + n.encode() + struct.pack("<I", 1) + struct.pack("<Q", 8) + struct.pack("<IQ", 0, 0)
    open(path, "wb").write(b + b"\0" * 64)
mt = fresh("mtp")
gguf(os.path.join(mt, "With-Head-Q4.gguf"), ["blk.0.attn_q.weight", "blk.1.attn_q.weight", "blk.2.nextn.eh_proj.weight"], [("x.nextn_predict_layers", 1)])
gguf(os.path.join(mt, "Plain-Q4.gguf"), ["blk.0.attn_q.weight", "blk.1.attn_q.weight"])
os.makedirs(os.path.join(mt, "other")); gguf(os.path.join(mt, "other", "Beside-Q4.gguf"), ["blk.0.attn_q.weight"]); open(os.path.join(mt, "other", "mtp-Beside.gguf"), "wb").write(b"GGUF" + b"\0" * 100)
ms = {m["name"]: m for m in ui.find_models([mt])}
check(ms["With-Head-Q4"]["mtp"] and ms["With-Head-Q4"]["mtp"]["head"] == 1 and ms["With-Head-Q4"]["mtp"]["declared"] == 1, f"MTP: a head inside the file is found ({ms['With-Head-Q4']['mtp']})")
check(ms["Plain-Q4"]["mtp"] and ms["Plain-Q4"]["mtp"]["head"] == 0 and ms["Plain-Q4"]["mtp"]["beside"] is None, "MTP: a plain model has none")
check(ms["Beside-Q4"]["mtp"]["head"] == 0 and ms["Beside-Q4"]["mtp"]["beside"] == "mtp-Beside.gguf" and "mtp-Beside" not in ms, "MTP: a separate mtp-*.gguf beside is reported, and is not listed as a model")

page = open(os.path.join(HERE, "ui", "index.html"), encoding="utf-8").read()
check("function mtpChip" in page and "mtpChip(m.mtp)" in page and "MTP head" in page and "not used" in page, "MTP: the page labels the models in its list")

shutil.rmtree(TMP, ignore_errors=True)
print("\nFAILED: " + "; ".join(fails) if fails else "\nall download tests passed")
sys.exit(1 if fails else 0)
