#!/usr/bin/env python3
"""LAUNCHER-007: resumable download and free-disk check (bin/moe_cache_fetch.py), model-free. A local HTTP server serves a 3 MB file; it can cut the
connection half way, ignore Range, or serve garbage. Deps.fetch_files is checked with a fake catalogue entry pointed at the same server (HF_ENDPOINT)."""
import hashlib, http.server, os, shutil, sys, tempfile, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin"))
import moe_cache_fetch as fx
import moe_cache_launch as L

fails = 0
def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what)
    fails += 0 if ok else 1

DATA = os.urandom(3 * 1024 * 1024 + 123)
SHA = hashlib.sha256(DATA).hexdigest()
S = {"cut_at": None, "ignore_range": False, "body": DATA, "ranges": [], "hits": 0}

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        S["hits"] += 1
        body, start = S["body"], 0
        rng = self.headers.get("Range")
        S["ranges"].append(rng)
        if rng and not S["ignore_range"]:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(body):
                self.send_response(416); self.send_header("Content-Length", "0"); self.end_headers(); return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        else:
            self.send_response(200)
        out = body[start:]
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        if S["cut_at"] is not None:
            n, S["cut_at"] = S["cut_at"], None            # cut once, mid-body
            self.wfile.write(out[:n]); self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(out)

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_address[1]}/m.gguf"
TMP = tempfile.mkdtemp(prefix="fetch_test_")
nosleep = lambda s: None
def lines():
    l = []
    return l, l.append

# ---- plain download ----
d = os.path.join(TMP, "a", "b"); dest = os.path.join(d, "m.gguf")
l, say = lines()
check(fx.fetch(URL, dest, len(DATA), SHA, say, sleep=nosleep, margin=0) is None and open(dest, "rb").read() == DATA, "plain download: file equals the served bytes, creates the folder")
check(not os.path.exists(dest + ".part"), "no .part left after success")
S["hits"] = 0
check(fx.fetch(URL, dest, len(DATA), SHA, say, margin=0) is None and S["hits"] == 0, "a complete file is not downloaded again")

# ---- cut mid-way, resumed in the same call ----
os.remove(dest); S["cut_at"] = 1_000_000; S["ranges"].clear()
l, say = lines()
err = fx.fetch(URL, dest, len(DATA), SHA, say, sleep=nosleep, margin=0)
check(err is None and open(dest, "rb").read() == DATA, "cut connection: resumed and the final file is complete and equal")
check(S["ranges"][0] is None and S["ranges"][1] and S["ranges"][1].startswith("bytes=") and int(S["ranges"][1][6:-1]) > 0, f"second request is a Range request ({S['ranges']})")
check(any("resuming" in x or "connection lost" in x for x in l), "the user is told about the resume")

# ---- a .part from an earlier run (process killed) ----
os.remove(dest)
open(dest + ".part", "wb").write(DATA[:2_000_000]); S["ranges"].clear()
check(fx.fetch(URL, dest, len(DATA), SHA, lambda x: None, margin=0) is None and S["ranges"] == ["bytes=2000000-"] and open(dest, "rb").read() == DATA, "an old .part continues at its size")

# ---- server ignores Range (200): start over, still correct ----
os.remove(dest); open(dest + ".part", "wb").write(DATA[:500_000]); S["ignore_range"] = True
check(fx.fetch(URL, dest, len(DATA), SHA, lambda x: None, margin=0) is None and open(dest, "rb").read() == DATA, "server without Range support: restarts, file correct")
S["ignore_range"] = False

# ---- bad prefix in .part: the hash at the end catches it, .part is deleted ----
os.remove(dest); open(dest + ".part", "wb").write(b"\0" * 10_000)
err = fx.fetch(URL, dest, len(DATA), SHA, lambda x: None, margin=0)
check(err and "sha256" in err and not os.path.exists(dest) and not os.path.exists(dest + ".part"), "corrupt resumed download: hash error, the bad .part is removed, no final file")
check(fx.fetch(URL, dest, len(DATA), SHA, lambda x: None, margin=0) is None and open(dest, "rb").read() == DATA, "...and the next run downloads clean")

# ---- .part bigger than expected, 416 ----
os.remove(dest); open(dest + ".part", "wb").write(DATA + b"x")
check(fx.fetch(URL, dest, len(DATA), SHA, lambda x: None, margin=0) is None and open(dest, "rb").read() == DATA, ".part larger than the file: discarded")

# ---- gives up after the retries, keeps the .part ----
os.remove(dest)
class Cut(H):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "100"); self.end_headers(); self.wfile.write(b"x" * 10); self.wfile.flush(); self.connection.close()
s2 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Cut); threading.Thread(target=s2.serve_forever, daemon=True).start()
err = fx.fetch(f"http://127.0.0.1:{s2.server_address[1]}/x", dest, 100, None, lambda x: None, retries=2, sleep=nosleep, margin=0)
check(err and "cut" in err and os.path.getsize(dest + ".part") > 0 and not os.path.exists(dest), "a server that always cuts: clear error, .part kept for the next run")
os.remove(dest + ".part")


# ---- free disk (fake low disk) ----
real_du = shutil.disk_usage
shutil.disk_usage = lambda p: shutil._ntuple_diskusage(100 * 10**9, 99 * 10**9, 1 * 10**9)
try:
    S["hits"] = 0
    err = fx.fetch(URL, dest, 20 * 10**9, SHA, lambda x: None)          # 20 GB needed, 1 GB free
    check(err and "20.0 GB" in err and "22.0 GB" in err and "only 1.0 GB is free" in err, "low disk: message has the needed size, the margin total and the free size")
    check(S["hits"] == 0 and not os.path.exists(dest + ".part"), "low disk: not one byte requested, no .part created")
    check(fx.free_check(os.path.join(TMP, "not", "yet", "there"), 5 * 10**8, 0) is None, "free_check: enough room passes, missing folder is looked up at its parent")
    check(fx.free_check(TMP, 5 * 10**8, 1e9) is not None, "free_check: the margin counts")
    # the resumed rest is what counts, not the whole file
    open(dest + ".part", "wb").write(b"x" * 10)
finally:
    shutil.disk_usage = real_du
os.path.exists(dest + ".part") and os.remove(dest + ".part")

# ---- .part held by someone else (the lock; on Windows an open handle) ----
open(dest + ".part", "wb").write(DATA[:1000])
holder = open(dest + ".part", "ab")
check(fx._lock(holder), "lock: first holder gets it")
err = fx.fetch(URL, dest, len(DATA), SHA, lambda x: None, margin=0)
if os.name == "nt":                      # mandatory lock: open itself may fail or the lock does; either way a sentence, no traceback, no download
    check(err and ("in use" in err or "cannot open" in err or "writing" in err), "Windows: a .part held open gives a clear error")
else:
    check(err and "in use" in err, "a .part locked by another download gives a clear error")
fx._unlock(holder); holder.close()
check(fx.fetch(URL, dest, len(DATA), SHA, lambda x: None, margin=0) is None and open(dest, "rb").read() == DATA, "after the holder lets go, the download resumes and completes")

# ---- rename retried when the new file is held (Windows AV): PermissionError twice, then ok ----
calls = {"n": 0}
real_replace = os.replace
def flaky(a, b):
    calls["n"] += 1
    if calls["n"] < 3:
        raise PermissionError("held open")
    return real_replace(a, b)
os.replace = flaky
try:
    open(os.path.join(TMP, "r.part"), "wb").write(b"abc")
    check(fx._replace(os.path.join(TMP, "r.part"), os.path.join(TMP, "r"), sleep=nosleep) is None and os.path.exists(os.path.join(TMP, "r")), "rename retries a PermissionError")
    calls["n"] = -100
    open(os.path.join(TMP, "r2.part"), "wb").write(b"abc")
    e = fx._replace(os.path.join(TMP, "r2.part"), os.path.join(TMP, "r2"), tries=2, sleep=nosleep)
    check(e and "holding the file open" in e and os.path.exists(os.path.join(TMP, "r2.part")), "rename gives up with a sentence and keeps the .part")
finally:
    os.replace = real_replace

# ---- Deps.fetch_files / download through the launcher, with HF_ENDPOINT at the fake server ----
folder = os.path.join(TMP, "models")
orig_cat, orig_models = L.pk.CAT, L.pk.MODELS
L.pk.CAT = {"t": {"files": [{"name": "m.gguf", "path": "sub/m.gguf", "bytes": len(DATA), "sha256": SHA}], "hf": {"repo": "o/r", "revision": "abc"}}}
L.pk.MODELS = {"t": {"repo": "o/r", "size": len(DATA)}}
S["body"], S["ranges"] = DATA, []
class Q(H):
    pass
H_paths = []
orig_get = H.do_GET
def get(self):
    H_paths.append(self.path); orig_get(self)
H.do_GET = get
os.environ["HF_ENDPOINT"] = f"http://127.0.0.1:{srv.server_address[1]}"
rec = {"id": "t", "download": {"folder": folder}}
S["cut_at"] = 700_000
real = L.Deps()
check(real.download(rec) == 0 and open(os.path.join(folder, "sub", "m.gguf"), "rb").read() == DATA, "Deps.download: cut + resume through the launcher, file in the catalogue path")
check(H_paths and H_paths[0] == "/o/r/resolve/abc/sub/m.gguf", f"the URL uses the pinned revision ({H_paths[:1]})")
check(real.verify(rec, lambda x: None) is None, "...and Deps.verify passes on it")
shutil.disk_usage = lambda p: shutil._ntuple_diskusage(100 * 10**9, 99 * 10**9, 10**6)
try:
    os.remove(os.path.join(folder, "sub", "m.gguf")); H_paths.clear()
    r = real.download(rec)
    check(isinstance(r, str) and "not enough free disk" in r and not H_paths, "Deps.download: low disk -> error text, no request made")
finally:
    shutil.disk_usage = real_du
L.pk.CAT, L.pk.MODELS = orig_cat, orig_models
srv.shutdown(); s2.shutdown()
shutil.rmtree(TMP, ignore_errors=True)
print("fetch_test:", "all passed" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
