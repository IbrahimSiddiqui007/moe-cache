#!/usr/bin/env python3
"""Model downloads for the control page (GUI-008). Standard library only.

One download at a time, file by file, only for the files the owner confirmed. Each file is written to <name>.part and renamed when its size is
right, so a cut-off download (cancel, closed page, lost connection) continues from the bytes already on disk (HTTP Range). When every file is here
the sha256 of each is checked with the picker's own verify_download (the same check `moe-cache launch` makes); a file that does not match is moved
to <name>.bad so a retry starts clean.

The catalogue is models.json (bin/moe_cache_pick.py CAT); the source is huggingface.co/<repo>/resolve/<revision>/<path>. Tests point BASE at a local
fake server and replace catalogue() with a one-file model; no real model is ever fetched by a test."""
import os, shutil, threading, time, urllib.error, urllib.request

import moe_cache_pick as pk

BASE = os.environ.get("MOE_CACHE_HF_BASE") or "https://huggingface.co"      # the variable is for tests
SPARE = 2 * 10**9          # free disk kept after a download (the picker's rule: size + 2 GB)
RETRIES = 5                # a lost connection is retried this many times in a row, each time from the bytes on disk
BACKOFF = 2.0              # seconds before a retry (tests shorten it)
CHUNK = 64 * 1024         # a read waits for this many bytes: small enough for a live progress bar and a quick cancel
TIMEOUT = 30               # seconds without a byte before a connection counts as lost


def gb(x):
    return f"{x / 1e9:.1f} GB" if x >= 1e8 else f"{x / 1e6:.1f} MB"


def catalogue():
    return pk.CAT


def safe_rel(p):
    """The catalogue path as a list of parts; refuses anything that would leave the model folder."""
    parts = str(p).replace("\\", "/").split("/")
    if not p or p.startswith("/") or ":" in parts[0] or any(x in ("", ".", "..") for x in parts):
        raise ValueError(f"unsafe file path in the catalogue: {p!r}")
    return parts


def file_path(folder, rel):
    return os.path.join(folder, *safe_rel(rel))


def free_disk(folder):
    """Free bytes where `folder` is or would be made (the nearest folder that exists)."""
    p = os.path.abspath(folder)
    while p and not os.path.isdir(p):
        up = os.path.dirname(p)
        if up == p:
            break
        p = up
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


def entry_files(entry):
    return [{"path": f["path"], "name": f.get("name") or os.path.basename(f["path"]), "bytes": int(f["bytes"]), "sha256": (f.get("sha256") or "").lower()}
            for f in entry["files"]]


def state_of(entry, folder):
    """What is on disk for one catalogue entry: per file complete / partial bytes, and what is still to fetch."""
    files = []
    for f in entry_files(entry):
        p = file_path(folder, f["path"])
        part = 0
        done = os.path.isfile(p) and os.path.getsize(p) == f["bytes"]
        if not done and os.path.isfile(p + ".part"):
            part = min(os.path.getsize(p + ".part"), f["bytes"])
        files.append(dict(f, complete=done, part=part, have=f["bytes"] if done else part))
    total = sum(f["bytes"] for f in files)
    have = sum(f["have"] for f in files)
    return {"files": files, "bytes": total, "have": have, "missing": total - have, "complete": all(f["complete"] for f in files)}


def need_free(entry, folder, spare=None):
    """Free disk the download needs: the bytes still to fetch plus the spare; the model's own minimum (DeepSeek's 240 GB) counts when nothing is here yet."""
    st = state_of(entry, folder)
    spare = SPARE if spare is None else spare
    need = st["missing"] + spare
    mf = float((entry.get("fit") or {}).get("min_free_disk_bytes") or 0)
    return max(need, mf) if st["have"] == 0 else need


def describe(k, entry, folder):
    st = state_of(entry, folder)
    free = free_disk(folder)
    need = need_free(entry, folder)
    return {"id": k, "name": entry.get("name", k), "license": entry.get("license"), "repo": (entry.get("hf") or {}).get("repo"),
            "revision": (entry.get("hf") or {}).get("revision"), "mtp": entry.get("mtp"), "uses": entry.get("uses"),
            "bytes": st["bytes"], "have": st["have"], "missing": st["missing"], "complete": st["complete"], "files": st["files"],
            "folder": folder, "free": free, "need_free": need, "fits_disk": free is None or free >= need,
            "has_checksums": all(f["sha256"] for f in st["files"])}


class Job:
    """The one running or last download. status: downloading | verifying | done | failed | cancelled."""
    def __init__(self, k, entry, folder, files):
        self.k, self.entry, self.folder = k, entry, folder
        self.files = files                                   # catalogue paths the owner confirmed
        self.status, self.message = "downloading", ""
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.cur, self.cur_done, self.cur_total = None, 0, 0
        self.total = sum(f["bytes"] for f in entry_files(entry) if f["path"] in files)
        self.base_done = 0                                   # bytes of finished files in this job
        self.t0, self.speed_at, self.speed_bytes, self.speed = time.time(), time.time(), 0, 0.0
        self.problems = []
        self.thread = None

    def public(self):
        with self.lock:
            done = self.base_done + self.cur_done
            return {"id": self.k, "name": self.entry.get("name", self.k), "status": self.status, "message": self.message, "file": self.cur,
                    "file_done": self.cur_done, "file_total": self.cur_total, "done": done, "total": self.total,
                    "speed": self.speed, "eta": (self.total - done) / self.speed if self.speed > 1 and self.status == "downloading" else None,
                    "problems": list(self.problems), "files": list(self.files), "folder": self.folder,
                    "running": self.status in ("downloading", "verifying")}

    def tick(self, n):
        with self.lock:
            self.cur_done += n
            now = time.time()
            if now - self.speed_at >= 1.0:
                inst = (self.cur_done + self.base_done - self.speed_bytes) / (now - self.speed_at)
                self.speed = inst if self.speed == 0 else 0.7 * self.speed + 0.3 * inst
                self.speed_at, self.speed_bytes = now, self.cur_done + self.base_done


class Downloads:
    def __init__(self):
        self.job = None
        self.lock = threading.Lock()
        self.on_change = None            # called after a finished download (the page's model list refreshes)

    def status(self):
        j = self.job
        return j.public() if j else {"status": "idle", "running": False}

    def start(self, k, folder, confirmed):
        """Starts the download of catalogue model k into `folder`. `confirmed` = the file paths the owner ticked: they must be exactly the files that
        are still missing (all of them, nothing else), so one OK never covers a file the page did not show."""
        cat = catalogue()
        if k not in cat:
            raise ValueError(f"unknown model '{k}'")
        entry = cat[k]
        with self.lock:
            if self.job and self.job.status in ("downloading", "verifying"):
                raise ValueError(f"a download is already running ({self.job.entry.get('name', self.job.k)}); cancel it first")
            st = state_of(entry, folder)
            todo = [f["path"] for f in st["files"] if not f["complete"]]
            if not todo:
                raise ValueError("every file of this model is already here")
            if sorted(confirmed or []) != sorted(todo):
                raise ValueError("the confirmed files are not the files still to fetch: " + ", ".join(todo))
            free, need = free_disk(folder), need_free(entry, folder)
            if free is not None and free < need:
                raise ValueError(f"not enough free disk: {gb(free)} free, {gb(need)} needed ({gb(st['missing'])} to fetch + {gb(SPARE)} spare)")
            os.makedirs(folder, exist_ok=True)
            j = self.job = Job(k, entry, folder, todo)
            j.thread = threading.Thread(target=self.run, args=(j,), daemon=True)
            j.thread.start()
        return j.public()

    def cancel(self):
        j = self.job
        if not j or j.status not in ("downloading", "verifying"):
            raise ValueError("no download is running")
        j.cancel.set()
        return j.public()

    def run(self, j):
        try:
            for f in entry_files(j.entry):
                if f["path"] not in j.files:
                    continue
                self.fetch(j, f)
                with j.lock:
                    j.base_done += f["bytes"]
                    j.cur_done = 0
            self.verify(j)
        except Cancelled:
            with j.lock:
                j.status, j.message = "cancelled", "cancelled; the part already on disk is kept and the next start continues from it"
        except Exception as e:
            with j.lock:
                j.status, j.message = "failed", str(e)
        if self.on_change:
            try:
                self.on_change(j)
            except Exception:
                pass

    def url(self, entry, rel):
        hf = entry.get("hf") or {}
        return f"{BASE}/{hf['repo']}/resolve/{hf.get('revision') or 'main'}/{rel}"

    def fetch(self, j, f):
        dest = file_path(j.folder, f["path"])
        part = dest + ".part"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with j.lock:
            j.cur, j.cur_total, j.cur_done = f["path"], f["bytes"], 0
        if os.path.isfile(dest) and os.path.getsize(dest) == f["bytes"]:
            with j.lock:
                j.cur_done = f["bytes"]
            return
        fails = 0
        while True:
            have = os.path.getsize(part) if os.path.isfile(part) else 0
            if have > f["bytes"]:
                os.truncate(part, 0)
                have = 0
            with j.lock:
                j.cur_done = have
            if have == f["bytes"]:
                break
            try:
                self.get(j, self.url(j.entry, f["path"]), part, have, f["bytes"])
                fails = 0
            except Cancelled:
                raise
            except (urllib.error.URLError, OSError, ConnectionError, TimeoutError) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code not in (500, 502, 503, 504, 429):
                    raise RuntimeError(f"{f['path']}: the server answered {e.code} {e.reason}")
                if isinstance(e, OSError) and e.errno == 28:
                    raise RuntimeError(f"{f['path']}: the disk is full")
                fails += 1
                if fails > RETRIES:
                    raise RuntimeError(f"{f['path']}: the connection kept failing ({e}); press Download again to continue from the bytes already here")
                with j.lock:
                    j.message = f"connection lost ({e}); retrying {fails}/{RETRIES}"
                end = time.time() + BACKOFF
                while time.time() < end:
                    if j.cancel.is_set():
                        raise Cancelled()
                    time.sleep(min(0.1, BACKOFF))
        if os.path.getsize(part) != f["bytes"]:
            raise RuntimeError(f"{f['path']}: size is {os.path.getsize(part)}, the catalogue says {f['bytes']}")
        os.replace(part, dest)
        with j.lock:
            j.message = ""

    def get(self, j, url, part, have, want):
        req = urllib.request.Request(url, headers={"User-Agent": "moe-cache-page", **({"Range": f"bytes={have}-"} if have else {})})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            if have and r.status != 206:               # the server ignored Range: start this file again
                have = 0
            with open(part, "ab" if have else "wb") as out:
                if not have:
                    out.truncate(0)
                    with j.lock:
                        j.cur_done = 0
                with j.lock:
                    j.message = ""
                while True:
                    if j.cancel.is_set():
                        raise Cancelled()
                    b = r.read(CHUNK)
                    if not b:
                        break
                    out.write(b)
                    j.tick(len(b))
            if os.path.getsize(part) < want:
                raise ConnectionError(f"the connection closed after {os.path.getsize(part)} of {want} bytes")

    def verify(self, j):
        entry = j.entry
        with j.lock:
            j.status, j.cur = "verifying", None
        files = [f for f in entry_files(entry) if f["path"] in j.files]
        done = {"n": 0}

        def say(line):
            with j.lock:
                j.message = line.replace("moe-cache: ", "")
            if j.cancel.is_set():
                raise Cancelled()
        problems, checked = pk.verify_download(j.k, j.folder, say, files=files)
        for pr in problems:
            rel = pr.split(":", 1)[0]
            p = file_path(j.folder, rel)
            if os.path.isfile(p) and "sha256" in pr:
                os.replace(p, p + ".bad")
        link = entry.get("link")
        if not problems and isinstance(link, dict):
            a, b = file_path(j.folder, link["file"]), file_path(j.folder, link["target"])
            try:
                if not os.path.exists(a) and os.path.exists(b):
                    os.link(b, a)
            except OSError as e:
                problems.append(f"the download is done, but linking {link['file']} failed ({e})")
        with j.lock:
            j.problems = problems
            if problems:
                j.status, j.message = "failed", ("checksum or link problem: " + "; ".join(problems) + ("; the bad file was kept as .bad, press Download to fetch it again" if any("sha256" in x for x in problems) else ""))
            else:
                j.status = "done"
                j.message = f"done: {checked} file(s) checked against the catalogue's sha256" if checked else "done (the catalogue lists no checksum for these files; not checked)"


class Cancelled(Exception):
    pass
