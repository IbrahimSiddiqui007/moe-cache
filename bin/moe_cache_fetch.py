#!/usr/bin/env python3
"""Resumable download for `launch --download` (LAUNCHER-007): a free-disk check before the first byte, HTTP Range resume of a cut download
(the .part file is kept), a sha256 / size check at the end, and only then the rename to the final name. Standard library only; no model is
needed to test it (tests/fetch_test.py serves a small file from a local HTTP server).

  free_check(folder, need_bytes, margin)    -> None, or one sentence with both numbers (needed, free)
  fetch(url, dest, size, sha256, ...)       -> None, or one sentence saying what went wrong; a cut connection is resumed, not restarted
"""
import hashlib, http.client, os, shutil, time
import urllib.error, urllib.request

MARGIN = 2e9           # free disk kept beyond the download itself (same margin as moe_cache_pick.need_free)
RETRIES = 5            # resumes after a cut connection before giving up (the .part stays; the next run continues)
CHUNK = 1 << 20


def gb(n):
    return f"{n / 1e9:.1f} GB"


def _existing_dir(path):
    """The nearest folder that exists at or above `path` (the download folder may not be created yet)."""
    p = os.path.abspath(path)
    while p and not os.path.isdir(p):
        up = os.path.dirname(p)
        if up == p:
            break
        p = up
    return p


def free_check(folder, need, margin=MARGIN):
    """None when `folder`'s drive has need + margin bytes free, else a sentence with both numbers. A drive that cannot be asked is not an error."""
    try:
        free = shutil.disk_usage(_existing_dir(folder)).free
    except OSError:
        return None
    if free >= need + margin:
        return None
    return (f"not enough free disk at {folder}: the download needs {gb(need)} plus {gb(margin)} of margin = {gb(need + margin)}, "
            f"but only {gb(free)} is free ({gb(need + margin - free)} missing). Nothing was downloaded. Free that space or use --models-dir on a bigger drive.")


def _lock(f):
    """Exclusive, non-blocking lock on the open .part file so two launches do not write one file. False when someone else holds it."""
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(f):
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def sha256_of(path, chunk=8 * CHUNK):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _replace(src, dst, tries=5, sleep=time.sleep):
    """os.replace with retries: on Windows a virus scanner or indexer can hold the new file for a moment (PermissionError)."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return None
        except PermissionError as e:
            if i == tries - 1:
                return f"could not rename {src} to {dst}: {e} (is another program holding the file open? The download is kept; run again)"
            sleep(1.0 + i)
        except OSError as e:
            return f"could not rename {src} to {dst}: {e}"


def fetch(url, dest, size, sha256=None, say=print, headers=None, retries=RETRIES, sleep=time.sleep, margin=MARGIN, timeout=60):
    """Download url to dest (size bytes, sha256 hex or None). Resumes dest + '.part' with a Range request. Returns None on success, else a sentence."""
    part = dest + ".part"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    except OSError as e:
        return f"cannot create the folder for {dest}: {e}"
    if os.path.isfile(dest) and os.path.getsize(dest) == size:
        say(f"     {os.path.basename(dest)} is already here ({gb(size)}); not downloaded again")
        return None
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    if have > size:
        say(f"     the old partial file is bigger than the expected {size} bytes; starting again")
        os.remove(part)
        have = 0
    err = free_check(os.path.dirname(os.path.abspath(dest)), size - have, margin)
    if err:
        return err
    try:
        f = open(part, "ab")
    except OSError as e:
        return f"cannot open {part}: {e}"
    try:
        if not _lock(f):
            return f"{part} is in use by another download (or another program holds it open); wait for it or close it, then run again"
        if have:
            say(f"     resuming {os.path.basename(dest)} at {gb(have)} of {gb(size)}")
        fails = 0
        while True:
            f.seek(0, os.SEEK_END)
            have = f.tell()
            if have >= size:
                break
            try:
                req = urllib.request.Request(url, headers={**(headers or {}), "Range": f"bytes={have}-"} if have else dict(headers or {}))
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    if have and r.status != 206:           # the server ignored the Range: the body is the whole file
                        f.truncate(0)
                        f.seek(0)
                        have = 0
                    elif have:
                        cr = r.headers.get("Content-Range") or ""
                        if not cr.startswith(f"bytes {have}-"):
                            return f"the server answered the resume at byte {have} with Content-Range '{cr}'; delete {part} and run again"
                    last = -1
                    while True:
                        b = r.read(CHUNK)
                        if not b:
                            break
                        f.write(b)
                        have += len(b)
                        pct = have * 100 // max(size, 1) // 10 * 10
                        if pct != last:
                            last = pct
                            say(f"     downloading {os.path.basename(dest)}: {pct} %")
                    f.flush()
                if have < size:
                    raise ConnectionError(f"connection closed at {have} of {size} bytes")
            except urllib.error.HTTPError as e:
                if e.code == 416:      # Range not satisfiable: the .part is not a prefix of what the server has -> start over
                    f.truncate(0)
                    f.seek(0)
                    fails += 1
                    if fails > retries:
                        return f"the server refuses to resume {url} (HTTP 416)"
                    continue
                return f"download of {url} failed: HTTP {e.code} {e.reason}"
            except (OSError, http.client.HTTPException) as e:
                net = isinstance(e, (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException))
                if not net:
                    return f"writing {part} failed: {e}"       # disk full, permission: resuming cannot help
                fails += 1
                if fails > retries:
                    return f"the download was cut {fails} times ({e}); {gb(os.path.getsize(part))} of {gb(size)} is kept in {part}, run the same command again to continue"
                say(f"     connection lost ({e}); resuming (try {fails} of {retries})")
                f.flush()
                sleep(min(2.0 * fails, 10.0))
    finally:
        _unlock(f)
        f.close()
    got = os.path.getsize(part)
    if got != size:
        os.remove(part)
        return f"{os.path.basename(dest)} is {got} bytes but {size} were expected; the partial file was deleted, run again"
    if sha256:
        say(f"     checking {os.path.basename(dest)} ...")
        h = sha256_of(part)
        if h != sha256.lower():
            os.remove(part)         # a resumed corrupt prefix would stay corrupt: start clean next time
            return f"{os.path.basename(dest)}: sha256 {h} but the catalogue says {sha256}; the partial file was deleted, run again"
    return _replace(part, dest, sleep=sleep)
