#!/usr/bin/env python3
"""tools/check_model_links.py (AUD-098): a deleted model repo (HTTP 404/410) fails, a Hugging Face outage (503, 429, network error) only warns.
Runs the tool in-process with urllib's urlopen replaced; no network."""
import contextlib, io, os, runpy, sys, urllib.error, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(HERE, "tools", "check_model_links.py")
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def run(opener):
    real = urllib.request.urlopen
    urllib.request.urlopen = opener
    out = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(out):
            try:
                runpy.run_path(TOOL, run_name="__main__")
            except SystemExit as e:
                code = e.code or 0
    finally:
        urllib.request.urlopen = real
    return code, out.getvalue()

def http_error(code):
    def op(url, timeout=None):
        raise urllib.error.HTTPError(url, code, "x", {}, None)
    return op

rc, out = run(http_error(404))
check(rc == 1 and "FAIL" in out and "404" in out, f"HTTP 404 (repo deleted) fails the check (rc {rc})")
rc, out = run(http_error(503))
check(rc == 0 and "warn" in out and "FAIL" not in out, f"HTTP 503 (outage) only warns (rc {rc})")
rc, out = run(http_error(429))
check(rc == 0 and "FAIL" not in out, f"HTTP 429 only warns (rc {rc})")
def neterr(url, timeout=None):
    raise OSError("no network")
rc, out = run(neterr)
check(rc == 0 and "FAIL" not in out, f"a network error only warns (rc {rc})")
print("model_links_test: " + ("PASS" if not fails else "FAILED: " + "; ".join(fails)))
sys.exit(1 if fails else 0)
