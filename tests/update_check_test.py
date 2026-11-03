#!/usr/bin/env python3
"""`moe_cache.py update --check` (AUD-092): the fetch and parse of GitHub's release lists, on FAKE JSON (urllib's urlopen is replaced; no network).
update_advice's pure wording is covered elsewhere; this drives update_cmd end to end: request headers, pre-releases taken, drafts skipped, llama.cpp's
non-bNNNN tags skipped, an offline / garbled / rate-limited answer degrading to a message, and nothing written or downloaded."""
import contextlib, io, json, os, sys, tempfile, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="update-test-")
sys.path.insert(0, os.path.join(ROOT, "bin"))
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

class Resp:
    def __init__(self, body): self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
    def read(self): return self.body
    def __enter__(self): return self
    def __exit__(self, *a): return False

def run(mine, repo, llama, ls_b=None):
    """repo / llama: JSON to answer with, or an Exception to raise, or bytes. Returns (output, requests)."""
    seen = []
    def urlopen(req, timeout=None):
        seen.append((req.full_url, dict(req.header_items()), timeout))
        ans = llama if "llama.cpp" in req.full_url else repo
        if isinstance(ans, Exception):
            raise ans
        return Resp(ans)
    old = (mc.urllib.request.urlopen, mc.VERSION, mc.ls_build, mc.say)
    out = []
    mc.urllib.request.urlopen, mc.VERSION = urlopen, mine
    mc.ls_build = lambda p: ls_b
    mc.say = lambda m="": out.append(str(m))
    try:
        rc = mc.update_cmd(["--check", "--llama-server", sys.executable])       # an existing executable: shutil.which finds it
    finally:
        mc.urllib.request.urlopen, mc.VERSION, mc.ls_build, mc.say = old
    return rc, "\n".join(out), seen

REL = [{"tag_name": "v0.9.9-draft", "draft": True}, {"tag_name": "v0.4.0-beta", "draft": False, "prerelease": True}, {"tag_name": "v0.3.5"}]
LL = [{"tag_name": "v0.1.0"}, {"tag_name": "b12100"}, {"tag_name": "b12000"}]

rc, out, seen = run("0.3.5", REL, LL, ls_b=11966)
check(rc == 0 and "moe-cache 0.4.0-beta is out (you have 0.3.5)" in out, "a newer pre-release is reported, the newer DRAFT is skipped")
check("your llama-server is build 11966, the newest release is b12100 (134 builds newer" in out, "llama.cpp: the first bNNNN tag is used, the old v0.1.0 tag is skipped; builds behind counted")
check("nothing was downloaded or installed" in out, "ends by saying nothing was downloaded")
urls = [s[0] for s in seen]
check(len(seen) == 2 and urls[0] == mc.REPO_API and urls[1] == mc.LLAMA_API, "exactly two requests: the moe-cache list and the llama.cpp list")
h = {k.lower(): v for k, v in seen[0][1].items()}
check(h.get("accept") == "application/vnd.github+json" and h.get("user-agent", "").startswith("moe-cache/") and "authorization" not in h and seen[0][2] == 10,
      "request: GitHub accept header, moe-cache user agent, no credentials, 10 s timeout")

rc, out, seen = run("0.3.5", [{"tag_name": "v0.3.5"}], [{"tag_name": "b11966"}], ls_b=11966)
check("moe-cache 0.3.5 is the newest release" in out and "(up to date)" in out, "equal versions: newest release / up to date")

rc, out, seen = run("0.5.0", [{"tag_name": "v0.3.5"}], [{"tag_name": "b11966"}], ls_b=None)
check("is newer than the newest release (0.3.5): a development copy" in out and "llama.cpp" not in out, "a newer local version: development copy; no llama-server found: no llama line")

rc, out, seen = run("0.3.5", urllib.error.URLError("offline"), OSError("offline"), ls_b=11966)
check(rc == 0 and "could not read the newest release (offline?)" in out and "newest release could not be read" in out, "offline: both lines degrade to a message, exit 0")

err403 = urllib.error.HTTPError(mc.REPO_API, 403, "rate limit exceeded", {}, None)
rc, out, seen = run("0.3.5", err403, [{"tag_name": "b11966"}], ls_b=11966)
check(rc == 0 and "could not read the newest release" in out and "(up to date)" in out, "GitHub rate limit (403) on one list does not hide the other")

rc, out, seen = run("0.3.5", b"<html>not json</html>", b'{"message": "x"}', ls_b=11966)
check(rc == 0 and "could not read the newest release" in out, "garbled / unexpected JSON: a message, no traceback")

rc, out, seen = run("0.3.5", {"tag_name": "v0.3.6"}, [{"tag_name": "b11966"}], ls_b=11966)
check("moe-cache 0.3.6 is out" in out, "a single release object (not a list) is read too")

rc, out, seen = run("0.3.5", [], [], ls_b=11966)
check(rc == 0 and "could not read the newest release" in out, "empty release lists: a message")

print("update_check_test: " + ("PASS" if not fails else "FAILED: " + "; ".join(fails)))
sys.exit(1 if fails else 0)
