#!/usr/bin/env python3
"""Tests for the warm-session saver (bin/moe_cache.py warm_keeper, warm_save_wait; idea N27) with a fake llama-server: no model; Linux and Windows.

  1. warm_save_wait: the old 20 s for small files; +20 s per GB; small growth waits 180 s + 30 s per GB, at most 10 min; new or shorter conversations count as big
  2. warm_keeper: a restored file sets the slot's saved size (a small growth after a restore is not saved at once); a big growth is saved;
     nothing is saved while a slot is processing; every save goes to .new.bin first and replaces the slot's file

   python3 tests/warm_test.py
"""
import json, os, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="warm-test-")
os.environ["XDG_CACHE_HOME"] = TMP
os.environ["XDG_CONFIG_HOME"] = TMP
sys.path.insert(0, os.path.join(ROOT, "bin"))
import moe_cache as mc  # noqa: E402

fails = 0


def check(cond, what):
    global fails
    print(("ok   " if cond else "FAIL ") + what)
    fails += 0 if cond else 1


# 1. the wait rule
W = mc.warm_save_wait
check(W(5000, None, None) == 20, "first save, size unknown: 20 s (the old idle_polls x poll)")
check(abs(W(100_000, None, 70_000) - (20 + 20 * 7.0)) < 1e-6, "first save of a 7 GB slot: 20 s + 20 s per GB = 160 s")
check(abs(W(100_500, 100_000, 70_000) - (180 + 30 * 7.035)) < 1e-6, "small growth (500 tokens) of a 7 GB slot: 180 s + 30 s per GB")
check(W(260_000, 250_000, 70_000) == 600, "small growth of an 18 GB slot: capped at 600 s")
check(W(130_000, 100_000, 70_000) == 20 + 20 * 9.1, "growth by a quarter: big change")
check(W(8_000, 2_500, 1_000) == 20 + 20 * 0.008, "growth by 4096+ tokens: big change")
check(W(40_000, 100_000, 70_000) == 20 + 20 * 2.8, "shorter conversation (a new one in the slot): big change")
check(W(3_000, 2_500, None) == 180, "small growth with the size unknown: 180 s")

# 2. warm_keeper against a fake server
SLOTDIR = os.path.join(TMP, "slots")
os.makedirs(SLOTDIR)
BPT = 10                                     # bytes per token of the fake slot files
state = {"slots": [{"id": 0, "id_task": 1, "n_prompt_tokens": 1000, "is_processing": False}], "saves": [], "restores": []}
lock = threading.Lock()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"status": "ok"})
        if self.path == "/slots":
            with lock:
                return self.reply(200, state["slots"])
        self.reply(404, {})

    def do_POST(self):
        u = urlparse(self.path)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        sid = int(u.path.rsplit("/", 1)[1])
        action = parse_qs(u.query).get("action", [""])[0]
        with lock:
            slot = next(x for x in state["slots"] if x["id"] == sid)
            if action == "save":
                with open(os.path.join(SLOTDIR, body["filename"]), "wb") as f:
                    f.write(b"\0" * (slot["n_prompt_tokens"] * BPT))
                state["saves"].append((sid, slot["n_prompt_tokens"], body["filename"]))
                return self.reply(200, {"n_saved": slot["n_prompt_tokens"]})
            if action == "restore":
                n = os.path.getsize(os.path.join(SLOTDIR, body["filename"])) // BPT
                slot["n_prompt_tokens"] = n
                state["restores"].append((sid, n))
                return self.reply(200, {"n_restored": n})
        self.reply(404, {})


srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

TAG = "m-c65536"
with open(os.path.join(SLOTDIR, TAG + "-s0.bin"), "wb") as f:     # a saved conversation of 20,000 tokens from an earlier run
    f.write(b"\0" * (20_000 * BPT))
state["slots"][0]["n_prompt_tokens"] = 0
POLL = 0.05
threading.Thread(target=mc.warm_keeper, args=(port, None, SLOTDIR, TAG), kwargs={"poll": POLL, "idle_polls": 2, "min_tokens": 2048},
                 daemon=True).start()


def wait_for(cond, secs):
    t = time.time() + secs
    while time.time() < t:
        with lock:
            if cond():
                return True
        time.sleep(0.02)
    with lock:
        return cond()


check(wait_for(lambda: state["restores"] == [(0, 20_000)], 3), "restore at start puts the 20,000-token file back into slot 0")
with lock:                                   # the agent's next turn adds 300 tokens: a small change against the restored 20,000
    state["slots"][0].update(id_task=2, n_prompt_tokens=20_300)
check(not wait_for(lambda: state["saves"], 1.0), "small growth after a restore is not saved after the short wait (it waits 180 s)")
with lock:                                   # while a slot is processing nothing is saved, even a big change
    state["slots"][0].update(id_task=3, n_prompt_tokens=40_000, is_processing=True)
check(not wait_for(lambda: state["saves"], 0.5), "nothing is saved while a slot is processing")
with lock:
    state["slots"][0]["is_processing"] = False
check(wait_for(lambda: len(state["saves"]) == 1, 3), "a big growth (20,000 -> 40,000) is saved after the short wait")
with lock:
    s = list(state["saves"])
check(s and s[0][1] == 40_000 and s[0][2].endswith(".new.bin"), "the save goes to the temporary .new.bin name")
check(wait_for(lambda: os.path.getsize(os.path.join(SLOTDIR, TAG + "-s0.bin")) == 40_000 * BPT, 2) and
      not os.path.exists(os.path.join(SLOTDIR, TAG + "-s0.new.bin")), "the slot's file is replaced and the temporary file is gone")
check(not wait_for(lambda: len(state["saves"]) > 1, 0.5), "the same state is not saved twice")

srv.shutdown()
print("all warm-session tests passed" if not fails else f"{fails} warm-session test(s) FAILED")
sys.exit(1 if fails else 0)
