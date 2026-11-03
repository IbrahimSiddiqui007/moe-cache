#!/usr/bin/env python3
"""GUI-016: the page's agent guess and "restored" marking on real chats, without a model.
The texts are what the real OpenCode and Hermes sent (tests/agent_sessions/real_prompts.json, captured by tools/capture_agent_prompts.py in a temp
HOME against a fake OpenAI server). Here they become saved slot files ('ggsq' header + tokens) of a fake llama-server that detokenizes them, and the
page's Runner.sessions() reads them as it does on a real machine. With MOE_REAL_AGENTS=1 and OpenCode / Hermes installed, the capture tool also runs
live and its result is compared with the fixture's first words.
   python3 tests/agent_guess_test.py"""
import http.server, json, os, shutil, struct, subprocess, sys, tempfile, threading

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-guess-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_ui as ui  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

real = json.load(open(os.path.join(HERE, "tests", "agent_sessions", "real_prompts.json"), encoding="utf-8"))
reqs = real["requests"]
get = lambda agent, kind: next(r for r in reqs if r["agent"] == agent and r["kind"] == kind)
check({(r["agent"], r["kind"]) for r in reqs} == {("OpenCode", "main"), ("OpenCode", "title"), ("Hermes", "main"), ("Hermes", "title")}, "fixture: main and title request of both agents")

# a chat template in front, as llama-server applies it; 4 characters per token (real tokenizers average 3-4: this is the strict side)
def convo(r):
    return "<|im_start|>system\n" + r["system"] + "<|im_end|>\n<|im_start|>user\n" + r["first_user"] + "<|im_end|>\n"
VOCAB, INDEX = [], {}
def tokenize(text):
    ids = []
    for i in range(0, len(text), 4):
        piece = text[i:i + 4]
        if piece not in INDEX:
            INDEX[piece] = len(VOCAB); VOCAB.append(piece)
        ids.append(INDEX[piece])
    return ids

# ---- guess on the real texts, as the page reads them: the first 600 tokens ----
def first600(r):
    return "".join(VOCAB[i] for i in tokenize(convo(r))[:600])
check(ui.guess_agent(first600(get("OpenCode", "main"))) == "OpenCode", "real OpenCode conversation -> OpenCode")
check(ui.guess_agent(first600(get("Hermes", "main"))) == "Hermes", "real Hermes conversation -> Hermes")
check(ui.guess_agent(first600(get("OpenCode", "title"))) is None and ui.guess_agent(first600(get("Hermes", "title"))) is None,
      "their title-making requests name no agent -> no guess (a slot that only made a title stays unlabelled)")
h = get("Hermes", "main")["system"].lower()
check("opencode" in h and "claude code" in h and h.find("hermes") < h.find("opencode") and h.find("hermes") < h.find("claude code"),
      f"Hermes' own prompt also names OpenCode (char {h.find('opencode')}) and Claude Code (char {h.find('claude code')}): the first name must win")
check(ui.guess_agent(get("Hermes", "main")["system"]) == "Hermes" and ui.guess_agent(get("OpenCode", "main")["system"]) == "OpenCode", "whole prompts: still the right agent")
check(ui.guess_agent("Run opencode ... you are Hermes") == "OpenCode" and ui.guess_agent("you are Hermes, not opencode") == "Hermes", "earliest name wins, not list order")
check(ui.guess_agent("") is None and ui.guess_agent(None) is None, "empty text -> no guess")

# ---- the whole path: slot files on disk, a running (fake) server that detokenizes, Runner.sessions() ----
class Fake(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, d):
        b = json.dumps(d).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.reply([{"id": 0, "is_processing": False, "n_prompt_tokens": 3000, "n_ctx": 65536}]) if self.path == "/slots" else self.send_response(404)
    def do_POST(self):
        q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.reply({"content": "".join(VOCAB[i] for i in q["tokens"])})
fake = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=fake.serve_forever, daemon=True).start()

model = os.path.join(TMP, "Fake-Model-Q4_K_M.gguf")
open(model, "wb").write(b"GGUF" + b"\0" * 1000)
settings = {"model": model, "ctx": 65536, "agent": True, "port": fake.server_address[1]}
tag = ui.session_tag(settings)
sd = ui.slot_dir(); os.makedirs(sd, exist_ok=True)
def slot_file(name, text):
    ids = tokenize(text)
    with open(os.path.join(sd, name), "wb") as f:
        f.write(struct.pack("<III", 0x67677371, 4, len(ids)) + struct.pack(f"<{len(ids)}i", *ids) + b"\0" * 2000)
slot_file(tag + "-s0.bin", convo(get("OpenCode", "main")))
slot_file(tag + "-s1.bin", convo(get("Hermes", "main")))
slot_file(tag + "-s2.bin", convo(get("Hermes", "title")))
slot_file(tag + "-s3.bin", convo(get("OpenCode", "main")))

# the launcher's own words for restoring (bin/moe_cache.py): the page matches them with regexes, so keep the two in step
src = open(os.path.join(HERE, "bin", "moe_cache.py"), encoding="utf-8").read()
check("warm session: restored a saved agent conversation (" in src and ") into slot " in src and "warm session: could not restore " in src,
      "the launcher still prints 'restored ... into slot N' and 'could not restore <file>' (the page reads both)")
r = ui.Runner()
r.state, r.settings, r.auto_key = "ready", settings, None
r.warm = [{"t": 1, "line": "restored a saved agent conversation (3000 tokens) into slot 0 in 1.2 s"},
          {"t": 2, "line": f"could not restore {tag}-s3.bin (HTTP 400): the slot file does not match this model"}]
ss = r.sessions()
by = {f["name"]: f for f in ss["files"]}
check(len(by) == 4 and all(f["this_server"] for f in by.values()), "all four slot files belong to this server")
check(by[tag + "-s0.bin"]["agent"] == "OpenCode" and by[tag + "-s1.bin"]["agent"] == "Hermes" and by[tag + "-s2.bin"]["agent"] is None,
      f"agent column: slot 0 OpenCode, slot 1 Hermes, a title-only slot none ({[by[tag + f'-s{i}.bin']['agent'] for i in range(3)]})")
check(by[tag + "-s0.bin"]["restored"] and not by[tag + "-s1.bin"]["restored"], "restored marking: slot 0 only")
check(by[tag + "-s3.bin"]["restore_failed"] and not by[tag + "-s0.bin"]["restore_failed"], "a file the launcher could not restore is marked")
check(by[tag + "-s3.bin"]["agent"] == "OpenCode" and by[tag + "-s3.bin"]["tokens"] > 600, "the token count is the file's")
r.state = "stopped"
ss = r.sessions()
check(all(f["agent"] is None for f in ss["files"]) and not any(f["restored"] for f in ss["files"]), "model stopped: no guess (the server must detokenize) and nothing marked restored")

# ---- live: the real agents, if installed (MOE_REAL_AGENTS=1) ----
if os.environ.get("MOE_REAL_AGENTS") == "1" and shutil.which("opencode") and shutil.which("hermes"):
    out = os.path.join(TMP, "live.json")
    subprocess.run([sys.executable, os.path.join(HERE, "tools", "capture_agent_prompts.py"), "--out", out], check=True, capture_output=True, timeout=600)
    live = json.load(open(out, encoding="utf-8"))["requests"]
    for agent in ("OpenCode", "Hermes"):
        m = next(x for x in live if x["agent"] == agent and x["kind"] == "main")
        check(ui.guess_agent(first600(m)) == agent, f"live {agent}: the real prompt is still recognised")
else:
    print("  skip live agents (set MOE_REAL_AGENTS=1 with opencode and hermes installed)")

shutil.rmtree(TMP, ignore_errors=True)
print("\nFAILED: " + "; ".join(fails) if fails else "\nagent guess tests passed")
sys.exit(1 if fails else 0)
