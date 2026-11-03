#!/usr/bin/env python3
"""`moe_cache.py connect NAME --check` against a small fake server (standard library http.server), every agent's path: it answers correctly, refuses the
key, omits tool calls, returns a malformed tool call, has a smaller context, lacks --jinja or an endpoint, or is not running. No model, temporary HOME.
   python3 tests/connect_check_test.py"""
import contextlib, http.server, io, json, os, sys, tempfile, threading

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-check-test-")
HOME = os.path.join(TMP, "home"); os.makedirs(HOME)
WORK = os.path.join(TMP, "work"); os.makedirs(WORK)
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
for v in ("CODEX_HOME", "PI_CODING_AGENT_DIR", "LLAMA_API_KEY", "XDG_CONFIG_HOME", "HERMES_HOME", "OPENCODE_CONFIG"):
    os.environ.pop(v, None)
os.chdir(WORK)
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc  # noqa: E402

KEY, MODEL = "k3y-abc123", "fake-moe"
MODE = {"m": "ok", "ctx": 65536}
fails = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def authed(self):
        if MODE["m"] == "badkey" and self.headers.get("Authorization") != "Bearer " + KEY:
            self.send(401, {"error": {"message": "Invalid API Key"}})
            return False
        return True

    def do_GET(self):
        if not self.authed():
            return
        if self.path == "/v1/models":
            self.send(200, {"data": [{"id": MODEL}]})
        elif self.path == "/props":
            self.send(200, {"default_generation_settings": {"n_ctx": MODE["ctx"]}})
        else:
            self.send(404, {"error": "nope"})

    def do_POST(self):
        if not self.authed():
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        proto = {"/v1/chat/completions": "chat", "/v1/responses": "responses", "/v1/messages": "messages"}.get(self.path)
        if not proto or (MODE["m"] == "noendpoint" and proto != "chat"):
            return self.send(404, {"error": "not found"})
        if body.get("model") != MODEL:
            return self.send(400, {"error": {"message": "model not found"}})
        raw = json.dumps(body)
        tools = bool(body.get("tools"))
        if tools and MODE["m"] == "nojinja":
            return self.send(500, {"error": {"message": "tools param requires --jinja flag"}})
        got_result = any(t in raw for t in ('"role": "tool"', "function_call_output", "tool_result"))
        call = tools and not got_result and MODE["m"] != "notools"
        args = '{"path": "notes.txt"}' if MODE["m"] != "malformed" else '{"path": '
        ev = []
        if proto == "chat":
            if call:
                ev = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": args[:8]}}]}}]},
                      {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": args[8:]}}]}}]}]
            else:
                ev = [{"choices": [{"delta": {"content": "Hello there friend"}}]}]
        elif proto == "responses":
            if call:
                ev = [{"type": "response.output_item.added", "item": {"type": "function_call", "id": "i1", "call_id": "c1", "name": "read_file", "arguments": ""}},
                      {"type": "response.function_call_arguments.delta", "item_id": "i1", "delta": args},
                      {"type": "response.output_item.done", "item": {"type": "function_call", "id": "i1", "call_id": "c1", "name": "read_file", "arguments": args}}]
            else:
                ev = [{"type": "response.output_text.delta", "delta": "Hello there friend"}, {"type": "response.completed"}]
        else:
            if call:
                ev = [{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "c1", "name": "read_file", "input": {}}},
                      {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": args[:8]}},
                      {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": args[8:]}}]
            else:
                ev = [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello there friend"}}]
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for e in ev:
            self.wfile.write(b"data: " + json.dumps(e).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_port}"
mc.key_for = lambda url, given=None: given or KEY
AGENTS = ["hermes", "opencode", "claude", "codex", "pi", "openclaw", "aider", "qwen"]


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = mc.connect(list(argv))
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


def result(agent, *extra):
    rc, out, err = run(agent, "--check", "--json", *extra)
    try:
        return rc, json.loads(out)
    except ValueError:
        print(out, err)
        return rc, {"steps": [], "verdict": out + err, "ok": False}


def steps(j):
    return {s["step"]: s for s in j["steps"]}


def setup(agent):
    """Write the agent's config the way `connect --write` does (hermes: by hand, it needs the hermes program)."""
    MODE.update(m="ok", ctx=65536)
    if agent == "hermes":
        os.makedirs(os.path.join(HOME, ".hermes"), exist_ok=True)
        open(os.path.join(HOME, ".hermes", "config.yaml"), "w").write(
            f"model:\n  provider: custom\n  base_url: {URL}/v1\n  default: {MODEL}\n  context_length: 65536\n  api_key: {KEY}\nother:\n  base_url: x\n")
        return 0
    return run(agent, "--write", "--url", URL, "--api-key", KEY)[0]


for ag in AGENTS:
    print(f"-- {ag}")
    check(setup(ag) == 0, f"{ag}: config written")
    rc, j = result(ag)
    st = steps(j)
    check(rc == 0 and j["ok"] and all(s["result"] in ("pass", "skip") for s in j["steps"]), f"{ag}: all steps pass ({[(s['step'], s['result']) for s in j['steps']]})")
    check(st["config"]["result"] == "pass" and j["config_written"] and j["url"] == URL and j["model"] == MODEL, f"{ag}: URL and model read from the agent's own config")
    check(st["models"]["result"] == "pass" and st["chat"]["result"] == "pass", f"{ag}: models and chat pass ({j['protocol']})")
    check(st["tool call"]["result"] == ("skip" if ag == "aider" else "pass"), f"{ag}: tool call " + ("skipped, text only" if ag == "aider" else "round trip passes"))
    check(st["window"]["result"] in ("pass",) or ag == "claude" and st["window"]["result"] == "pass", f"{ag}: window fits ({st['window']['detail']})")
    check(j["protocol"] == {"codex": "responses", "claude": "messages"}.get(ag, "chat"), f"{ag}: protocol {j['protocol']}")
    # failures
    if ag != "aider":
        for mode, step, word in (("notools", "tool call", "--agent"), ("malformed", "tool call", "malformed"), ("nojinja", "tool call", "--jinja")):
            MODE["m"] = mode
            rc, j = result(ag)
            check(rc == 1 and steps(j)[step]["result"] == "fail" and word in (j["verdict"] + steps(j)[step]["detail"]), f"{ag}: {mode} -> tool call fails with a fix ({j['verdict'][:70]})")
    if ag in ("codex", "claude"):
        MODE["m"] = "noendpoint"
        rc, j = result(ag)
        check(rc == 1 and steps(j)["chat"]["result"] == "fail" and "no /v1/" in j["verdict"], f"{ag}: missing endpoint named")
        check(steps(j)["tool call"]["result"] == "skip", f"{ag}: later steps skipped after the first failure")
    # window: the agent's config says 57344 (hermes 65536), the server has less
    if ag != "claude":
        MODE.update(m="ok", ctx=8192)
        rc, j = result(ag)
        check(rc == 1 and steps(j)["window"]["result"] == "fail" and "window" in j["verdict"], f"{ag}: smaller server context -> window fails ({j['verdict'][:80]})")
    else:
        MODE.update(m="ok", ctx=8192)
        rc, j = result(ag)
        check(rc == 1 and steps(j)["window"]["result"] == "fail", "claude: server context under 64000 fails")
    # key: the server now wants KEY, the config holds another one
    MODE.update(m="badkey", ctx=65536)
    cfgs = {"codex": ".codex/config.toml", "pi": ".pi/agent/models.json", "openclaw": ".openclaw/openclaw.json", "qwen": ".qwen/settings.json",
            "aider": ".aider.moe-cache.conf.yml", "opencode": ".config/opencode/opencode.json", "hermes": ".hermes/config.yaml", "claude": None}
    path = os.path.join(WORK, ".claude", "settings.local.json") if ag == "claude" else os.path.join(HOME, cfgs[ag])
    good = open(path).read()
    open(path, "w").write(good.replace(KEY, "wrong-key") if ag != "qwen" else good.replace(KEY, "wrong-key"))
    rc, j = result(ag)
    check(rc == 1 and steps(j)["models"]["result"] == "fail" and "key" in j["verdict"], f"{ag}: wrong key -> models fails, fix names the key ({j['verdict'][:70]})")
    open(path, "w").write(good)
    # model name: config names another model
    MODE["m"] = "ok"
    mpath = os.path.join(os.path.dirname(path), "settings.json") if ag == "pi" else path      # Pi names its default model in settings.json
    mgood = open(mpath).read()
    open(mpath, "w").write(mgood.replace(MODEL, "other-model"))
    rc, j = result(ag)
    check(rc == 1 and "name" in j["verdict"] and steps(j)["models"]["result"] == "fail", f"{ag}: wrong model name -> mismatch named ({j['verdict'][:70]})")
    open(mpath, "w").write(mgood)

print("-- no server / no config yet")
srv.shutdown(); srv.server_close()
rc, j = result("codex", "--url", URL)
check(rc == 1 and steps(j)["models"]["result"] == "fail" and "server MODEL.gguf --agent" in j["verdict"] and "on-demand" in j["verdict"], "nothing answers -> how to start a server")
os.remove(os.path.join(HOME, ".codex", "config.toml"))
rc, j = result("codex", "--url", URL)
check(steps(j)["config"]["result"] == "skip" and not j["config_written"], "no config yet: reported as a skip, not a failure of the config")
rc, out, _ = run("codex", "--check", "--url", URL)
check(rc == 1 and "step" in out and "result" in out and "FAIL" in out, "table output has the columns and a verdict")
rc, out, err = run("aider", "--json")
check(rc not in (0, None), "--json without --check is refused")

print("FAIL" if fails else "ok", len(fails), "failed" if fails else "")
sys.exit(1 if fails else 0)
