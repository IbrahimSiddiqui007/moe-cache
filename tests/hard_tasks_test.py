#!/usr/bin/env python3
"""MODELS-1141: tools/hard_tasks_run.py against a fake llama-server: right answers pass, wrong / missing answers fail, thinking tokens are counted up to </think>, a looping answer is flagged,
the cap is sent (and not sent when 0), a stop=limit answer counts as a cap hit; tools/hard_tasks.json is well formed (27 tasks, unique ids, every check kind known).
   python3 tests/hard_tasks_test.py"""
import http.server, json, os, subprocess, sys, tempfile, threading

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = json.load(open(os.path.join(HERE, "tools", "hard_tasks.json"), encoding="utf-8"))
fails = 0


def chk(name, cond):
    global fails
    print(("  ok   " if cond else "  FAIL ") + name)
    fails += 0 if cond else 1


BY_PROMPT = {t["prompt"]: t for t in TASKS}
SEEN = []


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self): self.send({"model_path": "fake.gguf", "build_info": "fake"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template": return self.send({"prompt": body["messages"][0]["content"]})
        if self.path == "/tokenize": return self.send({"tokens": list(range(len(body["content"].split())))})
        SEEN.append(body["n_predict"]); t = BY_PROMPT[body["prompt"]]; i = TASKS.index(t); ans = t["check"]["answer"]
        if i == 0: text, toks, stop = "word " * 40 + "</think>\n\nANSWER: " + ans, list(range(46)), "eos"                      # right, 40 thinking tokens
        elif i == 1: text, toks, stop = "hmm </think>\n\nANSWER: 000", list(range(5)), "eos"                              # wrong
        elif i == 2: text, toks, stop = "no answer line </think> done", list(range(5)), "eos"                              # no ANSWER line
        elif i == 3: text, toks, stop = "loop " * 50, [1, 2, 3] * 30, "limit"                                              # loops, never leaves thinking, cap hit
        else: text, toks, stop = "ok </think>\n\nANSWER: " + ans, list(range(6)), "eos"
        self.send({"content": text, "tokens": toks, "stop_type": stop})


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_port}"; out = os.path.join(tempfile.mkdtemp(), "run.json")
r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "tools", "hard_tasks_run.py"), "run", "--url", url, "--out", out, "--label", "fake", "--cap", "0", "--workers", "3"], capture_output=True, text=True)
chk("the runner exits 0", r.returncode == 0)
items = json.load(open(out))["items"]
chk("one item per task, in order", [x["id"] for x in items] == [t["id"] for t in TASKS])
chk("a right answer passes", items[0]["pass"] and items[4]["pass"] and all(x["pass"] for x in items[4:]))
chk("a wrong answer fails", not items[1]["pass"] and items[1]["answer"] == "000")
chk("no ANSWER line fails with answer None", not items[2]["pass"] and items[2]["answer"] is None)
chk("thinking tokens are counted up to </think>", items[0]["think_tokens"] == 40 and items[1]["think_tokens"] == 1)
chk("a looping answer is flagged and a clean one is not", items[3]["loop"] and not items[0]["loop"])
chk("stop=limit is a cap hit and all its tokens are thinking", items[3]["cap_hit"] and items[3]["think_tokens"] == items[3]["n_tokens"] and not items[0]["cap_hit"])
chk("cap 0 sends n_predict -1", set(SEEN) == {-1})
SEEN.clear()
r2 = subprocess.run([sys.executable, "-I", os.path.join(HERE, "tools", "hard_tasks_run.py"), "run", "--url", url, "--out", out, "--cap", "512", "--only", TASKS[0]["id"]], capture_output=True, text=True)
chk("a cap of 512 is sent as n_predict 512", r2.returncode == 0 and SEEN == [512])
s = subprocess.run([sys.executable, "-I", os.path.join(HERE, "tools", "hard_tasks_run.py"), "summary", out], capture_output=True, text=True)
chk("summary prints pass count and thinking tokens", "1/1 pass" in s.stdout and "thinking tokens" in s.stdout)
ids = [t["id"] for t in TASKS]
chk("the task file: 27 tasks, unique ids, known check kinds, an ANSWER instruction in every prompt", len(TASKS) == 27 and len(set(ids)) == 27 and all(t["check"]["kind"] in ("int", "word", "exact") and "ANSWER:" in t["prompt"] and t["why_hard"] for t in TASKS))
print("hard_tasks_test: " + ("FAIL" if fails else "PASS"))
sys.exit(1 if fails else 0)
