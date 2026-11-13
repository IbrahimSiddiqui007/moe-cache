#!/usr/bin/env python3
"""TEST-1142: bin/moe_cache_suite_run.py against a fake llama-server and a fake llama-perplexity: the speed probe, the task sections, the hard section, the KL section, a whole `quick` run into a private record,
the conversation converter, the CLI entry (`moe_cache.py bench-suite`), and that records are git-ignored (benchmarks/suites/).
   python3 tests/suite_run_test.py"""
import http.server, json, os, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_suite as S, moe_cache_suite_run as R      # noqa: E402
HARD = {t["prompt"]: t for t in json.load(open(os.path.join(HERE, "tools", "hard_tasks.json"), encoding="utf-8"))}
fails = 0
REQS = []; FAIL = [None]


def chk(name, cond):
    global fails
    print(("  ok   " if cond else "  FAIL ") + name)
    fails += 0 if cond else 1


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self): self.send({"model_path": "fake.gguf", "build_info": "fake-b1", "status": "ok", "total_slots": 4, "default_generation_settings": {"n_ctx": 12288}})

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template": return self.send({"prompt": b["messages"][0]["content"]})
        if self.path == "/tokenize": return self.send({"tokens": list(range(len(b["content"]) // 4))})
        p = b["prompt"]; REQS.append(self.path)
        if p == FAIL[0]:
            self.send_response(500); self.end_headers(); return
        if p in HARD: return self.send({"content": "x</think>\n\nANSWER: " + HARD[p]["check"]["answer"], "tokens": [1, 2, 3, 4], "stop_type": "eos"})
        self.send({"content": "391", "tokens": [1, 2], "stop_type": "eos", "timings": {"prompt_per_second": 321.0, "predicted_per_second": 12.5, "prompt_ms": 80.0, "predicted_ms": 20.0}})


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_port}"; tmp = tempfile.mkdtemp()

sp = R.run_speed_probe(url, 4096)
chk("speed probe: one band at the asked depth (with the tokenizer's own count), the server's tok/s and a TTFT, marked 1 run", [(b["depth"], b["prompt_tps"], b["decode_tps"]) for b in sp["bands"]] == [(4096, 321.0, 12.5)] and sp["bands"][0]["tokens"] > 0 and sp["ttft_ms"] == 100 and sp["decode_b1"] == 12.5 and sp["runs"] == 1)
chk("speed probe protocol hash depends on the depth", R.run_speed_probe(url, 4096)["protocol"] == sp["protocol"] and R.run_speed_probe(url, 8192)["protocol"] != sp["protocol"])
t = R.run_tasks(url, "off", 5, tmp)
chk("tasks section: 5 tasks run, the fake answers 391 so one question about 391 passes at most, think=off", t["n"] == 5 and t["think"] == "off" and t["cap"] == 64 and 0 <= t["pass"] <= 5)
h = R.run_hard(url, 0, 2, tmp, 65536)
chk("hard section: 27 tasks, all right with the fake, no loops, no cap hits, no errors", h["n"] == 27 and h["pass"] == 27 and h["loops"] == 0 and h["cap_hits"] == 0 and h["cap"] == 0 and h["errors"] == 0)
h2 = R.run_hard(url, 0, 2, tmp, 32768)
chk("hard protocol hash carries the served context (no cap = the context is the cap)", h["protocol"] != h2["protocol"])
FAIL[0] = next(iter(HARD)); tmp2 = tempfile.mkdtemp(); h3 = R.run_hard(url, 0, 2, tmp2, 65536); FAIL[0] = None
chk("one HTTP error in the hard set is that task's error, not the end of the run: 26 scored, 1 error", h3["n"] == 26 and h3["errors"] == 1 and h3["pass"] == 26)
n0 = len(REQS); h4 = R.run_hard(url, 0, 2, tmp2, 65536)
chk("resume: the second run redoes only the errored task", len(REQS) - n0 == 1 and h4["n"] == 27 and h4["errors"] == 0)
# fake perplexity: logs its arguments; prints KL lines for the second (--kl-divergence) call
fp = os.path.join(tmp, "fakeppl"); lg = os.path.join(tmp, "ppl.log")
open(fp, "w").write("#!/bin/sh\necho \"$*\" >> " + lg + "\ncase \"$*\" in *--kl-divergence-base*) echo 'Mean    KLD:   0.0123 +- 0.001'; echo 'Median  KLD:   0.0010'; echo '90.0%   KLD:   0.0300'; echo 'Same top p: 96.500 +- 0.1 %';; esac\n"); os.chmod(fp, 0o755)
served = {"ctk": "q8_0", "ctv": "q4_0", "plan_flags": ["-ngl", "20", "--n-cpu-moe", "3"]}
q = R.run_quality("model.gguf", served, 2, tmp, fp)
calls = open(lg).read().splitlines()
chk("quality section from the fake llama-perplexity: KL of the SERVED KV (q8_0/q4_0), reference f16", q["kl_mean"] == 0.0123 and q["top1_pct"] == 96.5 and q["windows"] == 2 and "q8_0/q4_0" in q["ref"] and q["served_kv"] == "q8_0/q4_0")
chk("the reference run uses f16 KV and the served plan flags; the candidate run the served KV types and the same plan flags", len(calls) == 2 and "-ctk f16 -ctv f16" in calls[0] and "-ctk q8_0 -ctv q4_0" in calls[1] and all("-ngl 20 --n-cpu-moe 3" in c for c in calls))
chk("the token count is the scored half of the windows (not windows x 4096), the window tokens are recorded too", q["tokens"] == 2 * 8192 // 2 - 2 and q["window_tokens"] == 2 * 8192)
chk("a served f16 KV cache has nothing to measure: no quality section", R.run_quality("model.gguf", {"ctk": "f16", "ctv": "f16"}, 2, tmp, fp) is None)
try:
    R.run_quality("model.gguf", {}, 2, tmp, fp); unk = False
except RuntimeError:
    unk = True
chk("unknown served KV types: refuses instead of guessing", unk)
wt = R.chat_windows_text(20000)
chk("chat-style windows: deterministic, 'User:'/'Assistant:' turns", wt == R.chat_windows_text(20000) and wt.startswith("User: ") and "\n\nAssistant: " in wt)
# conversation converter: a REAL tools/conversation_bench.py run (tests/data/conv_run_sample.json, captured on a laptop with --monitor-pid; turns trimmed)
conv = json.load(open(os.path.join(HERE, "tests", "data", "conv_run_sample.json")))
spd, mem = R.sections_from_conversation(conv, conv["ctx"])
chk("real conversation output: bands by depth, TTFT, final depth", len(spd["bands"]) >= 1 and spd["final_depth"] == conv["final_depth"] and spd.get("ttft_ms"))
chk("real conversation output: the memory section reads the monitor block (RSS and VRAM in GB)", mem is not None and mem["rss_gb"] == round(conv["monitor"]["process_ram_peak_mib"] / 1024, 2) and mem["vram_gb"] > 0)
nm = dict(conv); nm.pop("monitor"); chk("no monitor block (no pid): no memory section, not invented", R.sections_from_conversation(nm, conv["ctx"])[1] is None)
# helpers: served plan flags, command line of a process, free port, kill_tree, task subset
chk("plan_flags keeps where layers and experts sit, not the KV types or the port", R.plan_flags(["llama-server", "-m", "x.gguf", "-ngl", "99", "--n-cpu-moe", "5", "-ctk", "q8_0", "--port", "8140", "--cpu-moe", "-ot", "a=b"]) == ["-ngl", "99", "--n-cpu-moe", "5", "--cpu-moe", "-ot", "a=b"])
if os.name != "nt":
    chk("proc_cmdline reads a process's arguments", R.proc_cmdline(os.getpid()) and "suite_run_test" in " ".join(R.proc_cmdline(os.getpid())))
import socket
bs = socket.socket(); bs.bind(("127.0.0.1", 0)); busy = bs.getsockname()[1]
chk("free_port: a busy port is not reused (a leftover server must not answer for ours)", R.free_port(busy) != busy)
bs.close()
if os.name != "nt":
    pp = subprocess.Popen(["sh", "-c", "sleep 300 & sleep 300 & wait"], start_new_session=True); time.sleep(0.5)
    import moe_cache_record as rr
    kids = rr.probe().tree(pp.pid); R.kill_tree(pp.pid, grace=5); time.sleep(0.5)
    chk("kill_tree stops the process and its children", len(kids) >= 3 and not any(os.path.exists(f"/proc/{k}") and open(f"/proc/{k}/stat").read().split()[2] != "Z" for k in kids))
sp27 = {"bands": [{"depth": 65536, "decode_tps": 27.0}]}
chk("hard_budget: 65536 ctx at 27 tok/s -> 1.5 x 2427 s = 3640 s; a small ctx is held at 30 min; unknown speed -> 3600; capped at 6 h", R.hard_budget(sp27, 65536) == 3640 and R.hard_budget(sp27, 8192) == 1800 and R.hard_budget(None, 65536) == 3600 and R.hard_budget({"bands": [{"decode_tps": 0.5}]}, 65536) == 21600)
ts = R.task_subset(list(range(202)), 20)
chk("task subset: 20 of 202 spread over the whole file (the last tasks are used too)", len(ts) == 20 and ts[0] == 0 and ts[-1] > 185 and ts == sorted(set(ts)))
sv = R.served_facts(url)
chk("served facts from /props: ctx, slots, build (no process: no KV types claimed)", sv["ctx"] == 12288 and sv["np"] == 4 and sv["build"] == "fake-b1" and "ctk" not in sv)
# a whole quick run against the fake server (no launch)
rec = R.measure(url, "fake-model", "Q3", "quick", None, tempfile.mkdtemp(), 1, fp, progress=lambda m: None)
chk("quick measure: speed + tasks sections, valid, private, the served block is in the record", S.validate_record(rec) == [] and set(rec["sections"]) == {"speed", "tasks"} and rec["published"] is False and rec["served"]["ctx"] == 12288 and rec["served"]["np"] == 4)
chk("the record names the machine and the kit", rec["machine"].get("os") and rec["software"].get("kit") is not None)
# partial results: a section already saved is not measured again
pt = R.Partial(tempfile.mkdtemp(), "fake", "quick", {"k": 1}); pt.data["sections"]["speed"] = {"protocol": "x", "bands": []}; pt.save(); pt2 = R.Partial(os.path.dirname(pt.path), "fake", "quick", {"k": 1})
chk("a partial file with the same key loads; another key (other settings) does not", pt2.load() and not R.Partial(os.path.dirname(pt.path), "fake", "quick", {"k": 2}).load())
n0 = len(REQS); R.measure_server(url, "fake", "quick", tempfile.mkdtemp(), pt2, served=R.served_facts(url), progress=lambda m: None)
chk("resume: speed is kept, tasks are measured, and each finished section is saved at once", "tasks" in pt2.data["sections"] and pt2.data["sections"]["speed"]["protocol"] == "x" and json.load(open(pt2.path))["sections"].get("tasks"))
# R2: an errored hard section is not stored as done; a dead server stops the sections
_ho = dict(R.PROFILES["night"]); R.PROFILES["_t"] = dict(_ho, speed="probe", think=[], hard=True, kl_chunks=0)
FAIL[0] = next(iter(HARD)); _pt = R.Partial(tempfile.mkdtemp(), "fake", "_t", {"k": 9}); _w = tempfile.mkdtemp()
try:
    R.measure_server(url, "fake", "_t", _w, _pt, served=R.served_facts(url), progress=lambda m: None); _ok = False
except RuntimeError as e:
    _ok = "not finished" in str(e) and "error" in str(e)
FAIL[0] = None
chk("R2: a hard set with an errored task raises and stores no 'hard' section (the speed section is kept)", _ok and "hard" not in _pt.data["sections"] and "speed" in _pt.data["sections"])
R.measure_server(url, "fake", "_t", _w, _pt, served=R.served_facts(url), progress=lambda m: None)
chk("R2: the rerun resumes (only the errored task is redone) and stores the finished hard section", _pt.data["sections"].get("hard", {}).get("n") == 27 and _pt.data["sections"]["hard"]["errors"] == 0)
_pt2 = R.Partial(tempfile.mkdtemp(), "fake", "quick", {"k": 8}); _orig = R.server_alive; R.server_alive = lambda u, timeout=5: False
try:
    R.measure_server(url, "fake", "quick", tempfile.mkdtemp(), _pt2, served=R.served_facts(url), progress=lambda m: None); _ok = False
except RuntimeError as e:
    _ok = "does not answer" in str(e)
R.server_alive = _orig
chk("R2: a server that does not answer stops the sections and nothing is stored", _ok and not _pt2.data["sections"])
del R.PROFILES["_t"]
out = tempfile.mkdtemp(); r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "bench-suite", "run", "--url", url, "--name", "fake-model", "--quant", "Q3", "--profile", "quick", "--out", out], capture_output=True, text=True)
files = os.listdir(out)
chk("`moe_cache.py bench-suite run --url` writes one valid private record", r.returncode == 0 and len(files) == 1 and S.validate_record(json.load(open(os.path.join(out, files[0])))) == [])
r2 = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "bench-suite", "compare", os.path.join(out, files[0]), os.path.join(out, files[0])], capture_output=True, text=True)
chk("`moe_cache.py bench-suite compare` works", r2.returncode == 0 and "decode at 8k" in r2.stdout)
chk("benchmarks/suites/ is git-ignored (private by default)", "benchmarks/suites/" in open(os.path.join(HERE, ".gitignore")).read())
chk("profiles: quick has no KL and no hard; night has the hard set", R.PROFILES["quick"]["kl_chunks"] == 0 and not R.PROFILES["quick"]["hard"] and R.PROFILES["night"]["hard"])
print("suite_run_test: " + ("FAIL" if fails else "PASS"))
sys.exit(1 if fails else 0)
