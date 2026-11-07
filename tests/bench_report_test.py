#!/usr/bin/env python3
"""bench-report (R2-1): report sections and order, backend line, privacy, fit rule, identity, cold/warm stamp, run order. No model, no server:
the doctor, the launcher's plan, the runs and llama-server's device list are replaced by fakes. Same file on Linux and Windows CI."""
import json, os, re as _re_l, socket, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
import moe_cache_bench as mb  # noqa: E402
import moe_cache_ui as ui  # noqa: E402
import moe_cache_record as rr  # noqa: E402

FAILS = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


# 1. backend line from --list-devices
dev = lambda *names: "Available devices:\n" + "".join(f"  {n}: Some GPU (16304 MiB, 15890 MiB free)\n" for n in names)
for names, want in ((("CUDA0",), "CUDA"), (("Vulkan0",), "Vulkan"), (("ROCm0",), "ROCm-HIP"), (("SYCL0",), "SYCL"), ((), "CPU only"),
                    (("Vulkan0", "Vulkan1"), "Vulkan"), (("CUDA0", "Vulkan0"), "CUDA + Vulkan")):
    check(mb.backend_of(dev(*names)) == want, f"backend: {list(names) or 'no device'} -> {want}")
for names, want in ((("MTL0",), "Metal"), (("MUSA0",), "MUSA"), (("CANN0",), "CANN"), (("RPC0",), "RPC (remote)"), (("Foo0",), "Foo")):
    check(mb.backend_of(dev(*names)) == want, f"backend: {list(names)} -> {want}")
check(mb.backend_of("ggml_vulkan: Found 1 devices:\nAvailable devices:\n  CPU: x\n") == "CPU only", "backend: a CPU line is not a GPU")
check(mb.backend_of("") == "UNKNOWN - please state it" and mb.backend_of("error: unknown option") == "UNKNOWN - please state it", "backend: unreadable -> UNKNOWN")

# 2. order and plan summary
check(mb.order(2) == ["stock", "moe-cache", "moe-cache", "stock"] * 2, "order: stock, moe-cache, moe-cache, stock per pair")
line = "env MOE_CACHE_SIZE_GIB=auto llama-server -m x.gguf -c 8192 -ngl 99 --n-cpu-moe 30 -ub 512 -ctk q8_0"
check(mb.plan_summary(line) == "--n-cpu-moe 30, -ngl 99, -c 8192, -ub 512, -ctk q8_0, cache auto GiB", "plan summary from the launcher's command line")

# 3. identity verdicts
run = lambda mode, text, dec=10.0: {"mode": mode, "decode": dec, "prefill": 100.0, "text": text, "plugin_active": mode == "moe-cache"}
s = mb.summarize([run("stock", "a"), run("moe-cache", "a", 20.0), run("moe-cache", "a", 22.0), run("stock", "a")], exact=True)
check(s["identity"] == "IDENTICAL" and s["decode_x"] == 2.1 and s["moe-cache"]["decode_runs"] == [20.0, 22.0], "identity IDENTICAL, ratio 2.1x")
check(mb.summarize([run("stock", "a"), run("moe-cache", "b")], exact=True)["identity"] == "DIFFERENT", "identity DIFFERENT")
check(mb.summarize([run("stock", "a"), run("moe-cache", "b")], exact=False)["identity"] == "n/a (approximate mode)", "identity n/a for --fast")
check(mb.summarize([run("stock", "a"), {"mode": "moe-cache", "error": "x"}], exact=True)["identity"].startswith("not checked"), "identity not checked when a side failed")

# 4. the whole command with fakes: sections in order, privacy, files, exit codes
tmp = tempfile.mkdtemp(prefix="bench report ")                     # a space in the path
user = "zqxüser"; host = socket.gethostname()
os.environ["USER"] = os.environ["USERNAME"] = user
model_dir = os.path.join(tmp, f"{user} models")
os.makedirs(model_dir)
model = os.path.join(model_dir, "Flash-Next-Q2_0.gguf")
open(model, "wb").write(b"GGUF" + b"\0" * 1024)
calls = []
saved = (mb.doctor_facts, mb.launcher_plan, mb.run_one, mb.server_info, ui.flush_model_cache, ui.header_quant, mb.mc.port_busy)


def fake_run(model_, mode, a, port, text="Once upon a time"):
    calls.append(mode)
    return {"mode": mode, "decode": 10.0 if mode == "stock" else 20.0, "prefill": 100.0, "prompt_tokens": 2000, "ttft_s": 1.0, "text": FAKE_TEXT[mode],
            "plugin_active": mode == "moe-cache", "hit_rate": 90.0 if mode == "moe-cache" else None, "load_s": 5.0, "error": "",
            **({"agent_ttft_s": [9.0, 1.0, 0.5] if mode == "stock" else [4.0, 0.8, 0.4], "agent_decode": [10.0, 10.0, 10.0]} if getattr(a, "scenario", "") == "agent" else {})}

# 3b. the noise label: under 10 % is never a claim; the arms' own spread widens the band
r2 = lambda mode, dec, pre=100.0: {"mode": mode, "decode": dec, "prefill": pre, "text": "a", "plugin_active": mode == "moe-cache"}
nv = mb.summarize([r2("stock", 10.0, 100.0), r2("moe-cache", 19.0, 190.0), r2("moe-cache", 19.4, 192.0), r2("stock", 10.2, 101.0)], exact=True)["noise"]
check(nv["claimable"] and nv["decode"] == "faster" and nv["prefill"] == "faster" and "larger than the noise" in nv["sentence"] and "1.9x on prompt reading" in nv["sentence"],
      f"noise: 1.9x with tight runs is a claim ({nv['sentence']})")
nv = mb.summarize([r2("stock", 10.0), r2("moe-cache", 10.6), r2("moe-cache", 10.7), r2("stock", 10.1)], exact=True)["noise"]
check(not nv["claimable"] and nv["decode"] == "within noise" and nv["sentence"].startswith("Within noise") and "do not claim" in nv["sentence"], f"noise: +5 % is not a claim ({nv['sentence']})")
nv = mb.summarize([r2("stock", 8.0), r2("moe-cache", 11.0), r2("moe-cache", 9.0), r2("stock", 12.0)], exact=True)["noise"]
check(nv["noise_pct"] > 30 and nv["decode"] == "within noise", f"noise: noisy runs widen the band beyond 10 % ({nv['noise_pct']} %, 1.0x)")
nv = mb.summarize([r2("stock", 20.0), r2("moe-cache", 10.0), r2("moe-cache", 10.2), r2("stock", 20.2)], exact=True)["noise"]
check(nv["claimable"] and nv["decode"] == "slower" and "SLOWER" in nv["sentence"], "noise: a clear slowdown is reported as slower, not hidden")
check(not mb.summarize([r2("stock", 10.0)], exact=True)["noise"]["claimable"], "noise: a side without runs: nothing to claim")


# 3d. the record's JSON in the report stays valid JSON when a Windows path (escaped backslashes) is in it, and the path is still scrubbed
sc = mb.scrubber()
dd = {"record": {"record_schema": 1, "hardware": {"storage": {"drive": "C:\\Users\\Someone Else\\models\\x.gguf", "unc": "\\\\NAS\\share\\big\\y.gguf"}}, "notes": ["n"], "runs": [{"turns": [1], "arm": "stock"}]},
      "facts": {}, "summary": {}, "notes": []}
mb.report_md_saved = mb.report_md
mb.report_md = lambda d_, record_json=None: "head\n```json\n" + (record_json if record_json is not None else "?") + "\n```\ntail C:\\Users\\Someone Else\\z.gguf\n"
try:
    mdw = mb.scrubbed_report(dd, sc)
finally:
    mb.report_md = mb.report_md_saved
import re as _rw
blkw = _rw.search(r"```json\n(.*?)\n```", mdw, _rw.S).group(1)
try:
    jw = json.loads(blkw)
except ValueError as e_:
    jw = None
check(jw is not None and jw["hardware"]["storage"]["drive"] == "x.gguf" and jw["hardware"]["storage"]["unc"] == "y.gguf" and "Someone Else" not in mdw and "z.gguf" in mdw and "Users" not in mdw,
      f"a Windows path in the record: the JSON block stays valid and the path is cut to its last part ({blkw[:120]!r})")

# 3c. the ready-to-measure panel (reads only)
mdl = os.path.join(tempfile.mkdtemp(prefix="moe-ready-"), "x.gguf")
open(mdl, "wb").write(b"GGUF" + b"\0" * 4096)
saved_ready = (mb._on_battery, mb._big_programs, mb.mc.meminfo_total_gb, mb.mc.win_memory)
mb._on_battery = lambda: True; mb._big_programs = lambda limit_gb=1.0: [("browser", 3.2)]
mb.mc.meminfo_total_gb = lambda: 32.0; mb.mc.win_memory = lambda: (0, 0)
rd = mb.readiness(mdl, pairs=2)
lv = {c["id"]: c for c in rd["checks"]}
check(lv["size"]["level"] == "bad" and not rd["ready"] and lv["size"]["fix"], "ready: a model that fits in RAM is flagged bad, with the way out")
check(lv["power"]["level"] == "warn" and "charger" in lv["power"]["fix"] and lv["programs"]["level"] == "warn" and "browser 3.2 GB" in lv["programs"]["text"], "ready: battery and big programs are warnings with a fix")
mb._on_battery = lambda: False; mb._big_programs = lambda limit_gb=1.0: []
mb.mc.meminfo_total_gb = lambda: 0.000001
rd = mb.readiness(mdl, pairs=2)
check(rd["ready"] and {c["id"]: c for c in rd["checks"]}["power"]["level"] == "ok" and rd["checks"][-1]["id"] == "estimate" and rd["minutes"] >= 8, "ready: bigger than RAM on mains: ready, with an estimate")
check(mb.estimate(66, 2)[0] > mb.estimate(66, 1)[0] and mb.estimate(66, 1)[1] == 4 * 66, "ready: the estimate grows with the pairs; GB read = runs x model size")
mb._on_battery, mb._big_programs, mb.mc.meminfo_total_gb, mb.mc.win_memory = saved_ready

FAKE_TEXT = {"stock": "same", "moe-cache": "same"}
facts = {"system": "Linux 6.8", "cpu": "Test CPU", "ram": "32 GB total", "gpu": f"Vulkan0 (Test GPU), driver 1.2 on {host}", "moe-cache": "0.3.2",
         "disk": f"files in {os.path.expanduser('~')}/models", "drive speed": "3.1 GB/s", "model": r"D:\Private Stuff\models\x.gguf, 66 GB"}
mb.doctor_facts = lambda m, ls, ram: (False, dict(facts), [])
mb.server_info = lambda ls: (dev("Vulkan0"), "version: 11669 (620d2225c)")
mb.run_one = fake_run
ui.header_quant = lambda p: "Q2_0"
mb.mc.port_busy = lambda p: False
flush_ok = [True]
ui.flush_model_cache = lambda m, other: (flush_ok[0], "flushed" if flush_ok[0] else "no folder with other big files")
plan = ["plugin"]
mb.launcher_plan = lambda m, ls, ram, ctx, extra, limit_gb=None: (plan[0], "env MOE_CACHE_SIZE_GIB=auto llama-server -m x -c 8192 --n-cpu-moe 30")
try:
    out = os.path.join(tmp, "out dir")
    rc = mb.bench_report([model, "--out", out, "--llama-server", "llama-server"])
    files = sorted(os.listdir(out))
    check(rc == 0 and len(files) == 2 and files[0].endswith(".json") and files[1].endswith(".md"), f"exit 0, .md and .json written ({files})")
    md = open(os.path.join(out, files[1]), encoding="utf-8").read()
    js = open(os.path.join(out, files[0]), encoding="utf-8").read()
    order_ = ["moe-cache benchmark report", "**Result:**", "| OS |", "| **Backend** | **Vulkan** |", "| llama.cpp |", "| File |", "| Launcher plan |",
              "| decode tok/s", "| prompt tok/s", "Checks: tokens IDENTICAL", "No prompts, answers"]
    check("**Verdict:** moe-cache vs stock" in md and md.find("**Verdict:**") < md.find("| Machine |"), "the report carries the noise verdict above the tables")
    pos = [md.find(x) for x in order_]
    check(all(p >= 0 for p in pos) and pos == sorted(pos), "report sections present and in order")
    check("**COLD**" in md and calls == ["stock", "moe-cache", "moe-cache", "stock"], "COLD, and the runs went stock, moe-cache, moe-cache, stock")
    for bad, what in ((user, "user name"), (host, "host name"), (os.path.expanduser("~"), "home folder"), (model_dir, "model folder"), ("same", "answer text"),
                      ("Private Stuff", "another folder (Windows path)")):
        check(bad not in md and bad not in js, f"privacy: no {what} in the report or the .json")
    rj = json.loads(js)
    import moe_cache_record as rr
    check("Details (benchmark card format)" in md and "<details><summary>Record (JSON)" in md and rr.validate(rj["record"], rr.load_schema()) == [],
          "benchmark record: the card's detail lines and the record are in the report, and the record is valid against the schema")
    check(rj["record"]["outputs"]["identical_to_baseline"] == "yes" and rj["record"]["protocol"]["order"] == "ABBA" and len(rj["record"]["runs"]) == 4, "record: ABBA, four runs, identity yes")
    check("2.0x" in md, "decode ratio 2.0x in the result line")
    check(isinstance(rj.get("scrubbed"), dict) and rj["scrubbed"].get("user name or host name", 0) >= 1 and rj["scrubbed"].get("folder path", 0) >= 1, f"the .json lists what was removed, by kind ({rj.get('scrubbed')})")

    calls.clear(); flush_ok[0] = False
    rc = mb.bench_report([model, "--out", out, "--llama-server", "llama-server", "--flush-from", ""])
    md = open(os.path.join(out, sorted(f for f in os.listdir(out) if f.endswith(".md"))[-1]), encoding="utf-8").read()
    check("**WARM**" in md and md.count("cold start failed") == 1, "WARM stamp and one note (not one per run) when the cold start fails")
    flush_ok[0] = True

    FAKE_TEXT["moe-cache"] = "other"
    rc = mb.bench_report([model, "--out", out, "--warm"])
    md = open(os.path.join(out, sorted(f for f in os.listdir(out) if f.endswith(".md"))[-1]), encoding="utf-8").read()
    check(rc == 0 and "tokens DIFFERENT" in md and "tokens differ: stock and moe-cache ran" in md, "different tokens: reported with the explanation, exit 0")
    FAKE_TEXT["moe-cache"] = "same"


    # long chat (--chat): the arms play the real conversation against fake servers (as tests/conversation_bench_test.py); the report carries the band table, the
    # verdict and a record that validates; the whole record (with turns) goes to .record.json, the issue text has none
    import http.server, socketserver, threading, types
    sys.path.insert(0, os.path.join(HERE, "..", "tools"))
    import conversation_bench as cb

    class FakeSrv(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        log_message = lambda *a: None

        def do_GET(self):
            b = json.dumps({"default_generation_settings": {"n_ctx": 4096}}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            s_ = self.server
            msgs = body["messages"]; total = sum(len(m["content"]) // 4 + 8 for m in msgs)
            common = sum(len(b["content"]) // 4 + 8 for a_, b in zip(s_.last, msgs) if a_ == b) if s_.last else 0
            pn = total - common
            ans = "word " * 60
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": ans}}]}).encode() + b"\n\n")
            tm = {"prompt_n": pn, "cache_n": common, "prompt_ms": pn / s_.p * 1000, "prompt_per_second": s_.p, "predicted_n": 60, "predicted_ms": 60 / s_.d * 1000, "predicted_per_second": s_.d}
            self.wfile.write(b"data: " + json.dumps({"choices": [], "timings": tm, "usage": {"prompt_tokens": total, "completion_tokens": 60}}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            s_.last = msgs + [{"role": "assistant", "content": ans}]

    class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
    servers = {}
    for name, p_, d_ in (("stock", 100.0, 10.0), ("moe-cache", 200.0, 20.0)):
        sv = Srv(("127.0.0.1", 0), FakeSrv); sv.p, sv.d, sv.last = p_, d_, []
        threading.Thread(target=sv.serve_forever, daemon=True).start(); servers[name] = sv
    chat_calls = []

    def fake_chat(model_, mode, a, port, workdir, key, plan_line):
        chat_calls.append((mode, a.ctx, a.max_turns, key))
        ns = cb.default_args(ctx=a.ctx, max_turns=a.max_turns or None, timeout=30, server_cmd=plan_line, load_s=3.0)
        run = cb._one(ns, "http://127.0.0.1:%d" % servers[mode].server_address[1], mode, log=lambda m: None)
        o = cb.overall(run)
        return {"mode": mode, "error": "", "chat": run, "decode": o["decode_tps"], "prefill": o["prompt_tps"], "ttft_s": o["ttft_s"], "load_s": 3.0, "plugin_active": mode == "moe-cache"}
    saved_chat = (mb.run_chat_arm, mb.mc.stored_key)
    mb.run_chat_arm = fake_chat; mb.mc.stored_key = lambda create=False: "test-key"
    out2 = os.path.join(tmp, "chat out")
    rc = mb.bench_report([model, "--out", out2, "--chat", "--ctx", "4096", "--max-turns", "6", "--llama-server", "llama-server"])
    fl = sorted(os.listdir(out2))
    check(rc == 0 and any(f.endswith(".record.json") for f in fl), f"--chat: exit 0, the whole record is written next to the report ({fl})")
    mdc = open(os.path.join(out2, [f for f in fl if f.endswith(".md")][-1]), encoding="utf-8").read()
    check([c[0] for c in chat_calls] == ["stock", "moe-cache", "moe-cache", "stock"] and all(c[1] == 4096 and c[2] == 6 and c[3] == "test-key" for c in chat_calls), "--chat: A B B A, the context, the turn limit and the key reach every run")
    check("**Long chat**" in mdc and "| band |" in mdc and "2.00x" in mdc and "**Verdict:** moe-cache vs stock" in mdc and "not checked (long chat)" in mdc, "--chat: the report has the long-chat header, the band table (2.00x) and the verdict")
    recc = json.load(open(os.path.join(out2, [f for f in fl if f.endswith(".record.json")][0]), encoding="utf-8"))
    check(rr.validate(recc, rr.load_schema()) == [] and all("turns" in r for r in recc["runs"]) and recc["protocol"]["order"] == "ABBA", "--chat: the full record validates (turns kept, ABBA)")
    import re as _re
    issue_rec = json.loads(_re.search(r"```json\n(.*?)\n```", mdc, _re.S).group(1))
    check(all("turns" not in r for r in issue_rec["runs"]) and rr.validate(issue_rec, rr.load_schema()) == [] and len(mdc) < 60000, f"--chat: the issue text carries the record without per-turn lists, valid, under 60k characters ({len(mdc)})")
    check("test-key" not in mdc and "test-key" not in open(os.path.join(out2, [f for f in fl if f.endswith(".json") and not f.endswith("record.json")][0])).read(), "--chat: the API key is in no output file")
    check(mb.bench_report([model, "--out", out2, "--chat", "--scenario", "agent"]) == 2 and mb.bench_report([model, "--out", out2, "--chat", "--fast"]) == 2, "--chat with --scenario agent or --fast: refused (exit 2)")
    check((mb.chat_ctx(16), mb.chat_ctx(20), mb.chat_ctx(24), mb.chat_ctx(32), mb.chat_ctx(64), mb.chat_ctx(None)) == (16384, 16384, 32768, 65536, 65536, 65536), "chat context by RAM: 16k up to 20 GB, 32k up to 28 GB, else 65k")
    check(mb.estimate(66, 1, 65536)[0] > mb.estimate(66, 1, 0)[0] + 30, "the long chat's time estimate adds the conversation (about 10 min per run at 65k)")
    mb.run_chat_arm, mb.mc.stored_key = saved_chat
    for sv in servers.values():
        sv.shutdown()


    # memory-limit emulation (--limit-gb): refused where it cannot work, both arms get the limit, the report says EMULATED, the record says it and stays valid
    check(mb.limit_prefix(16)[:4] == ["systemd-run", "--user", "--scope", "--quiet"] and "MemoryMax=16G" in mb.limit_prefix(16) and "MemorySwapMax=0" in mb.limit_prefix(16) and mb.limit_prefix(16)[-1] == "--", "limit: systemd-run user scope, MemoryMax, no swap")
    saved_lim = mb.can_limit
    mb.can_limit = lambda: (False, "memory-limit emulation is Linux only")
    check(mb.bench_report([model, "--out", out, "--limit-gb", "16", "--warm"]) == 2, "limit: refused (exit 2) where the system cannot limit")
    mb.can_limit = lambda: (True, "")
    check(mb.bench_report([model, "--out", out, "--limit-gb", "2", "--warm"]) == 2 and mb.bench_report([model, "--out", out, "--limit-gb", "100000", "--warm"]) == 2, "limit: under 4 GB or not below the RAM: refused")
    seen_lim = []
    saved_run_one = mb.run_one
    mb.run_one = lambda m_, mode, a_, port: (seen_lim.append(a_.limit_gb), fake_run(m_, mode, a_, port))[1]
    plans_lim = []
    saved_plan = mb.launcher_plan
    mb.launcher_plan = lambda m_, ls_, ram_, ctx_, extra_, limit_gb=None: (plans_lim.append(limit_gb), saved_plan(m_, ls_, ram_, ctx_, extra_))[1]
    mb.mc.meminfo_total_gb = lambda: 64.0
    out3 = os.path.join(tmp, "limit out")
    rc = mb.bench_report([model, "--out", out3, "--limit-gb", "16", "--warm"])
    mdl_ = open(os.path.join(out3, [f for f in os.listdir(out3) if f.endswith(".md")][-1]), encoding="utf-8").read()
    recl = json.loads(_re_l.search(r"```json\n(.*?)\n```", mdl_, _re_l.S).group(1))
    check(recl["conditions"].get("memory_limit", "").startswith("EMULATED 16 GB") and rr.validate(recl, rr.load_schema()) == [], "limit: the record carries conditions.memory_limit and stays valid")
    check(rc == 0 and seen_lim == [16.0] * 4 and plans_lim == [16.0, 16.0], f"limit: every run and both plans get the limit ({seen_lim}, {plans_lim})")
    check("**EMULATED machine: 16 GB memory limit**" in mdl_ and "EMULATED machine: every server ran in a cgroup" in mdl_ and mdl_.find("EMULATED") < mdl_.find("**Result:**"), "limit: the report says EMULATED above the result and explains it in the notes")
    mb.run_one, mb.launcher_plan = saved_run_one, saved_plan
    mb.can_limit = saved_lim

    plan[0] = "stock"; calls.clear()
    check(mb.bench_report([model, "--out", out, "--warm"]) == 3 and calls == [], "model fits in RAM -> exit 3, nothing run")
    check(mb.bench_report([model, "--out", out, "--warm", "--force-plugin"]) == 0 and len(calls) == 4, "--force-plugin measures anyway")
    plan[0] = "plugin"

    rc = mb.bench_report([model, "--out", out, "--warm", "--scenario", "agent", "--gap", "0"])
    md = open(os.path.join(out, sorted(f for f in os.listdir(out) if f.endswith(".md"))[-1]), encoding="utf-8").read()
    check(rc == 0 and "agent turns: time to first token" in md and "9.0 / 1.0 / 0.5" in md and "4.0 / 0.8 / 0.4" in md, "--scenario agent: per-turn time to first token, stock and moe-cache")
    rc = mb.bench_report([model, "--out", out, "--warm"])
    md = open(os.path.join(out, sorted(f for f in os.listdir(out) if f.endswith(".md"))[-1]), encoding="utf-8").read()
    check("agent turns" not in md, "default scenario: no agent row")

    mb.doctor_facts = lambda m, ls, ram: (True, {}, ["[FAIL] plugin  not found"])
    check(mb.bench_report([model, "--out", out]) == 2, "a doctor FAIL stops with exit 2")
finally:
    mb.doctor_facts, mb.launcher_plan, mb.run_one, mb.server_info, ui.flush_model_cache, ui.header_quant, mb.mc.port_busy = saved

# AUD-032: UNC and \\?\ paths are scrubbed down to the file name like drive-letter paths
_sc = mb.scrubber()
for src, want in ((r"\\NAS\share\q.gguf", "q.gguf"), (r"model=\\NAS\my share\sub\q.gguf ok", "model=q.gguf ok"),
                  (r"\\?\C:\Users\zed\models\q.gguf", "q.gguf"), (r"\\?\UNC\NAS\share\dir\q.gguf", "q.gguf"),
                  (r"D:\Private Stuff\x.gguf", "x.gguf"), ("see http://a.example/b", "see http://a.example/b")):
    got = _sc(src)
    check(got == want and "NAS" not in got and "zed" not in got, f"scrub: {src!r} -> {want!r} (got {got!r})")

print("bench-report test: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
sys.exit(1 if FAILS else 0)
