#!/usr/bin/env python3
"""GUI-041: the Monitor view without a model: bin/moe_cache_monitor.py on its own, the Runner's hooks (log line, plugin snapshot, /metrics, tick, payload),
and the page's chart and log-filter code under node. Linux and Windows; standard library only.
   python3 tests/monitor_test.py"""
import json, os, re, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-mon-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
os.environ["XDG_CONFIG_HOME"] = os.path.join(TMP, "config")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_monitor as mm  # noqa: E402
import moe_cache_ui as ui  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

# ---- /metrics text ----
prom = "# HELP llamacpp:prompt_tokens_total x\n# TYPE x counter\nllamacpp:prompt_tokens_total 120\nllamacpp:prompt_seconds_total 2.5\nllamacpp:tokens_predicted_total 40\nllamacpp:requests_processing 1\nother 5\n"
p = mm.parse_prom(prom)
check(p == {"prompt_tokens_total": 120.0, "prompt_seconds_total": 2.5, "tokens_predicted_total": 40.0, "requests_processing": 1.0}, f"parse_prom keeps the llamacpp: series ({p})")

# ---- log classification ----
c = mm.classify
check(c("moe-cache: planning", True) == ("launcher", "info") and c("moe-cache: cache 6 GB", False) == ("plugin", "info"), "classify: moe-cache lines are launcher before the URL line, plugin after")
check(c("ui: the server ended (exit code 1)", False)[0] == "ui" and c("srv  update_slots: all slots are idle", False)[0] == "llama", "classify: ui and llama-server lines")
check(c("moe-cache: warning: low RAM", False)[1] == "warn" and c("llama_model_load: error: failed to load", False)[1] == "error", "classify: warning and error levels")
check(c("slot release: id 0 | task 0 | stop processing", False)[1] == "info", "classify: ordinary llama-server lines are info")

# ---- plan from the real llama-server arguments ----
plan = mm.plan_from_args("llama-server -m x.gguf -c 65536 -ngl 99 --n-cpu-moe 30 -ub 4096 -b 4096 -ctk q8_0 -ctv q8_0 -fa on -t 8 --parallel 1 --device CUDA0".split())
check(plan["ub"] == "4096" and plan["kv_k"] == "q8_0" and plan["cpu_moe_layers"] == "30" and plan["ngl"] == "99" and plan["device"] == "CUDA0" and plan["ctx"] == "65536", f"plan_from_args ({plan})")
check(mm.plan_from_args(["x", "--cpu-moe", "--ctx-size=8192"]) == {"cpu_moe_all": True, "ctx": "8192"}, "plan_from_args: --cpu-moe and --opt=value")
check(mm.split_command('a "b c" d') == ["a", "b c", "d"] or os.name == "nt", "split_command")

# ---- the Monitor: lines ----
m = mm.Monitor()
m.add_line(1, 100.0, "moe-cache: MTP on: 3 drafts")
m.add_line(2, 101.0, "moe-cache: web UI and API will be at http://127.0.0.1:8080")
m.add_line(3, 102.0, "slot update_slots: id 0 | task 5 | prompt processing done, n_tokens = 812, batch.n_tokens = 812")
m.add_line(4, 103.0, "prompt eval time =     400.00 ms /   812 tokens (    0.49 ms per token,  2030.00 tokens per second)")
m.add_line(5, 104.0, "slot print_timing: draft acceptance rate = 0.75000 (   30 accepted /    40 generated)")
m.add_line(6, 105.0, "       eval time =    2000.00 ms /    40 tokens (   50.00 ms per token,    20.00 tokens per second)")
m.add_line(7, 106.0, "moe-cache: approximate routing (skip): 1000 low-ranked picks looked at, 300 given up, 0 replaced by a resident expert")
check(m.starting is False and [x[2] for x in m.log][:2] == ["launcher", "launcher"], "lines up to the URL line are launcher")
m.add_line(8, 107.0, "moe-cache: executor by type q4_k ops=5")
check(m.log[-1][2] == "plugin", "a moe-cache line after the URL line is plugin")
check(m.ctx_depth == 812, "context depth from 'n_tokens ='")
check(len(m.requests) == 1 and m.requests[0]["prefill_tps"] == 2030.0 and m.requests[0]["decode_tps"] == 20.0 and m.requests[0]["decode_tokens"] == 40 and m.requests[0]["mtp"] == 0.75, f"one request row from the timing lines ({list(m.requests)})")
check(m.mtp["accepted"] == 30 and m.mtp["generated"] == 40, "MTP acceptance counted")
check(m.approx.get("mode") == "skip" and m.approx["given_up"] == 300, f"approximate-routing line parsed ({m.approx})")

# ---- per-request prompt length, reused tokens and duration (the Requests view) ----
m2 = mm.Monitor()
m2.add_line(1, 10.0, "slot update_slots: id 0 | task 7 | new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 1000")
m2.add_line(2, 11.0, "prompt eval time =     500.00 ms /   200 tokens (    2.50 ms per token,   400.00 tokens per second)")
m2.add_line(3, 12.0, "       eval time =    2000.00 ms /    40 tokens (   50.00 ms per token,    20.00 tokens per second)")
m2.add_line(4, 12.0, "      total time =    2500.00 ms /   240 tokens")
r2 = m2.requests[-1]
check(r2["prompt_total"] == 1000 and r2["prefill_tokens"] == 200 and r2["reused"] == 800 and r2["total_ms"] == 2500.0 and r2["decode_tokens"] == 40, f"request row: whole prompt, reused 800 of 1000, duration ({r2})")
m2.add_line(5, 13.0, "prompt eval time =     500.00 ms /   200 tokens (    2.50 ms per token,   400.00 tokens per second)")
m2.add_line(6, 14.0, "       eval time =    2000.00 ms /    40 tokens (   50.00 ms per token,    20.00 tokens per second)")
check("reused" not in m2.requests[-1], "a request whose prompt length was not logged has no reused figure (nothing invented)")
# the same from the lines this llama-server really prints (recorded on the laptop: no "new prompt" line; the slot choice and n_past say what was reused)
m3 = mm.Monitor()
REAL = ["0.01.987.913 I slot get_availabl: id  3 | task -1 | selected slot by LRU, t_last = -1",
        "0.14.827.333 I slot print_timing: id  3 | task 1 | prompt eval time =      586.50 ms /   115 tokens (    5.10 ms per token,   196.08 tokens per second)",
        "0.14.827.343 I slot print_timing: id  3 | task 1 |        eval time =   12252.30 ms /   150 tokens (   82.23 ms per token,    12.16 tokens per second)",
        "0.14.827.345 I slot print_timing: id  3 | task 1 |       total time =   12838.80 ms /   265 tokens",
        "0.14.827.432 I slot      release: id  3 | task 1 | stop processing: n_tokens = 264, truncated = 0",
        "0.14.837.617 I slot get_availabl: id  3 | task -1 | selected slot by LCP similarity, f_sim_best = 1.000 (> 0.100 thold), f_keep = 0.436",
        "0.14.837.928 W slot   operator(): id  3 | task 170 | need to evaluate at least 1 token for each active slot (n_past = 115, task.n_tokens() = 115)",
        "0.14.837.929 W slot   operator(): id  3 | task 170 | n_past was set to 114",
        "0.27.126.925 I slot print_timing: id  3 | task 170 | prompt eval time =      174.72 ms /     1 tokens (  174.72 ms per token,     5.72 tokens per second)",
        "0.27.126.930 I slot print_timing: id  3 | task 170 |        eval time =   12114.27 ms /   150 tokens (   81.30 ms per token,    12.30 tokens per second)",
        "0.27.126.930 I slot print_timing: id  3 | task 170 |       total time =   12288.99 ms /   151 tokens"]
for i, l in enumerate(REAL):
    m3.add_line(i + 1, 100.0 + i, l)
q1, q2 = m3.requests
check(q1["reused"] == 0 and q1["prompt_total"] == 115 and q1["total_ms"] == 12838.8, f"real log, empty slot: nothing reused, prompt 115 ({q1.get('reused')}, {q1.get('prompt_total')})")
check(q2["reused"] == 114 and q2["prompt_total"] == 115 and q2["prefill_tokens"] == 1 and q2["total_ms"] == 12288.99, f"real log, cached slot: 114 of 115 reused ({q2.get('reused')}, {q2.get('prompt_total')})")
mh = mm.Monitor()
mh.tick(1.0, None, None, None, None, None, False, hw={"gpu_util": 50.5, "cpu_sys": 12.3456, "io_r": None})
check(mh.samples[-1]["gpu_util"] == 50.5 and mh.samples[-1]["cpu_sys"] == 12.35 and mh.samples[-1]["io_r"] is None, "hardware readings join the sample row")
check(not mh.watched() and (mh.payload() and mh.watched()), "hardware is read only while somebody looked in the last 20 s")
check(isinstance(ui.token_mode_status({"kv_k": "q8_0", "ub": "4096", "env": {"MOE_CACHE_ROUTE": "skip"}}, True, {})[0], dict), "token modes come with their status")
tm = {x["id"]: x for x in ui.token_mode_status({"kv_k": "q8_0", "ub": "4096", "env": {"MOE_CACHE_ROUTE": "skip"}}, True, {"approx": "safe"})}
check(tm["kv8"]["active"] is True and tm["long-prompt"]["active"] is True and tm["mtp"]["active"] is True and tm["fast1"]["active"] is True and tm["fewer"]["active"] is False and tm["approx-tiers"]["active"] is True and tm["kv8"]["status"] == "won", "token modes: which are in use from the real arguments and environment")
check(ui.token_mode_status({"ub": "512"}, False, {})[3]["active"] is None, "token modes: unknown (not off) when the environment cannot be read")

# ---- snapshots -> rates and layer columns ----
s1 = {"hits": 100, "misses": 10, "hits_dec": 50, "misses_dec": 5, "read_bytes": 1_000_000_000, "ns_ssd_wait": 100_000_000, "n_ssd_wait": 4, "ns_cpu": 400_000_000, "ns_total": 1_000_000_000,
      "evictions": 10, "layer_hits": [10, 20, 0], "layer_misses": [0, 5, 0]}
s2 = dict(s1, hits=190, misses=20, hits_dec=90, misses_dec=10, read_bytes=3_000_000_000, ns_ssd_wait=600_000_000, n_ssd_wait=9, ns_cpu=900_000_000, ns_total=2_000_000_000, evictions=30,
          layer_hits=[20, 20, 0], layer_misses=[0, 15, 0])
m.add_snapshot(200.0, s1); m.add_snapshot(202.0, s2)
r = m.rates
check(abs(r["hit"] - 90 / 100) < 1e-9 and abs(r["hit_dec"] - 40 / 45) < 1e-9, f"interval hit rates ({r['hit']}, {r['hit_dec']})")
check(abs(r["ssd_gbs"] - 1.0) < 1e-9 and abs(r["ssd_wait"] - 0.25) < 1e-9 and r["ssd_waits"] == 5, f"SSD GB/s and wait over the interval ({r['ssd_gbs']}, {r['ssd_wait']})")
check(abs(r["cpu"] - 0.5) < 1e-9 and abs(r["evict"] - 10.0) < 1e-9, "executor share and evictions per second")
check(len(m.layer_cols) == 1 and m.layer_cols[0]["rate"] == [1.0, 0.0, None], f"per-layer column: layer 0 all hits, layer 1 all misses, layer 2 no lookups ({m.layer_cols[0]['rate']})")

# ---- GUI-042: the approximate modes' counters live, from the stats snapshot ----
ma = mm.Monitor()
sa = {"hits": 1, "misses": 1, "approx": {"route": 1, "considered": 1000, "skipped": 100, "substituted": 20, "fewer": 2, "fewer_considered": 500, "fewer_dropped": 300}}
sb = {"hits": 2, "misses": 1, "approx": {"route": 1, "considered": 3000, "skipped": 400, "substituted": 80, "fewer": 2, "fewer_considered": 1500, "fewer_dropped": 900}}
ma.add_snapshot(10.0, sa); ma.add_snapshot(12.0, sb)
ma.tick(13.0, None, None, None, None, None, True)
pa = ma.payload()
check(pa["approx"]["mode"] == "skip" and pa["approx"]["live"] is True and pa["approx"]["considered"] == 3000 and pa["approx"]["given_up"] == 400 and pa["approx"]["replaced"] == 80
      and pa["approx"]["fewer"] == 2 and pa["approx"]["fewer_dropped"] == 900, f"approximate counters from the snapshot ({pa['approx']})")
check(abs(pa["samples"][-1]["alt"] - ((400 + 80 + 900) - (100 + 20 + 300)) / 2) < 1e-6, f"picks changed per second between two snapshots ({pa['samples'][-1]['alt']})")
mb = mm.Monitor(); mb.add_snapshot(1.0, {"hits": 1}); mb.add_snapshot(3.0, {"hits": 2}); mb.tick(4.0, None, None, None, None, None, False)
check(mb.payload()["approx"] == {} and mb.payload()["samples"][-1]["alt"] is None, "no approximate mode: no counters and no rate (nothing invented)")
mc_ = mm.Monitor(); mc_.add_line(1, 1.0, "moe-cache: approximate routing (skip): 1000 low-ranked picks looked at, 300 given up, 0 replaced by a resident expert")
check(mc_.payload()["approx"]["mode"] == "skip" and "live" not in mc_.payload()["approx"], "an older plugin: the counters still come from its stop-time line")

# ---- GUI-043: decode steps the routing change touched, attributed to requests by time ----
ms = mm.Monitor()
ms.add_line(1, 100.0, "0.00.000.001 I slot launch_slot_: id  0 | task 1 | processing task, is_child = 0")
ms.add_line(2, 103.0, "0.00.003.000 I slot print_timing: id  0 | task 1 | prompt eval time =      50.00 ms /    10 tokens (    5.00 ms per token,   200.00 tokens per second)")
ms.add_line(3, 103.0, "0.00.003.000 I slot print_timing: id  0 | task 1 |        eval time =    2000.00 ms /     5 tokens (  400.00 ms per token,     2.50 tokens per second)")
ms.add_line(4, 103.0, "0.00.003.000 I slot print_timing: id  0 | task 1 |       total time =    2050.00 ms /    15 tokens")
ms.add_line(5, 103.1, "0.00.003.100 I slot      release: id  0 | task 1 | stop processing: n_tokens = 15, truncated = 0")
tail = [[1000 * 100 + 200 * k, [0, 3, 0, 5, 1][k], 1] for k in range(5)]          # plugin clock: 100.0 s + k * 0.2 s, picks changed per step
snap = {"hits": 1, "approx": {"route": 1, "considered": 10, "skipped": 9, "substituted": 0, "fewer": 0}, "steps": {"total": 5, "now_ms": 101000, "tail": tail}}
ms.add_snapshot(101.0, snap)
rk = ms.risk(100.0, 103.1)
check(rk["n"] == 5 and rk["changed_steps"] == 3 and rk["picks"] == 9 and rk["steps"][1] == [3, 1] and rk["lost"] == 0 and rk["final"] is False, f"steps in a request's window: 3 of 5 changed, 9 picks ({rk})")
ms.add_snapshot(105.0, dict(snap, steps={"total": 5, "now_ms": 105000, "tail": tail}))
check(ms.risk(100.0, 103.1)["n"] == 5 and ms.risk(100.0, 103.1)["final"] is True, "the same snapshot again adds no steps; a snapshot after the request makes the answer final")
rq = ms.payload()["requests"][-1]
check(rq["risk"]["steps"] == 5 and rq["risk"]["changed_steps"] == 3 and rq["risk"]["picks"] == 9 and rq["risk"]["final"] is True, f"the request row carries its at-risk summary ({rq.get('risk')})")
ms.add_snapshot(107.0, {"hits": 1, "approx": {"route": 1}, "steps": {"total": 2000, "now_ms": 107000, "tail": [[107000, 1, 1]] * 512}})
check(ms.steps_lost > 0 and len(ms.st_t) == 5 + 512, f"a gap bigger than one tail is counted as lost steps ({ms.steps_lost})")
check(ms.risk(50.0, 60.0)["n"] == 0 and mm.Monitor().risk(0, 1)["n"] == 0, "no steps outside the window, none without a plugin that sends them")
mo = mm.Monitor(); mo.add_snapshot(1.0, {"hits": 1}); check(mo.st_t == [] and "risk" not in (mo.payload()["requests"][-1] if mo.payload()["requests"] else {}), "no approximate mode: no steps and no at-risk summary")

# ---- /metrics rates, tick, payload ----
m.add_metrics(300.0, "llamacpp:prompt_tokens_total 100\nllamacpp:prompt_seconds_total 1\nllamacpp:tokens_predicted_total 10\nllamacpp:tokens_predicted_seconds_total 1\n")
m.add_metrics(302.0, "llamacpp:prompt_tokens_total 1100\nllamacpp:prompt_seconds_total 2\nllamacpp:tokens_predicted_total 10\nllamacpp:tokens_predicted_seconds_total 1\nllamacpp:requests_processing 1\n")
check(m.prom_rates == {"pre": 1000.0, "dec": None}, f"prompt and decode tok/s from the counters' difference ({m.prom_rates})")
m.tick(303.0, 18.5, 5e9, (20e9, 32e9), None, (6e9, 8e9), True)
pl = m.payload(0, 0, ctx_max=65536, plan={"ub": "4096"}, cap=28 * 2**30, totals={"state": "ready"})
smp = pl["samples"][-1]
check(smp["dec"] == 18.5 and smp["pre"] == 1000.0 and smp["hit"] == 0.9 and smp["ssd"] == 1.0 and smp["ram"] == 20e9 and smp["vram"] == 6e9 and smp["commit"] is None, f"the sample row ({smp})")
check(pl["layer_total"][:2] == [1.0, 0.571] and pl["cap"] == 28 * 2**30 and pl["ctx"] == {"depth": 812, "max": 65536} and pl["mtp"]["rate"] == 0.75, "payload: whole-run layer rates, cap, context, MTP rate")
check(json.loads(json.dumps(pl))["log_seq"] == 8 and len(pl["log"]) == 8 and len(m.payload(0, 5)["log"]) == 3, "payload: log lines after a sequence number")
check(len(m.payload(smp["t"])["samples"]) == 0, "payload: samples after a time")
check([x[0] for x in m.payload(0, 0, tail=3)["log"]] == [6, 7, 8], "payload: tail=3 keeps only the newest three lines")
for i in range(mm.MAX_LOG + 50):
    m.add_line(1000 + i, 400.0, "x")
check(len(m.log) == mm.MAX_LOG, f"the log ring holds {mm.MAX_LOG} lines")

# ---- off switch ----
os.environ["MOE_CACHE_MONITOR"] = "0"
off = mm.Monitor()
off.add_snapshot(1.0, s1); off.add_snapshot(3.0, s2); off.tick(3.0, 1, 1, None, None, None, False)
check(not off.samples and not off.layer_cols and off.payload()["enabled"] is False, "MOE_CACHE_MONITOR=0: no samples, no layer columns")
del os.environ["MOE_CACHE_MONITOR"]

# ---- export ----
ex = mm.export_text([(1, 0.0, "plugin", "warn", "hello")], "jsonl")
check(json.loads(ex)["text"] == "hello" and "[plugin/warn] hello" in mm.export_text([(1, 0.0, "plugin", "warn", "hello")]), "export as text and as JSON lines")

# ---- the Runner's hooks ----
rn = ui.Runner()
rn.add_line("moe-cache: planning")
rn.take_snapshot(s1); rn.take_snapshot(dict(s2))
rn.mon.add_snapshot(time.time() + 2, dict(s2, hits=300))       # a second interval for the monitor
rn.monitor_tick(1, None, False)       # nothing listens on port 1: /metrics fails, memory still sampled
pay = rn.monitor_payload(0, 0)
check(pay["log"] and pay["log"][0][4] == "moe-cache: planning" and "plan" in pay and "totals" in pay and pay["samples"], f"Runner: the payload has log, plan, totals and a sample ({list(pay)})")
check(rn.monitor_payload(1e12, 10 ** 9)["log"] == [] and rn.monitor_payload(1e12, 10 ** 9)["samples"] == [], "Runner: nothing newer than what the page has")
rn.reset({})
check(len(rn.mon.log) == 0, "a new run starts a new Monitor")

# ---- LAN allow-list, process arguments ----
check("/api/monitor" not in ui.REMOTE_GET and "/api/monitor" not in ui.REMOTE_POST, "/api/monitor is not on the LAN allow-list (it carries the whole log)")
a1 = ui.process_args(os.getpid())
check(isinstance(a1, tuple) and ui.process_args(os.getpid()) == a1, "process_args answers (argv, env) and is cached per pid")
check(ui.process_args(2 ** 22 + 1)[0] == [], "process_args: a pid that is not there gives no arguments")

# ---- the page's own code under node ----
node = shutil.which("node")
page = open(os.path.join(HERE, "ui", "index.html"), encoding="utf-8").read()
check('id="monopen"' in page and 'id="mondlg"' in page and "/api/monitor" in page, "the page has the Monitor button, view and route")
if node:
    js = re.search(r"<script[^>]*>(.*)</script>", page, re.S).group(1)
    pj = os.path.join(TMP, "page_all.js"); open(pj, "w", encoding="utf-8").write(js)
    o = subprocess.run([node, "--check", pj], capture_output=True, text=True, timeout=60)
    check(o.returncode == 0, "the page's whole script parses (node --check)" + ("" if o.returncode == 0 else ": " + o.stderr[:300]))
    pick = lambda pat: re.search(pat, js, re.S).group(0)
    fn = "".join(pick(p_) for p_ in (r"const nz = [^\n]*\n", r"const cfmt = [^\n]*\n", r"function niceMax\(v\) \{.*?\n\}\n", r"const hhmmss = [^\n]*\n", r"function monRange\(\) \{[^\n]*\n",
                                       r"function monLine\(id, series, o\) \{.*?\n\}\n", r"function monLogFilter\(\) \{.*?\n\}\n"))
    harness = ("const esc = s => String(s);\nconst M = {samples: [], cols: [], log: []};\nconst els = {};\n"
               "const $ = id => els[id] || (els[id] = {value: '', innerHTML: '', firstChild: {}, hidden: true, style: {}});\n"
               "const chartColours = () => ({moe:'#c00', gpu:'#00c', ok:'#0a0', warn:'#fa0', bad:'#f00', line:'#ccc', muted:'#888', text:'#000'});\n"
               "const document = {querySelectorAll: () => [{value:'plugin'},{value:'llama'}]};\n" + fn +
               "for (let i = 0; i < 30; i++) M.samples.push({t: 1000 + i * 2, dec: i < 10 ? null : 20 + i, pre: 500, ctx: 100 * i, hit: 0.9});\n"
               "$('monrange').value = '0';\n"
               "monLine('a', [{k: 'dec', name: 'decode', color: 'moe'}, {k: 'pre', name: 'prompt', color: 'gpu'}], {});\n"
               "const a = els['a'].innerHTML;\n"
               "monLine('b', [{k: 'hit', color: 'ok', scale: 100}], {ymax: 100, fmt: v => v.toFixed(0) + ' %', hline: {v: 50, label: 'cap'}});\n"
               "M.samples.length = 0; monLine('c', [{k: 'dec', color: 'moe'}], {});\n"
               "$('monq').value = '/ba[dr]/'; $('monlevel').value = 'warn';\n"
               "const f = monLogFilter();\n"
               "console.log(JSON.stringify({a, b: els['b'].innerHTML, c: els['c'].innerHTML, f1: f.test([1, 0, 'plugin', 'warn', 'a BAD thing']), f2: f.test([1, 0, 'plugin', 'info', 'a BAD thing']), f3: f.test([1, 0, 'ui', 'warn', 'bad']), f4: f.test([1, 0, 'llama', 'error', 'nothing'])}));\n")
    hf = os.path.join(TMP, "mon.js"); open(hf, "w", encoding="utf-8").write(harness)
    o = subprocess.run([node, hf], capture_output=True, text=True, encoding="utf-8", timeout=30)
    try:
        r_ = json.loads(o.stdout.strip().splitlines()[-1])
    except Exception:
        r_ = None; print(o.stdout[-300:], o.stderr[-600:])
    check(r_ is not None and r_["a"].count("<polyline") == 2 and "1000" not in r_["a"][:0] and "no data yet" in r_["c"] and "no data yet" not in r_["a"], "chart: two lines for two series, the gap before the first value breaks nothing, an empty chart says so")
    check(r_ is not None and "stroke-dasharray" in r_["b"] and "50 %" in r_["b"], "chart: a reference line (the cap) with the unit on the axis")
    check(r_ is not None and (r_["f1"], r_["f2"], r_["f3"], r_["f4"]) == (True, False, False, False), "log filter: regex search, source and level boxes")
    fn2 = re.search(r"function riskMarks\(cuts, steps\) \{.*?\n\}\n", js, re.S).group(0)
    h2 = os.path.join(TMP, "risk.js"); open(h2, "w", encoding="utf-8").write(fn2 + """
const cuts = [3, 6, 9, 12, 15], steps = [[0, 1], [2, 1], [0, 1], [5, 1], [1, 1]];
const a = riskMarks(cuts, steps);
const b = riskMarks([4, 8], [[0, 1], [0, 1], [3, 1], [0, 1]]);        // 2 pieces, 4 steps (drafted several at a time): proportional
const c = riskMarks([], []);
console.log(JSON.stringify({a, b, c}));""")
    o2 = subprocess.run([node, h2], capture_output=True, text=True, encoding="utf-8", timeout=30)
    try:
        r2 = json.loads(o2.stdout.strip().splitlines()[-1])
    except Exception:
        r2 = None; print(o2.stdout[-300:], o2.stderr[-500:])
    check(r2 is not None and r2["a"]["ranges"] == [[3, 6], [9, 15]] and r2["a"]["exact"] is True and r2["a"]["strip"] == [0, 1, 0, 1, 1], f"chat marks: pieces of changed steps become text ranges, neighbours merge ({r2 and r2['a']})")
    check(r2 is not None and r2["b"]["exact"] is False and r2["b"]["ranges"] == [[4, 8]] and r2["c"]["ranges"] == [], "chat marks: proportional mapping when pieces and steps differ; nothing for an empty answer")
else:
    print("  skip page checks: node not found")

print("monitor_test:", "FAILED" if fails else "ok", f"({len(fails)} failed)" if fails else "")
sys.exit(1 if fails else 0)
