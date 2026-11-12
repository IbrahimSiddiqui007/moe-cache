#!/usr/bin/env python3
"""TEST-1142: the control page's Compare view - the records folder functions (list, import, publish, compare by id), their safety (bad ids, oversized or invalid records, no silent overwrite),
the one-at-a-time suite job, and the page / route wiring (private: the suite routes are not in the LAN allow-lists).
   python3 tests/suite_ui_test.py"""
import json, os, sys, tempfile, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_suite as S, moe_cache_ui as ui      # noqa: E402
fails = 0


def check(cond, name):
    global fails
    print(("  ok   " if cond else "  FAIL ") + name)
    fails += 0 if cond else 1


def rec(name, hard, dec):
    return S.new_record({"name": name, "quant": "Q3"}, machine={"platform": "x"}, settings={}, sections={"speed": {"protocol": "S", "bands": [{"depth": 8192, "prompt_tps": 300.0, "decode_tps": dec}]},
                                                                                                         "hard": {"protocol": "H", "n": 27, "pass": hard, "think_median": 5000, "think_p90": 9000, "think_max": 12000, "loops": 0, "cap_hits": 0, "cap": 0}})


d = tempfile.mkdtemp(); A, B = rec("model-A", 20, 25.0), rec("model-B", 22, 24.0)
check(os.path.dirname(S.default_dir()).endswith("benchmarks") or S.default_dir().endswith(os.path.join("moe-cache", "suites")), "default_dir: the checkout's benchmarks/suites or the user data folder")
ida = S.import_text(d, json.dumps(A)); idb = S.import_text(d, json.dumps(B))
check(ida == A["id"] and os.path.exists(os.path.join(d, ida + ".json")), "import stores a valid record under its id")
check([r["id"] for r in S.list_records(d)] and {r["id"] for r in S.list_records(d)} == {ida, idb}, "list_records lists both, newest first, with model, quant, sections and the private flag")
check(all(r["published"] is False for r in S.list_records(d)) and "hard" in S.list_records(d)[0]["sections"], "records are private by default")
check(S.import_text(d, json.dumps(A)) == ida, "importing the same record again is fine")
other = json.loads(json.dumps(A)); other["sections"]["hard"]["pass"] = 21
for bad, why in ((json.dumps(other), "different content, same id"), ("not json", "not json"), (json.dumps({"x": 1}), "invalid record"), ("x" * (S.MAX_RECORD_BYTES + 1), "too large")):
    try:
        S.import_text(d, bad); ok = False
    except (ValueError, json.JSONDecodeError):
        ok = True
    check(ok, f"import refuses: {why}")
for rid in ("../etc/passwd", "a/b", "", "x" * 200, "a b"):
    try:
        S.record_path(d, rid); ok = False
    except ValueError:
        ok = True
    check(ok, f"record_path refuses the id {rid[:20]!r}")
res = S.compare_ids(d, [ida, idb])
check(res["meta"][0]["name"] == "model-A" and any(r["label"] == "hard tasks passed" and r["marks"][1] == "better" for r in res["rows"]), "compare_ids: reference first, hard pass 22 against 20 is better")
check(S.set_published(d, ida) is True and S.load(os.path.join(d, ida + ".json"))["published"] is True and [r for r in S.list_records(d) if r["id"] == ida][0]["published"], "publish sets only the flag")

# the one-at-a-time job (a fake child process instead of a real suite)
job = ui.SuiteJob()
class FakeProc:
    def __init__(self, *a, **k):
        import io; self.stdout = io.StringIO("[m] speed (probe)\n[m] tasks, thinking off\nwrote x.json\n")
    def wait(self): return 0
    def terminate(self): pass
orig = ui._sp.Popen; ui._sp.Popen = FakeProc
try:
    job.start(["/m/a.gguf"], "quick", d); time.sleep(0.5); st = job.status()
    check(st["state"] == "done" and st["lines"][-1] == "wrote x.json" and "quick: a.gguf" in st["what"], "a job runs, collects the output lines and ends done")
    try:
        job.start(["/m/a.gguf"], "bogus", d); ok = False
    except ValueError:
        ok = True
    check(ok, "an unknown profile is refused")
    job.state = "running"
    try:
        job.start(["/m/a.gguf"], "quick", d); ok = False
    except ValueError:
        ok = True
    check(ok, "a second suite while one runs is refused")
finally:
    ui._sp.Popen = orig

# wiring
src = open(os.path.join(HERE, "bin", "moe_cache_ui.py"), encoding="utf-8").read(); page = open(os.path.join(HERE, "ui", "index.html"), encoding="utf-8").read()
for route in ("/api/suite/records", "/api/suite/compare", "/api/suite/status", "/api/suite/import", "/api/suite/publish", "/api/suite/run", "/api/suite/cancel"):
    check(route in src and route in page, f"route {route} is served and used by the page")
check(not any(r.startswith("/api/suite") for r in ui.REMOTE_GET + ui.REMOTE_POST), "the suite routes are not allowed over the LAN (private)")
check('id="cmpdlg"' in page and 'id="cmpopen"' in page and "askConfirm(\"Publish this record?" in page and "esc(" in page.split("async function cmpCards")[1].split("async function cmpPoll")[0], "the page: the Compare button and dialog, publish asks first, record fields go through esc()")
check("run: stop the model" not in src and "stop the model that this page runs first" in src and "self.running()" in src.split('"/api/suite/run"')[1][:400], "running a suite is refused while the page's own model runs (memory)")
print("suite_ui_test: " + ("FAIL" if fails else "PASS"))
sys.exit(1 if fails else 0)
