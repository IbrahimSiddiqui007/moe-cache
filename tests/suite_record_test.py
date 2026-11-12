#!/usr/bin/env python3
"""TEST-1142: bin/moe_cache_suite.py - record building, validation, converters, protocol hashes, comparison (reference column, deltas, better/worse/noise, refusal of differing protocols), CLI, private-by-default.
   python3 tests/suite_record_test.py"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_suite as S      # noqa: E402
fails = 0


def chk(name, cond):
    global fails
    print(("  ok   " if cond else "  FAIL ") + name)
    fails += 0 if cond else 1


# protocol hash
chk("protocol_hash is stable and order independent", S.protocol_hash(a=1, b=2) == S.protocol_hash(b=2, a=1) and len(S.protocol_hash(a=1)) == 12)
chk("protocol_hash changes with any part", S.protocol_hash(cap=0) != S.protocol_hash(cap=512))

# converters
run = {"cap": 0, "items": [{"category": "sudoku", "pass": True, "think_tokens": 100, "loop": False, "cap_hit": False}, {"category": "sudoku", "pass": False, "think_tokens": 900, "loop": True, "cap_hit": True},
                           {"category": "graph", "pass": True, "think_tokens": 300, "loop": False, "cap_hit": False}]}
h = S.section_hard(run, "abc")
chk("section_hard: counts, median, p90, loops, cap hits, by category", h["n"] == 3 and h["pass"] == 2 and h["think_median"] == 300 and h["think_max"] == 900 and h["loops"] == 1 and h["cap_hits"] == 1 and h["by_category"]["sudoku"] == [1, 2])
q = S.section_quality(open(os.path.join(HERE, "tests", "data", "kl_log_sample.txt")).read(), "same model f16 KV", "w1", 8, 65536, "f16")
chk("section_quality parses the llama-perplexity KL lines", abs(q["kl_mean"] - 0.109258) < 1e-6 and abs(q["top1_pct"] - 94.643) < 1e-6 and q["kl_median"] > 0 and q["kl_p90"] > 0)
try:
    S.section_quality("nothing", "r"); chk("a KL log without a top-1 line is refused", False)
except ValueError:
    chk("a KL log without a top-1 line is refused", True)
yrun = {"items": [{"kind": "task", "text": "<think>x</think>\n391", "answers": [r"\b391\b"]}, {"kind": "task", "text": "no", "answers": [r"\byes\b"]}, {"kind": "open", "text": "zzz"}]}
t = S.section_tasks(yrun, "on", "th", 2048)
chk("section_tasks scores the task items only (thinking stripped)", t["n"] == 2 and t["pass"] == 1 and t["think"] == "on")

# records
def rec(name, hard_pass, think, dec32, proto="P", served=None, machine=None, quant="Q3", loops=1):
    return S.new_record({"name": name, "quant": quant}, machine=machine or {"cpu_model": "x", "ram_gib": 31}, software={"kit": "abc123"}, served=served or {"ctx": 32768, "np": 1, "ctk": "q8_0", "ctv": "q8_0", "build": "b1"}, settings={"kv": "q8_0"}, profile="standard", sections={
        "speed": {"protocol": "S1", "bands": [{"depth": 32768, "prompt_tps": 300.0, "decode_tps": dec32}]},
        "quality": {"protocol": "Q1", "ref": "same model f16 KV", "kl_mean": 0.1, "top1_pct": 95.0},
        "hard": dict(S.section_hard(run, "abc"), protocol=proto) | {"pass": hard_pass, "n": 27, "think_median": think, "loops": loops}})
A = rec("model-A", 9, 9000, 25.0); B = rec("model-B", 24, 5000, 24.9)
chk("a built record validates", S.validate_record(A) == [] and A["suite_record_schema"] == 1 and A["published"] is False)
bad = json.loads(json.dumps(A)); bad["sections"]["hard"]["pass"] = 99
chk("validation catches pass > n", any("pass > n" in p for p in S.validate_record(bad)))
bad2 = json.loads(json.dumps(A)); del bad2["model"]
chk("validation catches a missing required field", bool(S.validate_record(bad2)))

# compare
res = S.compare([A, B]); rows = {r["label"]: r for r in res["rows"]}
chk("first record is the reference (no delta, no mark)", rows["hard tasks passed of 27"]["deltas_pct"][0] is None and rows["hard tasks passed of 27"]["marks"][0] == "")
chk("higher is better: hard pass 24 against 9 of 27 is better (outside the 95 % interval), +167 %", rows["hard tasks passed of 27"]["marks"][1] == "better" and abs(rows["hard tasks passed of 27"]["deltas_pct"][1] - 166.666) < 0.01)
chk("lower is better: median thinking 5000 against 9000 is better", rows["thinking tokens, median"]["marks"][1] == "better")
chk("a small difference is '=' (noise): decode 24.9 against 25.0", rows["decode at 32k"]["marks"][1] == "=")
C = rec("model-C", 0, 20000, 20.0, loops=4); rc = {r["label"]: r for r in S.compare([A, C])["rows"]}
chk("worse is marked worse (decode -20 %, hard pass 0 against 9, loops 1 -> 4)", rc["decode at 32k"]["marks"][1] == "worse" and rc["loops"]["marks"][1] == "worse" and rc["hard tasks passed of 27"]["marks"][1] == "worse" and rc["thinking tokens, median"]["marks"][1] == "worse")
D = rec("model-D", 24, 5000, 25.0, proto="OTHER"); rd = S.compare([A, D])
chk("a section with another protocol hash is not compared and says why", all(v is None or i == 0 for r in rd["rows"] if r["section"] == "hard" for i, v in enumerate(r["values"])) and any("hard" in s and "different protocol" in s for s in rd["skipped"]))
chk("other sections of the same two records still compare", any(r["section"] == "speed" and r["values"][1] is not None for r in rd["rows"]))
txt = S.compare_text(res)
chk("compare_text prints both models and the deltas", "model-A Q3" in txt and "model-B Q3" in txt and "+167 % better" in txt)

# noise: pass counts get an interval, 0 -> 1 loops is not a verdict, one-run numbers need 10 %
near = {r["label"]: r for r in S.compare([rec("a", 20, 9000, 25.0, loops=0), rec("b", 24, 9000, 23.0, loops=1)])["rows"]}
chk("20 -> 24 of 27 is inside the noise of the pass count ('=')", near["hard tasks passed of 27"]["marks"][1] == "=")
chk("loops 0 -> 1 is '=' (not 'worse because the base is 0')", near["loops"]["marks"][1] == "=")
chk("a single-run speed change of -8 % is '=' (needs 10 %)", near["decode at 32k"]["marks"][1] == "=")
chk("every row says how it was measured: pass counts carry n, speed rows are kind 'run'", near["hard tasks passed of 27"]["kind"] == "pass" and near["hard tasks passed of 27"]["ns"] == [27, 27] and near["decode at 32k"]["kind"] == "run")
lo, hi = S.wilson(20, 27); chk("Wilson interval of 20/27 brackets 0.74 and is wide", lo < 0.74 < hi and hi - lo > 0.25)
# quality across different models is KV sensitivity, not a ranking
qa, qb = rec("m1", 20, 9000, 25.0), rec("m2", 20, 9000, 25.0, quant="Q4"); qb["sections"]["quality"]["kl_mean"] = 0.5
rq = {r["label"]: r for r in S.compare([qa, qb])["rows"]}
chk("KL between two different models carries no better/worse mark (it is the KV cost of each)", rq["KL mean (KV cost)"]["marks"][1] == "")
# server settings: what must match
E1 = rec("m", 20, 9000, 25.0); E2 = rec("m2", 20, 9000, 25.0, served={"ctx": 65536, "np": 1, "ctk": "f16", "ctv": "f16", "build": "b1"}); re_ = S.compare([E1, E2])
chk("another served context or KV type: speed, memory and hard are not compared, and the line says which setting", all(v is None or i == 0 for r in re_["rows"] if r["section"] in ("speed", "hard") for i, v in enumerate(r["values"])) and any("ctx" in s and "ctk" in s for s in re_["skipped"]))
chk("...and the warnings list the differences too", any("another ctx" in w for w in re_["warnings"]) and any("another ctk" in w for w in re_["warnings"]))
E3 = rec("m3", 20, 9000, 25.0, machine={"cpu_model": "y", "ram_gib": 64}); r3 = S.compare([E1, E3])
chk("another machine: speed is not compared (shows '-'), the task count still is", all(v is None or i == 0 for r in r3["rows"] if r["section"] == "speed" for i, v in enumerate(r["values"])) and any("machine" in s for s in r3["skipped"]))
E4 = rec("m4", 20, 9000, 25.0); E4["served"] = {}; r4 = S.compare([E1, E4])
chk("settings not recorded in one record: compared, with a warning that it could not be checked", any(r["section"] == "speed" and r["values"][1] is not None for r in r4["rows"]) and any("not recorded" in w for w in r4["warnings"]))
# scoring: only the text after </think> counts; a cut think block is a cut, not a pass
ok = S.task_answer_ok
chk("answer only in the reasoning: not a pass", ok("so it is 42 maybe</think>\nno idea", [r"42"], "stop", "on") == (False, False))
chk("answer after </think>: pass", ok("hmm</think>\nThe answer is 42", [r"42"], "stop", "on") == (True, False))
chk("cut while still thinking (stop=limit, no </think>): a cut, not a pass even if the reasoning says 42", ok("let me think... 42", [r"42"], "limit", "on") == (False, True))
chk("thinking off, short answer: pass", ok("42", [r"42"], "stop", "off") == (True, False))
ty = S.section_tasks({"items": [{"kind": "task", "text": "x 391", "answers": [r"391"], "stop": "limit"}, {"kind": "task", "text": "r</think>391", "answers": [r"391"], "stop": "stop"}]}, "on", "th", 2048)
chk("section_tasks counts cut answers and think blocks seen", ty["pass"] == 1 and ty["cut"] == 1 and ty["thinking_seen"] == 1)
eh = S.section_hard({"cap": 0, "items": run["items"] + [{"category": "graph", "error": "HTTPError", "pass": False, "think_tokens": 0, "loop": False, "cap_hit": False}]}, "abc", 65536)
chk("section_hard: an errored task is counted in `errors`, not in n or pass", eh["errors"] == 1 and eh["n"] == 3 and eh["pass"] == 2)
# CLI + private by default
d = tempfile.mkdtemp(); fa, fb, fh = (os.path.join(d, x) for x in ("a.json", "b.json", "hard.json"))
json.dump(A, open(fa, "w")); json.dump(B, open(fb, "w")); json.dump(S.section_hard(run, "abc"), open(fh, "w"))
cli = lambda *a: subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache_suite.py"), *a], capture_output=True, text=True)
r = cli("validate", fa, fb); chk("CLI validate: ok, exit 0", r.returncode == 0 and r.stdout.count("ok") == 2)
r = cli("compare", fa, fb); chk("CLI compare prints the table", r.returncode == 0 and "hard tasks passed of 27" in r.stdout)
r = cli("compare", fa, fb, "--json"); chk("CLI compare --json is valid JSON with rows", r.returncode == 0 and json.loads(r.stdout)["rows"])
out = os.path.join(d, "n.json"); r = cli("new", "--model", "m", "--quant", "Q4", "--section", f"hard={fh}", "--out", out)
chk("CLI new builds a valid private record from a section file", r.returncode == 0 and json.load(open(out))["published"] is False and S.validate_record(json.load(open(out))) == [])
r = cli("publish", out); chk("publish sets the flag and says nothing reads it yet", r.returncode == 0 and json.load(open(out))["published"] is True and "nothing reads" in r.stdout)
print("suite_record_test: " + ("FAIL" if fails else "PASS"))
sys.exit(1 if fails else 0)
