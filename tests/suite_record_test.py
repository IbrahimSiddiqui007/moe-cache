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
def rec(name, hard_pass, think, dec32, proto="P"):
    return S.new_record({"name": name, "quant": "Q3"}, machine={"cpu": "x"}, settings={"kv": "q8_0"}, profile="standard", sections={
        "speed": {"protocol": "S1", "bands": [{"depth": 32768, "prompt_tps": 300.0, "decode_tps": dec32}]},
        "quality": {"protocol": "Q1", "ref": "same model f16 KV", "kl_mean": 0.1, "top1_pct": 95.0},
        "hard": dict(S.section_hard(run, "abc"), protocol=proto) | {"pass": hard_pass, "n": 27, "think_median": think}})
A = rec("model-A", 20, 9000, 25.0); B = rec("model-B", 22, 5000, 24.9)
chk("a built record validates", S.validate_record(A) == [] and A["suite_record_schema"] == 1 and A["published"] is False)
bad = json.loads(json.dumps(A)); bad["sections"]["hard"]["pass"] = 99
chk("validation catches pass > n", any("pass > n" in p for p in S.validate_record(bad)))
bad2 = json.loads(json.dumps(A)); del bad2["model"]
chk("validation catches a missing required field", bool(S.validate_record(bad2)))

# compare
res = S.compare([A, B]); rows = {r["label"]: r for r in res["rows"]}
chk("first record is the reference (no delta, no mark)", rows["hard tasks passed"]["deltas_pct"][0] is None and rows["hard tasks passed"]["marks"][0] == "")
chk("higher is better: hard pass 22 against 20 is better, +10 %", rows["hard tasks passed"]["marks"][1] == "better" and abs(rows["hard tasks passed"]["deltas_pct"][1] - 10) < 1e-9)
chk("lower is better: median thinking 5000 against 9000 is better", rows["thinking tokens, median"]["marks"][1] == "better")
chk("a small difference is '=' (noise): decode 24.9 against 25.0", rows["decode at 32k"]["marks"][1] == "=")
C = rec("model-C", 10, 20000, 20.0); rc = {r["label"]: r for r in S.compare([A, C])["rows"]}
chk("worse is marked worse (decode and hard pass)", rc["decode at 32k"]["marks"][1] == "worse" and rc["hard tasks passed"]["marks"][1] == "worse" and rc["thinking tokens, median"]["marks"][1] == "worse")
D = rec("model-D", 22, 5000, 25.0, proto="OTHER"); rd = S.compare([A, D])
chk("a section with another protocol hash is not compared and says why", all(v is None or i == 0 for r in rd["rows"] if r["section"] == "hard" for i, v in enumerate(r["values"])) and any("hard" in s and "different protocol" in s for s in rd["skipped"]))
chk("other sections of the same two records still compare", any(r["section"] == "speed" and r["values"][1] is not None for r in rd["rows"]))
txt = S.compare_text(res)
chk("compare_text prints both models and the deltas", "model-A Q3" in txt and "model-B Q3" in txt and "+10 % better" in txt)

# CLI + private by default
d = tempfile.mkdtemp(); fa, fb, fh = (os.path.join(d, x) for x in ("a.json", "b.json", "hard.json"))
json.dump(A, open(fa, "w")); json.dump(B, open(fb, "w")); json.dump(S.section_hard(run, "abc"), open(fh, "w"))
cli = lambda *a: subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache_suite.py"), *a], capture_output=True, text=True)
r = cli("validate", fa, fb); chk("CLI validate: ok, exit 0", r.returncode == 0 and r.stdout.count("ok") == 2)
r = cli("compare", fa, fb); chk("CLI compare prints the table", r.returncode == 0 and "hard tasks passed" in r.stdout)
r = cli("compare", fa, fb, "--json"); chk("CLI compare --json is valid JSON with rows", r.returncode == 0 and json.loads(r.stdout)["rows"])
out = os.path.join(d, "n.json"); r = cli("new", "--model", "m", "--quant", "Q4", "--section", f"hard={fh}", "--out", out)
chk("CLI new builds a valid private record from a section file", r.returncode == 0 and json.load(open(out))["published"] is False and S.validate_record(json.load(open(out))) == [])
r = cli("publish", out); chk("publish sets the flag and says nothing reads it yet", r.returncode == 0 and json.load(open(out))["published"] is True and "nothing reads" in r.stdout)
print("suite_record_test: " + ("FAIL" if fails else "PASS"))
sys.exit(1 if fails else 0)
