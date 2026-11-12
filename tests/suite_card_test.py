#!/usr/bin/env python3
"""TEST-1142: bin/moe_cache_suite_card.py - one card per record, reference first, deltas with better / worse / flat classes, escaping of record fields, no external resources or scripts, dark mode, CLI.
   python3 tests/suite_card_test.py"""
import os, subprocess, sys, tempfile, json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_suite as S, moe_cache_suite_card as C      # noqa: E402
fails = 0


def chk(name, cond):
    global fails
    print(("  ok   " if cond else "  FAIL ") + name)
    fails += 0 if cond else 1


def rec(name, hard, think, dec, proto="P"):
    return S.new_record({"name": name, "quant": "Q3", "size_gb": 13.1}, machine={"platform": "Linux x"}, settings={}, sections={
        "speed": {"protocol": "S", "bands": [{"depth": 32768, "prompt_tps": 300.0, "decode_tps": dec}], "ttft_ms": 900},
        "quality": {"protocol": "Q", "ref": "f16", "kl_mean": 0.1, "top1_pct": 95.0},
        "hard": {"protocol": proto, "n": 27, "pass": hard, "think_median": think, "think_p90": think * 2, "think_max": think * 3, "loops": 1, "cap_hits": 0, "cap": 0}})


A, B = rec("model-A", 9, 9000, 25.0), rec('<script>alert(1)</script>', 24, 5000, 20.0)
page = C.render([A, B], "Cmp & <b>")
chk("one card per record, the first labelled reference", page.count('class="card"') == 2 and "reference" in page and "vs model-A Q3" in page)
chk("values and deltas: hard 9 -> 24 of 27 better (outside the interval), decode -20 % worse, thinking -44 % better", "+167 %" in page and "-20 %" in page and "-44 %" in page and 'class="d better"' in page and 'class="d worse"' in page)
chk("record fields are escaped (no live <script>, title escaped)", "<script>alert" not in page and "&lt;script&gt;" in page and "Cmp &amp; &lt;b&gt;" in page)
chk("self-contained: no external URL, no script tag, no link tag", "http://" not in page and "https://" not in page and "<script" not in page and "<link" not in page)
chk("dark mode and a viewport meta are there", "prefers-color-scheme:dark" in page and "viewport" in page)
chk("private badge shown", page.count(">private<") == 2)
D = rec("model-D", 25, 4000, 25.0, proto="OTHER"); pd = C.render([A, D])
chk("another protocol: not compared, the note says so, the value is a dash", "different protocol" in pd and "hard reasoning" in pd and "-</span></div>" in pd)
d = tempfile.mkdtemp(); fa, fb, out = (os.path.join(d, x) for x in ("a.json", "b.json", "c.html")); json.dump(A, open(fa, "w")); json.dump(B, open(fb, "w"))
r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "bench-suite", "card", fa, fb, "--out", out], capture_output=True, text=True)
chk("CLI: bench-suite card writes the page", r.returncode == 0 and os.path.getsize(out) > 1000 and "<!doctype html>" in open(out).read())
bad = os.path.join(d, "bad.json"); json.dump({"x": 1}, open(bad, "w"))
r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "bench-suite", "card", bad, "--out", out], capture_output=True, text=True)
chk("CLI: an invalid record is refused with exit 1", r.returncode == 1 and "invalid" in r.stderr)
near = C.render([rec("a", 20, 9000, 25.0), rec("b", 24, 9000, 25.0)])
chk("20 -> 24 of 27 is inside the noise of the pass count: grey '=' and no better/worse mark on that row", '+20 %' not in near and near.count('class="d better"') == 0)
chk("tiles say how many tasks (n=27) and that speed is one run", "n=27" in page and "1 run" in page)
chk("the page no longer claims 'same settings'", "same settings" not in page.lower() and "one run" in page)
print("suite_card_test: " + ("FAIL" if fails else "PASS"))
sys.exit(1 if fails else 0)
