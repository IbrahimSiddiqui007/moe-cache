#!/usr/bin/env python3
"""Quality harness: does a candidate configuration still answer like the reference one, and does it stay smart over long contexts and long sessions?

  bench/quality.py compare --ref URL --cand URL [--ref-key K] [--cand-key K] [--long 8000,32000] [--out report.json]
  bench/quality.py longctx --url URL [--key K] [--sizes 4000,16000,64000,120000]
  bench/quality.py judge-calibrate --url URL                              (is this server a usable judge of "same meaning"? needs 9 of 10)
  bench/quality.py collect --url URL [--key K] --out answers.json       (one server at a time: when two models cannot be loaded together)
  bench/quality.py compare-files --ref ref.json --cand cand.json [--judge-url URL] [--judge-key K]
  bench/quality.py soak    --url URL [--key K] [--minutes 30] [--out soak.json]

URL is a running llama-server (for example started from the benchmark page's Serve card). The reference is the exact configuration (stock llama.cpp, or the
plugin without approximations); the candidate is whatever you want to judge (an approximate kernel, expert substitution, a GPU cache, MTP...).

What `compare` does
  1. A bank of prompts with checkable answers (facts, arithmetic, reasoning, code, instruction following). Both servers answer at temperature 0; each answer is
     scored against the expected keywords. The score drop is the main number.
  2. Every pair of answers is also judged by the reference server: "do these answers reach the same conclusion with the same main reasons?" (SAME / DIFFERENT).
     Different words with the same meaning count as SAME, which is the criterion this project uses (identical tokens are not required).
  3. Degeneration checks: empty answers, repeated phrases, garbled text.
  4. Optional long-context retrieval at the sizes in --long (a fact hidden in a long text, asked at the end).
Exit status 1 when the candidate loses more than --max-drop points of accuracy (default 5) or agrees with the reference on fewer than --min-agree % (default 80).
"""
import argparse, difflib, json, os, random, re, sys, time, urllib.request, urllib.error

BANK = [
    # (id, category, prompt, expected regex list: the answer passes if ANY pattern matches, or None for judge-only prompts)
    ("fact1", "fact", "What is the capital of Australia? Answer in one sentence.", [r"canberra"]),
    ("fact2", "fact", "Who wrote the play Romeo and Juliet? Answer in one sentence.", [r"shakespeare"]),
    ("fact3", "fact", "What is the chemical symbol for gold? Answer briefly.", [r"\bAu\b"]),
    ("fact4", "fact", "Which planet is the largest in our solar system? One sentence.", [r"jupiter"]),
    ("fact5", "fact", "In what year did the first human land on the Moon? One sentence.", [r"1969"]),
    ("fact6", "fact", "What gas do plants absorb from the air for photosynthesis? One sentence.", [r"carbon dioxide|CO2|CO₂"]),
    ("math1", "math", "What is 17 multiplied by 23? Give the number and one line of working.", [r"\b391\b"]),
    ("math2", "math", "A train travels 150 km in 2.5 hours. What is its average speed in km/h? Show the division.", [r"\b60\b"]),
    ("math3", "math", "If x + 3 = 11, what is x * x? Think step by step, then give the number.", [r"\b64\b"]),
    ("math4", "math", "What is 15% of 240? Give the number.", [r"\b36\b"]),
    ("math5", "math", "How many minutes are in 3 days? Show the multiplication.", [r"4,?320"]),
    ("reason1", "reason", "Tom is taller than Anna. Anna is taller than Ben. Who is the shortest? Answer with the name and one reason.", [r"\bben\b"]),
    ("reason2", "reason", "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Answer yes or no and explain in one sentence.", [r"\bno\b|cannot|can't|not necessarily"]),
    ("reason3", "reason", "I have 3 apples. I give away 2, then buy 5 more, then eat 1. How many apples do I have? Show the steps.", [r"\b5\b"]),
    ("reason4", "reason", "Which is heavier, a kilogram of feathers or a kilogram of steel? Answer in one sentence.", [r"same|equal|neither|both"]),
    ("sky", "explain", "Why is the sky blue? Explain in 2 to 3 sentences.", [r"scatter"]),
    ("tide", "explain", "What causes the tides on Earth? Explain in 2 to 3 sentences.", [r"moon|lunar|gravit"]),
    ("seasons", "explain", "Why does Earth have seasons? Explain in 2 to 3 sentences.", [r"tilt|axis|axial"]),
    ("code1", "code", "Write a Python function is_palindrome(s) that returns True if s reads the same forwards and backwards. Only the code.", {"all": [r"def\s+is_palindrome", r"\[::-1\]|reversed\(|\.reverse\(|len\(s\)\s*//\s*2|while\s+\w+\s*<", r"return[^\n]*(==|!=|\band\b|\bnot\b)|==\s*\w+\[::-1\]|\bis_palindrome\("]}),
    ("code2", "code", "Write a Python one-liner that returns the sum of the squares of the numbers in a list nums. Only the code.", {"all": [r"sum\(|reduce|\+\=", r"\*\*\s*2|\bx\s*\*\s*x\b|\bpow\("]}),
    ("code3", "code", "In Python, what does len([1, 2, 3][1:]) evaluate to? Answer with the number.", [r"\b2\b"]),
    ("fmt1", "format", "List exactly three colours as a numbered list, nothing else.", {"all": [r"(?s)^\s*1[.)].*\n\s*2[.)].*\n\s*3[.)]", r"\A(?!.*\n\s*4[.)])"]}),
    ("fmt2", "format", "Reply with only the word YES in capital letters.", [r"^\s*YES\s*[.!]?\s*$"]),
    ("fmt3", "format", "Write one sentence that contains the word 'lighthouse' and ends with a question mark.", [r"lighthouse.*\?\s*$"]),
    ("sum1", "summarize", "Summarize in one sentence: 'The city council voted on Tuesday to extend the bus network to the eastern suburbs. The project will cost 40 million dollars and is expected to be finished by 2028. Opponents argued the money should go to road repair instead.'", {"all": [r"\bbus", r"east"]}),
    ("sum2", "summarize", "Extract the date and the place from: 'The annual conference will take place on 14 March 2027 at the Grand Hall in Lisbon.' Reply as 'date; place'.", [r"14 march 2027.*lisbon|2027.*lisbon"]),
    ("open1", "open", "Give two reasons why exercise is good for health. Be brief.", None),
    ("open2", "open", "Explain what a hash table is to a beginner in three sentences.", None),
    ("open3", "open", "What are the main trade-offs between SQL and NoSQL databases? Be concise.", None),
    ("open4", "open", "Describe how a refrigerator keeps food cold, in simple terms.", None),
]

FILLER = [
    "The committee reviewed the quarterly logistics report and noted that delivery times in the northern region had improved slightly over the previous period.",
    "A small bakery on the corner of the market square introduced a new rye loaf that quickly became popular with commuters on their way to the station.",
    "Researchers measured the temperature of the lake at several depths and recorded how the thermal layers shifted between spring and autumn.",
    "The museum extended its opening hours during the festival week and added two guided tours in the afternoon for visiting school groups.",
    "In the workshop, the apprentices practised cutting dovetail joints while the instructor explained how humidity affects seasoned timber.",
    "The river authority published a schedule for the maintenance of the lock gates, which will be closed one at a time over six weekends.",
    "A travelling theatre company rehearsed a comedy about a lighthouse keeper who mistakes a passing whale for a sinking ship.",
    "The city library digitised thousands of old postcards and made them searchable by street name, photographer and approximate year.",
]


def http_json(url, body=None, key="", timeout=3600):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None, h)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def complete(url, key, prompt, n=200, **kw):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0, "seed": 42, "cache_prompt": False}
    body.update(kw)
    t0 = time.time()
    try:
        r = http_json(url + "/completion", body, key)
    except Exception as e:
        return {"text": "", "error": str(e), "tps": 0, "prompt_n": 0, "wall": time.time() - t0}
    t = r.get("timings", {})
    return {"text": r.get("content", ""), "tps": t.get("predicted_per_second", 0), "prompt_n": t.get("prompt_n", 0),
            "prompt_ms": t.get("prompt_ms", 0), "wall": time.time() - t0}


def chat_prompt(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def ask(url, key, q, n=200):
    """Use the server's chat template when it has one; fall back to a plain prompt."""
    try:
        r = http_json(url + "/v1/chat/completions", {"messages": [{"role": "user", "content": q}], "max_tokens": n, "temperature": 0, "seed": 42}, key)
        return {"text": r["choices"][0]["message"].get("content") or "", "tps": 0}
    except Exception:
        return complete(url, key, q, n)


def passes(text, pats):
    """pats: None (judge only), a list (any pattern matches) or {"all": [...]} (every pattern must match)."""
    if pats is None:
        return None
    t = text.strip()
    if not t:
        return False
    if isinstance(pats, dict):
        return all(re.search(p, t, re.I | re.S | re.M) for p in pats["all"])
    return any(re.search(p, t, re.I | re.S | re.M) for p in pats)


def degenerate(text):
    t = text.strip()
    if not t:
        return "empty"
    words = re.findall(r"\w+", t.lower())
    if len(words) >= 24:
        grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
        if len(set(grams)) < 0.35 * len(grams):
            return "repeating"
    bad = sum(1 for ch in t if ord(ch) > 0x2FFF or ch == "�")
    if bad > 0.2 * len(t):
        return "garbled"
    return ""


def judge(ref_url, ref_key, q, a, b):
    p = ("You compare two answers to the same question. Decide whether they reach the same conclusion and give the same main reasons. "
         "Different wording does not matter; a different final answer, a different main reason or a factual disagreement does.\n\n"
         f"Question: {q}\n\nAnswer A: {a[:1500]}\n\nAnswer B: {b[:1500]}\n\nReply with exactly one word: SAME or DIFFERENT.")
    r = ask(ref_url, ref_key, p, 6)
    return "SAME" in r["text"].upper() and "DIFFERENT" not in r["text"].upper()


def needle_prompt(approx_tokens, depth, seed):
    rnd = random.Random(seed)
    code = f"{rnd.choice(['maple', 'cedar', 'birch', 'willow'])}-{rnd.randint(1000, 9999)}"
    vault = rnd.randint(11, 97)
    needle = f"The access code for vault {vault} is {code}."
    words_target = int(approx_tokens * 0.72)
    out, n = [], 0
    while n < words_target:
        s = rnd.choice(FILLER)
        out.append(s)
        n += len(s.split())
    out.insert(int(len(out) * depth), needle)
    text = " ".join(out)
    return text + f"\n\nQuestion: What is the access code for vault {vault}? Answer with the code only.\nAnswer:", code


def cmd_compare(a):
    rows, fails = [], []
    for pid, cat, q, pats in BANK:
        r = ask(a.ref, a.ref_key, q)
        c = ask(a.cand, a.cand_key, q)
        row = {"id": pid, "cat": cat, "ref": r["text"], "cand": c["text"], "ref_ok": passes(r["text"], pats), "cand_ok": passes(c["text"], pats),
               "sim": round(difflib.SequenceMatcher(None, r["text"], c["text"]).ratio(), 3), "ref_deg": degenerate(r["text"]), "cand_deg": degenerate(c["text"])}
        row["ref_err"], row["cand_err"] = bool(r.get("error")) or not r["text"].strip(), bool(c.get("error")) or not c["text"].strip()
        if row["ref_err"] or row["cand_err"]:
            row["same"] = False                      # an empty or failed answer is never "the same"
        else:
            row["same"] = True if r["text"].strip() == c["text"].strip() else judge(a.ref, a.ref_key, q, r["text"], c["text"])
        rows.append(row)
        print(f"{pid:8} ref={row['ref_ok']!s:5} cand={row['cand_ok']!s:5} same={row['same']!s:5} sim={row['sim']:.2f} {row['cand_deg']}", flush=True)
    chk = [r for r in rows if r["ref_ok"] is not None]
    ref_acc = 100 * sum(1 for r in chk if r["ref_ok"]) / max(1, len(chk))
    cand_acc = 100 * sum(1 for r in chk if r["cand_ok"]) / max(1, len(chk))
    agree = 100 * sum(1 for r in rows if r["same"]) / len(rows)
    deg = [r["id"] for r in rows if r["cand_deg"] and not r["ref_deg"]]
    longres = []
    for size in [int(x) for x in a.long.split(",") if x]:
        for depth in (0.1, 0.5, 0.9):
            q, code = needle_prompt(size, depth, seed=size + int(depth * 10))
            rr, cc = complete(a.ref, a.ref_key, q, 20), complete(a.cand, a.cand_key, q, 20)
            longres.append({"tokens": size, "depth": depth, "ref_ok": code in rr["text"], "cand_ok": code in cc["text"], "cand_tps": cc["tps"],
                            "prompt_n": cc["prompt_n"]})
            print(f"long {size:>6} depth {depth}: ref={code in rr['text']} cand={code in cc['text']}", flush=True)
    res = {"ref_accuracy": round(ref_acc, 1), "cand_accuracy": round(cand_acc, 1), "drop_points": round(ref_acc - cand_acc, 1), "agree_pct": round(agree, 1),
           "new_degenerate": deg, "long": longres, "rows": rows}
    print(f"\naccuracy: reference {ref_acc:.1f} %  candidate {cand_acc:.1f} %  (drop {ref_acc - cand_acc:.1f} points) · same meaning {agree:.1f} % · "
          f"new degenerate answers: {len(deg)}")
    for r in longres:
        if r["ref_ok"] and not r["cand_ok"]:
            fails.append(f"long-context retrieval lost at {r['tokens']} tokens, depth {r['depth']}")
    n_ref_err = sum(1 for r in rows if r["ref_err"])
    n_cand_err = sum(1 for r in rows if r["cand_err"])
    if n_ref_err > len(rows) // 5:
        fails.append(f"the reference gave {n_ref_err} empty or failed answers of {len(rows)}: the comparison is not meaningful")
    if n_cand_err > n_ref_err:
        fails.append(f"the candidate gave {n_cand_err - n_ref_err} more empty or failed answers than the reference")
    if ref_acc < a.min_ref_accuracy:
        fails.append(f"the reference itself only scores {ref_acc:.1f} % (needs {a.min_ref_accuracy:.0f} %): the servers are not answering properly")
    if ref_acc - cand_acc > a.max_drop:
        fails.append(f"accuracy dropped {ref_acc - cand_acc:.1f} points (limit {a.max_drop})")
    if agree < a.min_agree:
        fails.append(f"only {agree:.1f} % of answers keep the reference's meaning (limit {a.min_agree})")
    if deg:
        fails.append("degenerate answers: " + ", ".join(deg))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print("VERDICT: " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)))
    return 0 if not fails else 1


def cmd_longctx(a):
    try:
        n_ctx = int(http_json(a.url + "/props", None, a.key).get("default_generation_settings", {}).get("n_ctx", 0))
    except Exception:
        n_ctx = 0
    print(f"server context: {n_ctx or 'unknown'} tokens")
    res = []
    for size in [int(x) for x in a.sizes.split(",") if x]:
        if n_ctx and size + 64 > n_ctx:
            print(f"{size:>7}: skipped (server context is {n_ctx})")
            continue
        for depth in (0.1, 0.5, 0.9):
            q, code = needle_prompt(size, depth, seed=size + int(depth * 10))
            r = complete(a.url, a.key, q, 20)
            ok = code in r["text"]
            res.append({"tokens": size, "depth": depth, "ok": ok, "prompt_n": r["prompt_n"], "prefill_s": round(r.get("prompt_ms", 0) / 1000, 1), "tps": round(r["tps"], 2)})
            print(f"{size:>7} depth {depth}: {'found' if ok else 'MISSED'}  prompt {r['prompt_n']} tokens, prefill {r.get('prompt_ms', 0) / 1000:.1f} s, decode {r['tps']:.1f} tok/s", flush=True)
    miss = [r for r in res if not r["ok"]]
    print(f"\nretrieval: {len(res) - len(miss)}/{len(res)} found")
    return 0 if not miss else 1


CALIBRATION = [
    # (question, answer A, answer B, should be SAME)
    ("Why is the sky blue?", "Sunlight is scattered by air molecules and blue light scatters the most, so the sky looks blue.",
     "The sky appears blue because shorter blue wavelengths are scattered much more strongly by the atmosphere than red ones.", True),
    ("What causes the tides?", "The Moon's gravity pulls on the oceans and, to a lesser degree, the Sun's does too.",
     "Tides come mainly from the gravitational attraction of the Moon (and the Sun) on Earth's water.", True),
    ("What is 17 times 23?", "17 x 23 = 391.", "The product is 391.", True),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote it.", "It was written by Shakespeare around 1595.", True),
    ("Why does Earth have seasons?", "Because the Earth's axis is tilted relative to its orbit.", "Seasons happen since the planet's rotation axis is inclined, changing how directly sunlight hits each hemisphere.", True),
    ("Why is the sky blue?", "Because the ocean reflects its colour onto the sky.", "Air molecules scatter blue light more than red light.", False),
    ("What is 17 times 23?", "17 x 23 = 381.", "The product is 391.", False),
    ("Who wrote Romeo and Juliet?", "Christopher Marlowe wrote it.", "It was written by Shakespeare.", False),
    ("Why does Earth have seasons?", "Because the Earth is closer to the Sun in summer.", "Because of the tilt of the Earth's axis.", False),
    ("What causes the tides?", "Strong winds push the water back and forth.", "The Moon's gravity pulls on the oceans.", False),
]


def cmd_judge_calibrate(a):
    ok = 0
    for q, x, y, want in CALIBRATION:
        got = judge(a.url, a.key, q, x, y)
        ok += 1 if got == want else 0
        print(f"{'ok ' if got == want else 'BAD'} expected {'SAME' if want else 'DIFFERENT':9} judge said {'SAME' if got else 'DIFFERENT':9} {q}")
    print(f"\njudge accuracy on {len(CALIBRATION)} calibration pairs: {ok}/{len(CALIBRATION)} " + ("(usable)" if ok >= 9 else "(NOT reliable: do not trust 'same meaning' figures from this judge)"))
    return 0 if ok >= 9 else 1


def cmd_collect(a):
    out = {}
    for pid, cat, q, pats in BANK:
        r = ask(a.url, a.key, q, a.tokens)
        out[pid] = r["text"]
        print(f"{pid:8} {len(r['text'])} chars", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


def cmd_compare_files(a):
    ref, cand = json.load(open(a.ref)), json.load(open(a.cand))
    rows, ref_ok, cand_ok, n_chk = [], 0, 0, 0
    for pid, cat, q, pats in BANK:
        r, c = ref.get(pid, ""), cand.get(pid, "")
        same = None
        if r.strip() == c.strip():
            same = True
        elif a.judge_url:
            same = judge(a.judge_url, a.judge_key, q, r, c)
        row = {"id": pid, "cat": cat, "ref_ok": passes(r, pats), "cand_ok": passes(c, pats), "sim": round(difflib.SequenceMatcher(None, r, c).ratio(), 3),
               "same": same, "cand_deg": degenerate(c), "ref_deg": degenerate(r)}
        rows.append(row)
        if pats is not None:
            n_chk += 1
            ref_ok += 1 if row["ref_ok"] else 0
            cand_ok += 1 if row["cand_ok"] else 0
        print(f"{pid:8} ref={row['ref_ok']!s:5} cand={row['cand_ok']!s:5} same={same!s:5} sim={row['sim']:.2f} {row['cand_deg']}")
    judged = [r for r in rows if r["same"] is not None]
    print(f"\naccuracy: reference {100 * ref_ok / max(1, n_chk):.1f} %  candidate {100 * cand_ok / max(1, n_chk):.1f} % ({n_chk} checkable prompts)")
    if judged:
        print(f"same meaning: {100 * sum(1 for r in judged if r['same']) / len(judged):.1f} % of {len(judged)} judged/identical")
    print(f"mean text similarity: {sum(r['sim'] for r in rows) / len(rows):.2f}   new degenerate answers: {sum(1 for r in rows if r['cand_deg'] and not r['ref_deg'])}")
    fails = []
    empty_ref = sum(1 for pid, *_ in BANK if not ref.get(pid, "").strip())
    empty_cand = sum(1 for pid, *_ in BANK if not cand.get(pid, "").strip())
    if empty_ref > len(BANK) // 5:
        fails.append(f"{empty_ref} of {len(BANK)} reference answers are empty")
    if empty_cand > empty_ref:
        fails.append(f"{empty_cand - empty_ref} more empty candidate answers than reference answers")
    ref_acc, cand_acc = 100 * ref_ok / max(1, n_chk), 100 * cand_ok / max(1, n_chk)
    if ref_acc < 50:
        fails.append(f"the reference scores only {ref_acc:.1f} %")
    if ref_acc - cand_acc > 5:
        fails.append(f"accuracy dropped {ref_acc - cand_acc:.1f} points")
    print("VERDICT: " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)))
    return 0 if not fails else 1


def server_rss_mb(url):
    m = re.search(r":(\d+)", url.split("//")[-1])
    if not m:
        return 0
    port = m.group(1)
    try:
        out = os.popen(f"ss -ltnp 2>/dev/null | grep ':{port} '").read()
        pid = re.search(r"pid=(\d+)", out)
        if pid:
            for l in open(f"/proc/{pid.group(1)}/status"):
                if l.startswith("VmRSS"):
                    return int(l.split()[1]) // 1024
    except Exception:
        pass
    return 0


def cmd_soak(a):
    probes = [(pid, q, pats) for pid, cat, q, pats in BANK if pats is not None][:12]
    base = {}
    for pid, q, pats in probes:
        base[pid] = ask(a.url, a.key, q)["text"]
    t_end = time.time() + 60 * a.minutes
    log, rnd, nxt_probe, n = [], random.Random(1), time.time() + 60 * a.probe_every, 0
    ctx = []
    while time.time() < t_end:
        size = rnd.choice([200, 800, 3000, 9000])
        q, _ = needle_prompt(size, rnd.random(), seed=n)
        r = complete(a.url, a.key, q, 48)
        n += 1
        log.append({"t": int(time.time()), "kind": "load", "tokens": r["prompt_n"], "tps": round(r["tps"], 2), "rss_mb": server_rss_mb(a.url)})
        if time.time() >= nxt_probe:
            nxt_probe = time.time() + 60 * a.probe_every
            same = sum(1 for pid, q2, pats in probes if ask(a.url, a.key, q2)["text"].strip() == base[pid].strip())
            ok = sum(1 for pid, q2, pats in probes if passes(ask(a.url, a.key, q2)["text"], pats))
            tps = [x["tps"] for x in log[-10:] if x["kind"] == "load" and x["tps"]]
            row = {"t": int(time.time()), "kind": "probe", "same_as_first": f"{same}/{len(probes)}", "correct": f"{ok}/{len(probes)}",
                   "tps_recent": round(sum(tps) / max(1, len(tps)), 2), "rss_mb": server_rss_mb(a.url), "requests": n}
            log.append(row)
            print(json.dumps(row), flush=True)
    probe_rows = [x for x in log if x["kind"] == "probe"]
    print(f"\n{n} requests; probes: " + ", ".join(f"{p['same_as_first']} same/{p['correct']} ok" for p in probe_rows))
    if a.out:
        json.dump({"baseline": base, "log": log}, open(a.out, "w"), indent=1)
    bad = [p for p in probe_rows if int(p["correct"].split("/")[0]) < int(p["correct"].split("/")[1]) * 0.8]
    return 1 if bad else 0


CODEGEN = [
    ("fizzbuzz", "Write a Python function fizzbuzz(n) that returns a list of strings for 1..n: 'Fizz' for multiples of 3, 'Buzz' for 5, 'FizzBuzz' for both, else the number as a string.",
     "assert fizzbuzz(5) == ['1','2','Fizz','4','Buzz']\nassert fizzbuzz(15)[-1] == 'FizzBuzz'\nassert fizzbuzz(0) == []"),
    ("is_palindrome", "Write a Python function is_palindrome(s) that returns True if s reads the same forwards and backwards, ignoring case and non-alphanumeric characters.",
     "assert is_palindrome('A man, a plan, a canal: Panama') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True"),
    ("merge_intervals", "Write a Python function merge_intervals(iv) that takes a list of [start, end] lists and returns the merged, sorted list of non-overlapping [start, end] lists.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([]) == []"),
    ("fib", "Write a Python function fib(n) returning the n-th Fibonacci number with fib(0)=0 and fib(1)=1, computed iteratively.",
     "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(10) == 55\nassert fib(50) == 12586269025"),
    ("flatten", "Write a Python function flatten(x) that flattens an arbitrarily nested list of integers into a single flat list, preserving order.",
     "assert flatten([1,[2,[3,4]],5]) == [1,2,3,4,5]\nassert flatten([]) == []\nassert flatten([[[]]]) == []"),
    ("word_count", "Write a Python function word_count(text) that returns a dict mapping each lowercase word to how often it occurs; words are separated by whitespace and stripped of the characters .,!?",
     "assert word_count('The cat. the DOG, the end!') == {'the': 3, 'cat': 1, 'dog': 1, 'end': 1}\nassert word_count('') == {}"),
    ("rotate", "Write a Python function rotate(lst, k) that returns a new list rotated to the right by k positions (k may be larger than len(lst) or zero; empty list allowed).",
     "assert rotate([1,2,3,4,5], 2) == [4,5,1,2,3]\nassert rotate([1,2,3], 7) == [3,1,2]\nassert rotate([], 3) == []\nassert rotate([1,2], 0) == [1,2]"),
    ("is_prime", "Write a Python function is_prime(n) that returns True if the integer n is prime, False otherwise (n < 2 is not prime).",
     "assert [x for x in range(20) if is_prime(x)] == [2,3,5,7,11,13,17,19]\nassert is_prime(7919) is True\nassert is_prime(7917) is False"),
    ("roman_to_int", "Write a Python function roman_to_int(s) that converts a Roman numeral string (I V X L C D M, with subtractive pairs) to an integer.",
     "assert roman_to_int('III') == 3\nassert roman_to_int('IV') == 4\nassert roman_to_int('MCMXCIV') == 1994\nassert roman_to_int('LVIII') == 58"),
    ("balanced", "Write a Python function balanced(s) that returns True if the brackets ()[]{} in the string s are correctly nested and closed, ignoring all other characters.",
     "assert balanced('([]{})') is True\nassert balanced('(]') is False\nassert balanced('((') is False\nassert balanced('a(b)c') is True\nassert balanced('') is True"),
]


def extract_code(text):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return max(m, key=len) if m else ("" if "def " not in text else text[text.index("def "):])


def run_code(code, tests, timeout=10):
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", code + "\n\n" + tests], cwd=d, env={}, capture_output=True, text=True, timeout=timeout)
            return r.returncode == 0, (r.stderr or "")[-200:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def cmd_codegen(a):
    """10 small Python tasks; the generated function is run against assertions. Prints pass count; --out saves the details."""
    rows = []
    for name, q, tests in CODEGEN:
        r = ask(a.url, a.key, q + " Reply with one Python code block and nothing else.", a.tokens)
        code = extract_code(r["text"])
        ok, err = run_code(code, tests) if code else (False, "no code in the answer")
        rows.append({"task": name, "pass": ok, "error": err, "chars": len(r["text"])})
        print(f"{name:<16} {'PASS' if ok else 'FAIL'}  {err[:80].strip()}")
    n = sum(r["pass"] for r in rows)
    print(f"codegen: {n}/{len(rows)} pass")
    if a.out:
        json.dump({"pass": n, "total": len(rows), "rows": rows}, open(a.out, "w"), indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True); c.add_argument("--cand", required=True)
    c.add_argument("--ref-key", default=os.environ.get("REF_KEY", "")); c.add_argument("--cand-key", default=os.environ.get("CAND_KEY", ""))
    c.add_argument("--long", default="", help="comma separated context sizes in tokens for the retrieval test, e.g. 8000,32000")
    c.add_argument("--max-drop", type=float, default=5.0); c.add_argument("--min-ref-accuracy", type=float, default=50.0); c.add_argument("--min-agree", type=float, default=80.0); c.add_argument("--out", default="")
    l = sub.add_parser("longctx")
    l.add_argument("--url", required=True); l.add_argument("--key", default=os.environ.get("KEY", "")); l.add_argument("--sizes", default="4000,16000,32000")
    j = sub.add_parser("judge-calibrate")
    j.add_argument("--url", required=True); j.add_argument("--key", default=os.environ.get("KEY", ""))
    k = sub.add_parser("collect")
    k.add_argument("--url", required=True); k.add_argument("--key", default=os.environ.get("KEY", "")); k.add_argument("--out", required=True); k.add_argument("--tokens", type=int, default=200)
    f = sub.add_parser("compare-files")
    f.add_argument("--ref", required=True); f.add_argument("--cand", required=True); f.add_argument("--judge-url", default=""); f.add_argument("--judge-key", default="")
    g = sub.add_parser("codegen")
    g.add_argument("--url", required=True); g.add_argument("--key", default=os.environ.get("KEY", "")); g.add_argument("--tokens", type=int, default=1200); g.add_argument("--out", default="")
    s = sub.add_parser("soak")
    s.add_argument("--url", required=True); s.add_argument("--key", default=os.environ.get("KEY", ""))
    s.add_argument("--minutes", type=float, default=30); s.add_argument("--probe-every", type=float, default=5); s.add_argument("--out", default="")
    a = ap.parse_args()
    sys.exit({"compare": cmd_compare, "longctx": cmd_longctx, "soak": cmd_soak, "collect": cmd_collect, "judge-calibrate": cmd_judge_calibrate, "compare-files": cmd_compare_files, "codegen": cmd_codegen}[a.cmd](a))


if __name__ == "__main__":
    main()
