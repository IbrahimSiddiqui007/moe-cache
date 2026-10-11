#!/usr/bin/env python3
"""Quality harness: does a candidate configuration still answer like the reference one, and does it stay smart over long contexts and long sessions?

  bench/quality.py compare --ref URL --cand URL [--ref-key K] [--cand-key K] [--long 8000,32000] [--out report.json]
  bench/quality.py longctx --url URL [--key K] [--sizes 4000,16000,64000,120000]
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
    ("code1", "code", "Write a Python function is_palindrome(s) that returns True if s reads the same forwards and backwards. Only the code.", [r"def\s+is_palindrome", r"\[::-1\]|reversed|==\s*s\[::-1\]"]),
    ("code2", "code", "Write a Python one-liner that returns the sum of the squares of the numbers in a list nums. Only the code.", [r"sum\(|reduce|\+\="]),
    ("code3", "code", "In Python, what does len([1, 2, 3][1:]) evaluate to? Answer with the number.", [r"\b2\b"]),
    ("fmt1", "format", "List exactly three colours as a numbered list, nothing else.", [r"(?s)^\s*1[.)].*\n\s*2[.)].*\n\s*3[.)]"]),
    ("fmt2", "format", "Reply with only the word YES in capital letters.", [r"^\s*YES\s*[.!]?\s*$"]),
    ("fmt3", "format", "Write one sentence that contains the word 'lighthouse' and ends with a question mark.", [r"lighthouse.*\?\s*$"]),
    ("sum1", "summarize", "Summarize in one sentence: 'The city council voted on Tuesday to extend the bus network to the eastern suburbs. The project will cost 40 million dollars and is expected to be finished by 2028. Opponents argued the money should go to road repair instead.'", [r"bus", r"east"]),
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
    if pats is None:
        return None
    t = text.strip()
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True); c.add_argument("--cand", required=True)
    c.add_argument("--ref-key", default=os.environ.get("REF_KEY", "")); c.add_argument("--cand-key", default=os.environ.get("CAND_KEY", ""))
    c.add_argument("--long", default="", help="comma separated context sizes in tokens for the retrieval test, e.g. 8000,32000")
    c.add_argument("--max-drop", type=float, default=5.0); c.add_argument("--min-agree", type=float, default=80.0); c.add_argument("--out", default="")
    l = sub.add_parser("longctx")
    l.add_argument("--url", required=True); l.add_argument("--key", default=os.environ.get("KEY", "")); l.add_argument("--sizes", default="4000,16000,32000")
    s = sub.add_parser("soak")
    s.add_argument("--url", required=True); s.add_argument("--key", default=os.environ.get("KEY", ""))
    s.add_argument("--minutes", type=float, default=30); s.add_argument("--probe-every", type=float, default=5); s.add_argument("--out", default="")
    a = ap.parse_args()
    sys.exit({"compare": cmd_compare, "longctx": cmd_longctx, "soak": cmd_soak}[a.cmd](a))


if __name__ == "__main__":
    main()
