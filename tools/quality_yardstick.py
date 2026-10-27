#!/usr/bin/env python3
"""Quality yardstick 2.0: does a mode or a quant change the answers, and how much?

Runs a fixed set of greedy prompts against a running llama-server (stock or moe-cache, any OS) and saves the tokens.
Comparing two runs gives:
  flip rate      share of prompts whose greedy tokens differ from the reference run, and where the first difference is
  loops          prompts whose answer repeats a 12-token sequence 3+ times (the IQ2 failure mode: endless repetition)
  task score     exact-answer questions (arithmetic, facts, units, logic) checked against the expected answer
KL against a reference model's logits is a separate subcommand around llama-perplexity (--kl-divergence).

Standard library only (Windows and Linux). Greedy = temperature 0, top_k 1, fixed seed, no prompt cache, one request at a time.

usage:
  quality_yardstick.py run     --url http://127.0.0.1:8080 --out ref.json [--label stock] [--n-predict 256] [--no-think] [--prompts FILE.json]
  quality_yardstick.py compare REF.json TEST.json [TEST2.json ...]
  quality_yardstick.py kl      --perplexity BIN --model GGUF --text FILE --kld BASE.kld (--make-base | --test) [-- extra llama-perplexity args]

The reference run should be the most exact thing available: stock llama.cpp on the same file for a mode check (--fast, --hybrid),
or the higher-precision quant for a quant check (then flip rate measures quant damage, not identity). Compare runs made with
the same prompts, --n-predict and --no-think setting; `compare` refuses otherwise.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request

API_KEY = os.environ.get("LLAMA_API_KEY", "")   # set by --api-key; agent-mode servers need one (`moe_cache.py key` prints it)

# Open-ended prompts: long greedy answers, where flips and loops show. Kept short so the prompt cost is small.
OPEN = [
    "Explain how a refrigerator moves heat out of its inside, step by step.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a short docstring.",
    "Describe the water cycle to a ten-year-old.",
    "What are the main differences between TCP and UDP? Give one use case for each.",
    "Write a short story (about 150 words) about a lighthouse keeper who finds a message in a bottle.",
    "Summarise the causes of the First World War in a few paragraphs.",
    "Explain what a hash table is and how collisions are handled.",
    "Give a recipe for a simple tomato pasta sauce.",
    "Write a SQL query that returns the five customers with the highest total order value, given tables customers(id, name) and orders(id, customer_id, amount).",
    "Explain the difference between a virus and a bacterium.",
    "List ten tips for writing clear technical documentation, one sentence each.",
    "Explain how public-key cryptography lets two strangers agree on a secret.",
    "Write a bash script that counts the lines in every .txt file in a folder and prints the total.",
    "What happens in the body during a fever, and why can it help fight infection?",
    "Compare renting and buying a home: list the main pros and cons of each.",
    "Explain gradient descent and the role of the learning rate.",
    "Write a haiku about autumn, then explain the imagery you chose.",
    "Describe how a bill becomes a law in a parliamentary democracy, in general terms.",
    "Explain recursion with an example in JavaScript.",
    "Why is the sky blue and why are sunsets red?",
    "Write an email to a colleague asking to move a meeting from Tuesday to Thursday, politely.",
    "Explain what a mixture-of-experts language model is.",
    "Describe three sorting algorithms and their time complexity.",
    "Explain photosynthesis, including the light and dark reactions.",
]

# Exact-answer questions: (prompt, accepted answers as regexes, matched against the answer with thinking removed).
TASKS = [
    ("What is 17 * 23?", [r"\b391\b"]),
    ("What is 1234 + 5678?", [r"\b6,?912\b"]),
    ("What is 144 divided by 12?", [r"\b12\b"]),
    ("What is 15% of 80?", [r"\b12(\.0+)?\b"]),
    ("What is the square root of 169?", [r"\b13\b"]),
    ("What is 2 to the power of 10?", [r"\b1,?024\b"]),
    ("How many minutes are in 3.5 hours?", [r"\b210\b"]),
    ("How many centimetres are in 2.5 metres?", [r"\b250\b"]),
    ("What is the capital of Australia?", [r"canberra"]),
    ("What is the capital of Canada?", [r"ottawa"]),
    ("What is the chemical symbol for gold?", [r"\bau\b"]),
    ("Which planet is closest to the Sun?", [r"mercury"]),
    ("How many sides does a hexagon have?", [r"\b6\b|\bsix\b"]),
    ("In which year did the Second World War end?", [r"\b1945\b"]),
    ("Who wrote the play Romeo and Juliet?", [r"shakespeare"]),
    ("What is the boiling point of water at sea level in degrees Celsius?", [r"\b100\b"]),
    ("If Alice is taller than Bob and Bob is taller than Carol, who is the shortest?", [r"carol"]),
    ("A shop sells pens at 3 for 2 dollars. How many dollars do 12 pens cost?", [r"\b8\b|\beight\b"]),
    ("What is the next number in the sequence 2, 4, 8, 16, ...?", [r"\b32\b"]),
    ("How many days are in a leap year?", [r"\b366\b"]),
]
TASK_SUFFIX = " Answer with only the final answer, no explanation."
LOOP_N, LOOP_REPEATS = 12, 3


def post(url, path, body, timeout=1800):
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = "Bearer " + API_KEY
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get(url, path):
    try:
        req = urllib.request.Request(url.rstrip("/") + path, headers={"Authorization": "Bearer " + API_KEY} if API_KEY else {})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}


def strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.split("</think>")[-1]          # templates that open the think block in the prompt


def has_loop(tokens, n=LOOP_N, repeats=LOOP_REPEATS):
    seen = {}
    for i in range(len(tokens) - n + 1):
        k = tuple(tokens[i:i + n])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= repeats:
            return True
    return False


def cmd_run(a):
    global API_KEY
    API_KEY = a.api_key or API_KEY
    items = [{"kind": "open", "prompt": p} for p in OPEN] + [{"kind": "task", "prompt": p + TASK_SUFFIX, "answers": ans} for p, ans in TASKS]
    if a.prompts:
        items = json.load(open(a.prompts, encoding="utf-8"))
    props = get(a.url, "/props")
    out = {"label": a.label or "", "url": a.url, "n_predict": a.n_predict, "no_think": a.no_think, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "model": props.get("model_path", ""), "build": props.get("build_info", ""), "items": []}
    t0 = time.time()
    for i, it in enumerate(items):
        body = {"messages": [{"role": "user", "content": it["prompt"]}]}
        if a.no_think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        prompt = post(a.url, "/apply-template", body)["prompt"]
        n_pred = a.n_predict if it["kind"] == "open" else a.task_n_predict
        r = post(a.url, "/completion", {"prompt": prompt, "n_predict": n_pred, "temperature": 0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
                                         "repeat_penalty": 1.0, "presence_penalty": 0.0, "seed": 1, "cache_prompt": False, "return_tokens": True,
                                         "stream": False})
        rec = dict(it, tokens=r.get("tokens", []), text=r.get("content", ""), n_tokens=len(r.get("tokens", [])),
                   stop=r.get("stop_type", ""), tok_s=(r.get("timings") or {}).get("predicted_per_second"))
        out["items"].append(rec)
        print(f"[{i + 1}/{len(items)}] {it['kind']:4s} {rec['n_tokens']:4d} tok  {(rec['tok_s'] or 0):6.1f} tok/s  {it['prompt'][:50]!r}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    s = summary(out)
    print(f"saved {a.out}: {len(items)} prompts in {out['seconds']} s; loops {s['loops']}/{s['n']}; tasks {s['task_ok']}/{s['task_n']}")


def summary(run):
    loops = sum(1 for it in run["items"] if has_loop(it["tokens"]))
    tasks = [it for it in run["items"] if it["kind"] == "task"]
    ok = sum(1 for it in tasks if any(re.search(p, strip_think(it["text"]).lower()) for p in it["answers"]))
    return {"n": len(run["items"]), "loops": loops, "task_n": len(tasks), "task_ok": ok}


def cmd_compare(a):
    ref = json.load(open(a.ref, encoding="utf-8"))
    rs = summary(ref)
    rows = [("reference: " + (ref["label"] or a.ref), rs, None)]
    for path in a.test:
        t = json.load(open(path, encoding="utf-8"))
        if [it["prompt"] for it in t["items"]] != [it["prompt"] for it in ref["items"]] or t["n_predict"] != ref["n_predict"] or t["no_think"] != ref["no_think"]:
            sys.exit(f"{path}: different prompts, --n-predict or --no-think than the reference; rerun with the same settings")
        flips, first, agree = 0, [], []
        task_changed = 0
        for r_it, t_it in zip(ref["items"], t["items"]):
            rt, tt = r_it["tokens"], t_it["tokens"]
            m = min(len(rt), len(tt))
            d = next((j for j in range(m) if rt[j] != tt[j]), None)
            if d is None and len(rt) != len(tt):
                d = m
            if d is not None:
                flips += 1
                first.append(d)
            agree.append((d if d is not None else max(len(rt), 1)) / max(len(rt), 1))
            if r_it["kind"] == "task":
                ok_r = any(re.search(p, strip_think(r_it["text"]).lower()) for p in r_it["answers"])
                ok_t = any(re.search(p, strip_think(t_it["text"]).lower()) for p in t_it["answers"])
                task_changed += ok_r != ok_t
        first.sort()
        rows.append((t["label"] or path, summary(t), {"flips": flips, "n": len(ref["items"]), "median_first": first[len(first) // 2] if first else None,
                                                       "agree": sum(agree) / len(agree), "task_changed": task_changed}))
    print(f"{'run':<34} {'flip rate':>10} {'median 1st diff':>16} {'same prefix':>12} {'loops':>7} {'tasks right':>12} {'tasks changed':>14}")
    for name, s, c in rows:
        if c is None:
            print(f"{name[:34]:<34} {'-':>10} {'-':>16} {'-':>12} {s['loops']:>3}/{s['n']:<3} {s['task_ok']:>5}/{s['task_n']:<6} {'-':>14}")
        else:
            fr = f"{100 * c['flips'] / c['n']:.0f} % ({c['flips']})"
            md = "-" if c["median_first"] is None else f"token {c['median_first']}"
            print(f"{name[:34]:<34} {fr:>10} {md:>16} {100 * c['agree']:>10.0f} % {s['loops']:>3}/{s['n']:<3} {s['task_ok']:>5}/{s['task_n']:<6} {c['task_changed']:>14}")
    print("flip rate: prompts whose greedy tokens differ from the reference anywhere; same prefix: mean share of the reference answer reproduced before the "
          "first difference; tasks changed: questions right in one run and wrong in the other.")


def cmd_kl(a):
    extra = a.extra[1:] if a.extra[:1] == ["--"] else a.extra
    cmd = [a.perplexity, "-m", a.model, "-f", a.text, "--kl-divergence-base", a.kld, "-c", str(a.ctx), "--chunks", str(a.chunks)] + extra
    if a.test:
        cmd.append("--kl-divergence")
    print("running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    log = p.stdout
    with open(a.kld + (".test.log" if a.test else ".base.log"), "w", encoding="utf-8") as f:
        f.write(log)
    if p.returncode != 0:
        sys.exit(f"llama-perplexity failed (exit {p.returncode}); log in {f.name}")
    if not a.test:
        print(f"base logits saved to {a.kld}; log {f.name}")
        return
    want = ("Mean    KLD", "Maximum KLD", "99.0%   KLD", "Median  KLD", "Same top p", "RMS Δp", "Mean    PPL(Q)", "Mean    PPL(base)")
    for line in log.splitlines():
        if line.strip().startswith(want):
            print(line.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run", help="run the prompt set against a server and save the tokens")
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--out", required=True)
    r.add_argument("--label", default="")
    r.add_argument("--api-key", default="", help="server API key (default: env LLAMA_API_KEY)")
    r.add_argument("--n-predict", type=int, default=256, help="tokens per open-ended prompt (default 256)")
    r.add_argument("--task-n-predict", type=int, default=64, help="tokens per exact-answer question (default 64; raise it for thinking models)")
    r.add_argument("--no-think", action="store_true", help="pass enable_thinking=false to the chat template (Qwen-style thinking models)")
    r.add_argument("--prompts", help="JSON list of {kind: open|task, prompt, answers: [regex]} instead of the built-in set")
    c = sp.add_parser("compare", help="flip rate, loops and task score of runs against a reference run")
    c.add_argument("ref")
    c.add_argument("test", nargs="+")
    k = sp.add_parser("kl", help="KL divergence via llama-perplexity: --make-base on the reference, then --test on the candidate")
    k.add_argument("--perplexity", required=True, help="path to llama-perplexity(.exe)")
    k.add_argument("--model", required=True)
    k.add_argument("--text", required=True, help="plain text file, e.g. wiki.test.raw or any long document")
    k.add_argument("--kld", required=True, help="base logits file (written with --make-base, read with --test)")
    k.add_argument("--ctx", type=int, default=512)
    k.add_argument("--chunks", type=int, default=20)
    g = k.add_mutually_exclusive_group(required=True)
    g.add_argument("--make-base", action="store_true")
    g.add_argument("--test", action="store_true")
    k.add_argument("extra", nargs=argparse.REMAINDER, help="after --: more llama-perplexity args (e.g. -ngl 99 -t 16)")
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare, "kl": cmd_kl}[a.cmd](a)


if __name__ == "__main__":
    main()
