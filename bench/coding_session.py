#!/usr/bin/env python3
"""Coding-session benchmark: successive requests that share one large code context, the way an editor or agent uses a model.

  bench/coding_session.py --url http://127.0.0.1:8080 [--key K] [--repo DIR] [--tokens 16000] [--questions 6] [--out session.json]

What it does
  1. Builds a context of about --tokens tokens from real source files (default: this repository's src/, bench/ and docs/), as one block of text.
  2. Sends --questions requests, each = the same context + a different question, to the same slot with prompt reuse switched on
     (llama.cpp keeps the key/value cache of the common prefix, so only the question has to be processed).
  3. Then "edits" a file in the middle of the context and asks again: only the part after the edit has to be processed again.
  4. Reports, per request: tokens processed now, tokens reused, time to first token (prompt time), decode speed and the wall time.

The first request pays for the whole context (this is the number that makes long-context use slow); the following ones show how well prompt reuse works
on this machine, including whether the server keeps the earlier prompt in memory (--cache-ram) when other requests come in between.
"""
import argparse, json, os, sys, time, urllib.request

QUESTIONS = [
    "Which function is responsible for making an expert resident in memory before it is used? Answer in two sentences.",
    "What happens when the memory limit is smaller than what the cache needs? Answer in two sentences.",
    "List the environment variables that change the executor's behaviour.",
    "How does the code decide that the model's weights are bandwidth-bound? Answer in two sentences.",
    "Where are the statistics at exit printed, and what do the counters mean? Answer briefly.",
    "Summarise how a read of one expert from the model file works. Answer in three sentences.",
    "Which parts of this code only exist for tests? Answer briefly.",
    "What would you check first if the plugin did not load? Answer briefly.",
]


def build_context(repo, tokens):
    exts = (".cpp", ".h", ".py", ".sh", ".md")
    files = []
    for sub in ("src", "bench", "bin", "docs", "tests", "tools"):
        d = os.path.join(repo, sub)
        for dp, dn, fn in os.walk(d):
            for f in sorted(fn):
                if f.endswith(exts):
                    files.append(os.path.join(dp, f))
    budget = tokens * 2.0                 # characters; code is about 2.25 characters per token on the Flash-Next tokenizer (measured)
    cap = int(min(14000, max(1500, budget / 4)))   # at least four files share the budget
    parts, size = [], 0
    for f in files:
        try:
            t = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if len(t) > cap:                  # one huge file must not use the whole budget: keep its beginning
            t = t[:cap] + "\n... (file shortened)\n"
        chunk = f"\n=== file: {os.path.relpath(f, repo)} ===\n{t}\n"
        if size + len(chunk) > budget:
            if budget - size > 500:       # fill the rest of the budget with the beginning of this file
                parts.append(chunk[:int(budget - size)])
            break
        parts.append(chunk)
        size += len(chunk)
    return parts


def ask(url, key, prompt, n, slot=0):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0, "seed": 42, "cache_prompt": True, "id_slot": slot}
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/completion", json.dumps(body).encode(), h), timeout=7200))
    t = r.get("timings", {})
    return {"processed": t.get("prompt_n", 0), "reused": t.get("cache_n", r.get("tokens_cached", 0)), "ttft_s": round(t.get("prompt_ms", 0) / 1000, 1),
            "decode_tps": round(t.get("predicted_per_second", 0), 2), "wall_s": round(time.time() - t0, 1), "text": r.get("content", "")[:200]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default=os.environ.get("KEY", ""))
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--tokens", type=int, default=16000)
    ap.add_argument("--questions", type=int, default=6)
    ap.add_argument("--answer-tokens", type=int, default=64)
    ap.add_argument("--sections", type=int, default=1, help="warm the context in N growing prefixes first: llama-server keeps one checkpoint per request (at its end minus -ub), so an edit "
                    "later reuses everything before the nearest checkpoint instead of nothing")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    parts = build_context(a.repo, a.tokens)
    base = "You are reading a code base. Answer questions about it.\n" + "".join(parts) + "\n\n=== end of files ===\n"
    print(f"context: {len(parts)} files, {len(base)} characters (about {len(base) / 2.25:.0f} tokens)")
    rows = []
    head = "You are reading a code base. Answer questions about it.\n"
    for k in range(1, a.sections):
        idx = max(1, len(parts) * k // a.sections)
        r = ask(a.url, a.key, head + "".join(parts[:idx]), 1)
        print(f"warm-up {k}/{a.sections - 1}: {r['processed']} processed, {r['reused']} reused, {r['wall_s']} s")
    print(f"{'#':>2} {'what':<22} {'processed':>10} {'reused':>8} {'TTFT s':>8} {'tok/s':>7} {'wall s':>8}")
    for i in range(a.questions):
        q = QUESTIONS[i % len(QUESTIONS)]
        r = ask(a.url, a.key, base + "Question: " + q + "\nAnswer:", a.answer_tokens)
        r["what"] = "same context" if i else "first (cold)"
        rows.append(r)
        print(f"{i + 1:>2} {r['what']:<22} {r['processed']:>10} {r['reused']:>8} {r['ttft_s']:>8} {r['decode_tps']:>7} {r['wall_s']:>8}")
    # an edit in the middle of the context: everything before it can be reused, everything after has to be processed again
    mid = len(parts) // 2
    edited = parts[:mid] + [parts[mid] + "\n# NOTE: this file was edited by the user a moment ago.\n"] + parts[mid + 1:]
    base2 = "You are reading a code base. Answer questions about it.\n" + "".join(edited) + "\n\n=== end of files ===\n"
    r = ask(a.url, a.key, base2 + "Question: " + QUESTIONS[0] + "\nAnswer:", a.answer_tokens)
    r["what"] = "after an edit (middle)"
    rows.append(r)
    print(f"{len(rows):>2} {r['what']:<22} {r['processed']:>10} {r['reused']:>8} {r['ttft_s']:>8} {r['decode_tps']:>7} {r['wall_s']:>8}")
    if a.out:
        json.dump({"context_tokens_est": int(len(base) / 2.25), "requests": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
