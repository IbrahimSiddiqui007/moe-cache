#!/usr/bin/env python3
"""loop_check.py: does the model loop? 20 long-answer prompts against a running llama-server, counting repeated word n-grams (I45). Standard library only.

Low-bit quants (IQ2_*) tend to fall into repetition loops on long answers; sampling settings (the model card's temperature / top-p / top-k / min-p and a
presence penalty) are the usual cure. This sends the prompts WITHOUT any sampling fields by default, so the server's own defaults are what is measured:
run it once against `moe_cache.py server MODEL` and once against `moe_cache.py server MODEL --no-sampling-defaults`, same seed.

  python3 tools/loop_check.py --url http://127.0.0.1:8080 [--n 20] [--max-tokens 1500] [--json out.json] [--label NAME]

Per answer: words, the share of 8-word sequences that already appeared earlier in the same answer (repeat share), the most repeated 8-word sequence,
and LOOP when the answer ends in a repeat (the last 200 words are more than half repeats) or hit max_tokens while doing so. Summary: loops / answers,
median and max repeat share. Thinking text (reasoning_content) is counted with the answer: loops happen there too.
"""
import argparse, http.client, json, os, statistics, sys, time, urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_bench import _key_for  # noqa: E402  (moe-cache's stored key when the server asks for one)

PROMPTS = [
    "Explain in detail how a hash table works, including collision handling, resizing and the cost of each operation.",
    "Write a long, step-by-step guide to setting up a home vegetable garden from scratch.",
    "Describe the causes, main events and consequences of the French Revolution.",
    "Write a Python module that implements a priority queue with decrease-key, with docstrings and tests, and explain the design.",
    "Explain how TCP congestion control works: slow start, congestion avoidance, fast retransmit and fast recovery.",
    "Write a short story (at least 800 words) about a lighthouse keeper who finds a message in a bottle.",
    "Compare three sorting algorithms (quicksort, mergesort, heapsort): how each works, best/worst cases, memory, stability.",
    "Explain photosynthesis from light absorption to sugar production, including both stages and the role of each molecule.",
    "Plan a two-week trip through Japan for a first-time visitor, day by day, with transport and budget notes.",
    "Explain how a CPU executes an instruction: fetch, decode, execute, pipelining, branch prediction and caches.",
    "Write a detailed review of the trade-offs between microservices and a monolith for a 20-person startup.",
    "Explain the theory of evolution by natural selection with examples, common misconceptions and the evidence for it.",
    "Write a bash script that backs up a directory with rotation and logging, then explain every line.",
    "Describe how vaccines train the immune system, the main vaccine types and how each is developed and tested.",
    "Explain the differences between SQL isolation levels with an example anomaly for each.",
    "Write a long essay on the history of the printing press and its effects on European society.",
    "Explain how public-key cryptography works, with RSA and Diffie-Hellman worked through on small numbers.",
    "Give a detailed beginner's guide to running a first marathon: a 16-week plan, nutrition and injury prevention.",
    "Explain how a compiler turns source code into machine code: lexing, parsing, semantic analysis, optimisation, code generation.",
    "Describe the water cycle in detail and how climate change affects each part of it.",
]
N = 8


def repeat_stats(text):
    w = text.split()
    grams = [tuple(w[i:i + N]) for i in range(max(0, len(w) - N + 1))]
    seen, rep, counts = set(), [], {}
    for g in grams:
        rep.append(g in seen); seen.add(g); counts[g] = counts.get(g, 0) + 1
    top = max(counts.items(), key=lambda kv: kv[1]) if counts else ((), 0)
    tail = rep[-200:]
    return {"words": len(w), "repeat_share": round(sum(rep) / len(rep), 3) if rep else 0.0,
            "tail_repeat_share": round(sum(tail) / len(tail), 3) if tail else 0.0, "top_count": top[1], "top": " ".join(top[0])}


def ask(url, key, prompt, a):
    u = urllib.parse.urlparse(url)
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens, "seed": a.seed, "stream": False}
    if a.temperature is not None:
        body["temperature"] = a.temperature
    hdr = {"Content-Type": "application/json"}
    if key:
        hdr["Authorization"] = f"Bearer {key}"
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=a.timeout)
    c.request("POST", "/v1/chat/completions", json.dumps(body), hdr)
    r = c.getresponse(); data = r.read(); c.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {data[:200]!r}")
    j = json.loads(data)
    ch = j["choices"][0]; m = ch.get("message") or {}
    return (m.get("reasoning_content") or "") + "\n" + (m.get("content") or ""), ch.get("finish_reason"), (j.get("timings") or {})


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8080"); p.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY", ""))
    p.add_argument("--n", type=int, default=20); p.add_argument("--max-tokens", type=int, default=1500); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temperature", type=float, help="send this temperature (default: send none, the server's default applies)")
    p.add_argument("--json", default=""); p.add_argument("--label", default=""); p.add_argument("--timeout", type=float, default=1800)
    a = p.parse_args()
    key = a.api_key or _key_for(a.url)
    rows = []
    for i, pr in enumerate(PROMPTS[:a.n]):
        t0 = time.time()
        try:
            text, fin, tm = ask(a.url, key, pr, a)
            s = repeat_stats(text); s.update(finish=fin, s=round(time.time() - t0, 1), tok=tm.get("predicted_n"))
            s["loop"] = s["tail_repeat_share"] > 0.5
        except Exception as e:
            s = {"error": str(e)[:200], "loop": False}
        rows.append(s)
        if "error" in s:
            print(f"{i + 1:2d}  ERROR {s['error']}", flush=True)
        else:
            print(f"{i + 1:2d}  {s['words']:5d} words  repeat {s['repeat_share']:.3f}  tail {s['tail_repeat_share']:.3f}  {s['finish']:6s}  {s['s']:6.1f} s"
                  + ("  LOOP" if s["loop"] else "") + (f"  x{s['top_count']} '{s['top'][:60]}'" if s["top_count"] > 2 else ""), flush=True)
    ok = [r for r in rows if "error" not in r]
    summ = {"label": a.label, "answers": len(ok), "errors": len(rows) - len(ok), "loops": sum(r["loop"] for r in ok),
            "repeat_share_median": round(statistics.median([r["repeat_share"] for r in ok]), 3) if ok else None,
            "repeat_share_max": max((r["repeat_share"] for r in ok), default=None),
            "hit_max_tokens": sum(r.get("finish") == "length" for r in ok)}
    print(f"{a.label or 'result'}: {summ['loops']} loops in {summ['answers']} answers ({summ['errors']} errors); repeat share median "
          f"{summ['repeat_share_median']}, max {summ['repeat_share_max']}; {summ['hit_max_tokens']} hit max_tokens")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"summary": summ, "rows": rows}, f, indent=1)
    return 1 if summ["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
