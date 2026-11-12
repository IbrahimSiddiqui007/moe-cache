#!/usr/bin/env python3
"""MODELS-1141: runs tools/hard_tasks.json against a llama-server and scores it: pass (objective check), thinking tokens, total tokens, loops, prompts that hit the cap.
Standard library only (+ tools/quality_yardstick.py for the request helpers and the loop test).
  hard_tasks_run.py run     --url http://127.0.0.1:8080 --out OUT.json [--label L] [--tasks tools/hard_tasks.json] [--cap 0] [--workers 1] [--only ID,ID] [--no-think]
  hard_tasks_run.py summary OUT.json [OUT2.json ...]         one table per run (pass, thinking tokens median/mean/p90/max, loops, cap hits, by category)
  hard_tasks_run.py compare A.json B.json                    per task: pass and thinking tokens of both runs, and the totals
--cap 0 = no answer cap (the server's context is the only limit; a prompt that fills it stops with stop=limit and counts as a cap hit).
Request: the same as quality_yardstick.py (apply-template + /completion, temperature 0, top_k 1, seed 1, no prompt cache, return_tokens); the chat template's own thinking default is used (no
--no-think), so a thinking model thinks. Use llama-server -np N with --workers N to run N tasks at once (the answers do not change; the wall time does)."""
import argparse, json, os, re, statistics, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import quality_yardstick as Y      # noqa: E402


def final_answer(text):
    """The text after the last </think>, and in it the last `ANSWER: ...` line (None when there is none)."""
    body = text.split("</think>")[-1]
    m = re.findall(r"ANSWER:\s*(.+)", body)
    return m[-1].strip().strip("`*. ") if m else None


def check(task, answer):
    """True when `answer` is right for the task's objective check."""
    if answer is None:
        return False
    c = task["check"]; a = answer.strip().strip("`*. ")
    if c["kind"] == "int":
        return re.sub(r"[,\s_]", "", a) == c["answer"]
    if c["kind"] == "word":
        return re.sub(r"[^a-z]", "", a.lower()) == c["answer"]
    if c["kind"] == "exact":
        norm = lambda s: re.sub(r"[\s]", "", s).lower()
        return norm(a) == norm(c["answer"])
    raise ValueError(c["kind"])


def think_tokens(url, text, n_tokens, stop):
    """Number of tokens before </think> (the template opens the think block in the prompt); all of them when the answer never got out of the thinking."""
    if "</think>" not in text:
        return n_tokens
    head = text.split("</think>")[0]
    try:
        return len(Y.post(url, "/tokenize", {"content": head, "add_special": False}, 600)["tokens"])
    except Exception:
        return int(n_tokens * len(head) / max(1, len(text)))


def cmd_run(a):
    tasks = json.load(open(a.tasks, encoding="utf-8"))
    if a.only:
        keep = set(a.only.split(",")); tasks = [t for t in tasks if t["id"] in keep]
    props = Y.get(a.url, "/props")
    out = {"label": a.label or "", "url": a.url, "cap": a.cap, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "model": props.get("model_path", ""), "build": props.get("build_info", ""), "items": [None] * len(tasks)}
    lock = threading.Lock(); done = [0]; t0 = time.time()

    def work(i):
        t = tasks[i]; body = {"messages": [{"role": "user", "content": t["prompt"]}]}
        if a.no_think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        prompt = Y.post(a.url, "/apply-template", body)["prompt"]
        s = time.time()
        r = Y.post(a.url, "/completion", {"prompt": prompt, "n_predict": a.cap if a.cap > 0 else -1, "temperature": 0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                                           "presence_penalty": 0.0, "seed": 1, "cache_prompt": False, "return_tokens": True, "stream": False}, timeout=86400)
        text = r.get("content", ""); toks = r.get("tokens", []); stop = r.get("stop_type", "")
        ans = final_answer(text)
        rec = {"id": t["id"], "category": t["category"], "n_tokens": len(toks), "stop": stop, "think_tokens": think_tokens(a.url, text, len(toks), stop), "loop": Y.has_loop(toks),
               "answer": ans, "pass": check(t, ans), "cap_hit": stop == "limit", "seconds": round(time.time() - s), "text_tail": text[-300:]}
        with lock:
            out["items"][i] = rec; done[0] += 1
            print(f"[{done[0]}/{len(tasks)}] {t['id']:15s} {'PASS' if rec['pass'] else 'fail'} think {rec['think_tokens']:6d} total {rec['n_tokens']:6d} stop={stop} loop={rec['loop']} {rec['seconds']}s", flush=True)
            json.dump(out, open(a.out + ".partial", "w"), ensure_ascii=False)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, range(len(tasks))))
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(summary_text(out))


def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else 0


def summary_text(run):
    it = [x for x in run["items"] if x]; th = [x["think_tokens"] for x in it]; tot = [x["n_tokens"] for x in it]
    lines = [f"run {run.get('label') or run.get('model','')}: {sum(x['pass'] for x in it)}/{len(it)} pass, loops {sum(x['loop'] for x in it)}, cap hits {sum(x['cap_hit'] for x in it)} (cap {run.get('cap') or 'none'}), {run.get('seconds', 0)} s",
             f"  thinking tokens: median {int(statistics.median(th))} mean {int(statistics.mean(th))} p90 {pct(th, .9)} max {max(th)}; total tokens: median {int(statistics.median(tot))} max {max(tot)}; tasks over 8k thinking: {sum(1 for v in th if v > 8000)}"]
    cats = {}
    for x in it:
        cats.setdefault(x["category"], []).append(x)
    for c, xs in sorted(cats.items()):
        lines.append(f"  {c:20s} pass {sum(y['pass'] for y in xs)}/{len(xs)}  think median {int(statistics.median(y['think_tokens'] for y in xs))}  loops {sum(y['loop'] for y in xs)}")
    return "\n".join(lines)


def cmd_summary(a):
    for f in a.files:
        print(summary_text(json.load(open(f))))


def cmd_compare(a):
    A, B = (json.load(open(f)) for f in (a.a, a.b)); ia = {x["id"]: x for x in A["items"] if x}; ib = {x["id"]: x for x in B["items"] if x}
    print(f"{'task':16s} | {'A pass':6s} {'A think':>8s} {'A loop':6s} | {'B pass':6s} {'B think':>8s} {'B loop':6s}")
    for k in ia:
        if k in ib:
            x, y = ia[k], ib[k]
            print(f"{k:16s} | {'PASS' if x['pass'] else 'fail':6s} {x['think_tokens']:8d} {str(x['loop']):6s} | {'PASS' if y['pass'] else 'fail':6s} {y['think_tokens']:8d} {str(y['loop']):6s}")
    print("A:", summary_text(A).split("\n")[0]); print("B:", summary_text(B).split("\n")[0])
    print("A:", summary_text(A).split("\n")[1].strip()); print("B:", summary_text(B).split("\n")[1].strip())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--url", required=True); r.add_argument("--out", required=True); r.add_argument("--label", default=""); r.add_argument("--tasks", default=os.path.join(HERE, "hard_tasks.json"))
    r.add_argument("--cap", type=int, default=0); r.add_argument("--workers", type=int, default=1); r.add_argument("--only", default=""); r.add_argument("--no-think", action="store_true")
    s = sp.add_parser("summary"); s.add_argument("files", nargs="+"); c = sp.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    a = ap.parse_args(); {"run": cmd_run, "summary": cmd_summary, "compare": cmd_compare}[a.cmd](a)
