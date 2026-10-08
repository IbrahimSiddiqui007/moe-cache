#!/usr/bin/env python3
"""Ask a running llama-server for 4 fixed prompts at temperature 0 / seed 42 and save the token ids. usage: ident.py OUT.json [PORT] [N_TOKENS]"""
import json, os, sys, urllib.request, pathlib
out, port, n = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "8099"), int(sys.argv[3]) if len(sys.argv) > 3 else 100
P = ["Write a Python function that merges overlapping intervals and explain its time complexity.",
     "Explain how a hash table handles collisions, with examples.",
     "Write a SQL query that finds the top 3 customers by revenue per month, and explain which indexes would help.",
     "Explain the difference between processes and threads in Linux and show a small C example of each."]
if os.environ.get("IDENT_LONG"):
    # long prompts (many hundreds of tokens): these take the batched path that llama.cpp may hand to the GPU
    txt = (pathlib.Path(__file__).resolve().parent.parent / "README.md").read_text().splitlines(True)
    P = ["Summarize this text in two sentences:\n\n" + "".join(txt[a:b]) for a, b in [(0, 45), (40, 85)]]
res = []
for p in P:
    body = {"prompt": p, "n_predict": n, "temperature": 0, "seed": 42, "cache_prompt": False, "return_tokens": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    res.append(json.load(urllib.request.urlopen(req, timeout=3000))["tokens"])
json.dump(res, open(out, "w"))
