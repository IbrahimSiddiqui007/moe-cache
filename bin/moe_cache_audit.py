"""moe-cache audit (GUI-043): re-score an answer in exact mode, position by position (teacher forcing). Python 3.8+, standard library only.

An answer written with an approximate mode (q8_0 KV cache, MTP drafting, --fast, --fast=fewer, ...) is a sequence of tokens. The audit feeds the prompt and the answer
so far to a llama-server that runs in EXACT mode and asks for the next token with its top probabilities (/completion with the answer's own token ids, n_predict 1,
n_probs, temperature 0: the chosen token is the exact top-1). Where the exact top-1 is not the token the answer has, the answer could have gone another way.
The server keeps the prompt's KV cache between calls, so a position costs one decode step: the audit takes about as long as writing the answer did.

What it reports per position: the answer's token, the exact top-1 token with its probability, and the exact probability of the answer's token (when it is among the
top N asked for; otherwise "below p"). Summary: same-top-token share (the quality gate's own measure, docs/QUALITY_GATE.md) and the mean log-loss over the positions
whose probability is known. A KL needs the approximate run's own distribution, which a server does not keep; this is not a KL.

score(url, prompt_text, answer_text) works with any llama-server; see the functions' docstrings. The orchestration (starting an exact server beside the running one,
or pausing it) is in bin/moe_cache_ui.py (AuditJob).
"""
import json, math, time, urllib.error, urllib.request


class AuditError(Exception):
    pass


def _post(url, path, body, key=None, timeout=600):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode("utf-8"), headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raise AuditError(f"{path}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}")
    except (OSError, ValueError) as e:
        raise AuditError(f"{path}: {e}")


def render_prompt(url, messages, key=None):
    """The prompt text the server's chat template makes of `messages` (POST /apply-template)."""
    r = _post(url, "/apply-template", {"messages": messages}, key, 60)
    if not isinstance(r.get("prompt"), str):
        raise AuditError("/apply-template gave no prompt")
    return r["prompt"]


def tokenize(url, text, key=None, special=False):
    """[{id, piece}] for `text`; special=True for the prompt (BOS and the template's special tokens), False for the answer."""
    r = _post(url, "/tokenize", {"content": text, "add_special": special, "parse_special": special, "with_pieces": True}, key, 120)
    toks = r.get("tokens")
    if not isinstance(toks, list):
        raise AuditError("/tokenize gave no tokens")
    out = []
    for t in toks:
        if isinstance(t, dict) and "id" in t:
            out.append({"id": int(t["id"]), "piece": t.get("piece", "")})
        elif isinstance(t, int):                    # a server without with_pieces: ids only
            out.append({"id": t, "piece": ""})
    return out


def score(url, prompt_text, answer_text, key=None, top=20, max_tokens=4096, progress=None, cancel=None):
    """Teacher-force `answer_text` after `prompt_text` on the exact-mode server at `url`.
    Returns {n, truncated, prompt_tokens, same, same_pct, mean_logloss, unknown_p, tokens: [...], mismatches: [positions], seconds}.
    tokens[i] = {i, id, piece, exact_id, exact_piece, exact_p, p, p_below, same}; p is the exact probability of the answer's token when it is among the top N
    (else None and p_below = the N-th probability: the token is less likely than that). progress(done, total) is called after each position; cancel() -> True stops."""
    t0 = time.time()
    p_ids = [t["id"] for t in tokenize(url, prompt_text, key, special=True)]
    a = tokenize(url, answer_text, key, special=False)
    if not a:
        raise AuditError("the answer has no tokens")
    truncated = len(a) > max_tokens
    a = a[:max_tokens]
    a_ids = [t["id"] for t in a]
    rows, same, ll_sum, ll_n, unknown = [], 0, 0.0, 0, 0
    for k, tok in enumerate(a):
        if cancel and cancel():
            raise AuditError("cancelled")
        r = _post(url, "/completion", {"prompt": p_ids + a_ids[:k], "n_predict": 1, "temperature": 0, "n_probs": top, "cache_prompt": True}, key, 600)
        cps = r.get("completion_probabilities")
        if not isinstance(cps, list) or not cps:
            raise AuditError("the server gave no probabilities (/completion n_probs): is it a llama-server of this decade?")
        cp = cps[0]
        tops = cp.get("top_logprobs") or []
        exact_id = int(cp["id"])
        lp = next((t["logprob"] for t in tops if t.get("id") == tok["id"]), None)
        if lp is None and exact_id == tok["id"]:
            lp = cp.get("logprob")
        row = {"i": k, "id": tok["id"], "piece": tok["piece"], "exact_id": exact_id, "exact_piece": cp.get("token", ""), "exact_p": math.exp(cp["logprob"]) if cp.get("logprob") is not None else None,
               "p": math.exp(lp) if lp is not None else None, "p_below": None, "same": exact_id == tok["id"]}
        if lp is None:
            unknown += 1
            if tops:
                row["p_below"] = math.exp(min(t["logprob"] for t in tops))
        else:
            ll_sum += -lp; ll_n += 1
        same += row["same"]
        rows.append(row)
        if progress:
            progress(k + 1, len(a))
    n = len(rows)
    return {"n": n, "truncated": truncated, "prompt_tokens": len(p_ids), "same": same, "same_pct": 100.0 * same / n, "mean_logloss": (ll_sum / ll_n) if ll_n else None,
            "unknown_p": unknown, "tokens": rows, "mismatches": [r["i"] for r in rows if not r["same"]], "seconds": round(time.time() - t0, 1), "top": top}


def char_ranges(tokens, answer_text):
    """[[start, end, token index], ...] character ranges (in `answer_text`) of the tokens whose exact top-1 differs from the answer's token. The pieces are bytes of
    the UTF-8 text: they are laid end to end and cut at character boundaries; when the pieces do not add up to the text the ranges are not given (empty)."""
    raw = answer_text.encode("utf-8")
    pieces = [t["piece"].encode("utf-8") if isinstance(t["piece"], str) else bytes(t["piece"]) for t in tokens]
    if b"".join(pieces) != raw[:sum(len(p) for p in pieces)] or any(not p for p in pieces):
        return []
    # byte offset -> character offset
    off, cidx = {}, 0
    for i, ch in enumerate(answer_text):
        off[cidx] = i
        cidx += len(ch.encode("utf-8"))
    off[cidx] = len(answer_text)
    out, pos = [], 0
    for t, p in zip(tokens, pieces):
        a, b = pos, pos + len(p)
        pos = b
        if not t["same"] and a in off and b in off:
            out.append([off[a], off[b], t["i"]])
        elif not t["same"]:                         # a token that cuts a multi-byte character: widen to the characters it touches
            lo = max((k for k in off if k <= a), default=0)
            hi = min((k for k in off if k >= b), default=cidx)
            out.append([off[lo], off[hi], t["i"]])
    return out
