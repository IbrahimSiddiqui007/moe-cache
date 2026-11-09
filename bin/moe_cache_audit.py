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


def tokenize(url, text, key=None, special=False, parse_special=None):
    """[{id, piece}] for `text`; special=True for the prompt (BOS and the template's special tokens), False for the answer. parse_special=True on an answer that carries the
    template's own tags (the thinking part: '</think>' is one special token, and spelled out as text it would never match the model's token)."""
    r = _post(url, "/tokenize", {"content": text, "add_special": special, "parse_special": special if parse_special is None else parse_special, "with_pieces": True}, key, 120)
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


MAX_TOKENS = 16384        # R15-01: was 4096 (a long answer was cut and the page said "truncated"); one position is one cached /completion call, so the cost is the answer's own length


def score(url, prompt_text, answer_text, key=None, top=20, max_tokens=MAX_TOKENS, progress=None, cancel=None, parse_special=None):
    """Teacher-force `answer_text` after `prompt_text` on the exact-mode server at `url`.
    Returns {n, truncated, prompt_tokens, same, same_pct, mean_logloss, unknown_p, tokens: [...], mismatches: [positions], seconds}.
    tokens[i] = {i, id, piece, exact_id, exact_piece, exact_p, p, p_below, same}; p is the exact probability of the answer's token when it is among the top N
    (else None and p_below = the N-th probability: the token is less likely than that). progress(done, total) is called after each position; cancel() -> True stops."""
    t0 = time.time()
    p_ids = [t["id"] for t in tokenize(url, prompt_text, key, special=True)]
    a = tokenize(url, answer_text, key, special=False, parse_special=parse_special)
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


# ---- R15-01: what the audit may be given besides plain text ----

def image_size(data_url):
    """(width, height) of a base64 image data URL (png, gif, jpeg, webp) from its header bytes, or None. No image library."""
    import base64
    try:
        head = data_url.split(",", 1)[1][:87384]                    # the first 64 KiB of the file hold every header, a JPEG's start-of-frame included, unless a huge EXIF block comes first
        raw = base64.b64decode(head[:len(head) // 4 * 4], validate=False)
    except Exception:
        return None
    try:
        if raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR":
            return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
        if raw[:6] in (b"GIF87a", b"GIF89a"):
            return int.from_bytes(raw[6:8], "little"), int.from_bytes(raw[8:10], "little")
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            kind = raw[12:16]
            if kind == b"VP8X":
                return 1 + int.from_bytes(raw[24:27], "little"), 1 + int.from_bytes(raw[27:30], "little")
            if kind == b"VP8 ":
                return int.from_bytes(raw[26:28], "little") & 0x3FFF, int.from_bytes(raw[28:30], "little") & 0x3FFF
            if kind == b"VP8L":
                b = raw[21:25]
                bits = int.from_bytes(b, "little")
                return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if raw[:2] == b"\xff\xd8":                                # JPEG: walk the segments to the first start-of-frame
            i = 2
            while i + 9 < len(raw):
                if raw[i] != 0xFF:
                    i += 1
                    continue
                m = raw[i + 1]
                if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
                    i += 2
                    continue
                n = int.from_bytes(raw[i + 2:i + 4], "big")
                if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                    return int.from_bytes(raw[i + 7:i + 9], "big"), int.from_bytes(raw[i + 5:i + 7], "big")
                i += 2 + n
    except Exception:
        return None
    return None


def with_image_placeholders(messages):
    """(messages without the 'images' key, number of images shown as placeholders): every image of a message becomes '[image WxH]' in its text ('[image]' when the size cannot be
    read). The audit then re-scores the answer given those placeholders, not given the pictures: the exact server is text-only here (said in the result)."""
    out, n = [], 0
    for m in messages:
        imgs = m.get("images") if isinstance(m, dict) else None
        m2 = {k: v for k, v in m.items() if k != "images"}
        if imgs:
            marks = []
            for u in imgs:
                wh = image_size(u) if isinstance(u, str) else None
                marks.append(f"[image {wh[0]}x{wh[1]}]" if wh else "[image]")
            n += len(marks)
            m2["content"] = ((m2.get("content") or "") + ("\n" if m2.get("content") else "") + " ".join(marks))
        out.append(m2)
    return out, n


def answer_with_reasoning(url, messages, prompt, content, reasoning, key=None):
    """The text to teacher-force when the answer has a thinking part (`reasoning_content`): what the model's own chat template writes for an assistant turn with the thinking part
    and the answer, taken from /apply-template (so '<think>' / '</think>' and the newlines are the template's, not ours). Returns (text, how). Falls back to
    reasoning + '</think>' + answer when the template does not render the thinking part."""
    if not reasoning:
        return content, "answer only"
    try:
        full = render_prompt(url, list(messages) + [{"role": "assistant", "content": content, "reasoning_content": reasoning}], key)
        if full.startswith(prompt) and reasoning.strip()[:40] in full[len(prompt):]:
            tail = full[len(prompt):]
            end = tail.rfind(content)
            if end >= 0:
                return tail[:end + len(content)], "chat template"
    except Exception:
        pass
    return f"{reasoning.rstrip()}\n</think>\n\n{content}", "reasoning + </think> + answer (the chat template did not render the thinking part)"


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
