#!/usr/bin/env python3
"""S01 (audit idea): where does an agent's prompt stop being byte-identical to its previous request?

llama-server reuses the cached tokens only up to the first byte that differs from the last request of a slot; one changing line early in the prompt
(a date, a request id, a reordered tool list) makes it read the whole prompt again. Give this the request bodies of consecutive turns (JSON files with
a "messages" list, as sent to /v1/chat/completions, or a JSON-lines file with one body per line); it prints, per pair, the length of the identical
beginning, which message and role it ends in, a short excerpt around the first difference, and a hint when the changed text looks like a timestamp or id.

  python3 tools/prefix_check.py turn1.json turn2.json [turn3.json ...]
  python3 tools/prefix_check.py requests.jsonl
Reports only; it never changes a request and does not tell you the token count (use the server's "cache_n" / "prompt_n" for that).
"""
import json, re, sys


def bodies(paths):
    out = []
    for p in paths:
        txt = open(p, encoding="utf-8").read()
        try:
            out.append(json.loads(txt))
        except ValueError:
            out.extend(json.loads(line) for line in txt.splitlines() if line.strip())
    return out


def flat(body):
    """The prompt as text: the tools (if any), then each message as 'role: content' (content lists are joined by their text parts)."""
    parts = []
    if body.get("tools"):
        parts.append(("tools", json.dumps(body["tools"], sort_keys=False, ensure_ascii=False)))
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
        parts.append((m.get("role", "?"), c or ""))
    return parts


HINT = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}|\b\d{10,13}\b")


def compare(a, b):
    pa, pb = flat(a), flat(b)
    same = 0
    for i, (x, y) in enumerate(zip(pa, pb)):
        if x == y:
            same += len(x[0]) + len(x[1]) + 3
            continue
        n = next((k for k in range(min(len(x[1]), len(y[1]))) if x[1][k] != y[1][k]), min(len(x[1]), len(y[1])))
        if x[0] != y[0]:
            return same, i, x[0], "role differs", ""
        lo = max(0, n - 30)
        return same + n, i, x[0], f"...{x[1][lo:n + 30]!r} -> ...{y[1][lo:n + 30]!r}", ("looks like a timestamp or id: keep it out of the early prompt" if HINT.search(x[1][lo:n + 30] + y[1][lo:n + 30]) else "")
    return same, min(len(pa), len(pb)), "-", "one prompt is the start of the other (the normal case of a growing chat)", ""


if __name__ == "__main__":
    bs = bodies(sys.argv[1:])
    if len(bs) < 2:
        raise SystemExit(__doc__)
    for i in range(1, len(bs)):
        same, idx, role, what, hint = compare(bs[i - 1], bs[i])
        total = sum(len(r) + len(c) + 3 for r, c in flat(bs[i]))
        print(f"request {i} vs {i - 1}: identical start {same} of {total} characters ({100 * same / max(1, total):.0f} %), ends in message {idx} ({role}): {what}" + (f"\n    {hint}" if hint else ""))
