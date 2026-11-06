#!/usr/bin/env python3
"""conversation_bench.py (TEST-699): a REAL multi-turn conversation against a running llama-server, filled up to a context limit such as 65,536 tokens.

Single prompts and 8k contexts do not show what a user sees in a long coding or agent session: there the model reads a growing history, turn after turn, and
most of the history is reused from the server's prompt cache while every new paste still has to be read. This tool plays such a session: a fixed,
seeded script of user turns (small follow-ups, code pastes, tool-output-like logs, a few big pastes), each sent with the whole history so far
(cache_prompt on, temperature 0, a fixed max_tokens), the model's answer appended, until the next turn would not fit into the context.

  run       one conversation against one server       python3 tools/conversation_bench.py run --url http://127.0.0.1:8080 --arm moe-cache --ctx 65536 --out run1.json
  ab        stock and plugin servers that are both up   ... ab --stock-url U1 --plugin-url U2 --ctx 65536 --out-dir DIR      (order stock, plugin, plugin, stock)
  assemble  run files (in the order they were run) -> ONE bench record (docs/bench_record.schema.json) and the readable tables
  table     print the tables of a run file or a record

Per turn: tokens added, depth before and after, prompt and decode tok/s (the server's own timings), tokens reused from the cache, time to first token, and
the plugin's hits_dec / misses_dec of that turn when --stats-pid and --stats-log are given (the plugin must run with MOE_CACHE_STATS_SIGNAL=1; its
stderr goes to the log). Per depth band (a quarter of the context each): the same, summed over the turns that start in that band.
A value the server did not give is null in the run file and "not recorded" in the record; nothing is guessed. Standard library only. Docs: docs/CONVERSATION_BENCH.md."""
import argparse, datetime, hashlib, http.client, json, os, random, re, signal, statistics, sys, time, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
sys.path.insert(0, HERE)

CHARS_PER_TOKEN = 3.0       # deliberately low: a turn is judged too big a little early, never too late
MARGIN = 48                 # tokens of slack for the chat template around a turn
MIN_PROMPT_N = 64           # a turn that read fewer new tokens does not enter the prompt-speed mean (timer noise)
MIN_PREDICTED_N = 16        # the same for decode speed
SYSTEM = "You are a careful coding assistant working in the user's repository. Answer briefly."

# ---------------------------------------------------------------- the fixed script

_WORDS = ["cache", "expert", "layer", "router", "buffer", "token", "slot", "queue", "worker", "index", "block", "shard", "frame", "probe", "ledger", "window"]
_VERBS = ["load", "merge", "split", "flush", "scan", "pack", "route", "score", "trim", "fetch", "match", "write", "parse", "check", "apply"]
_FOLLOW = ["Why does {a} return early here?", "Can you rename {a} to {b} and keep the behaviour?", "What breaks if {a} is called twice?",
           "Is there an off-by-one in {a}?", "Add a short docstring to {a}.", "Could {a} be made faster without changing the result?",
           "Explain the difference between {a} and {b} in two sentences.", "Write a test for {a}.", "Does {a} need a lock?", "Thanks. Now look at {b}."]


def _ident(rng):
    return rng.choice(_VERBS) + "_" + rng.choice(_WORDS) + (("_" + rng.choice(_WORDS)) if rng.random() < 0.4 else "")


def make_code(rng, lines):
    out, name = [], _ident(rng)
    out.append(f"def {name}(state, {rng.choice(_WORDS)}, limit={rng.randint(2, 512)}):")
    for _ in range(max(1, lines - 2)):
        a, b = rng.choice(_WORDS), rng.choice(_WORDS)
        t = rng.random()
        if t < 0.30:
            out.append(f"    {a}_{rng.randint(0, 99)} = state.{_ident(rng)}({b}, {rng.randint(0, 4096)})")
        elif t < 0.55:
            out.append(f"    if {a} > {rng.randint(1, 9999)} and not state.{b}:")
            out.append(f"        raise ValueError(\"{a} {b} out of range at step {rng.randint(0, 999)}\")")
        elif t < 0.80:
            out.append(f"    for i in range({rng.randint(2, 64)}):")
            out.append(f"        {a}[i] = ({a}[i] * {rng.randint(2, 31)} + {rng.randint(0, 255)}) % {rng.randint(97, 65521)}")
        else:
            out.append(f"    # {rng.choice(_VERBS)} the {a} before the {b} is {rng.choice(_VERBS)}ed")
    out.append(f"    return {rng.choice(_WORDS)}")
    return "\n".join(out)


def make_log(rng, lines):
    out = []
    for i in range(lines):
        lvl = rng.choice(["INFO", "INFO", "INFO", "WARN", "DEBUG", "ERROR"])
        out.append(f"2026-11-04 12:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}.{rng.randint(0, 999):03d} {lvl} {_ident(rng)}: "
                   f"{rng.choice(_WORDS)}={rng.randint(0, 99999)} {rng.choice(_WORDS)}={rng.random():.4f} took {rng.randint(1, 900)} ms")
    return "\n".join(out)


def make_turn(seed, index, attempt=0):
    """The user message number `index`: deterministic from (seed, index). attempt > 0 gives a small follow-up (used when the planned turn does not fit)."""
    rng = random.Random(seed * 1000003 + index * 101 + attempt * 7)
    r = 3.0 if attempt else rng.random()
    a, b = _ident(rng), _ident(rng)
    if r < 0.45 or attempt:
        kind, text = "followup", rng.choice(_FOLLOW).format(a=a, b=b)
        if attempt:
            text = text[:120]
    elif r < 0.75:
        kind = "code"
        text = f"Here is `{a}` from `{b}.py`. Please review it:\n\n```python\n{make_code(rng, rng.randint(30, 140))}\n```"
    elif r < 0.90:
        kind = "tool_output"
        text = f"Output of the last run of `{a}`:\n\n```\n{make_log(rng, rng.randint(15, 90))}\n```\nWhat stands out?"
    else:
        kind = "big_paste"
        text = f"I pasted three files, `{a}.py`, `{b}.py` and the log:\n\n```python\n{make_code(rng, rng.randint(150, 260))}\n```\n\n```python\n" \
               f"{make_code(rng, rng.randint(150, 260))}\n```\n\n```\n{make_log(rng, rng.randint(60, 150))}\n```\nSummarise the problem."
    return {"index": index, "kind": kind, "text": text}


def load_script(path):
    """--script FILE: a JSON list of user messages (strings) played in order instead of the seeded script."""
    data = json.load(open(path, encoding="utf-8"))
    return [{"index": i, "kind": "script", "text": t if isinstance(t, str) else t.get("text", "")} for i, t in enumerate(data)]


# ---------------------------------------------------------------- the server side

def _conn(url, timeout):
    u = urllib.parse.urlparse(url)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout), u.path.rstrip("/")


def server_ctx(url, api_key, timeout=10):
    """n_ctx from /props, or None."""
    try:
        c, base = _conn(url, timeout)
        c.request("GET", base + "/props", headers={"Authorization": "Bearer " + api_key} if api_key else {})
        r = c.getresponse(); body = r.read(); c.close()
        j = json.loads(body)
        return int((j.get("default_generation_settings") or {}).get("n_ctx") or j.get("n_ctx") or 0) or None
    except Exception:
        return None


def chat(url, body, api_key, timeout):
    """One streamed chat request -> dict: text, reasoning (chars), ttft_s, wall_s, timings, usage, status, error."""
    res = {"text": "", "reasoning_chars": 0, "ttft_s": None, "wall_s": None, "timings": None, "usage": None, "status": None, "error": None}
    c, base = _conn(url, timeout)
    hdr = {"Content-Type": "application/json"}
    if api_key:
        hdr["Authorization"] = "Bearer " + api_key
    t0 = time.perf_counter()
    try:
        c.request("POST", base + "/v1/chat/completions", body=json.dumps(body).encode(), headers=hdr)
        r = c.getresponse()
        res["status"] = r.status
        if r.status != 200:
            res["error"] = f"HTTP {r.status}: {r.read(400).decode('utf-8', 'replace')}"
        else:
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                d = line[5:].strip()
                if d == b"[DONE]":
                    continue
                try:
                    j = json.loads(d)
                except ValueError:
                    continue
                for ch in j.get("choices") or []:
                    dl = ch.get("delta") or {}
                    if dl.get("content"):
                        res["text"] += dl["content"]
                    if dl.get("reasoning_content"):
                        res["reasoning_chars"] += len(dl["reasoning_content"])
                    if res["ttft_s"] is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("tool_calls")):
                        res["ttft_s"] = time.perf_counter() - t0
                if j.get("timings"):
                    res["timings"] = j["timings"]
                if j.get("usage"):
                    res["usage"] = j["usage"]
    except Exception as ex:
        res["error"] = f"{type(ex).__name__}: {ex}"
    res["wall_s"] = time.perf_counter() - t0
    c.close()
    return res


class Stats:
    """The plugin's own counters (hits_dec / misses_dec ...). The plugin runs with MOE_CACHE_STATS_SIGNAL=1 and prints one `moe-cache-stats {json}` line on
    request: Linux SIGUSR1 to llama-server's pid, Windows the named event Local\\moe-cache-stats-<pid>; its stderr is the log file."""
    def __init__(self, pid, log):
        self.pid, self.log = int(pid), log

    def lines(self):
        try:
            with open(self.log, "rb") as f:
                f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 400000))
                data = f.read().decode("utf-8", "replace")
        except OSError:
            return []
        return [l for l in data.splitlines() if "moe-cache-stats " in l]

    def ask(self):
        try:
            if os.name == "nt":
                import ctypes
                k = ctypes.windll.kernel32
                h = k.OpenEventW(0x0002, False, f"Local\\moe-cache-stats-{self.pid}")
                if not h:
                    return False
                k.SetEvent(h); k.CloseHandle(h)
            else:
                os.kill(self.pid, signal.SIGUSR1)
            return True
        except Exception:
            return False

    def snapshot(self, wait=4.0):
        n = len(self.lines())
        if not self.ask():
            return None
        t0 = time.time()
        while time.time() - t0 < wait:
            ls = self.lines()
            if len(ls) > n:
                try:
                    return json.loads(ls[-1].split("moe-cache-stats ", 1)[1])
                except ValueError:
                    return None
            time.sleep(0.1)
        return None


def proc_read_bytes(pid):
    """Bytes the process read from storage so far (Linux /proc/<pid>/io), or None where the OS gives no such counter (Windows: the plugin's own read_bytes
    counter is used instead, and a stock server's disk reads are not recorded)."""
    try:
        with open(f"/proc/{int(pid)}/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def delta(s0, s1, key):
    if s0 is None or s1 is None or key not in s0 or key not in s1:
        return None
    return s1[key] - s0[key]


# ---------------------------------------------------------------- one conversation

def run_conversation(url, arm, ctx, max_tokens=128, seed=42, api_key="", model="conversation-bench", timeout=1800, stats=None, max_turns=None,
                     script=None, no_think=False, ignore_eos=False, log=print, monitor_pid=None):
    """Play the conversation; returns the run dict (turns, bands, stop reason). monitor_pid: llama-server's pid, sampled during the whole run for VRAM, process
    RAM, free RAM and CPU peaks (the same Sampler as bench-report, bin/moe_cache_record.py)."""
    smp = None
    if monitor_pid:
        try:
            import moe_cache_record as rr
            smp = rr.Sampler(int(monitor_pid)); smp.start()
        except Exception as ex:
            log(f"no resource sampling: {type(ex).__name__}: {ex}")
    io0 = proc_read_bytes(monitor_pid) if monitor_pid else None
    turns, history = [], [{"role": "system", "content": SYSTEM}]
    prev_total, n, stop = 0, 0, "max turns reached"
    t_start = time.time()
    while max_turns is None or n < max_turns:
        pick = None
        for attempt in range(6):
            cand = (script[n] if (script and attempt == 0 and n < len(script)) else None) if script else make_turn(seed, n, attempt)
            if script and cand is None:
                stop = "script ended"; break
            est = len(cand["text"]) / CHARS_PER_TOKEN + MARGIN
            if prev_total + est + max_tokens <= ctx:
                pick = cand; break
            if script:
                stop = "context full"; break
        if pick is None:
            if stop == "max turns reached":
                stop = "context full"
            break
        history.append({"role": "user", "content": pick["text"]})
        body = {"model": model, "messages": history, "max_tokens": max_tokens, "temperature": 0, "seed": 42, "stream": True,
                "stream_options": {"include_usage": True}, "cache_prompt": True}
        if no_think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if ignore_eos:
            body["ignore_eos"] = True
        s0 = stats.snapshot() if stats else None
        r = chat(url, body, api_key, timeout)
        s1 = stats.snapshot() if stats else None
        tm, us = r["timings"] or {}, r["usage"] or {}
        prompt_len = (tm["prompt_n"] + tm.get("cache_n", 0)) if "prompt_n" in tm else us.get("prompt_tokens")
        pred_n = tm.get("predicted_n", us.get("completion_tokens"))
        row = {"turn": n, "kind": pick["kind"], "user_chars": len(pick["text"]), "error": r["error"],
               "tokens_added": (prompt_len - prev_total) if prompt_len is not None else None,
               "depth_before": prompt_len, "depth_after": (prompt_len + pred_n) if prompt_len is not None and pred_n is not None else None,
               "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"), "prompt_ms": tm.get("prompt_ms"),
               "prompt_tps": round(tm["prompt_per_second"], 2) if tm.get("prompt_per_second") is not None else None,
               "predicted_n": pred_n, "predicted_ms": tm.get("predicted_ms"),
               "decode_tps": round(tm["predicted_per_second"], 2) if tm.get("predicted_per_second") is not None else None,
               "ttft_s": round(r["ttft_s"], 3) if r["ttft_s"] is not None else None, "wall_s": round(r["wall_s"], 2),
               "hits_dec": delta(s0, s1, "hits_dec"), "misses_dec": delta(s0, s1, "misses_dec"),
               "read_gb": (lambda d: round(d / 1e9, 3) if d is not None else None)(delta(s0, s1, "read_bytes")),
               "answer_chars": len(r["text"]), "answer_sha256": hashlib.sha256(r["text"].encode("utf-8")).hexdigest()[:16]}
        turns.append(row)
        log(f"  [{arm}] turn {n:3d} {pick['kind']:<11} +{row['tokens_added']} tok  depth {row['depth_before']}->{row['depth_after']}  "
            f"prompt {row['prompt_tps']} tok/s ({row['prompt_n']} new, {row['cache_n']} cached)  decode {row['decode_tps']} tok/s  ttft {row['ttft_s']} s"
            + (f"  hits_dec {row['hits_dec']}/{row['misses_dec']}" if row["hits_dec"] is not None else ""))
        if r["error"]:
            stop = "error: " + str(r["error"])[:200]
            history.pop()
            break
        history.append({"role": "assistant", "content": r["text"] if r["text"] else "(no visible answer)"})
        prev_total = row["depth_after"] if row["depth_after"] is not None else prev_total + int(est)
        n += 1
    run = {"arm": arm, "url": url, "ctx": ctx, "max_tokens": max_tokens, "seed": seed, "turns": turns, "stop_reason": stop,
           "final_depth": turns[-1]["depth_after"] if turns else None, "wall_s": round(time.time() - t_start, 1),
           "user_turns_sha256": hashlib.sha256("\n".join(m["content"] for m in history if m["role"] == "user").encode("utf-8")).hexdigest()[:16],
           "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")}
    run["bands"] = bands(turns, ctx)
    if smp is not None:
        run["monitor"] = smp.stop()
    rb = [t["read_gb"] for t in turns if t["read_gb"] is not None]
    io1 = proc_read_bytes(monitor_pid) if monitor_pid else None
    if rb:
        run["disk_read_gb"], run["disk_read_source"] = round(sum(rb), 2), "the plugin's own read_bytes counter, summed over the turns"
    elif io0 is not None and io1 is not None:
        run["disk_read_gb"], run["disk_read_source"] = round((io1 - io0) / 1e9, 2), "/proc/<pid>/io read_bytes of llama-server over the run"
    return run


# ---------------------------------------------------------------- summaries

def band_edges(ctx, k=4):
    w = ctx / k
    return [(int(i * w), int((i + 1) * w)) for i in range(k)]


def bands(turns, ctx):
    """Per depth band (the turn's depth before its answer): turns, prompt tok/s (new tokens / prompt time, turns with >= 64 new tokens), decode tok/s
    (>= 16 generated), the share of the prompt that came from the cache, median time to first token, and the plugin's decode hit rate."""
    out = []
    for lo, hi in band_edges(ctx):
        ts = [t for t in turns if t["depth_before"] is not None and lo <= t["depth_before"] < hi and not t["error"]]
        d = {"band": f"{lo // 1024}k-{hi // 1024}k", "lo": lo, "hi": hi, "turns": len(ts)}
        pt = [t for t in ts if (t["prompt_n"] or 0) >= MIN_PROMPT_N and t["prompt_ms"]]
        dt = [t for t in ts if (t["predicted_n"] or 0) >= MIN_PREDICTED_N and t["predicted_ms"]]
        d["prompt_tps"] = round(sum(t["prompt_n"] for t in pt) / (sum(t["prompt_ms"] for t in pt) / 1000), 2) if pt else None
        d["prompt_turns"] = len(pt)
        d["decode_tps"] = round(sum(t["predicted_n"] for t in dt) / (sum(t["predicted_ms"] for t in dt) / 1000), 2) if dt else None
        d["decode_turns"] = len(dt)
        tot = sum(t["depth_before"] for t in ts if t["cache_n"] is not None)
        d["cache_reuse_pct"] = round(100 * sum(t["cache_n"] for t in ts if t["cache_n"] is not None) / tot, 1) if tot else None
        tt = [t["ttft_s"] for t in ts if t["ttft_s"] is not None]
        d["ttft_median_s"] = round(statistics.median(tt), 2) if tt else None
        h = [t for t in ts if t["hits_dec"] is not None and t["misses_dec"] is not None]
        hs, ms = sum(t["hits_dec"] for t in h), sum(t["misses_dec"] for t in h)
        d["hit_rate_dec_pct"] = round(100 * hs / (hs + ms), 1) if h and hs + ms else None
        out.append(d)
    return out


def overall(run):
    """Whole-run numbers for the record: weighted prompt and decode tok/s, median ttft, cache reuse."""
    ts = [t for t in run["turns"] if not t["error"]]
    pt = [t for t in ts if (t["prompt_n"] or 0) >= MIN_PROMPT_N and t["prompt_ms"]]
    dt = [t for t in ts if (t["predicted_n"] or 0) >= MIN_PREDICTED_N and t["predicted_ms"]]
    tot = sum(t["depth_before"] for t in ts if t["cache_n"] is not None and t["depth_before"] is not None)
    tt = [t["ttft_s"] for t in ts if t["ttft_s"] is not None]
    h = [t for t in ts if t["hits_dec"] is not None and t["misses_dec"] is not None]
    hs, ms = sum(t["hits_dec"] for t in h), sum(t["misses_dec"] for t in h)
    return {"hit_rate_dec_pct": 100 * hs / (hs + ms) if h and hs + ms else None,
            "prompt_tps": sum(t["prompt_n"] for t in pt) / (sum(t["prompt_ms"] for t in pt) / 1000) if pt else None,
            "decode_tps": sum(t["predicted_n"] for t in dt) / (sum(t["predicted_ms"] for t in dt) / 1000) if dt else None,
            "ttft_s": statistics.median(tt) if tt else None,
            "cache_hit_pct": 100 * sum(t["cache_n"] for t in ts if t["cache_n"] is not None) / tot if tot else None}


def _f(x, n=1):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


NO_CACHE = "n/a (no cache)"


def turn_table(run):
    rows = ["| turn | kind | tokens added | depth before | depth after | prompt tok/s | new / cached tokens | decode tok/s | first token s | hits_dec / misses_dec |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for t in run["turns"]:
        hm = (NO_CACHE if run["arm"] == "stock" else "-") if t["hits_dec"] is None else f"{t['hits_dec']} / {t['misses_dec']}"
        rows.append(f"| {t['turn']} | {t['kind']} | {_f(t['tokens_added'])} | {_f(t['depth_before'])} | {_f(t['depth_after'])} | {_f(t['prompt_tps'])} | "
                    f"{_f(t['prompt_n'])} / {_f(t['cache_n'])} | {_f(t['decode_tps'])} | {_f(t['ttft_s'], 2)} | {hm} |")
    return "\n".join(rows)


def band_table(runs):
    """Side-by-side per band: the mean over the runs of each arm (order of the runs kept in the header)."""
    arms = []
    for r in runs:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    def mean(arm, i, key):
        xs = [r["bands"][i][key] for r in runs if r["arm"] == arm and r["bands"][i][key] is not None]
        return sum(xs) / len(xs) if xs else None
    head = ["band", "turns"] + [f"{a} prompt tok/s" for a in arms] + [f"{a} decode tok/s" for a in arms] + [f"{a} cache reuse %" for a in arms] + \
           [f"{a} first token s" for a in arms] + [f"{a} hit rate %" for a in arms]
    if len(arms) == 2:
        head += [f"prompt {arms[1]} / {arms[0]}", f"decode {arms[1]} / {arms[0]}"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, b in enumerate(runs[0]["bands"]):
        cells = [b["band"], "/".join(str(r["bands"][i]["turns"]) for r in runs)]
        for key, n in (("prompt_tps", 1), ("decode_tps", 1), ("cache_reuse_pct", 1), ("ttft_median_s", 2), ("hit_rate_dec_pct", 1)):
            cells += [(NO_CACHE if key == "hit_rate_dec_pct" and a == "stock" else _f(mean(a, i, key), n)) for a in arms]
        if len(arms) == 2:
            for key in ("prompt_tps", "decode_tps"):
                x, y = mean(arms[0], i, key), mean(arms[1], i, key)
                cells.append(f"{y / x:.2f}x" if x and y else "-")
        rows.append("| " + " | ".join(cells) + " |")
    note = ("Runs in this order: " + ", ".join(f"{r['arm']} ({r['final_depth']} tokens, {len(r['turns'])} turns, stopped: {r['stop_reason']})" for r in runs) +
            ". Prompt tok/s = new tokens read / prompt time over turns with at least 64 new tokens; decode tok/s = over turns with at least 16 generated tokens; "
            "a band is the depth of the prompt before the answer. '-' = not measured or not enough turns; the stock server has no expert cache, so its hit rate is n/a.")
    return "\n".join(rows) + "\n\n" + note


# ---------------------------------------------------------------- the record

def build_record(runs, a):
    import moe_cache_record as rr
    # the plugin this record names is the one the moe-cache server ran (MOE_CACHE_LIB of its own command line), never a file search next to the launcher
    lib = next((r["settings"]["environment"]["MOE_CACHE_LIB"] for r in runs if isinstance(r.get("settings"), dict)
                and isinstance(r["settings"].get("environment"), dict) and r["settings"]["environment"].get("MOE_CACHE_LIB")), None)
    env = rr.collect_environment(a.model, a.llama_server, backend=a.backend, model_url=a.model_url, lib=lib if lib and os.path.isfile(lib) else None) \
        if (a.model or a.llama_server) else rr.collect_environment(None, None, listing="", version_line="", backend=a.backend, lib=lib if lib and os.path.isfile(lib) else None)
    if lib and not os.path.isfile(lib):
        env["software"].update(plugin_file=os.path.basename(lib.replace("\\", "/")), plugin_sha256_prefix=rr.NR)
    elif not lib:
        env["software"].update(plugin_file=rr.NR, plugin_sha256_prefix=rr.NR)
    entries, counts = [], {}
    for run in runs:
        counts[run["arm"]] = counts.get(run["arm"], 0) + 1
        o = overall(run)
        e = rr.run_entry(run["arm"], counts[run["arm"]], {"prompt_tps": o["prompt_tps"], "decode_tps": o["decode_tps"], "ttft_s": o["ttft_s"], "wall_s": run["wall_s"],
                                                           "prompt_tokens": run["final_depth"], "cache_hit_pct": o["cache_hit_pct"],
                                                           "load_s": run.get("load_s"), "disk_read_gb": run.get("disk_read_gb"),
                                                           "error": run["stop_reason"] if run["stop_reason"].startswith("error") else "",
                                                           "text_sha256": run["user_turns_sha256"]}, run.get("monitor"))
        e["hit_rate_dec_pct"] = rr.rnd(o["hit_rate_dec_pct"], 1) if o["hit_rate_dec_pct"] is not None else ("n/a (no cache)" if run["arm"] == "stock" else rr.NR)
        e["disk_read_source"] = run.get("disk_read_source", rr.NR)
        e["settings"] = run.get("settings") or rr.NR
        e["server_log_facts"] = run.get("server_log_facts") or rr.NR
        e["turns"] = run["turns"]
        e["bands"] = run["bands"]
        e["stop_reason"] = run["stop_reason"]
        entries.append(e)
    r0 = runs[0]
    seq = [r["arm"] for r in runs]
    order = "single" if len(seq) == 1 else "ABBA" if len(seq) == 4 and seq[0] == seq[3] and seq[1] == seq[2] and seq[0] != seq[1] else rr.NR
    depths = [r["final_depth"] for r in runs if r["final_depth"] is not None]
    proto = {"arms": sorted(counts), "order": order, "runs_per_arm": max(counts.values()), "prompt_set": f"conversation script, seed {r0['seed']}, filled to a {r0['ctx']}-token context",
             "prompt_sha256": r0["user_turns_sha256"], "prompt_tokens": max(depths) if depths else rr.NR,
             "n_predict": r0["max_tokens"], "temperature": 0, "seed": 42, "sampler": "greedy, cache_prompt on, chat template of the model"}
    given = getattr(a, "command_line", None) or getattr(a, "server_cmd", None)
    if given:
        st = rr.settings_from(given, env={}, log_lines=None)
        src = "the command line given to assemble / ab"
    else:
        first = next((r for r in runs if r.get("settings") and r["settings"].get("command_line") not in (None, rr.NR)), None)
        st = dict(first["settings"]) if first else rr.settings_from("", env={}, log_lines=None)
        src = f"the {first['arm']} run" if first else "nothing (no command line was given)"
    st["ctx"] = r0["ctx"] if st.get("ctx") in (None, rr.NR) else st["ctx"]
    cond = rr.conditions_now(cold={"cold": True, "warm": False}.get(a.cache), baseline=a.baseline, other_programs=a.other_programs)
    rec = rr.assemble(env, st, cond, proto, entries, reproduce={"commands": ["python tools/conversation_bench.py run --ctx %d --max-tokens %d --seed %d ..." % (r0["ctx"], r0["max_tokens"], r0["seed"])],
                                                                "kit": rr.NR, "model_url": rr.nr(a.model_url)},
                      notes=["conversation benchmark (TEST-699): the same seeded multi-turn conversation per run, history re-sent every turn with the prompt cache on",
                             "runs in this order: " + ", ".join(seq),
                             f"settings in this record: {src}; each run carries its own settings, server-log facts, resource peaks, load time and disk reads (runs[]), "
                             "because the arms can start llama-server with different GPU layers",
                             "resource peaks come from the bench-report sampler on llama-server's process tree; VRAM is the server's own share when the OS reports it, else the whole "
                             "card (runs[].vram_source says which); disk reads are the plugin's own counter or /proc/<pid>/io on Linux, else not recorded",
                             "per-turn and per-depth-band numbers are in runs[].turns and runs[].bands; the summary below is the mean over whole runs"])
    rec["outputs"]["identical_to_baseline"] = "not checked"
    return rec


# ---------------------------------------------------------------- command line

def _key(url, given):
    if given:
        return given
    try:
        import agent_bench
        return agent_bench._key_for(url)
    except Exception:
        return ""


def _run_args(p):
    p.add_argument("--ctx", type=int, default=65536, help="context limit to fill (default 65536; capped at the server's n_ctx when /props gives it)")
    p.add_argument("--max-tokens", type=int, default=128, help="answer length per turn (default 128)")
    p.add_argument("--seed", type=int, default=42, help="seed of the conversation script")
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--script", help="JSON list of user messages to play instead of the seeded script")
    p.add_argument("--api-key", default="")
    p.add_argument("--model", default="conversation-bench", help="model name sent in the request (the server ignores it unless --alias is used)")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--no-think", action="store_true", help="send chat_template_kwargs enable_thinking=false")
    p.add_argument("--ignore-eos", action="store_true", help="make every answer exactly --max-tokens long (the model may ramble)")
    p.add_argument("--stats-pid", type=int, help="llama-server's pid, to read the plugin's counters (needs --stats-log and MOE_CACHE_STATS_SIGNAL=1)")
    p.add_argument("--stats-log", help="the server's stderr log file")
    p.add_argument("--monitor-pid", type=int, help="llama-server's pid: VRAM, process RAM, free RAM and CPU peaks are sampled during the run (the bench-report sampler)")
    p.add_argument("--command-line", dest="server_cmd", help="the llama-server command line of this run (ctx, ub, GPU layers, flags go into the record); an 'env A=1 B=2 ' prefix "
                   "records environment variables; @FILE reads the text from a file")
    p.add_argument("--server-log", help="llama-server's log file (layer counts and buffer sizes are read from it)")
    p.add_argument("--load-s", type=float, help="seconds from the server start to its first ready answer (measured by the driver)")


def _one(a, url, arm, log=print):
    key = _key(url, a.api_key)
    ctx = a.ctx
    sc = server_ctx(url, key)
    if sc and sc < ctx:
        log(f"the server's context is {sc}, below --ctx {ctx}: using {sc}")
        ctx = sc
    stats = Stats(a.stats_pid, a.stats_log) if a.stats_pid and a.stats_log else None
    script = load_script(a.script) if a.script else None
    log(f"conversation: arm {arm}, {url}, context {ctx}, {a.max_tokens} tokens per answer, seed {a.seed}")
    run = run_conversation(url, arm, ctx, a.max_tokens, a.seed, key, a.model, a.timeout, stats, a.max_turns, script, a.no_think, a.ignore_eos, log, a.monitor_pid)
    try:
        import moe_cache_record as rr
        lines = []
        if a.server_log and os.path.exists(a.server_log):
            with open(a.server_log, "rb") as f:
                lines = f.read(40_000_000).decode("utf-8", "replace").splitlines()
        cmd = a.server_cmd or ""
        if cmd.startswith("@"):                  # --command-line @FILE: the text is in a file (quotes and spaces survive PowerShell's argument passing)
            with open(cmd[1:], encoding="utf-8-sig") as f:
                cmd = f.read().strip()
        run["command_line"] = cmd
        run["settings"] = rr.settings_from(cmd, env={}, log_lines=lines)
        run["server_log_facts"] = rr.parse_llama_log(lines)
    except Exception as ex:
        log(f"settings not read: {type(ex).__name__}: {ex}")
    if a.load_s is not None:
        run["load_s"] = round(a.load_s, 1)
    return run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--url", required=True); p.add_argument("--arm", required=True, help="stock or moe-cache (a label)"); p.add_argument("--out"); _run_args(p)
    p = sub.add_parser("ab"); p.add_argument("--stock-url", required=True); p.add_argument("--plugin-url", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--order", default="stock,moe-cache,moe-cache,stock"); _run_args(p)
    for q in (p,):
        q.add_argument("--llama-server"); q.add_argument("--backend"); q.add_argument("--baseline"); q.add_argument("--other-programs")
        q.add_argument("--cache", choices=["cold", "warm"]); q.add_argument("--model-file"); q.add_argument("--model-url")
    p = sub.add_parser("assemble"); p.add_argument("runs", nargs="+"); p.add_argument("--out", required=True)
    p.add_argument("--model", help="the model .gguf (hashes and quant are read from it)"); p.add_argument("--llama-server"); p.add_argument("--backend"); p.add_argument("--model-url")
    p.add_argument("--command-line"); p.add_argument("--baseline"); p.add_argument("--other-programs"); p.add_argument("--cache", choices=["cold", "warm"])
    p = sub.add_parser("table"); p.add_argument("file"); p.add_argument("--turns", action="store_true", help="also the per-turn table of every run")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        run = _one(a, a.url, a.arm)
        text = json.dumps(run, indent=1)
        if a.out:
            open(a.out, "w", encoding="utf-8").write(text)
        print(band_table([run]))
        return 0 if not run["stop_reason"].startswith("error") else 1
    if a.cmd == "ab":
        os.makedirs(a.out_dir, exist_ok=True)
        kinds = [k.strip() for k in a.order.split(",") if k.strip()]
        runs = []
        for i, k in enumerate(kinds, 1):
            url = a.stock_url if k == "stock" else a.plugin_url
            run = _one(a, url, k)
            open(os.path.join(a.out_dir, f"{i}-{k}.json"), "w", encoding="utf-8").write(json.dumps(run, indent=1))
            runs.append(run)
        a.model = a.model_file
        rec = build_record(runs, a)
        open(os.path.join(a.out_dir, "record.json"), "w", encoding="utf-8").write(json.dumps(rec, indent=1, ensure_ascii=False))
        print(band_table(runs))
        return 0
    if a.cmd == "assemble":
        runs = [json.load(open(f, encoding="utf-8")) for f in a.runs]
        rec = build_record(runs, a)
        import moe_cache_record as rr
        errs = rr.validate(rec, rr.load_schema()) if rr.load_schema() else []
        for e in errs:
            print("record: " + e, file=sys.stderr)
        open(a.out, "w", encoding="utf-8").write(json.dumps(rec, indent=1, ensure_ascii=False))
        print(band_table(runs))
        return 1 if errs else 0
    if a.cmd == "table":
        d = json.load(open(a.file, encoding="utf-8"))
        runs = d["runs"] if "runs" in d and "record_schema" in d else [d]
        for r in runs:
            r.setdefault("final_depth", r.get("prompt_tokens"))
            r.setdefault("ctx", d.get("settings", {}).get("ctx", 65536) if "settings" in d else 65536)
            r.setdefault("stop_reason", "")
            r.setdefault("seed", "-")
        print(band_table(runs))
        if a.turns:
            for r in runs:
                print(f"\n{r['arm']}:\n" + turn_table(r))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
