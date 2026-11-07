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
import argparse, datetime, hashlib, http.client, json, os, random, re, signal, statistics, subprocess, sys, time, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
sys.path.insert(0, HERE)

CHARS_PER_TOKEN = 3.0       # deliberately low: a turn is judged too big a little early, never too late
MARGIN = 48                 # tokens of slack for the chat template around a turn
MIN_PROMPT_N = 64           # a turn that read fewer new tokens does not enter the prompt-speed mean (timer noise)
MIN_PREDICTED_N = 16        # the same for decode speed
SYSTEM = "You are a careful coding assistant working in the user's repository. Answer briefly."
# TEST-699 fix (2026-11-07): with the old settings (thinking on, 128 tokens) the thinking block used the answer length and, from about turn 11, the model copied its own
# earlier 1-2 token replies, so the deep turns decoded a handful of tokens. The command line now asks for a visible answer: thinking off, 256 tokens, and every user
# message ends with ANSWER_HINT. --legacy brings the old behaviour back (comparable with the v0.3.7 records).
ANSWER_HINT = "\n\n(Please answer in at least three full sentences.)"
NEUTRAL_SYSTEM = "You are a careful coding assistant working in the user's repository."
MIN_MEDIAN_ANSWER = 24          # median answer tokens below this: the run carries a warning (a conversation of one-word answers measures nothing about decode)
SHORT_ANSWER = 8                # an answer of fewer tokens than this counts as short

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


def count_tokens(url, api_key, text, timeout=30):
    """The server's own token count of `text` (POST /tokenize); None when the server has no such endpoint or does not answer. The turn planning uses it instead of a
    characters-per-token guess: code and logs are much denser than 3 characters per token, and with 256-token answers the guess let a request run over the context."""
    try:
        c, base = _conn(url, timeout)
        hdr = {"Content-Type": "application/json"}
        if api_key:
            hdr["Authorization"] = "Bearer " + api_key
        c.request("POST", base + "/tokenize", body=json.dumps({"content": text, "add_special": False}).encode(), headers=hdr)
        r = c.getresponse()
        data = r.read()
        c.close()
        if r.status != 200:
            return None
        toks = json.loads(data).get("tokens")
        return len(toks) if isinstance(toks, list) else None
    except Exception:
        return None


def is_context_error(err):
    """True for the server's refusal of a request that does not fit the context (llama.cpp: HTTP 400 'the request (N tokens) exceeds the available context size (M tokens)')."""
    e = str(err or "").lower()
    return "400" in e and ("exceeds the available context" in e or "exceed_context_size" in e or "exceeds the context" in e or "context size" in e)


def chat(url, body, api_key, timeout):
    """One streamed chat request -> dict: text, reasoning (chars), ttft_s, wall_s, timings, usage, status, error."""
    res = {"text": "", "finish_reason": None, "reasoning_chars": 0, "ttft_s": None, "wall_s": None, "timings": None, "usage": None, "status": None, "error": None}
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
                    if ch.get("finish_reason"):
                        res["finish_reason"] = ch["finish_reason"]
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
    """Bytes the process read so far, or None. Linux: /proc/<pid>/io read_bytes (storage reads, mmap page-ins included). Windows: ReadTransferCount of
    GetProcessIoCounters (explicit ReadFile reads only: memory-mapped page-ins are NOT in it, so for a server started with --load-mode mmap it is a lower bound)."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            class IOC(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.OpenProcess.restype = wintypes.HANDLE
            h = k.OpenProcess(0x1000, False, int(pid))      # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return None
            try:
                c = IOC()
                return int(c.ReadTransferCount) if k.GetProcessIoCounters(h, ctypes.byref(c)) else None
            finally:
                k.CloseHandle(h)
        except Exception:
            return None
    try:
        with open(f"/proc/{int(pid)}/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def parse_drive_raw(text):
    """The cumulative Disk Read Bytes of one drive from `Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk ... | Select -Expand DiskReadBytesPersec` text:
    the first whole number in it, or None."""
    m = re.search(r"^\s*(\d+)\s*$", text or "", re.M)
    return int(m.group(1)) if m else None


def drive_read_bytes(letter):
    """Windows only: bytes ever read from the physical disk that holds drive `letter` (the raw cumulative counter behind 'PhysicalDisk\\Disk Read Bytes/sec'; the
    whole drive, other programs included, memory-mapped page-ins included). None elsewhere or when the counter cannot be read."""
    if os.name != "nt" or not letter or not re.fullmatch(r"[A-Za-z]", letter):
        return None
    ps = ("(Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk | Where-Object { $_.Name -like '* %s:' } | Select-Object -First 1).DiskReadBytesPersec" % letter.upper())
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_drive_raw(out)


def model_drive(cmd):
    """Drive letter of the -m path in a Windows llama-server command line, or None."""
    m = re.search(r'(?:^|\s)(?:-m|--model)[\s=]+"?([A-Za-z]):[\\/]', cmd or "")
    return m.group(1).upper() if m else None


def disk_read_facts(plugin_gb, nt, io0, io1, dr0, dr1, drive):
    """The run's disk-read fields. Order of trust: the plugin's own counter (plugin_gb: per-turn values, summed); on Linux /proc/<pid>/io; on Windows the model
    drive's raw disk counter (sees mapped page-ins), else the process ReadTransferCount (a lower bound with mmap). The process figure is always kept next to it.
    Nothing readable: no field (the record says "not recorded")."""
    out = {}
    if io0 is not None and io1 is not None:
        out["disk_read_process_gb"] = round((io1 - io0) / 1e9, 2)
        out["disk_read_process_source"] = ("/proc/<pid>/io read_bytes of llama-server over the run" if not nt else
                                           "Windows ReadTransferCount of llama-server over the run (explicit reads only; memory-mapped page-ins are not counted: a lower bound with --load-mode mmap)")
    if plugin_gb:
        out["disk_read_gb"], out["disk_read_source"] = round(sum(plugin_gb), 2), "the plugin's own read_bytes counter, summed over the turns"
    elif not nt and "disk_read_process_gb" in out:
        out["disk_read_gb"], out["disk_read_source"] = out["disk_read_process_gb"], out["disk_read_process_source"]
    elif dr0 is not None and dr1 is not None:
        out["disk_read_gb"] = round((dr1 - dr0) / 1e9, 2)
        out["disk_read_source"] = f"Windows PhysicalDisk raw Disk Read Bytes of the model's drive {drive}: over the run, whole drive (other programs and mapped page-ins included)"
    elif "disk_read_process_gb" in out:
        out["disk_read_gb"], out["disk_read_source"] = out["disk_read_process_gb"], out["disk_read_process_source"]
    return out


def delta(s0, s1, key):
    if s0 is None or s1 is None or key not in s0 or key not in s1:
        return None
    return s1[key] - s0[key]


# ---------------------------------------------------------------- one conversation

def run_conversation(url, arm, ctx, max_tokens=128, seed=42, api_key="", model="conversation-bench", timeout=1800, stats=None, max_turns=None,
                     script=None, no_think=False, ignore_eos=False, log=print, monitor_pid=None, drive=None, answer_hint=False, min_median=MIN_MEDIAN_ANSWER,
                     system=None, temperature=0.0, empty_marker="(no visible answer)", hint_text=None, keep_answers=True):
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
    dr0 = drive_read_bytes(drive) if drive else None
    hint = ANSWER_HINT if hint_text is None else hint_text
    turns, history = [], [{"role": "system", "content": SYSTEM if system is None else system}]
    prev_total, n, stop = 0, 0, "max turns reached"
    use_tokenize = True
    t_start = time.time()
    while max_turns is None or n < max_turns:
        pick = None
        for attempt in range(6):
            cand = (script[n] if (script and attempt == 0 and n < len(script)) else None) if script else make_turn(seed, n, attempt)
            if script and cand is None:
                stop = "script ended"; break
            full = cand["text"] + (hint if answer_hint else "")
            real = count_tokens(url, api_key, full) if use_tokenize else None
            if real is None and use_tokenize:
                use_tokenize = False                       # no /tokenize: the guess below, and it is not asked again
                log("  (the server has no /tokenize: turns are sized by characters, with the safety net of a context-full answer)")
            est = (real + MARGIN) if real is not None else (len(full) / CHARS_PER_TOKEN + MARGIN)
            if prev_total + est + max_tokens <= ctx:
                pick = cand; break
            if script:
                stop = "context full"; break
        if pick is None:
            if stop == "max turns reached":
                stop = "context full"
            break
        history.append({"role": "user", "content": pick["text"] + (hint if answer_hint else "")})
        body = {"model": model, "messages": history, "max_tokens": max_tokens, "temperature": temperature, "seed": 42, "stream": True,
                "stream_options": {"include_usage": True}, "cache_prompt": True}
        if no_think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if ignore_eos:
            body["ignore_eos"] = True
        s0 = stats.snapshot() if stats else None
        r = chat(url, body, api_key, timeout)
        if is_context_error(r["error"]) and turns:         # safety net: the size estimate was a little low; this is the end of the context, not a failure of the run (a refusal of the very first turn stays an error: the context is too small)
            history.pop()
            stop = "context full"
            log(f"  [{arm}] turn {n:3d}: the server says the request does not fit the context ({str(r['error'])[:120]}): stopping as context full")
            break
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
               "gpu_blocks": delta(s0, s1, "gpu_segments"),
               "gpu_up_saved_gb": (lambda d: round(d / 1e9, 3) if d is not None else None)(delta(s0, s1, "gpu_up_saved")),
               "ssd_wait_s": (lambda d: round(d / 1e9, 3) if d is not None else None)(delta(s0, s1, "ns_ssd_wait")),
               "answer_chars": len(r["text"]), "answer_sha256": hashlib.sha256(r["text"].encode("utf-8")).hexdigest()[:16],
               "finish_reason": r.get("finish_reason"), "reasoning_chars": r.get("reasoning_chars", 0)}
        if keep_answers:
            row["answer_head"] = r["text"][:400]          # the first 400 characters of the visible answer: what the 1-2 token answers actually say
        turns.append(row)
        log(f"  [{arm}] turn {n:3d} {pick['kind']:<11} +{row['tokens_added']} tok  depth {row['depth_before']}->{row['depth_after']}  "
            f"prompt {row['prompt_tps']} tok/s ({row['prompt_n']} new, {row['cache_n']} cached)  decode {row['decode_tps']} tok/s  ttft {row['ttft_s']} s"
            + (f"  hits_dec {row['hits_dec']}/{row['misses_dec']}" if row["hits_dec"] is not None else ""))
        if r["error"]:
            stop = "error: " + str(r["error"])[:200]
            history.pop()
            break
        history.append({"role": "assistant", "content": r["text"] if r["text"] else empty_marker})
        prev_total = row["depth_after"] if row["depth_after"] is not None else prev_total + int(est)
        n += 1
    run = {"arm": arm, "url": url, "ctx": ctx, "max_tokens": max_tokens, "seed": seed, "turns": turns, "stop_reason": stop,
           "final_depth": next((t["depth_after"] for t in reversed(turns) if t.get("depth_after") is not None), None), "wall_s": round(time.time() - t_start, 1),
           "user_turns_sha256": hashlib.sha256("\n".join(m["content"] for m in history if m["role"] == "user").encode("utf-8")).hexdigest()[:16],
           "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")}
    run["bands"] = bands(turns, ctx)
    run["thinking"] = not no_think
    run["answer_hint"] = bool(answer_hint)
    run["answer_stats"] = answer_stats(turns, min_median)
    for w in run["answer_stats"]["warnings"]:
        log("WARNING: " + w)
    if smp is not None:
        run["monitor"] = smp.stop()
    rb = [t["read_gb"] for t in turns if t["read_gb"] is not None]
    io1 = proc_read_bytes(monitor_pid) if monitor_pid else None
    dr1 = drive_read_bytes(drive) if drive else None
    # the plugin's own counters, run total = the sum over the turns (the same rule as the disk reads: time between turns is not counted)
    for field in ("gpu_blocks", "gpu_up_saved_gb", "ssd_wait_s"):
        vals = [t[field] for t in turns if t.get(field) is not None]
        if vals:
            run[field] = round(sum(vals), 3) if field != "gpu_blocks" else int(sum(vals))
    run.update(disk_read_facts(rb, os.name == "nt", io0, io1, dr0, dr1, drive))
    return run


# ---------------------------------------------------------------- summaries

def answer_stats(turns, min_median=MIN_MEDIAN_ANSWER):
    """How long the answers were: median tokens, the turns with a short (< SHORT_ANSWER tokens) or empty visible answer, and warnings when the median is under
    min_median or more than a quarter of the turns have no visible answer. Decode speed measured on one-word answers says little about decode."""
    n = [t["predicted_n"] for t in turns if t.get("predicted_n") is not None]
    empty = sum(1 for t in turns if t.get("answer_chars") == 0)
    short = sum(1 for x in n if x < SHORT_ANSWER)
    med = statistics.median(n) if n else None
    warnings = []
    if med is not None and med < min_median:
        warnings.append(f"the median answer is {med:g} tokens (under {min_median}): {short} of {len(n)} answers are shorter than {SHORT_ANSWER} tokens; deep-turn decode numbers "
                        "rest on very short answers and the conversation cannot judge meaning (use the default settings, not --legacy, or give --script with longer questions)")
    if turns and empty * 4 > len(turns):
        warnings.append(f"{empty} of {len(turns)} turns have no visible answer (thinking may use the whole --max-tokens)")
    # for finding the cause: does a short answer make the next one short (the model copying its own history), where does it start, and which kinds of user turn get short answers
    ps = [(t.get("predicted_n"), t.get("kind")) for t in turns if t.get("predicted_n") is not None]
    after_short = [b[0] < SHORT_ANSWER for a, b in zip(ps, ps[1:]) if a[0] < SHORT_ANSWER]
    after_long = [b[0] < SHORT_ANSWER for a, b in zip(ps, ps[1:]) if a[0] >= SHORT_ANSWER]
    by_kind = {}
    for x, k in ps:
        by_kind.setdefault(k, []).append(x)
    first_short = next((t["turn"] for t in turns if t.get("predicted_n") is not None and t["predicted_n"] < SHORT_ANSWER), None)
    reasons = {}
    for t in turns:
        reasons[str(t.get("finish_reason"))] = reasons.get(str(t.get("finish_reason")), 0) + 1
    return {"median_tokens": med, "short_turns": short, "empty_turns": empty, "turns": len(turns), "warnings": warnings, "first_short_turn": first_short,
            "p_short_after_short": round(sum(after_short) / len(after_short), 3) if after_short else None,
            "p_short_after_long": round(sum(after_long) / len(after_long), 3) if after_long else None,
            "median_by_kind": {k: statistics.median(v) for k, v in by_kind.items()}, "finish_reasons": reasons}

def cost_fit(turns, ub):
    """TEST-754: prompt time per turn = A x passes + B x (new tokens / 1k) + C x (new tokens / 1k) x (depth / 1k), least squares over the turns that read a prompt.
    passes = ceil(new tokens / ub) (every pass pays the fixed cost of going over the experts once); depth = tokens already cached + half the new ones.
    Returns {"a_s_per_pass", "b_s_per_1k", "c_s_per_1k_x_1k_depth", "turns", "ub", "rms_s"}, or None with fewer than 4 usable turns (or a flat system).
    With one -ub the passes column is nearly proportional to the tokens, so A and B only separate when -ub differs between runs: run the same chat at
    -ub 2048 and -ub 4096 and fit them together (pass several runs: cost_fit_runs)."""
    rows = []
    for t in turns:
        if t.get("error") or not t.get("prompt_n") or not t.get("prompt_ms") or not ub:
            continue
        n = t["prompt_n"]; d = (t.get("cache_n") or 0) + n / 2
        rows.append(([-(-n // ub), n / 1000, (n / 1000) * (d / 1000)], t["prompt_ms"] / 1000))
    return _lsq(rows, ub)


def cost_fit_runs(runs_with_ub):
    """The same fit over the turns of several runs, each (turns, ub): lets an -ub 4096 run separate A from B."""
    rows = []
    for turns, ub in runs_with_ub:
        for t in turns:
            if t.get("error") or not t.get("prompt_n") or not t.get("prompt_ms") or not ub:
                continue
            n = t["prompt_n"]; d = (t.get("cache_n") or 0) + n / 2
            rows.append(([-(-n // ub), n / 1000, (n / 1000) * (d / 1000)], t["prompt_ms"] / 1000))
    return _lsq(rows, "/".join(str(u) for _, u in runs_with_ub))


def _lsq(rows, ub):
    if len(rows) < 4:
        return None
    m = [[sum(r[0][i] * r[0][j] for r in rows) for j in range(3)] + [sum(r[0][i] * r[1] for r in rows)] for i in range(3)]
    for c in range(3):                                   # Gauss-Jordan with partial pivoting
        piv = max(range(c, 3), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-9:
            return None
        m[c], m[piv] = m[piv], m[c]
        m[c] = [v / m[c][c] for v in m[c]]
        for r in range(3):
            if r != c:
                f = m[r][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    A, B, C = m[0][3], m[1][3], m[2][3]
    rms = (sum((y - (A * x[0] + B * x[1] + C * x[2])) ** 2 for x, y in rows) / len(rows)) ** 0.5
    return {"a_s_per_pass": round(A, 3), "b_s_per_1k": round(B, 3), "c_s_per_1k_x_1k_depth": round(C, 4), "turns": len(rows), "ub": ub, "rms_s": round(rms, 3)}


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
                                                           "load_s": run.get("load_s"), "disk_read_gb": run.get("disk_read_gb"), "gpu_blocks": run.get("gpu_blocks"),
                                                           "error": run["stop_reason"] if run["stop_reason"].startswith("error") else "",
                                                           "text_sha256": run["user_turns_sha256"]}, run.get("monitor"))
        e["hit_rate_dec_pct"] = rr.rnd(o["hit_rate_dec_pct"], 1) if o["hit_rate_dec_pct"] is not None else ("n/a (no cache)" if run["arm"] == "stock" else rr.NR)
        e["disk_read_source"] = run.get("disk_read_source", rr.NR)
        for k in ("disk_read_process_gb", "disk_read_process_source", "gpu_up_saved_gb", "ssd_wait_s"):
            e[k] = run.get(k, rr.NR)
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
             "n_predict": r0["max_tokens"], "temperature": 0, "seed": 42, "sampler": "greedy, cache_prompt on, chat template of the model" + ("" if r0.get("thinking") is None else (", thinking " + ("on" if r0["thinking"] else "off") + (", answer hint on" if r0.get("answer_hint") else ", answer hint off")))}
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
    p.add_argument("--max-tokens", type=int, default=None, help="answer length per turn (default 256; 128 with --legacy)")
    p.add_argument("--seed", type=int, default=42, help="seed of the conversation script")
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--script", help="JSON list of user messages to play instead of the seeded script")
    p.add_argument("--api-key", default="")
    p.add_argument("--model", default="conversation-bench", help="model name sent in the request (the server ignores it unless --alias is used)")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--no-think", action="store_true", help="thinking off (the default now; kept so old command lines still work)")
    p.add_argument("--think", action="store_true", help="leave thinking to the server's default (the old behaviour; the thinking block uses the answer length)")
    p.add_argument("--no-answer-hint", action="store_true", help="do not end every user message with 'Please answer in at least three full sentences.'")
    p.add_argument("--legacy", action="store_true", help="the v0.3.7 settings: --think, --no-answer-hint and --max-tokens 128 (comparable with the old records)")
    p.add_argument("--system", help="the system message ('neutral' = the default without 'Answer briefly.'; any other text is used as it is; default: the original text)")
    p.add_argument("--hint", help="the sentence added to every user message instead of the default one")
    p.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (default 0, greedy)")
    p.add_argument("--empty-marker", default="(no visible answer)", help="what goes into the history when an answer has no visible text (default '(no visible answer)'; '' stores an empty answer)")
    p.add_argument("--no-keep-answers", action="store_true", help="do not save the first 400 characters of every answer in the run")
    p.add_argument("--min-median-answer", type=int, default=MIN_MEDIAN_ANSWER, help="warn when the median answer is under this many tokens (default %d)" % MIN_MEDIAN_ANSWER)
    p.add_argument("--ignore-eos", action="store_true", help="make every answer exactly --max-tokens long (the model may ramble)")
    p.add_argument("--stats-pid", type=int, help="llama-server's pid, to read the plugin's counters (needs --stats-log and MOE_CACHE_STATS_SIGNAL=1)")
    p.add_argument("--stats-log", help="the server's stderr log file")
    p.add_argument("--monitor-pid", type=int, help="llama-server's pid: VRAM, process RAM, free RAM and CPU peaks are sampled during the run (the bench-report sampler)")
    p.add_argument("--command-line", dest="server_cmd", help="the llama-server command line of this run (ctx, ub, GPU layers, flags go into the record); an 'env A=1 B=2 ' prefix "
                   "records environment variables; @FILE reads the text from a file")
    p.add_argument("--server-log", action="append", help="llama-server's log file (layer counts, threads and buffer sizes are read from it); repeat for stdout and stderr, the plugin prints to stdout")
    p.add_argument("--ub", type=int, default=2048, help="the server's -ub, used for the cost-model fit when the command line does not say (default 2048)")
    p.add_argument("--load-s", type=float, help="seconds from the server start to its first ready answer (measured by the driver)")


def default_args(**over):
    """The `run` options with their defaults (what the command line would give), overridden by keyword: for a driver in Python (bench-report --chat)."""
    p = argparse.ArgumentParser()
    _run_args(p)
    a = p.parse_args([])
    vars(a).update(over)
    return a


def _settings(a):
    """-> (max_tokens, no_think, answer_hint) from the command line: the new defaults, or the v0.3.7 ones under --legacy / --think / --no-answer-hint."""
    legacy = getattr(a, "legacy", False)
    think = legacy or getattr(a, "think", False)
    hint = not (legacy or getattr(a, "no_answer_hint", False))
    mt = a.max_tokens if a.max_tokens is not None else (128 if legacy else 256)
    return mt, not think, hint


def _one(a, url, arm, log=print):
    a.max_tokens, no_think, hint = _settings(a)
    key = _key(url, a.api_key)
    ctx = a.ctx
    sc = server_ctx(url, key)
    if sc and sc < ctx:
        log(f"the server's context is {sc}, below --ctx {ctx}: using {sc}")
        ctx = sc
    stats = Stats(a.stats_pid, a.stats_log) if a.stats_pid and a.stats_log else None
    script = load_script(a.script) if a.script else None
    log(f"conversation: arm {arm}, {url}, context {ctx}, {a.max_tokens} tokens per answer, seed {a.seed}, temperature {a.temperature}, system {'neutral' if a.system == 'neutral' else ('custom' if a.system else 'original')}, thinking {'off' if no_think else 'server default'}, answer hint {'on' if hint else 'off'}")
    cmd = a.server_cmd or ""
    try:
        if cmd.startswith("@"):                  # --command-line @FILE: the text is in a file (quotes and spaces survive PowerShell's argument passing)
            with open(cmd[1:], encoding="utf-8-sig") as f:
                cmd = f.read().strip()
    except OSError as ex:
        log(f"command line file not read: {ex}")
        cmd = ""
    run = run_conversation(url, arm, ctx, a.max_tokens, a.seed, key, a.model, a.timeout, stats, a.max_turns, script, no_think, a.ignore_eos, log, a.monitor_pid,
                           model_drive(cmd) if os.name == "nt" else None, hint, a.min_median_answer,
                           NEUTRAL_SYSTEM if a.system == "neutral" else a.system, a.temperature, a.empty_marker, a.hint, not a.no_keep_answers)
    try:
        import moe_cache_record as rr
        lines = []
        for lf in (a.server_log or []):
            if os.path.exists(lf):
                with open(lf, "rb") as f:
                    lines += f.read(40_000_000).decode("utf-8", "replace").splitlines()
        run["command_line"] = cmd
        run["settings"] = rr.settings_from(cmd, env={}, log_lines=lines)
        run["server_log_facts"] = rr.parse_llama_log(lines)
    except Exception as ex:
        log(f"settings not read: {type(ex).__name__}: {ex}")
    ub = (run.get("settings") or {}).get("ub") if isinstance(run.get("settings"), dict) else None
    try:
        ub = int(ub) if ub else a.ub
    except (TypeError, ValueError):
        ub = a.ub
    fit = cost_fit(run["turns"], ub)
    run["cost_fit"] = fit
    log("cost-model fit per turn (prompt time = A per pass + B per 1k tokens + C per 1k tokens x 1k depth): " + (json.dumps(fit) if fit else "not enough turns")
        + "; A and B separate only with a second -ub (the -ub 4096 arm)")
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
