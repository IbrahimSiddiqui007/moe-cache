#!/usr/bin/env python3
"""agent_bench.py: agent-shaped benchmark for llama-server (idea I30). Standard library only; Linux and Windows.

Coding agents send a few large requests (5-20k tokens of tools and instructions), then many small follow-ups (tool results), with pauses while
the agent runs its tools. Decode tok/s says little about that; time to first token, prompt reuse and total task time do.

  record  a recording proxy: point the agent at it, it forwards to a real server and writes every request (body, start / end time) to a file
          python3 tools/agent_bench.py record --upstream http://127.0.0.1:8080 --port 8199 --out session.json --name "opencode: fix calc.py"
          (then run the agent against http://127.0.0.1:8199/v1; Ctrl+C or --idle SECONDS ends the recording)
  replay  sends the recorded requests again, with the same bodies, the same overlap and the same pauses between them, and reports per request:
          time to first token, prompt tokens processed and reused, decode tok/s; then the whole task time
          python3 tools/agent_bench.py replay session.json --url http://127.0.0.1:8080 [--json out.json] [--label stock]
  show    prints a recorded session or a saved replay result (--json files)
  table   puts saved replay results side by side: python3 tools/agent_bench.py table out/*.json

Replay notes: generation is greedy (temperature 0, fixed seed) and capped at the recorded number of tokens, so two exact servers (stock and
moe-cache) produce the same tokens. The bodies hold the conversation as recorded, so each follow-up re-sends the agent's original assistant message;
if the model now writes something else, the server re-processes that message (counted in "processed"). Pauses are the agent's own time between a
response and its next request (tool runs, its own bookkeeping) and are kept as recorded.
"""
import argparse, getpass, http.client, http.server, json, os, re, socket, statistics, sys, threading, time, urllib.parse

# ---------------------------------------------------------------- record

def _scrub_subs():
    """Patterns for recorded files that may be shared: the home folder (as a path, either slash direction) and the user and host names
    (whole words only). Applied to parsed JSON string values, so escaped backslashes in Windows paths are matched too."""
    home = os.path.expanduser("~").rstrip("\\/")
    parts = [x for x in re.split(r"[\\/]+", home) if x]
    lead = r"[\\/]" if home[:1] in "\\/" else ""
    pats = [(re.compile(lead + r"[\\/]".join(map(re.escape, parts)) + r"(?![\w])", re.I), "/home/user")]
    for n, r in ((getpass.getuser(), "user"), (socket.gethostname(), "host")):
        if n and len(n) >= 3:
            pats.append((re.compile(r"(?<![\w])" + re.escape(n) + r"(?![\w])", re.I), r))
    return pats


def _scrub(v, pats):
    if isinstance(v, str):
        for rx, r in pats:
            v = rx.sub(r, v)
        return v
    if isinstance(v, list):
        return [_scrub(x, pats) for x in v]
    if isinstance(v, dict):
        return {k: _scrub(x, pats) for k, x in v.items()}
    return v


def _sse_summary(raw):
    """Summary of a streamed chat completion: completion tokens, timings, first-token time is measured by the caller."""
    out = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            continue
        try:
            j = json.loads(d)
        except ValueError:
            continue
        if j.get("timings"):
            out["timings"] = j["timings"]
        if j.get("usage"):
            out["usage"] = j["usage"]
    return out


def record(a):
    up = urllib.parse.urlparse(a.upstream)
    subs = [] if a.no_scrub else _scrub_subs()   # recorded files may be shared (token counts change by a few)
    t0 = time.perf_counter()
    reqs, lock = [], threading.Lock()
    last = [time.perf_counter()]

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *x):
            pass

        def _fwd(self, method):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            ts = time.perf_counter()
            hdr = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
            c = http.client.HTTPConnection(up.hostname, up.port or 80, timeout=3600)
            c.request(method, self.path, body=body, headers=hdr)
            r = c.getresponse()
            self.send_response(r.status)
            chunked = r.getheader("Transfer-Encoding", "").lower() == "chunked" or r.getheader("Content-Length") is None
            for k, v in r.getheaders():
                if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                    self.send_header(k, v)
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            data, first = bytearray(), None
            if chunked:
                self.end_headers()
                while True:
                    b = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                    if not b:
                        break
                    first = first or time.perf_counter()
                    data += b
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b)); self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
            else:
                data = r.read(); first = time.perf_counter()
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            te = time.perf_counter()
            c.close()
            e = {"method": method, "path": self.path, "t_start": round(ts - t0, 3), "t_end": round(te - t0, 3), "status": r.status}
            if body:
                try:
                    e["body"] = _scrub(json.loads(body.decode("utf-8")), subs)
                except ValueError:
                    e["body_raw"] = _scrub(body.decode("utf-8", "replace"), subs)
            if self.path.split("?")[0].endswith("/chat/completions") or self.path.split("?")[0].endswith("/completions"):
                rec = _sse_summary(bytes(data)) if (e.get("body") or {}).get("stream") else {}
                if not rec:
                    try:
                        j = json.loads(bytes(data)); rec = {"timings": j.get("timings"), "usage": j.get("usage")}
                    except ValueError:
                        pass
                rec["t_first_byte"] = round((first or te) - t0, 3)
                e["recorded"] = rec
            with lock:
                reqs.append(e); last[0] = time.perf_counter()
                tm = (e.get("recorded") or {}).get("timings") or {}
                print(f"  {len(reqs):3d}  {method} {self.path}  {e['t_start']:7.1f}-{e['t_end']:7.1f} s"
                      + (f"  prompt {tm.get('prompt_n')} new + {tm.get('cache_n')} reused, {tm.get('predicted_n')} generated" if tm else ""), flush=True)

        def do_GET(self):
            self._run("GET")

        def do_POST(self):
            self._run("POST")

        def _run(self, method):
            try:
                self._fwd(method)
            except ConnectionError:   # the agent hung up (its own timeout; ConnectionAbortedError on Windows): that request is not recorded
                print(f"  {method} {self.path}: the client closed the connection; not recorded", flush=True)
                self.close_connection = True

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H)
    srv.daemon_threads = True
    print(f"recording: point the agent at http://127.0.0.1:{a.port}/v1 (forwards to {a.upstream}); Ctrl+C ends", flush=True)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    try:
        while True:
            time.sleep(0.5)
            if a.idle and reqs and time.perf_counter() - last[0] > a.idle:
                break
            if a.stop_file and os.path.exists(a.stop_file):
                break
    except KeyboardInterrupt:
        pass
    srv.shutdown()
    reqs.sort(key=lambda e: e["t_start"])
    out = {"format": "moe-cache agent session 1", "name": a.name or "", "recorded": time.strftime("%Y-%m-%d %H:%M"), "upstream_model": a.model or "",
           "notes": a.notes or "", "requests": reqs}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {a.out}: {len(reqs)} requests, {sum(1 for e in reqs if 'recorded' in e)} completions")

# ---------------------------------------------------------------- replay

def _schedule(reqs):
    """For each request: the event it waited on in the recording (start or end of an earlier request, the latest one before its own start) and
    the delay after it. Requests that overlapped in the recording overlap in the replay; a pause after a response stays a pause."""
    out = []
    for i, e in enumerate(reqs):
        best = (None, None, 0.0)   # (index, 'start'|'end', time)
        for j in range(i):
            for kind in ("start", "end"):
                t = reqs[j]["t_" + kind]
                if t <= e["t_start"] + 1e-6 and (best[0] is None or t >= best[2]):
                    best = (j, kind, t)
        out.append((best[0], best[1], max(0.0, e["t_start"] - best[2]) if best[0] is not None else 0.0))
    return out


def _send(url, e, a, res):
    u = urllib.parse.urlparse(url)
    base = u.path.rstrip("/")
    path = e["path"]
    if base and not path.startswith(base):
        path = base + path
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=a.timeout)
    hdr = {"Content-Type": "application/json"}
    if a.api_key:
        hdr["Authorization"] = "Bearer " + a.api_key
    body = None
    is_comp = "recorded" in e and "body" in e
    if "body" in e:
        b = dict(e["body"])
        if is_comp:
            b["stream"] = True
            b["stream_options"] = {"include_usage": True}
            b["temperature"] = 0.0; b["seed"] = 42; b.pop("top_p", None)
            b["cache_prompt"] = True   # llama-server's default; explicit so a server with another default still reuses the prefix
            if a.model:
                b["model"] = a.model
            rec_n = ((e["recorded"].get("timings") or {}).get("predicted_n") or (e["recorded"].get("usage") or {}).get("completion_tokens"))
            if rec_n and not a.no_cap:
                b["max_tokens"] = int(rec_n)
                b.pop("max_completion_tokens", None)
        body = json.dumps(b).encode()
    elif "body_raw" in e:
        body = e["body_raw"].encode()
    try:
        c.request(e["method"], path, body=body, headers=hdr)
        r = c.getresponse()
        res["status"] = r.status
        first, buf, tim, usage = None, b"", None, None
        while True:
            line = r.readline()
            if not line:
                break
            if not is_comp:
                continue
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
                if first is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("tool_calls")):
                    first = time.perf_counter()
            if j.get("timings"):
                tim = j["timings"]
            if j.get("usage"):
                usage = j["usage"]
        res["t_first"] = first
        res["timings"] = tim
        res["usage"] = usage
    except Exception as ex:   # a failed request is reported, the replay goes on
        res["error"] = f"{type(ex).__name__}: {ex}"
    res["t_end"] = time.perf_counter()
    c.close()


def replay(a):
    with open(a.session, encoding="utf-8") as f:
        s = json.load(f)
    reqs = [e for e in s["requests"] if a.all_requests or "recorded" in e]
    sched = _schedule(reqs)
    n = len(reqs)
    res = [dict() for _ in range(n)]
    started = [threading.Event() for _ in range(n)]
    done = [threading.Event() for _ in range(n)]
    print(f"replay {a.session} ({s.get('name') or '-'}): {n} requests against {a.url}" + (f" [{a.label}]" if a.label else ""), flush=True)
    if a.wait_health:
        u = urllib.parse.urlparse(a.url)
        for _ in range(a.wait_health):
            try:
                c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=5); c.request("GET", "/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                pass
            time.sleep(1)
    T0 = time.perf_counter()

    # t_start must exist before dependants read it: set it before signalling
    def run_wrapped(i):
        dep, kind, delay = sched[i]
        if dep is not None:
            (started if kind == "start" else done)[dep].wait()
            base = res[dep].get("t_start" if kind == "start" else "t_end", time.perf_counter())
            w = base + delay * a.gap_scale - time.perf_counter()
            if w > 0:
                time.sleep(w)
        res[i]["t_start"] = time.perf_counter()
        started[i].set()
        try:
            _send(a.url, reqs[i], a, res[i])
        except Exception as ex:
            res[i]["error"] = f"{type(ex).__name__}: {ex}"
        finally:
            res[i].setdefault("t_end", time.perf_counter())
            done[i].set()
        r = res[i]; tm = r.get("timings") or {}
        ttft = f"{r['t_first'] - r['t_start']:.2f}" if r.get("t_first") else "-"
        tps = f"{tm['predicted_per_second']:.1f}" if tm.get("predicted_per_second") else "-"
        print(f"  {i + 1:3d}/{n}  at {r['t_start'] - T0:7.1f} s  ttft {ttft} s  prompt {tm.get('prompt_n', '-')} new / {tm.get('cache_n', '-')} reused  "
              f"{tm.get('predicted_n', '-')} tok at {tps} tok/s  total {r['t_end'] - r['t_start']:.1f} s" + (f"  ERROR {r['error']}" if r.get("error") else ""), flush=True)

    ths = [threading.Thread(target=run_wrapped, args=(i,), daemon=True) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    T1 = time.perf_counter()
    rows = []
    for i, (e, r) in enumerate(zip(reqs, res)):
        tm = r.get("timings") or {}
        rows.append({"i": i + 1, "at": round(r["t_start"] - T0, 3), "ttft": round(r["t_first"] - r["t_start"], 3) if r.get("t_first") else None,
                     "total": round(r["t_end"] - r["t_start"], 3), "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                     "prompt_ms": tm.get("prompt_ms"), "predicted_n": tm.get("predicted_n"), "predicted_per_second": tm.get("predicted_per_second"),
                     "recorded_prompt_n": ((e["recorded"].get("timings") or {}).get("prompt_n") if "recorded" in e else None),
                     "recorded_cache_n": ((e["recorded"].get("timings") or {}).get("cache_n") if "recorded" in e else None),
                     "recorded_predicted_n": ((e["recorded"].get("timings") or {}).get("predicted_n") if "recorded" in e else None),
                     "status": r.get("status"), "error": r.get("error")})
    summ = summarize(rows, T1 - T0, sum(d for _, _, d in sched))
    warn = []
    for r in rows:
        if r["prompt_n"] is None and not r["error"]:
            warn.append(f"request {r['i']}: no timings in the response (not llama-server, or an old one): prompt and decode columns are empty")
        elif r["recorded_prompt_n"] is not None and r["prompt_n"] is not None and r["prompt_n"] > 2 * r["recorded_prompt_n"] + 256:
            warn.append(f"request {r['i']}: {r['prompt_n']} prompt tokens processed, {r['recorded_prompt_n']} in the recording "
                        "(the server reused less: other slot, other template, or the re-sent assistant message differs)")
        if r["recorded_predicted_n"] and r["predicted_n"] is not None and r["predicted_n"] < r["recorded_predicted_n"]:
            warn.append(f"request {r['i']}: generated {r['predicted_n']} of the recorded {r['recorded_predicted_n']} tokens (the model stopped earlier)")
    summ["warnings"] = warn
    for w in warn:
        print("  warning: " + w, flush=True)
    out = {"session": a.session, "name": s.get("name"), "label": a.label, "url": a.url, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": summ, "rows": rows}
    print_summary(out)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 1 if any(r["error"] or (r["status"] or 0) >= 400 for r in rows) else 0


def summarize(rows, total, pauses):
    comp = [r for r in rows if r["ttft"] is not None]
    fu = [r["ttft"] for r in comp[1:]]
    big = max(comp, key=lambda r: (r["prompt_n"] or 0), default=None)
    return {"requests": len(rows), "task_s": round(total, 2), "pauses_s": round(pauses, 2), "first_ttft_s": comp[0]["ttft"] if comp else None,
            "followup_ttft_median_s": round(statistics.median(fu), 3) if fu else None, "followup_ttft_mean_s": round(statistics.mean(fu), 3) if fu else None,
            "followup_ttft_sum_s": round(sum(fu), 2) if fu else None,
            "prompt_processed": sum(r["prompt_n"] or 0 for r in rows), "prompt_reused": sum(r["cache_n"] or 0 for r in rows),
            "generated": sum(r["predicted_n"] or 0 for r in rows),
            "largest_prompt": {"i": big["i"], "processed": big["prompt_n"], "ttft_s": big["ttft"]} if big else None,
            "decode_tps_median": round(statistics.median([r["predicted_per_second"] for r in rows if r["predicted_per_second"] and (r["predicted_n"] or 0) >= 16]), 2)
            if any(r["predicted_per_second"] and (r["predicted_n"] or 0) >= 16 for r in rows) else None,
            "errors": sum(1 for r in rows if r["error"] or (r["status"] or 0) >= 400)}


def print_summary(out):
    s = out["summary"]
    print(f"\n{out.get('label') or ''} {out.get('name') or ''}".strip())
    print("| # | at (s) | TTFT (s) | prompt processed | reused | generated | decode tok/s | request (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in out["rows"]:
        if r["ttft"] is None and r["prompt_n"] is None:
            continue
        ttft = "-" if r["ttft"] is None else f"{r['ttft']:.2f}"
        tps = "-" if not r["predicted_per_second"] else f"{r['predicted_per_second']:.1f}"
        print(f"| {r['i']} | {r['at']:.1f} | {ttft} | {r['prompt_n']} | {r['cache_n']} | {r['predicted_n']} | {tps} | {r['total']:.1f} |")
    print(f"task {s['task_s']} s (of which {s['pauses_s']} s recorded agent pauses); first TTFT {s['first_ttft_s']} s; follow-up TTFT median "
          f"{s['followup_ttft_median_s']} s, sum {s['followup_ttft_sum_s']} s; prompt {s['prompt_processed']} processed / {s['prompt_reused']} reused; "
          f"{s['generated']} generated, decode median {s['decode_tps_median']} tok/s; errors {s['errors']}")


def show(a):
    with open(a.file, encoding="utf-8") as f:
        s = json.load(f)
    if "summary" in s:
        print_summary(s); return
    reqs = s["requests"]
    print(f"{a.file}: {s.get('name')}  recorded {s.get('recorded')}  {len(reqs)} requests")
    sched = _schedule([e for e in reqs])
    for i, (e, (dep, kind, d)) in enumerate(zip(reqs, sched)):
        tm = (e.get("recorded") or {}).get("timings") or {}
        msgs = (e.get("body") or {}).get("messages") or []
        print(f"  {i + 1:3d} {e['method']} {e['path']:24s} {e['t_start']:7.1f}-{e['t_end']:7.1f} s  after {'-' if dep is None else f'{kind} of {dep + 1}'} +{d:.1f} s"
              + (f"  {len(msgs)} messages, {len((e.get('body') or {}).get('tools') or [])} tools, prompt {tm.get('prompt_n')} new + {tm.get('cache_n')} reused, "
                 f"{tm.get('predicted_n')} generated" if tm else ""))


def table(a):
    """One row per saved replay result: label, session, first-turn TTFT, follow-up TTFT, task time."""
    print("| run | session | first TTFT (s) | follow-up TTFT median / sum (s) | task (s) | prompt processed / reused | decode median (tok/s) |")
    print("|---|---|---|---|---|---|---|")
    groups = {}
    for fn in a.files:
        with open(fn, encoding="utf-8") as f:
            o = json.load(f)
        s = o["summary"]
        sess = (o.get("name") or "").split(":")[0]
        print(f"| {o.get('label') or fn} | {sess} | {s['first_ttft_s']:.1f} | {s['followup_ttft_median_s']:.1f} / "
              f"{s['followup_ttft_sum_s']:.1f} | {s['task_s']:.1f} | {s['prompt_processed']} / {s['prompt_reused']} | {s['decode_tps_median']} |")
        groups.setdefault((sess, (o.get("label") or fn).split()[0]), []).append(s)
    # several runs per side (label "stock run 1", "mc run 2", ...): min and median of each, so one slow run does not decide
    if any(len(v) > 1 for v in groups.values()):
        print("\n| session | side | runs | first TTFT min / median (s) | follow-up TTFT median, min / median (s) | task min / median (s) |")
        print("|---|---|---|---|---|---|")
        for (sess, side), v in groups.items():
            mm = lambda k: f"{min(x[k] for x in v):.1f} / {statistics.median(x[k] for x in v):.1f}"
            print(f"| {sess} | {side} | {len(v)} | {mm('first_ttft_s')} | {mm('followup_ttft_median_s')} | {mm('task_s')} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("record", help="recording proxy")
    r.add_argument("--upstream", required=True, help="the real server, e.g. http://127.0.0.1:8080")
    r.add_argument("--port", type=int, default=8199)
    r.add_argument("--out", required=True)
    r.add_argument("--name", default="")
    r.add_argument("--model", default="", help="note: the model the session was recorded with")
    r.add_argument("--notes", default="")
    r.add_argument("--idle", type=float, default=0, help="stop after this many seconds without a request (0 = only Ctrl+C / --stop-file)")
    r.add_argument("--stop-file", default="", help="stop when this file exists")
    r.add_argument("--no-scrub", action="store_true", help="keep the home folder, user and host names in the recording")
    p = sp.add_parser("replay", help="replay a recorded session against a server")
    p.add_argument("session")
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--label", default="")
    p.add_argument("--json", default="", help="write the per-request rows and the summary here")
    p.add_argument("--model", default="", help="model name to send (default: as recorded; llama-server ignores it with one model)")
    p.add_argument("--api-key", default="")
    p.add_argument("--gap-scale", type=float, default=1.0, help="scale the recorded pauses (0 = none)")
    p.add_argument("--no-cap", action="store_true", help="do not cap generation at the recorded number of tokens")
    p.add_argument("--all-requests", action="store_true", help="also replay non-completion requests (GET /v1/models ...)")
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--wait-health", type=int, default=0, help="wait up to N seconds for /health first")
    w = sp.add_parser("show", help="print a recorded session or a replay result")
    w.add_argument("file")
    t = sp.add_parser("table", help="saved replay results (--json files) side by side")
    t.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "record":
        record(a)
    elif a.cmd == "replay":
        sys.exit(replay(a))
    elif a.cmd == "table":
        table(a)
    else:
        show(a)


if __name__ == "__main__":
    main()
