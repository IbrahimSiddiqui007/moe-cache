#!/usr/bin/env python3
"""`moe_cache.py connect NAME --check`: proves that an agent's setup works end to end, the way that agent talks to the server.

Five steps, each timed:  config (URL, key, model, window as the agent's own config file holds them, or as `connect --write` would write them) ->
models (GET /v1/models answers, lists the model) -> chat (one short streamed request in the agent's protocol returns text) ->
tool call (one request with one tool returns a well-formed call; the tool result is sent back and the model answers) -> window (the agent's
window fits the server's context). The first failure ends in a one-line verdict with the most likely fix. Standard library only; it never
starts or stops a server and writes nothing.
"""
import http.client, json, os, re, sys, time, urllib.parse

PROTOCOL = {"hermes": "chat", "opencode": "chat", "pi": "chat", "openclaw": "chat", "aider": "chat", "qwen": "chat", "codex": "responses", "claude": "messages"}
NO_TOOLS = ("aider",)               # Aider edits through text, it sends no tool definitions
MIN_WINDOW = {"hermes": 64000}      # Hermes Agent refuses a smaller context_length
QWEN_KEY_VAR = "MOE_CACHE_API_KEY"
TOOL = {"name": "read_file", "description": "Read a text file and return its content.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path of the file"}}, "required": ["path"]}}
ASK = "Use the read_file tool to read the file notes.txt, then tell me in one sentence what it says."
RESULT = "The meeting is at 3 pm in room 4."
START = "start one: `python3 bin/moe_cache.py server MODEL.gguf --agent` (or `python3 bin/moe_cache.py on-demand` for your picked model)"


# ---- reading the agent's config ----

def _json(path):
    try:
        data = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{path} cannot be read ({e.__class__.__name__}): rerun connect --write, or fix the file")
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _clean_key(k):
    return None if k in (None, "", "none") else str(k)


def _root(url):
    """Server root from a base URL with or without /v1."""
    url = (url or "").rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def _strip(model, prefix):
    return model[len(prefix):] if isinstance(model, str) and model.startswith(prefix) else model


def config_path(agent, given):
    home, env = os.path.expanduser("~"), os.environ
    if given:
        return given
    if agent == "codex":
        return os.path.join(env.get("CODEX_HOME") or os.path.join(home, ".codex"), "config.toml")
    if agent == "pi":
        return os.path.join(env.get("PI_CODING_AGENT_DIR") or os.path.join(home, ".pi", "agent"), "models.json")
    if agent == "openclaw":
        return os.path.join(home, ".openclaw", "openclaw.json")
    if agent == "qwen":
        return os.path.join(home, ".qwen", "settings.json")
    if agent == "aider":
        return os.path.join(home, ".aider.moe-cache.conf.yml")
    if agent == "claude":
        return os.path.join(os.getcwd(), ".claude", "settings.local.json")
    if agent == "hermes":
        h = env.get("HERMES_HOME") or (os.path.join(env.get("LOCALAPPDATA") or home, "hermes") if os.name == "nt" else os.path.join(home, ".hermes"))
        return os.path.join(h, "config.yaml")
    d = os.path.join(env.get("XDG_CONFIG_HOME") or os.path.join(home, ".config"), "opencode")
    jc = os.path.join(d, "opencode.jsonc")
    if os.path.isfile(jc):
        try:
            json.load(open(jc, encoding="utf-8"))
            return jc
        except (OSError, ValueError):
            pass
    return os.path.join(d, "opencode.json")


def _yaml_uncomment(v):
    """A YAML value without its trailing `# comment` (a # inside quotes, or not after a space, stays)."""
    q = None
    for i, ch in enumerate(v):
        if q:
            if ch == "\\" and q == '"':
                continue
            if ch == q:
                q = None
        elif ch in "\"'" and (i == 0 or v[i - 1] in " \t[{,:"):
            q = ch
        elif ch == "#" and (i == 0 or v[i - 1] in " \t"):
            return v[:i].rstrip()
    return v.strip()


def _yaml_flow(v):
    """'{a: 1, b: [x, y]}' -> dict (best effort, scalars as text/number); None when v is not a flow mapping."""
    v = v.strip()
    if not (v.startswith("{") and v.endswith("}")):
        return None
    out, depth, cur, q, parts = {}, 0, "", None, []
    for ch in v[1:-1]:
        if q:
            q = None if ch == q else q
        elif ch in "\"'":
            q = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        cur += ch
    parts.append(cur)
    for part in parts:
        k, sep, val = part.partition(":")
        if sep and k.strip():
            out[k.strip().strip("\"'")] = _yaml_scalar(val)
    return out


def _yaml_scalar(v):
    """One YAML value as Python: quotes removed, comment dropped, numbers read; a flow list/mapping comes back as its text."""
    v = _yaml_uncomment(v)
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        try:
            return json.loads(v) if v[0] == '"' else v[1:-1].replace("''", "'")
        except ValueError:
            return v[1:-1]
    try:
        return json.loads(v)
    except ValueError:
        return v


def read_config(agent, cfg):
    """{"url","key","model","window"} as the agent's config holds them, None when the file or our entry in it is missing; ValueError when unreadable."""
    if not os.path.isfile(cfg):
        return None
    if agent == "codex":
        text = open(cfg, encoding="utf-8").read()
        m = re.search(r"^\[\s*model_providers\s*\.\s*\"?moe-cache\"?\s*\](.*?)(?=^\s*\[|\Z)", text, re.M | re.S)
        if not m:
            return None
        top = re.split(r"^\s*\[", text, 1, re.M)[0]

        def val(body, k):
            mm = re.search(r"^%s\s*=\s*(\"(?:[^\"\\]|\\.)*\"|[0-9]+)" % k, body, re.M)
            return json.loads(mm.group(1)) if mm else None
        return {"url": val(m.group(1), "base_url"), "key": _clean_key(val(m.group(1), "experimental_bearer_token")), "model": val(top, "model"),
                "window": val(top, "model_context_window")}
    if agent == "aider":
        kv = {}
        for ln in open(cfg, encoding="utf-8").read().splitlines():
            mm = re.match(r"^([\w-]+):(?:\s+(.*))?$", ln)
            if mm:
                kv[mm.group(1)] = _yaml_scalar(mm.group(2) or "")
        if "openai-api-base" not in kv:
            return None
        model, window = _strip(kv.get("model"), "openai/"), None
        meta = kv.get("model-metadata-file")
        if meta and os.path.isfile(meta):
            try:
                window = _json(meta).get(f"openai/{model}", {}).get("max_input_tokens")
            except ValueError:
                pass
        return {"url": kv["openai-api-base"], "key": _clean_key(kv.get("openai-api-key")), "model": model, "window": window}
    if agent == "hermes":
        kv, inside = {}, False
        for ln in open(cfg, encoding="utf-8").read().splitlines():
            fm = re.match(r"^model:\s*(\{.*)$", ln)
            if fm:
                kv.update({k: str(v) for k, v in (_yaml_flow(_yaml_uncomment(fm.group(1))) or {}).items()})
                inside = False
            elif re.match(r"^model:\s*(#.*)?$", ln):
                inside = True
            elif inside and ln and not ln[0].isspace() and not ln.startswith("#"):
                inside = False
            elif inside:
                mm = re.match(r"^\s+([\w.]+):(?:\s+(.*?))?\s*$", ln)
                if mm:
                    kv[mm.group(1)] = str(_yaml_scalar(mm.group(2) or ""))
        if kv.get("provider") != "custom" or not kv.get("base_url"):
            return None
        return {"url": kv["base_url"], "key": _clean_key(kv.get("api_key")), "model": kv.get("default"),
                "window": int(kv["context_length"]) if kv.get("context_length", "").isdigit() else None}
    data = _json(cfg)
    if agent == "claude":
        env = data.get("env") if isinstance(data.get("env"), dict) else {}
        if not env.get("ANTHROPIC_BASE_URL"):
            return None
        return {"url": env["ANTHROPIC_BASE_URL"], "key": _clean_key(env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_API_KEY")),
                "model": env.get("ANTHROPIC_MODEL"), "window": None}
    if agent == "opencode":
        prov = (data.get("provider") or {}).get("moe-cache")
        if not isinstance(prov, dict):
            return None
        model = _strip(data.get("model"), "moe-cache/") if str(data.get("model", "")).startswith("moe-cache/") else next(iter(prov.get("models") or {}), None)
        opts = prov.get("options") or {}
        return {"url": opts.get("baseURL"), "key": _clean_key(opts.get("apiKey")), "model": model,
                "window": (((prov.get("models") or {}).get(model) or {}).get("limit") or {}).get("context")}
    if agent in ("pi", "openclaw"):
        if agent == "pi":
            prov = (data.get("providers") or {}).get("moe-cache")
            sd = os.path.join(os.path.dirname(cfg), "settings.json")
            model = None
            try:
                s = _json(sd) if os.path.isfile(sd) else {}
                model = s.get("defaultModel") if s.get("defaultProvider") == "moe-cache" else None
            except ValueError:
                pass
        else:
            prov = ((data.get("models") or {}).get("providers") or {}).get("moe-cache")
            prim = (((data.get("agents") or {}).get("defaults") or {}).get("model"))
            prim = prim.get("primary") if isinstance(prim, dict) else prim
            model = _strip(prim, "moe-cache/") if isinstance(prim, str) and prim.startswith("moe-cache/") else None
        if not isinstance(prov, dict):
            return None
        rows = [r for r in prov.get("models") or [] if isinstance(r, dict)]
        model = model or (rows[0].get("id") if rows else None)
        row = next((r for r in rows if r.get("id") == model), {})
        return {"url": prov.get("baseUrl"), "key": _clean_key(prov.get("apiKey")), "model": model, "window": row.get("contextWindow")}
    ents = [e for e in ((data.get("modelProviders") or {}).get("openai") or []) if isinstance(e, dict) and e.get("envKey") == QWEN_KEY_VAR]      # qwen
    if not ents:
        return None
    model = (data.get("model") or {}).get("name") or ents[0].get("id")
    ent = next((e for e in ents if e.get("id") == model), ents[0])
    return {"url": ent.get("baseUrl"), "key": _clean_key((data.get("env") or {}).get(QWEN_KEY_VAR)), "model": model,
            "window": (ent.get("generationConfig") or {}).get("contextWindowSize")}


# ---- talking to the server ----

class Reply:
    def __init__(self):
        self.status, self.error, self.text, self.reasoning, self.calls = 0, "", "", "", []


def _conn(root, timeout):
    u = urllib.parse.urlparse(root)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout), u.path.rstrip("/")


def get(root, path, key, timeout=10):
    """(status, text); status 0 when the server cannot be reached. Redirects are not followed (they would carry the key along)."""
    try:
        c, base = _conn(root, timeout)
        c.request("GET", base + path, headers=({"Authorization": "Bearer " + key} if key else {}))
        r = c.getresponse()
        return r.status, r.read().decode(errors="replace")
    except Exception as e:
        return 0, str(e)


def _sse(resp):
    for raw in resp:
        ln = raw.decode(errors="replace").strip()
        if ln.startswith("data:"):
            d = ln[5:].strip()
            if d and d != "[DONE]":
                try:
                    yield json.loads(d)
                except ValueError:
                    continue


def stream(root, proto, key, body, timeout):
    """One streamed request in the agent's protocol -> Reply (text, reasoning, calls[{id,name,args}], or status/error)."""
    r = Reply()
    path = {"chat": "/v1/chat/completions", "responses": "/v1/responses", "messages": "/v1/messages"}[proto]
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if key:
        h["Authorization"] = "Bearer " + key
    if proto == "messages":
        h["anthropic-version"] = "2023-06-01"
    try:
        c, base = _conn(root, timeout)
        c.request("POST", base + path, json.dumps(dict(body, stream=True)), h)
        resp = c.getresponse()
        r.status = resp.status
        if resp.status != 200:
            r.error = resp.read().decode(errors="replace")[:400]
            return r
        calls = {}
        for ev in _sse(resp):
            if not isinstance(ev, dict):
                continue
            if ev.get("error") or ev.get("type") in ("error", "response.failed"):
                r.error = json.dumps(ev.get("error") or ev)[:400]
                break
            t = ev.get("type", "")
            if proto == "chat":
                ch = (ev.get("choices") or [{}])[0].get("delta") or {}
                r.text += ch.get("content") or ""
                r.reasoning += ch.get("reasoning_content") or ""
                for tc in ch.get("tool_calls") or []:
                    c0 = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                    c0["id"] = tc.get("id") or c0["id"]
                    f = tc.get("function") or {}
                    c0["name"] += f.get("name") or ""
                    c0["args"] += f.get("arguments") or ""
            elif proto == "responses":
                if t == "response.output_text.delta":
                    r.text += ev.get("delta") or ""
                elif t in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                    r.reasoning += ev.get("delta") or ""
                elif t in ("response.output_item.added", "response.output_item.done") and (ev.get("item") or {}).get("type") == "function_call":
                    it = ev["item"]
                    c0 = calls.setdefault(it.get("id") or ev.get("output_index", 0), {"id": "", "name": "", "args": ""})
                    c0["id"], c0["name"] = it.get("call_id") or it.get("id") or c0["id"], it.get("name") or c0["name"]
                    if t.endswith("done") or it.get("arguments"):
                        c0["args"] = it.get("arguments") or c0["args"]
                elif t == "response.function_call_arguments.delta":
                    c0 = calls.setdefault(ev.get("item_id", 0), {"id": "", "name": "", "args": ""})
                    c0["args"] += ev.get("delta") or ""
                elif t == "response.function_call_arguments.done" and ev.get("arguments"):
                    calls.setdefault(ev.get("item_id", 0), {"id": "", "name": "", "args": ""})["args"] = ev["arguments"]
            else:
                if t == "content_block_start" and (ev.get("content_block") or {}).get("type") == "tool_use":
                    cb = ev["content_block"]
                    calls[ev.get("index", 0)] = {"id": cb.get("id", ""), "name": cb.get("name", ""), "args": ""}
                elif t == "content_block_delta":
                    d = ev.get("delta") or {}
                    if d.get("type") == "text_delta":
                        r.text += d.get("text") or ""
                    elif d.get("type") == "thinking_delta":
                        r.reasoning += d.get("thinking") or ""
                    elif d.get("type") == "input_json_delta":
                        calls.setdefault(ev.get("index", 0), {"id": "", "name": "", "args": ""})["args"] += d.get("partial_json") or ""
        r.calls = list(calls.values())
    except Exception as e:
        r.status, r.error = r.status or 0, f"{e.__class__.__name__}: {e}"
    return r


def request_body(proto, model, history, tools, max_tokens):
    """The agent's request: `history` is a list of ("user", text) | ("call", id, name, args) | ("result", id, text) | ("assistant", text)."""
    if proto == "chat":
        msgs = []
        for h in history:
            if h[0] in ("user", "assistant"):
                msgs.append({"role": h[0], "content": h[1]})
            elif h[0] == "call":
                msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": h[1], "type": "function", "function": {"name": h[2], "arguments": h[3]}}]})
            else:
                msgs.append({"role": "tool", "tool_call_id": h[1], "content": h[2]})
        b = {"model": model, "messages": msgs, "max_tokens": max_tokens}
        if tools:
            b["tools"] = [{"type": "function", "function": TOOL}]
        return b
    if proto == "responses":
        items = []
        for h in history:
            if h[0] == "user":
                items.append({"role": "user", "content": [{"type": "input_text", "text": h[1]}]})
            elif h[0] == "call":
                items.append({"type": "function_call", "call_id": h[1], "name": h[2], "arguments": h[3]})
            elif h[0] == "result":
                items.append({"type": "function_call_output", "call_id": h[1], "output": h[2]})
        b = {"model": model, "input": items, "max_output_tokens": max_tokens}
        if tools:
            b["tools"] = [dict(TOOL, type="function")]
        return b
    msgs = []
    for h in history:
        if h[0] == "user":
            msgs.append({"role": "user", "content": h[1]})
        elif h[0] == "call":
            msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": h[1], "name": h[2], "input": json.loads(h[3])}]})
        elif h[0] == "result":
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": h[1], "content": h[2]}]})
    b = {"model": model, "messages": msgs, "max_tokens": max_tokens}
    if tools:
        b["tools"] = [{"name": TOOL["name"], "description": TOOL["description"], "input_schema": TOOL["parameters"]}]
    return b


# ---- the check ----

def http_fix(status, err, what, agent):
    """(detail, fix) for a failed request."""
    low = err.lower()
    if status == 200:
        return f"the stream died with an error: {err[:140]}", "the server failed while answering: see its log for the reason"
    if status == 0:
        return f"no answer ({err[:120]})", f"the connection broke: is the server still running? {START}"
    if status in (401, 403):
        return f"HTTP {status}: key rejected", f"the key is wrong: run `python3 bin/moe_cache.py connect {agent} --write` again (it writes the current key)"
    if status == 404:
        return f"HTTP 404 on {what}", f"this server has no {what}: update llama.cpp (agent mode needs a recent build); start it with --agent"
    if "jinja" in low or ("tool" in low and status in (400, 422, 500)):
        return f"HTTP {status}: {err[:140]}", "the server is not in agent mode: restart it with `--agent` (it adds --jinja, which tool calls need)"
    if "context" in low or "exceed" in low or "too large" in low:
        return f"HTTP {status}: {err[:140]}", "the request is larger than the server's context: restart the server with a bigger --ctx"
    if "model" in low and ("not found" in low or "unknown" in low or "does not exist" in low):
        return f"HTTP {status}: {err[:140]}", f"the model name is wrong: run `python3 bin/moe_cache.py connect {agent} --write` again"
    return f"HTTP {status}: {err[:140]}", "see the server's log for the reason"


def check_agent(agent, url_given=None, config=None, as_json=False, timeout=180, default_url="http://127.0.0.1:8080", fallback_key=None, window_fn=None):
    """Run the five steps; print the table (or JSON); 0 when nothing failed, else 1."""
    proto = PROTOCOL[agent]
    steps, fix, t_all = [], [], [None]
    cfg_file = config_path(agent, config)
    ctx = {"url": None, "key": None, "model": None, "window": None, "models": [], "server_ctx": None, "written": False}

    def add(name, ok, t0, detail, fixtext=None):
        steps.append({"step": name, "result": "pass" if ok is True else "fail" if ok is False else "skip", "ms": int((time.time() - t0) * 1000), "detail": detail})
        if ok is False and not fix:
            fix.append(fixtext or "")
        return ok

    # 1 config
    t0 = time.time()
    try:
        conf = read_config(agent, cfg_file)
    except ValueError as e:
        conf = None
        add("config", False, t0, str(e), f"rerun `python3 bin/moe_cache.py connect {agent} --write` (it repairs or replaces the entry), or fix the file by hand")
    if conf:
        ctx["written"] = True
        ctx.update(url=_root(conf["url"]) if conf.get("url") else None, key=conf["key"], model=conf["model"], window=conf["window"])
        if not ctx["url"]:
            add("config", False, t0, f"{cfg_file} has no server URL", f"rerun `python3 bin/moe_cache.py connect {agent} --write`")
        else:
            add("config", True, t0, f"{cfg_file}: {ctx['url']}, model {ctx['model'] or '(none)'}, window {ctx['window'] or '(none set)'}, key {'set' if ctx['key'] else 'none'}")
    elif not steps:
        add("config", None, t0, f"no moe-cache entry in {cfg_file} yet: checking what `connect {agent} --write` would write")
    if url_given and conf is None:
        ctx["url"] = _root(url_given)
    ctx["url"] = ctx["url"] or _root(url_given or default_url)
    if not conf:
        ctx["key"] = fallback_key(ctx["url"]) if fallback_key else None
    root, key = ctx["url"], ctx["key"]
    cfg_ok = not any(s["result"] == "fail" for s in steps)

    def need(*names):
        return any(s["step"] in names and s["result"] != "pass" for s in steps)

    # 2 models
    t0 = time.time()
    if not cfg_ok:
        add("models", None, t0, "needs a readable config")
    else:
        st, body = get(root, "/v1/models", key)
        ids = []
        if st == 200:
            try:
                ids = [m["id"] for m in json.loads(body)["data"]]
            except (ValueError, KeyError, TypeError):
                pass
        ctx["models"] = ids
        if st == 0:
            add("models", False, t0, f"nothing answers at {root}", f"no server at {root}: {START}")
        elif st != 200:
            d, f = http_fix(st, body, "/v1/models", agent)
            if st in (401, 403) and fallback_key:
                alt = fallback_key(root) if not conf else None
                good = key != alt and alt and get(root, "/v1/models", alt)[0] == 200
                f = (f"the key in {cfg_file} is rejected: run `python3 bin/moe_cache.py connect {agent} --write` (writes the current key)" if good or conf
                     else f"the server wants an API key: `python3 bin/moe_cache.py key` prints it, pass it with --api-key")
            add("models", False, t0, d, f)
        elif not ids:
            add("models", False, t0, "/v1/models lists no model", "the server has no model loaded: see its log")
        else:
            model = ctx["model"] or ids[0]
            ctx["model"] = model
            if model not in ids:
                add("models", False, t0, f"model '{model}' is not served; the server lists: {', '.join(ids[:4])}",
                    f"the model name does not match: run `python3 bin/moe_cache.py connect {agent} --write` (writes '{ids[0]}')")
            else:
                add("models", True, t0, f"{', '.join(ids[:3])}" + (" ..." if len(ids) > 3 else ""))
    model, ready = ctx["model"], not need("config", "models")

    # 3 chat
    t0 = time.time()
    if not ready:
        add("chat", None, t0, "needs steps 1-2")
    else:
        r = stream(root, proto, key, request_body(proto, model, [("user", "Say hello in three words.")], False, 64), timeout)
        if r.status != 200 and r.status or r.error:
            d, f = http_fix(r.status, r.error, {"chat": "/v1/chat/completions", "responses": "/v1/responses", "messages": "/v1/messages"}[proto], agent)
            add("chat", False, t0, d, f)
        elif r.text.strip():
            add("chat", True, t0, f"{proto}: \"{r.text.strip()[:50]}\"")
        elif r.reasoning.strip():
            add("chat", True, t0, f"{proto}: reasoning only within 64 tokens (a thinking model; fine)")
        else:
            add("chat", False, t0, f"{proto}: the stream ended without text", "the reply was empty: see the server's log; the model may not fit this server's chat template")

    # 4 tool call
    t0 = time.time()
    if agent in NO_TOOLS:
        add("tool call", None, t0, "skipped: Aider sends no tool calls, text only is checked")
    elif need("chat") or not ready:
        add("tool call", None, t0, "needs steps 1-3")
    else:
        hist = [("user", ASK)]
        r = stream(root, proto, key, request_body(proto, model, hist, True, 512), timeout)
        if (r.status and r.status != 200) or r.error:
            d, f = http_fix(r.status, r.error, "tool calls", agent)
            add("tool call", False, t0, d, f)
        elif not r.calls:
            add("tool call", False, t0, f"no tool call; the model wrote: \"{(r.text or r.reasoning).strip()[:60]}\"",
                "the model answered in text instead of calling the tool: start the server with `--agent` (adds --jinja) and use a model with tool calling")
        else:
            c = r.calls[0]
            try:
                args = json.loads(c["args"] or "{}")
                bad = None if isinstance(args, dict) else "arguments are not a JSON object"
                bad = bad or (None if c["name"] == TOOL["name"] else f"unknown tool '{c['name']}'")
                bad = bad or (None if isinstance(args.get("path"), str) and args["path"] else "required argument 'path' is missing")
            except ValueError:
                bad = f"arguments are not valid JSON: {c['args'][:50]!r}"
            if bad:
                add("tool call", False, t0, "malformed tool call: " + bad,
                    "the model returned a broken tool call: use a model with tool-call training, or a newer llama.cpp (its parser for this model); agents will loop or stall on this")
            else:
                cid = c["id"] or "call_check1"
                hist += [("call", cid, c["name"], json.dumps(args)), ("result", cid, RESULT)]
                r2 = stream(root, proto, key, request_body(proto, model, hist, True, 512), timeout)
                if (r2.status and r2.status != 200) or r2.error:
                    d, f = http_fix(r2.status, r2.error, "tool results", agent)
                    add("tool call", False, t0, "call ok, but sending the result back failed: " + d, f)
                elif r2.text.strip():
                    add("tool call", True, t0, f"read_file({json.dumps(args)[:40]}) -> answer \"{r2.text.strip()[:40]}\"")
                else:
                    add("tool call", False, t0, "call ok, but no answer after the tool result" + (" (it called the tool again)" if r2.calls else ""),
                        "the model does not continue after a tool result: check the chat template (agent mode / --jinja) and try another model")

    # 4b system messages (AGENTS-992): a client may send a system message after the first turn (seen through Claude Code); some chat templates (Qwen3.8's) raise on them and every request answers HTTP 500
    if agent == "claude" and not need("chat") and ready:
        t0 = time.time()
        body = {"model": model, "max_tokens": 16, "messages": [
            {"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."},
            {"role": "system", "content": "Reminder: answer briefly."}, {"role": "user", "content": "Say hello in three words."}]}
        r = stream(root, "chat", key, body, timeout)
        r.error = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", r.error or "")      # text from the server goes to the terminal: no control characters
        low = r.error.lower()
        if r.status == 404:
            add("system messages", None, t0, "skipped: no /v1/chat/completions to test a system message in the middle of a conversation")
        elif (r.status and r.status != 200) or r.error:
            if "system message" in low or "beginning" in low:
                add("system messages", False, t0, f"HTTP {r.status}: ...{r.error[-100:]}",
                    "this model's chat template refuses a system message in the middle of a conversation (Claude Code produced one): start the server with moe-cache-server (it uses a "
                    "copy of the template fixed by one line for Qwen3.8), or pass --chat-template-file FIXED.jinja after the --")
            else:
                d, f = http_fix(r.status, r.error, "system messages", agent)
                add("system messages", False, t0, d, f)
        else:
            add("system messages", True, t0, "a system message in the middle of a conversation is accepted")

    # 5 window
    t0 = time.time()
    if need("config", "models"):
        add("window", None, t0, "needs steps 1-2")
    else:
        st, body = get(root, "/props", key)
        sctx = None
        if st == 200:
            try:
                sctx = int(json.loads(body)["default_generation_settings"]["n_ctx"])
            except (ValueError, KeyError, TypeError):
                pass
        win = ctx["window"] if conf else (window_fn(sctx) if window_fn and sctx else None)
        need_min = MIN_WINDOW.get(agent)
        if sctx is None:
            add("window", None, t0, "the server does not report its context (/props): not checked")
        elif agent == "claude":
            ok = sctx >= 64000
            add("window", ok, t0, f"server context {sctx}" + ("" if ok else " (Claude Code's own prompt is long)"),
                "the server's context is too small for Claude Code: restart with `--agent` or --ctx 65536 or more")
        elif win is None:
            add("window", None, t0, f"no window in the config; server context {sctx}")
        elif win > sctx:
            add("window", False, t0, f"agent window {win} > server context {sctx}",
                f"the window is bigger than the server's: run `python3 bin/moe_cache.py connect {agent} --write` (sets {window_fn(sctx) if window_fn else sctx}), or restart the server with a bigger --ctx")
        elif need_min and (win < need_min or sctx < need_min):
            add("window", False, t0, f"window {win}, server context {sctx}: {agent} needs at least {need_min}",
                "restart the server with `--agent` or --ctx 65536, then connect again")
        else:
            add("window", True, t0, f"agent window {win} <= server context {sctx}" + ("" if conf else " (would be written)"))

    ok = not any(s["result"] == "fail" for s in steps)
    if ok:
        verdict = (f"{agent} will work against {root} (model {model})" + ("" if ctx["written"] else f"; its config is not written yet: run `python3 bin/moe_cache.py connect {agent} --write`")
                   + ("; tool calls were not tested (Aider uses none)" if agent in NO_TOOLS else ""))
    else:
        verdict = "FAIL: " + (fix[0] or "see the failed step")
    out = {"agent": agent, "ok": ok, "url": root, "model": model, "protocol": proto, "config_file": cfg_file, "config_written": ctx["written"],
           "steps": steps, "verdict": verdict}
    if as_json:
        print(json.dumps(out, indent=2), flush=True)
    else:
        rows = [(s["step"], s["result"], f"{s['ms'] / 1000:.1f} s", s["detail"]) for s in steps]
        w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(("step", "result", "time"))]
        print(f"moe-cache connect {agent} --check   ({proto} protocol, {root})")
        print(f"  {'step':<{w[0]}}  {'result':<{w[1]}}  {'time':>{w[2]}}  detail")
        for r in rows:
            print(f"  {r[0]:<{w[0]}}  {r[1]:<{w[1]}}  {r[2]:>{w[2]}}  {r[3]}")
        print(("PASS: " if ok else "") + verdict, flush=True)
    return 0 if ok else 1
