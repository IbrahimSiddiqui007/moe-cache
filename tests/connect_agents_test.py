#!/usr/bin/env python3
"""`moe_cache.py connect codex|pi|openclaw|aider|qwen`: what is printed, what --write merges, the first backup kept, malformed files refused.
No model, no server (server_info / server_models are replaced), temporary HOME; standard library only.
   python3 tests/connect_agents_test.py"""
import contextlib, io, json, os, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-connect-test-")
HOME = os.path.join(TMP, "home"); os.makedirs(HOME)
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
for v in ("CODEX_HOME", "PI_CODING_AGENT_DIR", "LLAMA_API_KEY"):
    os.environ.pop(v, None)
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc  # noqa: E402

try:
    import tomllib
except ImportError:
    tomllib = None
KEY = "k3y-abc123"
mc.server_info = lambda url, key=None: ("fake-moe", 65536)
mc.server_models = lambda url, key=None: ["fake-moe", "second"]
mc.key_for = lambda url, given=None: given or KEY
fails = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)


def run(*argv):
    """(exit code or message, stdout, stderr) of connect(argv)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = mc.connect(list(argv))
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


def put(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)


def mode(path):
    return os.stat(path).st_mode & 0o777


check(mc.agent_window(65536) == 57344 and mc.agent_window(131072) == 114688 and mc.agent_window(8192) == 7168, "window = server context minus 12.5 % headroom, whole 1024s")

# ---- Codex CLI ----
cx = os.path.join(HOME, ".codex", "config.toml")
rc, out, _ = run("codex", "--url", "http://127.0.0.1:8080")
check(rc == 0 and 'wire_api = "responses"' in out and "model_context_window = 57344" in out and KEY not in out and not os.path.exists(cx), "codex: preview prints the settings, hides the key, writes nothing")
rc, out, _ = run("codex", "--write")
txt = open(cx).read()
check(rc == 0 and mode(cx) == 0o600 and f'experimental_bearer_token = "{KEY}"' in txt and 'web_search = "disabled"' in txt, "codex: missing file -> written, private, key inside")
if tomllib:
    d = tomllib.loads(txt)
    check(d["model_provider"] == "moe-cache" and d["model"] == "fake-moe" and d["model_providers"]["moe-cache"]["base_url"] == "http://127.0.0.1:8080/v1", "codex: the file parses and holds provider, model, base URL")
check(not os.path.exists(cx + ".bak"), "codex: no backup when there was no file")
put(cx, 'model = "gpt-x"\nmodel_provider = "openai"\n\n[model_providers.mine]\nname = "mine"\nbase_url = "http://h/v1"\n\n[projects."/a"]\ntrust_level = "trusted"\n')
orig = open(cx).read()
rc, _, _ = run("codex", "--write")
txt = open(cx).read()
check(rc == 0 and open(cx + ".bak").read() == orig, "codex: existing file -> first backup is the original")
check("#moe-cache-off# model = \"gpt-x\"" in txt and "[model_providers.mine]" in txt and '[projects."/a"]' in txt, "codex: user's top-level keys commented out, their tables kept")
if tomllib:
    d = tomllib.loads(txt)
    check(d["model"] == "fake-moe" and "mine" in d["model_providers"] and d["projects"]["/a"]["trust_level"] == "trusted", "codex: merged file parses with both providers")
run("codex", "--write", "--api-key", "new-key")
txt = open(cx).read()
check(txt.count("[model_providers.moe-cache]") == 1 and "new-key" in txt and KEY not in txt and open(cx + ".bak").read() == orig, "codex: second connect replaces our block only; backup untouched")
put(cx, '[model_providers.moe-cache]\nbase_url = "x"\n')
rc, _, _ = run("codex", "--write")
check(isinstance(rc, str) and "already defines" in rc and open(cx).read().startswith("[model_providers"), "codex: a hand-written moe-cache provider is refused, file untouched")
if tomllib:
    put(cx, "model = = broken\n")
    rc, _, _ = run("codex", "--write")
    check(isinstance(rc, str) and "not valid TOML" in rc and open(cx).read() == "model = = broken\n", "codex: malformed TOML refused, untouched")
os.remove(cx)

# ---- Pi ----
pm = os.path.join(HOME, ".pi", "agent", "models.json"); ps = os.path.join(HOME, ".pi", "agent", "settings.json")
rc, out, _ = run("pi")
check(rc == 0 and '"contextWindow": 57344' in out and "defaultProvider" in out and KEY not in out and not os.path.exists(pm), "pi: preview")
run("pi", "--write")
d, s = json.load(open(pm)), json.load(open(ps))
p = d["providers"]["moe-cache"]
check(p["api"] == "openai-completions" and p["apiKey"] == KEY and [m["id"] for m in p["models"]] == ["fake-moe", "second"] and p["models"][0]["contextWindow"] == 57344
      and s == {"defaultProvider": "moe-cache", "defaultModel": "fake-moe"} and mode(pm) == 0o600 and mode(ps) == 0o600, "pi: models.json provider + settings.json default written, private")
orig_m = json.dumps({"providers": {"ollama": {"baseUrl": "http://localhost:11434/v1", "api": "openai-completions", "apiKey": "ollama", "models": [{"id": "x"}]}}})
orig_s = json.dumps({"defaultProvider": "anthropic", "defaultModel": "claude-x", "theme": "dark"})
put(pm, orig_m); put(ps, orig_s)
for f in (pm + ".bak", ps + ".bak"):
    os.path.exists(f) and os.remove(f)
run("pi", "--write"); run("pi", "--write")
d, s = json.load(open(pm)), json.load(open(ps))
check("ollama" in d["providers"] and "moe-cache" in d["providers"] and s["theme"] == "dark" and s["defaultProvider"] == "moe-cache", "pi: existing provider and settings kept, ours added")
check(open(pm + ".bak").read() == orig_m and open(ps + ".bak").read() == orig_s, "pi: first backups hold the originals after two connects")
put(pm, "{ // comment\n}")
rc, _, _ = run("pi", "--write")
check(isinstance(rc, str) and "not plain JSON" in rc and open(pm).read() == "{ // comment\n}", "pi: commented file refused, untouched")
put(pm, '{"providers": []}')
rc, _, _ = run("pi", "--write")
check(isinstance(rc, str) and "not an object" in rc, "pi: providers that is not an object refused")
put(pm, "[]")
rc, _, _ = run("pi", "--write")
check(isinstance(rc, str) and "not a Pi models.json object" in rc, "pi: top-level array refused")

# ---- OpenClaw ----
oc = os.path.join(HOME, ".openclaw", "openclaw.json")
rc, out, _ = run("openclaw")
check(rc == 0 and '"primary": "moe-cache/fake-moe"' in out and KEY not in out and not os.path.exists(oc), "openclaw: preview")
run("openclaw", "--write")
d = json.load(open(oc))
check(d["models"]["providers"]["moe-cache"]["baseUrl"] == "http://127.0.0.1:8080/v1" and d["agents"]["defaults"]["model"]["primary"] == "moe-cache/fake-moe"
      and d["models"]["providers"]["moe-cache"]["models"][0]["contextWindow"] == 57344 and mode(oc) == 0o600, "openclaw: provider + primary model written, private")
orig = json.dumps({"models": {"mode": "replace", "providers": {"x": {"baseUrl": "u"}}}, "agents": {"defaults": {"model": "anthropic/claude-x", "workspace": "/w"}}, "gateway": {"port": 1}})
put(oc, orig); os.remove(oc + ".bak") if os.path.exists(oc + ".bak") else None
run("openclaw", "--write"); run("openclaw", "--write")
d = json.load(open(oc))
check(d["models"]["mode"] == "replace" and "x" in d["models"]["providers"] and d["gateway"] == {"port": 1} and d["agents"]["defaults"]["workspace"] == "/w"
      and d["agents"]["defaults"]["model"] == {"primary": "moe-cache/fake-moe"} and open(oc + ".bak").read() == orig, "openclaw: other settings kept, string model turned into primary, first backup kept")
put(oc, "{\n  // json5\n  models: {},\n}")
rc, _, _ = run("openclaw", "--write")
check(isinstance(rc, str) and "not plain JSON" in rc and "json5" in open(oc).read(), "openclaw: JSON5 file refused, untouched")
put(oc, '{"agents": {"defaults": 3}}')
rc, _, _ = run("openclaw", "--write")
check(isinstance(rc, str) and "not an object" in rc, "openclaw: agents.defaults of the wrong type refused")

# ---- Qwen Code ----
qw = os.path.join(HOME, ".qwen", "settings.json")
rc, out, _ = run("qwen")
check(rc == 0 and "MOE_CACHE_API_KEY" in out and KEY not in out and not os.path.exists(qw), "qwen: preview")
run("qwen", "--write")
d = json.load(open(qw))
e = d["modelProviders"]["openai"]
check(d["env"]["MOE_CACHE_API_KEY"] == KEY and [x["id"] for x in e] == ["fake-moe", "second"] and e[0]["generationConfig"]["contextWindowSize"] == 57344
      and d["security"]["auth"]["selectedType"] == "openai" and d["model"]["name"] == "fake-moe" and mode(qw) == 0o600, "qwen: env key, provider entries, auth type, model written, private")
orig = json.dumps({"modelProviders": {"openai": [{"id": "mine", "envKey": "OTHER", "baseUrl": "http://h/v1"}]}, "ui": {"theme": "x"}, "env": {"A": "1"}})
put(qw, orig); os.remove(qw + ".bak") if os.path.exists(qw + ".bak") else None
run("qwen", "--write"); run("qwen", "--write")
d = json.load(open(qw))
check([x["id"] for x in d["modelProviders"]["openai"]] == ["mine", "fake-moe", "second"] and d["ui"] == {"theme": "x"} and d["env"]["A"] == "1" and open(qw + ".bak").read() == orig,
      "qwen: user's entry kept, ours replaced (not duplicated), first backup kept")
put(qw, '{"modelProviders": {"openai": {}}}')
rc, _, _ = run("qwen", "--write")
check(isinstance(rc, str) and "not a list" in rc, "qwen: openai entries of the wrong type refused")
put(qw, "not json")
rc, _, _ = run("qwen", "--write")
check(isinstance(rc, str) and "not plain JSON" in rc and open(qw).read() == "not json", "qwen: malformed file refused, untouched")

# ---- Aider ----
ac = os.path.join(HOME, ".aider.moe-cache.conf.yml"); am = os.path.join(HOME, ".aider.moe-cache.model.metadata.json")
rc, out, _ = run("aider")
check(rc == 0 and "openai-api-base" in out and KEY not in out and not os.path.exists(ac), "aider: preview")
run("aider", "--write")
y = open(ac).read()
m = json.load(open(am))
check(f'openai-api-key: "{KEY}"' in y and 'model: "openai/fake-moe"' in y and am in y and m["openai/fake-moe"]["max_input_tokens"] == 57344 and "openai/second" in m
      and m["openai/fake-moe"]["litellm_provider"] == "openai" and mode(ac) == 0o600, "aider: own config + metadata written, private")
put(am, json.dumps({"openai/other": {"max_input_tokens": 1}}))
run("aider", "--write")
check(set(json.load(open(am))) == {"openai/other", "openai/fake-moe", "openai/second"} and json.load(open(am + ".bak")) == {"openai/other": {"max_input_tokens": 1}}, "aider: metadata merged, first backup kept")
put(am, "{broken")
rc, _, _ = run("aider", "--write")
check(isinstance(rc, str) and "not plain JSON" in rc, "aider: malformed metadata refused")
check(not os.path.exists(os.path.join(HOME, ".aider.conf.yml")), "aider: the user's own .aider.conf.yml is never created or touched")

# ---- a server that answers 401 is not "no server" (found with the real agents: temporary HOME without the stored key) ----
_si, _http = mc.server_info, mc.http
mc.server_info = lambda url, key=None: (None, None)
mc.http = lambda url, body=None, api_key=None, timeout=10: (401, "")
rc, _, _ = run("codex")
check(isinstance(rc, str) and "answers 401" in rc and "--api-key" in rc, "401 from the server: the message says it wants its API key")
mc.http = lambda url, body=None, api_key=None, timeout=10: (0, "")
rc, _, _ = run("codex")
check(isinstance(rc, str) and "no llama-server answers" in rc, "unreachable server: still 'no llama-server answers'")
mc.server_info, mc.http = _si, _http

# ---- --config, and the help line ----
alt = os.path.join(TMP, "alt", "c.toml")
run("codex", "--write", "--config", alt)
check(os.path.isfile(alt), "--config PATH writes there")
check("codex|pi|openclaw|aider|qwen" in open(os.path.join(HERE, "bin", "moe_cache.py")).read().split('"""')[1], "the usage text lists the new agents")
print("\n%d failed" % len(fails) if fails else "\nall ok")
sys.exit(1 if fails else 0)
