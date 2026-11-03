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
# ---- writer robustness (audit 2026-11-03: AUD-066 .. AUD-078) ----
W = os.path.join(TMP, "w"); os.makedirs(W)

# AUD-066: a symlinked config stays a link; the target is replaced, the backup sits beside the target
if os.name != "nt":
    real = os.path.join(W, "dotfiles", "opencode.json"); link = os.path.join(W, "cfg", "opencode.json")
    put(real, '{"provider": {"mine": {}}}'); os.makedirs(os.path.dirname(link)); os.symlink(real, link)
    rc, _, _ = run("opencode", "--write", "--config", link)
    check(rc == 0 and os.path.islink(link) and "moe-cache" in json.load(open(real))["provider"] and os.path.isfile(real + ".bak") and not os.path.exists(link + ".bak"),
          "opencode: a symlinked config stays a link, the real file is merged, the backup is beside it")
    real = os.path.join(W, "dotfiles", "settings.local.json"); link = os.path.join(W, "cfg", "settings.local.json")
    put(real, '{"permissions": {}}'); os.symlink(real, link)
    rc, _, _ = run("claude", "--write", "--config", link)
    check(rc == 0 and os.path.islink(link) and json.load(open(real))["env"]["ANTHROPIC_MODEL"] == "fake-moe", "claude: a symlinked settings file stays a link")
    real = os.path.join(W, "dotfiles", "qwen.json"); link = os.path.join(W, "cfg", "qwen.json")
    put(real, '{"ui": 1}'); os.symlink(real, link)
    rc, _, _ = run("qwen", "--write", "--config", link)
    check(rc == 0 and os.path.islink(link) and json.load(open(real))["ui"] == 1 and os.path.isfile(real + ".bak"), "qwen: a symlinked config stays a link (shared writer)")

# AUD-070: CRLF kept, non-ASCII not escaped, final newline
oc = os.path.join(W, "crlf", "opencode.json")
os.makedirs(os.path.dirname(oc))
open(oc, "wb").write('{\r\n  "theme": "dunkel-ü"\r\n}\r\n'.encode("utf-8"))
rc, _, _ = run("opencode", "--write", "--config", oc)
raw = open(oc, "rb").read()
check(rc == 0 and raw.endswith(b"\r\n") and b"\n" not in raw.replace(b"\r\n", b"") and "dunkel-ü".encode("utf-8") in raw and b"\\u00" not in raw, "opencode: CRLF kept, non-ASCII not escaped, final newline")
fresh = os.path.join(W, "fresh", "opencode.json")
run("opencode", "--write", "--config", fresh)
check(open(fresh, "rb").read().endswith(b"}\n"), "opencode: a new file ends with a newline")
cxc = os.path.join(W, "crlf", "config.toml")
open(cxc, "wb").write(b'model = "x"\r\n[projects."/a"]\r\ntrust_level = "trusted"\r\n')
if tomllib:
    rc, _, _ = run("codex", "--write", "--config", cxc)
    raw = open(cxc, "rb").read()
    check(rc == 0 and b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), "codex: CRLF file stays CRLF")

# AUD-071: the opencode backup is the first file, not re-copied when the file has no moe-cache provider
oc = os.path.join(W, "second", "opencode.json")
put(oc, '{"theme": "first"}')
run("opencode", "--write", "--config", oc)
put(oc, '{"theme": "second"}')                       # the user replaced the file by one without our provider
run("opencode", "--write", "--config", oc)
check(json.load(open(oc + ".bak")) == {"theme": "first"}, "opencode: a second connect on a file without our provider keeps the first backup")

# AUD-072: an OSError is a message, not a traceback
dirasfile = os.path.join(W, "isdir", "opencode.json"); os.makedirs(dirasfile)
rc, _, _ = run("opencode", "--write", "--config", dirasfile)
check(isinstance(rc, str) and "opencode was not connected" in rc, f"opencode: a directory where the file should be -> one message ({str(rc)[:100]})")
bad = os.path.join(W, "cp1252", "config.toml"); os.makedirs(os.path.dirname(bad))
open(bad, "wb").write(b'# caf\xe9\nmodel = "x"\n')
if tomllib:
    rc, _, _ = run("codex", "--write", "--config", bad)
    check(isinstance(rc, str) and "not UTF-8" in rc and open(bad, "rb").read().startswith(b"# caf\xe9"), "codex: a file that is not UTF-8 -> a message, untouched")

# AUD-069: no TOML check below Python 3.11 -> codex --write refuses
_tl = sys.modules.get("tomllib", "absent"); sys.modules["tomllib"] = None
rc, _, _ = run("codex", "--write", "--config", os.path.join(W, "notoml", "config.toml"))
if _tl == "absent":
    del sys.modules["tomllib"]
else:
    sys.modules["tomllib"] = _tl
check(isinstance(rc, str) and "Python 3.11" in rc and not os.path.exists(os.path.join(W, "notoml", "config.toml")), "codex --write without tomllib: refused with the reason, nothing written")

# AUD-073: Pi writes both files or neither
if os.name != "nt":
    pd = os.path.join(W, "pi"); os.makedirs(os.path.join(pd, "settings.json"))          # a directory where settings.json should be
    rc, _, _ = run("pi", "--write", "--config", os.path.join(pd, "models.json"))
    check(isinstance(rc, str) and not os.path.exists(os.path.join(pd, "models.json")) and not [f for f in os.listdir(pd) if f.endswith(".tmp")],
          f"pi: when settings.json cannot be written models.json is not left behind, no temporary files remain ({str(rc)[:80]})")

# AUD-078: the ambient LLAMA_API_KEY goes only to a server on this machine
seen = []
_kf = mc.key_for
mc.key_for = lambda url, given=None: (seen.append((url, given)) or given)
os.environ["LLAMA_API_KEY"] = "ENVKEY"
run("pi", "--url", "http://remote.example:8080")
run("pi", "--url", "http://127.0.0.1:8080")
run("pi", "--url", "http://remote.example:8080", "--api-key", "TYPED")
del os.environ["LLAMA_API_KEY"]
mc.key_for = _kf
check(seen == [("http://remote.example:8080", None), ("http://127.0.0.1:8080", "ENVKEY"), ("http://remote.example:8080", "TYPED")],
      f"LLAMA_API_KEY is not sent to a remote --url (a typed --api-key is) ({seen})")

# AUD-068: previews of hermes / opencode / claude never print the key
for ag in ("hermes", "opencode", "claude"):
    rc, out, err = run(ag)
    check(rc == 0 and KEY not in out + err and "<your API key" in out + err, f"{ag}: the preview hides the key")

# AUD-077: the Claude settings backup is not beside the project file
proj = os.path.join(W, "proj"); cl = os.path.join(proj, ".claude", "settings.local.json")
put(cl, '{"env": {"ANTHROPIC_API_KEY": "mine"}}')
rc, out, _ = run("claude", "--write", "--config", cl)
bk = mc.claude_backup_path(cl)
check(rc == 0 and not os.path.exists(cl + ".bak") and json.load(open(bk))["env"] == {"ANTHROPIC_API_KEY": "mine"}, "claude: the backup of the project's settings file is outside the project folder")
os.makedirs(os.path.join(W, "repo", "." + "git")); os.makedirs(os.path.join(W, "repo", "sub"))
check("repo" in (mc.project_folder_hint(os.path.join(W, "repo", "sub")) or "") and mc.project_folder_hint(os.path.join(W, "repo")) is None, "claude: a sub-folder of a project gets a hint")

# AUD-067 / 073 / 074: Hermes through a stand-in `hermes` (a script that keeps its config as a JSON file)
if os.name != "nt":
    hb = os.path.join(W, "hbin"); os.makedirs(hb)
    hc = os.path.join(W, "hermes-config.yaml")
    fake = os.path.join(hb, "hermes")
    open(fake, "w").write('#!/usr/bin/env python3\nimport json, os, sys\nC = %r\n'
                          'a = sys.argv[1:]\nd = json.load(open(C)) if os.path.exists(C) else {}\n'
                          'if a[:2] == ["config", "path"]: print(C)\n'
                          'elif a[:2] == ["config", "get"]: print(d.get(a[2], ""))\n'
                          'elif a[:2] == ["config", "set"]:\n'
                          '    if os.environ.get("FAKE_HERMES_FAIL") == a[2]: print("boom", file=sys.stderr); sys.exit(1)\n'
                          '    d[a[2]] = a[3]; json.dump(d, open(C, "w"))\n' % hc)
    os.chmod(fake, 0o755)
    os.environ["PATH"] = hb + os.pathsep + os.environ["PATH"]
    json.dump({"model.provider": "openrouter", "model.default": "gpt-x", "model.base_url": "https://openrouter.ai/api/v1"}, open(hc, "w"))
    orig_h = open(hc).read()
    rc, out, err = run("hermes", "--write")
    check(rc == 0 and open(hc + ".bak").read() == orig_h and json.load(open(hc))["model.provider"] == "custom", "hermes: --write sets the model and backs up the old config")
    check(KEY not in out + err, "hermes: the key is not printed")
    run("hermes", "--write")                                                      # our own setting again: no new backup
    check(len([f for f in os.listdir(W) if f.startswith("hermes-config.yaml.bak")]) == 1 and open(hc + ".bak").read() == orig_h, "hermes: a second connect does not bury the first backup")
    json.dump({"model.provider": "custom", "model.default": "ollama-m", "model.base_url": "http://127.0.0.1:11434/v1"}, open(hc, "w"))   # the user's own local server
    orig2 = open(hc).read()
    run("hermes", "--write")
    baks = sorted(f for f in os.listdir(W) if f.startswith("hermes-config.yaml.bak"))
    check(len(baks) == 2 and open(hc + ".bak").read() == orig_h and any(open(os.path.join(W, f)).read() == orig2 for f in baks if f != "hermes-config.yaml.bak"),
          f"hermes: the user's own local server is backed up too, the first .bak is never overwritten ({baks})")
    os.environ["FAKE_HERMES_FAIL"] = "model.default"
    rc, _, _ = run("hermes", "--write")
    del os.environ["FAKE_HERMES_FAIL"]
    check(isinstance(rc, str) and "partly set up" in rc and "model.provider" in rc and "model.default" in rc, f"hermes: a failure half-way says what is done and what is not ({str(rc)[:120]})")

print("\n%d failed" % len(fails) if fails else "\nall ok")
sys.exit(1 if fails else 0)
