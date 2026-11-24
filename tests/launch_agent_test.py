#!/usr/bin/env python3
"""AGENTS-865: `moe-cache launch claude|opencode|codex` starts the agent with the API key ONLY in the child's environment: no config file is written
anywhere (temp HOME, temp cwd are scanned for the key afterwards), the key is not on the agent's command line, the agent's own config stays untouched.
Fake agents on PATH record their environment and arguments; no model, no server (server_info / server_models are replaced); standard library only.
   python3 tests/launch_agent_test.py"""
import contextlib, io, json, os, stat, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-launchagent-test-")
HOME, CWD, BIN = (os.path.join(TMP, d) for d in ("home", "work", "bin"))
for d in (HOME, CWD, BIN, os.path.join(TMP, "cache")):
    os.makedirs(d)
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
os.environ["XDG_CONFIG_HOME"] = os.path.join(HOME, ".config")
os.environ.pop("LLAMA_API_KEY", None)
os.environ["PATH"] = BIN + os.pathsep + os.environ["PATH"]
os.chdir(CWD)
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc  # noqa: E402
import moe_cache_launch as ml  # noqa: E402

KEY = "k3y-abc123-SECRET"
REC = os.path.join(TMP, "rec")
mc.server_info = lambda url, key=None: ("fake-moe", 65536)
mc.server_models = lambda url, key=None: ["fake-moe", "second"]
mc.key_for = lambda url, given=None, trusted=False: given or KEY
fails = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)


def fake_agent(name):
    """A script that writes its arguments and environment to REC/<name>.json and exits 7."""
    p = os.path.join(BIN, name)
    with open(p, "w") as f:
        f.write('#!/usr/bin/env python3\nimport json, os, sys\nos.makedirs(%r, exist_ok=True)\njson.dump({"argv": sys.argv[1:], "env": dict(os.environ)}, open(os.path.join(%r, %r + ".json"), "w"))\nsys.exit(7)\n' % (REC, REC, name))
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = ml.launch(list(argv))
    return rc, out.getvalue()


def key_in_files():
    hits = []
    for root in (HOME, CWD, os.path.join(TMP, "cache")):
        for dp, _, fs in os.walk(root):
            for fn in fs:
                p = os.path.join(dp, fn)
                try:
                    if KEY in open(p, errors="replace").read():
                        hits.append(p)
                except OSError:
                    pass
    return hits


# the user's own agent configs exist before and must stay byte for byte
own = {os.path.join(HOME, ".config", "opencode", "opencode.json"): '{"theme": "mine"}\n', os.path.join(HOME, ".codex", "config.toml"): 'model = "gpt"\n',
       os.path.join(CWD, ".claude", "settings.local.json"): '{"env": {"X": "1"}}\n'}
for p, t in own.items():
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(t)
before = {p: open(p).read() for p in own}

for agent in ("claude", "opencode", "codex"):
    fake_agent(agent)
    rc, out = run(agent, "--url", "http://127.0.0.1:8080", "--", "--extra", "arg")
    rec = json.load(open(os.path.join(REC, agent + ".json")))
    check(rc == 7, f"{agent}: the agent's exit code is returned ({rc})")
    check(rec["argv"][-2:] == ["--extra", "arg"], f"{agent}: arguments after -- reach the agent")
    check(not any(KEY in a for a in rec["argv"]), f"{agent}: the key is not on the agent's command line")
    check(KEY not in out, f"{agent}: the launcher's own text does not print the key")
    check(any(KEY in v for k, v in rec["env"].items()), f"{agent}: the key is in the agent's environment")
    check(KEY not in os.environ.get("ANTHROPIC_AUTH_TOKEN", "") and KEY not in "".join(os.environ.values()), f"{agent}: the launcher's own environment is unchanged")
    env = rec["env"]
    if agent == "claude":
        check(env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080" and env["ANTHROPIC_AUTH_TOKEN"] == KEY and env["ANTHROPIC_MODEL"] == "fake-moe" and env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0", "claude: base URL, token, model, attribution header")
    if agent == "opencode":
        cfg = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        check(cfg["model"] == "moe-cache/fake-moe" and cfg["provider"]["moe-cache"]["options"]["baseURL"] == "http://127.0.0.1:8080/v1" and set(cfg["provider"]["moe-cache"]["models"]) == {"fake-moe", "second"}, "opencode: provider entry in OPENCODE_CONFIG_CONTENT")
    if agent == "codex":
        a = " ".join(rec["argv"])
        check('model_providers.moe-cache.env_key="MOE_CACHE_API_KEY"' in a and env["MOE_CACHE_API_KEY"] == KEY and "experimental_bearer_token" not in a, "codex: key by env_key, not by a command-line token")

check(key_in_files() == [], f"no file under HOME / cwd / cache holds the key ({key_in_files()})")
check({p: open(p).read() for p in own} == before, "the user's own claude / opencode / codex configs are untouched")
check(sorted(os.listdir(CWD)) == [".claude"] and os.listdir(os.path.join(CWD, ".claude")) == ["settings.local.json"], "nothing new in the working folder")

# no key at all (open server): nothing in the environment names one
mc.key_for = lambda url, given=None, trusted=False: None
fake_agent("claude"); rc, out = run("claude")
rec = json.load(open(os.path.join(REC, "claude.json")))
check(rec["env"]["ANTHROPIC_AUTH_TOKEN"] == "none", "server without a key: token 'none'")
# agent not installed / server not answering: a message, exit 1, nothing started
os.remove(os.path.join(BIN, "codex")); os.environ["PATH"] = BIN      # a real codex elsewhere on this PC must not count
rc, out = run("codex")
check(rc == 1 and "not on PATH" in out, "agent not installed: message, exit 1")
mc.server_info = lambda url, key=None: (None, None)
rc, out = run("claude")
check(rc == 1 and "no llama-server answers" in out, "no server: message, exit 1")
print("FAILED: " + "; ".join(fails) if fails else "all ok"); sys.exit(1 if fails else 0)
