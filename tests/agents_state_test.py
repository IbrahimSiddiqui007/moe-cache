#!/usr/bin/env python3
"""GUI-025: the control page reads OpenCode's and Hermes' config files back to show the real connected state. No agent, no server: fake config folders.
   python3 tests/agents_state_test.py"""
import json, os, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-agents-state-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
os.environ["XDG_CONFIG_HOME"] = os.path.join(TMP, "config")
os.environ["HERMES_HOME"] = os.path.join(TMP, "hermes")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_ui as ui  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def put(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)

URL = "http://127.0.0.1:8080"
oc = os.path.join(TMP, "config", "opencode")
hm = os.path.join(TMP, "hermes", "config.yaml")

# nothing written yet
st = ui.agents_state(URL)["agents"]
check([a["agent"] for a in st] == ["opencode", "hermes"] and not any(a["found"] or a["connected"] for a in st), "no config files: neither agent is found or connected")

# OpenCode
block = {"model": "moe-cache/m", "provider": {"moe-cache": {"options": {"baseURL": "http://127.0.0.1:8080/v1"}}}}
put(oc + "/opencode.json", json.dumps(block))
o = ui.opencode_state(URL)
check(o["connected"] and o["found"] and o["model"] == "moe-cache/m" and o["file"].endswith("opencode.json"), "OpenCode: provider at this server and moe-cache/ default model: connected")
check(not ui.opencode_state("http://127.0.0.1:8081")["connected"] and ui.opencode_state("http://127.0.0.1:8081")["found"], "OpenCode: another port: found but not connected")
check(ui.opencode_state("http://localhost:8080/")["connected"], "OpenCode: localhost and a trailing slash are the same server")
check(not ui.opencode_state(None)["connected"], "OpenCode: no server running: not connected")
put(oc + "/opencode.jsonc", '// written by OpenCode Desktop\n{\n  /* mine */ "model": "anthropic/claude", // comment with "quotes"\n  "theme": "x,]",\n}\n')
o = ui.opencode_state(URL)
check(o["found"] and not o["connected"] and o["model"] == "anthropic/claude", "OpenCode: a later opencode.jsonc with its own default model wins: not connected (comments and trailing comma read)")
os.remove(oc + "/opencode.jsonc")
put(oc + "/opencode.json", "{ not json")
check(not ui.opencode_state(URL)["found"], "OpenCode: a broken file is skipped, no error")

# Hermes
put(hm, "# top\nagent:\n  max_turns: 5\nmodel:\n  provider: custom   # ours\n  base_url: \"http://127.0.0.1:8080/v1\"\n  default: 'qwen-x'\n  extra:\n    base_url: nested\n  api_key: ${MOE_CACHE_API_KEY}\nkanban:\n  provider: other\n")
h = ui.hermes_state(URL)
check(h["connected"] and h["found"] and h["model"] == "qwen-x" and h["url"] == "http://127.0.0.1:8080/v1", f"Hermes: custom provider at this server: connected ({h})")
check(not ui.hermes_state("http://127.0.0.1:9999")["connected"] and ui.hermes_state("http://127.0.0.1:9999")["found"], "Hermes: another port: found but not connected")
put(hm, "model:\n  provider: openrouter\n  base_url: https://openrouter.ai/api/v1\n  default: gpt-x\n")
h = ui.hermes_state(URL)
check(not h["found"] and not h["connected"] and h["url"] is None, "Hermes: the user's own OpenRouter setup is not ours")
put(hm, "model: some-name\nother:\n  provider: custom\n")
check(not ui.hermes_state(URL)["found"], "Hermes: `model` as a plain name (no mapping): not connected")
os.remove(hm)
check(not ui.hermes_state(URL)["found"], "Hermes: no config.yaml: not connected")
check(ui.yaml_model_block("model:\n  a: 1\n  b: \"x # y\"\n") == {"a": "1", "b": "x # y"}, "yaml: quoted values keep a # inside")
sys.exit(1 if fails else 0)
