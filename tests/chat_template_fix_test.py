#!/usr/bin/env python3
"""AGENTS-992: the chat-template fix for Claude Code's mid-conversation system messages, and the launcher stage that applies it.
   python3 tests/chat_template_fix_test.py
Qwen3.8-27B's own template raises "System message must be at the beginning." (HTTP 500, seen through Claude Code on the PC, 2026-11-09). tests/data/qwen38_27b_midsystem_patched.jinja is the
owner's hand-patched file; its original differs by exactly one line, which is re-created here by the reverse edit, so the test needs no model file."""
import os, struct, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner")); sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "tests"))
import chat_template_fix as ctf   # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

patched = open(os.path.join(HERE, "tests", "data", "qwen38_27b_midsystem_patched.jinja"), encoding="utf-8").read()
original = patched.replace(ctf.USER_TURN, ctf.RAISE_LINE)
check(original != patched and ctf.needs_fix(original) and not ctf.needs_fix(patched), "the original template raises on a later system message, the patched one does not")
check(ctf.fix(original) == patched, "fix(original) is the owner's hand-patched template, byte for byte")
check(ctf.fix(patched) is None, "an already fixed template is left alone")
check(ctf.fix("{{ messages }}") is None and not ctf.needs_fix("plain chatml") and not ctf.needs_fix(None), "templates without the raise are left alone")
odd = original.replace("set content = render_content(", "set body = render_content(")
check(ctf.fix(odd) is None, "an unknown shape (no rendered `content` before the raise) is not guessed at")
mtp = "{%- for message in messages %}\n    {%- set content = render_content(message.content, true)|trim %}\n    {%- if message.role == \"system\" %}\n        {%- if not loop.first %}\n            " + ctf.RAISE_LINE + "\n        {%- endif %}\n    {%- elif message.role == \"user\" %}x{%- endif %}{%- endfor %}"
check(ctf.fix(mtp) and "raise_exception" not in ctf.fix(mtp), "the second known shape (MTP builds: raise inside `if not loop.first`) is fixed too")

print("rendering with jinja2 (skipped when it is not installed)")
try:
    import jinja2
except ImportError:
    jinja2 = None
    if os.environ.get("REQUIRE_JINJA2"):      # CI installs it: a skipped rendering check would hide a template that no longer fails/renders
        check(False, "jinja2 is required (REQUIRE_JINJA2) but not installed")
if jinja2:
    def render(tpl, msgs):
        env = jinja2.Environment(loader=jinja2.BaseLoader(), extensions=["jinja2.ext.loopcontrols"])
        def boom(m): raise RuntimeError(m)
        env.globals["raise_exception"] = boom
        env.filters["tojson"] = lambda v, **k: __import__("json").dumps(v)
        return env.from_string(tpl).render(messages=msgs, add_generation_prompt=True, tools=None)
    conv = [{"role": "system", "content": "You are Claude Code."}, {"role": "user", "content": "Fix the bug."}, {"role": "assistant", "content": "Looking."},
            {"role": "system", "content": "<system-reminder>todo list changed</system-reminder>"}, {"role": "user", "content": "Go on."}]
    try:
        render(original, conv); raised = None
    except Exception as e:
        raised = str(e)
    check(raised and "System message must be at the beginning" in raised, "original template: the Claude Code conversation raises (the HTTP 500)")
    out = render(patched, conv)
    check("<|im_start|>user\n<system-reminder>todo list changed</system-reminder><|im_end|>" in out, "fixed template: the later system message is a user turn")
    check(out.count("<|im_start|>system") == 1 and "You are Claude Code." in out, "fixed template: the first system message is still the system prompt")
    plain = conv[:3] + [{"role": "user", "content": "Go on."}]
    check(render(original, plain) == render(patched, plain), "a conversation without a later system message renders identically with both templates")
    # Which /v1/messages bodies produce a system message after the first turn? llama-server's Anthropic converter (tools/server/server-chat.cpp, server_chat_convert_anthropic_to_oai)
    # turns the top-level `system` into message 0 and passes every entry of `messages` through with its own role. Checked on the laptop with the real llama-server
    # (0.5.0-dev 7fe450e) and this template: a Claude Code shaped body (top-level system, <system-reminder> text blocks in user turns) answers 200 with the ORIGINAL template;
    # a body with a role "system"/"developer" entry after a user turn answers 500 with exactly this error. So the trigger is such an entry, not the top-level system field.
    def anthropic_to_oai(body):
        out = []
        if body.get("system") is not None:
            s = body["system"]
            out.append({"role": "system", "content": s if isinstance(s, str) else "".join(b.get("text", "") for b in s if b.get("type") == "text")})
        for m in body["messages"]:
            c = m["content"]
            out.append({"role": m["role"], "content": c if isinstance(c, str) else "".join(b.get("text", "") for b in c if b.get("type") == "text")})
        return out
    def ok_original(body):
        try:
            render(original, anthropic_to_oai(body)); return True
        except RuntimeError as e:
            assert "System message must be at the beginning" in str(e); return False
    cc_shape = {"system": [{"type": "text", "text": "You are Claude Code."}], "messages": [
        {"role": "user", "content": [{"type": "text", "text": "<system-reminder>x</system-reminder>"}, {"type": "text", "text": "hi"}]}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]}
    check(ok_original(cc_shape), "Claude Code shaped body (top-level system, reminders inside user turns): the ORIGINAL template accepts it (so it is not the trigger)")
    for name in ("system", "developer"):
        body = {"system": "s", "messages": [{"role": "user", "content": "hi"}, {"role": name, "content": [{"type": "text", "text": "reminder"}]}, {"role": "user", "content": "go"}]}
        check(not ok_original(body) and "<|im_start|>user\nreminder<|im_end|>" in render(patched, anthropic_to_oai(body)),
              f"a role \"{name}\" entry after a user turn in /v1/messages: the ORIGINAL template raises (HTTP 500), the fixed one renders it as a user turn")
    check(not ok_original({"messages": [{"role": "user", "content": "hi"}, {"role": "system", "content": "r"}]}), "no top-level system and a system entry after the first user turn: raises with the original template")
    check(ok_original({"system": "s", "messages": [{"role": "system", "content": "more"}, {"role": "user", "content": "hi"}]}), "system entries at the start of messages are merged, not an error (unchanged)")
    two = [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}] + conv[1:3] + [{"role": "user", "content": "x"}]
    check(render(patched, two) == render(original, two), "two leading system messages are merged as before (unchanged)")

print("reading a model and the launcher stage")
def make_model(path, template):
    def string(x):
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv = [("general.architecture", 8, string("llama")), ("llama.block_count", 4, struct.pack("<I", 1))]
    if template is not None:
        kv.append(("tokenizer.chat_template", 8, string(template)))
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv))
    for k, t, v in kv:
        head += string(k) + struct.pack("<I", t) + v
    head += string("token_embd.weight") + struct.pack("<I", 1) + struct.pack("<Q", 32) + struct.pack("<IQ", 0, 0)
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head + b"\0" * 128)
import moe_cache as mc   # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    bad, good, none = (os.path.join(tmp, n) for n in ("bad.gguf", "good.gguf", "none.gguf"))
    make_model(bad, original); make_model(good, "{{ messages }}"); make_model(none, None)
    check(ctf.model_template(bad) == original and ctf.model_template(none) is None and ctf.model_template(os.path.join(tmp, "missing")) is None, "the template is read from the GGUF header; none/missing -> None")
    cache = os.path.join(tmp, "cache")
    path, text = ctf.fixed_template_file(bad, cache)
    check(path and "\\" not in path and os.path.isfile(path) and open(path, encoding="utf-8").read() == patched, "the fixed template is written to the cache folder, forward slashes in the path")
    check(ctf.fixed_template_file(bad, cache)[0] == path, "the same model gives the same file")
    check(ctf.fixed_template_file(good, cache) == (None, None) and ctf.fixed_template_file(none, cache) == (None, None), "a model whose template is fine gets nothing")
    p2, _ = ctf.fixed_template_file(bad, os.path.join(tmp, "dry"), write=False)
    check(p2 and not os.path.exists(os.path.join(tmp, "dry")), "a dry run names the file but writes nothing")
    saved = mc.CACHE_DIR
    mc.CACHE_DIR = cache
    try:
        a = type("A", (), {})()
        extra = []
        add = mc.template_stage(a, extra, bad)
        check(add[:1] == ["--chat-template-file"] and add[1] == path and "--jinja" in add, f"launcher stage: --chat-template-file + --jinja for the failing model ({add})")
        r = mc.template_stage(a, ["--jinja"], bad)
        check(r[:1] == ["--chat-template-file"] and "--jinja" not in r, "--jinja already given: not added twice")
        check(mc.template_stage(a, ["--chat-template-file", "mine.jinja"], bad) == [] and mc.template_stage(a, ["--chat-template", "chatml"], bad) == [] and mc.template_stage(a, ["--no-jinja"], bad) == [],
              "the user's own --chat-template[-file] / --no-jinja win")
        odd_model = os.path.join(tmp, "odd.gguf"); make_model(odd_model, original.replace("set content = render_content(", "set body = render_content("))
        said = []; sv_say = mc.say; mc.say = lambda m, err=False: said.append(m)
        try:
            check(mc.template_stage(a, [], odd_model) == [] and any("does not recognise" in m for m in said), "a template that raises but has an unknown shape: nothing added, and a warning says to pass your own")
        finally:
            mc.say = sv_say
        check(mc.template_stage(a, [], good) == [] and mc.template_stage(a, [], none) == [], "a model with a fine or no template: nothing added")
    finally:
        mc.CACHE_DIR = saved

print("FAIL" if fails else "all ok")
sys.exit(1 if fails else 0)
