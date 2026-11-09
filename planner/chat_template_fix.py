"""AGENTS-992: chat templates that reject a system message in the middle of a conversation.

A client that sends a system (or developer) message after the first turn makes the chat template of Qwen3.8 (27B, Flash-Next, the MTP builds) raise
"System message must be at the beginning.", so llama-server answers HTTP 500 (seen through Claude Code on the PC, 2026-11-09, Qwen3.8-27B). llama-server's /v1/messages turns the
top-level `system` field into message 0 (fine) but passes a role "system"/"developer" entry inside `messages` through as is: that entry is the trigger (tests/chat_template_fix_test.py);
a plain Claude Code body with a top-level system and reminders inside user turns does not trigger it on the laptop's llama-server.
The fix is one line of the template: a later system (or developer) message is rendered as a user turn instead of raising. The fixed template is made from
the model's OWN template (read from the GGUF header), so it always matches the model, and is given to llama-server with --chat-template-file.
The owner's hand-patched file for Qwen3.8-27B (handover model_screen/tmpl/qwen38_27b_patched.jinja) differs from the model's template by exactly this line;
tests/data/qwen38_27b_midsystem_patched.jinja is that file and the test checks that fix() reproduces it byte for byte."""
import hashlib
import os
import re

RAISE_LINE = "{{- raise_exception('System message must be at the beginning.') }}"
USER_TURN = "{{- '<|im_start|>user\\n' + content + '<|im_end|>' + '\\n' }}"
MARK = "System message must be at the beginning"


def needs_fix(text):
    """True when the template raises for a system message that is not first."""
    return isinstance(text, str) and MARK in text


def fix(text):
    """The template with the raise replaced by a user turn (the message's content, ChatML), or None when it has no such raise or it is not in the known shape
    (the raise must sit in a branch where `content` is already rendered, as in the Qwen3.8 templates; any other shape is left alone rather than guessed)."""
    if not needs_fix(text) or RAISE_LINE not in text:
        return None
    head, _, tail = text.partition(RAISE_LINE)
    if "set content = render_content(" not in head:
        return None
    out = head + USER_TURN + tail
    return None if MARK in out else out


def model_template(model):
    """The chat template string in the GGUF header (first shard), or None. Never raises."""
    try:
        import gguf_plan as gp
        t = gp.read_meta(gp.shards(model)[0]).get("tokenizer.chat_template")
        return t if isinstance(t, str) else None
    except Exception:
        return None


def fixed_template_file(model, cache_dir, write=True):
    """(path of the fixed template file, text) for a model whose template raises on a later system message, else (None, None). The file is named by the hash of the
    fixed text and written once, unless write=False (a dry run only names it) (forward slashes in the returned path: llama-server on Windows wants them in this option's value)."""
    t = model_template(model)
    f = fix(t) if needs_fix(t) else None
    if not f:
        return None, None
    d = os.path.join(cache_dir, "templates")
    if write:
        os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "midsystem-" + hashlib.sha1(f.encode("utf-8")).hexdigest()[:12] + ".jinja")
    if write and (not os.path.isfile(path) or open(path, encoding="utf-8").read() != f):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f)
        os.replace(tmp, path)
    return path.replace("\\", "/"), f


def user_chose_template(extra):
    """True when the user's own flags after -- already pick a template (they win)."""
    return any(x in ("--chat-template", "--chat-template-file", "--no-jinja") or x.startswith(("--chat-template=", "--chat-template-file=")) for x in extra)
