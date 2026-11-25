#!/usr/bin/env python3
"""LAUNCHER-1244 (R19): --exact turns off the model-card sampling defaults and the chat-template fix too, not only the token-changing modes.
Model-free: a 35B header twin (sparse file), fake machine, dry-runs only. The template stage is replaced by a stub that returns a fix file, so the check is
"does --exact keep the launcher from adding it" (the real fix is tested in chat_template_fix_test.py). Linux / macOS only (sparse file)."""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(HERE, "tests"), os.path.join(HERE, "planner"), os.path.join(HERE, "bin")]
import launch_plan_test as lp   # noqa: E402
import moe_cache as mc          # noqa: E402
from twin35b import make_twin   # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


if os.name == "nt":
    print("exact-defaults checks: sparse header twin, Linux / macOS only")
    sys.exit(0)

with tempfile.TemporaryDirectory() as tmp:
    twin = os.path.join(tmp, "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf")
    make_twin(twin)

    def plan(opts, extra=()):
        real = mc.template_stage
        state = {}

        def patch():
            def stub(a, extra_, model, dry=False):
                return [] if mc.arg_after(extra_, "--chat-template-file") else ["--chat-template-file", "FIXED.jinja", "--jinja"]
            mc.template_stage = stub
        try:
            r = lp.run_case("linux", ["--ctx", "65536", "--vram", "15"] + opts, list(extra), twin, tmp, call=patch)
        finally:
            mc.template_stage = real
        out = r["stdout"] + r["stderr"]
        cmd = next((ln for ln in out.splitlines() if "llama-server -m" in ln), "")
        return out, cmd

    out, cmd = plan([])
    check("--temp 1" in cmd and "sampling from the" in out, "default: the model-card sampling defaults are set")
    check("--chat-template-file FIXED.jinja" in cmd, "default: the chat-template fix is added (stubbed stage)")
    out, cmd = plan(["--exact"])
    check(cmd != "" and "--temp" not in cmd and "--top-p" not in cmd and "--top-k" not in cmd and "--presence-penalty" not in cmd and "sampling from the" not in out,
          "--exact --dry-run sets no model-card sampling defaults")
    check(cmd != "" and "--chat-template-file" not in cmd and "--jinja" not in cmd, "--exact --dry-run adds no chat-template fix")
    out, cmd = plan(["--exact"], ["--temp", "0.2"])
    check("--temp 0.2" in cmd, "--exact with your own --temp after --: yours stays")
    out, cmd = plan([], ["--chat-template-file", "MINE.jinja"])
    check("MINE.jinja" in cmd and "FIXED.jinja" not in cmd, "your own --chat-template-file still wins without --exact")

print("exact-defaults checks: " + (f"{len(fails)} failed" if fails else "all passed"))
sys.exit(1 if fails else 0)
