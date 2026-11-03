#!/usr/bin/env python3
"""CLEANUP-017: planner/plan.py prints its environment lines in the syntax of the shell in use (bash export, fish set -x, PowerShell $env:). Header-only synthetic model.
   python3 tests/plan_shell_test.py"""
import os, shlex, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "planner"))
import launch_plan_test as lpt  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

TMP = tempfile.mkdtemp(prefix="moe-shell-test-")
model = os.path.join(TMP, "my model's.gguf")
lpt.make_synthetic_model(model)

def plan(*args, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("SHELL", "FISH_VERSION")}
    e.update(env)
    r = subprocess.run([sys.executable, os.path.join(HERE, "planner", "plan.py"), model, "--vram-gib", "8", "--ram-gib", "16", *args], capture_output=True, text=True, env=e)
    return r.returncode, r.stdout

rc, out = plan("--shell", "bash")
gg = [l for l in out.splitlines() if l.startswith("export MOE_CACHE_GGUF=")]
check(rc == 0 and "export MOE_CACHE_STATS=1" in out and "export MOE_CACHE_RAM_GIB=16.0" in out, "bash: simple values stay as they were (export K=V)")
check(gg and shlex.split(gg[0][len("export "):]) == ["MOE_CACHE_GGUF=" + model], "bash: a path with a space and a quote survives a shell round trip")
rc, out = plan("--shell", "fish")
check("set -x MOE_CACHE_STATS '1'" in out and "export " not in out, "fish: set -x NAME 'value', no export")
check("set -x MOE_CACHE_GGUF '" + model.replace("\\", "\\\\").replace("'", "\\'") + "'" in out, "fish: quote inside the path is escaped with a backslash")
rc, out = plan("--shell", "powershell")
check("$env:MOE_CACHE_STATS = '1'" in out and "$env:MOE_CACHE_GGUF = '" + model.replace("'", "''") + "'" in out, "PowerShell: $env:NAME = 'value', quote doubled")
check("export " in plan(SHELL="/usr/bin/zsh")[1], "auto: zsh gets export lines")
check("set -x MOE_CACHE_STATS" in plan(SHELL="/usr/bin/fish")[1], "auto: $SHELL ending in fish gets set -x")
check("set -x MOE_CACHE_STATS" in plan(FISH_VERSION="4.0.0")[1], "auto: FISH_VERSION gets set -x")
rc, out = plan("--json", "--shell", "fish")
check(rc == 0 and out.lstrip().startswith("{"), "--json is unchanged by --shell")
sys.exit(1 if fails else 0)
