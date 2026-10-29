#!/usr/bin/env python3
"""`moe_cache.py diffusion` (S3 of the diffusion plan): the command and environment it builds for llama-diffusion-cli. No model, no llama.cpp: a fake
   cli file and a small fake gguf are enough because --dry-run only prints. Same checks on Linux and Windows.
   python3 tests/diffusion_test.py"""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH = os.path.join(HERE, "bin", "moe_cache.py")
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

tmp = tempfile.mkdtemp()
model = os.path.join(tmp, "m.gguf"); open(model, "wb").write(b"GGUF" + b"\0" * 4096)     # 4 KB: "fits" in any memory
cli = os.path.join(tmp, "llama-diffusion-cli" + (".exe" if os.name == "nt" else "")); open(cli, "w").write("x")
lib = os.path.join(tmp, "plugin.so"); open(lib, "wb").write(b"moe-cache-api=3\0")

def run(*args, env=None):
    e = dict(os.environ, XDG_CACHE_HOME=tmp, MOE_CACHE_LIB=lib)
    e.pop("MOE_CACHE_MAX_RAM_GB", None); e.update(env or {})
    r = subprocess.run([sys.executable, LAUNCH, "diffusion", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, env=e)
    return r.returncode, r.stdout + r.stderr

# a model that fits: stock, no plugin variables
rc, out = run(model, "-p", "hi", "--cli", cli, "--dry-run")
check(rc == 0 and "running stock llama-diffusion-cli" in out and "MOE_CACHE" not in out.split("moe-cache: running")[-1], "model fits: stock run, no plugin environment")
# forced plugin: environment, memory settings, experts on the CPU side
rc, out = run(model, "-p", "hi", "--cli", cli, "--always-plugin", "--ram", "3", "--max-ram", "20", "--load-method", "path", "--dry-run")
last = [l for l in out.splitlines() if "-ngl" in l][-1] if "-ngl" in out else ""; last = last.replace(chr(34), "")      # PowerShell form: $env:X="v"
check(rc == 0 and "GGML_BACKEND_PATH=" in last and "MOE_CACHE_RAM_GIB=2.794" in last and "MOE_CACHE_MAX_RAM_GB=20" in last and "MOE_CACHE_GGUF=" in last, "plugin run: backend path, cache size (3 GB = 2.794 GiB), cap, gguf")
check(" -ngl 0 " in last and "-ub 256 -b 256 -c 256" in last and "--diffusion-steps 64" in last and "--diffusion-block-length 32" in last, "default command: -ngl 0, canvas 256, 64 steps, blocks of 32")
check("canvas 256 tokens" in out and "64 steps" in out and "blocks of 32" in out, "prints canvas and steps")
check("-p hi" in last and "-s 42" in last, "prompt and seed passed")
# RND1 style: timestep schedule, no block length
rc, out = run(model, "-p", "hi", "--cli", cli, "--stock", "--eps", "0.001", "--algorithm", "1", "--temp", "0.5", "--canvas", "512", "--steps", "100", "--dry-run")
last = [l for l in out.splitlines() if "-ngl" in l][-1] if "-ngl" in out else ""; last = last.replace(chr(34), "")      # PowerShell form: $env:X="v"
check(rc == 0 and "--diffusion-eps 0.001" in last and "--diffusion-block-length" not in last and "--diffusion-algorithm 1" in last and "-ub 512" in last, "eps schedule: no block length, algorithm 1, canvas 512")
# llama.cpp's own asserts, caught before the model loads
rc, out = run(model, "-p", "hi", "--cli", cli, "--canvas", "250", "--dry-run")
check(rc != 0 and "multiple of --block" in out, "canvas not a multiple of the block: refused")
rc, out = run(model, "-p", "hi", "--cli", cli, "--steps", "63", "--dry-run")
check(rc != 0 and "multiple of the number of blocks" in out, "steps not divisible by the blocks: refused")
# --n-cpu-moe is ignored by llama-diffusion-cli: say so
rc, out = run(model, "-p", "hi", "--cli", cli, "--stock", "--dry-run", "--", "--n-cpu-moe", "20")
check(rc == 0 and "ignores --n-cpu-moe" in out, "--n-cpu-moe after -- gets a warning")
rc, out = run(model, "-p", "hi", "--cli", os.path.join(tmp, "nope"), "--stock")
check(rc != 0 and "llama-diffusion-cli not found" in out, "missing llama-diffusion-cli: clear message")
rc, out = run(model, "--stock", "--always-plugin", "--cli", cli, "--dry-run")
check(rc != 0 and "cannot go with" in out, "--stock with --always-plugin refused")
print("FAIL: %d" % len(fails) if fails else "all ok")
sys.exit(1 if fails else 0)
