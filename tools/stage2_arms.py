#!/usr/bin/env python3
"""IQ3_XXS stage 2 arms (iq3_prompt_speed.md ranks 1 and 3): the four launcher settings of the conversation A B B A, checked with the launcher's own --dry-run
(no model run, no llama-server needed), and the PC commands for them.

  arm  what                                               launcher options          environment (PC, before the launcher)
  A    shipped                                            --ub=2048                  -
  B    rank 1: -ub 4096 with the output-row cap           --ub=4096                  LLAMA_N_OUTPUTS_MAX=1   (patch 0042, in the ci/llama-patched server)
  C    B plus q8_0 KV cache (rank 3, existing --fast=kv)  --ub=4096 --fast=kv        LLAMA_N_OUTPUTS_MAX=1
  D    A plus q8_0 KV cache (separates KV from the buffer) --ub=2048 --fast=kv       -

Order for the chain: A B C D D C B A (two runs per arm, the first and last arm are the pair that shows drift). Keep rules: docs/internal and the report
(parts/reports/iq3_prompt_speed.md, stage 2): B keeps if overall prompt >= +20 % and decode within -2 %; C keeps if prompt >= +3 %, decode >= 0 and the quality gate passes.

    python3 tools/stage2_arms.py MODEL.gguf [--ctx 65536] [--ram 24]      check the four arms with --dry-run and print the PC lines
Exit 1 when an arm's llama-server line lacks what it must have (or has what it must not)."""
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARMS = {
    "A": dict(opts=["--ub=2048"], env={}, must=["-ub 2048"], mustnot=["-ctk"]),
    "B": dict(opts=["--ub=4096"], env={"LLAMA_N_OUTPUTS_MAX": "1"}, must=["-ub 4096"], mustnot=["-ctk"]),
    "C": dict(opts=["--ub=4096", "--fast=kv"], env={"LLAMA_N_OUTPUTS_MAX": "1"}, must=["-ub 4096", "-ctk q8_0", "-ctv q8_0"], mustnot=[]),
    "D": dict(opts=["--ub=2048", "--fast=kv"], env={}, must=["-ub 2048", "-ctk q8_0", "-ctv q8_0"], mustnot=[]),
}
ORDER = "A B C D D C B A".split()


def dry_run(model, opts, ctx, ram):
    cmd = [sys.executable, os.path.join(ROOT, "bin", "moe_cache.py"), "server", model, "--ctx", str(ctx), "--ram", str(ram)] + opts + ["--dry-run"]
    env = dict(os.environ, LD_PRELOAD="")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    return r.returncode, r.stdout + r.stderr


def check(model, ctx=65536, ram=24, runner=None):
    """-> list of problems (empty when every arm's command line is as defined). runner(opts) -> (exit code, text) replaces the real launcher call (tests)."""
    bad = []
    for name, a in ARMS.items():
        rc, out = runner(a["opts"]) if runner else dry_run(model, a["opts"], ctx, ram)
        line = next((l for l in out.split("\n") if re.search(r"llama-server\S* -m ", l)), "")
        if rc != 0 or not line:
            bad.append(f"arm {name}: the dry run failed (exit {rc}): {out.strip().splitlines()[-1] if out.strip() else 'no output'}")
            continue
        bad += [f"arm {name}: '{m}' missing from the llama-server line" for m in a["must"] if m not in line]
        bad += [f"arm {name}: '{m}' must not be there" for m in a["mustnot"] if m in line]
    return bad


def pc_lines(model, ctx=65536, ram=24):
    out = []
    for name in sorted(ARMS):
        a = ARMS[name]
        env = "".join(f'$env:{k}="{v}"; ' for k, v in a["env"].items())
        out.append(f"arm {name}: {env}python bin\\moe_cache.py server {model} --ctx {ctx} --ram {ram} {' '.join(a['opts'])}   (conversation_bench_pair.ps1 starts the launcher itself: put the options in its -ExtraArgs and the variables in the session)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model"); ap.add_argument("--ctx", type=int, default=65536); ap.add_argument("--ram", type=float, default=24)
    a = ap.parse_args(argv)
    bad = check(a.model, a.ctx, a.ram)
    print("order:", " ".join(ORDER))
    print("\n".join(pc_lines(a.model, a.ctx, a.ram)))
    if bad:
        print("\n".join(bad), file=sys.stderr)
        return 1
    print("all four arms: the launcher's dry run gives the defined llama-server options")
    return 0


if __name__ == "__main__":
    sys.exit(main())
