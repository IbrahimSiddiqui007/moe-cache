#!/usr/bin/env python3
"""tools/stage2_arms.py: the four IQ3_XXS stage 2 arms give the defined llama-server options in the launcher's --dry-run (synthetic MoE header, no model run).
   python3 tests/stage2_arms_test.py"""
import os, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, os.path.join(HERE, "tools"))
import launch_plan_test as lp   # noqa: E402
import stage2_arms as sa        # noqa: E402

with tempfile.TemporaryDirectory() as t:
    m = os.path.join(t, "synthetic-moe.gguf"); lp.make_synthetic_model(m)
    def runner(opts):
        r = lp.run_case("linux", ["--ram", "24", "--ctx", "65536", "--always-plugin"] + opts, [], m, t)
        return r["rc"], r["stdout"] + r["stderr"]
    bad = sa.check(m, runner=runner)
    for b in bad:
        print("  FAIL " + b)
    if not bad:
        print("  ok   arms A B C D: -ub 2048 / 4096 and the q8_0 KV flags are in the dry-run line as defined")
    lines = sa.pc_lines(m)
    ok = len(lines) == 4 and "LLAMA_N_OUTPUTS_MAX" in lines[1] and "LLAMA_N_OUTPUTS_MAX" in lines[2] and "LLAMA_N_OUTPUTS_MAX" not in lines[0]
    print(("  ok   " if ok else "  FAIL ") + "the PC lines carry the output-row cap on B and C only")
    sys.exit(0 if not bad and ok else 1)
