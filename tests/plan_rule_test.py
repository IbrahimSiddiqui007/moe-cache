#!/usr/bin/env python3
"""MOE_CACHE_PLAN_RULE helpers and the GPU memory log parsers (no model, no GPU): the rule's reserve never falls when -ub or the context grows,
llama-fit-params output and Windows GPU Process Memory counters are parsed, the log is off unless its switch is set."""
import os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
import gguf_plan as gp  # noqa: E402

fails = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)


lay = {"n_layers": 48, "expert_bytes_per_layer": 704 * 2**20, "other_bytes": 4 * 2**30}
ctxs = (8192, 32768, 65536, 131072, 262144)
check(all(gp.rule_buffer_mib("/none", lay, c, u2)[0] >= gp.rule_buffer_mib("/none", lay, c, u1)[0] for c in ctxs for u1, u2 in zip(gp.PLAN_UBS, gp.PLAN_UBS[1:])),
      "plan rule: reserve non-decreasing in -ub")
check(all(gp.formula_buffer_mib(c2, u) >= gp.formula_buffer_mib(c1, u) for u in gp.PLAN_UBS for c1, c2 in zip(ctxs, ctxs[1:])), "plan rule: formula non-decreasing in context")
old = [gp.compute_buffer_guess_mib(65536, u) for u in (1024, 2048, 3072, 4096)]
check(old != sorted(old), f"the old table is still not monotone at 64k (why the rule exists): {old}")
check(gp.parse_fit_params_compute_mib("CUDA0 3664 1840 1309 \nHost 68651 0 222 \n") == 1309.0 and gp.parse_fit_params_compute_mib("junk") is None,
      "llama-fit-params output parsed (Host line skipped, no figure -> None)")
cl = ("\\\\PC\\gpu process memory(pid_42_luid_0x0_phys_0)\\dedicated usage|1073741824;\\\\PC\\gpu process memory(pid_42_luid_0x0_phys_0)\\shared usage|2097152;"
      "\\\\PC\\gpu process memory(pid_7_luid_0x0_phys_0)\\shared usage|99999999")
check(gp.parse_gpu_process_counters(cl, 42) == (1024.0, 2.0) and gp.parse_gpu_process_counters("", 42) == (None, None),
      f"Windows GPU Process Memory counters parsed per process ({gp.parse_gpu_process_counters(cl, 42)})")
check(gp.start_gpu_mem_log(1, print, False, {}) is None, "GPU memory log is off unless MOE_CACHE_GPU_MEM_LOG is set")
os.environ.pop("MOE_CACHE_PLAN_RULE", None)
check(not gp.plan_rule_on(), "plan rule is off by default")
print(f"plan_rule_test: {len(fails)} failed")
sys.exit(1 if fails else 0)
