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
check(old == sorted(old), f"the table grows with the micro-batch at 64k (fixed 2026-11-02): {old}")
check(gp.parse_fit_params_compute_mib("CUDA0 3664 1840 1309 \nHost 68651 0 222 \n") == 1309.0 and gp.parse_fit_params_compute_mib("junk") is None,
      "llama-fit-params output parsed (Host line skipped, no figure -> None)")
cl = ("\\\\PC\\gpu process memory(pid_42_luid_0x0_phys_0)\\dedicated usage|1073741824;\\\\PC\\gpu process memory(pid_42_luid_0x0_phys_0)\\shared usage|2097152;"
      "\\\\PC\\gpu process memory(pid_7_luid_0x0_phys_0)\\shared usage|99999999")
check(gp.parse_gpu_process_counters(cl, 42) == (1024.0, 2.0) and gp.parse_gpu_process_counters("", 42) == (None, None),
      f"Windows GPU Process Memory counters parsed per process ({gp.parse_gpu_process_counters(cl, 42)})")
check(gp.start_gpu_mem_log(1, print, False, {}) is None, "GPU memory log is off unless MOE_CACHE_GPU_MEM_LOG is set")
os.environ.pop("MOE_CACHE_PLAN_RULE", None)
check(not gp.plan_rule_on(), "plan rule is off by default")

# AUD-031 / AUD-038: plan.py takes --vram-gib 0 as a value (not "not given": a fake nvidia-smi that reports 24 GB free must be ignored) and a header with
# head_count 0 ends with a message instead of a ZeroDivisionError traceback; gguf_plan.kv_bytes answers None for it
import json, stat, struct, subprocess, tempfile
def _s(x):
    b = x.encode(); return struct.pack("<Q", len(b)) + b
def _u32(k, v): return _s(k) + struct.pack("<I", 4) + struct.pack("<I", v)
def _mini(path, heads):
    kvs = [_s("general.architecture") + struct.pack("<I", 8) + _s("x"), _u32("x.block_count", 2), _u32("x.expert_count", 4), _u32("x.expert_used_count", 2),
           _u32("x.attention.head_count", heads), _u32("x.attention.head_count_kv", heads), _u32("x.embedding_length", 64)]
    tens = [("blk.0.ffn_up_exps.weight", 0), ("blk.1.ffn_up_exps.weight", 4096), ("blk.0.attn_q.weight", 8192)]
    t = b"".join(_s(n) + struct.pack("<I", 1) + struct.pack("<Q", 8) + struct.pack("<I", 0) + struct.pack("<Q", o) for n, o in tens)
    h = b"GGUF" + struct.pack("<IQQ", 3, len(tens), len(kvs)) + b"".join(kvs) + t
    h += b"\0" * (-len(h) % 32)
    open(path, "wb").write(h + b"\0" * 12288)
if os.name != "nt":
    d = tempfile.mkdtemp(prefix="plan-zero-")
    fake = os.path.join(d, "nvidia-smi")
    open(fake, "w").write("#!/bin/sh\necho 24000\n"); os.chmod(fake, 0o755)
    env = dict(os.environ, PATH=d + os.pathsep + os.environ.get("PATH", ""), MOE_PLAN_MINI="1")
    m = os.path.join(d, "ok.gguf"); _mini(m, 4)
    r = subprocess.run([sys.executable, os.path.join(HERE, "planner", "plan.py"), m, "--ctx", "1024", "--vram-gib", "0", "--ram-gib", "16", "--json"],
                       capture_output=True, text=True, env=env, timeout=60)
    try:
        info = json.loads(r.stdout)
        check(info["n_cpu_moe"] == 2 and info["free_vram_GiB"] == 0, f"plan.py --vram-gib 0 keeps every expert on the CPU (n_cpu_moe {info['n_cpu_moe']}, free {info['free_vram_GiB']})")
    except Exception as e:
        check(False, f"plan.py --vram-gib 0: {e} {r.stderr[-200:]}")
    z = os.path.join(d, "zero.gguf"); _mini(z, 0)
    r = subprocess.run([sys.executable, os.path.join(HERE, "planner", "plan.py"), z, "--vram-gib", "8", "--ram-gib", "16", "--json"],
                       capture_output=True, text=True, env=env, timeout=60)
    check(r.returncode != 0 and "ZeroDivisionError" not in r.stderr and "head_count" in r.stderr, f"plan.py: head_count 0 -> a message, no traceback ({r.stderr[-120:]!r})")
    check(gp.kv_bytes(z, 1024) is None, "gguf_plan.kv_bytes: head_count 0 -> None")
print(f"plan_rule_test: {len(fails)} failed")
sys.exit(1 if fails else 0)
