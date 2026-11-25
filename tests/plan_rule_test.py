#!/usr/bin/env python3
"""MOE_CACHE_PLAN_RULE helpers and the GPU memory log parsers (no model, no GPU): the rule's reserve never falls when -ub or the context grows,
llama-fit-params output and Windows GPU Process Memory counters are parsed, the log is off unless its switch is set."""
import os, re, sys

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
check(gp.plan_rule_active("/x/llama-fit-params") is False, "plan_rule_active: off by default even when llama-fit-params is found (LAUNCHER-1236)")
_saved_default = gp.PLAN_RULE_DEFAULT_ON
gp.PLAN_RULE_DEFAULT_ON = True
check(gp.plan_rule_active("/x/f") is True, "plan_rule_active: default on with llama-fit-params when PLAN_RULE_DEFAULT_ON")
check(gp.plan_rule_active(None) is False, "plan_rule_active: default on without llama-fit-params stays off")
os.environ["MOE_CACHE_PLAN_RULE"] = "0"
check(gp.plan_rule_active("/x/f") is False, "MOE_CACHE_PLAN_RULE=0 forces the old table even with the default on")
gp.PLAN_RULE_DEFAULT_ON = _saved_default
os.environ["MOE_CACHE_PLAN_RULE"] = "1"
check(gp.plan_rule_active(None) is True, "MOE_CACHE_PLAN_RULE=1 turns the rule on even without llama-fit-params")
os.environ.pop("MOE_CACHE_PLAN_RULE", None)
check("2.5 %" in gp.plan_table_lines([], 8192)[-1], "plan_table_lines footer names the decode cost per layer (2.5 %)")

# LAUNCHER-1236 follow-up: every tune() call takes the plan's fit_params, so the stock-vs-plugin choice, the fast-kv hint and the retunes
# use the same compute-buffer rule as the plan (rule off -> None -> the table, as before)
import inspect
check("fit_params" in inspect.signature(gp.fast_kv_hint).parameters, "fast_kv_hint takes fit_params")
src = open(os.path.join(HERE, "bin", "moe_cache.py")).read().splitlines()
tune_stmts = ["\n".join(src[i:i + 3]) for i, l in enumerate(src) if re.search(r"\btune\(model, ctx", l)]
with_fp = sum(1 for s in tune_stmts if "fit_params" in s)
check(with_fp >= 4, f"tune(model, ctx calls carrying fit_params: {with_fp} (the two plugin_for_partial_offload calls and the two oom retunes; the main plan ones pass it via **kw/**fkw)")

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
# LAUNCHER-1236 critic: tune() itself obeys the switch. The measured buffer (llama-fit-params) is stubbed, so no GPU and no binary: with the rule off the stub is never
# asked and the table decides; with MOE_CACHE_PLAN_RULE=1 (or the default flipped on) the stub's number decides, a bigger reserve leaves fewer expert layers on the GPU;
# =0 beats the default flag; the default flag without a fit_params binary changes nothing.
_MIB = 1 << 20
_rd = tempfile.mkdtemp(prefix="plan-rule-tune-")
_rm = os.path.join(_rd, "rule.gguf")
def _rs(x):
    b = x.encode(); return struct.pack("<Q", len(b)) + b
_rt = [("token_embd.weight", 300 * _MIB)] + [(f"blk.{i}.ffn_{k}_exps.weight", 240 * _MIB) for i in range(40) for k in ("gate", "up", "down")] + [(f"blk.{i}.attn_q.weight", 4 * _MIB) for i in range(40)]
_rk = [_rs("general.architecture") + struct.pack("<I", 8) + _rs("tst")]
_ro, _rtot = [], 0
for _n, _sz in _rt:
    _ro.append(_rtot); _rtot += _sz
_rh = b"GGUF" + struct.pack("<IQQ", 3, len(_rt), len(_rk)) + b"".join(_rk) + b"".join(_rs(n) + struct.pack("<IQIQ", 1, 8, 0, o) for (n, _), o in zip(_rt, _ro))
with open(_rm, "wb") as _f:
    _f.write(_rh + b"\0" * (-len(_rh) % 32)); _f.truncate(len(_rh) + (-len(_rh) % 32) + _rtot)       # sparse
_asked = []
_stub = {"mib": 100.0}
_orig_fp, _orig_pf = gp.fit_params_buffer_mib, gp.prefetch_fit_params
gp.fit_params_buffer_mib = lambda fp, path, ctx, ub, kv_q8=False, timeout=90: (_asked.append((fp, ub)), _stub["mib"])[1]
gp.prefetch_fit_params = lambda *a, **k: None
def _tune(fp="/x/llama-fit-params"):
    _asked.clear()
    return gp.tune(_rm, 8192, free=14 * 2**30, want_cache=False, fit_params=fp)
try:
    os.environ.pop("MOE_CACHE_PLAN_RULE", None)
    r_off = _tune()
    check(not _asked, f"tune: rule off by default, fit_params given -> llama-fit-params never asked ({len(_asked)} asks)")
    _stub["mib"] = 100.0; os.environ["MOE_CACHE_PLAN_RULE"] = "1"
    r_small = _tune()
    check(_asked and all(a[0] == "/x/llama-fit-params" for a in _asked), f"tune: MOE_CACHE_PLAN_RULE=1 asks the fit_params stub ({len(_asked)} asks)")
    _stub["mib"] = 6000.0
    r_big = _tune()
    check(r_big["n_cpu_moe"] > r_small["n_cpu_moe"], f"tune: a 6000 MiB measured reserve puts more layers on the CPU than 100 MiB ({r_big['n_cpu_moe']} vs {r_small['n_cpu_moe']})")
    os.environ["MOE_CACHE_PLAN_RULE"] = "0"; _stub["mib"] = 6000.0
    r_zero = _tune()
    check(not _asked and r_zero["n_cpu_moe"] == r_off["n_cpu_moe"], "tune: MOE_CACHE_PLAN_RULE=0 = the table (same answer as the default, stub not asked)")
    os.environ.pop("MOE_CACHE_PLAN_RULE", None)
    gp.PLAN_RULE_DEFAULT_ON = True
    r_def = _tune()
    check(_asked and r_def["n_cpu_moe"] == r_big["n_cpu_moe"], "tune: PLAN_RULE_DEFAULT_ON with fit_params -> the rule decides, as with =1")
    r_nofp = _tune(None)
    check(not _asked and r_nofp["n_cpu_moe"] == r_off["n_cpu_moe"], "tune: PLAN_RULE_DEFAULT_ON without fit_params -> the table, nothing asked")
finally:
    gp.fit_params_buffer_mib, gp.prefetch_fit_params = _orig_fp, _orig_pf
    gp.PLAN_RULE_DEFAULT_ON = _saved_default
    os.environ.pop("MOE_CACHE_PLAN_RULE", None)
    import shutil; shutil.rmtree(_rd, ignore_errors=True)      # the sparse model is ~28 GiB of apparent size
print(f"plan_rule_test: {len(fails)} failed")
sys.exit(1 if fails else 0)
