#!/usr/bin/env python3
"""LAUNCHER-1152: the MoE planner sizes the GPU compute buffer from llama-fit-params on the user's machine (same build as llama-server), not from the table
measured on one model (Flash-Next) on our PC; without the tool it keeps the table and the launcher says so in one line. No model, no GPU, no tool: the
model layouts are the real header figures (35B IQ3_XXS and Flash-Next IQ3_XXS heads, sessions/bugfix_2026-11-09/files/pretag_logs/*_head64M.bin) and the
fake llama-fit-params answers with the compute buffers the PC's own server logs printed (room_probe_buffer_lines.txt, 2026-11-13; night chain 2026-11-07).
   python3 tests/compute_buffer_test.py"""
import os, sys, tempfile, types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
sys.path.insert(0, os.path.join(HERE, "bin"))
import gguf_plan as gp  # noqa: E402

TMP = tempfile.mkdtemp(prefix="moe-cbuf-test-")
os.environ["MOE_CACHE_FIT_CACHE"] = os.path.join(TMP, "cache", "fit.json")
os.environ.pop("MOE_CACHE_FIT_PARAMS", None)
os.environ.pop("MOE_CACHE_PLAN_RULE", None)
fails = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)


GiB = 2**30
# header figures (gguf_plan.model_layout of the 64 MiB heads) and KV bytes as kv_bytes() sizes them (q8_0, 4 slots)
MODELS = {
    "35b.gguf": ({"n_layers": 40, "expert_bytes_per_layer": 277715353.6, "other_bytes": 2091543040, "table_bytes": 0, "total_bytes": 13200157184, "mtp_blocks": 0},
                 {32768: 591.25 * 2**20, 16384: 295.6 * 2**20}),
    "fn.gguf": ({"n_layers": 48, "expert_bytes_per_layer": 894020266.67, "other_bytes": 4113241600, "table_bytes": 0, "total_bytes": 75840000000, "mtp_blocks": 0},
                {65536: 1224 * 2**20, 32768: 612 * 2**20}),
}
# what llama-server itself printed on the PC (Vulkan1 compute buffer size, MiB): 35B 32k q8_0 -ub 4096 = 864.03 / 864.16 (plugin and stock arms alike);
# the old-plan arm printed 432.08 (taken here as -ub 2048, half the 4096 figure); Flash-Next 64k -ub 4096 = 4660 (f16 KV, night chain 2026-11-07, item 15)
SERVER_LOG = {("35b.gguf", 32768, 4096): 864.0, ("35b.gguf", 32768, 2048): 432.0, ("fn.gguf", 65536, 4096): 4660.0,
              ("35b.gguf", 32768, 1024): 216.0, ("35b.gguf", 32768, 512): 108.0,          # the rest of the ladder: made up (the fake must answer every candidate)
              ("fn.gguf", 65536, 2048): 2021.0, ("fn.gguf", 65536, 1024): 1010.0, ("fn.gguf", 65536, 512): 505.0}

real_layout, real_kv, real_meta = gp.model_layout, gp._kv_or_flat, gp.sampling_meta
gp.model_layout = lambda p: dict(MODELS[os.path.basename(p)][0])
gp._kv_or_flat = lambda p, ctx, kv_q8=False: MODELS[os.path.basename(p)][1][ctx]
gp.sampling_meta = lambda p: {"arch": "qwen35moe"}
gp.shards = lambda p: [p]
gp.gpu_identity = lambda: "vulkan:test-gpu"

calls = []


def fake_run(cmd, **kw):
    calls.append(cmd)
    m, ctx, ub = os.path.basename(cmd[cmd.index("-m") + 1]), int(cmd[cmd.index("-c") + 1]), int(cmd[cmd.index("-ub") + 1])
    v = SERVER_LOG.get((m, ctx, ub))
    out = f"Vulkan1 12190 340 {int(v)} \nHost 397 0 384 \n" if v else "error: failed\n"
    return types.SimpleNamespace(stdout=out, returncode=0 if v else 1)


gp.subprocess.run = fake_run
srv_dir = os.path.join(TMP, "llama"); os.makedirs(srv_dir)
ls = os.path.join(srv_dir, "llama-server"); open(ls, "w").close()
tool = os.path.join(srv_dir, "llama-fit-params"); open(tool, "w").close()
for name in MODELS:
    open(os.path.join(TMP, name), "wb").write(b"GGUF" + bytes(60))
m35, mfn = os.path.join(TMP, "35b.gguf"), os.path.join(TMP, "fn.gguf")
ROOM = 12.83 * GiB          # the room the branch planned with on the PC (2e8be30 dry run: --n-cpu-moe 23)


def plan(model, ctx, fit_params=None, **kw):
    gp._FIT_CACHE.clear()
    return gp.tune(model, ctx, free=kw.pop("free", ROOM), kv_q8=True, want_cache=False, reserve_gb=0.5, long_prompt=True, fit_params=fit_params, **kw)


# 1) the old plan is reproduced without a tool, and says where its figure came from
old = plan(m35, 32768)
check(old["n_cpu_moe"] == 23 and old["ub"] == 4096 and old["compute_mib"] == 4660 and old["compute_source"] == "table",
      f"35B 32k without llama-fit-params: the table's 4660 MiB, --n-cpu-moe 23 as on the PC ({old['n_cpu_moe']}, {old['compute_mib']}, {old['compute_source']})")
# 2) with the tool: the server's own 864 MiB, 14 more expert layers on the GPU
new = plan(m35, 32768, tool)
check(new["n_cpu_moe"] == 9 and new["ub"] == 4096 and new["compute_mib"] == 864 and new["compute_source"] == "llama-fit-params",
      f"35B 32k measured: 864 MiB, --n-cpu-moe 9 (PC single runs: 4 -> 71.7, 12 -> 71.2, 23 -> 53.1 tok/s) ({new['n_cpu_moe']}, {new['compute_mib']})")
cmd = next(c for c in calls if c[c.index("-ub") + 1] == "4096")
check(cmd[0] == tool and cmd[cmd.index("-c") + 1] == "32768" and "q8_0" in cmd and cmd[cmd.index("-b") + 1] == "4096",
      "the tool is asked for this context, -ub (with -b raised to match) and KV type")
# 3) remembered: a second start does not run the tool again
n = len(calls)
again = plan(m35, 32768, tool)
check(again["n_cpu_moe"] == 9 and len(calls) == n, f"a second start reads the remembered figure ({len(calls) - n} tool runs)")
# 4) Flash-Next 64k: the table's one measurement is this model's own figure, so the plan does not move
fo, fnew = plan(mfn, 65536, free=ROOM), plan(mfn, 65536, tool, free=ROOM)
check(fo["n_cpu_moe"] == fnew["n_cpu_moe"] and fnew["compute_mib"] == 4660 and fnew["compute_source"] == "llama-fit-params",
      f"Flash-Next 64k: measured 4660 MiB = the table's figure, same plan ({fo['n_cpu_moe']} -> {fnew['n_cpu_moe']})")
# 5) the tool fails (no figure): the table, and the source says so
bad = plan(m35, 16384, tool, free=ROOM)
check(bad["compute_source"] == "table" and bad["compute_mib"] == gp.compute_buffer_guess_mib(16384, bad["ub"]),
      f"no figure from the tool (16k not in the fake's answers): the table's number ({bad['compute_mib']} MiB at -ub {bad['ub']})")
# 6) the table still decides the candidates: -ub 4096 only in long-prompt mode, a user's -ub is kept
short = gp.tune(m35, 32768, free=ROOM, kv_q8=True, want_cache=False, reserve_gb=0.5, fit_params=tool)
check(short["ub"] == 2048 and short["compute_mib"] == 432 and short["compute_source"] == "llama-fit-params",
      f"without long-prompt mode the ladder starts at 2048 (measured {short['compute_mib']} MiB)")
asked = gp.tune(m35, 32768, free=ROOM, kv_q8=True, want_cache=False, reserve_gb=0.5, fit_params=tool, ub=4096)
check(asked["ub"] == 4096 and asked["compute_mib"] == 864, "a -ub the user asked for is measured too")
# 7) the plan rule (MOE_CACHE_PLAN_RULE=1) keeps its own path
os.environ["MOE_CACHE_PLAN_RULE"] = "1"
r = plan(m35, 32768, tool)
check(r["compute_source"] == "rule", "MOE_CACHE_PLAN_RULE=1: the rule's own lookup, not this one")
os.environ.pop("MOE_CACHE_PLAN_RULE")

# 8) which tool: the one of llama-server's own build; PATH only when asked; MOE_CACHE_FIT_PARAMS names one or turns it off
check(gp.find_fit_params(ls, use_path=False) == tool, "llama-fit-params next to llama-server is found")
lone = os.path.join(TMP, "alone"); os.makedirs(lone); open(os.path.join(lone, "llama-server"), "w").close()
pathdir = os.path.join(TMP, "pathbin"); os.makedirs(pathdir); pt = os.path.join(pathdir, "llama-fit-params"); open(pt, "w").close(); os.chmod(pt, 0o755)
old_path = os.environ.get("PATH", "")
os.environ["PATH"] = pathdir + os.pathsep + old_path
check(gp.find_fit_params(os.path.join(lone, "llama-server"), use_path=False) is None, "a llama-fit-params of another build on PATH is not used by the plan")
check(gp.find_fit_params(os.path.join(lone, "llama-server")) == pt, "the multi-GPU check still looks on PATH")
os.environ["PATH"] = old_path
os.environ["MOE_CACHE_FIT_PARAMS"] = "off"
check(gp.find_fit_params(ls, use_path=False) is None and gp.fit_params_off(), "MOE_CACHE_FIT_PARAMS=off: no measurement")
os.environ["MOE_CACHE_FIT_PARAMS"] = pt
check(gp.find_fit_params(os.path.join(lone, "llama-server"), use_path=False) == pt, "MOE_CACHE_FIT_PARAMS=PATH names the tool")
os.environ.pop("MOE_CACHE_FIT_PARAMS")

# 9) the launcher's plain line (rule C16) for each source
import moe_cache as mc  # noqa: E402
l1 = mc.compute_buffer_line(new, tool)
check(l1 and "864 MiB at -ub 4096" in l1 and "measured" in l1 and "this GPU" in l1, f"measured line: {l1}")
l2 = mc.compute_buffer_line(old, None)
check(l2 and "4660 MiB" in l2 and "estimate" in l2 and "no llama-fit-params next to llama-server" in l2 and "errs high" in l2, f"fallback line: {l2}")
check("gave no figure" in mc.compute_buffer_line(bad, tool), "fallback line when the tool ran but gave nothing")
os.environ["MOE_CACHE_FIT_PARAMS"] = "off"
check("MOE_CACHE_FIT_PARAMS=off" in mc.compute_buffer_line(old, None), "fallback line names the switch when it is off")
os.environ.pop("MOE_CACHE_FIT_PARAMS")
check(mc.compute_buffer_line(r, tool) is None and mc.compute_buffer_line(None, None) is None, "no line under the plan rule (it prints its own) or without a tuner answer")
check(mc.plan_fit_params(ls) == tool and mc.plan_fit_params(os.path.join(lone, "llama-server")) is None, "Machine.fit_params: the tool next to llama-server, else None")

print(f"compute_buffer_test: {len(fails)} failed")
sys.exit(1 if fails else 0)
