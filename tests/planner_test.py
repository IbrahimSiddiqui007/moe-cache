#!/usr/bin/env python3
"""Planner / tuner / launcher checks on a real MoE GGUF (I13; CI runs it on Granite 1B). No GPU needed: VRAM is passed in.
   python3 tests/planner_test.py MODEL.gguf
Checks rules that must hold for any model: more VRAM never puts more expert layers on the CPU, the margin is respected, the micro-batch is one the
tuner knows, plan.py's answer is in range, and the launcher passes --vram to the tuner in GiB (audit A12) and --gpu-margin through (audit A05)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
import gguf_plan  # noqa: E402

model = sys.argv[1]
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

lay = gguf_plan.model_layout(model)
check(lay["n_layers"] > 0 and lay["expert_bytes_per_layer"] > 0, f"layout: {lay['n_layers']} layers, {lay['expert_bytes_per_layer'] / 1e6:.1f} MB experts per layer")

GIB = 2**30
prev = None
for gib in (2, 3, 4, 6, 8, 12, 16, 24):
    t = gguf_plan.tune(model, 8192, free=gib * GIB, want_cache=False)
    check(t["ub"] in (512, 1024, 2048), f"tune at {gib} GiB: ub {t['ub']} is a known micro-batch")
    ncm = min(t["n_cpu_moe"], lay["n_layers"])
    if prev is not None:
        check(ncm <= prev, f"tune at {gib} GiB: {ncm} CPU expert layers <= {prev} with less VRAM")
    prev = ncm

# a larger margin can only keep the same or more expert layers on the CPU
a = gguf_plan.tune(model, 8192, free=8 * GIB, want_cache=False, reserve_gb=0.5)
b = gguf_plan.tune(model, 8192, free=8 * GIB, want_cache=False, reserve_gb=3.0)
check(min(b["n_cpu_moe"], lay["n_layers"]) >= min(a["n_cpu_moe"], lay["n_layers"]), "a 3 GB margin keeps at least as many expert layers on the CPU as 0.5 GB")

# plan.py (the launcher's first opinion) answers in range
out = subprocess.run([sys.executable, os.path.join(HERE, "planner", "plan.py"), model, "--ctx", "8192", "--vram-gib", "8", "--ram-gib", "16", "--json"],
                     capture_output=True, text=True, timeout=120)
try:
    info = json.loads(out.stdout)
    check(0 <= info["n_cpu_moe"] <= lay["n_layers"], f"plan.py: n_cpu_moe {info['n_cpu_moe']} within 0..{lay['n_layers']}")
except Exception as e:
    check(False, f"plan.py --json: {e} {out.stderr[-300:]}")

# the launcher's dry run: --vram is GiB for the tuner too, --gpu-margin reaches the tuner
def dry(*extra):
    env = dict(os.environ)
    env.setdefault("MOE_CACHE_LIB", os.path.join(HERE, "build", "libggml-moe-cache.so"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "server", model, "--ram", "4", "--always-plugin", "--dry-run", *extra],
                       capture_output=True, text=True, timeout=300, env=env)
    return r.stdout + r.stderr
def ncmoe(text):
    import re
    m = re.search(r"--n-cpu-moe (\d+)", text)
    return int(m.group(1)) if m else None
for v in ("2", "4", "8"):
    want = min(gguf_plan.tune(model, 8192, free=float(v) * GIB, want_cache=False)["n_cpu_moe"], 999)
    got = ncmoe(dry("--vram", v))
    check(got == want, f"launcher --vram {v}: --n-cpu-moe {got} equals the tuner at {v} GiB ({want})")
lo, hi = ncmoe(dry("--vram", "8", "--gpu-margin", "0.2")), ncmoe(dry("--vram", "8", "--gpu-margin", "3"))
check(lo is not None and hi is not None and hi >= lo, f"launcher --gpu-margin 3 keeps at least as many CPU expert layers ({hi}) as 0.2 ({lo})")

# Windows core pinning (WINPIN): the launcher's parser of GetLogicalProcessorInformationEx records, on synthetic records (no Windows needed)
import importlib.util, struct  # noqa: E402
_sp = importlib.util.spec_from_file_location("moe_cache_launcher", os.path.join(HERE, "bin", "moe_cache.py"))
L = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(L)
def core(eff, mask, group=0):
    r = bytearray(48); struct.pack_into("<II", r, 0, 0, 48); r[9] = eff; struct.pack_into("<H", r, 30, 1); struct.pack_into("<QH", r, 32, mask, group)
    return bytes(r)
lunar = b"".join([core(1, 1 << i) for i in range(4)] + [core(0, 1 << i) for i in range(4, 8)])           # 4 P + 4 LPE, no hyper-threading
check(L.hybrid_masks(L.parse_core_records(lunar)) == (0xF, [4, 5, 6, 7]), "WINPIN: Lunar Lake 4P+4E -> -C 0xf, efficiency CPUs 4-7")
raptor = b"".join([core(1, 3 << (2 * i)) for i in range(6)] + [core(0, 1 << i) for i in range(12, 16)])    # 6 P with HT + 4 E
check(L.hybrid_masks(L.parse_core_records(raptor)) == (0x555, [12, 13, 14, 15]), "WINPIN: Raptor Lake 6P(HT)+4E -> -C 0x555 (one thread per P core)")
check(L.hybrid_masks(L.parse_core_records(b"".join(core(0, 1 << i) for i in range(8)))) is None, "WINPIN: one efficiency class -> no pinning")
check(L.parse_core_records(core(1, 1, group=1)) == [], "WINPIN: processor group 1 is ignored")
_g = set(); L.parse_core_records(lunar + core(1, 1, group=1), _g)
check(_g == {0, 1}, "WINPIN: the processor groups seen are reported (more than one: the launcher does not pin)")

# agent mode (AG1, I29): what --agent adds to the llama-server command, and the config snippets
ag = dry("--agent")
for want in ("-c 65536", "--jinja", "--ctx-checkpoints 8", "--alias ", "--slot-save-path ", "MOE_CACHE_HOST_RESERVE_MIB=1024"):
    check(want in ag, f"--agent: the command has '{want}'")
check("-c 32768" in dry("--agent", "--ctx", "32768"), "--agent --ctx 32768 keeps the given context")
check("--slot-save-path" not in dry("--agent", "--no-warm"), "--agent --no-warm: no warm-session slot directory")
check("--jinja" not in dry(), "no --agent: the command is unchanged (no --jinja added)")
check("-ctk q8_0 -ctv q8_0" in dry("--fast") and "-ctk" not in dry(), "--fast adds a q8_0 KV cache (I6); the exact default does not")
check(dry("--fast", "--", "-ctk", "f16").count("-ctk") == 1, "--fast keeps a -ctk given after the --")
check("MOE_CACHE_ROUTE_DROP=4" in dry("--fast=2") and "MOE_CACHE_ROUTE_DROP" not in dry("--fast"), "--fast=2 sets MOE_CACHE_ROUTE_DROP=4, --fast does not")
snip = L.agent_snippets(8099, "m1", 65536, None)
check("base_url: http://127.0.0.1:8099/v1" in snip and '"baseURL": "http://127.0.0.1:8099/v1"' in snip and "context_length: 65536" in snip,
      "agent snippets: Hermes and OpenCode blocks name the base URL and the context")

# starter profiles (I18): every shipped profile is a valid XUSE file without a fingerprint, named <arch>-<layers>x<experts>
import glob as _glob, re  # noqa: E402
k = gguf_plan.model_key(model)
check(k is not None and re.fullmatch(r"[A-Za-z0-9_.-]+-\d+x\d+", k or "") is not None, f"model_key: '{k}' has the form arch-LAYERSxEXPERTS")
for sp in _glob.glob(os.path.join(HERE, "profiles", "*-*x*.usage")):
    d = open(sp, "rb").read()
    n_l, n_e = struct.unpack("<ii", d[4:12])
    check(d[:4] == b"XUSE" and len(d) == 12 + 4 * n_l * n_e and str(n_e) == sp.rsplit("x", 1)[1].split(".")[0],
          f"starter profile {os.path.basename(sp)}: XUSE, {n_l} x {n_e}, no fingerprint")

print(f"planner checks: {len(fails)} failed")
sys.exit(1 if fails else 0)
