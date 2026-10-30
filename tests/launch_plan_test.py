#!/usr/bin/env python3
"""XA-2 golden test: `moe_cache.py server --dry-run` must print the same lines (argv, environment, reasons, warnings) before and after the launch decisions
move into one LaunchPlan. A matrix of option sets runs in-process on two FAKE machines (a Linux box with an 8 GB NVIDIA card, a Windows PC with a 16 GB AMD
card); every outside fact (memory, GPU list, commit, driver, CPU topology, the planner and llama-server child processes) is replaced, so the output depends
only on the launcher's code and on the model's GGUF header.
   python3 tests/launch_plan_test.py                       the same on a synthetic 18 GB MoE header (sparse file, no download): tests/launch_plan_golden_synth.json
   python3 tests/launch_plan_test.py MODEL.gguf            compare with tests/launch_plan_golden.json (skipped when the model is not the recorded one)
   python3 tests/launch_plan_test.py MODEL.gguf --record   write the golden file from the current code
The recorded model is Qwen3.6-35B-A3B (Ollama blob sha256-d372de8e...): reading its header is cheap, nothing is loaded."""
import contextlib, io, json, os, sys, tempfile, types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
sys.path.insert(0, os.path.join(HERE, "planner"))
GOLDEN = os.path.join(HERE, "tests", "launch_plan_golden.json")
GIB = 2**30

import moe_cache as mc            # noqa: E402
import gguf_plan                  # noqa: E402

LINUX_LIST = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB, 7600 MiB free)\n"
WIN_LIST = "Available devices:\n  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15000 MiB free)\n  Vulkan1: AMD Radeon(TM) Graphics (512 MiB, 400 MiB free)\n"
HELP_TEXT = "usage: llama-server\n  --cache-ram N\n  --spec-type draft-mtp\n"

MACHINES = {
    "linux": dict(windows=False, total=32 * GIB, avail=24e9, listing=LINUX_LIST, nvidia=True, free_vram=7600 * 2**20, total_vram=8188 * 2**20,
                  win_mem=(None, None), commit=(None, None), ded={}, driver="", cores=[]),
    "win": dict(windows=True, total=32e9, avail=24e9, listing=WIN_LIST, nvidia=False, free_vram=15000 * 2**20, total_vram=16304 * 2**20,
                win_mem=(32e9, 24e9), commit=(60e9, 20e9), ded={"AMD Radeon RX 9070 XT": 16 * GIB, "AMD Radeon(TM) Graphics": 512 * 2**20},
                driver="32.0.21001.1", cores=[(1, [0, 2, 4, 6, 8, 10, 12, 14]), (0, [16, 17, 18, 19])]),
}

# (name, launcher options before --, options after --)
CASES = [
    ("plain_ram12", ["--ram", "12"], []),
    ("fits_stock_ram32", ["--ram", "32"], []),
    ("fits_always_plugin", ["--ram", "32", "--always-plugin"], []),
    ("stock_flag", ["--ram", "12", "--stock"], []),
    ("fast", ["--ram", "12", "--fast"], []),
    ("fast2", ["--ram", "12", "--fast=2"], []),
    ("fast_kv", ["--ram", "12", "--fast=kv"], []),
    ("fast_fewer", ["--ram", "12", "--fast=fewer"], []),
    ("hybrid", ["--ram", "12", "--hybrid"], []),
    ("exact_128k", ["--ram", "12", "--ctx", "131072", "--exact"], []),
    ("ctx_8k", ["--ram", "12", "--ctx", "8192"], []),
    ("ctx_64k", ["--ram", "12", "--ctx", "65536"], []),
    ("ctx_128k", ["--ram", "12", "--ctx", "131072"], []),
    ("ctx_256k", ["--ram", "12", "--ctx", "262144"], []),
    ("ctx_256k_vram", ["--ram", "12", "--ctx", "262144", "--vram", "12"], []),
    ("ctx_128k_stock", ["--ram", "12", "--ctx", "131072", "--stock"], []),
    ("ctx_128k_ctk_user", ["--ram", "12", "--ctx", "131072"], ["-ctk", "f16", "-ctv", "f16"]),
    ("agent", ["--ram", "12", "--agent"], []),
    ("agent_ctx32k", ["--ram", "12", "--agent", "--ctx", "32768"], []),
    ("agent_no_warm", ["--ram", "12", "--agent", "--no-warm"], []),
    ("vram8", ["--ram", "12", "--vram", "8"], []),
    ("vram2_margin", ["--ram", "12", "--vram", "2", "--gpu-margin", "1"], []),
    ("ub1024", ["--ram", "12", "--ub", "1024"], []),
    ("ub2048_vram8", ["--ram", "12", "--ub", "2048", "--vram", "8"], []),
    ("ub_user_after", ["--ram", "12"], ["-ub", "256"]),
    ("gpu_cache_auto", ["--ram", "12", "--vram", "8", "--gpu-cache", "auto"], []),
    ("gpu_cache_2000", ["--ram", "12", "--vram", "8", "--gpu-cache", "2000"], []),
    ("n_cpu_moe_20", ["--ram", "12", "--n-cpu-moe", "20"], []),
    ("n_cpu_moe_0", ["--ram", "12", "--n-cpu-moe", "0"], []),
    ("no_pin_no_ecores", ["--ram", "12", "--no-pin", "--no-ecores"], []),
    ("max_ram_20", ["--ram", "12", "--max-ram", "20"], []),
    ("cache_ram_user", ["--ram", "12"], ["--cache-ram", "512"]),
    ("ram_low_cram0", ["--ram", "8"], []),
    ("mtp", ["--ram", "12", "--mtp"], []),
    ("no_mtp", ["--ram", "12", "--no-mtp"], []),
    ("port_alias_extra", ["--ram", "12", "--port", "8123"], ["--alias", "xyz", "--host", "0.0.0.0"]),
    ("no_ram_arg", [], []),
    ("profile_name", ["--ram", "12", "--profile", "work"], []),
]


def make_synthetic_model(path, n_layers=40, n_expert=256, ne0=2048, ne1=512):
    """A MoE GGUF with a real header and no data: the file is sparse (its size is what the launcher's decisions read), 40 layers x 3 expert tensors of
    [2048, 512, 256] Q4_K (18 GB in all, like a mid-size MoE), so every decision (fits or not, GPU layers, micro-batch) can be recorded without a download."""
    import struct
    def string(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    infos, off = [], 0
    def add(name, dims, ty, nbytes):
        nonlocal off
        infos.append((name, dims, ty, off))
        off += (nbytes + 31) // 32 * 32
    add("token_embd.weight", [ne0, 32000], 1, ne0 * 32000 * 2)
    for l in range(n_layers):
        add(f"blk.{l}.attn_norm.weight", [ne0], 0, ne0 * 4)
        add(f"blk.{l}.attn_q.weight", [ne0, ne0], 12, ne0 // 256 * 144 * ne0)
        add(f"blk.{l}.ffn_gate_inp.weight", [ne0, n_expert], 0, ne0 * n_expert * 4)
        for nm in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            add(f"blk.{l}.{nm}.weight", [ne0, ne1, n_expert], 12, ne0 // 256 * 144 * ne1 * n_expert)
    kv = [("general.architecture", 8, string("qwen35moe")), ("general.name", 8, string("synthetic-moe")),
          ("qwen35moe.block_count", 4, struct.pack("<I", n_layers)), ("qwen35moe.expert_count", 4, struct.pack("<I", n_expert)),
          ("qwen35moe.expert_used_count", 4, struct.pack("<I", 8)), ("qwen35moe.context_length", 4, struct.pack("<I", 262144)),
          ("qwen35moe.embedding_length", 4, struct.pack("<I", ne0)), ("qwen35moe.attention.head_count", 4, struct.pack("<I", 16)),
          ("qwen35moe.attention.head_count_kv", 4, struct.pack("<I", 2))]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(kv))
    for k, t, v in kv:
        head += string(k) + struct.pack("<I", t) + v
    for name, dims, ty, o in infos:
        head += string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, o)
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head)
        f.truncate(len(head) + off)        # sparse: the tensor data is never written


def fake_machine(m, tmp):
    """Patch every outside fact in the launcher module (restored by the caller)."""
    mc.WINDOWS = m["windows"]
    mc.CACHE_DIR = os.path.join(tmp, "cache")
    mc.KEY_FILE = os.path.join(mc.CACHE_DIR, "api.key")
    mc.mem_total = lambda: m["total"]
    mc._mem_for_server = lambda ram_gb: float(ram_gb) * 1e9 if ram_gb else m["avail"]
    mc.win_memory = lambda: m["win_mem"]
    mc.win_commit = lambda: m["commit"]
    mc.win_dedicated_vram = lambda: m["ded"]
    mc.win_driver_version = lambda name: m["driver"]
    mc.win_cores = lambda: m["cores"]
    mc.list_devices = lambda ls: m["listing"]
    mc.nvidia_reports = lambda: m["nvidia"]
    mc.ls_found_line = lambda ls: None
    mc.ls_looked = lambda: "(fake)"
    mc.ls_api = lambda ls: "3"
    mc.lib_api = lambda lib: "3"
    mc.ldd = lambda path: "\tlibggml-base.so.0 => /x/libggml-base.so.0.26.0 (0x0)\n\tlibggml-cpu.so.0 => /x/libggml-cpu.so.0 (0x0)\n"
    mc.stored_key = lambda create=False: "TESTKEY-" + "x" * 24
    mc.vk_cache_cap = lambda ls, model, listing=None: (None, None)
    mc.default_llama_server = lambda refresh=False: os.path.join(tmp, "llama-server")
    mc.python_cmd = lambda: "python3"
    gguf_plan.free_vram_bytes = lambda: m["free_vram"]
    gguf_plan.gpu_total_free_bytes = lambda: (m["total_vram"], m["free_vram"])
    real_access = os.access
    os.access = lambda p, mode, *a, **k: False if str(p).startswith("/sys/devices/cpu_") else real_access(p, mode, *a, **k)

    calls = []
    def fake_run(cmd, *a, **k):
        calls.append(list(cmd) if isinstance(cmd, (list, tuple)) else cmd)
        out = ""
        if isinstance(cmd, (list, tuple)) and len(cmd) > 1 and str(cmd[1]).endswith("plan.py"):
            out = '{"n_cpu_moe": 21}'
        elif isinstance(cmd, (list, tuple)) and "--help" in cmd:
            out = HELP_TEXT
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    mc.subprocess.run = fake_run
    return calls, real_access


def run_case(machine_name, opts, extra, model, tmp, flag="--dry-run"):
    m = MACHINES[machine_name]
    saved = (mc.WINDOWS, mc.CACHE_DIR, mc.KEY_FILE, mc.subprocess.run, os.access, os.environ.copy())
    for k in [k for k in os.environ if k.startswith("MOE_CACHE_") or k in ("LLAMA_API_KEY", "LLAMA_SERVER", "GGML_OP_OFFLOAD_MIN_BATCH")]:
        del os.environ[k]
    out, err = io.StringIO(), io.StringIO()
    rc = None
    calls, _ = fake_machine(m, tmp)
    ls = os.path.join(tmp, "llama-server")
    lib = os.path.join(tmp, "libggml-moe-cache.so")
    open(ls, "w").close(); open(lib, "w").close()
    mc.pick_lib = lambda ls_: lib
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = mc.server([model, "--llama-server", ls, flag, *opts] + (["--"] + extra if extra else []))
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 1
                print("SystemExit:", e.code, file=sys.stderr)
    finally:
        mc.WINDOWS, mc.CACHE_DIR, mc.KEY_FILE, mc.subprocess.run, os.access = saved[:5]
        os.environ.clear(); os.environ.update(saved[5])

    def norm(s):
        for a, b in ((model, "<MODEL>"), (tmp, "<TMP>"), (HERE, "<HERE>")):
            s = s.replace(a, b)
        return s
    return {"rc": rc, "stdout": norm(out.getvalue()), "stderr": norm(err.getvalue()), "calls": [norm(" ".join(map(str, c))) for c in calls]}


class FakeMachine(mc.Machine):
    """Fixed answers, no probes: what a test (or a caller that already measured the machine) passes to launch_plan."""
    def __init__(self, tmp, vram_mib=8188, free_mib=7600, avail=24e9):
        self.tmp, self.vram_mib, self.free_mib, self.avail = tmp, vram_mib, free_mib, avail
    windows = False
    def mem_total(self): return 32 * GIB
    def mem_cap(self): return 28 * GIB
    def mem_for_server(self, ram_gb): return float(ram_gb) * 1e9 if ram_gb else self.avail
    def list_devices(self, ls): return f"Available devices:\n  CUDA0: Fake GPU ({self.vram_mib} MiB, {self.free_mib} MiB free)\n"
    def nvidia_reports(self): return True
    def pick_lib(self, ls): return os.path.join(self.tmp, "libggml-moe-cache.so")
    def ls_found_line(self, ls): return None
    def ls_looked(self): return "(fake)"
    def ls_api(self, ls): return "3"
    def lib_api(self, lib): return "3"
    def load_env(self, ls, lib, method="auto"): return [("LD_PRELOAD", lib), ("MOE_CACHE_PRELOAD", "1")], "preload"
    def vk_cache_cap(self, ls, model, listing=None): return None, None
    def free_vram_bytes(self): return self.free_mib * 2**20
    def gpu_total_free_bytes(self): return (self.vram_mib * 2**20, self.free_mib * 2**20)
    def has_cpu_atom(self): return False
    def has_cpu_core(self): return False
    def run(self, cmd, **kw):
        out = '{"n_cpu_moe": 21}' if str(cmd[1]).endswith("plan.py") else "--cache-ram N\n"
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")


def injection_test(model):
    """launch_plan() takes its facts from the Machine it is given, prints nothing when quiet, and a bigger GPU changes the plan."""
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for f in ("llama-server", "libggml-moe-cache.so"):
            open(os.path.join(tmp, f), "w").close()
        saved = (mc.CACHE_DIR, mc.KEY_FILE, mc.default_llama_server, mc.WINDOWS)
        mc.CACHE_DIR = os.path.join(tmp, "cache"); mc.KEY_FILE = os.path.join(mc.CACHE_DIR, "api.key")
        mc.WINDOWS = False
        mc.default_llama_server = lambda refresh=False: os.path.join(tmp, "llama-server")
        try:
            plans = []
            for vram, free in ((8188, 7600), (24576, 23000)):
                a, extra, argv = mc.parse_server([model, "--llama-server", os.path.join(tmp, "llama-server"), "--ram", "12", "--no-pin", "--plan-json"])
                out = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                    plan = mc.launch_plan(a, extra, argv, machine=FakeMachine(tmp, vram, free), quiet=True)
                plans.append(plan)
                ok = out.getvalue() == "" and plan.engine == "plugin" and plan.reasons and plan.cmd[0].endswith("llama-server") and "--n-cpu-moe" in plan.cmd
                print("  " + ("ok  " if ok else "FAIL") + f" injected machine ({vram} MiB GPU): engine {plan.engine}, --n-cpu-moe {plan.n_cpu_moe}, -ub {plan.ub}, nothing printed")
                bad += not ok
            ok = int(plans[1].n_cpu_moe) < int(plans[0].n_cpu_moe)
            print("  " + ("ok  " if ok else "FAIL") + f" a bigger GPU keeps more expert layers on the GPU ({plans[0].n_cpu_moe} -> {plans[1].n_cpu_moe} on the CPU)")
            bad += not ok
        finally:
            mc.CACHE_DIR, mc.KEY_FILE, mc.default_llama_server, mc.WINDOWS = saved
    return bad


def fingerprint(model):
    import hashlib
    h = hashlib.sha256()
    with open(model, "rb") as f:
        h.update(f.read(1 << 20))
    return f"{os.path.getsize(model)}:{h.hexdigest()[:16]}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    record = "--record" in sys.argv
    global GOLDEN
    synth_dir = None
    if args:
        model = args[0]
    else:                                      # no model: the synthetic one (CI, any machine); its golden file is tests/launch_plan_golden_synth.json
        synth_dir = tempfile.mkdtemp(prefix="launch-plan-")
        model = os.path.join(synth_dir, "synthetic-moe.gguf")
        make_synthetic_model(model)
        GOLDEN = os.path.join(HERE, "tests", "launch_plan_golden_synth.json")
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mname in MACHINES:
            for cname, opts, extra in CASES:
                results[f"{mname}/{cname}"] = run_case(mname, opts, extra, model, tmp)
    fp = fingerprint(model)
    inj_bad = injection_test(model)
    # --plan-json is the same plan as --dry-run: the lines the page shows (dry_run_lines) are the dry run's lines in the same order, nothing else is printed
    consistent_bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for mname in MACHINES:
            for cname, opts, extra in CASES:
                r = run_case(mname, opts, extra, model, tmp, flag="--plan-json")
                base = results[f"{mname}/{cname}"]
                want = [l for l in (base["stdout"] + base["stderr"]).splitlines() if l.strip()]
                if base["rc"] not in (0, None):      # a refusal (SystemExit) is the same refusal, with the same message, and no plan
                    msg = lambda x: [l for l in x["stderr"].splitlines() if l.startswith("SystemExit")]
                    if r["rc"] != base["rc"] or msg(r) != msg(base):
                        consistent_bad += 1
                        print("  FAIL", f"{mname}/{cname}", "--plan-json refuses differently from --dry-run")
                    continue
                try:
                    plan = json.loads(r["stdout"])
                    got = [l.replace(tmp, "<TMP>").replace(model, "<MODEL>").replace(HERE, "<HERE>") for l in plan["dry_run_lines"]]
                    ok = got == want and r["stderr"] == "" and plan["engine"] in ("stock", "plugin") and plan["argv"]
                except Exception as e:
                    ok = False
                    got = [str(e)]
                if not ok:
                    consistent_bad += 1
                    print("  FAIL", f"{mname}/{cname}", "--plan-json differs from --dry-run")
                    for x, y in zip(want + [""] * len(got), got + [""] * len(want)):
                        if x != y:
                            print("      dry-run :", x[:200]); print("      plan-json:", y[:200]); break
    consistent_bad += inj_bad
    if consistent_bad:
        print(f"{consistent_bad} checks failed (--plan-json vs --dry-run, injected machine)")
    if record:
        json.dump({"model": fp, "cases": results}, open(GOLDEN, "w"), indent=1, sort_keys=True)
        print(f"recorded {len(results)} cases for model {fp}")
        return 1 if consistent_bad else 0
    gold = json.load(open(GOLDEN))
    bad = 0
    # every case must at least run (no traceback) on any model
    for k, r in results.items():
        if "Traceback" in r["stderr"]:
            print("  FAIL", k, "traceback"); bad += 1
    if gold["model"] != fp:
        print(f"  note: the golden file is for model {gold['model']}, this one is {fp}: smoke test only ({len(results)} cases ran)")
        return 1 if (bad or consistent_bad) else 0
    for k, r in sorted(results.items()):
        g = gold["cases"].get(k)
        if g != r:
            bad += 1
            print("  FAIL", k)
            for key in ("rc", "stdout", "stderr", "calls"):
                if g and g[key] != r[key]:
                    a, b = (g[key] if isinstance(g[key], str) else "\n".join(g[key])).splitlines(), (r[key] if isinstance(r[key], str) else "\n".join(r[key])).splitlines()
                    for x, y in zip(a + [""] * len(b), b + [""] * len(a)):
                        if x != y:
                            print(f"      {key}: golden  {x[:240]}\n      {key}: now     {y[:240]}")
                            break
    print(f"{len(results) - bad} of {len(results)} cases identical to the golden file; --plan-json agrees with --dry-run in {len(results) - consistent_bad}")
    return 1 if (bad or consistent_bad) else 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        import shutil
        for d in [d for d in os.listdir(tempfile.gettempdir()) if d.startswith('launch-plan-')]:
            shutil.rmtree(os.path.join(tempfile.gettempdir(), d), ignore_errors=True)
    sys.exit(rc)
