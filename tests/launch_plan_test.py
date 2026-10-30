#!/usr/bin/env python3
"""XA-2 golden test: `moe_cache.py server --dry-run` must print the same lines (argv, environment, reasons, warnings) before and after the launch decisions
move into one LaunchPlan. A matrix of option sets runs in-process on two FAKE machines (a Linux box with an 8 GB NVIDIA card, a Windows PC with a 16 GB AMD
card); every outside fact (memory, GPU list, commit, driver, CPU topology, the planner and llama-server child processes) is replaced, so the output depends
only on the launcher's code and on the model's GGUF header.
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


def run_case(machine_name, opts, extra, model, tmp):
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
                rc = mc.server([model, "--llama-server", ls, "--dry-run", *opts] + (["--"] + extra if extra else []))
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


def fingerprint(model):
    import hashlib
    h = hashlib.sha256()
    with open(model, "rb") as f:
        h.update(f.read(1 << 20))
    return f"{os.path.getsize(model)}:{h.hexdigest()[:16]}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    record = "--record" in sys.argv
    if not args:
        raise SystemExit(__doc__)
    model = args[0]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mname in MACHINES:
            for cname, opts, extra in CASES:
                results[f"{mname}/{cname}"] = run_case(mname, opts, extra, model, tmp)
    fp = fingerprint(model)
    if record:
        json.dump({"model": fp, "cases": results}, open(GOLDEN, "w"), indent=1, sort_keys=True)
        print(f"recorded {len(results)} cases for model {fp}")
        return 0
    gold = json.load(open(GOLDEN))
    bad = 0
    # every case must at least run (no traceback) on any model
    for k, r in results.items():
        if "Traceback" in r["stderr"]:
            print("  FAIL", k, "traceback"); bad += 1
    if gold["model"] != fp:
        print(f"  note: the golden file is for model {gold['model']}, this one is {fp}: smoke test only ({len(results)} cases ran)")
        return 1 if bad else 0
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
    print(f"{len(results) - bad} of {len(results)} cases identical to the golden file")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
