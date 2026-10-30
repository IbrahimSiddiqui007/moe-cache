#!/usr/bin/env python3
"""bench-report (R2-1): report sections and order, backend line, privacy, fit rule, identity, cold/warm stamp, run order. No model, no server:
the doctor, the launcher's plan, the runs and llama-server's device list are replaced by fakes. Same file on Linux and Windows CI."""
import json, os, socket, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
import moe_cache_bench as mb  # noqa: E402
import moe_cache_ui as ui  # noqa: E402

FAILS = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


# 1. backend line from --list-devices
dev = lambda *names: "Available devices:\n" + "".join(f"  {n}: Some GPU (16304 MiB, 15890 MiB free)\n" for n in names)
for names, want in ((("CUDA0",), "CUDA"), (("Vulkan0",), "Vulkan"), (("ROCm0",), "ROCm-HIP"), (("SYCL0",), "SYCL"), ((), "CPU only"),
                    (("Vulkan0", "Vulkan1"), "Vulkan"), (("CUDA0", "Vulkan0"), "CUDA + Vulkan")):
    check(mb.backend_of(dev(*names)) == want, f"backend: {list(names) or 'no device'} -> {want}")
for names, want in ((("MTL0",), "Metal"), (("MUSA0",), "MUSA"), (("CANN0",), "CANN"), (("RPC0",), "RPC (remote)"), (("Foo0",), "Foo")):
    check(mb.backend_of(dev(*names)) == want, f"backend: {list(names)} -> {want}")
check(mb.backend_of("ggml_vulkan: Found 1 devices:\nAvailable devices:\n  CPU: x\n") == "CPU only", "backend: a CPU line is not a GPU")
check(mb.backend_of("") == "UNKNOWN - please state it" and mb.backend_of("error: unknown option") == "UNKNOWN - please state it", "backend: unreadable -> UNKNOWN")

# 2. order and plan summary
check(mb.order(2) == ["stock", "moe-cache", "moe-cache", "stock"] * 2, "order: stock, moe-cache, moe-cache, stock per pair")
line = "env MOE_CACHE_SIZE_GIB=auto llama-server -m x.gguf -c 8192 -ngl 99 --n-cpu-moe 30 -ub 512 -ctk q8_0"
check(mb.plan_summary(line) == "--n-cpu-moe 30, -ngl 99, -c 8192, -ub 512, -ctk q8_0, cache auto GiB", "plan summary from the launcher's command line")

# 3. identity verdicts
run = lambda mode, text, dec=10.0: {"mode": mode, "decode": dec, "prefill": 100.0, "text": text, "plugin_active": mode == "moe-cache"}
s = mb.summarize([run("stock", "a"), run("moe-cache", "a", 20.0), run("moe-cache", "a", 22.0), run("stock", "a")], exact=True)
check(s["identity"] == "IDENTICAL" and s["decode_x"] == 2.1 and s["moe-cache"]["decode_runs"] == [20.0, 22.0], "identity IDENTICAL, ratio 2.1x")
check(mb.summarize([run("stock", "a"), run("moe-cache", "b")], exact=True)["identity"] == "DIFFERENT", "identity DIFFERENT")
check(mb.summarize([run("stock", "a"), run("moe-cache", "b")], exact=False)["identity"] == "n/a (approximate mode)", "identity n/a for --fast")
check(mb.summarize([run("stock", "a"), {"mode": "moe-cache", "error": "x"}], exact=True)["identity"].startswith("not checked"), "identity not checked when a side failed")

# 4. the whole command with fakes: sections in order, privacy, files, exit codes
tmp = tempfile.mkdtemp(prefix="bench report ")                     # a space in the path
user = "zqxüser"; host = socket.gethostname()
os.environ["USER"] = os.environ["USERNAME"] = user
model_dir = os.path.join(tmp, f"{user} models")
os.makedirs(model_dir)
model = os.path.join(model_dir, "Flash-Next-Q2_0.gguf")
open(model, "wb").write(b"GGUF" + b"\0" * 1024)
calls = []
saved = (mb.doctor_facts, mb.launcher_plan, mb.run_one, mb.server_info, ui.flush_model_cache, ui.header_quant, mb.mc.port_busy)


def fake_run(model_, mode, a, port, text="Once upon a time"):
    calls.append(mode)
    return {"mode": mode, "decode": 10.0 if mode == "stock" else 20.0, "prefill": 100.0, "prompt_tokens": 2000, "ttft_s": 1.0, "text": FAKE_TEXT[mode],
            "plugin_active": mode == "moe-cache", "hit_rate": 90.0 if mode == "moe-cache" else None, "load_s": 5.0, "error": "",
            **({"agent_ttft_s": [9.0, 1.0, 0.5] if mode == "stock" else [4.0, 0.8, 0.4], "agent_decode": [10.0, 10.0, 10.0]} if getattr(a, "scenario", "") == "agent" else {})}


FAKE_TEXT = {"stock": "same", "moe-cache": "same"}
facts = {"system": "Linux 6.8", "cpu": "Test CPU", "ram": "32 GB total", "gpu": f"Vulkan0 (Test GPU), driver 1.2 on {host}", "moe-cache": "0.3.2",
         "disk": f"files in {os.path.expanduser('~')}/models", "drive speed": "3.1 GB/s", "model": r"D:\Private Stuff\models\x.gguf, 66 GB"}
mb.doctor_facts = lambda m, ls, ram: (False, dict(facts), [])
mb.server_info = lambda ls: (dev("Vulkan0"), "version: 11669 (620d2225c)")
mb.run_one = fake_run
ui.header_quant = lambda p: "Q2_0"
mb.mc.port_busy = lambda p: False
flush_ok = [True]
ui.flush_model_cache = lambda m, other: (flush_ok[0], "flushed" if flush_ok[0] else "no folder with other big files")
plan = ["plugin"]
mb.launcher_plan = lambda m, ls, ram, ctx, extra: (plan[0], "env MOE_CACHE_SIZE_GIB=auto llama-server -m x -c 8192 --n-cpu-moe 30")
try:
    out = os.path.join(tmp, "out dir")
    rc = mb.bench_report([model, "--out", out, "--llama-server", "llama-server"])
    files = sorted(os.listdir(out))
    check(rc == 0 and len(files) == 2 and files[0].endswith(".json") and files[1].endswith(".md"), f"exit 0, .md and .json written ({files})")
    md = open(os.path.join(out, files[1]), encoding="utf-8").read()
    js = open(os.path.join(out, files[0]), encoding="utf-8").read()
    order_ = ["moe-cache benchmark report", "**Result:**", "| OS |", "| **Backend** | **Vulkan** |", "| llama.cpp |", "| File |", "| Launcher plan |",
              "| decode tok/s", "| prompt tok/s", "Checks: tokens IDENTICAL", "No prompts, answers"]
    pos = [md.find(x) for x in order_]
    check(all(p >= 0 for p in pos) and pos == sorted(pos), "report sections present and in order")
    check("**COLD**" in md and calls == ["stock", "moe-cache", "moe-cache", "stock"], "COLD, and the runs went stock, moe-cache, moe-cache, stock")
    for bad, what in ((user, "user name"), (host, "host name"), (os.path.expanduser("~"), "home folder"), (model_dir, "model folder"), ("same", "answer text"),
                      ("Private Stuff", "another folder (Windows path)")):
        check(bad not in md and bad not in js, f"privacy: no {what} in the report or the .json")
    json.loads(js)
    check("2.0x" in md, "decode ratio 2.0x in the result line")

    calls.clear(); flush_ok[0] = False
    rc = mb.bench_report([model, "--out", out, "--llama-server", "llama-server", "--flush-from", ""])
    md = open(os.path.join(out, sorted(f for f in os.listdir(out) if f.endswith(".md"))[-1]), encoding="utf-8").read()
    check("**WARM**" in md and md.count("cold start failed") == 1, "WARM stamp and one note (not one per run) when the cold start fails")
    flush_ok[0] = True

    FAKE_TEXT["moe-cache"] = "other"
    rc = mb.bench_report([model, "--out", out, "--warm"])
    md = open(os.path.join(out, sorted(f for f in os.listdir(out) if f.endswith(".md"))[-1]), encoding="utf-8").read()
    check(rc == 0 and "tokens DIFFERENT" in md and "tokens differ: stock and moe-cache ran" in md, "different tokens: reported with the explanation, exit 0")
    FAKE_TEXT["moe-cache"] = "same"

    plan[0] = "stock"; calls.clear()
    check(mb.bench_report([model, "--out", out, "--warm"]) == 3 and calls == [], "model fits in RAM -> exit 3, nothing run")
    check(mb.bench_report([model, "--out", out, "--warm", "--force-plugin"]) == 0 and len(calls) == 4, "--force-plugin measures anyway")
    plan[0] = "plugin"

    rc = mb.bench_report([model, "--out", out, "--warm", "--scenario", "agent", "--gap", "0"])
    md = open(os.path.join(out, sorted(f for f in os.listdir(out) if f.endswith(".md"))[-1]), encoding="utf-8").read()
    check(rc == 0 and "agent turns: time to first token" in md and "9.0 / 1.0 / 0.5" in md and "4.0 / 0.8 / 0.4" in md, "--scenario agent: per-turn time to first token, stock and moe-cache")
    rc = mb.bench_report([model, "--out", out, "--warm"])
    md = open(os.path.join(out, sorted(f for f in os.listdir(out) if f.endswith(".md"))[-1]), encoding="utf-8").read()
    check("agent turns" not in md, "default scenario: no agent row")

    mb.doctor_facts = lambda m, ls, ram: (True, {}, ["[FAIL] plugin  not found"])
    check(mb.bench_report([model, "--out", out]) == 2, "a doctor FAIL stops with exit 2")
finally:
    mb.doctor_facts, mb.launcher_plan, mb.run_one, mb.server_info, ui.flush_model_cache, ui.header_quant, mb.mc.port_busy = saved

print("bench-report test: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
sys.exit(1 if FAILS else 0)
