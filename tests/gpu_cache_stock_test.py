#!/usr/bin/env python3
"""CACHE-1187 / LAUNCHER-1188: the stock-engine GPU expert cache (--gpu-cache auto / MIB). The micro-batch stays -ub 2048 while the cache still gets 1.5 GiB at that batch and drops
to 1024 below it; the plugin path keeps 1024; `--load-mode none` only when the LAUNCHER chose the cache and the file + prompt cache + reserve fit the server's RAM (not with --exact,
-ot, the user's own load flag, --stock, or a cache flag the user passed after --).   python3 tests/gpu_cache_stock_test.py   (no model: a synthetic MoE header)"""
import os, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(HERE, "tests"), os.path.join(HERE, "planner"), os.path.join(HERE, "bin")]
import launch_plan_test as lp      # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def stock_env():
    os.environ["MOE_CACHE_STOCK_WHEN_FITS"] = "1"        # the stock engine without the "partial offload" plugin rule (a fake plugin library exists in the test)

def plan(model, tmp, opts, extra=(), stock=True, vram=8):
    r = lp.run_case("linux", ["--vram", str(vram)] + list(opts), list(extra), model, tmp, call=stock_env if stock else None)
    out = r["stdout"]
    c = next((l for l in out.splitlines() if "llama-server" in l and " -m " in l), "")
    g = lambda k: c.split(k)[1].split()[0] if k in c.split() or (k in c) else None
    return dict(lm=g("--load-mode "), ub=g("-ub "), cache=g("--moe-cache-mib "), stock="without the plugin" in out, out=out)

with tempfile.TemporaryDirectory() as tmp:
    model = os.path.join(tmp, "moe.gguf"); lp.make_synthetic_model(model, n_layers=40, ne1=256)      # about 8.7 GiB, 40 expert layers
    p = plan(model, tmp, ["--gpu-cache", "auto"])
    check(p["stock"] and p["cache"] and p["ub"] == "2048", f"stock + --gpu-cache auto keeps -ub 2048 (cache {p['cache']} MiB)")
    check(p["lm"] == "none" and "--load-mode none" in p["out"], "stock + --gpu-cache auto + RAM room: --load-mode none, says why")
    p = plan(model, tmp, ["--gpu-cache", "auto", "--exact"])
    check(p["lm"] == "mmap", "--exact keeps mmap")
    p = plan(model, tmp, ["--gpu-cache", "auto"], ["--load-mode", "mmap"])
    check(p["lm"] == "mmap", "the user's own --load-mode after -- stands")
    p = plan(model, tmp, ["--gpu-cache", "auto"], ["-ot", "exps=CPU"])
    check(p["lm"] != "none", "-ot of your own: no --load-mode none")
    p = plan(model, tmp, [], ["--moe-cache-mib", "2000"])
    check(p["lm"] == "mmap" and p["cache"] == "2000", "a cache flag the user passed after --: mmap stays (the launcher did not choose the cache)")
    p = plan(model, tmp, ["--gpu-cache", "auto", "--max-ram", "9"])
    check(p["lm"] == "mmap", "not enough RAM for file + prompt cache + reserve: mmap stays")
    p = plan(model, tmp, ["--gpu-cache", "auto", "--always-plugin"], stock=False)      # LAUNCHER-1236: this model nearly fits the card, so only --always-plugin keeps the plugin
    check(not p["stock"] and p["ub"] == "1024" and p["lm"] == "mmap", "plugin path unchanged: -ub 1024, mmap")
    p = plan(model, tmp, ["--gpu-cache", "auto"], vram=3.4)
    check(p["stock"] and p["ub"] == "1024" and p["cache"] is not None, f"cache under 1.5 GiB at -ub 2048 (little VRAM): -ub 1024 buys the room (cache {p['cache']} MiB)")
    p = plan(model, tmp, [])
    check(p["lm"] == "mmap" and p["cache"] is None, "no --gpu-cache, llama-server without --moe-cache-mib: nothing turns on, mmap")

    # --- the default (owner 2026-11-16): on only on Linux + CUDA llama-server + stock engine MoE with experts on the CPU, a llama-server that has --moe-cache-mib
    lp.HELP_TEXT += "  --moe-cache-mib N\n"
    def dplan(opts, extra=(), machine="linux", stock=True):
        r = lp.run_case(machine, ["--vram", "8"] + list(opts), list(extra), model, tmp, call=stock_env if stock else None)
        c = next((l for l in r["stdout"].splitlines() if "llama-server" in l and " -m " in l), "")
        g = lambda k: c.split(k)[1].split()[0] if k in c else None
        return dict(cache=g("--moe-cache-mib "), ub=g("-ub "), lm=g("--load-mode "), why="GPU expert cache on by default" in r["stdout"], ncmoe=g("--n-cpu-moe "))
    d = dplan([])
    check(d["why"] and d["cache"] and d["ncmoe"] == "999" and d["ub"] == "2048" and d["lm"] == "none", f"default on Linux + CUDA + stock MoE: cache {d['cache']} MiB, -ub 2048, --load-mode none, the plan line says why")
    for what, o, e in (("--gpu-cache off", ["--gpu-cache", "off"], []), ("--exact", ["--exact"], []), ("--stock", ["--stock"], []), ("your own --n-cpu-moe", ["--n-cpu-moe", "20"], []),
                       ("your own --moe-cache-mib after --", [], ["--moe-cache-mib", "2000"]), ("your own -ot after --", [], ["-ot", "exps=CPU"]), ("a draft model after --", [], ["-md", "draft.gguf"])):
        d = dplan(o, e)
        check(not d["why"] and (d["cache"] is None or what.startswith("your own --moe")), f"default off with {what}")
    d = dplan([], machine="win")
    check(not d["why"] and d["cache"] is None, "default off on Windows / Vulkan")
    lp.MACHINES["linux_vk"] = dict(lp.MACHINES["linux"], listing=lp.WIN_LIST)
    d = dplan([], machine="linux_vk")
    check(not d["why"] and d["cache"] is None, "default off on Linux with a Vulkan llama-server")
    d = dplan(["--gpu-cache", "auto"], machine="linux_vk")
    check(d["cache"] is not None and d["ub"] == "1024" and d["lm"] == "mmap", f"--gpu-cache auto asked on Vulkan: the old -ub 1024 rule and mmap stay (cache {d['cache']})")
    lp.MACHINES["linux_2cuda"] = dict(lp.MACHINES["linux"], listing=lp.LINUX_LIST + "  CUDA1: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB, 7600 MiB free)\n")
    d = dplan([], machine="linux_2cuda")
    check(not d["why"] and d["cache"] is None, "default off with two CUDA devices (not measured)")
    d = dplan(["--always-plugin"], stock=False)          # LAUNCHER-1236: the nearly-fitting model runs stock unless the plugin is asked for
    check(not d["why"], "default off when the plugin engine runs")
    lp.HELP_TEXT = lp.HELP_TEXT.replace("  --moe-cache-mib N\n", "")
    d = dplan([])
    check(not d["why"] and d["cache"] is None, "default off when llama-server has no --moe-cache-mib")
print("gpu_cache_stock_test: " + ("all passed" if not fails else f"{len(fails)} FAILED"))
sys.exit(1 if fails else 0)
