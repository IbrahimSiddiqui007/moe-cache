#!/usr/bin/env python3
"""The plan file (bin/moe_cache_planfile.py): `server --dry-run --plan-file PATH` writes one JSON row per decided setting (value, source, reason), the sources are
the right ones for a few known cases (your flag = user, the stock path still shows its --n-cpu-moe), and the summary is printed. Same fake machines as launch_plan_test.py
(no model, no GPU). The 'commands unchanged' proof is launch_plan_test.py (golden) itself."""
import json, os, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests"))
import launch_plan_test as lp   # noqa: E402


def row(rec, name):
    return next(r for r in rec["settings"] if r["setting"] == name)


import moe_cache_planfile as pf  # noqa: E402


def main():
    bad = 0
    def check(ok, what):
        nonlocal bad
        print(("  ok   " if ok else "  FAIL ") + what)
        bad += 0 if ok else 1
    with tempfile.TemporaryDirectory() as tmp:
        model = os.path.join(tmp, "synthetic-moe.gguf")
        lp.make_synthetic_model(model)
        def plan(opts, extra=(), machine="linux"):
            path = os.path.join(tmp, "plan.json")
            if os.path.exists(path):
                os.unlink(path)
            r = lp.run_case(machine, opts + ["--plan-file", path], list(extra), model, tmp)
            return r, (json.load(open(path)) if os.path.exists(path) else None)
        # LAUNCHER-1312: the real Inkling command (server log 2026-11-28, chain30 c1271_trace_inkling): --n-cpu-moe 0 plus the planner's 13 layers by -ot
        ot = "blk\\.(0|1|2|3|4|5|6|18|19|20|21|22|23)\\.ffn_(gate_up|gate|up|down)_exps\\.weight=CPU"
        check(pf.cpu_expert_layers(["llama-server", "--n-cpu-moe", "0", "-ot", ot]) == [0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21, 22, 23], "1312: the -ot rule is read as the real CPU expert layers (13)")
        check(pf.cpu_expert_layers(["llama-server", "--n-cpu-moe", "0"]) == [] and pf.cpu_expert_layers(["-ot", "exps=CPU"]) == [], "1312: no launcher -ot rule = no layers")
        r, rec = plan(["--ram", "12"])
        check(rec is not None and rec["schema"] == 1 and rec["settings"], "a dry run with --plan-file writes the file")
        check(all(x["source"] in ("probe", "measured", "constant", "user", "llama") and x["why"] for x in rec["settings"]), "every row has a source and a reason")
        check("launch plan" in r["stdout"], "the summary is printed")
        r, rec = plan(["--ram", "12", "--n-cpu-moe", "20"])
        check(row(rec, "n_cpu_moe")["source"] == "user" and row(rec, "n_cpu_moe")["value"] == "20", "--n-cpu-moe is the user's")
        r, rec = plan(["--ram", "12"], ["-ub", "256"])
        check(row(rec, "ub")["source"] == "user" and row(rec, "ub")["value"] == "256", "-ub after -- is the user's")
        r, rec = plan(["--ram", "12", "--stock"])
        check(rec["engine"] == "stock" and row(rec, "n_cpu_moe")["value"] is not None, "the stock path shows its final --n-cpu-moe")
        r, rec = plan(["--ram", "12", "--ctx", "131072"])
        check(row(rec, "context")["source"] == "user" and row(rec, "kv_type")["value"] == "q8_0/q8_0" and row(rec, "kv_type")["source"] == "constant", "context from --ctx is the user's; the 128k KV type is ours")
        r, rec = plan(["--ram", "12", "--stock"])
        check(row(rec, "engine")["source"] == "user", "--stock is the user's")
        r, rec = plan(["--ram", "12", "--leave-room"])
        check(row(rec, "threads")["source"] == "constant" and "leave-room" in row(rec, "threads")["why"], "--leave-room threads are labelled as --leave-room, not as pinning")
        r, rec = plan(["--ram", "12", "--mtp"])
        check(rec is None and r["rc"] not in (None, 0), "--mtp on a model without an MTP head stops, no plan file")
        r, rec = plan(["--ram", "12", "--ctx", "65536"], machine="win")
        check(row(rec, "ub")["source"] in ("probe", "user", "constant") and rec["engine"] in ("plugin", "stock"), "a Windows fake machine builds the file")
        # stubs for paths the fake machines do not reach: a stored auto-tune pick, the Windows commit check
        import types
        def stub(cmd, tune=None, changed=None, **kw):
            a = types.SimpleNamespace(stock=False, ub="auto", ctx=None, vram=None, ram=None, mtp=None, leave_room=False, dense_kv=False, lp_kv=False, fast=None, approx_spec=False)
            return types.SimpleNamespace(a=a, cmd=cmd, env=[], plan_env=[], argv_in=[], user_flags={}, user_extra=[], commit_changed=changed or {}, autotune=tune, tuned={"why": []},
                                         planner_info={"free_vram_GiB": 7.5}, ncmoe_user=None, engine="plugin", model="m.gguf", port=8080, ctx=8192, avail=None)
        base = ["llama-server", "--n-cpu-moe", "27", "-ub", "1024", "-c", "8192"]
        rec2 = pf.build(stub(base, {"changed": True, "state": "tuned", "picked": "ramlock"}))
        check(row(rec2, "n_cpu_moe")["source"] == "probe", "a ramlock pick does not make --n-cpu-moe 'measured'")
        rec2 = pf.build(stub(base, {"changed": True, "state": "cached", "picked": "ncmoe-2"}))
        check(row(rec2, "n_cpu_moe")["source"] == "measured", "an ncmoe pick makes --n-cpu-moe 'measured'")
        rec2 = pf.build(stub(base, None, {"ub": ("2048", "1024"), "n_cpu_moe": ("25", "27")}))
        check(row(rec2, "ub")["source"] == "probe" and "commit" in row(rec2, "ub")["why"] and "commit" in row(rec2, "n_cpu_moe")["why"], "a value the commit check changed says so")
        path = os.path.join(tmp, "x", "p.json")
        check(pf.write(os.path.join(tmp, "y", "p.json"), {"bad": object()}) is False, "a value that cannot be written never raises")
        pf.write(path, rec2)
        pf.remove(path, rec2)
        check(not os.path.exists(path), "remove deletes the file this launcher wrote")
        pf.write(path, rec2)
        pf.remove(path, {"other": 1})
        check(os.path.exists(path), "remove keeps a file another launcher wrote")
        os.unlink(path)
        check(pf.write(path, {"schema": 1}) and json.load(open(path)) == {"schema": 1}, "write makes the folder and the file")
    print("plan file: all checks passed" if not bad else f"{bad} checks failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
