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
        path = os.path.join(tmp, "x", "p.json")
        import moe_cache_planfile as pf
        check(pf.write(path, {"schema": 1}) and json.load(open(path)) == {"schema": 1}, "write makes the folder and the file")
    print("plan file: all checks passed" if not bad else f"{bad} checks failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
