#!/usr/bin/env python3
"""launcher_audit_rows_test: LAUNCHER-1234 rows 14 (sampling/template failures are said, not silent) and 25 (check --ram takes a number; no bare O_DIRECT). Fake inputs, no model."""
import contextlib, io, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
import moe_cache as mc  # noqa: E402
FAILS = []


def check(c, w):
    print(("  ok   " if c else "  FAIL ") + w)
    if not c:
        FAILS.append(w)


err = io.StringIO()
with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
    r = mc.sampling_args(False, False, "/nonexistent/model.gguf", [])
check(r == [] and "sampling defaults" in err.getvalue(), "row 14: unreadable model: no sampling args and one line says so")
err = io.StringIO()
with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
    r = mc.template_stage(types_ns := type("A", (), {})(), [], "/nonexistent/model.gguf", dry=True)
check(r == [] and ("chat template" in err.getvalue() or err.getvalue() == ""), "row 14: unreadable model: no template flags (a line when the fix could not run)")
src = open(os.path.join(HERE, "..", "bin", "moe_cache.py"), encoding="utf-8").read()
check(not re.search(r"os\.O_RDONLY \| os\.O_DIRECT", src), "row 25: no bare os.O_DIRECT")
check('add_argument("model", nargs="?"); ap.add_argument("--llama-server", default=default_llama_server()); ap.add_argument("--ram", type=gb_arg)' in src, "row 25: check --ram is typed")
r = mc.fit_line(5.0, 6.0)
check(r and r[0] and "5.0 GB + 1.5 GB headroom = 6.5 GB is over the 6.0 GB" in r[1], "row 9: 5 GB model, 6 GB limit: over (model + headroom), text states the sum")
r = mc.fit_line(4.0, 6.0)
check(r and not r[0] and "fits in 6.0 GB" in r[1], "row 9: 4 GB model, 6 GB limit: fits")
check(mc.fit_line(4.0, None) is None, "row 9: no limit: no line")
# row 28 (source check, not a run: the lambda lives inside plan_launch): the auto-tune's commit check gets the same argv as the real one (argv + the user's own flags after --)
check("commit_fits(a, c_cmd, c_env, list(argv) + user_extra," in src, "row 28: auto-tune fits() passes argv + user_extra to the commit check (source check)")
sys.exit(1 if FAILS else 0)
