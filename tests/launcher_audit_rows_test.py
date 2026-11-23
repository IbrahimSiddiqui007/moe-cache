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
sys.exit(1 if FAILS else 0)
