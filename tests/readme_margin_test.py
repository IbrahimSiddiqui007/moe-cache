#!/usr/bin/env python3
"""PREFILL-103: the README tells how to reserve GPU memory for the desktop, and the option it names exists in the launcher. Text check.
   python3 tests/readme_margin_test.py"""
import os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc  # noqa: E402

readme = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
ok1 = "`--gpu-margin GiB`" in readme
a, _, _ = mc.parse_server(["m.gguf", "--gpu-margin", "3"])
ok2 = a.gpu_margin == 3.0
ok3 = "--gpu-margin GiB" in mc.SERVER_HELP
print(("  ok   " if ok1 else "  FAIL ") + "README names --gpu-margin GiB")
print(("  ok   " if ok2 else "  FAIL ") + "the launcher parses --gpu-margin 3")
print(("  ok   " if ok3 else "  FAIL ") + "the server help describes it")
sys.exit(0 if ok1 and ok2 and ok3 else 1)
