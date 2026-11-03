#!/usr/bin/env python3
"""CI guard (AUD-094): tests/ui_test.py skips its JavaScript-logic checks, with exit 0, when `node` is not on the PATH. CI must not lose those checks
silently, so this test fails when node is missing or cannot run a one-line script. tools/run_quick_tests.sh runs it when CI or REQUIRE_NODE is set."""
import shutil, subprocess, sys

node = shutil.which("node")
if not node:
    print("FAIL: node is not on the PATH, so the control page's JavaScript checks in tests/ui_test.py would be skipped silently")
    sys.exit(1)
r = subprocess.run([node, "-e", "process.stdout.write(String(6*7))"], capture_output=True, text=True, timeout=60)
if r.returncode != 0 or r.stdout.strip() != "42":
    print(f"FAIL: {node} did not run a one-line script (rc {r.returncode}): {r.stderr.strip()[:200]}")
    sys.exit(1)
print(f"require_node_test: ok ({node})")
