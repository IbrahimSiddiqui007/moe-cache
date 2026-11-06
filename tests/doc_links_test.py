#!/usr/bin/env python3
"""Every relative link in the repository's Markdown files points at a file or folder that exists (tools/check_doc_links.py), and the README stays short.
   python3 tests/doc_links_test.py"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = subprocess.run([sys.executable, os.path.join(HERE, "tools", "check_doc_links.py")], capture_output=True, text=True)
print(r.stdout.strip())
n = len(open(os.path.join(HERE, "README.md"), encoding="utf-8").read().split("\n"))
print(f"README.md: {n} lines (limit 200)")
sys.exit(0 if r.returncode == 0 and n < 200 else 1)
