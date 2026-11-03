#!/usr/bin/env python3
"""A workflow step name with ": " (or " #") in it must be quoted: unquoted it is invalid YAML, GitHub then ignores the whole file (no workflow_dispatch, every
push run fails in 0 s). 53caa47 did this to windows.yml. Text check, no yaml library; uses PyYAML as well when it is installed.
   python3 tests/workflow_yaml_test.py"""
import glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad = []
files = sorted(glob.glob(os.path.join(ROOT, ".github", "workflows", "*.yml")))
for f in files:
    text = open(f, encoding="utf-8").read()
    for n, line in enumerate(text.splitlines(), 1):
        m = re.match(r"\s*(?:-\s+)?(name|run-name|description):\s+(.*)$", line)
        if not m:
            continue
        v = m.group(2).strip()
        if v and v[0] not in "\"'|>[{&*!" and (": " in v or " #" in v or v.endswith(":")):
            bad.append(f"{os.path.basename(f)}:{n}: unquoted '{m.group(1)}' value with ': ' or ' #' in it")
    try:
        import yaml
        yaml.safe_load(text)
    except ImportError:
        pass
    except Exception as e:
        bad.append(f"{os.path.basename(f)}: not valid YAML ({str(e).splitlines()[-1]})")
for b in bad:
    print("  FAIL " + b)
print(("  ok   " if not bad else "  FAIL ") + f"{len(files)} workflow files: step names are valid YAML")
sys.exit(1 if bad else 0)
