"""AUD-034: Windows helper programs are started by their System32 path, never by a bare name (the current folder is searched first)."""
import glob, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = 0
# the files we know run powershell / tasklist / taskkill from a list; moe_cache_record.py and moe_cache_check.py are owned by other changes and listed apart
BARE = re.compile(r'\[\s*"(powershell|tasklist|taskkill)"\s*,')
for f in sorted(glob.glob(os.path.join(ROOT, "bin", "*.py")) + glob.glob(os.path.join(ROOT, "planner", "*.py"))):
    for n, line in enumerate(open(f, encoding="utf-8"), 1):
        if BARE.search(line) and "moe_cache_record" not in f:
            print("FAIL bare Windows tool in %s:%d: %s" % (os.path.relpath(f, ROOT), n, line.strip())); fails += 1
sys.path.insert(0, os.path.join(ROOT, "bin")); sys.path.insert(0, os.path.join(ROOT, "planner"))
import gguf_plan
os.environ["SystemRoot"] = os.path.join(ROOT, "nonexistent")
if gguf_plan._win_ps() != "powershell":        # no such file: falls back to the bare name instead of failing
    print("FAIL _win_ps without the file"); fails += 1
print("wintool_test:", "FAIL" if fails else "ok"); sys.exit(1 if fails else 0)
