#!/usr/bin/env python3
"""WINDOWS-082: tools/windows_test.ps1 caps llama-server's memory with a job object and refuses a model bigger than the cap. PowerShell is not available on
the Linux CI, so this checks the script's text against the launcher's rule (mem_cap: RAM minus max(4 GiB, 12.5 %)) and that the pieces are all there; the
Windows workflow runs the script itself (it prints "job memory limit ... read back").
   python3 tests/win_memcap_test.py"""
import os, re, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc
bad = 0
def check(ok, what):
    global bad
    print(("  ok   " if ok else "  FAIL ") + what)
    bad += 0 if ok else 1
ps = open(os.path.join(HERE, "tools", "windows_test.ps1"), encoding="utf-8").read()
check("$ramBytes - [math]::Max(4GB, 0.125 * $ramBytes)" in ps, "the cap formula is RAM minus max(4 GiB, 12.5 %)")
os.environ.pop("MOE_CACHE_MAX_RAM_GB", None)
mc.mem_total = lambda: 32 * 2**30
check(abs(mc.mem_cap() - 28 * 2**30) < 1, f"...and the launcher's rule gives the same 28 GiB on a 32 GiB PC ({mc.mem_cap() / 2**30:.1f})")
check("MOE_CACHE_MAX_RAM_GB" in ps, "MOE_CACHE_MAX_RAM_GB overrides it, as in the launcher")
check("0x2000 | 0x200" in ps and "JobMemoryLimit" in ps and "AssignProcessToJobObject" in ps, "job object: KILL_ON_JOB_CLOSE + JOB_MEMORY, process assigned")
check("QueryInformationJobObject" in ps and "read back" in ps, "the limit is read back and printed")
check(re.search(r"\$modelBytes -gt \$capBytes -and -not \$AllowBigModel", ps) is not None and "throw" in ps, "a model bigger than the cap is refused unless -AllowBigModel")
check("[double]$MemLimitGB = 0" in ps and "[switch]$AllowBigModel" in ps, "parameters -MemLimitGB and -AllowBigModel exist")
check(ps.count("@\"") == 1 and re.search(r'^"@', ps, re.M) is not None, "the C# here-string is closed at the start of a line")
print("win_memcap_test:", "all passed" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
