#!/usr/bin/env python3
"""tools/quality_gate.py --window (QUALITY-014): the default window stays 512 with -b 8 -ub 8 (the gate's protocol, unchanged); a longer window sets -c and a 512-token batch; the record says the window."""
import os, sys, types
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
import quality_gate as qg      # noqa: E402

fails = 0
def check(ok, what):
    global fails
    print(("  ok   " if ok else "  FAIL ") + what); fails += not ok

a = types.SimpleNamespace(perplexity="ppl", model="m.gguf", text="t.txt", chunks=3, ngl="99", n_cpu_moe="999", common_args="")
c = qg.ppl_cmd(a, "base.kld", True, ["-ctk", "q8_0"])
check(c[c.index("-c") + 1] == "512" and c[c.index("-b") + 1] == "8" and c[c.index("-ub") + 1] == "8", "no --window: -c 512, -b 8, -ub 8 (the gate's own protocol, unchanged)")
check(c[-3:] == ["-ctk", "q8_0", "--kl-divergence"] and "--kl-divergence-base" in c, "the candidate command still carries its KV arguments and --kl-divergence")
a.window = 32768
c = qg.ppl_cmd(a, "base.kld", False, [])
check(c[c.index("-c") + 1] == "32768" and c[c.index("-b") + 1] == "512" and c[c.index("-ub") + 1] == "512" and "--kl-divergence" not in c, "--window 32768: -c 32768, -b / -ub 512, the stock side has no --kl-divergence")
a.window = 512
check(qg.ppl_cmd(a, "b", True, []) == qg.ppl_cmd(types.SimpleNamespace(**{k: v for k, v in vars(a).items() if k != "window"}), "b", True, []), "--window 512 equals the old default exactly")
import argparse, subprocess
h = subprocess.run([sys.executable, os.path.join(HERE, "tools", "quality_gate.py"), "--help"], capture_output=True, text=True).stdout
check("--window" in h, "--window is in --help")
print("PASS (0 failed)" if not fails else f"FAIL ({fails} failed)")
sys.exit(1 if fails else 0)
