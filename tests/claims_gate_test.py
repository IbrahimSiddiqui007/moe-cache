#!/usr/bin/env python3
"""tools/claims_gate.py: a waiver must name claim ids and an expiry; expired, unnamed or too-broad waivers are refused (no model needed)."""
import datetime, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import claims_gate as g

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok: fails.append(what)

TODAY = datetime.date(2026, 11, 10)
LINT = ("README.md:119: claim stale: chain1103-iq3xxs: moe-cache is now 0.3.6, the claim is valid up to 0.3.5\n"
        "README.md:92: unclaimed: 2.0x\n"
        "README.md: 9 lines show performance numbers; 0 with a current claim; 2 problems\n")
W = "waived: owner 2026-11-05; expires 2026-11-12; claims chain1103-*; files README.md; reason: v0.3.6 not re-measured yet"
rc, msg = g.run(LINT, 1, W, TODAY)
check(rc == 0 and "waiver" in msg, "a waiver naming the claims, the file and an unexpired date passes (with a warning)")
check(g.run("", 0, None, TODAY)[0] == 0, "clean lint passes without a waiver")
check(g.run(LINT, 1, None, TODAY)[0] == 1, "problems and no waiver fail")
check(g.run(LINT, 1, W.replace("expires 2026-11-12", "expires 2026-11-09"), TODAY)[0] == 1, "an expired waiver is refused")
check(g.run(LINT, 1, W.replace("chain1103-*", "*"), TODAY)[0] == 1, "a bare * claim list is refused")
check(g.run(LINT, 1, W.replace("claims chain1103-*; ", ""), TODAY)[0] == 1, "a waiver with no claims field is refused")
check(g.run(LINT, 1, W.replace("expires 2026-11-12", "expires 2026-12-30"), TODAY)[0] == 1, "more than 30 days is refused")
check(g.run(LINT, 1, W.replace("chain1103-*", "card-decode-*"), TODAY)[0] == 1, "a stale id the waiver does not name still fails")
check(g.run(LINT, 1, W.replace("files README.md", "files docs/benchmark_card.html"), TODAY)[0] == 1, "an unclaimed number in a file the waiver does not name still fails")
check(g.run(LINT, 1, W + "\nsecond line", TODAY)[0] == 1, "a second line is refused")
check(g.run(LINT, 1, W.replace("reason: v0.3.6 not re-measured yet", "reason: x"), TODAY)[0] == 1, "a reason shorter than 8 characters is refused")
check(g.run("docs/x.md:3: withdrawn: claim foo\n", 1, W, TODAY)[0] == 1, "other kinds of problem are not waivable")
print("claims gate test: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)
