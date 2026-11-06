#!/usr/bin/env python3
"""Compare the logit files of two layerwise_check runs: max |diff| and whether the best token agrees on every step.   python3 tests/layerwise_compare.py A.bin B.bin N_VOCAB"""
import struct, sys
a, b, nv = open(sys.argv[1], "rb").read(), open(sys.argv[2], "rb").read(), int(sys.argv[3])
if len(a) != len(b):
    sys.exit(f"different sizes: {len(a)} vs {len(b)}")
fa, fb = struct.unpack(f"<{len(a)//4}f", a), struct.unpack(f"<{len(b)//4}f", b)
worst, same = 0.0, True
for r in range(len(fa) // nv):
    ra, rb = fa[r*nv:(r+1)*nv], fb[r*nv:(r+1)*nv]
    worst = max(worst, max(abs(x - y) for x, y in zip(ra, rb)))
    same = same and ra.index(max(ra)) == rb.index(max(rb))
print(f"rows {len(fa)//nv}: max |diff| {worst:.3g}, best token same on every row: {same}")
sys.exit(0 if same and worst < 1e-3 else 1)
