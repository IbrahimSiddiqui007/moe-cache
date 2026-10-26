#!/usr/bin/env python3
"""make_starter_profile.py MODEL.gguf PROFILE.usage [OUT_DIR]: write a starter usage profile for MODEL's family to OUT_DIR (default profiles/) as
<architecture>-<layers>x<experts>.usage, without the model fingerprint, so every quant of that model can warm-start from it (idea I18).
The launcher copies it to the user's profile path when there is none yet; from then on it learns from the user's own prompts."""
import os, struct, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
from gguf_plan import model_key
model, prof = sys.argv[1], sys.argv[2]
out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "profiles")
key = model_key(model)
if not key:
    raise SystemExit("no architecture / block_count / expert_count in the GGUF header")
data = open(prof, "rb").read()
if data[:4] != b"XUSE":
    raise SystemExit("not an XUSE usage profile")
n_l, n_e = struct.unpack("<ii", data[4:12])
body = data[12:12 + 4 * n_l * n_e]
if len(body) != 4 * n_l * n_e:
    raise SystemExit("profile is truncated")
if n_e != int(key.rsplit("x", 1)[1]):
    raise SystemExit(f"profile has {n_e} experts per layer but {key} has {key.rsplit('x', 1)[1]}")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, key + ".usage")
with open(out, "wb") as f:
    f.write(b"XUSE" + struct.pack("<ii", n_l, n_e) + body)       # no trailing fingerprint: accepted for every quant of the family
print(f"{out}: {n_l} layers x {n_e} experts")
