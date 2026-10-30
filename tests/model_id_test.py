#!/usr/bin/env python3
"""One model identity (XA-1, bin/moe_cache_model.py): stable for the same file, different for a changed header, right for split models, the old saved-state key unchanged,
usage profiles keep their name and warm start, and a profile learned on another file with the same name is left alone. Synthetic GGUF headers (the 18 GB MoE one is
sparse, nothing is written); no model, no llama-server; Linux and Windows; standard library only.
   python3 tests/model_id_test.py"""
import hashlib, importlib.util, json, os, re, shutil, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
sys.path.insert(0, os.path.join(HERE, "tests"))
import moe_cache_model as mm  # noqa: E402
from pack_test import make_gguf  # noqa: E402
spec = importlib.util.spec_from_file_location("launch_plan_test", os.path.join(HERE, "tests", "launch_plan_test.py"))
lpt = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(lpt)
    make_moe = lpt.make_synthetic_model
except Exception:                       # a tree without the launch-plan test: the small pack_test models are enough for most checks
    make_moe = None

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

TMP = tempfile.mkdtemp(prefix="moe-model-id-")
def p(*parts):
    f = os.path.join(TMP, *parts)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    return f

# ---- the fields, on the synthetic MoE header ----
if make_moe:
    a = p("a", "synthetic-moe.gguf"); make_moe(a)
    m = mm.identity(a)
    check(m.arch == "qwen35moe" and m.n_layers == 40 and m.n_expert == 256 and m.expert_quant == "Q4_K" and len(m.shards) == 1 and m.size == os.path.getsize(a),
          f"fields: {m.summary()}, {m.size / 1e9:.1f} GB")
    check(m.expert_bytes == 40 * 3 * (2048 // 256 * 144) * 512 * 256, f"expert bytes from the tensor table ({m.expert_bytes / 1e9:.2f} GB)")
    check(re.fullmatch(r"[0-9a-f]{16}", m.header_fp) and re.fullmatch(r"[0-9a-f]{16}", m.sample_fp) and m.id.startswith("synthetic_moe-") and m.header_fp in m.id and "/" not in m.id,
          f"id: {m.id}")
    check("path" not in m.to_dict(paths=False) and "path" in m.to_dict(), "to_dict(paths=False) has no path (for reports)")
    # same bytes elsewhere: same identity; a changed header: another one
    b = p("b", "synthetic-moe.gguf"); make_moe(b)
    mm._CACHE.clear()
    mb = mm.identity(b)
    check(mb.header_fp == m.header_fp and mb.sample_fp == m.sample_fp and mb.id == m.id and mb.path != m.path, "the same file in another folder: the same id")
    check(mm.identity(a) is mm.identity(a), "cached per file (size and modification time)")
    c = p("c", "synthetic-moe.gguf"); make_moe(c, n_expert=128)
    check(mm.identity(c).header_fp != m.header_fp and mm.identity(c).id != m.id and mm.identity(c).n_expert == 128, "a changed tensor table (128 experts): another header hash and id")
    d = p("d", "synthetic-moe.gguf"); make_moe(d, n_layers=39)
    check(mm.identity(d).header_fp != m.header_fp and mm.identity(d).n_layers == 39, "one layer fewer: another header hash")

# ---- same header, other weights (a fine-tune): header_fp equal, sample_fp and the saved-state key differ ----
x1, x2 = p("x", "small-moe.gguf"), p("y", "small-moe.gguf")
make_gguf(x1, seed=1); make_gguf(x2, seed=2)
i1, i2 = mm.identity(x1), mm.identity(x2)
check(i1.header_fp == i2.header_fp and i1.sample_fp != i2.sample_fp and i1.id == i2.id, "same header, other weights: header hash equal, sample hash differs")
check(mm.slot_id(x1, 8192) != mm.slot_id(x2, 8192), "the saved-state key follows the weights too (a fine-tune does not get another model's prompt state)")

# ---- split models ----
s1, s2 = p("s", "split-00001-of-00002.gguf"), p("s", "split-00002-of-00002.gguf")
make_gguf(s1, n_layers=3); make_gguf(s2, n_layers=2, seed=3)
mm._CACHE.clear()
ms = mm.identity(s1)
check(len(ms.shards) == 2 and ms.size == os.path.getsize(s1) + os.path.getsize(s2) and ms.name == "split" and [os.path.basename(x) for x, _ in ms.shards] == [os.path.basename(s1), os.path.basename(s2)],
      f"split model: two shards, summed size, name without the shard suffix ({ms.name})")
make_gguf(s2, n_layers=4, seed=3)
mm._CACHE.clear()
check(mm.identity(s1).header_fp != ms.header_fp, "the second shard's header is part of the identity")
os.remove(s2)
mm._CACHE.clear()
check(len(mm.identity(s1).shards) == 1 and mm.identity(s1).header_fp != ms.header_fp, "a missing shard shows (one shard found, another identity)")

# ---- the saved-state key is the old one, byte for byte ----
def old_model_identity(model, ctx=None, ctk=None, ctv=None):       # the launcher's function before XA-1, copied
    try:
        size = os.path.getsize(model)
        h = hashlib.sha256()
        with open(model, "rb") as f:
            h.update(f.read(1 << 20))
            for frac in (0.5, 0.75, 1.0):
                f.seek(max(0, min(int(size * frac), size - (64 << 10)) - (32 << 10)))
                h.update(f.read(64 << 10))
    except OSError:
        return ""
    return "-".join([re.sub(r"[^A-Za-z0-9_]", "_", os.path.basename(model))[:40], str(size), h.hexdigest()[:16], f"c{ctx}", ctk or "f16", ctv or "f16"])
files = [x1, x2, s1] + ([a] if make_moe else [])
check(all(mm.slot_id(f, c, k, v) == old_model_identity(f, c, k, v) for f in files for c, k, v in ((8192, None, None), (131072, "q8_0", "q8_0"), (None, "f16", "q4_0"))),
      "slot_id equals the old model_identity for every file and setting (saved sessions are still found)")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc  # noqa: E402
check(mc.model_identity(x1, 4096, "q8_0", "q8_0") == old_model_identity(x1, 4096, "q8_0", "q8_0") and mc.model_identity(os.path.join(TMP, "nope.gguf")) == "", "the launcher's model_identity is slot_id; an unreadable file gives \"\"")

# ---- unreadable and not-GGUF files ----
bad = p("bad", "notgguf.gguf"); open(bad, "wb").write(b"hello world" * 100)
try:
    mm.identity(bad); check(False, "a file that is not GGUF is refused")
except (ValueError, OSError):
    check(True, "a file that is not GGUF is refused (ValueError)")
try:
    mm.identity(os.path.join(TMP, "missing.gguf")); check(False, "a missing file is refused")
except OSError:
    check(True, "a missing file is refused (OSError)")

# ---- usage profiles ----
cache = os.path.join(TMP, "cache"); os.makedirs(cache)
said = []
say = lambda msg, err=False: said.append(msg)
base = os.path.join(cache, "small-moe.gguf.usage")
check(mm.profile_path(x1, "", cache, dry=True, say=say) == base and not os.path.exists(base + ".id"), "a dry run gives the old profile name and writes nothing")
open(base, "wb").write(b"XUSE")          # a profile learned before this module: no note
check(mm.profile_path(x1, "", cache, dry=False, say=say) == base and open(base, "rb").read() == b"XUSE", "an old profile with no note keeps its name and its bytes (warm start kept)")
rec = json.load(open(base + ".id"))
check(rec["header_fp"] == i1.header_fp and rec["size"] == i1.size, f"... and the note adopts it ({rec})")
check(mm.profile_path(x2, "", cache, dry=False, say=say) == base and not said, "the same header with other weights (a fine-tune) shares the profile, no message")
named = mm.profile_path(x1, "coding", cache, dry=False, say=say)
check(named == os.path.join(cache, "small-moe.gguf.coding.usage") and os.path.exists(named + ".id"), "a named profile keeps its name too")
other = p("other", "small-moe.gguf"); make_gguf(other, n_layers=5)       # another file with the same name and another header
oi = mm.identity(other)
alt = os.path.join(cache, "small-moe.gguf." + oi.header_fp[:8] + ".usage")
r = mm.profile_path(other, "", cache, dry=True, say=say)
check(r == alt and len(said) == 1 and "another model file with the same name" in said[0] and not os.path.exists(alt + ".id") and open(base, "rb").read() == b"XUSE",
      f"another file with the same name: not that profile, a new name with the header's hash, said why, a dry run writes nothing ({os.path.basename(r)})")
said.clear()
r = mm.profile_path(other, "", cache, dry=False, say=say)
check(r == alt and os.path.exists(alt + ".id") and open(base, "rb").read() == b"XUSE", "... and a real start records the new profile's note; the old profile is untouched")
said.clear()
check(mm.profile_path(other, "", cache, dry=False, say=say) == alt and not said, "the next start finds its own profile: no message")
check(mm.profile_path(x1, "", cache, dry=False, say=say) == base and not said, "the first model still has its profile")
check(mm.profile_path(os.path.join(TMP, "missing.gguf"), "", cache, say=say) == os.path.join(cache, "missing.gguf.usage"), "an unreadable model: the old name (the launcher reports the missing file itself)")

print("model id test: " + ("FAIL" if fails else "PASS"))
sys.exit(1 if fails else 0)
