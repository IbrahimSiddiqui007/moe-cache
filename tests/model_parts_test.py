#!/usr/bin/env python3
"""model_parts() finds every shard of a split GGUF with glob: the folder and file name are the user's own text, so '[', ']' and '*' in them must match
themselves (a folder 'models [new]' returned no shards: the size was 0 and the launcher's stock-or-plugin decision, the planner and the page all saw an empty model).
   python3 tests/model_parts_test.py"""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bin"))
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

tmp = tempfile.mkdtemp(prefix="parts-")
for d in ("plain", "models [new]", "a*b", "q?x"):
    os.makedirs(os.path.join(tmp, d))
    stem = "Big" if d == "plain" else "Big [Q4]"
    names = [f"{stem}-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]
    for n in names:
        open(os.path.join(tmp, d, n), "wb").write(b"x")
    open(os.path.join(tmp, d, f"{stem}-00001-of-00004.gguf"), "wb").write(b"x")      # another split of the same name: not a part
    got = mc.model_parts(os.path.join(tmp, d, names[0]))
    check(sorted(os.path.basename(p) for p in got) == names, f"all 3 shards found in '{d}' ({len(got)})")
sys.path.insert(0, os.path.join(ROOT, "planner"))
import gguf_plan  # noqa: E402
import moe_cache_pick as picker  # noqa: E402
d = os.path.join(tmp, "models [new]")
first = os.path.join(d, "Big [Q4]-00001-of-00003.gguf")
check(len(gguf_plan.shards(first)) == 3, "the planner's shards() finds all 3 in a folder with brackets")
open(os.path.join(d, "Other [x].gguf"), "wb").write(b"x")
picker.MODELS = dict(picker.MODELS, probe=dict(name="p", size=1, pattern=r"^Other \[x\]\.gguf$", repo="r", include="i"))
picker.is_draft_head = lambda p: False
check("probe" in picker.find_local([d], ollama_stores=[]), "the picker finds a model in a folder whose name has brackets")
single = os.path.join(tmp, "plain", "one.gguf")
check(mc.model_parts(single) == [single], "a single file is itself")
sys.exit(1 if fails else 0)
