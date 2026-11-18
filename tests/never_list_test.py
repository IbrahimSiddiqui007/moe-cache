#!/usr/bin/env python3
"""LAUNCHER-1225 step 2: the never-list (planner/never_list.json + never_list.py). Every row is well formed and names its evidence; a listed setting is vetoed only
inside its scope (backend, dense/MoE, context); the auto-tune's candidate filter drops a listed candidate and keeps C0; a setting the user gave gets a note, never a refusal."""
import json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
sys.path.insert(0, os.path.join(HERE, "bin"))
import never_list as nl              # noqa: E402
import moe_cache as mc               # noqa: E402
import moe_cache_autotune as at      # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok   " if ok else "  FAIL ") + what)
    fails += not ok


with open(nl.PATH, encoding="utf-8") as f:
    raw = json.load(f)
check(raw.get("schema") == 1 and len(raw["rows"]) == len(nl.rows()) >= 5, f"never_list.json loads, every row kept ({len(nl.rows())} rows)")
ids = [r["id"] for r in nl.rows()]
check(len(ids) == len(set(ids)), "row ids are unique")
check(all(r.get("loss") and r.get("source") and ("value" in r or "min" in r) for r in nl.rows()), "every row says the measured loss, its source row and the value it lists")
check({"dflash-draft", "vk-no-host-visible-vidmem", "ub-8192", "mtp-dense-128k"} <= set(ids), "seeded from the ledger losses: DFlash draft, Vulkan host-visible vidmem off, -ub 8192, MTP dense 128k")

vk = {"backend": "vulkan", "model": "moe", "ctx": 65536}
cu = {"backend": "cuda", "model": "moe", "ctx": 65536}
check(nl.backend_of("Available devices:\n  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15000 MiB free)\n") == "vulkan", "backend from the device listing: Vulkan")
check(nl.backend_of("Available devices:\n  CUDA0: NVIDIA GeForce RTX 4060 (8188 MiB, 7600 MiB free)\n") == "cuda", "backend from the device listing: CUDA")
check(nl.backend_of("") is None, "no listing: backend unknown")
env_edit = [{"dim": "vk", "env": "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "value": "1"}]
check(nl.veto(env_edit, vk) and not nl.veto(env_edit, cu) and not nl.veto(env_edit, {"ctx": 65536}), "Vulkan env row: vetoed on Vulkan only (CUDA and an unknown backend: not)")
ub8k = [{"dim": "ub", "flag": "-ub", "value": "8192", "also": ["--ubatch-size"]}]
check(nl.veto(ub8k, cu) and not nl.veto([dict(ub8k[0], value="4096")], cu), "-ub 8192 (and up) vetoed, -ub 4096 not")
check(nl.veto([{"dim": "ub", "flag": "--ubatch-size", "value": "16384"}], cu), "-ub row also matches the long spelling")
mtp = [{"dim": "mtp", "flag": "--spec-type", "value": "draft-mtp"}]
check(nl.veto(mtp, {"backend": "vulkan", "model": "dense", "ctx": 131072}) and not nl.veto(mtp, {"backend": "vulkan", "model": "dense", "ctx": 65536})
      and not nl.veto(mtp, {"backend": "vulkan", "model": "moe", "ctx": 131072}), "MTP row: dense at 128k+ only (64k dense and 128k MoE stay tunable)")
check(nl.veto([{"dim": "ncmoe", "flag": "--n-cpu-moe", "delta": -2}], cu) is None, "placement deltas are never listed")

# the auto-tune filter in the launcher
cands = [at.Candidate("C0", "plan", []), at.Candidate("ub8192", "-ub 8192", ub8k), at.Candidate("np1", "-np 1", [{"dim": "np", "flag": "-np", "value": "1"}])]
kept, out = mc.never_filter(cands, cu)
check([c.id for c in kept] == ["C0", "np1"] and out == ["ub8192 (PREFILL-185)"], f"never_filter drops the listed candidate, keeps C0 and the rest ({[c.id for c in kept]}, {out})")
kept, out = mc.never_filter(cands[:1], cu)
check([c.id for c in kept] == ["C0"] and out == [], "C0 alone: kept")

# notes for the user's own settings
notes = nl.user_notes(["llama-server", "-ub", "8192", "--spec-type", "draft-dflash"], [], cu, environ={})
check(len(notes) == 2 and all("yours is kept" in n for n in notes) and any("PREFILL-185" in n for n in notes) and any("DECODE-704" in n for n in notes),
      "user's -ub 8192 and DFlash: two notes naming the evidence, nothing refused")
check(nl.user_notes(["llama-server", "-ub", "4096"], [], cu, environ={}) == [], "an unlisted value: no note")
check(len(nl.user_notes(["llama-server"], [], vk, environ={"GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM": "1"})) == 1, "the user's environment variable on Vulkan: one note")
check(nl.user_notes(["llama-server"], [], cu, environ={"GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM": "1"}) == [], "...on CUDA: out of scope, no note")
check(nl.user_notes(["llama-server", "-ub=8192"], [], cu, environ={}) != [], "-ub=8192 spelling is seen too")
check(nl.rows(os.path.join(HERE, "no-such-file.json")) == [], "a missing list vetoes nothing")

print("PASS (0 failed)" if not fails else f"FAIL ({fails} failed)")
sys.exit(1 if fails else 0)
