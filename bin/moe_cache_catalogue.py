#!/usr/bin/env python3
"""The model catalogue (models.json at the repo root): one list of the models moe-cache recommends, read by `pick`, `launch`, `models`,
tools/gen_models_md.py (docs/MODELS.md) and tools/check_model_links.py. Standard library only. This file only loads and checks the JSON and turns an
entry into the dict shape the picker has always used (`legacy`); the fit rule is moe_cache_pick.fit()."""
import json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.environ.get("MOE_CACHE_MODELS_JSON") or os.path.join(HERE, "models.json")      # the variable is for tests
USES = ("agents", "chat", "docs")
METRICS = {"decode_tps": "tok/s decode", "prompt_tps": "tok/s prompt", "tasks_correct": "tasks right (of 202)", "agent_session_s": "s agent session"}


def load(path=None):
    try:
        with open(path or PATH, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        raise SystemExit(f"moe-cache: the model catalogue {path or PATH} cannot be read ({e}); models.json belongs next to bin/ (reinstall, or git checkout models.json)")


def problems(doc):
    """Everything wrong with a catalogue, as a list of sentences (empty = fine). The numbers must say where they come from."""
    bad, ids = [], set()
    machines = doc.get("machines") or {}
    for m in doc.get("models") or []:
        i = m.get("id", "?")
        if i in ids:
            bad.append(f"{i}: duplicate id")
        ids.add(i)
        for k in ("id", "name", "family", "quant", "bytes", "files", "hf", "local_pattern", "uses", "avoid_uses", "fit", "measured", "notes", "warnings"):
            if k not in m:
                bad.append(f"{i}: missing {k}")
        if m.get("files") and m.get("bytes") != sum(f["bytes"] for f in m["files"]):
            bad.append(f"{i}: bytes is not the sum of the files")
        for f in m.get("files", []):
            if not (f.get("name") and f.get("bytes") and (f.get("sha256") is None or len(f["sha256"]) == 64)):
                bad.append(f"{i}: file entry {f.get('name')} incomplete")
        for u in list(m.get("uses", [])) + list(m.get("avoid_uses", [])):
            if u not in USES:
                bad.append(f"{i}: unknown use {u}")
        for r in m.get("measured", []):
            if r.get("metric") not in METRICS or r.get("machine") not in machines or not r.get("date") or not r.get("source") or not isinstance(r.get("value"), (int, float)):
                bad.append(f"{i}: measured entry {r} needs metric, value, machine, date and source")
    return bad


def includes(hf):
    inc = hf["include"]
    return [inc] if isinstance(inc, str) else list(inc)


def legacy(m):
    """The picker's old per-model dict: name, size, pattern, repo, include (+ sha256, min_free, link, ollama)."""
    out = dict(name=m["name"], size=m["bytes"], pattern=m["local_pattern"], repo=m["hf"]["repo"], include=m["hf"]["include"])
    if len(m["files"]) == 1 and m["files"][0].get("sha256"):
        out["sha256"] = m["files"][0]["sha256"]
    if m["fit"].get("min_free_disk_bytes"):
        out["min_free"] = float(m["fit"]["min_free_disk_bytes"])
    if m.get("link"):
        out["link"] = (m["link"]["file"], m["link"]["target"])
    if (m.get("ollama") or {}).get("pull"):
        out["ollama"] = m["ollama"]["pull"]
    return out


def models_dict(doc=None):
    return {m["id"]: m for m in (doc or load())["models"]}
