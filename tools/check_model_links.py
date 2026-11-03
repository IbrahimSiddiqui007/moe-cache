#!/usr/bin/env python3
"""Check that every model in the catalogue (models.json) still exists on Hugging Face: the repo answers, its --include pattern matches the .gguf files the
catalogue lists (as `hf download --include` matches them, on the full path), and their sizes and sha256 (the LFS oid) are the catalogue's. Metadata only,
nothing is downloaded. Exit 1 when a repo is gone, a pattern matches no .gguf or a listed shard is missing; a size / sha256 / file-list difference or a
network error only warns (an HF outage must not fail CI; a changed file says the numbers measured on the old one need a look)."""
import fnmatch, json, os, sys, urllib.error, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_catalogue as cat  # noqa: E402

doc = cat.load()
bad = [f"FAIL catalogue: {p}" for p in cat.problems(doc)]
for b in bad:
    print(b)
for m in doc["models"]:
    key, repo = m["id"], m["hf"]["repo"]
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            files = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (404, 410):          # the repo is gone (a 5xx / 429 / 403 is Hugging Face having a bad day: AUD-098)
            print(f"FAIL {key}: {repo} answers HTTP {e.code}"); bad.append(key)
        else:
            print(f"warn {key}: Hugging Face answers HTTP {e.code} for {repo} (not treated as a deleted repo)")
        continue
    except Exception as e:
        print(f"warn {key}: could not reach Hugging Face ({e})"); continue
    pats = cat.includes(m["hf"])
    hit = {f["path"]: f for f in files if f.get("type") == "file" and f["path"].endswith(".gguf") and any(fnmatch.fnmatch(f["path"], x) for x in pats)}
    if not hit:
        print(f"FAIL {key}: no .gguf in {repo} matches --include {pats}"); bad.append(key); continue
    notes = []
    for f in m["files"]:
        h = hit.get(f["path"])
        if not h:
            print(f"FAIL {key}: shard {f['path']} is gone from {repo}"); bad.append(key); continue
        size, oid = (h.get("lfs") or {}).get("size") or h.get("size", 0), (h.get("lfs") or {}).get("oid")
        if size != f["bytes"]:
            notes.append(f"{f['name']}: {size} bytes on HF vs {f['bytes']} in the catalogue")
        if oid and f.get("sha256") and oid != f["sha256"]:
            notes.append(f"{f['name']}: sha256 changed")
    extra = sorted(set(hit) - {f["path"] for f in m["files"]})
    if extra:
        notes.append("new files match the pattern: " + ", ".join(extra))
    total = sum((f.get("lfs") or {}).get("size") or f.get("size", 0) for f in hit.values())
    print(f"ok   {key}: {len(hit)} file(s), {total / 1e9:.1f} GB" + "".join(f"\n  warn {n}" for n in notes))
sys.exit(1 if bad else 0)
