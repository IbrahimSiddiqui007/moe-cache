#!/usr/bin/env python3
"""Check that every model the picker can recommend still exists on Hugging Face: the repo answers, its --include pattern matches .gguf files
(as `hf download --include` matches them, on the full path), and their total size is near the size the picker states. Metadata only, nothing is
downloaded. Exit 1 when a repo is gone or a pattern matches no .gguf; a size difference or a network error only warns (an HF outage must not
fail CI)."""
import fnmatch, importlib.util, json, os, sys, urllib.error, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("moe_cache_pick", os.path.join(HERE, "bin", "moe_cache_pick.py"))
pick = importlib.util.module_from_spec(spec); sys.path.insert(0, os.path.join(HERE, "bin")); spec.loader.exec_module(pick)

bad = 0
for key, m in pick.MODELS.items():
    url = f"https://huggingface.co/api/models/{m['repo']}/tree/main?recursive=true"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            files = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"FAIL {key}: {m['repo']} answers HTTP {e.code}"); bad += 1; continue
    except Exception as e:
        print(f"warn {key}: could not reach Hugging Face ({e})"); continue
    hit = [f for f in files if f.get("type") == "file" and f["path"].endswith(".gguf") and fnmatch.fnmatch(f["path"], m["include"])]
    if not hit:
        print(f"FAIL {key}: no .gguf in {m['repo']} matches --include {m['include']}"); bad += 1; continue
    total = sum((f.get("lfs") or {}).get("size") or f.get("size", 0) for f in hit)
    note = "" if abs(total - m["size"]) <= 0.05 * m["size"] else f" (warn: {total / 1e9:.1f} GB on HF vs {m['size'] / 1e9:.1f} GB in the picker)"
    print(f"ok   {key}: {len(hit)} file(s), {total / 1e9:.1f} GB{note}")
sys.exit(1 if bad else 0)
