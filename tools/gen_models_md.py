#!/usr/bin/env python3
"""Write the model table of docs/MODELS.md from models.json (between the two marker lines); `--check` exits 1 when the doc differs from what the JSON
would give (CI: tests/models_catalogue_test.py and the quick workflow). Standard library only."""
import os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_catalogue as cat  # noqa: E402

DOC = os.path.join(HERE, "docs", "MODELS.md")
BEGIN = "<!-- models:begin (generated from models.json by tools/gen_models_md.py: edit the JSON, then run the tool; CI checks it) -->"
END = "<!-- models:end -->"
USE_WORDS = {"agents": "agents", "chat": "chat", "docs": "documents"}


def speed(m, machines):
    """The best-known decode number: the strongest machine first, its shortest context."""
    dec = sorted((r for r in m["measured"] if r["metric"] == "decode_tps" and r.get("covers_fit", True)),
                 key=lambda r: (-machines[r["machine"]]["cap_gb"], r.get("ctx") or 0))
    if not dec:
        return "not measured"
    r = dec[0]
    return f"{r['value']:g} tok/s decode" + (f" at {r['ctx'] // 1024}k ctx" if r.get("ctx") else "") + f" ({machines[r['machine']]['name'].split(':')[0]}, {r['source'].split(';')[0]}, {r['date']})"


def needs(m):
    f = m["fit"]
    if f.get("stock_only"):
        return "must fit in the server's RAM (stock llama.cpp)" + (f"; llama.cpp b{f['min_build']}+" if f.get("min_build") else "")
    bits = ["stock if it fits, else moe-cache"]
    if f.get("min_cap_gb"):
        bits.append(f"server memory {f['min_cap_gb']} GB+")
    if f.get("min_vram_gib"):
        bits.append(f"GPU {f['min_vram_gib']} GiB+")
    if f.get("needs_ssd") or f.get("needs_ssd_if_oversize"):
        bits.append("SSD")
    if f.get("min_free_disk_bytes"):
        bits.append(f"{f['min_free_disk_bytes'] / 1e9:.0f} GB free disk")
    if f.get("min_build"):
        bits.append(f"llama.cpp b{f['min_build']}+")
    return ", ".join(bits)


def render(doc=None):
    doc = doc or cat.load()
    ms = doc["machines"]
    out = [BEGIN, "",
           "| Id | Model | Quant | Size (files) | Good for | Needs | Measured | MTP head | License |", "|---|---|---|---|---|---|---|---|---|"]
    for m in doc["models"]:
        out.append("| " + " | ".join([
            f"`{m['id']}`", m["name"], m["quant"], f"{m['bytes'] / 1e9:.1f} GB ({len(m['files'])})",
            ", ".join(USE_WORDS[u] for u in m["uses"]) + (" (not " + ", ".join(USE_WORDS[u] for u in m["avoid_uses"]) + ")" if m["avoid_uses"] else ""),
            needs(m), speed(m, ms), m["mtp"] or "unknown", m["license"] or "not stated"]) + " |")
    out += ["", "Machines the numbers come from: " + "; ".join(f"**{k}** = {v['name']}" for k, v in ms.items()) + ". Every number, with its date and RESULTS section or claim, is in `models.json`.", ""]
    for m in doc["models"]:
        items = m["notes"] + ["**Warning:** " + w for w in m["warnings"]]
        if items:
            out.append(f"- `{m['id']}` ([{m['hf']['repo']}](https://huggingface.co/{m['hf']['repo']})): " + " ".join(items))
    out += ["", END]
    return "\n".join(out)


def apply(text, block):
    a, b = text.index(BEGIN), text.index(END) + len(END)
    return text[:a] + block + text[b:]


def main(argv):
    text = open(DOC, encoding="utf-8").read()
    new = apply(text, render())
    if "--check" in argv:
        if new != text:
            print("docs/MODELS.md is out of date: run python3 tools/gen_models_md.py")
            return 1
        print("docs/MODELS.md matches models.json")
        return 0
    open(DOC, "w", encoding="utf-8", newline="\n").write(new)
    print("docs/MODELS.md written")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
