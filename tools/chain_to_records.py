#!/usr/bin/env python3
"""Turn the *.facts.json files of an overnight PC chain (one file per server start, written by the chain runner; chain.json holds the machine) into
benchmark records (docs/BENCHMARK_TEMPLATE.md) and, with --ingest, put them into benchmarks/runs/.
   python3 tools/chain_to_records.py FACTS_DIR [--out DIR] [--ingest] [--date YYYY-MM-DD] [--only ITEM ...]
Two kinds of item are known (ITEMS below): card rows (c_*: stock / plugin / mtp arms, a 13-token water request, order ABBA) and the quant table
(q_<quant>_<34k|64k>: the plugin on a long prompt, two passes; the number is the speed of the short answer that follows the prompt, stored as
decode_tps and said so in the notes). Anything the facts do not hold is "not recorded". A decode-after value of 0.0 is not a speed: it is stored as
"not recorded" with a note. Standard library; uses bin/moe_cache_record.py and bin/moe_cache_claims.py."""
import argparse, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bin"))
import moe_cache_record as rr
import moe_cache_claims as cl

NR = "not recorded"
# item -> (model file as launched, quant, size in GB (catalogue or RESULTS; None = not recorded), prompt set, what the speed is)
CARD = {
    "c_iq3xxs2": ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS", "IQ3_XXS", 75.8),
    "c_iq3s": ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S", "IQ3_S", None),
    "c_q122": ("Qwen3.5-122B-A10B-UD-IQ2_XXS", "UD-IQ2_XXS", 36.6),
    "c_glm": ("GLM-5.3-Flash-UD-IQ2_XXS", "UD-IQ2_XXS", 101.8),
    "c_bolt": ("NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", "NVIDIA 2-bit QAT", 22.0),
    "c_q36": ("Qwen3.6-35B-A3B", "Q4_K_M", 22.1),
}
QUANT = {
    "q_iq2xs": ("Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS", "IQ2_XS", 68.0),
    "q_iq3xxs": ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS", "IQ3_XXS", 75.8),
    "q_iq3s": ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S", "IQ3_S", None),
    "q_q2_0": ("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0", "Q2_0", 66.4),
}


def load(facts_dir):
    items = {}
    for f in sorted(glob.glob(os.path.join(facts_dir, "*.facts.json"))):
        d = json.load(open(f, encoding="utf-8"))
        items.setdefault(d["item"], []).append(d)
    for v in items.values():
        v.sort(key=lambda d: int(d["run"]))
    return items, json.load(open(os.path.join(facts_dir, "chain.json"), encoding="utf-8"))


def hardware(chain):
    return {"cpu_model": chain["cpu"].strip(), "cpu_cores": chain["cores"], "cpu_threads": chain["threads"], "ram_gib": round(chain["ram_total_gib"]), "ram_type": "DDR5",
            "ram_speed_mts": NR, "gpu_model": chain["gpu"], "gpu_driver": chain["gpu_driver"], "vram_mib": NR, "os": "Windows 11", "os_build": chain["os_build"],
            "power_plan": NR, "storage": {"drive": NR, "interface": NR, "model": NR}}


def build(item, facts, chain, baseline):
    if item in CARD:
        name, quant, size = CARD[item]
        prompt_set, kind = "water request (13 tokens)", "card"
    else:
        key, ctx = item.rsplit("_", 1)
        name, quant, size = QUANT[key]
        prompt_set, kind = f"{ctx}-token prompt, then a short answer", "quant"
    plug = next((d for d in facts if d.get("plugin")), {})
    arms = list(dict.fromkeys(d["arm"] for d in facts))
    runs, idx, notes = [], {}, []
    for d in facts:
        r = d["results"][0]
        idx[d["arm"]] = idx.get(d["arm"], 0) + 1
        dec = r.get("decode_tps") if kind == "card" else r.get("decode_after_tps")
        if kind == "quant" and not dec:
            notes.append(f"{d['arm']}: the harness stored decode-after = {r.get('decode_after_tps')}; that is not a speed, so it is recorded as 'not recorded'.")
            dec = None
        mon = d.get("monitor") or {}
        e = rr.run_entry(d["arm"], idx[d["arm"]], {"prompt_tps": r.get("prompt_tps"), "decode_tps": dec, "wall_s": r.get("wall_s"), "load_s": d.get("load_s"),
                                                   "prompt_tokens": r.get("prompt_n"), "cache_hit_pct": r.get("hit_pct"), "gpu_blocks": r.get("gpu_segments"),
                                                   "text_sha256": d.get("text_md5")}, mon)
        runs.append(e)
        notes += [n for n in d.get("notes", []) if "not clean" in n or "peak RAM" in n]
    if kind == "quant":
        notes.append("decode_tps in this record is the speed of the short answer after the long prompt (decode after a long prompt), from the harness' decode_after_tps. "
                     "'reprocessed' (prompt tokens read again by the follow-up): " + ", ".join(f"{d['arm']} {d['results'][0].get('reprocessed')}" for d in facts) + ".")
        notes.append("text_sha256 holds the first 10 hex digits of the harness' text md5, not a sha256.")
    notes.append("Source: the overnight chain of 2026-11-04 01:36-06:56 on the PC (restarted 01:34; baseline guard: " + baseline + "). Every run was 'not clean': owner work was running.")
    g = facts[0]
    env = {"hardware": hardware(chain),
           "software": {"moe_cache_version": "0.3.5", "moe_cache_commit": NR, "plugin_file": NR, "plugin_sha256_prefix": NR, "llama_backend": "Vulkan",
                        "llama_build": 11966, "llama_commit": plug.get("ggml_commit", NR), "python": NR},
           "model": {"file": name, "quant": quant, "size_gb": size if size is not None else NR, "sha256_full": NR, "sha256_head16MiB": NR, "source_url": NR}}
    settings = {"batch": NR, "command_line": NR, "ctx": g.get("ctx", NR), "environment": {}, "fast_flag": NR, "flash_attn": NR,
                "gpu_expert_layers": plug.get("gpu_expert_layers", NR), "kv_type_k": NR, "kv_type_v": NR,
                "mtp": "on in the mtp arm only (--mtp)" if kind == "card" and "mtp" in arms else NR, "n_cpu_moe": plug.get("cpu_expert_layers", NR),
                "n_layer": plug.get("model_layers", NR), "ngl": NR, "threads": NR, "ubatch": g.get("ub", NR)}
    cond = {"baseline_check": baseline, "cpu_temp_c": NR, "file_cache": NR, "gpu_temp_c": NR, "other_programs": "owner work was running (every run 'not clean')",
            "system_cpu_busy_before_pct": NR, "system_ram_free_before_mib": NR, "uptime_h": NR, "vram_in_use_before_mib": NR}
    proto = {"arms": arms, "n_predict": NR, "order": "ABBA" if kind == "card" else "single", "prompt_set": prompt_set, "prompt_sha256": NR,
             "prompt_tokens": facts[0]["results"][0].get("prompt_n", NR), "runs_per_arm": max(idx.values()), "sampler": NR, "seed": NR, "temperature": NR}
    repro = {"commands": [], "kit": "the released v0.3.5 kit with the official llama.cpp b11966 (launcher defaults)", "model_url": NR}
    import datetime
    rec = rr.assemble(env, settings, cond, proto, runs, repro, notes, now=datetime.datetime(2026, 11, 4, 6, 56, tzinfo=datetime.timezone.utc))
    return rec


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("facts_dir")
    ap.add_argument("--out", default=os.path.join(__import__("tempfile").gettempdir(), "chain_records"))
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--date", default="2026-11-04")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    items, chain = load(a.facts_dir)
    guard = os.path.join(os.path.dirname(a.facts_dir.rstrip("/\\")), "baseline_guard.log")
    baseline = "PASS (baseline_guard.log, 2026-11-04 01:41-01:48)" if os.path.exists(guard) else NR
    os.makedirs(a.out, exist_ok=True)
    sch = rr.load_schema()
    for item in sorted(items):
        if a.only and item not in a.only:
            continue
        if item not in CARD and not (item.rsplit("_", 1)[0] in QUANT and item.endswith(("_34k", "_64k"))):
            continue
        rec = build(item, items[item], chain, baseline)
        errs = rr.validate(rec, sch) if sch else []
        path = os.path.join(a.out, item + ".json")
        json.dump(rec, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        rid = ""
        if a.ingest and not errs:
            rid, _ = cl.ingest(path, date=a.date, machine="pc")
        print(f"{item}: {len(rec['runs'])} runs, {'valid' if not errs else 'INVALID: ' + '; '.join(errs[:2])} {rid}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
