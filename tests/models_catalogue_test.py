#!/usr/bin/env python3
"""The model catalogue (models.json): its shape and sources, the one fit rule (8/16/24 GB GPUs x 16/32/64 GB RAM x 32k/64k/256k context), pick and launch
reading it, `moe_cache.py models`, and that docs/MODELS.md is what tools/gen_models_md.py makes from it. No model, no llama-server, no network.
   python3 tests/models_catalogue_test.py"""
import contextlib, io, json, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-cat-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
os.environ.pop("MOE_CACHE_MAX_RAM_GB", None)
os.environ["HOME"] = os.environ["USERPROFILE"] = os.path.join(TMP, "home")
os.environ.pop("OLLAMA_MODELS", None)
sys.path.insert(0, os.path.join(HERE, "bin"))
sys.path.insert(0, os.path.join(HERE, "tools"))
import moe_cache as mc  # noqa: E402
import moe_cache_catalogue as cat  # noqa: E402
import moe_cache_pick as pk  # noqa: E402
import moe_cache_models as mm  # noqa: E402
import moe_cache_launch as ln  # noqa: E402
import gen_models_md as gen  # noqa: E402
pk.OLLAMA_SYSTEM_ROOTS = []

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

GIB = 2**30
def machine(ram_gib, vram_gib, kind="SSD", build=11966, local=None, disk_gb=800):
    t = ram_gib * GIB
    return {"os": "test", "ram_total": t, "ram_cap": t - max(4 * GIB, t / 8), "gpu": "GPU" if vram_gib else None, "vram": vram_gib * GIB if vram_gib else None,
            "disk_path": "/models", "disk_free": disk_gb * 1e9, "disk_kind": kind, "avx2": True, "avx512": False, "llama_build": build, "local": local or {}}

def v(spec, k, ctx, use):
    return pk.fit(spec, k, ctx, use)["verdict"]

# ---- the file ----
doc = cat.load()
check(cat.problems(doc) == [], "models.json: every entry complete, bytes = sum of the shards, every number has machine, date and source (" + "; ".join(cat.problems(doc)) + ")")
ids = [m["id"] for m in doc["models"]]
check(ids == list(pk.MODELS) and len(ids) == 8, f"the picker's MODELS are the catalogue's ({len(ids)} models)")
check(all(f["sha256"] and re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) for m in doc["models"] for f in m["files"]), "every shard has its sha256 (the HF LFS oid)")
check(all(m["hf"]["revision"] and m["license"] for m in doc["models"]), "every model has an HF revision and a licence")
check(all(re.search(r"RESULTS \d|claim |benchmarks/|docs/", r["source"]) for m in doc["models"] for r in m["measured"]), "every measured number names its RESULTS section, claim or file")
check(pk.MODELS["gpt-oss-120b"]["sha256"] == "582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d" and pk.MODELS["deepseek-v41-flash"]["link"][0].endswith("00006-of-00006.gguf")
      and pk.MODELS["deepseek-v41-flash"]["min_free"] == 240e9 and pk.MODELS["qwen3.6-35b"]["ollama"] == "qwen3.6:35b-a3b", "legacy view keeps sha256, link, min_free and the Ollama tag")
bad = json.loads(json.dumps(doc)); bad["models"][0]["measured"][0]["source"] = ""
check(any("source" in p for p in cat.problems(bad)), "a number without a source is refused")

# ---- the fit rule ----
pc = machine(32, 16)
check(v(pc, "bolt", 65536, "agents") == "fits" and v(pc, "bolt", 131072, "agents") == "fits" and v(pc, "bolt", 262144, "agents") == "fits", "32 GB + 16 GB GPU: Bolt fits at 64k, 128k and 256k (measured to 228k tokens)")
check(v(pc, "flash-next-iq3_xxs", 32768, "chat") == "fits" and v(pc, "flash-next-q2_0", 32768, "chat") == "fits", "32 GB + 16 GB GPU: both Flash-Next files fit for chat")
check(v(pc, "flash-next-iq3_xxs", 262144, "agents") == "tight" and "longer than any we ran" in pk.fit(pc, "flash-next-iq3_xxs", 262144, "agents")["reason"], "IQ3_XXS at 256k: tight, 'longer than any we ran' (only Q2_0 was run at 256k)")
check(v(pc, "flash-next-q2_0", 262144, "agents") == "fits", "Q2_0 at 256k: fits (RESULTS 64, 67)")
check(v(pc, "deepseek-v41-flash", 32768, "chat") == "tight" and v(pc, "deepseek-v41-flash", 65536, "agents") == "no", "DeepSeek: tight for chat (never run), no for agents")
small = machine(16, 8)
check(v(small, "flash-next-q2_0", 32768, "chat") == "no" and v(small, "flash-next-q2_0", 16384, "chat") == "tight", "16 GB + 8 GB GPU: Q2_0 no at 32k, tight at 16k (--fast only)")
check(v(small, "bolt", 65536, "agents") == "no" and v(small, "glm-5.3", 32768, "chat") == "no", "16 GB: Bolt and GLM do not fit in the memory the server may use: no")
check(v(small, "qwen3.6-35b", 65536, "agents") == "fits", "16 GB + 8 GB GPU: Qwen3.6 fits (measured at 12 GB + 8 GB GPU)")
check(v(machine(32, 8), "bolt", 65536, "agents") == "tight" and v(machine(32, 8), "flash-next-q2_0", 32768, "chat") == "tight", "32 GB + 8 GB GPU: Bolt and Q2_0 tight (measured only with a 16 GB GPU)")
check(v(machine(32, 0), "flash-next-q2_0", 32768, "chat") == "no" and v(machine(32, 0), "bolt", 65536, "agents") == "tight", "no GPU: Flash-Next no (needs 6 GiB of VRAM), Bolt tight")
check(v(machine(64, 24), "glm-5.3", 32768, "chat") == "no" and v(machine(128, 16), "glm-5.3", 32768, "chat") == "fits", "GLM: no at 64 GB, fits at 128 GB")
check(v(machine(32, 16, kind="HDD"), "flash-next-q2_0", 32768, "chat") == "no" and v(machine(32, 16, kind="HDD"), "bolt", 65536, "agents") == "fits", "hard disk: bigger-than-RAM models no, a model that fits is fine")
check(v(machine(32, 16, kind=None), "flash-next-q2_0", 32768, "chat") == "tight", "drive type unknown: bigger-than-RAM models tight")
check(v(machine(32, 16, build=11000), "bolt", 65536, "agents") == "no" and "b11966" in pk.fit(machine(32, 16, build=11000), "bolt", 65536, "agents")["reason"], "llama.cpp b11000: Bolt no, the reason names b11966")
check(v(machine(32, 16, build=None), "bolt", 65536, "agents") == "tight", "llama.cpp build unknown: Bolt tight")
check(v(machine(8, 0), "qwen3.6-35b", 262144, "agents") == "no" and "KV cache" in pk.fit(machine(8, 0), "qwen3.6-35b", 262144, "agents")["reason"], "8 GB RAM, no GPU, 256k: the KV cache alone does not fit: no")
check(v(machine(32, 16), "gpt-oss-120b", 65536, "agents") == "tight", "gpt-oss at 64k for agents: tight (untested)")

# the full grid, every model: each answer is one of the three and has a reason; verdicts only get worse as the machine shrinks
rank = {"fits": 0, "tight": 1, "no": 2}
grid_ok = True
for k in pk.CAT:
    for use in pk.USES:
        for ctx in (32768, 65536, 262144):
            last = None
            for ram in (64, 32, 16):
                for vram in (24, 16, 8):
                    f = pk.fit(machine(ram, vram), k, ctx, use)
                    grid_ok &= f["verdict"] in rank and bool(f["reason"])
            for ram in (64, 32, 16):            # at a fixed GPU, less RAM never fits better
                a, b = (pk.fit(machine(r, 16), k, ctx, use)["verdict"] for r in (ram, ram // 2 if ram > 16 else 16))
                grid_ok &= rank[b] >= rank[a] or a == b
check(grid_ok, "8/16/24 GB GPU x 16/32/64 GB RAM x 32k/64k/256k x every use: always a verdict and a reason; less RAM never fits better")

# ---- pick and launch read it ----
SERIES = [(8, 16), (16, 16), (24, 16), (8, 32), (16, 32), (24, 32), (8, 64), (16, 64), (24, 64)]     # (vram, ram)
ok = True
for vram, ram in SERIES:
    for use in pk.USES:
        r = pk.recommend(machine(ram, vram), use)
        ok &= r["fit"]["verdict"] in ("fits", "tight") and set(r["fit"]) == {"verdict", "reason", "reasons"} and all("fit" in a for a in r["alternatives"])
check(ok, "pick: every recommendation on the grid carries a fit verdict (never 'no') and its alternatives too")
r = pk.recommend(pc, "chat")
check(r["id"] == "flash-next-q2_0" and r["fit"]["verdict"] == "fits" and r["alternatives"][0]["id"] == "flash-next-iq3_xxs", "PC chat: Q2_0 is the default Flash-Next file (owner 2026-11-05), IQ3_XXS the higher-quality sibling, verdict fits")
r = pk.recommend(pc, "agents")
check(r["id"] == "bolt" and r["ctx"] == 131072 and r["fit"]["verdict"] == "fits", "PC agents: Bolt at 128k, fits")
with contextlib.redirect_stdout(io.StringIO()) as out:
    pk.pick(["--use", "agents", "--json", "--models-dir", TMP])
js = json.loads(out.getvalue())
check(js["recommendations"][0]["fit"]["verdict"] in rank and js["recommendations"][0]["fit"]["reason"], "pick --json: the recommendation has fit.verdict and fit.reason")
lines = "\n".join(ln.candidate_lines(1, r, True))
check("fit for this PC: fits (" in lines, "launch: the candidate card shows the fit verdict and reason")
check(ln.pk.MODELS is pk.MODELS, "launch reads the picker's catalogue view")

# ---- the models command ----
def run_models(argv, spec):
    with contextlib.redirect_stdout(io.StringIO()) as o:
        mm.models(argv, spec)
    return o.getvalue()
here = os.path.join(TMP, "m", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
os.makedirs(os.path.dirname(here)); open(here, "wb").write(b"x")
spec = machine(32, 16, local={"qwen3.6-35b": here})
txt = run_models([], spec)
check("FITS  bolt" in txt and "downloaded: " + here in txt and "not downloaded" in txt, "models: verdict, downloaded path and 'not downloaded'")
check("RESULTS 65" in txt and "27.9 tok/s decode" in txt and "2026-10-28" in txt, "models: the measured speed comes with its source and date")
check("glm-5.3" not in txt and "do not fit this PC" in txt, "models: what does not fit is hidden and counted")
check("glm-5.3" in run_models(["--all"], spec), "models --all: lists the models that do not fit")
rows = json.loads(run_models(["--json", "--all"], spec))
check([r["id"] for r in rows["models"]][:1] == ["qwen3.6-35b"] or rows["models"][0]["fit"]["verdict"] == "fits", "models --json: sorted fits, tight, no")
check({"id", "name", "fit", "bytes", "downloaded", "path", "speed", "license", "source"} <= set(rows["models"][0]) and rows["ctx"] == 65536 and rows["use"] == "agents", "models --json: the fields")
q = next(r for r in rows["models"] if r["id"] == "qwen3.6-35b")
check(q["downloaded"] and q["path"] == here and q["bytes"] == 1, "models --json: a downloaded model reports its path and the file's own size")
g = next(r for r in json.loads(run_models(["--json", "--all"], machine(16, 8)))["models"] if r["id"] == "glm-5.3")
check(g["speed"] is None and "not measured on a PC like this one" in run_models(["--all"], machine(16, 8)), "models: 'not measured' where nothing was measured on a PC like this")
check("fit" in json.loads(run_models(["--json", "--use", "chat", "--ctx", "32768"], spec))["models"][0] and json.loads(run_models(["--json", "--ctx", "32768"], spec))["ctx"] == 32768, "models --ctx / --use")
r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "models", "--help"], capture_output=True, text=True, env=dict(os.environ))
check(r.returncode == 0 and "--all" in r.stdout, "moe_cache.py models is wired in (--help)")
r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "--help"], capture_output=True, text=True, env=dict(os.environ))
check("moe_cache.py models" in r.stdout + r.stderr, "moe_cache.py help lists models")

# ---- the doc and the tools ----
check(gen.main(["--check"]) == 0, "docs/MODELS.md is what tools/gen_models_md.py makes from models.json")
changed = json.loads(json.dumps(doc)); changed["models"][0]["quant"] = "Q9_X"
check(gen.apply(open(gen.DOC, encoding="utf-8").read(), gen.render(changed)) != open(gen.DOC, encoding="utf-8").read(), "the doc check notices a changed JSON")
src = open(os.path.join(HERE, "tools", "check_model_links.py"), encoding="utf-8").read()
check("moe_cache_catalogue" in src and "pick.MODELS" not in src, "tools/check_model_links.py reads the catalogue")
for f in ("tools/make_release.sh", "tools/make_windows_kit.sh"):
    check("models.json" in open(os.path.join(HERE, f), encoding="utf-8").read(), f"{f} ships models.json")

# a damaged or incomplete models.json: one plain sentence, no traceback (AUD-040)
for label, text in (("cut-off JSON", open(os.path.join(HERE, "models.json"), encoding="utf-8").read()[:500]),
                    ("a model without its files", json.dumps({"schema": 1, "machines": {}, "models": [{"id": "x"}]})),
                    ("not an object", "[1, 2]")):
    bad = os.path.join(TMP, "bad_models.json"); open(bad, "w", encoding="utf-8").write(text)
    r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "models"], capture_output=True, text=True,
                       env=dict(os.environ, MOE_CACHE_MODELS_JSON=bad), timeout=60)
    check(r.returncode != 0 and "Traceback" not in r.stderr and "model catalogue" in (r.stdout + r.stderr), f"{label}: `models` says the catalogue is damaged, no traceback")

print("models catalogue test: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)
