#!/usr/bin/env python3
"""Model picker (bin/moe_cache_pick.py): the machine -> model table on fake machines, the settings it writes, and `server` / `on-demand` /
`connect` reading the pick. No model, no llama-server, nothing downloaded; Linux and Windows; standard library only.
   python3 tests/picker_test.py"""
import json, os, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-pick-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")          # before the import: the launcher reads it once
os.environ.pop("MOE_CACHE_MAX_RAM_GB", None)
os.environ["HOME"] = os.environ["USERPROFILE"] = os.path.join(TMP, "home")      # not this machine's ~/.ollama or ~/models
os.environ.pop("OLLAMA_MODELS", None)
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc  # noqa: E402
import moe_cache_pick as pk  # noqa: E402
pk.OLLAMA_SYSTEM_ROOTS = []                                  # the Linux service's stores of this machine do not count

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

GIB = 2**30
def cap(ram_gib):         # the launcher's hard cap, as mem_cap() computes it from the total
    t = ram_gib * GIB
    return t - max(4 * GIB, t / 8)

def machine(ram_gib, vram_gib, disk_gb=500, kind="SSD", build=11966, local=None, avx2=True):
    return {"os": "test", "ram_total": ram_gib * GIB, "ram_cap": cap(ram_gib), "gpu": "GPU" if vram_gib else None,
            "vram": vram_gib * GIB if vram_gib else None, "disk_path": "/models", "disk_free": disk_gb * 1e9, "disk_kind": kind,
            "avx2": avx2, "avx512": False, "llama_build": build, "local": local or {}}

check(abs(cap(32) - 28 * GIB) < 1 and abs(cap(16) - 12 * GIB) < 1 and abs(cap(64) - 56 * GIB) < 1, "fake caps follow RAM minus max(4 GiB, 12.5 %)")

# ---- the table: (machine, use) -> model id, context, measured or not ----
TABLE = [
    ("16 GB RAM / 6 GB VRAM laptop", machine(16, 6), {"agents": ("qwen3.6-35b", 65536), "chat": ("qwen3.6-35b", 32768), "docs": ("qwen3.6-35b", 65536)}),
    ("32 GB / 8 GB", machine(32, 8), {"agents": ("bolt", 65536), "chat": ("flash-next-q2_0", 32768), "docs": ("bolt", 65536)}),
    ("32 GB / 16 GB (the PC)", machine(32, 16), {"agents": ("bolt", 131072), "chat": ("flash-next-q2_0", 32768), "docs": ("bolt", 131072)}),
    ("64 GB / 16 GB", machine(64, 16), {"agents": ("bolt", 131072), "chat": ("flash-next-iq3_xxs", 32768), "docs": ("bolt", 131072)}),
    ("32 GB, no GPU", machine(32, 0), {"agents": ("qwen3.6-35b", 65536), "chat": ("qwen3.6-35b", 32768), "docs": ("qwen3.6-35b", 65536)}),
]
recs = {}
for label, m, want in TABLE:
    for use, (mid, ctx) in want.items():
        r = pk.recommend(m, use)
        recs[(label, use)] = r
        check(r["id"] == mid and r["ctx"] == ctx, f"{label}, {use}: {mid} at {ctx} (got {r['id']} at {r['ctx']})")

# measured vs untested wording: only where docs/RESULTS has a number for that class
pc = "32 GB / 16 GB (the PC)"
check(not recs[(pc, "agents")]["untested"] and "47.8 s" in recs[(pc, "agents")]["measured"], "PC agents: measured (OpenCode 47.8 s)")
check(not recs[(pc, "chat")]["untested"] and "22-25 tok/s" in recs[(pc, "chat")]["measured"], "PC chat: measured (Q2_0 22-25 tok/s)")
check(recs[("32 GB / 8 GB", "agents")]["untested"] and recs[("32 GB / 8 GB", "agents")]["measured"] == "untested on this class of machine",
      "32/8 agents (Bolt, measured only with a 16 GB GPU): 'untested on this class of machine'")
check(recs[("32 GB / 8 GB", "chat")]["untested"] and "IQ2_XS" in recs[("32 GB / 8 GB", "chat")]["why"][0], "32/8 chat: untested, names the sibling file that was measured")
check(recs[("64 GB / 16 GB", "chat")]["untested"], "64/16 chat (IQ3_XXS): untested on this class")
check(not recs[("16 GB RAM / 6 GB VRAM laptop", "agents")]["untested"] and "22.6" in recs[("16 GB RAM / 6 GB VRAM laptop", "agents")]["measured"],
      "16/6 agents: Qwen3.6 with moe-cache at 12 GB was measured (22.6 tok/s)")
check("9.9" in recs[("32 GB, no GPU", "agents")]["measured"], "no GPU: the no-GPU measurement is quoted")
check(any("without a GPU" in w for w in recs[("32 GB, no GPU", "agents")]["warnings"]), "no GPU + agents: warns that long prompts are slow")

# engine follows the launcher's own stock-or-plugin rule (moe_cache.decide: model + 1.5 GB in the memory the server may use)
check(recs[(pc, "agents")]["engine"].startswith("stock") and recs[(pc, "chat")]["engine"].startswith("moe-cache"), "engine: Bolt stock, Flash-Next moe-cache on 32 GB")
check(recs[("16 GB RAM / 6 GB VRAM laptop", "chat")]["engine"].startswith("moe-cache"), "engine: Qwen3.6 (21.7 GB) needs moe-cache on 16 GB")
check(recs[(pc, "agents")]["args"] == ["--agent", "--ctx", "131072"] and "--agent" not in recs[(pc, "chat")]["args"], "args: --agent only for agents")
check(any(a["id"] == "flash-next-q2_0" for a in recs[(pc, "agents")]["alternatives"]), "PC agents: the Flash-Next line is offered as the alternative")
check(any(a["id"] == "flash-next-q2_0" and "--fast" in a["why"] for a in recs[("16 GB RAM / 6 GB VRAM laptop", "chat")]["alternatives"]),
      "16/6 chat: Flash-Next offered only as the --fast alternative (measured at 12 GB)")

# downloads: printed with size and source, never run here
d = recs[(pc, "agents")]["download"]
check(d and d["size"] == pk.MODELS["bolt"]["size"] and "huggingface.co/nvidia/" in d["source"] and d["command"].startswith("hf download nvidia/"),
      f"download command with size and source ({d and d['command']})")
check("ollama" in recs[("32 GB, no GPU", "chat")]["download"], "Qwen3.6: the Ollama alternative is printed too")

# ---- memory with the context, and a model of the user's own choosing is warned about, never refused ----
r = pk.recommend(machine(32, 16), "agents")
mem = r["memory"]
check(mem["model"] == pk.MODELS["bolt"]["size"] and mem["kv"] > 0 and abs(mem["total"] - mem["model"] - mem["kv"]) < 1 and mem["available"] > mem["total"],
      f"memory estimate = model + KV at the chosen context ({mem['kv'] / 1e9:.1f} GB, {mem['kv_how']})")
fn = pk.kv_estimate("flash-next-q2_0", 262144)["bytes"] / 2**20
check(abs(fn - (6144 + 1536 + 450)) < 1, f"Flash-Next KV at 256k = the PC's measured buffers ({fn:.0f} MiB)")
check(pk.kv_estimate("bolt", 131072)["bytes"] > pk.kv_estimate("bolt", 65536)["bytes"], "KV grows with the context")
r = pk.recommend(machine(16, 6), "chat", model="glm-5.3")
check(r["id"] == "glm-5.3" and r["ctx"] > 0 and r["untested"], f"a model the user picks is used even if too big for the machine ({r['id']})")
r = pk.recommend(machine(16, 6, disk_gb=20), "agents", model="flash-next-iq3_xxs")
check(r["id"] == "flash-next-iq3_xxs" and any("free disk" in w for w in r["warnings"]) and "chosen by you" in r["why"][0],
      "chosen model without disk space: used, with the disk warning")
r = pk.recommend(machine(32, 16, build=9000), "agents", model="bolt")
check(r["id"] == "bolt" and any("b11966" in w for w in r["warnings"]), "chosen Bolt on an old llama.cpp: used, with the version warning")
r = pk.recommend(machine(16, 6), "chat", model="bolt")
check(r["id"] == "bolt", "chosen model not in the rules for this machine: still used")
try:
    pk.recommend(machine(32, 16), "chat", model="gpt-5"); check(False, "unknown model refused")
except ValueError:
    check(True, "unknown model id refused")

# ---- special cases ----
r = pk.recommend(machine(32, 16, build=9000), "agents")
check(r["id"] == "qwen3.6-35b" and any("b11966" in w for w in r["warnings"]), f"llama.cpp older than b11966: no Bolt, and why ({r['id']})")
r = pk.recommend(machine(32, 16, build=None), "agents")
check(r["id"] == "bolt" and any("could not be read" in w for w in r["warnings"]), "llama.cpp build unknown: Bolt, with the version note")
r = pk.recommend(machine(32, 16, kind="HDD"), "chat")
check(r["id"] == "bolt" and any("hard disk" in w for w in r["warnings"]), f"HDD: only models that fit in RAM ({r['id']})")
r = pk.recommend(machine(32, 16, disk_gb=50), "chat")
check(r["id"] == "qwen3.5-122b" and any("free disk" in w for w in r["warnings"]), f"50 GB free disk: the 122B instead of Flash-Next ({r['id']})")
r = pk.recommend(machine(32, 16, disk_gb=10), "chat")
check(r["id"] == "flash-next-q2_0" and r["fits_disk"] is False and r["download"] is None and r["local"] is None,
      f"10 GB free disk, nothing local: still the model that suits the machine, but no download is offered ({r['id']})")
check(abs(r["disk_missing"] - (pk.MODELS["flash-next-q2_0"]["size"] + 2e9 - 10e9)) < 1 and "disk is too small" in r["why"][0] and "missing" in r["why"][0]
      and any("GB of disk missing" in w for w in r["warnings"]), f"10 GB free disk: says plainly how much disk is missing ({r['disk_missing'] / 1e9:.1f} GB)")
check(any("not on this machine and" in l and "no download is offered" in l for l in pk.rec_lines(r)) and not any("download (you run it" in l for l in pk.rec_lines(r)),
      "10 GB free disk: the text output has no download line")
r = pk.recommend(machine(32, 16, disk_gb=10, local={"qwen3.6-35b": "/m/q36.gguf"}), "chat")
check(r["id"] == "qwen3.6-35b" and r["fits_disk"] and r["local"], "no disk for the best fit but a smaller model is already here: that one, no shortage message")
r = pk.recommend(machine(32, 16, disk_gb=10, local={"glm-5.3": "/m/glm-00001-of-00004.gguf"}), "agents")
check(r["id"] == "bolt" and not r["fits_disk"] and any(a["id"] == "glm-5.3" and "already on this machine" in a["why"] for a in r["alternatives"]),
      "a local model that is not the best fit is listed as 'already here' next to the shortage message")
r = pk.recommend(machine(32, 16, disk_gb=10), "agents", model="flash-next-q2_0")
check(r["id"] == "flash-next-q2_0" and not r["fits_disk"] and r["download"] is None, "a model chosen by the user without disk space: used, warned, no download line")
r = pk.recommend(machine(32, 16, disk_gb=10, local={"flash-next-q2_0": "/m/Q2_0-00001-of-00002.gguf"}), "chat")
check(r["id"] == "flash-next-q2_0" and r["local"] and r["download"] is None, "a model already here needs no disk and no download")
r = pk.recommend(machine(32, 8, local={"flash-next-iq3_xxs": "/m/IQ3.gguf"}), "chat")
check(r["id"] == "flash-next-iq3_xxs" and "already on this machine" in r["why"][0], "the other Flash-Next file is already here: used instead of a 66 GB download")
r = pk.recommend(machine(128, 16), "chat")
check(r["id"] == "glm-5.3" and r["untested"], "128 GB RAM chat: GLM-5.3 fits in RAM, untested on this class")
r = pk.recommend(machine(32, 16, avx2=False), "chat")
check(any("AVX2" in w for w in r["warnings"]), "no AVX2: warned")
try:
    pk.recommend(machine(32, 16), "gaming"); check(False, "unknown use refused")
except ValueError:
    check(True, "unknown use refused")

# llama-server's build number from --version (the PC's own output, and a distro build with no number)
import subprocess
class _R:
    def __init__(s, t): s.stdout, s.stderr = "", t
for text, want in (("version: 0.7.0-dev (build 11669, commit 620d2225c)\nbuilt with Clang", 11669), ("version: 0.5.0-dev (build 1, commit 7fe450e)", None),
                   ("version: 11966 (ff0d396e3)", 11966), ("no version here", None)):
    real, pk.subprocess.run = pk.subprocess.run, (lambda *a, **k: _R(text))
    got = pk.llama_build("x"); pk.subprocess.run = real
    check(got == want, f"llama_build({text.splitlines()[0]!r}) = {got}")
r = pk.recommend(machine(32, 16, build=11669), "agents")
check(r["id"] == "qwen3.6-35b" and any("b11966" in w for w in r["warnings"]), "the PC's llama-vk (build 11669): no Bolt, the version note says why")

# ---- settings: apply() writes ui.json; server / on-demand without a model read it ----
model = os.path.join(TMP, "models", "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf")
os.makedirs(os.path.dirname(model))
open(model, "wb").write(b"GGUF" + b"\0" * 60)
check(pk.find_local([os.path.dirname(model)]) == {"bolt": model}, "find_local: the Bolt file is found by name")
os.makedirs(mc.CACHE_DIR, exist_ok=True)
json.dump({"last": {"port": 8085, "model": "/old.gguf"}, "folders": ["/x"]}, open(pk.SETTINGS_FILE, "w"))
check(pk.picked() is None, "no pick yet: picked() is None")
r = pk.recommend(machine(32, 16, local={"bolt": model}), "agents")
pk.apply(r)
s = json.load(open(pk.SETTINGS_FILE))
check(s["last"]["model"] == model and s["last"]["ctx"] == 131072 and s["last"]["agent"] is True and s["last"]["port"] == 8085 and s["folders"] == ["/x"],
      f"apply: 'last' points at the file with the context and agent mode, the rest kept ({s['last']})")
check(s["pick"]["id"] == "bolt" and s["pick"]["use"] == "agents", "apply: 'pick' records what and why")
check(pk.picked() == (model, 131072, True), f"picked(): (file, context, agent) ({pk.picked()})")
r2 = pk.recommend(machine(32, 16), "chat")
pk.apply(r2)
s = json.load(open(pk.SETTINGS_FILE))
check(s["pick"]["id"] == "flash-next-q2_0" and s["pick"]["model"] is None and s["last"]["model"] == model,
      "apply of a model not downloaded yet: remembered, but Start keeps the last file")
check(pk.picked() is None, "picked(): None while the picked file is not here")
check("you picked" in pk.pick_missing() and "--use chat --apply" in pk.pick_missing(), f"pick_missing: names the pick and the next step ({pk.pick_missing()[:80]})")
try:
    mc.server(["--dry-run"]); _m = ""
except SystemExit as e:
    _m = str(e)
check("you picked" in _m, f"server without a model, pick not downloaded: says so, not 'none picked' ({_m[:100]})")
check("-ggml0.26-" in mc.api_advice("3") and "-ggml0.26-" not in mc.api_advice("2") or os.name == "nt", f"api_advice: the right tarball per API ({mc.api_advice('3')[:70]})")
pk.apply(r)
# audit 2026-10-29 R4-2: ui.json private, written atomically under a lock; the page's save keeps what `pick --apply` wrote meanwhile
if os.name != "nt":
    check(os.stat(pk.SETTINGS_FILE).st_mode & 0o777 == 0o600, f"apply: ui.json is private (0600, {oct(os.stat(pk.SETTINGS_FILE).st_mode & 0o777)})")
import threading
def _writer(i):
    for j in range(25):
        pk.update_settings(lambda d, i=i, j=j: d.__setitem__(f"w{i}", j))
_ts = [threading.Thread(target=_writer, args=(i,)) for i in range(4)]
[t.start() for t in _ts]; [t.join() for t in _ts]
s = json.load(open(pk.SETTINGS_FILE))
check(all(s.get(f"w{i}") == 24 for i in range(4)) and s["folders"] == ["/x"] and s["pick"]["id"] == "bolt",
      f"update_settings: 4 writers x 25 updates, none lost, other keys kept ({ {k: v for k, v in s.items() if k != 'last'} })")
for i in range(4):
    pk.update_settings(lambda d, i=i: d.pop(f"w{i}", None))
import moe_cache_ui as _ui
_st = _ui.Store("llama-server")                      # the page opened before the pick below
pk.apply(r)                                          # `pick --apply` while the page is open
_st.saved.setdefault("folders", []).append("/y"); _st.save("folders")
s = json.load(open(pk.SETTINGS_FILE))
check(s["pick"]["id"] == "bolt" and s["folders"] == ["/x", "/y"] and _st.saved["pick"]["id"] == "bolt",
      f"the page's save keeps the pick made meanwhile and takes it in ({s['pick']['id']}, {s['folders']})")
_st.saved["folders"] = ["/x"]; _st.save("folders")

# ---- Ollama: its manifests map a tag to a GGUF blob that llama-server reads; found and listed as local ----
import hashlib
oroot = os.path.join(TMP, "ollama", "models")
def ollama_model(tag_path, payload):
    dg = "sha256:" + hashlib.sha256(payload).hexdigest()
    os.makedirs(os.path.join(oroot, "blobs"), exist_ok=True)
    blob = os.path.join(oroot, "blobs", dg.replace(":", "-"))
    open(blob, "wb").write(payload)
    man = os.path.join(oroot, "manifests", *tag_path.split("/"))
    os.makedirs(os.path.dirname(man), exist_ok=True)
    json.dump({"schemaVersion": 2, "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": dg, "size": len(payload)},
                                              {"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:" + "0" * 64, "size": 1}]}, open(man, "w"))
    return blob
b36 = ollama_model("registry.ollama.ai/library/qwen3.6/35b-a3b", b"GGUF" + b"\0" * 5000)
bhf = ollama_model("hf.co/someone/Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/Q2_K", b"GGUF" + b"\0" * 7000)
open(os.path.join(oroot, "blobs", "sha256-" + "f" * 64), "wb").write(b"GGUF")      # a blob no manifest names
check(pk.ollama_manifests(oroot) and sorted(n for n, _ in pk.ollama_manifests(oroot)) == ["library/qwen3.6:35b-a3b", "someone/Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF:Q2_K"],
      f"ollama manifests: tag -> blob, the unnamed blob is ignored ({[n for n, _ in pk.ollama_manifests(oroot)]})")
check(pk.ollama_name(b36) == "qwen3.6:35b-a3b" and pk.ollama_name(os.path.join(oroot, "blobs", "sha256-" + "f" * 64)) is None, "ollama_name: the tag of a blob; None for an unnamed one")
found = pk.find_ollama(roots=[oroot])
check(found == {"qwen3.6-35b": b36, "bolt": bhf}, f"find_ollama: Qwen3.6 and Bolt tags are recognised ({sorted(found)})")
os.environ["OLLAMA_MODELS"] = oroot
check(pk.ollama_roots() == [oroot], "ollama_roots: $OLLAMA_MODELS first")
check(pk.find_local([os.path.join(TMP, "empty-folder")])["qwen3.6-35b"] == b36, "find_local: Ollama's blob counts as local")
fl = pk.find_local([os.path.dirname(model)])
check(fl["bolt"] == model and fl["qwen3.6-35b"] == b36, "find_local: a file in the model folders wins over Ollama's copy")
check(pk.size_of("qwen3.6-35b", {"qwen3.6-35b": b36}) == os.path.getsize(b36) and pk.size_of("bolt", {}) == pk.MODELS["bolt"]["size"], "size_of: the file's own size when it is here")
ro = pk.recommend(machine(16, 6, local={"qwen3.6-35b": b36}), "agents")
check(ro["id"] == "qwen3.6-35b" and ro["local"] == b36 and ro["download"] is None and ro["size"] == os.path.getsize(b36), "a picked Ollama model is local: no download offered")
check(mc.default_alias(b36) == "qwen3.6-35b-a3b" and mc.default_alias("/m/Big-00001-of-00002.gguf") == "Big" and mc.default_alias("/m/sha256-" + "e" * 64) == "sha256-" + "e" * 64,
      "launcher alias: an Ollama blob gets its tag as the name; ordinary files as before")
del os.environ["OLLAMA_MODELS"]

# `connect claude`: the env block, never a file without --write, a merge with the first backup kept, a private file
env = mc.claude_env("http://127.0.0.1:8080/", "my-model", "KEY123")
check(env == {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8080", "ANTHROPIC_AUTH_TOKEN": "KEY123", "ANTHROPIC_MODEL": "my-model", "ANTHROPIC_SMALL_FAST_MODEL": "my-model"},
      f"connect claude: base URL without /v1 or a trailing slash, token, model ({env})")
check(mc.claude_env("http://h:1", "m", None)["ANTHROPIC_AUTH_TOKEN"] == "none", "connect claude: a server without a key gets the placeholder token 'none'")
txt = mc.claude_snippets(env, 65536)
check('export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"' in txt and '$env:ANTHROPIC_MODEL = "my-model"' in txt, "connect claude: bash and PowerShell lines")
check("KEY123" not in mc.claude_snippets(env, 65536, hide_key=True), "connect claude: the key can be hidden")
cfgdir = os.path.join(TMP, "proj", ".claude")
cfg = os.path.join(cfgdir, "settings.local.json")
check(not os.path.exists(cfg), "connect claude: nothing written before --write")
os.makedirs(cfgdir)
json.dump({"permissions": {"allow": ["Bash(ls)"]}, "env": {"FOO": "1"}}, open(cfg, "w"))
mc.write_claude_settings(cfg, env)
d = json.load(open(cfg))
check(d["permissions"] == {"allow": ["Bash(ls)"]} and d["env"]["FOO"] == "1" and d["env"]["ANTHROPIC_MODEL"] == "my-model", "connect claude --write: merged, the user's other settings kept")
mc.write_claude_settings(cfg, dict(env, ANTHROPIC_MODEL="other"))
check(json.load(open(cfg + ".bak"))["env"] == {"FOO": "1"} and json.load(open(cfg))["env"]["ANTHROPIC_MODEL"] == "other", "connect claude --write twice: the first .bak (what the user had) is kept")
if os.name != "nt":
    check(os.stat(cfg).st_mode & 0o077 == 0 and os.stat(cfg + ".bak").st_mode & 0o077 == 0, "connect claude: the settings file and its backup are private")
open(cfg, "w").write("{ not json")
try:
    mc.write_claude_settings(cfg, env); check(False, "connect claude: broken JSON refused")
except SystemExit:
    check(open(cfg).read() == "{ not json", "connect claude: broken JSON is not touched")

# prompt cache line (item 1): what the launcher chose for agents, in one plain line (the control page lists every 'moe-cache:' line it logs)
n = mc.prompt_cache_note("2048", ["--ctx-checkpoints", "8"], True)
check(n.startswith("moe-cache: prompt cache: up to 2048 MiB") and "8 checkpoints per slot" in n and "default number of slots" in n and "--cache-ram MiB" in n, f"prompt cache line: size, slots, checkpoints ({n})")
n = mc.prompt_cache_note("0", ["-np", "2"], True)
check("off (--cache-ram 0)" in n and "2 slots (-np)" in n, f"prompt cache line: off below 16 GB, -np named ({n})")
check("no --cache-ram" in mc.prompt_cache_note(None, [], False) and "checkpoint" not in mc.prompt_cache_note("2048", ["--ctx-checkpoints", "8"], False), "prompt cache line: old llama-server; no checkpoints outside agent mode")

# the margin the launcher keeps next to what it places (--gpu-margin) is taken off the VRAM in the estimate
a0 = pk.recommend(machine(32, 16), "agents")["memory"]["available"]
a2 = pk.recommend(dict(machine(32, 16), gpu_margin=2), "agents")["memory"]
check(abs(a0 - a2["available"] - (2 - pk.GPU_MARGIN_GIB) * GIB) < 1 and a2["gpu_margin_gib"] == 2, f"gpu margin: --gpu-margin 2 takes 2 GiB off the VRAM in the estimate ({(a0 - a2['available']) / GIB:.2f} GiB less)")
check(any("--gpu-margin 2" in w for w in pk.recommend(machine(32, 16), "agents")["warnings"]), "gpu margin: the long-context agent pick names --gpu-margin 2 for a GPU that also draws the screen")

# `server` without a model uses the pick (stops right after the model check: MOE_CACHE_LIB points to a missing file)
import io, contextlib
os.environ["MOE_CACHE_LIB"] = os.path.join(TMP, "no-plugin.so")
err = io.StringIO()
with contextlib.redirect_stderr(err), contextlib.redirect_stdout(err):
    try:
        mc.server(["--llama-server", os.path.join(TMP, "no-llama-server"), "--always-plugin", "--dry-run"])
    except SystemExit as e:
        msg = str(e)
check("no model given: using the picked one" in err.getvalue() and "context 131072, agent mode" in err.getvalue(),
      f"server without a model: uses the pick ({err.getvalue().strip()[:160]})")
check("MOE_CACHE_LIB points to a file that does not exist" in msg, f"server without a model: got past the model check ({msg[:120]})")
os.remove(pk.SETTINGS_FILE)
try:
    mc.server(["--dry-run"]); msg = ""
except SystemExit as e:
    msg = str(e)
check("moe_cache.py pick" in msg, f"server without a model and no pick: points at `pick` ({msg[:120]})")

# `connect` with no server answering names the picked model
pk.apply(r)
try:
    mc.connect(["opencode", "--url", "http://127.0.0.1:9"]); msg = ""
except SystemExit as e:
    msg = str(e)
check("picked model" in msg and os.path.basename(model) in msg, f"connect without a server: names the picked model ({msg[:160]})")

# `pick --use agents --apply --json` end to end on this machine (real specs; nothing downloaded)
out = io.StringIO()
with contextlib.redirect_stdout(out):
    rc = pk.pick(["--use", "agents", "--json", "--models-dir", os.path.dirname(model)])
j = json.loads(out.getvalue())
check(rc == 0 and j["recommendations"][0]["use"] == "agents" and "ram_cap" in j["machine"], "pick --json: machine and one recommendation")

print("picker test:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
