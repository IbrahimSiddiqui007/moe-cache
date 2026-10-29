#!/usr/bin/env python3
"""Planner / tuner / launcher checks on a real MoE GGUF (I13; CI runs it on Granite 1B). No GPU needed: VRAM is passed in.
   python3 tests/planner_test.py MODEL.gguf
Checks rules that must hold for any model: more VRAM never puts more expert layers on the CPU, the margin is respected, the micro-batch is one the
tuner knows, plan.py's answer is in range, and the launcher passes --vram to the tuner in GiB (audit A12) and --gpu-margin through (audit A05)."""
import json, os, subprocess, sys, types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
import gguf_plan  # noqa: E402

model = sys.argv[1]
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

lay = gguf_plan.model_layout(model)
check(lay["n_layers"] > 0 and lay["expert_bytes_per_layer"] > 0, f"layout: {lay['n_layers']} layers, {lay['expert_bytes_per_layer'] / 1e6:.1f} MB experts per layer")

GIB = 2**30
prev = None
for gib in (2, 3, 4, 6, 8, 12, 16, 24):
    t = gguf_plan.tune(model, 8192, free=gib * GIB, want_cache=False)
    check(t["ub"] in (512, 1024, 2048), f"tune at {gib} GiB: ub {t['ub']} is a known micro-batch")
    ncm = min(t["n_cpu_moe"], lay["n_layers"])
    if prev is not None:
        check(ncm <= prev, f"tune at {gib} GiB: {ncm} CPU expert layers <= {prev} with less VRAM")
    prev = ncm

# a larger margin can only keep the same or more expert layers on the CPU
a = gguf_plan.tune(model, 8192, free=8 * GIB, want_cache=False, reserve_gb=0.5)
b = gguf_plan.tune(model, 8192, free=8 * GIB, want_cache=False, reserve_gb=3.0)
check(min(b["n_cpu_moe"], lay["n_layers"]) >= min(a["n_cpu_moe"], lay["n_layers"]), "a 3 GB margin keeps at least as many expert layers on the CPU as 0.5 GB")

# plan.py (the launcher's first opinion) answers in range
out = subprocess.run([sys.executable, os.path.join(HERE, "planner", "plan.py"), model, "--ctx", "8192", "--vram-gib", "8", "--ram-gib", "16", "--json"],
                     capture_output=True, text=True, timeout=120)
try:
    info = json.loads(out.stdout)
    check(0 <= info["n_cpu_moe"] <= lay["n_layers"], f"plan.py: n_cpu_moe {info['n_cpu_moe']} within 0..{lay['n_layers']}")
except Exception as e:
    check(False, f"plan.py --json: {e} {out.stderr[-300:]}")

# the launcher's dry run: --vram is GiB for the tuner too, --gpu-margin reaches the tuner
def dry(*extra):
    env = dict(os.environ)
    env.setdefault("MOE_CACHE_LIB", os.path.join(HERE, "build", "libggml-moe-cache.so"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "server", model, "--ram", "4", "--always-plugin", "--dry-run", *extra],
                       capture_output=True, text=True, timeout=300, env=env)
    out = r.stdout + r.stderr
    if "plugin not built" in out or "MOE_CACHE_LIB points to a file that does not exist" in out:
        # stop with the reason instead of failing every launcher check (a fresh worktree has no build/; 2026-10-27 that read as 16 planner bugs)
        sys.exit(f"planner_test: the launcher checks need the plugin: run ./build.sh first (or set MOE_CACHE_LIB). Launcher said: {out.strip().splitlines()[-1]}")
    return out
def ncmoe(text):
    import re
    m = re.search(r"--n-cpu-moe (\d+)", text)
    return int(m.group(1)) if m else None
for v in ("2", "4", "8"):
    want = min(gguf_plan.tune(model, 8192, free=float(v) * GIB, want_cache=False)["n_cpu_moe"], 999)
    got = ncmoe(dry("--vram", v))
    check(got == want, f"launcher --vram {v}: --n-cpu-moe {got} equals the tuner at {v} GiB ({want})")
lo, hi = ncmoe(dry("--vram", "8", "--gpu-margin", "0.2")), ncmoe(dry("--vram", "8", "--gpu-margin", "3"))
check(lo is not None and hi is not None and hi >= lo, f"launcher --gpu-margin 3 keeps at least as many CPU expert layers ({hi}) as 0.2 ({lo})")

# Windows core pinning (WINPIN): the launcher's parser of GetLogicalProcessorInformationEx records, on synthetic records (no Windows needed)
import importlib.util, struct  # noqa: E402
_sp = importlib.util.spec_from_file_location("moe_cache_launcher", os.path.join(HERE, "bin", "moe_cache.py"))
L = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(L)
# AMD driver version for the checkpoint message (I44): fake winreg with two display adapters
_keys = {"": ["0000", "0001", "Properties"], "0000": {"DriverDesc": "AMD Radeon(TM) Graphics", "DriverVersion": "31.0.1.1"},
         "0001": {"DriverDesc": "AMD Radeon RX 9070 XT", "DriverVersion": "32.0.23034.4"}}
class _K:
    def __init__(self, n): self.n = n
    def __enter__(self): return self
    def __exit__(self, *a): return False
def _open(parent, sub):
    n = sub if sub in _keys else ""
    return _K(n)
def _query(k, v):
    if k.n not in _keys or not isinstance(_keys[k.n], dict) or v not in _keys[k.n]: raise OSError
    return (_keys[k.n][v], 1)
_fake = types.SimpleNamespace(HKEY_LOCAL_MACHINE=0, OpenKey=_open, QueryInfoKey=lambda k: (3, 0, 0), EnumKey=lambda k, i: _keys[""][i], QueryValueEx=_query)
_oldw, _oldm = L.WINDOWS, sys.modules.get("winreg")
L.WINDOWS = True; sys.modules["winreg"] = _fake
check(L.win_driver_version("AMD Radeon RX 9070 XT") == "32.0.23034.4", "win_driver_version: the discrete adapter's DriverVersion")
check(L.win_driver_version("AMD Radeon(TM) Graphics") == "31.0.1.1", "win_driver_version: the integrated adapter's own version")
check(L.win_driver_version("No such GPU") == "", "win_driver_version: unknown adapter gives ''")
check(L.driver_older("32.0.23034.4") and not L.driver_older("32.0.32015.7029") and not L.driver_older("32.0.32015.7030") and not L.driver_older("32.1.0.1")
      and L.driver_older("31.0.99999.9") and L.driver_older("32.0.32015.7028"), "driver_older: dotted integers compared (32.0.23034.4 old, 32.0.32015.7029 fixed)")
check(L.driver_older("") and L.driver_older("abc") and L.driver_older(None), "driver_older: unknown version counts as older (workaround stays)")
check(not L.driver_older("32.0.32015.7029.1") and not L.driver_older("33"), "driver_older: longer and shorter version strings")
L.WINDOWS = False
check(L.win_driver_version("AMD Radeon RX 9070 XT") == "", "win_driver_version: '' off Windows")
L.WINDOWS = _oldw
if _oldm is None: sys.modules.pop("winreg", None)
else: sys.modules["winreg"] = _oldm

# API key (option C): stable per-user key, 0600, not on the command line, hidden in the launcher's printout, CORS allow-list as environment
import tempfile  # noqa: E402
_kd = tempfile.mkdtemp()
L.CACHE_DIR = _kd; L.KEY_FILE = os.path.join(_kd, "sub", "api.key")
check(L.stored_key() == "", "api key: none until created")
k1 = L.stored_key(create=True); k2 = L.stored_key(create=True)
check(len(k1) >= 30 and k1 == k2 and L.stored_key() == k1, "api key: created once, stable, urlsafe")
if os.name != "nt":
    check(oct(os.stat(L.KEY_FILE).st_mode & 0o777) == "0o600", "api key: file mode 0600")
x = ["-c", "8192", "--api-key", "SECRET", "--alias", "m"]
check(L.take_api_key_from_args(x) == "SECRET" and x == ["-c", "8192", "--alias", "m"], "api key: --api-key KEY removed from llama-server's arguments")
x = ["--api-key=S2", "-c", "1"]
check(L.take_api_key_from_args(x) == "S2" and x == ["-c", "1"], "api key: --api-key=KEY form removed too")
check(L.take_api_key_from_args(["-c", "1"]) is None, "api key: none given -> None")
check(L.key_for("http://127.0.0.1:1", "given") == "given", "api key: an explicit key wins (no request made)")
snip = L.agent_snippets(8080, "m", 65536, "TOPSECRET", hide_key=True)
check("TOPSECRET" not in snip and "bin/moe_cache.py key" in snip, "api key: the launcher's agent printout does not show the key")
check("TOPSECRET" in L.agent_snippets(None, "m", 65536, "TOPSECRET", base="http://x/v1", only="hermes"), "api key: connect's own printout still has the key")

def core(eff, mask, group=0):
    r = bytearray(48); struct.pack_into("<II", r, 0, 0, 48); r[9] = eff; struct.pack_into("<H", r, 30, 1); struct.pack_into("<QH", r, 32, mask, group)
    return bytes(r)
lunar = b"".join([core(1, 1 << i) for i in range(4)] + [core(0, 1 << i) for i in range(4, 8)])           # 4 P + 4 LPE, no hyper-threading
check(L.hybrid_masks(L.parse_core_records(lunar)) == (0xF, [4, 5, 6, 7]), "WINPIN: Lunar Lake 4P+4E -> -C 0xf, efficiency CPUs 4-7")
raptor = b"".join([core(1, 3 << (2 * i)) for i in range(6)] + [core(0, 1 << i) for i in range(12, 16)])    # 6 P with HT + 4 E
check(L.hybrid_masks(L.parse_core_records(raptor)) == (0x555, [12, 13, 14, 15]), "WINPIN: Raptor Lake 6P(HT)+4E -> -C 0x555 (one thread per P core)")
check(L.hybrid_masks(L.parse_core_records(b"".join(core(0, 1 << i) for i in range(8)))) is None, "WINPIN: one efficiency class -> no pinning")
check(L.parse_core_records(core(1, 1, group=1)) == [], "WINPIN: processor group 1 is ignored")
_g = set(); L.parse_core_records(lunar + core(1, 1, group=1), _g)
check(_g == {0, 1}, "WINPIN: the processor groups seen are reported (more than one: the launcher does not pin)")

# agent mode (AG1, I29): what --agent adds to the llama-server command, and the config snippets
ag = dry("--agent")
for want in ("-c 65536", "--jinja", "--ctx-checkpoints 8", "--alias ", "--slot-save-path ", "MOE_CACHE_HOST_RESERVE_MIB=1024"):
    check(want in ag, f"--agent: the command has '{want}'")
check("-c 32768" in dry("--agent", "--ctx", "32768"), "--agent --ctx 32768 keeps the given context")
check("--slot-save-path" not in dry("--agent", "--no-warm"), "--agent --no-warm: no warm-session slot directory")
if gguf_plan.mtp_layers(model)[0] == 0:
    check("--spec-type" not in ag and "MTP off: the model has no MTP head" in ag, "--agent on a model without an MTP head: no --spec-type, one line why")
else:
    check(("MTP on:" in ag) == ("--spec-type draft-mtp --spec-draft-n-max 2" in ag) and "MTP o" in ag, "--agent on a model with an MTP head: one line, args only when on")
check("--jinja" not in dry(), "no --agent: the command is unchanged (no --jinja added)")
check("MOE_CACHE_RAM_GIB=3.725" in dry(), "--ram 4 (GB) reaches the plugin as MOE_CACHE_RAM_GIB=3.725 (GiB), not 4 (B1)")
check("-ctk q8_0 -ctv q8_0" in dry("--fast") and "-ctk" not in dry(), "--fast adds a q8_0 KV cache (I6); the exact default does not")
check(dry("--fast", "--", "-ctk", "f16").count("-ctk") == 1, "--fast keeps a -ctk given after the --")
check("MOE_CACHE_ROUTE_DROP=4" in dry("--fast=2") and "MOE_CACHE_ROUTE_DROP" not in dry("--fast"), "--fast=2 sets MOE_CACHE_ROUTE_DROP=4, --fast does not")
snip = L.agent_snippets(8099, "m1", 65536, None)
check("base_url: http://127.0.0.1:8099/v1" in snip and '"baseURL": "http://127.0.0.1:8099/v1"' in snip and "context_length: 65536" in snip,
      "agent snippets: Hermes and OpenCode blocks name the base URL and the context")

# starter profiles (I18): every shipped profile is a valid XUSE file without a fingerprint, named <arch>-<layers>x<experts>
import glob as _glob, re  # noqa: E402
k = gguf_plan.model_key(model)
check(k is not None and re.fullmatch(r"[A-Za-z0-9_.-]+-\d+x\d+", k or "") is not None, f"model_key: '{k}' has the form arch-LAYERSxEXPERTS")
for sp in _glob.glob(os.path.join(HERE, "profiles", "*-*x*.usage")):
    d = open(sp, "rb").read()
    n_l, n_e = struct.unpack("<ii", d[4:12])
    check(d[:4] == b"XUSE" and len(d) == 12 + 4 * n_l * n_e and str(n_e) == sp.rsplit("x", 1)[1].split(".")[0],
          f"starter profile {os.path.basename(sp)}: XUSE, {n_l} x {n_e}, no fingerprint")

# MTP head (I43): header reader on synthetic GGUFs, the decision table, and the launcher's argparse defaults
def _gguf(path, declared, tensors):
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv = [s("general.architecture") + struct.pack("<I", 8) + s("tst")]
    if declared is not None:
        kv.append(s("tst.nextn_predict_layers") + struct.pack("<II", 4, declared))
    ti = b"".join(s(n) + struct.pack("<IQIQ", 1, 8, 0, 32 * i) for i, n in enumerate(tensors))
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(kv) + ti
    open(path, "wb").write(head + b"\0" * (-len(head) % 32) + b"\0" * 32 * len(tensors))
_md = tempfile.mkdtemp()
for name, declared, tensors, want in [("head", 1, ["blk.0.attn_q.weight", "blk.1.nextn.eh_proj.weight", "blk.1.nextn.enorm.weight"], (1, 1)),
                                      ("dropped", 1, ["blk.0.attn_q.weight"], (0, 1)), ("none", None, ["blk.0.attn_q.weight"], (0, 0))]:
    _p = os.path.join(_md, name + ".gguf"); _gguf(_p, declared, tensors)
    got = gguf_plan.mtp_layers(_p)
    check(got == want, f"mtp_layers: '{name}' file -> {got} (want {want})")
# the tuner's line names the model's own layers apart from the MTP head's block (it has experts too: Qwen3.6 41 = 40 + 1)
_p = os.path.join(_md, "mtpexp.gguf"); _gguf(_p, 1, ["blk.0.ffn_up_exps.weight", "blk.1.ffn_up_exps.weight", "blk.1.nextn.eh_proj.weight"])
_l = gguf_plan.model_layout(_p)
_w = [w for w in gguf_plan.tune(_p, 4096, free=8 * 2**30, want_cache=False)["why"] if "stay on the GPU" in w]
check(_l["n_layers"] == 2 and _l["mtp_blocks"] == 1 and _w and _w[0].startswith("1 of 1 layers of experts, plus the MTP head's block"),
      f"tuner wording with an MTP block: {_l.get('mtp_blocks')} block, {_w}")
_h, _d = gguf_plan.mtp_layers(model)
check(_h == 0 or _d >= 1, f"mtp_layers on the test model: {_h} head layer(s), header declares {_d} (a file with nextn tensors declares them)")
D = L.mtp_decide
check(D(None, True, 1, 1, True, False)[0] is True, "mtp: --agent + head + draft-mtp in llama-server -> on")
check(D(None, False, 1, 1, True, False)[0] is False and "--mtp" in D(None, False, 1, 1, True, False)[1], "mtp: plain server + head -> off with a --mtp hint")
check(D(True, False, 1, 1, True, False)[0] is True, "mtp: --mtp + head -> on")
check(D(False, True, 1, 1, True, False) == (False, "moe-cache: MTP off (--no-mtp)"), "mtp: --no-mtp wins over --agent")
check(D(None, True, 1, 1, False, False)[0] is False and "draft-mtp" in D(None, True, 1, 1, False, False)[1], "mtp: llama-server without draft-mtp -> off, says why")
check(D(None, True, 0, 0, True, False)[0] is False and D(None, False, 0, 0, True, False)[1] is None, "mtp: no head -> off; silent outside --agent")
check("dropped" in D(None, True, 0, 1, True, False)[1], "mtp: header declares a head the quant dropped -> says so")
check(D(None, True, 1, 1, True, True)[0] is False, "mtp: your own --spec-type is left alone")
for args in [(True, True, 0, 0, True, False), (True, True, 1, 1, False, False)]:
    try:
        D(*args); check(False, f"mtp: --mtp with {args} must stop")
    except SystemExit as e:
        check("--mtp:" in str(e), f"mtp: --mtp stops with a reason ({str(e)[:60]}...)")
_said = []; _say = L.say; L.say = lambda m, err=False: _said.append(m)
check(L.mtp_args(None, True, "llama-server", os.path.join(_md, "none.gguf"), []) == [] and len(_said) == 1, "mtp_args: no head -> no args, one line")
check(L.mtp_args(True, False, "llama-server", os.path.join(_md, "head.gguf"), ["--spec-type", "ngram-simple"]) == [], "mtp_args: --mtp keeps your own --spec-type")
L.say = _say

# model-card sampling (I45): the decision table on header dicts, and the header reader on the synthetic GGUF
S = L.sampling_decide
_q = {"arch": "qwen35moe", "name": "", "basename": "", "sampling": {}}
_a, _w = S(_q, "m.gguf", [], False)
check(_a == ["--temp", "1", "--top-p", "0.95", "--top-k", "20", "--min-p", "0", "--presence-penalty", "1.5"] and "thinking" in _w, "sampling: Qwen3.6 thinking = card values, one line")
check(S(_q, "m.gguf", [], True)[0][:2] == ["--temp", "0.6"] and "--presence-penalty" in S(_q, "m.gguf", [], True)[0], "sampling: --agent uses the card's coding values")
check(S(_q, "m.gguf", ["--reasoning", "off"], False)[0][:4] == ["--temp", "0.7", "--top-p", "0.8"], "sampling: --reasoning off -> non-thinking values")
check(S(_q, "m.gguf", ["--chat-template-kwargs", '{"enable_thinking": false}'], False)[0][1] == "0.7", "sampling: enable_thinking false -> non-thinking")
check("--temp" not in S(_q, "m.gguf", ["--temp", "0.3"], False)[0] and "--top-k" in S(_q, "m.gguf", ["--temp", "0.3"], False)[0], "sampling: your --temp is kept, the rest set")
_g = dict(_q, sampling={"temp": 0.6, "top_p": 0.95, "top_k": 20})
_a, _w = S(_g, "m.gguf", [], False)
check(_a == ["--min-p", "0", "--presence-penalty", "1.5"] and "from the GGUF" in _w, "sampling: values the GGUF carries are left to llama.cpp")
check(S({"arch": "qwen4exp", "sampling": {}}, "m.gguf", [], False)[0][-2:] == ["--presence-penalty", "0"], "sampling: Qwen3.8 thinking has presence 0 (its card)")
check(S({"arch": "qwen3moe", "name": "Qwen3-30B-A3B-Instruct-2507", "sampling": {}}, "m.gguf", [], False)[0][1] == "0.7", "sampling: an Instruct name -> non-thinking")
check(S({"arch": "glm5-next", "sampling": {}}, "m.gguf", [], False)[0] == ["--temp", "1", "--top-p", "0.95"], "sampling: GLM gets only what its card states")
check(S({"arch": "granitemoe", "sampling": {}}, "m.gguf", [], False) == ([], None), "sampling: unknown family -> nothing, no line")
check(gguf_plan.sampling_meta(os.path.join(_md, "head.gguf"))["arch"] == "tst", "sampling_meta reads the architecture")
check(L.sampling_args(True, False, os.path.join(_md, "head.gguf"), []) == [], "sampling: --no-sampling-defaults -> nothing")

# connect opencode --write: the global file that wins (OpenCode merges opencode.json, then opencode.jsonc)
_od = tempfile.mkdtemp(); _oj = os.path.join(_od, "opencode.json"); _ojc = os.path.join(_od, "opencode.jsonc")
_env = os.environ.pop("OPENCODE_CONFIG", None)
check(L.opencode_config_target(_oj) == (_oj, []), "opencode target: no .jsonc -> opencode.json, no warning")
open(_ojc, "w").write('{"$schema": "https://opencode.ai/config.json"}')
check(L.opencode_config_target(_oj) == (_ojc, []), "opencode target: plain-JSON opencode.jsonc (made by OpenCode) is the file written")
open(_ojc, "w").write('// mine\n{"model": "x/y"}')
_t, _w = L.opencode_config_target(_oj)
check(_t == _oj and len(_w) == 1 and '"model"' in _w[0], "opencode target: .jsonc with comments and its own model -> opencode.json + a warning")
if _env is not None:
    os.environ["OPENCODE_CONFIG"] = _env

# a micro-batch the user chose is planned around (RESULTS 61: a chosen -ub skipped the tuner, so the long-prompt block found no VRAM)
_f = gguf_plan.tune(model, 40960, free=16 * GIB, want_cache=False, ub=4096)
_g = gguf_plan.tune(model, 40960, free=16 * GIB, want_cache=False, ub=512)
check(_f["ub"] == 4096 and any("as asked" in w for w in _f["why"]), "tune ub=4096: the asked micro-batch is kept")
check(_f["n_cpu_moe"] >= _g["n_cpu_moe"], f"tune: a bigger asked micro-batch keeps at least as many CPU expert layers ({_f['n_cpu_moe']} >= {_g['n_cpu_moe']})")
check(gguf_plan.compute_buffer_guess_mib(65536, 2048) >= gguf_plan.compute_buffer_mib(65536, 1024) and gguf_plan.compute_buffer_guess_mib(8192, 4096) > 0,
      "compute buffer guess: unmeasured pairs get an estimate that errs high")
check(gguf_plan.compute_buffer_guess_mib(524288, 512) >= 2 * gguf_plan.compute_buffer_mib(262144, 512),
      "compute buffer guess: past the largest measured context it grows with the context (T0a: a flat 0.55 GB at 256k starved the long-prompt block)")
for _c in sorted(gguf_plan._COMPUTE_MIB):      # audit 2026-10-29 L1: -ub 2048 at 128k/256k was guessed at a flat 990 MiB (3150 / 3979 measured)
    check(gguf_plan.compute_buffer_guess_mib(_c, 2048) >= gguf_plan.compute_buffer_guess_mib(_c, 1024) >= gguf_plan.compute_buffer_guess_mib(_c, 512),
          f"compute buffer guess at {_c}: 2048 >= 1024 >= 512 ({[gguf_plan.compute_buffer_guess_mib(_c, u) for u in (2048, 1024, 512)]})")
check(gguf_plan.compute_buffer_guess_mib(262144, 2048) >= 3979 and gguf_plan.compute_buffer_mib(262144, 2048) is None,
      "compute buffer: 2048 at 256k guessed at its measured size but still never auto-picked")
_run = gguf_plan.subprocess.run      # audit 2026-10-29 L4: two NVIDIA GPUs print two lines; the tuner read none
gguf_plan.subprocess.run = lambda *a, **k: types.SimpleNamespace(stdout="24564, 300\n12288, 100\n")
try:
    check(gguf_plan.free_vram_bytes() == (24564 - 300) * 1048576, "free VRAM with two NVIDIA GPUs: the first GPU's line")
finally:
    gguf_plan.subprocess.run = _run
_h = gguf_plan.tune(model, 524288, free=16 * GIB, want_cache=False)
check(_h["compute_mib"] and _h["compute_mib"] >= gguf_plan.compute_buffer_mib(262144, 512), f"tune past the table: compute buffer from the scaled guess ({_h['compute_mib']})")

# the launcher's "plugin disabled itself" warning: only the plugin's lines that say it is not caching, not its "Windows refused ..." warnings (F5, PC)
check(L.plugin_off_line(b"moe-cache: no experts found - plugin disabled, the model loads the normal way"), "plugin off: setup failure line")
check(L.plugin_off_line(b"moe-cache: this plugin was built for ggml backend API 3 but the server uses ggml 0.24.0 (API 2): not loaded, the model loads the normal way. x"),
      "plugin off: backend API mismatch line")
check(not L.plugin_off_line(b"moe-cache: MOE_CACHE_WIN_VIEW: Windows refused a working-set minimum of 21.97 GiB (error 1314): resident experts may be trimmed"),
      "plugin off: a refused working-set minimum is a warning, the plugin keeps running")
check(not L.plugin_off_line(b"moe-cache: warning: Windows refused memory for an expert (error 1455): the cache gives experts back and shrinks"),
      "plugin off: a refused expert commit is a warning")
# drive probe (N8): unbuffered random reads of the model file; None for a file too small to probe
_small = os.path.join(_od, "small.bin"); open(_small, "wb").write(b"\0" * 4096)
check(L.drive_read_speed(_small, seconds=0.2) is None, "drive probe: a file under 64 MiB is not probed")
if len(sys.argv) > 1 and os.path.getsize(sys.argv[1]) >= 64 << 20:
    _g = L.drive_read_speed(sys.argv[1], seconds=0.5)
    check(_g is None or _g > 0, f"drive probe on the model: {_g if _g is None else round(_g, 2)} GB/s (None = no unbuffered reads on this file system)")

# T5: --gpu-cache on Vulkan is capped so each of llama.cpp's cache banks fits in one device tensor. Synthetic sparse GGUF with Flash-Next IQ2_XS's
# expert layout (three groups: 34, 11 and 3 layers): at a 2 GiB limit the cap is 7712 MiB (7709 on the real file, its tensors are padded; 8909 aborted on the PC, 5900 loaded)
def _gguf_sized(path, tensors):
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv = [s("general.architecture") + struct.pack("<I", 8) + s("tst")]
    offs, o = [], 0
    for _, size in tensors:
        offs.append(o); o += size
    ti = b"".join(s(n) + struct.pack("<IQIQ", 1, 8, 0, off) for (n, _), off in zip(tensors, offs))
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(kv) + ti
    with open(path, "wb") as f:
        f.write(head + b"\0" * (-len(head) % 32)); f.truncate(len(head) + (-len(head) % 32) + o)     # sparse: no data written
_MB = 1 << 20
_lay = [(256, 256, 225)] * 34 + [(206, 206, 225)] * 11 + [(175, 175, 225)] * 3
_t = [("token_embd.weight", 300 * _MB)]
for _i, (_gt, _up, _dn) in enumerate(_lay):
    _t += [(f"blk.{_i}.ffn_gate_exps.weight", _gt * _MB), (f"blk.{_i}.ffn_up_exps.weight", _up * _MB), (f"blk.{_i}.ffn_down_exps.weight", _dn * _MB),
           (f"blk.{_i}.attn_q.weight", 4 * _MB)]
_p = os.path.join(_md, "banks.gguf"); _gguf_sized(_p, _t)
_c2, _c4, _c64 = (gguf_plan.gpu_cache_cap_mib(_p, b) for b in (2 << 30, 4 << 30, 64 << 30))
check(_c2 == 7712 and _c4 == 15424 and _c64 is None, f"gpu_cache_cap_mib: 2 GiB -> {_c2} (want 7712), 4 GiB -> {_c4} (want 15424), 64 GiB -> {_c64} (want None)")
_cm = gguf_plan.gpu_cache_cap_mib(model, 2 << 30)
check(_cm is None or _cm > 0, f"gpu_cache_cap_mib on the test model: {_cm}")
_lim = L.vk_tensor_limits()
check(isinstance(_lim, list) and all(isinstance(n, str) and b > 0 for n, b in _lim), f"vk_tensor_limits runs: {_lim}")
_run = L.subprocess.run
L.subprocess.run = lambda *a, **k: types.SimpleNamespace(stdout=json.dumps([["AMD Radeon(TM) Graphics", 4 << 30], ["AMD Radeon RX 9070 XT", 2 << 30]]) + "\n")
_vk = "Available devices:\n  Vulkan0: AMD Radeon(TM) Graphics (24000 MiB, 20000 MiB free)\n  Vulkan1: AMD Radeon RX 9070 XT (16304 MiB, 15600 MiB free)\n"
_cu = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB, 7000 MiB free)\n"
_a, _b = L.vk_cache_cap("ls", _p, _vk), L.vk_cache_cap("ls", _p, _cu)
L.subprocess.run = _run
check(_a == (7712, 2 << 30) and _b == (None, None), f"vk_cache_cap: Vulkan listing -> {_a} (smallest listed limit), CUDA listing -> {_b} (no cap)")

# N26: KV + recurrent state from the header (Flash-Next-like hybrid), and the flat fallback for MLA (GLM-like)
def _meta_gguf(path, kv):
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    body = b""
    for k, v in kv:
        body += s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str) else struct.pack("<II", 4, v))
    head = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + body
    open(path, "wb").write(head + b"\0" * (-len(head) % 32))
_fx = os.path.join(_md, "fnx_kv.gguf")
_meta_gguf(_fx, [("general.architecture", "qwen4exp"), ("qwen4exp.block_count", 48), ("qwen4exp.attention.head_count", 24),
                 ("qwen4exp.attention.head_count_kv", 2), ("qwen4exp.attention.key_length", 256), ("qwen4exp.attention.value_length", 256),
                 ("qwen4exp.full_attention_interval", 4), ("qwen4exp.attention.indexer.key_length", 128), ("qwen4exp.ssm.inner_size", 6144),
                 ("qwen4exp.ssm.state_size", 128), ("qwen4exp.ssm.group_count", 16), ("qwen4exp.ssm.time_step_rank", 48), ("qwen4exp.ssm.conv_kernel", 4)])
_k = gguf_plan.kv_bytes(_fx, 65536)
_k256 = gguf_plan.kv_bytes(_fx, 262144)
check(_k256 is not None and abs(_k256 / 2**20 - (6144 + 1536 + 450.28)) < 5, f"kv_bytes: Flash-Next layout at 256k = {(_k256 or 0) / 2**20:.0f} MiB (PC llama-server: 6144 + 1536 + 450 MiB)")
_kq = gguf_plan.kv_bytes(_fx, 65536, kv_q8=True)
check(_kq is not None and _kq < _k, f"kv_bytes: q8_0 KV is smaller ({(_kq or 0) / 2**30:.3f} GiB)")
_gl = os.path.join(_md, "glm_kv.gguf")
_meta_gguf(_gl, [("general.architecture", "glm5-next"), ("glm5-next.block_count", 46), ("glm5-next.attention.head_count", 64),
                 ("glm5-next.attention.key_length_mla", 256)])
check(gguf_plan.kv_bytes(_gl, 65536) is None, "kv_bytes: MLA layout -> None (the flat 40 KB per token stays)")

# N28: at 128k+ in exact mode, one line suggesting --fast (q8_0 KV) when it frees 2+ GPU expert layers; nothing below 128k
def _full_gguf(path, kv, tensors):
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    body = b"".join(s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str) else struct.pack("<II", 4, v)) for k, v in kv)
    offs, o = [], 0
    for _, size in tensors:
        offs.append(o); o += size
    ti = b"".join(s(n) + struct.pack("<IQIQ", 1, 8, 0, off) for (n, _), off in zip(tensors, offs))
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + body + ti
    with open(path, "wb") as f:
        f.write(head + b"\0" * (-len(head) % 32)); f.truncate(len(head) + (-len(head) % 32) + o)
_fkv = [("general.architecture", "qwen4exp"), ("qwen4exp.block_count", 48), ("qwen4exp.attention.head_count", 24), ("qwen4exp.attention.head_count_kv", 2),
        ("qwen4exp.attention.key_length", 256), ("qwen4exp.attention.value_length", 256), ("qwen4exp.full_attention_interval", 4),
        ("qwen4exp.attention.indexer.key_length", 128), ("qwen4exp.ssm.inner_size", 6144), ("qwen4exp.ssm.state_size", 128),
        ("qwen4exp.ssm.group_count", 16), ("qwen4exp.ssm.time_step_rank", 48), ("qwen4exp.ssm.conv_kernel", 4)]
_ft = [("token_embd.weight", 3 << 30)] + [(f"blk.{i}.ffn_down_exps.weight", 700 << 20) for i in range(48)]
_fp = os.path.join(_md, "fnx_full.gguf"); _full_gguf(_fp, _fkv, _ft)
_h256, _h64 = gguf_plan.fast_kv_hint(_fp, 262144, free=15.06 * 2**30), gguf_plan.fast_kv_hint(_fp, 65536, free=15.06 * 2**30)
check(_h256 and "--fast" in _h256 and "more layers of experts on the GPU" in _h256 and _h64 is None, f"fast_kv_hint: 256k -> {_h256}; 64k -> {_h64}")

# hard memory cap (owner 2026-10-29): physical RAM minus max(4 GiB, 12.5 %), MOE_CACHE_MAX_RAM_GB overrides (0 = off), --ram cannot exceed it
_mt, _env = L.mem_total, os.environ.pop("MOE_CACHE_MAX_RAM_GB", None)
_caps = {}
for _gib in (16, 32, 64):
    L.mem_total = lambda g=_gib: g * 2**30
    _caps[_gib] = L.mem_cap() / 2**30
check(_caps == {16: 12.0, 32: 28.0, 64: 56.0}, f"mem_cap: 16/32/64 GiB machines -> {_caps} (want 12/28/56 GiB)")
L.mem_total = lambda: 32 * 2**30
os.environ["MOE_CACHE_MAX_RAM_GB"] = "20"
check(L.mem_cap() == 20e9 and L.mem_for_server("40") == 20e9 and L.mem_for_server("10") == 10e9,
      f"mem_cap: override 20 GB; --ram 40 capped to it, --ram 10 kept ({L.mem_cap()}, {L.mem_for_server('40')}, {L.mem_for_server('10')})")
os.environ["MOE_CACHE_MAX_RAM_GB"] = "0"
check(L.mem_cap() is None and L.mem_for_server("40") == 40e9, "mem_cap: MOE_CACHE_MAX_RAM_GB=0 turns the cap off")
os.environ.pop("MOE_CACHE_MAX_RAM_GB")
if _env is not None:
    os.environ["MOE_CACHE_MAX_RAM_GB"] = _env
L.mem_total = _mt

print(f"planner checks: {len(fails)} failed")
sys.exit(1 if fails else 0)
