#!/usr/bin/env python3
"""QUALITY-970: the opt-in `--approx safe|careful|aggressive` tiers of moe-cache-server.

Off by default. A tier is a named group of switches that already exist (q8_0 KV cache, the model's own MTP draft head, --fast, --fast=2); each mode in it carries
the quality gate it must pass (docs/QUALITY_GATE.md, the thresholds are copied here and a test keeps them equal), and the models it was measured on.
The startup canary re-measures the tier on a short text against the exact setting of the same model and steps the tier down when the KL divergence is past
the gate. Standard library only. Everything here that decides (resolve, canary_decide, parse) is pure, so tests/approx_test.py runs without a model.

A canary guards, it never proves: passing it does not make a mode a default (docs/QUALITY_GATE.md: defaults need the full gate and the owner's decision).
"""
import json, os, re, subprocess, hashlib, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TIER_ORDER = ["exact", "safe", "careful", "aggressive"]       # lowest to highest; the canary steps down this list

# gate level -> (max mean KLD, min same-top %, tasks tolerance). The same numbers as LEVELS in tools/quality_gate.py and the table in docs/QUALITY_GATE.md.
GATE_LEVELS = {
    "exact": (1e-5, 100.0, 0),
    "kv": (0.015, 95.0, 2),
    "fast": (0.040, 90.0, 4),
    "fewer": (0.060, 88.0, 5),
}

# mode id -> description. class: E expert skipping, K KV precision, S speculation (one mode per class in a tier).
# verified: lower-case substrings of the model file name this mode has a gate record for ("granite" = the CI smoke guard only, not a quality statement).
MODES = {
    "kv8": {"class": "K", "gate": "kv", "switch": "--fast=kv  (-ctk q8_0 -ctv q8_0)", "changes_answers": True,
            "plain": "the KV cache (the chat's memory) is stored in 8 bit instead of 16 bit: half the memory, answers can differ in a few words",
            "measured": "mean KL 0.0045-0.015 against stock (RESULTS 53, R11; Flash-Next chat 0.014-0.015, same top token 94-95 %, meaning kept in 10 of 10 checks at 55k)",
            "verified": ["flash-next", "qwen3.6", "qwen3.8", "granite"]},
    "spec-exact": {"class": "S", "gate": "kv", "switch": "--mtp  (the model's own draft head, checked by the model)", "changes_answers": True,
                   "plain": "the model guesses 2 tokens ahead with its own draft head and checks them; only when the model has that head. The text can differ slightly (batch rounding)",
                   "measured": "Qwen3.6 on the PC 45.6 -> 70.3 tok/s; GLM-5.3 code text differed slightly",
                   "verified": ["*"]},
    "fast1": {"class": "E", "gate": "fast", "switch": "--fast  (+ q8_0 KV cache)", "changes_answers": True,
              "plain": "experts that are rarely picked and not in RAM are skipped, and the KV cache is stored in 8 bit; about 1 token in 40-50 comes out different",
              "measured": "about +10 %; KL 0.0053 (RESULTS 31) to 0.025 with top-1 93.6 % (25.3)",
              "verified": ["flash-next", "qwen3.6", "granite"]},
    "fast2": {"class": "E", "gate": "fewer", "switch": "--fast=2  (+ q8_0 KV cache)", "changes_answers": True,
              "plain": "up to 4 low-ranked experts per token that are not in RAM are skipped; about 1 token in 28 comes out different. For memory much smaller than the model",
              "measured": "Flash-Next at 12 GB +43 %; KL 0.022; gate level 'fewer' (0.060) is the contract limit for this mode",
              "verified": ["flash-next", "granite"]},
}

# tier -> modes (cumulative in effect: careful and aggressive keep the exact speculation of safe; the expert mode replaces kv8 because --fast already stores the KV in 8 bit)
TIERS = {
    "safe": {"modes": ["kv8", "spec-exact"], "gate": "kv", "changes_answers": True,
             "plain": "answers are NOT identical to stock: 8-bit KV cache and (if the model has the head) its own drafting; every expert is still computed"},
    "careful": {"modes": ["fast1", "spec-exact"], "gate": "fast", "changes_answers": True,
                "plain": "answers are NOT identical to stock: rarely picked experts that are not in RAM are skipped (about 1 token in 40-50 differs), 8-bit KV cache, own drafting"},
    "aggressive": {"modes": ["fast2", "spec-exact"], "gate": "fewer", "changes_answers": True,
                   "plain": "answers are NOT identical to stock: up to 4 low-ranked experts per token are skipped (about 1 token in 28 differs); fastest, least faithful"},
}


# --- LAUNCHER-984: the table of every mode that changes tokens (owner rule 2026-11-09: no opt-in; ON by default when measured as won; --exact turns all of them off) ---
# status: "won" = a paired run shows a speed win AND the quality check passed -> the default enables it; "lost" = measured slower or the meaning changed -> off;
# "unmeasured" = no paired run with a quality check yet -> off until measured. The launcher reads default_on() at each place a default is decided, so changing a status here changes the
# default and nothing else; tests/token_modes_test.py proves the default enables exactly the "won" rows and that --exact enables none.
TOKEN_MODES = [
    {"id": "kv8", "what": "q8_0 KV cache from 128k context on a GPU machine", "switch": "--fast=kv (default from 128k)", "status": "won",
     "reason": "about 2x prompt speed at 256k on a 16 GB card; gate level kv passed (KL 0.0045-0.015, docs/QUALITY_GATE.md); owner 2026-10-30"},
    {"id": "long-prompt", "what": "long-prompt mode from 32k context: -ub 4096 with the q8_0 KV cache, where the memory plan has room", "switch": "--long-prompt (default from 32k)", "status": "won",
     "reason": "Flash-Next IQ3_XXS, 65k conversation, A C C A: prompts +77.0 %, answers +1.8 % (claim v038-longprompt-default); KL 0.014-0.015, meaning kept 10 of 10 at 55k"},
    {"id": "mtp", "what": "the model's own MTP draft head, in agent mode when the MoE model has the head and llama-server supports it (a dense model: only with --mtp, it measured slower)", "switch": "--mtp (default with --agent)", "status": "won",
     "reason": "Qwen3.6 on the PC: decode 45.6 -> 70.3 tok/s; prompt processing 5-8 % slower; GLM-5.3 code text differed slightly"},
    {"id": "fast1", "what": "skip rarely picked experts that are not in RAM, with the q8_0 KV cache", "switch": "--fast", "status": "unmeasured",
     "reason": "about +10 % where memory is much smaller than the model; no paired run with a quality check at 64k on the PC yet"},
    {"id": "fast2", "what": "skip up to 4 low-ranked experts per token that are not in RAM", "switch": "--fast=2", "status": "unmeasured",
     "reason": "Flash-Next at 12 GB on the laptop +43 %, no gain at 24 GB; not measured on the PC"},
    {"id": "fewer", "what": "drop the 2 lowest-ranked experts of every token on the CPU-held layers", "switch": "--fast=fewer", "status": "unmeasured",
     "reason": "gate level fewer (KL 0.060) set, nothing measured"},
    {"id": "approx-tiers", "what": "the named groups of the modes above", "switch": "--approx safe|careful|aggressive", "status": "unmeasured", "group": True,
     "reason": "a tier is only a group of the rows above; it is as good as its rows"},
]
DEPRECATED_SWITCHES = ("--fast", "--fast=2", "--fast=fewer", "--approx")        # still work as aliases that force the mode on; the table decides the default


def token_mode(mode_id):
    return next(m for m in TOKEN_MODES if m["id"] == mode_id)


def default_on(mode_id):
    """True when the default launch enables this mode: its status in TOKEN_MODES is "won"."""
    return token_mode(mode_id)["status"] == "won"


def default_ids():
    return [m["id"] for m in TOKEN_MODES if m["status"] == "won" and not m.get("group")]


def default_expert_level():
    """The --fast level the default launch uses: the strongest expert-skipping row whose status is "won" (fewer, then fast2, then fast1), else None. All three are "unmeasured" today."""
    for mid, lvl in (("fewer", "fewer"), ("fast2", "2"), ("fast1", "1")):
        if default_on(mid):
            return lvl
    return None


class ApproxError(Exception):
    """A refusal with a plain-words message (the launcher turns it into SystemExit)."""


def gate_records(root=None):
    """profiles/gates/*.json: {"match": "model-name-substring", "modes": {"kv8": {"pass": true, "kl": 0.0051, "kit": "...", "date": "..."}}}; written by the gate run.
    A record with pass:false refuses the mode for matching models; a record with pass:true verifies it. No folder = no records."""
    d = os.path.join(root or HERE, "profiles", "gates")
    out = []
    try:
        names = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    except OSError:
        return out
    for f in names:
        try:
            r = json.load(open(os.path.join(d, f), encoding="utf-8"))
            if isinstance(r, dict) and isinstance(r.get("modes"), dict) and r.get("match"):
                out.append(r)
        except (OSError, ValueError):
            pass
    return out


def mode_status(mode_id, model_name, records=()):
    """-> "verified" | "unverified" | "failed" for this mode and this model file name (lower-cased substring match)."""
    n = (model_name or "").lower()
    for r in records:
        if str(r["match"]).lower() in n and mode_id in r["modes"]:
            return "verified" if r["modes"][mode_id].get("pass") else "failed"
    pats = MODES[mode_id]["verified"]
    return "verified" if "*" in pats or any(p in n for p in pats) else "unverified"


def resolve(tier, model_name, off=(), unverified=False, user_fast=None, records=None):
    """The settings a tier turns on. -> dict(tier, fast (None|"1"|"2"|"kv"), kv (bool: 8-bit KV even without --fast), spec (bool: ask for MTP where the model has the head),
    modes (list of dicts id/status/switch/plain), lines (what to say), gate (level name)).
    off: mode ids removed from the tier (--approx-off). unverified: allow modes without a gate record for this model (--approx-unverified).
    user_fast: the level the user gave with --fast; it wins over the tier's expert mode and is reported. Raises ApproxError on a refusal."""
    if tier == "exact":
        return dict(tier="exact", fast=None, kv=False, spec=False, modes=[], lines=["moe-cache: --approx exact: nothing approximate is switched on (the plugin only)"], gate="exact")
    if tier not in TIERS:
        raise ApproxError(f"--approx {tier}: unknown tier (exact, safe, careful, aggressive)")
    recs = gate_records() if records is None else records
    t = TIERS[tier]
    for o in off:
        if o not in MODES:
            raise ApproxError(f"--approx-off {o}: unknown mode (known: {', '.join(sorted(MODES))})")
    chosen = [m for m in t["modes"] if m not in off]
    lines, modes, bad = [], [], []
    for m in chosen:
        st = mode_status(m, model_name, recs)
        modes.append(dict(id=m, status=st, switch=MODES[m]["switch"], plain=MODES[m]["plain"], gate=MODES[m]["gate"], measured=MODES[m]["measured"]))
        if st == "failed":
            bad.append(f"{m} (a gate record for this model says it FAILED)")
        elif st == "unverified" and not unverified:
            bad.append(f"{m} (no gate record for this model)")
    if bad:
        raise ApproxError(f"--approx {tier}: refused: " + "; ".join(bad) + ". Modes need a gate record for the model (docs/QUALITY_GATE.md). Drop one with --approx-off MODE, "
                          "or --approx-unverified to run it anyway (the control page shows a red badge)")
    ids = [m["id"] for m in modes]
    fast = ("1" if "fast1" in ids else "2" if "fast2" in ids else None)
    kv = "kv8" in ids
    if fast and user_fast and user_fast != fast:
        lines.append(f"moe-cache: --approx {tier}: your --fast={user_fast} wins over the tier's expert mode (--fast={fast}); the tier's gate no longer describes this run")
        fast = None
    lines.append(f"moe-cache: --approx {tier}: {t['plain']}. Gate: {t['gate']} level (docs/QUALITY_GATE.md). Modes: " + "; ".join(f"{m['id']} [{m['status']}]" for m in modes))
    for m in modes:
        if m["status"] == "unverified":
            lines.append(f"moe-cache: --approx-unverified: {m['id']} has NO gate record for this model: it runs, nobody measured its quality here")
    return dict(tier=tier, fast=fast, kv=kv, spec="spec-exact" in ids, modes=modes, lines=lines, gate=t["gate"])


def lower_tier(tier):
    i = TIER_ORDER.index(tier)
    return TIER_ORDER[max(i - 1, 0)]


def passes(level, kld, top1):
    """Does a measured (mean KLD, same-top %) meet this gate level? A value that could not be read never passes."""
    kmax, tmin, _ = GATE_LEVELS[level]
    return kld is not None and top1 is not None and kld <= kmax and top1 >= tmin


def canary_decide(tier, measure):
    """Step the tier down until its canary passes. measure(tier) -> (mean KLD, same-top %) against the exact setting, or None when it could not be run.
    -> (final tier, history [(tier, kld, top1, ok)], note). A measurement that cannot be taken keeps the tier and says so (the canary guards, it never blocks on its own failure
    to run). A tier whose measurement fails drops to the next lower one; "exact" needs no measurement."""
    hist = []
    while tier != "exact":
        got = measure(tier)
        if got is None:
            return tier, hist, "canary could not run: the tier stays as asked, nothing was checked"
        kld, top1 = got
        ok = passes(TIERS[tier]["gate"], kld, top1)
        hist.append((tier, kld, top1, ok))
        if ok:
            return tier, hist, None
        tier = lower_tier(tier)
    return "exact", hist, "every tier failed its canary: running exact"


def parse_ppl(text):
    """(mean KLD, same-top %) from llama-perplexity --kl-divergence output; None for a value that is missing."""
    def num(pat):
        m = re.search(pat + r"\s*:\s*(-?[0-9.]+)", text)
        return float(m.group(1)) if m else None
    return num(r"Mean +KLD"), num(r"Same top p")


# --- running the canary (needs llama-perplexity and the model; the launcher calls it, tests use canary_decide with a fake measure) ---------------------------------

CANARY_TEXT = os.path.join(HERE, "tools", "approx_canary.txt")
CANARY_CTX = 256        # two chunks of 256 tokens; llama-perplexity scores the second half of each: about 256 scored tokens ("about 300 tokens")
KEEP = ("-ngl", "--n-gpu-layers", "--n-cpu-moe", "-ot", "--override-tensor", "--device", "-dev", "-sm", "--split-mode", "-t", "--threads", "--no-mmap", "--mlock", "-mg", "--main-gpu", "--no-repack")
FLAGS = ("--no-mmap", "--mlock", "--no-repack")           # keep flags without a value
ROUTE_ENV = ("MOE_CACHE_ROUTE", "MOE_CACHE_ROUTE_DROP", "MOE_CACHE_FEWER")


def model_args_from_cmd(cmd):
    """The placement flags of a llama-server command line (layers on the GPU, experts on the CPU, tensor overrides, devices, threads) as a list for llama-perplexity."""
    out, i = [], 0
    while i < len(cmd):
        t = cmd[i]
        if t in KEEP:
            if t in FLAGS:
                out.append(t)
            elif i + 1 < len(cmd):
                out += [t, cmd[i + 1]]; i += 1
        i += 1
    return out


def kv_args_from_cmd(cmd):
    out = []
    for i, t in enumerate(cmd[:-1]):
        if t in ("-ctk", "-ctv", "--cache-type-k", "--cache-type-v"):
            out += [t, cmd[i + 1]]
    return out


def canary_key(model, lib, tier):
    """The stamp a passed or failed canary is stored under: model file (path, size, mtime), plugin library (mtime), and the tier. A new model file, a new kit or a new tier measures again."""
    def sig(p):
        try:
            s = os.stat(p); return f"{os.path.abspath(p)}:{s.st_size}:{int(s.st_mtime)}"
        except OSError:
            return f"{p}:?"
    return hashlib.sha256(f"{sig(model)}|{sig(lib) if lib else ''}|{tier}".encode()).hexdigest()[:20]


def stamp_read(dirpath, key):
    try:
        return json.load(open(os.path.join(dirpath, f"canary_{key}.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stamp_write(dirpath, key, rec):
    try:
        os.makedirs(dirpath, exist_ok=True)
        json.dump(rec, open(os.path.join(dirpath, f"canary_{key}.json"), "w", encoding="utf-8"))
    except OSError:
        pass


def run_ppl(perplexity, model, text, base_file, test, env, args, timeout=3600):
    """One llama-perplexity run on the canary text: test=False writes the exact base logits, test=True compares with them. -> output text or None on failure."""
    cmd = [perplexity, "-m", model, "-f", text, "-c", str(CANARY_CTX), "--chunks", "2", "-b", "8", "-ub", "8", "-fa", "on", "--kl-divergence-base", base_file] + args + (["--kl-divergence"] if test else [])
    try:
        p = subprocess.run(cmd, env=dict(os.environ, **env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


def make_measure(perplexity, model, text, place_args, kv_args_by_tier, env_by_tier, base_env, workdir):
    """-> measure(tier) for canary_decide, using llama-perplexity: the base logits are written once with the exact setting (no route env, f16 KV), then each tier's own
    environment and KV arguments are compared with them. None when a run fails."""
    base = os.path.join(workdir, "canary_base.kld")
    state = {"base": False}

    def measure(tier):
        if not state["base"]:
            if run_ppl(perplexity, model, text, base, False, dict(base_env), place_args) is None:
                return None
            state["base"] = True
        env = dict(base_env); env.update(env_by_tier(tier))
        out = run_ppl(perplexity, model, text, base, True, env, place_args + kv_args_by_tier(tier))
        if out is None:
            return None
        kld, top1 = parse_ppl(out)
        return None if kld is None or top1 is None else (kld, top1)
    return measure


def tier_env(tier):
    """Plugin environment of a tier (the same settings quality_gate.py uses for the matching level)."""
    return {"safe": {}, "careful": {"MOE_CACHE_ROUTE": "skip"}, "aggressive": {"MOE_CACHE_ROUTE": "skip", "MOE_CACHE_ROUTE_DROP": "4"}}.get(tier, {})


def tier_kv(tier):
    return ["-ctk", "q8_0", "-ctv", "q8_0"] if tier in ("safe", "careful", "aggressive") else []
