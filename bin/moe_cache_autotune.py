#!/usr/bin/env python3
"""LAUNCHER-994: measured auto-tune. On the first start for a model + machine + context the launcher tries a few candidate plans (at most BUDGET_S seconds in
total), measures prompt and decode speed on each, keeps the fastest and stores it; later starts read the stored pick (`--retune` measures again).

This file is the logic only: the candidates, the pick, the cache file and the text. Starting llama-server and sending the probe is the `runner` the caller
passes in (bin/moe_cache.py: autotune_stage), so tests drive it with fake timings.

A candidate is the plan as it is plus ONE or TWO edits ("dims"): ub (-ub), ncmoe (--n-cpu-moe, stored as a delta so a changed GPU memory later cannot
carry a stale absolute value over), kv (-ctk/-ctv), ramlock (MOE_CACHE_RAM_LOCK + MOE_CACHE_DROP_READAHEAD), np (-np 1). A dimension the user set (flag or
environment) is never tuned; --exact and --stock skip the tuner."""
import hashlib
import json
import os
import time

VERSION = 2                 # bump when the meaning of a stored pick changes: old files are then ignored
BUDGET_S = 180.0            # the owner's limit for the whole first start tuning, loads included
MARGIN = 0.05               # a candidate must beat the plan as it is by this much (run-to-run noise on one probe is about this big)
PROMPT_TOKENS = 1500        # the probe prompt (a prompt speed at this length ranks plans; it is not a 64k number)
DECODE_TOKENS = 512         # owner rule: decode probes are >= 512 tokens or say they are short
SHORT_MIN = 64              # below this a decode figure says nothing
SCORE_PROMPT = 4000         # score = seconds for this many prompt tokens + DECODE_TOKENS generated tokens ...
SCORE_PROMPT_AGENT = 16000  # ... (agents send long prompts)
RAMLOCK_ENV = (("MOE_CACHE_RAM_LOCK", "1"), ("MOE_CACHE_DROP_READAHEAD", "1"))


def flag_value(cmd, *names):
    v = None
    for i, t in enumerate(cmd):
        if t in names and i + 1 < len(cmd):
            v = cmd[i + 1]
    return v


def set_flag(cmd, name, value, also=()):
    """cmd with `name value` replaced (all copies) or appended before a trailing '--'-free end; `also` = other names that mean the same flag (long form)."""
    names = (name,) + tuple(also)
    out, i, done = [], 0, False
    while i < len(cmd):
        if cmd[i] in names and i + 1 < len(cmd):
            if not done:
                out += [cmd[i], str(value)]
                done = True
            i += 2
            continue
        out.append(cmd[i])
        i += 1
    if not done:
        out += [name, str(value)]
    return out


def set_env(env, key, value):
    return [(k, v) for k, v in env if k != key] + [(key, value)]


class Candidate:
    def __init__(self, cid, label, edits):
        self.id, self.label, self.edits = cid, label, edits       # edits: [{"dim", "flag"|"env", "value"|"delta"}]

    @property
    def dims(self):
        return sorted({e["dim"] for e in self.edits})

    def to_dict(self):
        return {"id": self.id, "label": self.label, "edits": self.edits}

    @staticmethod
    def from_dict(d):
        return Candidate(d["id"], d["label"], d["edits"])


def apply_edits(cmd, env, edits, locked=()):
    """(cmd, env, skipped dims): the edits applied to a plan, minus the dimensions in `locked` (the user's flags win)."""
    cmd, env, skipped = list(cmd), list(env), []
    for e in edits:
        if e["dim"] in locked:
            skipped.append(e["dim"])
            continue
        if "env" in e:
            env = set_env(env, e["env"], e["value"])
        elif e["flag"] == "--n-cpu-moe":
            cur = flag_value(cmd, "--n-cpu-moe", "-ncmoe")
            if cur is not None and str(cur).isdigit():
                cmd = set_flag(cmd, "--n-cpu-moe", max(0, int(cur) + int(e["delta"])), also=("-ncmoe",))
        else:
            cmd = set_flag(cmd, e["flag"], e["value"], also=tuple(e.get("also", ())))
    return cmd, env, sorted(set(skipped))


def locked_dims(a, argv, environ, windows=False, agent=False, project=False, engine="plugin"):
    """The dimensions the tuner must leave alone: what the user gave on the command line (before or after `--`) or in the environment, plus what does not apply here.
    Returns {dim: reason}."""
    import re
    user = " ".join(argv)
    has = lambda *names: any(re.search(r"(^| )" + re.escape(n) + r"( |=|$)", user) for n in names)
    out = {}
    if getattr(a, "ub", "auto") != "auto" or has("-ub", "--ubatch-size", "-b", "--batch-size"):
        out["ub"] = "your -ub / -b"
    if getattr(a, "exact", False) or has("-ctk", "--cache-type-k", "-ctv", "--cache-type-v"):
        out["kv"] = "--exact" if getattr(a, "exact", False) else "your -ctk / -ctv"
    if (has("-ot", "--override-tensor", "-ngl", "--n-gpu-layers", "--gpu-layers", "--n-cpu-moe", "-ncmoe", "--cpu-moe") or getattr(a, "n_cpu_moe", None) is not None
            or getattr(a, "hybrid", False)):
        out["ncmoe"] = "your --n-cpu-moe / -ngl / -ot"
    if has("-np", "--parallel") or agent or project:
        out["np"] = "your -np" if has("-np", "--parallel") else "agent / project mode keeps its slots"
    if "MOE_CACHE_RAM_LOCK" in environ or "MOE_CACHE_DROP_READAHEAD" in environ:
        out["ramlock"] = "your MOE_CACHE_RAM_LOCK / DROP_READAHEAD"
    elif windows or engine != "plugin":
        out["ramlock"] = "Linux plugin engine only"
    return out


def build_candidates(cmd, env, locked, ctx, n_layers=None):
    """The plan as it is (C0) plus up to 3 candidates built from the unlocked dimensions, best guesses first. Every candidate differs from C0 in one dimension
    (the first two also in a second one only when that is needed to make room: a bigger -ub with more experts on the GPU is not tried together)."""
    c0 = Candidate("C0", "the plan as it is", [])
    cands = []
    ncmoe = flag_value(cmd, "--n-cpu-moe", "-ncmoe")
    ub = flag_value(cmd, "-ub", "--ubatch-size")
    byid = {}
    if "ncmoe" not in locked and ncmoe is not None and str(ncmoe).isdigit():
        n = int(ncmoe)
        if n >= 2:
            byid["ncmoe-2"] = Candidate("ncmoe-2", "2 more expert layers on the GPU", [{"dim": "ncmoe", "flag": "--n-cpu-moe", "delta": -2}])
        if n_layers is None or n + 2 <= n_layers:
            byid["ncmoe+2"] = Candidate("ncmoe+2", "2 more expert layers on the CPU (less GPU memory in use)", [{"dim": "ncmoe", "flag": "--n-cpu-moe", "delta": 2}])
    # LAUNCHER-1043: the probe reads a PROMPT_TOKENS-token prompt, so a micro-batch above that size does the same work as one at that size, and a KV type is judged on
    # a short context. The planner's long-prompt defaults (-ub 4096, q8_0 KV: measured at 65k) can therefore neither be improved nor be undone by this probe: no KV
    # candidate, and -ub only upward from a small plan (512 -> 1024), never above the prompt length.
    if "ub" not in locked and ub is not None and str(ub).isdigit():
        u = int(ub)
        alt = u * 2
        if alt <= PROMPT_TOKENS:
            byid[f"ub{alt}"] = Candidate(f"ub{alt}", f"-ub {alt} (plan: {u})", [{"dim": "ub", "flag": "-ub", "value": str(alt), "also": ["--ubatch-size"]}])
    if "ramlock" not in locked:
        byid["ramlock"] = Candidate("ramlock", "RAM_LOCK + DROP_READAHEAD", [{"dim": "ramlock", "env": k, "value": v} for k, v in RAMLOCK_ENV])
    if "np" not in locked:
        byid["np1"] = Candidate("np1", "-np 1 (one slot)", [{"dim": "np", "flag": "-np", "value": "1", "also": ["--parallel"]}])
    # priority: more GPU layers, the batch size, the KV type, then the rest; ncmoe+2 only stands in when a slot is free
    order = ["ncmoe-2"] + [k for k in byid if k.startswith("ub")] + ["ncmoe+2", "ramlock", "np1"]
    out = [byid[k] for k in order if k in byid]
    return [c0] + out[:3]


def key_parts(model_id, ctx, machine_sig, locked, agent, engine, limits=None):
    """limits: the memory the user capped (--ram, --vram, --max-ram, --gpu-margin, --leave-room): a pick measured with more memory is not replayed with less."""
    return {"v": VERSION, "model": model_id, "ctx": int(ctx), "machine": machine_sig, "locked": sorted(locked), "agent": bool(agent), "engine": engine, "limits": limits or {}}


def key_of(parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def cache_path(cache_dir, parts):
    return os.path.join(cache_dir, "autotune", key_of(parts) + ".json")


def load(cache_dir, parts):
    try:
        with open(cache_path(cache_dir, parts), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) and d.get("version") == VERSION and d.get("key") == parts else None
    except (OSError, ValueError):
        return None


def save(cache_dir, parts, result):
    p = cache_path(cache_dir, parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(result, version=VERSION, key=parts), f, indent=1)
    os.replace(tmp, p)
    return p


def mark_bad(cache_dir, parts, why):
    """LAUNCHER-1045: the stored pick did not load: keep the file (so the next start knows and says why) but never apply its pick again; --retune measures anew."""
    d = load(cache_dir, parts)
    if not d:
        return False
    d["bad"] = {"at": time.strftime("%Y-%m-%d %H:%M"), "why": str(why)[:300]}
    save(cache_dir, parts, {k: v for k, v in d.items() if k not in ("version", "key")})
    return True


LOCK_STALE_S = 600.0        # the budget is 180 s plus the loads around it; a lock older than this is a launcher that died


def _pid_alive(pid):
    if os.name == "nt":
        return True         # no cheap check without extra modules: only the age counts there
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, ValueError):
        return True


def lock_path(cache_dir):
    return os.path.join(cache_dir, "autotune", "tuning.lock")


def acquire_lock(cache_dir, now=time.time, alive=_pid_alive):
    """LAUNCHER-1041: one tuning at a time on this computer (two loads at once double the memory and spoil each other's numbers). -> (True, None) or (False, pid of the
    launcher that holds it). A lock of a dead process or older than LOCK_STALE_S is taken over."""
    p = lock_path(cache_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                with open(p, encoding="utf-8") as f:
                    held = json.load(f)
            except (OSError, ValueError):
                held = {}
            if held and now() - float(held.get("t", 0)) < LOCK_STALE_S and alive(held.get("pid", 0)):
                return False, held.get("pid")
            try:
                os.unlink(p)            # stale: take it over (the loop tries once more)
            except OSError:
                return False, held.get("pid")
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "t": now()}, f)
        return True, None
    return False, None


def release_lock(cache_dir):
    try:
        with open(lock_path(cache_dir), encoding="utf-8") as f:
            if int(json.load(f).get("pid", -1)) != os.getpid():
                return
        os.unlink(lock_path(cache_dir))
    except (OSError, ValueError):
        pass


def score(row, agent=False):
    """Seconds to read a prompt and write DECODE_TOKENS tokens at this row's speeds (lower is better); None without both speeds."""
    pp, dec = row.get("prompt_tps"), row.get("decode_tps")
    if not pp or not dec or pp <= 0 or dec <= 0:
        return None
    return (SCORE_PROMPT_AGENT if agent else SCORE_PROMPT) / pp + DECODE_TOKENS / dec


def aborted_result(rows, cid, why, total, budget_s, agent):
    return {"rows": rows, "total_s": round(total, 1), "budget_s": budget_s, "agent": agent, "prompt_tokens": PROMPT_TOKENS, "decode_tokens": DECODE_TOKENS,
            "tuned_at": time.strftime("%Y-%m-%d %H:%M"), "picked": "C0", "changed": False, "state": "aborted",
            "why": f"the memory guard stopped the probe of {cid} ({str(why)[:160]}): the tuning was aborted, nothing was stored, the plan as it is runs. It is tried again at the next start"}


def tune(cands, runner, agent=False, budget_s=BUDGET_S, clock=time.monotonic, say=lambda t: None, progress=lambda t: None):
    """Measure the candidates in order and pick. runner(cand, left_s, decode_tokens) -> {"ok", "load_s", "prompt_tps", "decode_tps", "decode_n", "error"}.
    C0 (the plan as it is) always runs first with whatever time there is; each later candidate starts only when the time left covers a load and probe like
    C0's. When C0 cannot be measured the result is `too_slow` and nothing is picked. Returns the result dict that is stored."""
    t0 = clock()
    rows, base_cost = [], None
    aborted = lambda rows_, cid, why: aborted_result(rows_, cid, why, clock() - t0, budget_s, agent)
    for c in cands:
        left = budget_s - (clock() - t0)
        row = {"id": c.id, "label": c.label, "status": "not measured", "edits": c.edits}
        n = DECODE_TOKENS
        if c.id != "C0":
            b0 = rows[0]
            fixed = (base_cost - DECODE_TOKENS / b0["decode_tps"]) * 1.15      # load + prompt part of C0's probe, with a margin
            room = left - fixed
            if room < SHORT_MIN / b0["decode_tps"]:
                row["error"] = f"no time left in the {budget_s:.0f} s budget ({left:.0f} s left, load and prompt alone take about {fixed:.0f} s)"
                rows.append(row)
                continue
            n = min(DECODE_TOKENS, int(b0["decode_tps"] * room))
        progress(f"trying plan {len(rows) + 1} of {len(cands)}: {c.id} ({c.label})")      # LAUNCHER-1041: the control page lists this while the server is still "starting"
        s = clock()
        try:
            r = runner(c, left, n) or {}
        except Exception as e:                      # a runner that raises is a failed candidate, never a failed start
            r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        took = clock() - s
        if r.get("guard_stop"):             # LAUNCHER-1042: the memory guard stopped a probe: no more loads under memory pressure, nothing is stored, the plan as it is runs
            row.update(status="stopped", error=str(r["guard_stop"])[:200])
            rows.append(row)
            say(row_line(row))
            return aborted(rows, c.id, r["guard_stop"])
        row.update({k: r.get(k) for k in ("load_s", "prompt_tps", "decode_tps", "decode_n", "error")})
        row["took_s"] = round(took, 1)
        if r.get("ok") and r.get("prompt_tps") and r.get("decode_tps"):
            row["status"] = "measured"
            row["short_probe"] = bool((r.get("decode_n") or n) < DECODE_TOKENS)
            row["score_s"] = round(score(row, agent), 2)
            if c.id == "C0":
                base_cost = took
        else:
            row["status"] = "failed"
        rows.append(row)
        say(row_line(row))
        if c.id == "C0" and row["status"] != "measured":
            break
    # the plan as it is, a second time (RULES 4: a screen needs A and A2): C0 ran first, cold, so a later plan can look faster only because the files and clocks were warm
    if rows and rows[0]["status"] == "measured" and base_cost:
        left = budget_s - (clock() - t0)
        if left >= base_cost * 1.15:
            c0b = {"id": "C0b", "label": "the plan as it is, again (noise check)", "status": "not measured", "edits": []}
            progress("trying the plan as it is a second time (noise check)")
            try:
                r = runner(Candidate("C0", c0b["label"], []), left, DECODE_TOKENS) or {}
            except Exception as e:
                r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            if r.get("guard_stop"):         # the same rule as in the loop: a guard stop anywhere aborts the tune and stores nothing
                c0b.update(status="stopped", error=str(r["guard_stop"])[:200])
                rows.append(c0b)
                say(row_line(c0b))
                return aborted(rows, "C0b", r["guard_stop"])
            c0b.update({k: r.get(k) for k in ("load_s", "prompt_tps", "decode_tps", "decode_n", "error")})
            if r.get("ok") and r.get("prompt_tps") and r.get("decode_tps"):
                c0b.update(status="measured", short_probe=False, score_s=round(score(c0b, agent), 2))
            else:
                c0b["status"] = "failed"
            rows.append(c0b)
            say(row_line(c0b))
    total = clock() - t0
    base = rows[0] if rows else None
    res = {"rows": rows, "total_s": round(total, 1), "budget_s": budget_s, "agent": agent, "prompt_tokens": PROMPT_TOKENS, "decode_tokens": DECODE_TOKENS,
           "tuned_at": time.strftime("%Y-%m-%d %H:%M"), "picked": "C0", "changed": False}
    if not base or base["status"] != "measured":
        res.update(state="too_slow", why=f"the plan as it is could not be measured in {budget_s:.0f} s ({(base or {}).get('error') or 'no result'}); nothing was changed. "
                                         "A model that takes longer than the budget to load is not tuned automatically; --retune tries again")
        return res
    again = next((r for r in rows if r["id"] == "C0b" and r["status"] == "measured"), None)
    noise = ""
    if again:
        spread = max(abs(again["decode_tps"] - base["decode_tps"]) / max(again["decode_tps"], base["decode_tps"]), abs(again["prompt_tps"] - base["prompt_tps"]) / max(again["prompt_tps"], base["prompt_tps"]))
        if again["score_s"] < base["score_s"]:
            base = again                                    # the harder baseline: the better of its two runs
        if spread > MARGIN:
            noise = f" (the plan as it is moved {spread * 100:.0f} % between its two runs: a cold first run or noise, the better run is the baseline)"
        res["baseline_spread"] = round(spread, 3)
    best = min((r for r in rows if r["status"] == "measured" and r["id"] != "C0b"), key=lambda r: r["score_s"])
    gain = 1 - best["score_s"] / base["score_s"] if best["id"] != "C0" else 0.0
    need = max(MARGIN, 2 * res.get("baseline_spread", 0.0))      # LAUNCHER-1045 (margin part): a plan must beat the baseline by more than twice the baseline's own run-to-run spread
    if best["id"] != "C0" and gain >= need:
        res.update(picked=best["id"], changed=True, state="tuned",
                   why=f"{best['id']} ({best['label']}) was {gain * 100:.0f} % faster than the plan as it is for a {SCORE_PROMPT_AGENT if agent else SCORE_PROMPT}-token prompt plus {DECODE_TOKENS} generated tokens "
                       f"(prompt {best['prompt_tps']:.0f} vs {base['prompt_tps']:.0f} tok/s, decode {best['decode_tps']:.1f} vs {base['decode_tps']:.1f} tok/s)" + noise)
    else:
        near = f"; the best other was {best['id']} at {gain * 100:.0f} %" if best["id"] != "C0" else ""
        res.update(state="tuned", partial=any(r["status"] == "not measured" for r in rows), why=f"the plan as it is stays: no candidate was {need * 100:.0f} % faster{near}" + noise
                                     + ("" if sum(r["status"] == "measured" and r["id"] != "C0b" for r in rows) > 1 else " (only the plan itself could be measured in the time; --retune tries again, e.g. with the model already in the page cache)"))
    return res


def row_line(r):
    if r["status"] == "measured":
        sp = f" (short probe, {r.get('decode_n')} tokens)" if r.get("short_probe") else ""
        return f"{r['id']:<9} {r['label']}: prompt {r['prompt_tps']:.0f} tok/s, decode {r['decode_tps']:.1f} tok/s{sp}, load {r.get('load_s') or 0:.0f} s"
    return f"{r['id']:<9} {r['label']}: {r['status']}" + (f" ({str(r['error'])[:120]})" if r.get("error") else "")


PICK_MAX_AGE_DAYS = 30      # a stored pick was measured on one day's drivers, llama.cpp build and desktop load: after 30 days it is measured again (LAUNCHER-994 review)


def pick_age_days(result, now=None):
    """Days since the stored result was tuned ("tuned_at" is local time, "%Y-%m-%d %H:%M"); None when it has no readable date."""
    import time as _t
    try:
        t = _t.mktime(_t.strptime(result.get("tuned_at") or "", "%Y-%m-%d %H:%M"))
    except (ValueError, OverflowError, TypeError):
        return None
    return ((now if now is not None else _t.time()) - t) / 86400.0


def edits_fit_free_vram(stored, free_now, slack_mib=512):
    """The stored pick may add GPU use only when the free GPU memory now is at least what it was when the pick was measured (minus slack_mib); False when either figure is missing."""
    then = stored.get("free_vram_mib")
    if not then or not free_now:
        return False
    return free_now / 2**20 >= then - slack_mib


def adds_gpu_use(cmd, edits):
    """True when the pick puts more on the GPU than the plan it is applied to: expert layers moved from the CPU (--n-cpu-moe lower) or a bigger micro-batch (-ub: a bigger
    compute buffer). Such a pick was measured with the free GPU memory of that day."""
    for e in edits or []:
        if e.get("flag") == "--n-cpu-moe" and int(e.get("delta", 0) or 0) < 0:
            return True
        if e.get("flag") in ("-ub", "--ubatch-size"):
            try:
                if int(e.get("value")) > int(flag_value(cmd, "-ub", "--ubatch-size") or 0):
                    return True
            except (TypeError, ValueError):
                pass
    return False


def replay_edits(cmd, stored, free_now, slack_mib=512):
    """(edits to apply at this start, note or None). A stored pick is applied at every later start. An edit that puts more on the GPU than the plan (--n-cpu-moe lower, a bigger
    -ub) was measured with the free GPU memory of that day: it is left out for this start unless the free memory now is at least that (minus slack_mib); the other edits of the pick
    still apply (partial replay: the idea is from the outside review of 7c3d024). A stored result without the figure, or no reading now, cannot be compared: held back too."""
    edits = pick_edits(stored)
    add = [e for e in edits if adds_gpu_use(cmd, [e])]
    if not add or edits_fit_free_vram(stored, free_now, slack_mib):
        return edits, None
    then = stored.get("free_vram_mib")
    what = ", ".join(e.get("flag", "?") + (f" {e['delta']:+d}" if "delta" in e else f" {e.get('value')}") for e in add)
    if then and free_now:
        why = f"{free_now / 2**30:.1f} GiB of GPU memory is free now against {then / 1024:.1f} GiB when it was measured"
    else:
        why = "the free GPU memory cannot be compared with the day it was measured (not recorded, or not readable now)"
    return [e for e in edits if e not in add], f"the stored pick's part that puts more on the GPU ({what}) is not applied at this start: {why} (--retune measures again)"


def pick_edits(result):
    """The stored pick's edits ([] for the plan as it is)."""
    for r in result.get("rows", []):
        if r["id"] == result.get("picked"):
            return r.get("edits") or []
    return []


def summary_lines(result, cached=False, skipped=(), locked=None):
    """The lines the launcher prints and the control page shows."""
    head = "moe-cache: auto-tune: " + ("cached result" + (f" from {result.get('tuned_at')}" if result.get("tuned_at") else "") + " for this model, machine and context (--retune measures again)"
                                      if cached else f"measured {len([r for r in result['rows'] if r['status'] == 'measured'])} of {len(result['rows'])} plans in {result['total_s']:.0f} s "
                                      f"(budget {result['budget_s']:.0f} s)")
    out = [head]
    probe = (f"moe-cache: auto-tune probe: a {result.get('prompt_tokens', PROMPT_TOKENS)}-token prompt and {result.get('decode_tokens', DECODE_TOKENS)} generated tokens per plan "
             "(it ranks the plans; it is not a 64k figure)")
    if not cached:
        out.append(probe)
    for r in result["rows"]:
        out.append("moe-cache: auto-tune:   " + row_line(r))
    out.append("moe-cache: auto-tune: " + ("picked " + result["picked"] + ": " if result.get("changed") else "") + result.get("why", ""))
    if locked:
        out.append("moe-cache: auto-tune: left alone (yours or not applicable): " + ", ".join(f"{d} ({w})" for d, w in sorted(locked.items())))
    if skipped:
        out.append("moe-cache: auto-tune: not applied (your flags win): " + ", ".join(skipped))
    return out
