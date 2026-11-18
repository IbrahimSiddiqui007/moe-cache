"""The never-list (LAUNCHER-1225 step 2): settings measured as a loss, read from never_list.json next to this file.

A row is (kind flag|env, name, value or "*" or a minimum, scope, loss, source). Scope keys (all optional, all must hold): "backend" (cuda / vulkan / rocm / sycl / metal / cpu,
from the server's device listing), "model" (dense / moe), "ctx_min" (context at or above). An unknown scope fact (backend None) does NOT match: a row only vetoes where its
evidence applies.

Two uses: veto(edits, facts) for the auto-tune's candidates (a listed candidate is never built), and user_notes(cmd, env, facts) for what the user set (a note, never a refusal)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "never_list.json")
_ROWS = None


def rows(path=None):
    """The rows of the list (cached for the default file); [] when the file is missing or unreadable (nothing is vetoed then)."""
    global _ROWS
    if path is None and _ROWS is not None:
        return _ROWS
    try:
        with open(path or PATH, encoding="utf-8") as f:
            r = [x for x in (json.load(f).get("rows") or []) if isinstance(x, dict) and x.get("id") and x.get("kind") in ("flag", "env") and x.get("name")]
    except (OSError, ValueError, AttributeError):
        r = []
    if path is None:
        _ROWS = r
    return r


def backend_of(listing):
    """cuda / vulkan / rocm / sycl / metal from the first device of llama-server's --list-devices text; None when unknown."""
    import re
    m = re.search(r"^\s+(CUDA|Vulkan|ROCm|HIP|SYCL|Metal|MTL)\d*:", listing or "", re.M)
    if not m:
        return None
    return {"CUDA": "cuda", "Vulkan": "vulkan", "ROCm": "rocm", "HIP": "rocm", "SYCL": "sycl", "Metal": "metal", "MTL": "metal"}[m.group(1)]


def in_scope(row, facts):
    sc = row.get("scope") or {}
    if "backend" in sc and facts.get("backend") != sc["backend"]:
        return False
    if "model" in sc and facts.get("model") != sc["model"]:
        return False
    if "ctx_min" in sc and int(facts.get("ctx") or 0) < int(sc["ctx_min"]):
        return False
    return True


def value_hits(row, value):
    if value is None:
        return False
    if "min" in row:
        try:
            return int(value) >= int(row["min"])
        except (TypeError, ValueError):
            return False
    want = row.get("value", "*")
    return want == "*" or str(value) == str(want)


FLAG_SPELLINGS = {"-ub": ("--ubatch-size",), "--spec-type": ()}


def hit(kind, name, value, facts, names=()):
    """The row that lists this setting in this scope, else None. names: other spellings of the flag (-ub / --ubatch-size)."""
    for r in rows():
        spell = {r["name"], *FLAG_SPELLINGS.get(r["name"], ())}
        if r["kind"] == kind and (name in spell or spell & set(names)) and value_hits(r, value) and in_scope(r, facts):
            return r
    return None


def veto(edits, facts):
    """The row that forbids a candidate (a list of auto-tune edits {"flag"|"env", "value"}), else None. Delta edits (--n-cpu-moe) are placements, never listed."""
    for e in edits or []:
        if "env" in e:
            r = hit("env", e["env"], e.get("value"), facts)
        elif "value" in e:
            r = hit("flag", e.get("flag"), e.get("value"), facts, tuple(e.get("also", ())))
        else:
            r = None
        if r:
            return r
    return None


def user_notes(cmd, env, facts, environ=None):
    """One line per listed setting found in the command or the environment (the plan's and the user's own), said once: what was measured; the setting stays."""
    out, seen = [], set()
    def flag_val(names):
        v = None
        for i, t in enumerate(cmd):
            if t in names and i + 1 < len(cmd):
                v = cmd[i + 1]
            elif any(t.startswith(n + "=") for n in names):
                v = t.split("=", 1)[1]
        return v
    envd = dict(environ if environ is not None else os.environ)
    envd.update(dict(env or []))
    for r in rows():
        if r["id"] in seen or not in_scope(r, facts):
            continue
        v = flag_val((r["name"],) + FLAG_SPELLINGS.get(r["name"], ())) if r["kind"] == "flag" else envd.get(r["name"])
        if value_hits(r, v):
            seen.add(r["id"])
            what = f"{r['name']} {v}" if r["kind"] == "flag" else f"{r['name']}={v}"
            out.append(f"moe-cache: note: {what} was measured as a loss ({r['loss']}; {r['source']}); the launcher never picks it on its own; yours is kept")
    return out


# ---- answer classes (LAUNCHER-1225 step 2) ----
INPUT_SETTINGS = ("free_gpu_gib", "ram_for_server_gib")      # plan-file rows that are measured inputs, not settings


def answer_class(name, path=None):
    """(class, gate) of an auto-tune dimension or plan-file setting: class exact / placement / quality / input; gate = the TOKEN_MODES row of a quality setting.
    A name missing from the table is ("quality", None): never picked on its own (fail closed). Plan-file inputs (gpu_* sizes, free memory) are ("input", None)."""
    if name in INPUT_SETTINGS or str(name).startswith(("gpu_", "walk_floor")):
        return "input", None
    try:
        with open(path or PATH, encoding="utf-8") as f:
            t = json.load(f).get("answer_classes") or {}
    except (OSError, ValueError, AttributeError):
        t = {}
    r = t.get(name)
    if not isinstance(r, dict) or r.get("class") not in ("exact", "placement", "quality"):
        return "quality", None
    return r["class"], r.get("gate")


def gate_passed(gate):
    """True when the quality-gate row `gate` of TOKEN_MODES (bin/moe_cache_approx.py) has status 'won'; False when unknown."""
    if not gate:
        return False
    try:
        import sys
        b = os.path.join(os.path.dirname(HERE), "bin")
        if b not in sys.path:
            sys.path.insert(0, b)
        import moe_cache_approx as ax
        return ax.default_on(gate)
    except Exception:
        return False


def tuner_may_pick(edits):
    """(True, None) when the auto-tune may try and pick a candidate with these edits on its own: every dimension is exact or placement, or quality with a passed gate.
    Else (False, "dim (why)")."""
    for e in edits or []:
        dim = e.get("dim")
        cls, gate = answer_class(dim)
        if cls == "quality" and not gate_passed(gate):
            return False, f"{dim} ({'its quality gate ' + gate + ' has not passed' if gate else 'no answer class: counts as changing answers'})"
    return True, None
