#!/usr/bin/env python3
"""make_starter_profile.py: write starter usage profiles for a model family (idea I18, ledger CACHE-146).

  make_starter_profile.py MODEL.gguf PROFILE.usage [OUT_DIR]
      write the single given XUSE profile to OUT_DIR (default profiles/) as <arch>-<layers>x<experts>.usage, without the
      model fingerprint, so every quant of that model can warm-start from it.

  make_starter_profile.py --agent MODEL.gguf PROFILE1.usage [PROFILE2.usage ...] [--out-dir DIR]
      merge several recorded sessions into one starter: every profile must be XUSE with the model's n_layers x n_experts,
      each is scaled to the same total weight (its counts / its sum, so one long session does not dominate), the scaled
      profiles are summed and rescaled so the merged sum equals the mean of the non-zero input sums (rounded to int32, clamped >= 0),
      then written as DIR/<key>.agent.usage (no fingerprint, same header). This is what a PC run of per-agent replays feeds in.

  make_starter_profile.py --export-prompts OUT.json SESSION.json [SESSION2.json ...]
      collect the replay list for such a PC run from server recordings (keys: requests[] with method/path/body, e.g.
      tests/agent_sessions/opencode_stats.json): OUT.json = {"format": "moe-cache starter prompts 1", "requests":
      [body of every POST /v1/chat/completions request, in order, with "stream" forced false]}. Nothing is run here.
"""
import json, os, struct, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "planner"))
from gguf_plan import model_key


def read_usage(path):
    """One XUSE profile -> (n_layers, n_experts, counts). SystemExit with a clear message when it is not one."""
    data = open(path, "rb").read()
    if data[:4] != b"XUSE":
        raise SystemExit(f"{path}: not an XUSE usage profile")
    n_l, n_e = struct.unpack("<ii", data[4:12])
    body = data[12:12 + 4 * n_l * n_e]
    if len(body) != 4 * n_l * n_e:
        raise SystemExit(f"{path}: profile is truncated")
    return n_l, n_e, list(struct.unpack(f"<{n_l * n_e}i", body))


def merge_usage(profiles):
    """CACHE-146: merge count lists of equal length into one. Each profile is scaled to the same total weight
    (counts / its own sum, a zero-sum profile contributes zeros and does not divide by zero), the scaled profiles
    are summed, then rescaled so the merged sum equals the mean of the non-zero input sums (rounded, clamped >= 0). Pure: no I/O."""
    n = len(profiles)
    sums = [sum(p) for p in profiles]
    scaled = [[c / s for c in p] if s else [0.0] * len(p) for p, s in zip(profiles, sums)]
    acc = [sum(col) for col in zip(*scaled)] if n else []
    live = [x for x in sums if x]                 # an empty session carries no information: it does not pull the merged weight down
    target = sum(live) / len(live) if live else 0.0
    norm = sum(acc)
    scale = target / norm if norm else 0.0
    return [max(0, min(2**31 - 1, int(round(v * scale)))) for v in acc]


def export_prompts(out_path, sessions):
    """The replay list a PC run feeds to the server to produce the per-agent usage profiles: the body of every
    POST /v1/chat/completions request, in order, with "stream" forced false. Nothing is run here."""
    reqs = []
    for path in sessions:
        data = json.load(open(path))
        for r in data.get("requests", []):
            if r.get("method") == "POST" and r.get("path") == "/v1/chat/completions":
                body = dict(r.get("body") or {})
                body["stream"] = False
                reqs.append(body)
    with open(out_path, "w") as f:
        json.dump({"format": "moe-cache starter prompts 1", "requests": reqs}, f)
    print(f"{out_path}: {len(reqs)} chat requests from {len(sessions)} session(s)")


def run_agent(model, profiles, out_dir):
    key = model_key(model)
    if not key:
        raise SystemExit("no architecture / block_count / expert_count in the GGUF header")
    want_e = int(key.rsplit("x", 1)[1])
    want_l = int(key.rsplit("-", 1)[1].split("x")[0])
    usages = [read_usage(p) for p in profiles]
    for p, (n_l, n_e, _) in zip(profiles, usages):
        if (n_l, n_e) != (want_l, want_e):
            raise SystemExit(f"{p}: profile is {n_l}x{n_e} but {key} is {want_l}x{want_e}")
    merged = merge_usage([u[2] for u in usages])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, key + ".agent.usage")
    with open(out, "wb") as f:
        f.write(b"XUSE" + struct.pack("<ii", want_l, want_e) + struct.pack(f"<{len(merged)}i", *merged))
        f.truncate(12 + 4 * want_l * want_e)         # no trailing fingerprint: accepted for every quant of the family
    print(f"{out}: {want_l} layers x {want_e} experts from {len(profiles)} session(s)")


def main(argv):
    args = list(argv[1:])
    def take(flag):
        if flag in args:
            i = args.index(flag)
            val = args[i + 1]
            del args[i:i + 2]
            return val
        return None
    out_dir = take("--out-dir") or os.path.join(HERE, "profiles")
    if "--export-prompts" in args:
        i = args.index("--export-prompts")
        out, sessions = args[i + 1], args[i + 2:]
        if not sessions:
            raise SystemExit("--export-prompts needs at least one session recording")
        export_prompts(out, sessions)
    elif "--agent" in args:
        i = args.index("--agent")
        model, profiles = args[i + 1], args[i + 2:]
        if not profiles:
            raise SystemExit("--agent needs the model GGUF and at least one PROFILE.usage")
        run_agent(model, profiles, out_dir)
    else:
        model, prof = args[0], args[1]
        out_dir = args[2] if len(args) > 2 else out_dir
        key = model_key(model)
        if not key:
            raise SystemExit("no architecture / block_count / expert_count in the GGUF header")
        n_l, n_e, _ = read_usage(prof)
        if n_e != int(key.rsplit("x", 1)[1]):
            raise SystemExit(f"profile has {n_e} experts per layer but {key} has {key.rsplit('x', 1)[1]}")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, model_key(model) + ".usage")
        with open(out, "wb") as f:
            data = open(prof, "rb").read()
            f.write(b"XUSE" + data[4:12 + 4 * n_l * n_e])       # no trailing fingerprint: accepted for every quant of the family
        print(f"{out}: {n_l} layers x {n_e} experts")


if __name__ == "__main__":
    main(sys.argv)
