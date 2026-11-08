#!/usr/bin/env python3
"""GPU expert tier replay: how many decode picks a VRAM bank of B (layer, expert) slots serves, per policy. Kill check for the GPU tier (CACHE-049).
usage: gpu_tier_replay.py TRACE.jsonl [--fracs 0.0625,0.125,0.19] [--every 1,4,16] [--swaps 24,96] [--decay 0.7,0.97]
TRACE: routing .jsonl ({"req","phase","layer","experts"}), decode rows only; each request is one conversation start.
Same bank size B for every arm (B = frac x layers x experts). The first 10 % of each request's decode tokens are warm-up and not counted.
The fixed arms get a profile built from the OTHER requests (leave-one-request-out), like the shipped generic profile; adaptive arms start from that bank.
Arms:
  whole-layers   the layers with the most use get ALL their experts (what --n-cpu-moe does today with free VRAM)
  global-top     the B most used (layer, expert) pairs of the profile, global ranking (today's --hybrid bank)
  layer-top      equal slots per layer, each layer keeps its own most used experts (per-layer slot ranges, fixed)
  layer-prop     slots per layer proportional to the layer's profile concentration, most used experts inside
  adapt-glob     global-top + the plugin's hybrid_adapt rule (misses since last round, decay 0.97, swap inside the layer)
  adapt-layer    layer-top + decayed-count rule: every E tokens, up to S swaps, ALL counts x decay per round, a missing expert comes in only if its
                 count beats the coldest resident expert of ITS layer
  oracle-layer   best static equal-per-layer bank for this request (upper bound for fixed per-layer ranges)
Policy ideas (per-layer slot ranges, decayed counts, swap batches) are from published designs; no code copied."""
import argparse, collections, json, sys


def load(path):
    reqs = collections.OrderedDict()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["phase"] != "decode":
                continue
            reqs.setdefault(r["req"], []).append((r["layer"], r["experts"]))
    out = {}
    for q, rows in reqs.items():                      # a layer id that does not grow starts a new token
        toks, cur, last = [], {}, -1
        for layer, ex in rows:
            if layer <= last and cur:
                toks.append(cur); cur = {}
            cur[layer] = ex; last = layer
        if cur:
            toks.append(cur)
        out[q] = toks
    return out


def counts(reqs, skip=()):
    c = collections.Counter()
    for q, toks in reqs.items():
        if q in skip:
            continue
        for tok in toks:
            for l, ex in tok.items():
                for e in ex:
                    c[(l, e)] += 1
    return c


def per_layer_top(prof, l, n):
    return {k for k, _ in sorted(((k, v) for k, v in prof.items() if k[0] == l), key=lambda kv: -kv[1])[:n]}


def banks(prof, B, n_layers, n_exp):
    res = {}
    res["global-top"] = {k for k, _ in prof.most_common(B)}
    per = B // n_layers
    lt = set()
    for l in range(n_layers):
        lt |= per_layer_top(prof, l, per)
    res["layer-top"] = lt
    tot = collections.Counter()
    for (l, e), v in prof.items():
        tot[l] += v
    conc = {}
    for l in range(n_layers):
        top = sorted((v for k, v in prof.items() if k[0] == l), reverse=True)[:max(1, n_exp // 8)]
        conc[l] = sum(top) / max(1, tot[l])
    z = sum(conc.values())
    lp = set()
    for l in range(n_layers):
        lp |= per_layer_top(prof, l, min(n_exp, int(round(B * conc[l] / z))))
    res["layer-prop"] = lp
    byl = sorted(tot, key=lambda l: -tot[l])[:max(1, B // n_exp)]
    res["whole-layers"] = {(l, e) for l in byl for e in range(n_exp)}
    return res


def run(toks, start, bank, every, swaps, decay, mode, prof, minmiss=2.0):
    bank = set(bank)
    score = collections.defaultdict(float)
    miss = collections.Counter()
    seen = collections.Counter()
    hits = picks = ups = 0
    for t, tok in enumerate(toks):
        for l, ex in tok.items():
            for e in ex:
                k = (l, e)
                h = k in bank
                if t >= start:
                    picks += 1; hits += h
                if mode == "decay":
                    seen[k] += 1
                if h:
                    score[k] += 1.0
                else:
                    miss[k] += 1.0
        if not every or (t + 1) % every:
            continue
        if mode == "glob":
            for k in list(score):
                score[k] *= decay ** every
            done = 0
            for k, m in miss.most_common():
                if done >= swaps or m < minmiss:
                    break
                pool = [b for b in bank if b[0] == k[0]]
                if not pool:
                    continue
                o = min(pool, key=lambda b: score[b] + 0.001 * prof.get(b, 0))
                if score[o] >= m:
                    continue
                bank.discard(o); score.pop(o, None); bank.add(k); score[k] = m; ups += 1; done += 1
            miss.clear()
        else:
            for k in seen:
                seen[k] *= decay
            cand = sorted(((v, k) for k, v in seen.items() if k not in bank), reverse=True)
            cold = {}
            for k in bank:
                cold.setdefault(k[0], []).append((seen[k], k))
            for l in cold:
                cold[l].sort()
            done = 0
            for v, k in cand:
                if done >= swaps:
                    break
                lst = cold.get(k[0])
                if not lst:
                    continue
                cv, ck = lst[0]
                if cv >= v:
                    continue
                lst.pop(0)
                bank.discard(ck); bank.add(k); ups += 1; done += 1
    return 100.0 * hits / max(1, picks), ups / max(1, len(toks))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--fracs", default="0.0625,0.125,0.19", help="bank size as share of all (layer, expert) pairs")
    ap.add_argument("--every", default="1,4,16")
    ap.add_argument("--swaps", default="24,96")
    ap.add_argument("--decay", default="0.7,0.97", help="per-round decay for the decayed-count arm")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    reqs = load(a.trace)
    all_c = counts(reqs)
    n_layers = max(l for l, _ in all_c) + 1
    n_exp = max(e for _, e in all_c) + 1
    ntok = sum(len(t) for t in reqs.values())
    top = len(next(iter(next(iter(reqs.values()))[0].values())))
    print(f"trace: {len(reqs)} requests, {ntok} decode tokens, {n_layers} layers x {n_exp} experts, top-{top}")
    rows = []
    for frac in map(float, a.fracs.split(",")):
        B = int(frac * n_layers * n_exp)
        pooled = collections.OrderedDict()
        for q, toks in reqs.items():
            start = max(1, len(toks) // 10)
            prof = counts(reqs, skip=(q,))
            bk = banks(prof, B, n_layers, n_exp)
            ora = counts({q: toks[start:]})
            oracle = set()
            for l in range(n_layers):
                oracle |= per_layer_top(ora, l, B // n_layers)
            arms = [(n, b, 0, 0, 0.97, "glob") for n, b in bk.items()]
            arms.append(("oracle-layer", oracle, 0, 0, 0.97, "glob"))
            for ev in map(int, a.every.split(",")):
                for sw in map(int, a.swaps.split(",")):
                    arms.append((f"adapt-glob e{ev} s{sw}", bk["global-top"], ev, sw, 0.97, "glob"))
                    for dc in map(float, a.decay.split(",")):
                        arms.append((f"adapt-layer e{ev} s{sw} d{dc}", bk["layer-top"], ev, sw, dc, "decay"))
            for name, b, ev, sw, dc, mode in arms:
                h, up = run(toks, start, b, ev, sw, dc, mode, prof)
                n = len(toks) - start
                p = pooled.setdefault(name, [0.0, 0.0, 0])
                p[0] += h * n; p[1] += up * n; p[2] += n
        print(f"\nbank = {frac*100:.2f} % of all experts = {B} slots ({B / n_layers:.1f} per layer)")
        print(f"{'arm':<34} {'hit %':>7} {'uploads/token':>14}")
        for name, (hs, ss, n) in pooled.items():
            print(f"{name:<34} {hs / n:>7.1f} {ss / n:>14.2f}")
            rows.append({"frac": frac, "arm": name, "hit": hs / n, "uploads_per_token": ss / n})
    if a.json:
        json.dump(rows, sys.stdout)


if __name__ == "__main__":
    main()
