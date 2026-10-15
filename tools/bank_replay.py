#!/usr/bin/env python3
"""Offline replay of the hybrid GPU bank (MOE_CACHE_HYBRID) on a MOE_CACHE_TRACE_IDS recording: how many picks a bank of B experts serves.
usage: bank_replay.py TRACE.bin PROFILE.usage --bank 1828 [--every 8,32,128] [--swaps 4,16,64]

Policies (bank of B (layer, expert) slots, same size for every policy):
  fixed     the B most used experts of the usage profile (what the plugin does today)
  adaptive  starts as fixed; every N tokens up to K of the most missed experts since the last swap replace the coldest bank expert of their own layer
            (decayed recent hits + a small share of the profile count) when they were used more often; upload cost = swaps made
  oracle    the best static bank for this trace (upper bound for any static choice)
The first 10 % of the tokens are warm-up for every policy and not counted."""
import argparse, collections, struct, sys
ptok = 1.0

def load_trace(path):
    toks, cur, last = [], {}, -1
    with open(path, "rb") as f:
        b = f.read()
    o = 0
    while o + 4 <= len(b):
        layer, n = struct.unpack_from("<hh", b, o); o += 4
        ids = struct.unpack_from("<%dh" % n, b, o); o += 2 * n
        if layer <= last and cur:
            toks.append(cur); cur = {}
        cur[layer] = ids; last = layer
    if cur:
        toks.append(cur)
    return toks

def load_profile(path):
    b = open(path, "rb").read()
    if b[:4] != b"XUSE":
        sys.exit("profile: only the XUSE format is supported")
    nl, ne = struct.unpack_from("<ii", b, 4)
    c = struct.unpack_from("<%df" % (nl * ne), b, 12)
    return {(l, e): c[l * ne + e] for l in range(nl) for e in range(ne) if c[l * ne + e] > 0}

def run(toks, start, bank, every=0, swaps=0, prof=None, decay=0.97, per_layer=True, seed=0.0, prior=0.0, minmiss=2, duel=0):
    # duel = W > 0: also count what the original profile bank would have served (shadow); every swap round compare the two over the last W tokens:
    # the adapted bank behind -> no swaps this round, and swap back the profile experts that the shadow says were used more
    base = set(bank)
    bank = set(bank)
    win_a = collections.deque(); win_b = collections.deque()
    # seed > 0: a slot starts at seed x its steady-state score from the profile (per-token use / (1 - decay)), so a swap needs evidence that beats the profile
    score = {k: seed * prof.get(k, 0.0) / ptok / (1.0 - decay) if seed else 0.0 for k in bank}
    missed = collections.Counter()
    hits = picks = uploads = 0
    for t, tok in enumerate(toks):
        ha = hb = 0
        for layer, ids in tok.items():
            for e in ids:
                k = (layer, e)
                h = k in bank
                ha += h; hb += k in base
                if t >= start:
                    picks += 1; hits += h
                if h:
                    score[k] += 1.0
                else:
                    missed[k] += 1
        if duel:
            win_a.append(ha); win_b.append(hb)
            if len(win_a) > duel:
                win_a.popleft(); win_b.popleft()
        if every and (t + 1) % every == 0 and duel and sum(win_a) < sum(win_b):
            # the profile bank is ahead: bring back profile experts in place of adapted ones that did worse, no new swaps.
            # As the plugin (hybrid_adapt): scores decay here too, layers are visited in order, at most `swaps` in all
            for k in score:
                score[k] *= decay ** every
            done = 0
            for layer in sorted({k[0] for k in base}):
                for k in sorted((k for k in base - bank if k[0] == layer), key=lambda k: -prof.get(k, 0.0)):
                    if done >= swaps:
                        break
                    pool = [b for b in bank - base if b[0] == layer]
                    if not pool:
                        break
                    o_ = min(pool, key=lambda b: score[b])
                    bank.discard(o_); del score[o_]; bank.add(k); score[k] = 0.0; uploads += 1; done += 1
            missed.clear()
            continue
        if every and (t + 1) % every == 0:
            for k in score:
                score[k] *= decay ** every
            done = 0
            for k, m in missed.most_common():
                if done >= swaps or m < minmiss:
                    break
                if per_layer:          # the plugin's banks have a fixed number of slots per layer: the new expert replaces the coldest one of its own layer
                    pool = [b for b in bank if b[0] == k[0]]
                else:
                    pool = list(bank)
                if not pool:
                    continue
                val = lambda b: score[b] + (prior * prof.get(b, 0.0) / ptok / (1.0 - decay) if prior else 0.001 * prof.get(b, 0.0))
                o_ = min(pool, key=val)
                # no gain: the coldest bank expert was used at least as often. Without --prior this is the plugin's rule (the profile only breaks
                # ties when picking the slot; the comparison uses the raw score). --prior is a replay-only experiment, not in the plugin
                if (val(o_) >= m + prior * prof.get(k, 0.0) / ptok / (1.0 - decay)) if prior else (score[o_] >= m):
                    continue
                bank.discard(o_); del score[o_]
                bank.add(k); score[k] = float(m)
                uploads += 1; done += 1
            missed.clear()
    return 100.0 * hits / max(1, picks), uploads

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace"); ap.add_argument("profile"); ap.add_argument("--bank", type=int, required=True)
    ap.add_argument("--every", default="8,32,128"); ap.add_argument("--swaps", default="4,16,64")
    ap.add_argument("--seed", default="0", help="comma list of profile seed weights to try (0 = slots start cold)")
    ap.add_argument("--prior", default="0", help="comma list: weight of the profile in a slot's value, kept for good (0 = only a tie-breaker)")
    ap.add_argument("--duel", default="0", help="comma list: window (tokens) of the shadow comparison with the profile bank (0 = off)")
    ap.add_argument("--minmiss", default="2", help="comma list: misses since the last swap before an expert may come in")
    a = ap.parse_args()
    toks = load_trace(a.trace)
    prof = load_profile(a.profile)
    global ptok
    n_used = max(len(ids) for tok in toks for ids in tok.values())
    ptok = max(1.0, sum(prof.values()) / (max(len(t) for t in toks) * n_used))      # tokens the profile has seen
    start = len(toks) // 10
    fixed = [k for k, _ in sorted(prof.items(), key=lambda kv: -kv[1])[:a.bank]]
    use = collections.Counter(k for tok in toks[start:] for l, ids in tok.items() for k in ((l, e) for e in ids))
    oracle = [k for k, _ in use.most_common(a.bank)]
    n_tok = len(toks) - start
    print(f"trace: {len(toks)} decode tokens ({n_tok} counted), {max(len(t) for t in toks)} layers; profile: {len(prof)} experts with counts; bank {a.bank} experts")
    f, _ = run(toks, start, fixed)
    o, _ = run(toks, start, oracle)
    print(f"{'policy':<36} {'bank hit %':>10} {'experts uploaded per token':>27}")
    print(f"{'fixed (profile, today)':<36} {f:>10.1f} {0:>27.2f}")
    print(f"{'oracle static (this trace)':<36} {o:>10.1f} {0:>27.2f}")
    for ev in map(int, a.every.split(",")):
        for sw in map(int, a.swaps.split(",")):
            for sd in map(float, a.seed.split(",")):
                for pr in map(float, a.prior.split(",")):
                    for mm in map(int, a.minmiss.split(",")):
                        for du in map(int, a.duel.split(",")):
                            h, up = run(toks, start, fixed, ev, sw, prof, seed=sd, prior=pr, minmiss=mm, duel=du)
                            print(f"{'every %d, %d swaps, duel %d' % (ev, sw, du):<36} {h:>10.1f} {up / len(toks):>27.2f}")

if __name__ == "__main__":
    main()
