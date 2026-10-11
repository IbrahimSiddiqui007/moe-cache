#!/usr/bin/env python3
"""Offline replay of a MOE_CACHE_TRACE_IDS recording (decode steps only): cache policies (D4) and next-layer predictor recall (D2).
usage: trace_replay.py TRACE.bin [--expert-mb 1.44] [--first-half-profile]
Record format: int16 layer, int16 n, n x int16 expert id; one record per MoE layer per decoded token (layers ascend within a token)."""
import struct, sys, collections, argparse

def load(path):
    toks, cur, last = [], {}, -1
    with open(path, "rb") as f:
        while True:
            h = f.read(4)
            if len(h) < 4:
                break
            layer, n = struct.unpack("<hh", h)
            ids = struct.unpack("<%dh" % n, f.read(2 * n))
            if layer <= last and cur:
                toks.append(cur); cur = {}
            cur[layer] = ids; last = layer
    if cur:
        toks.append(cur)
    return toks

class LRU:
    def __init__(s, cap): s.cap, s.d = cap, collections.OrderedDict()
    def access(s, k):
        hit = k in s.d
        if hit: s.d.move_to_end(k)
        else:
            s.d[k] = 1
            if len(s.d) > s.cap: s.d.popitem(last=False)
        return hit

class TwoQ:                                   # 2Q: new entries go to a FIFO (25 %), re-referenced ones to the main LRU (75 %)
    def __init__(s, cap): s.a = collections.OrderedDict(); s.m = LRU(max(1, int(cap * 0.75))); s.acap = max(1, cap - s.m.cap); s.ghost = collections.OrderedDict()
    def access(s, k):
        if k in s.m.d: s.m.access(k); return True
        if k in s.a: return True
        if k in s.ghost:
            del s.ghost[k]; s.m.access(k); return False
        s.a[k] = 1
        if len(s.a) > s.acap:
            old, _ = s.a.popitem(last=False); s.ghost[old] = 1
            if len(s.ghost) > s.acap * 4: s.ghost.popitem(last=False)
        return False

class PinLRU:                                 # most frequent pairs of the profile are pinned (never evicted); the rest is LRU
    def __init__(s, cap, pinned): s.pin = set(pinned); s.lru = LRU(max(1, cap - len(s.pin)))
    def access(s, k): return True if k in s.pin else s.lru.access(k)

def replay(toks, mk, warm=None):
    c = mk(); hits = n = 0
    for tk in (warm or []):
        for layer in sorted(tk):
            for e in tk[layer]: c.access((layer, e))
    for tk in toks:
        for layer in sorted(tk):
            for e in tk[layer]:
                n += 1; hits += c.access((layer, e))
    return hits / max(1, n)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--expert-mb", type=float, default=1.44)
    a = ap.parse_args()
    toks = load(a.trace); half = len(toks) // 2
    prof, test = toks[:half], toks[half:]
    freq = collections.Counter((l, e) for tk in prof for l, ids in tk.items() for e in ids)
    pairs = len(set((l, e) for tk in toks for l, ids in tk.items() for e in ids))
    print(f"tokens {len(toks)} (profile {len(prof)}, test {len(test)}), layers {len(toks[0])}, distinct pairs {pairs}")
    print("cache policy hit rate on the second half (cold start, then warm after replaying the first half):")
    print(f"{'GiB':>6} {'entries':>8} {'LRU':>7} {'2Q':>7} {'pin25+LRU':>10} {'pin50+LRU':>10} {'pin75+LRU':>10}")
    for gib in (4, 8, 12, 16, 19.5):
        cap = int(gib * 1024 / a.expert_mb)
        top = [k for k, _ in freq.most_common(cap)]
        row = [replay(test, lambda: LRU(cap), prof), replay(test, lambda: TwoQ(cap), prof)]
        for f in (0.25, 0.5, 0.75):
            pin = top[:int(cap * f)]
            row.append(replay(test, lambda: PinLRU(cap, pin), prof))
        print(f"{gib:>6} {cap:>8} " + " ".join(f"{100*r:>6.1f}%" if i < 2 else f"{100*r:>9.1f}%" for i, r in enumerate(row)))
    # predictor recall for the next layer, evaluated on the second half
    co = collections.defaultdict(collections.Counter)      # (layer, expert) -> Counter of next-layer experts seen with it
    for tk in prof:
        for l in sorted(tk):
            if l + 1 in tk:
                for e in tk[l]:
                    for g in tk[l + 1]: co[(l, e)][g] += 1
    prev = {}
    for tk in prof:
        for l, ids in tk.items(): prev[l] = set(ids)
    res = collections.defaultdict(lambda: [0, 0])
    for tk in test:
        for l in sorted(tk):
            if l + 1 not in tk: continue
            truth = set(tk[l + 1]); sc = collections.Counter()
            for e in tk[l]:
                for g, c in co[(l, e)].items(): sc[g] += c
            for name, k in (("co-occurrence top10", 10), ("co-occurrence top20", 20), ("co-occurrence top40", 40)):
                pred = {g for g, _ in sc.most_common(k)}
                res[name][0] += len(truth & pred); res[name][1] += len(truth)
            pv = prev.get(l + 1, set())
            res["same layer, previous token"][0] += len(truth & pv); res["same layer, previous token"][1] += len(truth)
            un = pv | {g for g, _ in sc.most_common(10)}
            res["previous token + co-occ top10"][0] += len(truth & un); res["previous token + co-occ top10"][1] += len(truth)
        for l, ids in tk.items(): prev[l] = set(ids)
    print("recall of the experts picked at layer N+1 (from layer N's picks and the first-half statistics):")
    for k, (h, n) in res.items(): print(f"  {k:<32} {100*h/max(1,n):5.1f}%")
main()
