#!/usr/bin/env python3
"""S10-1: replay saved decode routing traces under RAM-cache replacement policies, hit rate per cache size.
usage: policy_replay.py TRACE [--sizes 10,20,30,50] [--policies lru,lfu,2q,pin50,crf4] [--json]
TRACE: MOE_CACHE_TRACE_IDS .bin (int16 layer, int16 n, n x int16 expert) or a routing .jsonl ({"phase","req","layer","experts"}; decode rows only).
Cache unit = one (layer, expert) pair (all the same size, as in tools/trace_replay.py); size = percent of layers x experts per layer.
Each layer's picks are one op: its experts are visited in ascending id (as the plugin does) and none of them may be evicted while the op runs
(the plugin's in_use protection). Hit rate = share of expert picks found resident, counted on the second half after the first half was replayed
(warm), and on the whole trace from a cold start. Policies:
  lru     plain LRU (what the plugin does)
  lfu     least frequently used (plain counts, persistent)       2q  2Q (25 % FIFO, 75 % LRU, ghost list)
  pin50   half the cache pinned to the first half's most used pairs, the rest LRU (RESULTS 12)
  crf<K>[h<H>]  llama.cpp PR #31233 ClassLRU: decayed use count CRF = old*2^(-dt/H)+1 (H = 128 decode tokens, kept after eviction), K classes
          class = min(K-1, floor(log2 CRF)), one LRU list per class (entry sits in the class of its last use), victim = the class's oldest
          eligible entry with the largest age / 2^class."""
import struct, sys, json, math, heapq, argparse, collections, time


def load(path):
    """-> list of decode tokens; each token = list of (layer, [experts]) in layer order."""
    toks, cur, last = [], [], -1
    def flush():
        nonlocal cur
        if cur: toks.append(cur)
        cur = []
    if path.endswith(".jsonl"):
        req = None
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["phase"] != "decode": continue
                if r["req"] != req or r["layer"] <= last: flush()
                req, last = r["req"], r["layer"]
                cur.append((r["layer"], sorted(r["experts"])))
    else:
        d = open(path, "rb").read(); i = 0
        while i < len(d):
            layer, n = struct.unpack_from("<hh", d, i); ids = struct.unpack_from("<%dh" % n, d, i + 4); i += 4 + 2 * n
            if layer <= last: flush()
            last = layer; cur.append((layer, sorted(ids)))
    flush()
    return toks


class Base:
    """access(key, prot, t) -> hit; prot = set of keys of the running op (not evictable)."""
    def __init__(s, cap): s.cap = cap


class LRU(Base):
    def __init__(s, cap): super().__init__(cap); s.d = collections.OrderedDict()
    def access(s, k, prot, t):
        d = s.d
        if k in d: d.move_to_end(k); return True
        while len(d) >= s.cap:
            for v in d:
                if v not in prot: del d[v]; break
            else: break
        d[k] = 1
        return False


class LFU(Base):
    """lazy heap of (count at push, seq, key); stale entries (count moved on or key gone) are dropped when popped"""
    def __init__(s, cap): super().__init__(cap); s.res = set(); s.cnt = collections.Counter(); s.h = []; s.seq = 0
    def _push(s, k): s.seq += 1; heapq.heappush(s.h, (s.cnt[k], s.seq, k))
    def access(s, k, prot, t):
        s.cnt[k] += 1
        if k in s.res: s._push(k); return True
        while len(s.res) >= s.cap:
            held = []; v = None
            while s.h:
                c, q, x = heapq.heappop(s.h)
                if x not in s.res or c != s.cnt[x]: continue
                if x in prot: held.append((c, q, x)); continue
                v = x; break
            for e in held: heapq.heappush(s.h, e)
            if v is None: break
            s.res.discard(v)
        s.res.add(k); s._push(k)
        return False


class TwoQ(Base):
    def __init__(s, cap):
        super().__init__(cap); s.mcap = max(1, int(cap * 0.75)); s.acap = max(1, cap - s.mcap)
        s.m = collections.OrderedDict(); s.a = collections.OrderedDict(); s.g = collections.OrderedDict()
    def _drop(s, od, prot):
        for v in od:
            if v not in prot: del od[v]; return v
        return None
    def access(s, k, prot, t):
        if k in s.m: s.m.move_to_end(k); return True
        if k in s.a: return True
        if k in s.g:
            del s.g[k]
            if len(s.m) >= s.mcap: s._drop(s.m, prot)
            s.m[k] = 1
        else:
            if len(s.a) >= s.acap:
                v = s._drop(s.a, prot)
                if v is not None:
                    s.g[v] = 1
                    if len(s.g) > s.acap * 4: s.g.popitem(last=False)
            s.a[k] = 1
        return False


class PinLRU(Base):
    def __init__(s, cap, pinned): super().__init__(cap); s.pin = set(pinned); s.lru = LRU(max(1, cap - len(s.pin)))
    def access(s, k, prot, t): return True if k in s.pin else s.lru.access(k, prot, t)


class ClassLRU(Base):
    def __init__(s, cap, K=4, H=128.0):
        super().__init__(cap); s.K, s.H = K, H
        s.lists = [collections.OrderedDict() for _ in range(K)]
        s.cls = {}                      # resident key -> class list it sits in
        s.crf = {}; s.last = {}         # persistent (ghost) scores
        s.last_use = {}
    def access(s, k, prot, t):
        c = s.crf.get(k, 0.0) * 2.0 ** (-(t - s.last.get(k, t)) / s.H) + 1.0
        s.crf[k] = c; s.last[k] = t
        cl = min(s.K - 1, int(math.log2(c)))
        hit = k in s.cls
        if hit:
            del s.lists[s.cls[k]][k]
        else:
            while len(s.cls) >= s.cap:
                best, bv = None, -1.0
                for ci, lst in enumerate(s.lists):
                    for v in lst:                       # oldest first
                        if v not in prot:
                            sc = (t - s.last_use[v]) / (1 << ci)
                            if sc > bv: bv, best = sc, (ci, v)
                            break
                if best is None: break
                del s.lists[best[0]][best[1]]; del s.cls[best[1]]
        s.lists[cl][k] = 1; s.cls[k] = cl
        s.last_use[k] = t
        return hit


def run(toks, mk, warm=(), t0=0):
    c = mk(); t = t0
    for tk in warm:
        for layer, ids in tk:
            prot = {(layer, e) for e in ids}
            for e in ids: c.access((layer, e), prot, t)
        t += 1
    hits = n = 0
    for tk in toks:
        for layer, ids in tk:
            prot = {(layer, e) for e in ids}
            for e in ids:
                n += 1; hits += c.access((layer, e), prot, t)
        t += 1
    return hits / max(1, n)


def factory(name, cap, prof):
    if name == "lru": return lambda: LRU(cap)
    if name == "lfu": return lambda: LFU(cap)
    if name == "2q": return lambda: TwoQ(cap)
    if name.startswith("pin"):
        f = int(name[3:]) / 100
        freq = collections.Counter((l, e) for tk in prof for l, ids in tk for e in ids)
        top = [k for k, _ in freq.most_common(int(cap * f))]
        return lambda: PinLRU(cap, top)
    if name.startswith("crf"):
        body = name[3:]; H = 128.0
        if "h" in body: body, h = body.split("h"); H = float(h)
        K = int(body or 4)
        return lambda: ClassLRU(cap, K, H)
    raise SystemExit("unknown policy " + name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("--sizes", default="10,20,30,50")
    ap.add_argument("--policies", default="lru,lfu,2q,pin50,crf3,crf4,crf5,crf4h64,crf4h256")
    ap.add_argument("--cold", action="store_true", help="also report the whole trace from a cold start")
    a = ap.parse_args()
    toks = load(a.trace); half = len(toks) // 2
    prof, test = toks[:half], toks[half:]
    layers = max(l for tk in toks for l, _ in tk) + 1
    nexp = max(e for tk in toks for _, ids in tk for e in ids) + 1
    total = layers * nexp
    picks = sum(len(ids) for tk in test for _, ids in tk)
    print(f"{a.trace}: {len(toks)} decode tokens, {layers} layers x {nexp} experts = {total} pairs, {picks} picks in the test half")
    pols = a.policies.split(",")
    print("hit rate % on the second half (warm: first half replayed before)" + (", [cold whole trace]" if a.cold else ""))
    print(f"{'size':>5} " + " ".join(f"{p:>9}" for p in pols))
    for sz in [int(x) for x in a.sizes.split(",")]:
        cap = max(1, total * sz // 100); row = []
        for p in pols:
            t0 = time.time(); r = run(test, factory(p, cap, prof), prof, 0)
            s = f"{100*r:8.2f}"
            if a.cold: s += f"/{100*run(toks, factory(p, cap, []), (), 0):.1f}"
            row.append(s); print(f"  [{p} {sz}% {time.time()-t0:.1f}s]", file=sys.stderr)
        print(f"{sz:>4}% " + " ".join(f"{x:>9}" for x in row))

if __name__ == "__main__":
    main()
