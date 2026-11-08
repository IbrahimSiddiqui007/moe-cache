#!/usr/bin/env python3
"""Analysis of a recorded decode routing trace (MOE_CACHE_TRACE_IDS .bin) for the hybrid GPU bank: ceilings, drift, per-layer value.
usage: gpu_tier_trace_analysis.py TRACE.bin SHIPPED.usage [--bank 4104] [--cpu-us 122] [--layer-us 270] [--window 48]
(a) ceilings for a bank of --bank experts with the per-layer slot counts of the profile's global top (what the plugin builds):
    fixed (the profile bank), MIN (offline optimum per layer with swaps on any miss: Belady with bypass), rate-limited oracle (every R tokens up to S swaps, future knowledge of the next 8 tokens)
    and the plugin's own rule (hybrid_adapt) for reference.
(b) drift: Jaccard overlap of each layer's top-N used experts between adjacent windows of --window tokens, and between the trace's top-N and the profile bank's experts of that layer;
    repeat rate: share of a layer's picks at token t that were also picked at t-1.
(c) per-layer value: hits per token x --cpu-us saved against --layer-us paid; layers that lose; total net per token with the bank on everywhere vs only where it wins."""
import argparse, collections, struct, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_tier_replay as g
import gpu_tier_flash_replay as fr


def load(path):
    d = open(path, "rb").read(); i = 0; toks = []; cur = {}; last = -1
    while i < len(d):
        layer, n = struct.unpack_from("<hh", d, i); ids = struct.unpack_from("<%dh" % n, d, i + 4); i += 4 + 2 * n
        if layer <= last and cur:
            toks.append(cur); cur = {}
        cur[layer] = list(ids); last = layer
    if cur:
        toks.append(cur)
    return toks


def hits_of(toks, bank_by_layer):
    h = [0] * 64; p = [0] * 64
    for tok in toks:
        for l, ex in tok.items():
            for e in ex:
                p[l] += 1; h[l] += (l, e) in bank_by_layer
    return h, p


def min_policy(toks, bank):
    """offline MIN per layer with bypass; swaps on a miss when the missed expert is needed sooner than the farthest resident"""
    n = len(toks); layers = sorted({l for t in toks for l in t}); hits = picks = ups = 0
    res = {l: {e for (ll, e) in bank if ll == l} for l in layers}
    for l in layers:
        uses = collections.defaultdict(list)
        for t, tok in enumerate(toks):
            for e in tok.get(l, []):
                uses[e].append(t)
        ptr = collections.defaultdict(int)
        def nxt(e, t):
            u = uses.get(e, []); k = ptr[e]
            while k < len(u) and u[k] <= t: k += 1
            ptr[e] = k
            return u[k] if k < len(u) else 10 ** 9
        for t, tok in enumerate(toks):
            cur = set(tok.get(l, []))
            for e in tok.get(l, []):
                picks += 1
                if e in res[l]: hits += 1
            for e in tok.get(l, []):
                if e in res[l]: continue
                cand = [x for x in res[l] if x not in cur]
                if not cand: continue
                v = max(cand, key=lambda x: nxt(x, t))
                if nxt(e, t) < nxt(v, t):
                    res[l].discard(v); res[l].add(e); ups += 1
    return 100.0 * hits / picks, ups / n


def oracle_rate(toks, bank, every, swaps, horizon=8):
    bankb = set(bank); hits = picks = ups = 0
    for t, tok in enumerate(toks):
        if t % every == 0:
            fut = collections.Counter()
            for tt in toks[t:t + horizon]:
                for l, ex in tt.items():
                    for e in ex: fut[(l, e)] += 1
            cands = []
            for l in {k[0] for k in bankb}:
                res = sorted((fut[k], k) for k in bankb if k[0] == l)
                new = sorted(((c, k) for k, c in fut.items() if k[0] == l and k not in bankb), reverse=True)
                for (cv, ck), (nv, nk) in zip(res, new):
                    if nv > cv: cands.append((nv - cv, ck, nk))
            cands.sort(key=lambda x: -x[0])
            used = set()
            for gain, ck, nk in cands[:swaps]:
                if ck in bankb and nk not in bankb: bankb.discard(ck); bankb.add(nk); ups += 1
        for l, ex in tok.items():
            for e in ex:
                picks += 1; hits += (l, e) in bankb
    return 100.0 * hits / picks, ups / len(toks)


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace"); ap.add_argument("shipped"); ap.add_argument("--bank", type=int, default=4104)
    ap.add_argument("--cpu-us", type=float, default=122); ap.add_argument("--layer-us", type=float, default=270); ap.add_argument("--window", type=int, default=48)
    a = ap.parse_args()
    toks = load(a.trace); prof = fr.load_usage(a.shipped); nt = len(toks)
    layers = sorted({l for t in toks for l in t}); n_picks_layer = len(next(iter(toks[0].values())))
    bank = {k for k, _ in prof.most_common(a.bank)}
    print(f"trace: {nt} decode tokens, {len(layers)} layers, top-{n_picks_layer}; bank {len(bank)} experts from the shipped profile")
    # (a)
    print("\n(a) bank hit % on the real chat trace (cold bank, all tokens counted)")
    f, _ = g.run(toks, 0, bank, 0, 0, 0.97, "glob", prof)
    rows = [("fixed (shipped profile bank)", f, 0.0)]
    for ev, sw in ((8, 16), (4, 24), (4, 96)):
        h, u = g.run(toks, 0, bank, ev, sw, 0.97, "glob", prof); rows.append((f"plugin rule, every {ev} tokens, {sw} swaps", h, u))
    own = g.counts({0: toks}); oracle_static = {k for k, _ in own.most_common(len(bank))}
    h, _ = g.run(toks, 0, oracle_static, 0, 0, 0.97, "glob", prof); rows.append(("best static bank of this trace (global top, in-sample)", h, 0.0))
    for ev, sw in ((8, 16), (4, 24), (4, 96)):
        h, u = oracle_rate(toks, bank, ev, sw); rows.append((f"future-knowing oracle at that rate, every {ev}, {sw} swaps", h, u))
    h, u = min_policy(toks, bank); rows.append(("MIN per layer (offline optimum, swaps on any miss)", h, u))
    print(f"{'policy':<62} {'bank hit %':>10} {'swaps/token':>12}")
    for name, h, u in rows:
        print(f"{name:<62} {h:>10.1f} {u:>12.1f}")
    # (b)
    print(f"\n(b) drift and repeat (windows of {a.window} tokens)")
    N = max(1, a.bank // len(layers))
    wins = [toks[i:i + a.window] for i in range(0, nt - a.window + 1, a.window)]
    rows = []
    for l in layers:
        tops = []
        for w in wins:
            c = collections.Counter(e for tok in w for e in tok.get(l, []))
            tops.append({e for e, _ in c.most_common(N)})
        adj = [jaccard(tops[i], tops[i + 1]) for i in range(len(tops) - 1)]
        pb = {e for (ll, e) in bank if ll == l}
        c_all = collections.Counter(e for tok in toks for e in tok.get(l, []))
        top_all = {e for e, _ in c_all.most_common(len(pb))}
        rep = sum(len(set(toks[t].get(l, [])) & set(toks[t - 1].get(l, []))) for t in range(1, nt)) / (n_picks_layer * (nt - 1))
        rows.append((sum(adj) / max(1, len(adj)), jaccard(top_all, pb), rep))
    m = lambda i: sum(r[i] for r in rows) / len(rows)
    print(f"mean over layers: Jaccard of the top-{N} sets between adjacent windows {m(0):.2f}; Jaccard of the trace's top set with the profile bank's set of that layer {m(1):.2f}; repeat rate t -> t-1 {m(2):.2f}")
    # (c)
    print(f"\n(c) per-layer value: saved {a.cpu_us:.0f} us per bank hit against {a.layer_us:.0f} us paid per layer step")
    hf, pf = hits_of(toks, bank)
    adapt_bank = None
    out = []
    for l in layers:
        hpt = hf[l] / nt
        out.append((l, hpt, hpt * a.cpu_us - a.layer_us, len({e for (ll, e) in bank if ll == l})))
    win = [o for o in out if o[2] > 0]; lose = [o for o in out if o[2] <= 0]
    tot_all = sum(o[2] for o in out); tot_win = sum(o[2] for o in win)
    print(f"hits per token and layer: mean {sum(o[1] for o in out) / len(out):.2f} (break-even {a.layer_us / a.cpu_us:.2f}); layers that win {len(win)}, lose {len(lose)}")
    print(f"net per token with the fixed bank on everywhere: {tot_all / 1000:+.2f} ms; only in the winning layers: {tot_win / 1000:+.2f} ms (slots of the losing layers: {sum(o[3] for o in lose)} of {len(bank)})")
    print("losing layers (layer: hits/token, slots): " + ", ".join(f"{o[0]}: {o[1]:.2f}, {o[3]}" for o in lose[:20]) + (" ..." if len(lose) > 20 else ""))
    print("best layers: " + ", ".join(f"{o[0]}: {o[1]:.2f}, {o[3]}" for o in sorted(out, key=lambda x: -x[1])[:8]))
    # reallocation of the losing layers' slots to the winners by greedy marginal gain; two versions:
    #   in-sample (ranking and hits from the trace itself: optimistic) and out-of-sample (ranking from the shipped profile, hits measured on the trace)
    per_layer_c = {l: collections.Counter(e for tok in toks for e in tok.get(l, [])).most_common() for l in layers}
    per_layer_p = {}
    for l in layers:
        pc = sorted(((e, c) for (ll, e), c in prof.items() if ll == l), key=lambda x: -x[1]); tot = sum(c for _, c in pc) or 1
        per_layer_p[l] = [(e, n_picks_layer * c / tot) for e, c in pc]
    base_slots = {l: len({e for (ll, e) in bank if ll == l}) for l in layers}
    pool = sum(base_slots[o[0]] for o in lose); on = [o[0] for o in win]
    for name, rank in (("in-sample", per_layer_c), ("out-of-sample (profile ranking)", per_layer_p)):
        slots = dict(base_slots)
        for l in [o[0] for o in lose]: slots[l] = 0
        for _ in range(pool):
            best = max(on, key=lambda l: (rank[l][slots[l]][1] if slots[l] < len(rank[l]) else 0))
            slots[best] += 1
        tot_h = 0
        for l in on:
            chosen = {e for e, _ in rank[l][:slots[l]]}
            tot_h += sum(1 for tok in toks for e in tok.get(l, []) if e in chosen)
        net = sum(sum(1 for tok in toks for e in tok.get(l, []) if e in {x for x, _ in rank[l][:slots[l]]}) / nt * a.cpu_us - a.layer_us for l in on)
        print(f"bank off in the {len(lose)} losing layers, their {pool} slots to the winners ({name}): picks served {100.0 * tot_h / (nt * n_picks_layer * len(layers)):.1f} % of all picks; net {net / 1000:+.2f} ms per token")


if __name__ == "__main__":
    main()
