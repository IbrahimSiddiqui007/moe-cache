#!/usr/bin/env python3
"""Bank hit replay on a REAL decode routing trace (MOE_CACHE_TRACE_IDS .bin) with the shipped usage profile and with a decode-only profile.
usage: gpu_tier_flash_replay.py TRACE.bin COUNTS.json SHIPPED.usage [--fracs 0.073,0.125,0.167] [--window 161]
COUNTS.json: tokens per request, in order (the .bin has no request marks). Leave-one-request-out for the decode-only profile; the shipped profile is used as is.
The bank is chosen from the profile (global top by count, as the plugin does), then optionally adapted with the plugin's rule; hit rate = share of decode picks served by the bank.
--window N: only the first N decode tokens of each request, from a cold bank (a short answer after a long prompt)."""
import argparse, collections, json, struct, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_tier_replay as g


def load_bin(path, counts):
    d = open(path, "rb").read(); i = 0; toks = []; cur = {}; last = -1
    while i < len(d):
        layer, n = struct.unpack_from("<hh", d, i); ids = struct.unpack_from("<%dh" % n, d, i + 4); i += 4 + 2 * n
        if layer <= last and cur:
            toks.append(cur); cur = {}
        cur[layer] = list(ids); last = layer
    if cur:
        toks.append(cur)
    toks = toks[len(toks) - sum(counts):]      # a stray first decode step (a 1-token prompt batch) is dropped
    out = collections.OrderedDict(); k = 0
    for r, n in enumerate(counts):
        out[r] = toks[k:k + n]; k += n
    assert k == len(toks), (k, len(toks))
    return out


def load_usage(path):
    b = open(path, "rb").read(); nl, ne = struct.unpack_from("<ii", b, 4); c = struct.unpack_from("<%df" % (nl * ne), b, 12)
    return collections.Counter({(l, e): c[l * ne + e] for l in range(nl) for e in range(ne) if c[l * ne + e] > 0})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace"); ap.add_argument("counts"); ap.add_argument("shipped")
    ap.add_argument("--fracs", default="0.073,0.125,0.167"); ap.add_argument("--window", type=int, default=0)
    a = ap.parse_args()
    reqs = load_bin(a.trace, json.load(open(a.counts))); shipped = load_usage(a.shipped)
    n_layers = max(l for t in reqs.values() for tok in t for l in tok) + 1; n_exp = 512
    print(f"trace: {len(reqs)} requests, {sum(len(t) for t in reqs.values())} decode tokens, {n_layers} layers x {n_exp} experts; window {a.window or 'all (10 % warm-up)'}")
    arms = [("shipped profile, fixed", "ship", 0, 0), ("shipped profile + shipped swaps (8,16)", "ship", 8, 16), ("shipped profile + (4,24)", "ship", 4, 24),
            ("decode-only profile, fixed", "dec", 0, 0), ("decode-only + shipped swaps (8,16)", "dec", 8, 16), ("decode-only + (4,24)", "dec", 4, 24), ("decode-only + (4,96)", "dec", 4, 96)]
    for frac in map(float, a.fracs.split(",")):
        B = int(frac * n_layers * n_exp)
        res = collections.OrderedDict((n[0], [0.0, 0]) for n in arms)
        for q, toks in reqs.items():
            if a.window:
                toks = toks[:a.window]
            if len(toks) < 20:
                continue
            start = 0 if a.window else max(1, len(toks) // 10)
            dprof = g.counts(reqs, skip=(q,))
            banks = {"ship": {k for k, _ in shipped.most_common(B)}, "dec": {k for k, _ in dprof.most_common(B)}}
            for name, src, ev, sw in arms:
                prof = shipped if src == "ship" else dprof
                h, _ = g.run(toks, start, banks[src], ev, sw, 0.97, "glob", prof)
                n = len(toks) - start; res[name][0] += h * n; res[name][1] += n
        print(f"\nbank = {frac*100:.1f} % of experts = {B} experts ({B * 3.08 / 1024:.1f} GiB at 3.08 MiB each)")
        print(f"{'arm':<42} {'bank hit %':>10}")
        for name, (hs, n) in res.items():
            print(f"{name:<42} {hs / n:>10.1f}")


if __name__ == "__main__":
    main()
