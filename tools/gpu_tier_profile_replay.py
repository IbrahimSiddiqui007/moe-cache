#!/usr/bin/env python3
"""Does a prompt-heavy usage profile hurt the hybrid bank? Replay of the bank hit rate in decode when the bank is chosen from a profile with a given prompt weight.
usage: gpu_tier_profile_replay.py TRACE.jsonl [--fracs 0.125,0.19] [--weights 0,1,30,300] [--every 8,4] [--swaps 16,24]
TRACE: routing .jsonl with prefill AND decode rows. Profile of the bank = decode picks of the other requests + W x their prompt picks (W = 0: decode only;
the plugin today counts every group, so a 64k prompt makes W large). Leave-one-request-out; the same bank size for every arm; first 10 % of the decode tokens are warm-up.
Arms: fixed (the bank as chosen), adapt (the plugin's hybrid_adapt rule from gpu_tier_replay.py: misses since the last round replace the coldest expert of the layer)."""
import argparse, collections, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_tier_replay as g


def load_both(path):
    pre = collections.defaultdict(collections.Counter)
    for line in open(path):
        r = json.loads(line)
        if r["phase"] == "prefill":
            for e in r["experts"]:
                pre[r["req"]][(r["layer"], e)] += 1
    return pre


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace"); ap.add_argument("--fracs", default="0.125,0.19"); ap.add_argument("--weights", default="0,1,30,300")
    ap.add_argument("--window", type=int, default=0, help="count only the first N decode tokens of each request, from a cold bank (no warm-up): a short answer after a long prompt")
    ap.add_argument("--every", default="8,4"); ap.add_argument("--swaps", default="16,24")
    a = ap.parse_args()
    dec = g.load(a.trace); pre = load_both(a.trace)
    allc = g.counts(dec); n_layers = max(l for l, _ in allc) + 1; n_exp = max(e for _, e in allc) + 1
    ntok_p = sum(sum(c.values()) for c in pre.values()); ntok_d = sum(sum(g.counts({q: t}).values()) for q, t in dec.items())
    print(f"trace: {len(dec)} requests, decode picks {ntok_d}, prompt picks {ntok_p} (ratio {ntok_p / ntok_d:.2f}), {n_layers} layers x {n_exp} experts")
    base_ratio = ntok_p / ntok_d
    for frac in map(float, a.fracs.split(",")):
        B = int(frac * n_layers * n_exp)
        print(f"\nbank = {frac*100:.1f} % of experts = {B} slots")
        print(f"{'prompt weight (x trace ratio)':<32} {'prompt:decode in profile':>24} {'fixed hit %':>12} " + " ".join(f"{'adapt e%s s%s' % (e, s):>14}" for e, s in zip(a.every.split(","), a.swaps.split(","))))
        for W in map(float, a.weights.split(",")):
            res = collections.defaultdict(lambda: [0.0, 0])
            for q, toks in dec.items():
                if a.window:
                    toks = toks[:a.window]
                start = 0 if a.window else max(1, len(toks) // 10)
                prof = collections.Counter(g.counts(dec, skip=(q,)))
                for q2, c in pre.items():
                    if q2 != q:
                        for k, v in c.items():
                            prof[k] += W * v
                bank = {k for k, _ in prof.most_common(B)}
                n = len(toks) - start
                h, _ = g.run(toks, start, bank, 0, 0, 0.97, "glob", prof)
                res["fixed"][0] += h * n; res["fixed"][1] += n
                for e, s in zip(a.every.split(","), a.swaps.split(",")):
                    h, _ = g.run(toks, start, bank, int(e), int(s), 0.97, "glob", prof)
                    res[(e, s)][0] += h * n; res[(e, s)][1] += n
            ratio = W * base_ratio
            row = f"{'W=%g' % W:<32} {ratio:>24.2f} {res['fixed'][0] / res['fixed'][1]:>12.1f} "
            row += " ".join(f"{res[(e, s)][0] / res[(e, s)][1]:>14.1f}" for e, s in zip(a.every.split(","), a.swaps.split(",")))
            print(row)


if __name__ == "__main__":
    main()
