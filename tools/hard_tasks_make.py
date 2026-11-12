#!/usr/bin/env python3
"""MODELS-1141: builds tools/hard_tasks.json, a set of hard long-reasoning tasks with objective checks (standard library only, deterministic: fixed seeds).
Each task: {id, category, prompt, check: {kind, ...}, why_hard}. The prompt always ends with 'Finish with one line: ANSWER: <answer>'. Checks live in tools/hard_tasks_run.py (check()).
  python3 tools/hard_tasks_make.py [--out tools/hard_tasks.json] [--verify]      --verify re-solves every task with the reference solver and fails on a mismatch"""
import itertools, json, os, random, sys

FINISH = "\n\nThink as long as you need. Finish with one line exactly in the form `ANSWER: <answer>`."


# ---- sudoku -------------------------------------------------------------------------------------------------------------------------------------
def sudoku_solutions(grid, limit=2):
    g = [row[:] for row in grid]; sols = []
    def ok(r, c, v):
        if any(g[r][i] == v or g[i][c] == v for i in range(9)): return False
        br, bc = r // 3 * 3, c // 3 * 3
        return all(g[br + i][bc + j] != v for i in range(3) for j in range(3))
    def rec():
        if len(sols) >= limit: return
        best = None
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    cand = [v for v in range(1, 10) if ok(r, c, v)]
                    if best is None or len(cand) < len(best[2]): best = (r, c, cand)
        if best is None: sols.append([row[:] for row in g]); return
        r, c, cand = best
        for v in cand:
            g[r][c] = v; rec(); g[r][c] = 0
            if len(sols) >= limit: return
    rec(); return sols

def make_sudoku(seed, givens):
    rng = random.Random(seed); base = [[(3 * (r % 3) + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]
    digs = list(range(1, 10)); rng.shuffle(digs); base = [[digs[v - 1] for v in row] for row in base]
    rows = [b * 3 + i for b in rng.sample(range(3), 3) for i in rng.sample(range(3), 3)]; cols = [b * 3 + i for b in rng.sample(range(3), 3) for i in rng.sample(range(3), 3)]
    sol = [[base[r][c] for c in cols] for r in rows]; g = [row[:] for row in sol]
    cells = [(r, c) for r in range(9) for c in range(9)]; rng.shuffle(cells); n = 81
    for r, c in cells:
        if n <= givens: break
        keep = g[r][c]; g[r][c] = 0
        if len(sudoku_solutions(g)) == 1: n -= 1
        else: g[r][c] = keep
    return g, sol


# ---- cryptarithm --------------------------------------------------------------------------------------------------------------------------------
def crypt_solutions(words, result, limit=2):
    letters = sorted(set("".join(words) + result)); lead = {w[0] for w in words + [result]}; out = []
    for perm in itertools.permutations(range(10), len(letters)):
        m = dict(zip(letters, perm))
        if any(m[l] == 0 for l in lead): continue
        val = lambda w: int("".join(str(m[c]) for c in w))
        if sum(val(w) for w in words) == val(result):
            out.append(m)
            if len(out) >= limit: break
    return out


# ---- einstein-style logic grid -------------------------------------------------------------------------------------------------------------------
def make_logic(seed, n=5, k=5):
    rng = random.Random(seed)
    cats = {"nationality": ["Brit", "Swede", "Dane", "Norwegian", "German"], "colour": ["red", "green", "white", "yellow", "blue"], "drink": ["tea", "coffee", "milk", "beer", "water"],
            "pet": ["dog", "bird", "cat", "horse", "fish"], "sport": ["tennis", "chess", "golf", "rugby", "squash"]}[:0] if False else None
    names = {"nationality": ["Brit", "Swede", "Dane", "Norwegian", "German"], "colour": ["red", "green", "white", "yellow", "blue"], "drink": ["tea", "coffee", "milk", "beer", "water"],
             "pet": ["dog", "bird", "cat", "horse", "fish"], "sport": ["tennis", "chess", "golf", "rugby", "squash"]}
    keys = list(names)[:k]
    sol = {c: rng.sample(names[c], n) for c in keys}          # sol[c][house] = value
    pos = {(c, v): h for c in keys for h, v in enumerate(sol[c])}
    facts = []
    for _ in range(400):
        (c1, v1), (c2, v2) = rng.sample(list(pos), 2)
        if c1 == c2: continue
        h1, h2 = pos[(c1, v1)], pos[(c2, v2)]
        kind = rng.choice(["same", "same", "left", "next", "notsame"])
        if kind == "same" and h1 == h2: facts.append(("same", c1, v1, c2, v2))
        elif kind == "left" and h1 + 1 == h2: facts.append(("left", c1, v1, c2, v2))
        elif kind == "next" and abs(h1 - h2) == 1: facts.append(("next", c1, v1, c2, v2))
        elif kind == "notsame" and h1 != h2: facts.append(("notsame", c1, v1, c2, v2))
    # house-position facts
    for _ in range(3):
        c, v = rng.choice(list(pos)); facts.append(("at", c, v, pos[(c, v)] + 1))
    def solve_count(fs, limit=2):
        cnt = [0]
        perms = {c: list(itertools.permutations(names[c], n)) for c in keys}
        def holds(assign):
            for f in fs:
                if f[0] == "at":
                    _, c, v, h = f
                    if c in assign and assign[c][h - 1] != v: return False
                else:
                    _, c1, v1, c2, v2 = f
                    if c1 in assign and c2 in assign:
                        h1, h2 = assign[c1].index(v1), assign[c2].index(v2)
                        if f[0] == "same" and h1 != h2: return False
                        if f[0] == "left" and h1 + 1 != h2: return False
                        if f[0] == "next" and abs(h1 - h2) != 1: return False
                        if f[0] == "notsame" and h1 == h2: return False
            return True
        def rec(i, assign):
            if cnt[0] >= limit: return
            if i == len(keys): cnt[0] += 1; return
            for p in perms[keys[i]]:
                assign[keys[i]] = p
                if holds(assign): rec(i + 1, assign)
                del assign[keys[i]]
        rec(0, {}); return cnt[0]
    rng.shuffle(facts); chosen = []
    for f in facts:                                         # add facts until the solution is unique, then drop redundant ones
        chosen.append(f)
        if len(chosen) >= 6 and solve_count(chosen) == 1: break
    for f in list(chosen):
        t = [x for x in chosen if x is not f]
        if solve_count(t) == 1: chosen = t
    def txt(f):
        if f[0] == "at": return f"The {f[2]} ({f[1]}) lives in house {f[3]}."
        _, c1, v1, c2, v2 = f
        a, b = f"the {v1} ({c1})", f"the {v2} ({c2})"
        return {"same": f"{a.capitalize()} and {b} are in the same house.", "left": f"{a.capitalize()} is in the house directly to the left of {b}.",
                "next": f"{a.capitalize()} and {b} are in neighbouring houses.", "notsame": f"{a.capitalize()} and {b} are not in the same house."}[f[0]]
    c, v = rng.choice([(c, v) for c in keys[1:] for v in names[c]]); ask_house = pos[(c, v)] + 1
    asked = keys[0]; ans = sol[asked][ask_house - 1]
    q = f"There are {n} houses in a row, numbered 1 (left) to {n} (right). Each house has one value of each category: " + "; ".join(f"{cc}: {', '.join(names[cc])}" for cc in keys) + ".\nClues:\n" + "\n".join(f"- {txt(f)}" for f in chosen) + f"\nQuestion: what is the {asked} of the person whose {c} is {v}? Answer with that single word."
    return q, ans.lower(), len(chosen)


def tasks():
    T = []
    for i, (seed, g) in enumerate([(11, 24), (12, 23), (13, 22), (14, 23)]):
        grid, sol = make_sudoku(seed, g)
        rows = "\n".join(" ".join(str(v) if v else "." for v in r) for r in grid)
        T.append({"id": f"sudoku_{i+1}", "category": "sudoku", "prompt": "Solve this 9x9 sudoku (. = empty). It has exactly one solution.\n\n" + rows + "\n\nGive the completed grid as 81 digits in reading order, no spaces, as the answer." + FINISH,
                  "check": {"kind": "exact", "answer": "".join(str(v) for r in sol for v in r), "grid": rows.replace(" ", "").replace("\n", "")}, "why_hard": f"{g} givens: long chains of eliminations, easy to slip one digit; every wrong digit fails the whole grid."})
    for i, (words, res) in enumerate([(["SEND", "MORE"], "MONEY"), (["TWO", "TWO"], "FOUR"), (["EAT", "THAT"], "APPLE"), (["CROSS", "ROADS"], "DANGER"), (["BASE", "BALL"], "GAMES")]):
        sols = crypt_solutions(words, res)
        if len(sols) != 1: continue
        m = sols[0]; ans = ",".join(f"{c}={m[c]}" for c in sorted(m))
        T.append({"id": f"cryptarithm_{i+1}", "category": "cryptarithm", "prompt": f"Each letter stands for a different digit 0-9 (same letter, same digit; no leading zero). Solve {' + '.join(words)} = {res}. Answer with the letter=digit pairs sorted alphabetically, comma separated, like `A=1,B=2`." + FINISH,
                  "check": {"kind": "exact", "answer": ans}, "why_hard": f"{len(m)} letters, carry reasoning with unique solution; models branch and backtrack a lot."})
    n = 0
    for seed in range(200, 260):
        if n >= 4: break
        q, ans, nf = make_logic(seed)
        n += 1; T.append({"id": f"logicgrid_{n}", "category": "logic grid", "prompt": q + FINISH, "check": {"kind": "word", "answer": ans}, "why_hard": f"Einstein-style 5x5 grid with {nf} clues and a unique solution: needs systematic case analysis."})
    rng = random.Random(7)
    for i in range(3):
        a, b = rng.randint(1000000, 9999999), rng.randint(1000000, 9999999); c = rng.randint(100000, 999999)
        T.append({"id": f"bigarith_{i+1}", "category": "long arithmetic", "prompt": f"Compute exactly: {a} * {b} + {c} * {c}. Answer with the integer only." + FINISH, "check": {"kind": "int", "answer": str(a * b + c * c)}, "why_hard": "14-digit products by hand: thinking models write out partial products and re-check them, a long, error-prone chain."})
    for i in range(3):
        base, e, m = rng.randint(3, 19), rng.randint(2 ** 20, 2 ** 24), rng.choice([1000003, 998244353, 1000000007][:2 + i % 1])
        T.append({"id": f"modexp_{i+1}", "category": "modular arithmetic", "prompt": f"Compute {base}^{e} mod {m}. Answer with the integer only." + FINISH, "check": {"kind": "int", "answer": str(pow(base, e, m))}, "why_hard": "Square-and-multiply with 20+ squarings of 9-digit numbers modulo a prime: long exact arithmetic, one slip ruins the answer."})
    for i in range(3):
        sd = 300 + i; r = random.Random(sd); n_nodes = 14 + 2 * i; edges = {}
        for u in range(n_nodes):
            for v in r.sample(range(n_nodes), 3):
                if u != v: edges[(u, v)] = r.randint(1, 9)
        import heapq
        dist = {0: 0}; cnt = {0: 1}; pq = [(0, 0)]
        adj = {}
        for (u, v), w in edges.items(): adj.setdefault(u, []).append((v, w))
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, 1e9): continue
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, 1e9): dist[v] = nd; cnt[v] = cnt[u]; heapq.heappush(pq, (nd, v))
                elif nd == dist[v]: cnt[v] += cnt[u]
        tgt = max(dist, key=lambda x: (dist[x], x)); lst = "; ".join(f"{u}->{v}:{w}" for (u, v), w in sorted(edges.items()))
        T.append({"id": f"graph_{i+1}", "category": "graph", "prompt": f"A directed graph has nodes 0..{n_nodes-1} and weighted edges (u->v:weight): {lst}.\nHow many different shortest paths are there from node 0 to node {tgt}? Answer with `<shortest distance>,<number of shortest paths>`." + FINISH,
                  "check": {"kind": "exact", "answer": f"{dist[tgt]},{cnt[tgt]}"}, "why_hard": f"{len(edges)} edges: Dijkstra by hand with path counting; ties are easy to miss."})
    for i, (L, forb) in enumerate([(14, ["0110", "1001"]), (16, ["000", "1111"]), (12, ["10101", "0110"])]):
        c = sum(1 for bits in itertools.product("01", repeat=L) if not any(f in "".join(bits) for f in forb))
        T.append({"id": f"count_{i+1}", "category": "combinatorics", "prompt": f"How many binary strings of length {L} contain none of the substrings {', '.join(forb)}? Answer with the integer only." + FINISH, "check": {"kind": "int", "answer": str(c)}, "why_hard": "Needs a transfer-matrix / DP over overlapping forbidden patterns; brute-force by hand is hopeless, so the model must set up and iterate a recurrence exactly."})
    prog = ["def f(n):\n    a, b, steps = 0, 1, 0\n    while n > 1:\n        if n % 2: n = 3 * n + 1\n        else: n //= 2\n        a, b = b, (a + b) % 1000\n        steps += 1\n    return steps, a\nprint(f({N}))", "def f(s):\n    x = 7\n    for i, ch in enumerate(s):\n        x = (x * 31 + ord(ch) * (i + 1)) % 10007\n    return x\nprint(f('{S}'))", "def f(n):\n    t = [0] * (n + 1); t[0] = 1\n    for c in (1, 5, 10, 25, 50):\n        for v in range(c, n + 1): t[v] += t[v - c]\n    return t[n]\nprint(f({M}))"]
    for i, (src, kw) in enumerate([(prog[0], {"N": 837799}), (prog[1], {"S": "the quick brown fox jumps over the lazy dog 1234567890"}), (prog[2], {"M": 1000})]):
        code = src.format(**kw); import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): exec(code, {})
        T.append({"id": f"trace_{i+1}", "category": "program tracing", "prompt": f"What does this program print? Give the exact output (for a tuple print it as Python prints it).\n\n```python\n{code}\n```" + FINISH, "check": {"kind": "exact", "answer": buf.getvalue().strip()}, "why_hard": "Many iterations of a state update (hundreds of steps / a 1000-entry DP table): the model has to simulate exactly or find a closed form."})
    return T


if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(os.path.dirname(os.path.abspath(__file__)), "hard_tasks.json")
    T = tasks()
    if "--verify" in sys.argv:
        for t in T:
            if t["category"] == "sudoku":
                g = [[int(ch) if ch.isdigit() else 0 for ch in t["check"]["grid"][r * 9:(r + 1) * 9]] for r in range(9)]
                s = sudoku_solutions(g); assert len(s) == 1 and "".join(str(v) for r in s[0] for v in r) == t["check"]["answer"], t["id"]
        print("verified", len(T))
    json.dump(T, open(out, "w"), indent=1, ensure_ascii=False)
    import collections
    print(len(T), "tasks:", dict(collections.Counter(t["category"] for t in T)), "->", out)
