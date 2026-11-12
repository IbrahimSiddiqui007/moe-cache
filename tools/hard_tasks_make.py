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


def verify_all(T):
    """Re-derive every task's answer FROM ITS PROMPT TEXT with code that does not share the generator's data (MODELS-1141/M4): a prompt that does not match its check fails here."""
    import re as _re
    for t in T:
        p, ans, cat = t["prompt"], t["check"]["answer"], t["category"]
        if cat == "sudoku":
            g = [[int(ch) if ch.isdigit() else 0 for ch in t["check"]["grid"][r * 9:(r + 1) * 9]] for r in range(9)]
            s_ = sudoku_solutions(g); assert len(s_) == 1 and "".join(str(v) for r in s_[0] for v in r) == ans, t["id"]
            rows = [ln.split() for ln in p.split("\n\n")[1].splitlines()]; assert "".join(c if c != "." else "." for r in rows for c in r) == t["check"]["grid"], t["id"] + ": grid in prompt differs from the check"
        elif cat == "cryptarithm":
            m = _re.search(r"Solve ([A-Z]+(?: \+ [A-Z]+)*) = ([A-Z]+)", p); words, res = m.group(1).split(" + "), m.group(2)
            sols = crypt_solutions(words, res, limit=3); assert len(sols) == 1, t["id"] + ": not unique"
            mp = dict(kv.split("=") for kv in ans.split(",")); assert sum(int("".join(mp[c] for c in w)) for w in words) == int("".join(mp[c] for c in res)) and len(set(mp.values())) == len(mp), t["id"]
        elif cat == "long arithmetic":
            m = _re.search(r"exactly: (\d+) \* (\d+) \+ (\d+) \* (\d+)", p); x, y, z, w = map(int, m.groups()); assert str(x * y + z * w) == ans and z == w, t["id"]
        elif cat == "modular arithmetic":
            m = _re.search(r"Compute (\d+)\^(\d+) mod (\d+)", p); b_, e_, md = map(int, m.groups()); r_ = 1
            for bit in bin(e_)[2:]: r_ = r_ * r_ % md; r_ = r_ * b_ % md if bit == "1" else r_
            assert str(r_) == ans, t["id"]
        elif cat == "graph":
            m = _re.search(r"weighted edges \(u->v:weight\): (.*?)\.\nHow many", p, _re.S); E = [tuple(map(int, x)) for x in _re.findall(r"(\d+)->(\d+):(\d+)", m.group(1))]
            n_ = int(_re.search(r"nodes 0\.\.(\d+)", p).group(1)) + 1; INF = 10 ** 9; D = [[INF] * n_ for _ in range(n_)]
            for u, v, w in E: D[u][v] = min(D[u][v], w)
            for i in range(n_): D[i][i] = 0
            for k in range(n_):
                for i in range(n_):
                    for j in range(n_): D[i][j] = min(D[i][j], D[i][k] + D[k][j])
            tgt = int(_re.search(r"from node 0 to node (\d+)", p).group(1)); cnt = {0: 1}
            for v in sorted((x for x in range(n_) if D[0][x] < INF), key=lambda x: D[0][x]):
                if v: cnt[v] = sum(cnt[u] for u, vv, w in E if vv == v and u in cnt and D[0][u] + w == D[0][v])
            assert f"{D[0][tgt]},{cnt[tgt]}" == ans, t["id"]
        elif cat == "combinatorics":
            m = _re.search(r"length (\d+) contain none of the substrings ([01, ]+)\?", p); L = int(m.group(1)); forb = [x.strip() for x in m.group(2).split(",")]; k = max(map(len, forb)) - 1
            st = {"": 1}
            for _ in range(L):
                nx = {}
                for suf, c in st.items():
                    for bit in "01":
                        w = suf + bit
                        if any(w.endswith(f) for f in forb): continue
                        key = w[-k:] if k else ""; nx[key] = nx.get(key, 0) + c
                st = nx
            assert str(sum(st.values())) == ans, t["id"]
        elif cat == "program tracing":
            import io, contextlib
            code = _re.search(r"```python\n(.*?)\n```", p, _re.S).group(1); buf = io.StringIO()
            with contextlib.redirect_stdout(buf): exec(code, {})
            assert buf.getvalue().strip() == ans, t["id"]
        elif cat == "logic grid":
            names = {"nationality": ["Brit", "Swede", "Dane", "Norwegian", "German"], "colour": ["red", "green", "white", "yellow", "blue"], "drink": ["tea", "coffee", "milk", "beer", "water"], "pet": ["dog", "bird", "cat", "horse", "fish"], "sport": ["tennis", "chess", "golf", "rugby", "squash"]}
            cl = [ln[2:] for ln in p.splitlines() if ln.startswith("- ")]; cons = []
            tk = lambda x: _re.findall(r"the (\w+) \((\w+)\)", x.lower())
            for c in cl:
                low = c.lower(); pr = [(v, k_) for v, k_ in tk(c)]
                if "lives in house" in low: cons.append(("at", pr[0], int(_re.search(r"house (\d)", low).group(1)) - 1))
                elif "same house" in low and "not" not in low: cons.append(("same", pr[0], pr[1]))
                elif "not in the same house" in low: cons.append(("notsame", pr[0], pr[1]))
                elif "directly to the left" in low: cons.append(("left", pr[0], pr[1]))
                elif "neighbouring" in low: cons.append(("next", pr[0], pr[1]))
                else: raise AssertionError(t["id"] + ": clue not understood: " + c)
            keys = list(names)[:5]; sols = []
            def ok(asg):
                for c in cons:
                    if c[0] == "at":
                        v, k_ = c[1]
                        if k_ in asg and asg[k_].index(v.capitalize() if v.capitalize() in names[k_] else v) != c[2]: return False
                        continue
                    (v1, k1), (v2, k2) = c[1], c[2]
                    if k1 in asg and k2 in asg:
                        f = lambda v, k_: asg[k_].index(next(x for x in names[k_] if x.lower() == v)); h1, h2 = f(v1, k1), f(v2, k2)
                        if c[0] == "same" and h1 != h2 or c[0] == "notsame" and h1 == h2 or c[0] == "left" and h1 + 1 != h2 or c[0] == "next" and abs(h1 - h2) != 1: return False
                return True
            def go(i, asg):
                if len(sols) > 1: return
                if i == len(keys): sols.append(dict(asg)); return
                for pm in itertools.permutations(names[keys[i]]):
                    asg[keys[i]] = pm
                    if ok(asg): go(i + 1, asg)
                    del asg[keys[i]]
            go(0, {}); assert len(sols) == 1, t["id"] + f": {len(sols)} solutions"
            q = _re.search(r"Question: what is the (\w+) of the person whose (\w+) is (\w+)\?", p); assert q, t["id"] + ": question not understood"
            ck, ck2, v = q.group(1), q.group(2), q.group(3); h = sols[0][ck2].index(next(x for x in names[ck2] if x.lower() == v)); assert sols[0][ck][h].lower() == ans, t["id"]
        else:
            raise AssertionError("no verifier for category " + cat)


if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(os.path.dirname(os.path.abspath(__file__)), "hard_tasks.json")
    T = tasks()
    if "--verify" in sys.argv:
        verify_all(T)
        print("verified", len(T), "tasks (every category re-solved from its own prompt text)")
    json.dump(T, open(out, "w"), indent=1, ensure_ascii=False)
    import collections
    print(len(T), "tasks:", dict(collections.Counter(t["category"] for t in T)), "->", out)
