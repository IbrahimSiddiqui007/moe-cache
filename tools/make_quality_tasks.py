#!/usr/bin/env python3
"""Builds tools/quality_tasks.json, the 100+ exact-answer task set for quality_yardstick.py (`run --tasks tools/quality_tasks.json`).

Every answer is computed here (arithmetic and dates with Python, code-output tasks by running the snippet, logic puzzles by brute force),
so the file can be regenerated and re-checked: `python3 -I tools/make_quality_tasks.py` rewrites the JSON and fails on any inconsistency.
Questions are original. The checker is the yardstick's own: any listed regex matching the lowercased answer (thinking removed) counts as right.
"""
import datetime as dt, itertools, json, os, re, subprocess, sys
from fractions import Fraction

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_tasks.json")
tasks = []          # (category, prompt, [regex])
NUM_L, NUM_R = r"(?<![\d.,])", r"(?!\d|[.,]\d)"


def add(cat, prompt, *regexes):
    tasks.append((cat, prompt, list(regexes)))


def num(v):
    """Regex for a number: optional minus sign, optional thousands commas, optional trailing zeros for decimals."""
    v = Fraction(v)
    neg = v < 0
    v = abs(v)
    ip, fp = int(v), v - int(v)
    digits = str(ip)
    grp = []
    while len(digits) > 3:
        grp.insert(0, digits[-3:])
        digits = digits[:-3]
    s = digits + "".join(",?" + g for g in grp)
    if fp == 0:
        s += r"(\.0+)?"
    else:
        frac = str(fp.numerator * 10 ** 12 // fp.denominator).rjust(12, "0").rstrip("0")
        assert fp.denominator in (2, 4, 5, 8, 10, 20, 25, 50, 100, 1000), fp
        s += r"\." + frac + "0*"
    return (r"[-−]" if neg else "") + NUM_L + s + NUM_R


def N(cat, prompt, expr):
    v = Fraction(eval(expr, {"Fraction": Fraction}))
    add(cat, prompt, num(v))
    return v


def W(cat, prompt, *words):
    add(cat, prompt, *[r"\b" + re.escape(w.lower()) + r"\b" for w in words])


# ---- the 20 questions of yardstick 2.0 (kept so old results stay comparable) ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quality_yardstick as qy
for p, ans in qy.TASKS:
    add("v2 original", p, *ans)

# ---- arithmetic ----
C = "arithmetic"
for p, e in [("What is 48 * 25?", "48*25"), ("What is 7919 - 2468?", "7919-2468"), ("What is 156 * 12?", "156*12"),
             ("What is 9,000 divided by 45?", "9000/45"), ("What is 13 squared?", "13**2"), ("What is 3 to the power of 5?", "3**5"),
             ("What is 25% of 360?", "Fraction(25,100)*360"), ("What is 8.5 + 4.75?", "Fraction('8.5')+Fraction('4.75')"),
             ("What is 1000 - 387?", "1000-387"), ("What is the sum of the integers from 1 to 20?", "sum(range(1,21))"),
             ("What is 6 + 4 * 5 - 3?", "6+4*5-3"), ("What is (14 + 6) * (9 - 4)?", "(14+6)*(9-4)"),
             ("What is 2 to the power of 12?", "2**12"), ("What is 999 * 3?", "999*3"), ("What is the remainder when 100 is divided by 7?", "100%7"),
             ("What is the greatest common divisor of 84 and 36?", "__import__('math').gcd(84,36)"),
             ("What is the least common multiple of 6 and 8?", "__import__('math').lcm(6,8)")]:
    N(C, p, e)
for p, e in [("A train leaves at 9:40 and arrives at 12:15 the same day. How many minutes is the journey?", "(12*60+15)-(9*60+40)"),
             ("A bakery makes 36 rolls per tray and fills 14 trays. It sells 400 rolls. How many are left?", "36*14-400"),
             ("Maya buys 3 notebooks at 4 dollars each and 2 pens at 1.50 dollars each. She pays with a 20 dollar bill. How many dollars of change?", "20-(3*4+2*Fraction(3,2))"),
             ("A tank holds 240 litres and is 3/8 full. How many litres are in it?", "240*Fraction(3,8)"),
             ("A car drives 60 km/h for 2.5 hours, then 80 km/h for 1.5 hours. How many km in total?", "60*Fraction(5,2)+80*Fraction(3,2)"),
             ("Seven friends split a bill of 245 dollars equally. How many dollars does each pay?", "245/7"),
             ("A rectangle is 18 cm long and 7 cm wide. What is its perimeter in cm?", "2*(18+7)"),
             ("A rectangle is 18 cm long and 7 cm wide. What is its area in square cm?", "18*7"),
             ("A jacket costs 120 dollars and is discounted by 30%. What is the sale price in dollars?", "120*Fraction(70,100)"),
             ("A recipe for 4 people uses 300 g of rice. How many grams are needed for 10 people?", "300*10/4"),
             ("Sam saves 15 dollars a week. After how many weeks will Sam have saved 240 dollars?", "240/15"),
             ("A class has 28 students; 3/7 of them are girls. How many boys are in the class?", "28-28*Fraction(3,7)"),
             ("Three consecutive integers add up to 72. What is the largest?", "[n + 2 for n in range(-100, 100) if n + n + 1 + n + 2 == 72][0]"),
             ("The average of four numbers is 18. Three of them are 10, 20 and 30. What is the fourth?", "4*18-60"),
             ("A phone battery drops 4% every hour. Starting at 100%, what percent is left after 9 hours?", "100-4*9"),
             ("A pool fills at 12 litres per minute. How many minutes does it take to put 1,500 litres in it?", "1500/12"),
             ("Tickets cost 12 dollars for adults and 7 dollars for children. What do 3 adults and 5 children pay in dollars?", "3*12+5*7"),
             ("If 5 machines make 5 parts in 5 minutes, how many minutes do 100 machines need to make 100 parts?", "5"),
             ("A number is doubled and then increased by 9, giving 51. What is the number?", "(51-9)/2")]:
    N(C, p, e)

# ---- unit conversions ----
C = "units"
for p, e in [("Convert 68 degrees Fahrenheit to Celsius.", "(68-32)*Fraction(5,9)"), ("Convert 37 degrees Celsius to Fahrenheit.", "37*Fraction(9,5)+32"),
             ("How many metres are in 5 kilometres?", "5*1000"), ("How many seconds are in 7 minutes?", "7*60"),
             ("How many minutes are in 2 hours and 15 minutes?", "2*60+15"), ("How many grams are in 3.2 kilograms?", "Fraction('3.2')*1000"),
             ("How many litres are in 250 millilitres?", "Fraction(250,1000)"), ("How many inches are in 5 feet (12 inches per foot)?", "5*12"),
             ("Convert 12 inches to centimetres (1 inch = 2.54 cm).", "12*Fraction('2.54')"), ("Convert 90 km/h to metres per second.", "Fraction(90*1000,3600)"),
             ("How many megabytes are in 4.5 gigabytes (1 GB = 1000 MB)?", "Fraction('4.5')*1000"),
             ("How many feet are in 12 yards (3 feet per yard)?", "12*3"), ("How many ounces are in 3 pounds (16 ounces per pound)?", "3*16"),
             ("How many hours are in 3 days?", "3*24"), ("How many months are in 7.5 years?", "Fraction('7.5')*12"),
             ("How many millimetres are in 0.35 metres?", "Fraction('0.35')*1000"), ("How many seconds are in one day?", "24*3600"),
             ("A recipe needs 1.5 litres of milk. How many cups is that if a cup is 250 ml?", "Fraction('1.5')*1000/250")]:
    N(C, p, e)

# ---- code output (snippets are executed to get the answer) ----
C = "code output"
SNIPPETS = [
    "print(sum(range(1, 11)))",
    "x = [3, 1, 2]\nx.sort()\nprint(x[0], x[-1])",
    "print(len('hello world'))",
    "print('abc' * 3)",
    "d = {'a': 1, 'b': 2}\nd['c'] = d['a'] + d['b']\nprint(sorted(d.items())[-1])",
    "print(10 // 3, 10 % 3)",
    "print([i * i for i in range(5)])",
    "s = 0\nfor i in range(1, 6):\n    if i % 2:\n        s += i\nprint(s)",
    "print('banana'.count('a'))",
    "a = [1, 2, 3]\nb = a\nb.append(4)\nprint(len(a))",
    "print(' '.join(w[0] for w in 'north east south west'.split()))",
    "def f(n):\n    return 1 if n < 2 else n * f(n - 1)\nprint(f(5))",
    "print(max('apple', 'pear', 'fig', key=len))",
    "print(7 / 2, 7 // 2)",
    "print(bool([]), bool([0]))",
    "t = (1, 2, 3)\nprint(t[1:])",
    "print('Hello'[::-1])",
    "x = 5\nwhile x > 0:\n    x -= 2\nprint(x)",
    "print(sorted({3, 1, 3, 2, 1}))",
    "print(int('12') + int('30'))",
    "print(list(range(10, 0, -3)))",
    "print(round(2.675, 1), abs(-4))",
    "def g(a, b=2):\n    return a ** b\nprint(g(3), g(2, 5))",
    "print(len(set('mississippi')))",
    "words = ['pear', 'fig', 'apple']\nprint(sorted(words, key=len))",
]
for s in SNIPPETS:
    r = subprocess.run([sys.executable, "-I", "-c", s], capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, (s, r.stderr)
    out = r.stdout.strip()
    pat = r"\s+".join(re.escape(tok) for tok in out.lower().split())
    # accept the output as a whole token run; a bare-number output must not sit inside a longer number
    pat = (NUM_L + pat + NUM_R) if re.fullmatch(r"\d+", out) else pat
    add(C, "What exactly does this Python program print?\n```python\n" + s + "\n```", pat)

# ---- string manipulation ----
C = "strings"
for p, e in [("Write the word 'stressed' backwards.", "'stressed'[::-1]"), ("Write the word 'planet' with its letters in alphabetical order.", "''.join(sorted('planet'))"),
             ("Remove all vowels (a, e, i, o, u) from the word 'education'.", "''.join(c for c in 'education' if c not in 'aeiou')"),
             ("Write the word 'notebook' backwards.", "'notebook'[::-1]"),
             ("What is the third letter of the word 'keyboard'?", "'keyboard'[2]"), ("What is the last letter of the word 'mountain'?", "'mountain'[-1]"),
             ("Write the first letters of the words in 'portable network graphics format' joined together.", "''.join(w[0] for w in 'portable network graphics format'.split())"),
             ("Replace every 'a' with 'o' in the word 'banana'.", "'banana'.replace('a','o')"),
             ("Take the word 'ornament', remove its first and last letters.", "'ornament'[1:-1]"),
             ("Write the word 'garden' twice with a hyphen between the copies.", "'garden-garden'")]:
    v = eval(e)
    add(C, p, r"(?<![a-z])" + re.escape(v) + r"(?![a-z])")
for p, e in [("How many times does the letter 'r' appear in the word 'strawberry'?", "'strawberry'.count('r')"),
             ("How many vowels (a, e, i, o, u) are in the word 'encyclopedia'?", "sum(c in 'aeiou' for c in 'encyclopedia')"),
             ("How many letters are in the word 'extraordinary'?", "len('extraordinary')"),
             ("How many words are in the sentence 'the quick brown fox jumps over the lazy dog'?", "len('the quick brown fox jumps over the lazy dog'.split())"),
             ("How many times does the letter 'e' appear in 'independence'?", "'independence'.count('e')"),
             ("How many distinct letters are in the word 'bookkeeper'?", "len(set('bookkeeper'))")]:
    N(C, p, e)
for w in ("level", "planet"):
    add(C, f"Is the word '{w}' a palindrome? Answer yes or no.", r"\byes\b" if w == w[::-1] else r"\bno\b")
add(C, "Which word comes first in alphabetical order: 'mango', 'melon', 'maple' or 'marble'?", r"\b" + min("mango melon maple marble".split()) + r"\b")

# ---- logic ----
C = "logic"


def order_puzzle(names, cons, ask, want):
    """Brute-force a left-to-right order; every constraint is a lambda over a position dict. Requires a unique solution."""
    sols = [p for p in itertools.permutations(names) if all(c({n: i for i, n in enumerate(p)}) for c in cons)]
    assert len(sols) == 1, sols
    return sols[0]


s = order_puzzle("ABCD", [lambda p: p["A"] < p["B"], lambda p: p["C"] < p["A"], lambda p: p["B"] < p["D"]], 0, 0)
# C A B D (oldest to youngest if positions are age rank)
assert s == ("C", "A", "B", "D")
W(C, "Four runners finish a race. Ana finishes before Ben. Cy finishes before Ana. Ben finishes before Dee. Who finishes third?", "ben")
W(C, "Four runners finish a race. Ana finishes before Ben. Cy finishes before Ana. Ben finishes before Dee. Who finishes last?", "dee")
s = order_puzzle("ABCDE", [lambda p: p["A"] < p["B"], lambda p: p["B"] < p["C"], lambda p: p["E"] < p["A"], lambda p: p["D"] > p["C"]], 0, 0)
assert s == ("E", "A", "B", "C", "D")
W(C, "Five books are on a shelf in a row. Red is left of Green. Green is left of Blue. Black is left of Red. Pink is right of Blue. Which book is in the middle?", "green")
for stmt, want in [("All bloops are razzies. All razzies are lazzies. Are all bloops lazzies?", "yes"),
                   ("All cats are animals. Some animals are fast. Does it follow that some cats are fast?", "no"),
                   ("No fish can fly. A salmon is a fish. Can a salmon fly?", "no"),
                   ("If it rains, the street is wet. The street is not wet. Did it rain?", "no")]:
    add(C, stmt + " Answer yes or no.", r"\b" + want + r"\b")


def knights(a_says, b_says):
    """Brute force: knights always tell the truth, knaves always lie. Return the sets of (A,B) types consistent with the statements."""
    out = []
    for A, B in itertools.product([True, False], repeat=2):
        if (a_says(A, B) == A) and (b_says(A, B) == B):
            out.append((A, B))
    return out


# A: "B is a knave."  B: "A and I are the same type."
sol = knights(lambda A, B: not B, lambda A, B: A == B)
assert sol == [(True, False)] or len(sol) == 1, sol
add(C, "On an island, knights always tell the truth and knaves always lie. A says: 'B is a knave.' B says: 'A and I are the same type.' What is A: knight or knave?",
    r"\bknight\b" if sol[0][0] else r"\bknave\b")
# A: "We are both knaves."
sol = knights(lambda A, B: (not A) and (not B), lambda A, B: B)
add(C, "On an island, knights always tell the truth and knaves always lie. A says: 'We are both knaves.' (B says nothing.) What is A: knight or knave?",
    r"\bknight\b" if sol[0][0] else r"\bknave\b")
assert sol == [(False, True)], sol
N(C, "A farmer has 17 sheep. All but 9 run away. How many sheep are left?", "9")
N(C, "How many months of the year have 28 days (at least)?", "12")
N(C, "What is the next number in the sequence 3, 6, 11, 18, 27, ...?", "27+11")
N(C, "What is the next number in the sequence 1, 1, 2, 3, 5, 8, 13, ...?", "21")
N(C, "What is the next number in the sequence 100, 90, 81, 73, ...? (the differences shrink by 1 each step)", "73-7")
W(C, "Monday, Wednesday, Friday, ... what is the next day of the week in this pattern?", "sunday")
pets = [p for p in itertools.permutations(["cat", "dog", "fish"]) if p[0] != "cat" and p[1] == "dog"]
assert len(pets) == 1 and pets[0][2] == "cat"
W(C, "Alice, Bob and Carol each own exactly one pet, all different: a cat, a dog or a fish. Alice is allergic to fur, and Bob's pet barks. Who owns the cat?", "carol")
N(C, "Two fathers and two sons go fishing and each catches one fish, for exactly 3 fish in total (nobody is left out). How many people are fishing?", "3")

# ---- dates ----
C = "dates"
D = dt.date
for d in [D(2024, 3, 1), D(2023, 12, 25), D(2026, 2, 14), D(2000, 1, 1), D(2025, 7, 4)]:
    W(C, f"What day of the week is {d.isoformat()}? Answer with the weekday name.", d.strftime("%A"))
for a, b in [(D(2024, 1, 15), D(2024, 3, 10)), (D(2023, 11, 20), D(2024, 1, 5)), (D(2025, 2, 1), D(2025, 3, 1))]:
    N(C, f"How many days are there from {a.isoformat()} to {b.isoformat()}?", str((b - a).days))
for d, n in [(D(2024, 2, 20), 15), (D(2025, 12, 28), 10), (D(2023, 1, 31), 30), (D(2024, 12, 1), 61)]:
    r = d + dt.timedelta(days=n)
    add(C, f"What date is {n} days after {d.isoformat()}? Answer in YYYY-MM-DD format.", re.escape(r.isoformat()))
for y in (1900, 2100, 2024, 2023):
    leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
    assert leap == __import__("calendar").isleap(y)
    add(C, f"Is {y} a leap year? Answer yes or no.", r"\byes\b" if leap else r"\bno\b")
N(C, "How many days are in February 2100?", "28")
W(C, "If today is Wednesday, which weekday is it 10 days from now? Answer with the weekday name.", "saturday")
assert ("mon tue wed thu fri sat sun".split()[(2 + 10) % 7]) == "sat"
add(C, "A meeting starts at 14:35 and lasts 1 hour 50 minutes. At what time does it end? Answer as HH:MM.", r"(?<!\d)16:25(?!\d)")
add(C, "What month comes 7 months after October? Answer with the month name.", r"\bmay\b")

# ---- facts with one stable short answer ----
C = "facts"
for p, ans in [("What is the capital of Japan?", "Tokyo"), ("What is the chemical symbol for sodium?", "Na"), ("What is the largest planet in the Solar System?", "Jupiter"),
               ("Who wrote the novel Pride and Prejudice?", "Austen"), ("What is the capital of Egypt?", "Cairo"), ("What is the chemical symbol for iron?", "Fe"),
               ("What is the capital of Brazil?", "Brasilia|Brasília"), ("What is the capital of Turkey?", "Ankara"), ("What is the capital of Switzerland?", "Bern"),
               ("What is the capital of Kenya?", "Nairobi"), ("Which is the largest ocean on Earth?", "Pacific"), ("What is the currency of Japan?", "yen"),
               ("Who painted the Mona Lisa?", "Leonardo|da Vinci"), ("What gas do plants absorb from the air for photosynthesis?", "carbon dioxide|CO2"),
               ("Which planet is known as the Red Planet?", "Mars"), ("What is the chemical formula of table salt?", "NaCl"),
               ("Which organelle is called the powerhouse of the cell?", "mitochondri"), ("What is the hardest natural substance?", "diamond"),
               ("Which gas makes up most of Earth's atmosphere?", "nitrogen"), ("What is the chemical symbol for silver?", "Ag"),
               ("Which mountain range contains Mount Everest?", "Himalaya"), ("What is the chemical symbol for potassium?", "K"),
               ("Who wrote the dystopian novel Nineteen Eighty-Four?", "Orwell"), ("What is the largest mammal?", "blue whale"),
               ("What is the capital of Italy?", "Rome")]:
    add(C, p, r"\b(" + "|".join(re.escape(a.lower()) for a in ans.split("|")) + r")" + ("" if ans in ("mitochondri", "Himalaya") else r"\b"))
for p, v in [("In which year did the Berlin Wall fall?", 1989), ("In which year did Apollo 11 land humans on the Moon?", 1969), ("How many chromosomes does a typical human cell have?", 46),
             ("How many bones does an adult human have?", 206), ("What is the atomic number of carbon?", 6), ("How many strings does a standard guitar have?", 6),
             ("How many sides does an octagon have?", 8), ("How many continents are there on Earth by the usual seven-continent count?", 7),
             ("At what temperature in degrees Fahrenheit does water freeze at sea level?", 32), ("How many letters are in the English alphabet?", 26),
             ("What is the smallest prime number?", 2), ("In which year did the French Revolution begin?", 1789)]:
    N(C, p, str(v))
add(C, "What is 255 in hexadecimal? Use lowercase letters, no prefix.", r"(?<![0-9a-f])ff(?![0-9a-f])")
add(C, "What is the decimal number 37 written in binary?", r"(?<![01])" + bin(37)[2:] + r"(?![01])")
add(C, "What is the number 49 written as a Roman numeral?", r"\bxlix\b")
assert bin(37)[2:] == "100101" and hex(255) == "0xff"
add(C, "What is the binary number 101101 in decimal?", num(int("101101", 2)))

# ---- tool-call style JSON ----


def toolcall(name, args, request, spec):
    look = "".join(f'(?=.*"{k}"\\s*:\\s*{v})' for k, v in args)
    prompt = (f"You can call this tool: {spec}\n{request}\nReply with only a JSON object of the form "
              '{"name": ..., "arguments": {...}} and nothing else.')
    add("tool call", prompt, r"(?s)^" + f'(?=.*"name"\\s*:\\s*"{name}")' + look + ".*")


toolcall("get_weather", [("city", '"paris"'), ("unit", '"celsius"')], "What is the weather in Paris right now, in celsius?",
         'get_weather(city: string, unit: "celsius" or "fahrenheit")')
toolcall("add_numbers", [("a", "17"), ("b", "25")], "Add 17 and 25 using the tool.", "add_numbers(a: number, b: number)")
toolcall("set_timer", [("minutes", "20")], "Set a timer for twenty minutes.", "set_timer(minutes: integer)")
toolcall("send_email", [("to", '"sam@example\\.com"'), ("subject", '"lunch"')], "Email sam@example.com with the subject lunch.",
         "send_email(to: string, subject: string)")
toolcall("search_files", [("query", '"invoice"'), ("limit", "5")], "Find at most 5 files matching the word invoice.", "search_files(query: string, limit: integer)")
toolcall("convert_currency", [("amount", "100"), ("from", '"usd"'), ("to", '"eur"')], "Convert 100 US dollars (USD) to euros (EUR).",
         'convert_currency(amount: number, from: string (ISO code), to: string (ISO code))')


if __name__ == "__main__":
    seen = set()
    items = []
    for cat, prompt, regs in tasks:
        assert prompt not in seen, prompt
        seen.add(prompt)
        for r in regs:
            re.compile(r)
        items.append({"kind": "task", "category": cat, "prompt": prompt, "answers": regs})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    from collections import Counter
    c = Counter(i["category"] for i in items)
    print(f"{len(items)} tasks -> {OUT}")
    for k, v in c.items():
        print(f"  {k:14s} {v}")
