#!/usr/bin/env python3
"""Turn docs/benchmark_card.html into Markdown (tables, lists, claim markers kept) and put it into docs/RESULTS_CURRENT.md between
`<!-- card:begin -->` and `<!-- card:end -->`, so GitHub shows the whole card, not only an image of its top. The README rule: numbers, card, PNGs and README
change in one step.

    python3 tools/card_markdown.py            # print the Markdown
    python3 tools/card_markdown.py --write    # rewrite the block in README.md
    python3 tools/card_markdown.py --check    # exit 1 if README.md's block differs from the card (the quick tests run this)

Python standard library only. The decode chart (an SVG drawn by script from the `data` list at the end of the card) becomes a table with the same numbers
and the same test details."""
import json
import os
import re
import sys
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CARD = os.path.join(ROOT, "docs", "benchmark_card.html")
README = os.path.join(ROOT, "docs", "RESULTS_CURRENT.md")
BEGIN, END = "<!-- card:begin -->", "<!-- card:end -->"
VOID = {"br", "meta", "link", "img", "hr", "input"}


class Node:
    def __init__(self, tag, attrs=None):
        self.tag, self.attrs, self.kids = tag, dict(attrs or {}), []

    def cls(self):
        return self.attrs.get("class", "")


class Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs)
        self.stack[-1].kids.append(n)
        if tag not in VOID:
            self.stack.append(n)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].kids.append(data)

    def handle_comment(self, data):
        self.stack[-1].kids.append(("comment", data))


def find(n, pred):
    for k in n.kids:
        if isinstance(k, Node):
            if pred(k):
                yield k
            yield from find(k, pred)


def inline(n):
    """inline Markdown of a node's children"""
    out = []
    for k in n.kids:
        if isinstance(k, str):
            out.append(re.sub(r"\s+", " ", k))
        elif isinstance(k, tuple):
            m = re.match(r"\s*claim:(\S+)\s*$", k[1])
            if m:
                out.append("<!-- claim:%s -->" % m.group(1))
        elif k.tag in ("b", "strong"):
            t = inline(k).strip()
            out.append("**%s**" % t if t else "")
        elif k.tag == "code":
            out.append("`%s`" % inline(k))
        elif k.tag == "br":
            out.append("<br>")
        elif k.tag == "a":
            out.append("[%s](%s)" % (inline(k).strip(), k.attrs.get("href", "")))
        elif k.tag == "span" and k.cls() == "sub":
            out.append("<br><sub>%s</sub>" % inline(k).strip())
        elif k.tag in ("script", "style", "svg"):
            pass
        else:
            out.append(inline(k))
    s = "".join(out)
    return re.sub(r"[ \t]+", " ", s)


def cell(n):
    return inline(n).strip().replace("|", "\\|") or " "


def table(n):
    head = next(find(n, lambda k: k.tag == "thead"), None)
    body = next(find(n, lambda k: k.tag == "tbody"), None)
    heads = [cell(c).replace("<br><sub>", " <sub>").replace("<br>", " ") for c in find(head, lambda k: k.tag == "th")]
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    pending = None
    for tr in [k for k in body.kids if isinstance(k, Node) or isinstance(k, tuple)]:
        if isinstance(tr, tuple):                      # a claim comment after </tr> belongs to the row just written
            m = re.match(r"\s*claim:(\S+)\s*$", tr[1])
            if m and len(lines) > 2:
                lines[-1] = lines[-1][:-2] + " <!-- claim:%s --> |" % m.group(1)
            continue
        cells = [cell(c) for c in tr.kids if isinstance(c, Node) and c.tag in ("td", "th")]
        cells += [" "] * (len(heads) - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def blocks(n, h=3):
    out = []
    for k in n.kids:
        if not isinstance(k, Node):
            continue
        t, c = k.tag, k.cls()
        if t == "h2":
            out.append("\n" + "#" * h + " " + inline(k).strip() + "\n")
        elif t == "h3":
            out.append("**%s**\n" % inline(k).strip())
        elif t == "p":
            out.append(inline(k).strip() + "\n")
        elif t == "ul":
            out.append("\n".join("- " + inline(li).strip() for li in k.kids if isinstance(li, Node) and li.tag == "li") + "\n")
        elif t == "pre":
            out.append("```\n" + "".join(x for x in k.kids if isinstance(x, str)).strip("\n") + "\n```\n")
        elif t == "table":
            out.append(table(k) + "\n")
        elif t in ("script", "style", "svg", "footer", "header", "head"):
            continue
        elif t == "div" and c == "chart":
            continue
        elif t == "div" and "tiles" in c:
            rows = ["| Headline | What it is |", "|---|---|"]
            for tile in k.kids:
                if isinstance(tile, Node):
                    b = next(find(tile, lambda x: x.tag == "b"))
                    sp = next(find(tile, lambda x: x.tag == "span"))
                    rows.append("| %s | %s |" % (inline(b).strip(), inline(sp).strip().replace("|", "\\|")))
            out.append("\n".join(rows) + "\n")
        else:
            out.extend(blocks(k, h))
    return out


def render():
    src = open(CARD, encoding="utf-8").read()
    data = json.loads(re.search(r"var data = (\[.*?\n  \]);", src, flags=re.S).group(1))
    tree = Tree()
    tree.feed(src[src.find('<div class="wrap">'):])
    header = next(find(tree.root, lambda k: k.tag == "header"))
    eyebrow = inline(next(find(header, lambda k: k.cls() == "eyebrow"))).strip()
    lede = inline(next(find(header, lambda k: k.cls() == "lede"))).strip()
    machine = [inline(s).strip() for s in find(next(find(header, lambda k: k.cls() == "machine")), lambda k: k.tag == "span")]
    parts = ["**%s**\n" % eyebrow, lede + "\n",
             "| Machine | |\n|---|---|\n" + "\n".join("| %s | %s |" % (a, b) for a, b in zip(["CPU", "GPU", "RAM", "OS", "llama.cpp builds"], machine)) + "\n"]
    after_decode = False
    for b in blocks(tree.root):
        if after_decode and b.strip().startswith("### "):      # the details of the decode rows close the decode section
            parts.append(details_table(data))
            after_decode = False
        parts.append(b)
        if b.strip().startswith("### Decode speed"):
            after_decode = True
    return "\n".join(parts)


def details_table(data):
    lines = ["Test details for each row (the card draws the decode table as bars; same numbers):\n",
             "| Model | Size, drive | stock tok/s | moe-cache tok/s | with MTP tok/s | Test details |", "|---|---|---|---|---|---|"]
    for d in data:
        lines.append("| %s | %s | %.2f | %.2f | %s | %s |" % (d["m"], d["s"], d["stock"], d["mc"], ("%.2f" % d["mtp"]) if d.get("mtp") else "–",
                                                          "<br>".join(x.replace("|", "\\|") for x in d["d"])))
    return "\n".join(lines) + "\n"


def block():
    text = re.sub(r"\n{3,}", "\n\n", render()).strip()
    return BEGIN + "\n\n" + text + "\n\n" + END


def main():
    md = block()
    if "--write" in sys.argv or "--check" in sys.argv:
        text = open(README, encoding="utf-8").read()
        m = re.search(re.escape(BEGIN) + r".*?" + re.escape(END), text, flags=re.S)
        if not m:
            print("docs/RESULTS_CURRENT.md has no %s ... %s block" % (BEGIN, END))
            sys.exit(1)
        if "--check" in sys.argv:
            if m.group(0) != md:
                print("docs/RESULTS_CURRENT.md's card block differs from docs/benchmark_card.html: run python3 tools/card_markdown.py --write")
                sys.exit(1)
            print("docs/RESULTS_CURRENT.md's card block matches the card")
            return
        open(README, "w", encoding="utf-8", newline="\n").write(text[:m.start()] + md + text[m.end():])
        print("docs/RESULTS_CURRENT.md card block written (%d lines)" % md.count("\n"))
        return
    print(md)


if __name__ == "__main__":
    main()
