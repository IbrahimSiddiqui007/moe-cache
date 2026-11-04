#!/usr/bin/env python3
"""A name used inside a function of bin/moe_cache.py that is neither local, nor a parameter, nor defined at module level, nor a builtin would raise NameError only
on the path that reaches it. ENGINE-029 moves code into stage functions; this finds a variable that was left behind in the old function, on every path, without
running them. No pyflakes needed: the compiler's symbol table tells which names a function takes from outside.   python3 tests/undefined_names_test.py"""
import builtins, os, symtable, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad = []
for rel in ("bin/moe_cache.py", "bin/moe_cache_ui.py", "bin/moe_cache_on_demand.py", "bin/moe_cache_launch.py"):
    path = os.path.join(HERE, rel)
    src = open(path, encoding="utf-8").read()
    top = symtable.symtable(src, path, "exec")
    module_names = set(top.get_identifiers()) | set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    def walk(t):
        for c in t.get_children():
            if c.get_type() == "function" or c.get_type() == "class":
                if c.get_type() == "function":
                    for s in c.get_symbols():
                        if s.is_global() and s.is_referenced() and s.get_name() not in module_names:
                            bad.append(f"{rel}: {c.get_name()}() uses '{s.get_name()}', which is not defined anywhere it can see")
                walk(c)
    walk(top)
for b in bad:
    print("  FAIL " + b)
print(("  ok   " if not bad else "  FAIL ") + "no function uses a name that does not exist (bin/moe_cache*.py)")
sys.exit(1 if bad else 0)
