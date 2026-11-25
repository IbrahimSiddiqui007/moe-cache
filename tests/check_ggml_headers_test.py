#!/usr/bin/env python3
"""TEST-1235 audit row 26: tools/check_ggml_headers.py exits 1 for a header that is missing at the commit and for lines inserted inside one of our structs (a layout change);
a new declaration between structs stays a safe addition (exit 0). Temporary git repo, no llama.cpp checkout, no network."""
import os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(HERE, "tools", "check_ggml_headers.py")
fails = 0
def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what); fails += 0 if ok else 1
BASE = "struct a {\n    int x;\n    int y;\n};\n\nint f(void);\n"
def run(upstream, extra_vendored=()):
    with tempfile.TemporaryDirectory() as d:
        tree, vend = os.path.join(d, "tree"), os.path.join(d, "vend")
        os.makedirs(os.path.join(tree, "ggml/include")); os.makedirs(os.path.join(vend, "include")); os.makedirs(os.path.join(vend, "src"))
        open(os.path.join(tree, "ggml/include/ggml.h"), "w").write(upstream)
        open(os.path.join(vend, "include/ggml.h"), "w").write(BASE)
        for f in extra_vendored:
            open(os.path.join(vend, "include", f), "w").write(BASE)
        g = lambda *a: subprocess.run(["git", "-C", tree, "-c", "user.email=t@t", "-c", "user.name=t", *a], capture_output=True, text=True, check=True)
        g("init", "-q"); g("add", "-A"); g("commit", "-qm", "x")
        r = subprocess.run([sys.executable, TOOL, "--tree", tree, "--commit", "HEAD", "--vendored", vend], capture_output=True, text=True)
        return r.returncode, r.stdout
rc, out = run(BASE); check(rc == 0, "identical header: exit 0")
rc, out = run(BASE + "int g(void);\n"); check(rc == 0 and "add " in out, "a new declaration after our lines: safe, exit 0")
rc, out = run(BASE.replace("    int y;\n", "    int y;\n    int z;\n") ); check(rc == 1 and "INSIDE" in out, f"a member inserted inside our struct: exit 1 ({out.strip().splitlines()[0][:60]})")
rc, out = run(BASE.replace("int x;", "long x;")); check(rc == 1 and "DIFF" in out, "a changed line: exit 1")
rc, out = run(BASE, extra_vendored=("ggml-gone.h",)); check(rc == 1 and "not available" in out, "a vendored header missing at the commit: exit 1 (was 0)")
print("check_ggml_headers: " + (f"{fails} failed" if fails else "all passed")); sys.exit(1 if fails else 0)
