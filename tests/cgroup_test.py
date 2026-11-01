"""Model-free test of the launcher's cgroup memory-limit reader (A3-06): a limit on a PARENT cgroup applies to a leaf that says "max".
Run: python3 tests/cgroup_test.py   (fake /proc/self/cgroup and /sys/fs/cgroup trees in a temp folder; no root, no model)"""
import os, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc

bad = 0
def check(cond, msg):
    global bad
    print(("  ok   " if cond else "  FAIL ") + msg)
    bad += 0 if cond else 1

def put(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)

with tempfile.TemporaryDirectory() as t:
    root = os.path.join(t, "cg"); proc = os.path.join(t, "proc")
    # v2: leaf says max, its parent slice has 8 GiB, a grandparent 20 GiB
    put(proc, "0::/user.slice/app.slice/run.scope\n")
    put(root + "/user.slice/app.slice/run.scope/memory.max", "max\n")
    put(root + "/user.slice/app.slice/memory.max", str(8 * 2**30) + "\n")
    put(root + "/user.slice/memory.max", str(20 * 2**30) + "\n")
    check(mc.cgroup_mem_limit(proc, root) == 8 * 2**30, "v2: the parent's 8 GiB limit applies to a leaf that says max")
    put(root + "/user.slice/app.slice/run.scope/memory.max", str(4 * 2**30) + "\n")
    check(mc.cgroup_mem_limit(proc, root) == 4 * 2**30, "v2: the leaf's own smaller limit wins")
    put(proc, "0::/\n")
    check(mc.cgroup_mem_limit(proc, root) is None, "v2: the root cgroup has no limit file: None")
    put(proc, "0::/nothing/here\n")
    check(mc.cgroup_mem_limit(proc, root) is None, "v2: a path with no files at all: None")
    # v1: "no limit" is a huge number
    put(proc, "4:memory:/docker/abc\n3:cpu,cpuacct:/docker/abc\n")
    put(root + "/memory/docker/abc/memory.limit_in_bytes", "9223372036854771712\n")
    put(root + "/memory/docker/memory.limit_in_bytes", str(6 * 2**30) + "\n")
    check(mc.cgroup_mem_limit(proc, root) == 6 * 2**30, "v1: the huge 'no limit' value is ignored, the parent's 6 GiB applies")
    check(mc.cgroup_mem_limit(os.path.join(t, "missing"), root) is None, "no /proc/self/cgroup (Windows, macOS): None")
print("cgroup_test: " + ("all checks passed" if not bad else f"{bad} FAILED"))
sys.exit(1 if bad else 0)
