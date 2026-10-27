#!/usr/bin/env python3
"""Hostile GGUF headers (audit 2026-10-27): the header readers the launcher and the UI run on every model file they list must refuse lengths that
point past the file, nested arrays and odd alignments without allocating a model-sized buffer, and must not put a path-like architecture name into
a profile file name. No model needed: the files are a few hundred bytes (one is made sparse to look large).
  python3 tests/gguf_hostile_test.py"""
import os, resource, struct, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "planner"))
import gguf_plan  # noqa: E402

fails = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv_str(k, v):
    return s(k) + struct.pack("<I", 8) + s(v)


def kv_u32(k, v):
    return s(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def gguf(kvs, n_tensors=0, tensors=b"", pad=0):
    h = b"GGUF" + struct.pack("<IQQ", 3, n_tensors, len(kvs)) + b"".join(kvs) + tensors
    return h + b"\0" * pad


tmp = tempfile.mkdtemp(prefix="gguf-hostile-")


def write(name, data, sparse_to=0):
    p = os.path.join(tmp, name)
    with open(p, "wb") as f:
        f.write(data)
        if sparse_to:
            f.truncate(sparse_to)       # looks like a big model file, takes no disk space
    return p


def refused(fn, path):
    try:
        fn(path)
    except (ValueError, struct.error):
        return True
    except Exception as e:  # noqa: BLE001
        print("    unexpected", type(e).__name__, e)
        return False
    return False


rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
big = 8 << 30   # 8 GiB sparse: an unbounded read of "the rest of the file" would need that much RAM

# a key length of almost the whole (sparse) file
p = write("bigstr.gguf", b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", big - 64), big)
check(refused(gguf_plan.read_tensors, p) and refused(gguf_plan.model_key, p),
      "a string length near the file size is refused (read_tensors, model_key)")

# a uint8 array as long as the file, then a string array as long as the file
arr = s("x.arr") + struct.pack("<I", 9) + struct.pack("<IQ", 0, big + 100)
p = write("bigarr.gguf", gguf([arr]), big)
check(refused(gguf_plan.read_tensors, p), "a byte array that runs past the end of the file is refused")
arr = s("x.arr") + struct.pack("<I", 9) + struct.pack("<IQ", 0, big - 200)
p = write("inarr.gguf", gguf([arr]), big)
check(gguf_plan.read_tensors(p) == [], "a byte array as long as the file (valid) is skipped, not read")
arr = s("x.arr") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 1 << 60)
p = write("strarr.gguf", gguf([arr]))
check(refused(gguf_plan.read_tensors, p), "a string array with 2^60 entries is refused at once")
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
check(rss1 - rss0 < 200 * 1024, f"no big allocation while refusing them (max RSS grew {(rss1 - rss0) // 1024} MB)")

# arrays of arrays: no writer makes them, the old reader recursed as deep as the file said
nested = s("x.nest") + struct.pack("<I", 9) + b"".join(struct.pack("<IQ", 9, 1) for _ in range(5000)) + struct.pack("<IQ", 4, 0)
p = write("nested.gguf", gguf([nested]))
check(refused(gguf_plan.read_tensors, p), "nested arrays are refused (no RecursionError)")

# alignment 0 divided by zero
p = write("align0.gguf", gguf([kv_u32("general.alignment", 0)]))
check(refused(gguf_plan.read_tensors, p), "general.alignment = 0 is refused")
p = write("align3.gguf", gguf([kv_u32("general.alignment", 3)]))
check(refused(gguf_plan.read_tensors, p), "a non-power-of-two alignment is refused")

# too many tensor dimensions
t = s("blk.0.ffn_up_exps.weight") + struct.pack("<I", 1 << 30)
p = write("dims.gguf", gguf([], 1, t))
check(refused(gguf_plan.read_tensors, p), "a tensor with 2^30 dimensions is refused")

# the architecture becomes part of a profile file name
for arch, want in (("../../etc/x", None), ("/tmp/x", None), ("a\\b", None), ("x..y", None), ("qwen3moe", "qwen3moe-48x128"), ("gpt-oss", "gpt-oss-48x128")):
    p = write("arch.gguf", gguf([kv_str("general.architecture", arch), kv_u32(f"{arch}.block_count", 48), kv_u32(f"{arch}.expert_count", 128)]))
    got = gguf_plan.model_key(p)
    check(got == want, f"model_key with architecture {arch!r}: {got!r}")

# the planner's own reader (used when the gguf package is missing or fails): same files, clean error, small memory
plan = os.path.join(HERE, "..", "planner", "plan.py")
for name in ("bigstr.gguf", "bigarr.gguf", "strarr.gguf", "nested.gguf", "align0.gguf", "dims.gguf"):
    r = subprocess.run([sys.executable, "-c", "import resource, runpy, struct, sys; sys.argv = sys.argv[1:]\n"
                        "try:\n runpy.run_path(sys.argv[0], run_name='__main__')\n"
                        "except (ValueError, struct.error) as e: print('refused:', type(e).__name__, e)\n"
                        "except BaseException as e: print('other:', type(e).__name__, e)\n"
                        "print('maxrss', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)",
                        plan, os.path.join(tmp, name), "--json"],
                       capture_output=True, text=True, timeout=120, env=dict(os.environ, MOE_PLAN_MINI="1"))
    out = r.stdout + r.stderr
    rss = int(out.rsplit("maxrss", 1)[1].split()[0]) if "maxrss" in out else 1 << 30
    check("refused:" in out and rss < 300 * 1024,
          f"plan.py (built-in reader) refuses {name} with {rss // 1024} MB ({out.strip().splitlines()[0][:90] if out.strip() else r.returncode})")

for f in os.listdir(tmp):
    os.unlink(os.path.join(tmp, f))
os.rmdir(tmp)
print("gguf hostile test:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
