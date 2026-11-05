#!/usr/bin/env python3
"""Wiring test for three default-on guards (Critic round 4, item 1). The unit tests of these guards (gpu_guard_test, gpu_fallback_test, lru_rule_test,
rebalance_test) test pure helpers, so they stay green when the guarded call at the REAL call site is deleted or bypassed. This test reads the real
sources (src/mc_compute.inl, src/mc_state_lru.inl, src/mc_state.inl, src/mc_state_rebalance.inl) and fails when:
  GPU_CATCH      gpu_segment() / hybrid_group() run outside mcguard::guarded(), or the guarded wrappers are no longer what the decode path calls,
                 or the wrapper no longer cleans up (gpu_failed_cleanup) and logs ("the GPU block threw" / "the hybrid GPU block threw")
  LRU rule       the sibling path in mc_state_lru.inl no longer asks mclru::refresh_recency_of_resident(cur_scan) before it moves a resident expert to MRU
  idle rebalance the idle thread no longer calls rb_choose() or the switch default is no longer on (MOE_CACHE_IDLE_REBALANCE=0 turns it off)
To prove that each check can fail, the test also applies the deletion / bypass to an in-memory copy of the source and requires the check to complain.
What this does NOT do: run the guard against a real GPU fault (a Vulkan fault-injection run with MOE_CACHE_GPU_CATCH=2 needs a CI job with Vulkan and the plugin).
   python3 tests/guard_wiring_test.py"""
import os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def read(rel):
    return open(os.path.join(HERE, rel), encoding="utf-8").read()
def strip_comments(t):
    return re.sub(r"//[^\n]*", "", t)
def body(t, header):
    """text of the function whose header line starts with `header`, up to the closing brace at column 0"""
    i = t.find(header)
    if i < 0:
        return ""
    j = t.find("\n}\n", i)
    return t[i:j + 3] if j > 0 else t[i:]

def check_gpu_catch(compute, backend, hybrid):
    p = []
    c = strip_comments(compute)
    seg, grp = body(c, "bool gpu_segment_guarded("), body(c, "bool hybrid_group_guarded(")
    for name, b, inner in (("gpu_segment_guarded", seg, "gpu_segment("), ("hybrid_group_guarded", grp, "hybrid_group(")):
        if not b:
            p.append(f"{name} is gone")
            continue
        if "mcguard::guarded(" not in b:
            p.append(f"{name} no longer goes through mcguard::guarded()")
        if inner not in b:
            p.append(f"{name} no longer calls {inner[:-1]}")
        if "gpu_failed_cleanup(c)" not in b:
            p.append(f"{name} no longer cleans up after a throw (gpu_failed_cleanup)")
        if "MOE_CACHE_GPU_CATCH" not in b:
            p.append(f"{name} no longer reads MOE_CACHE_GPU_CATCH")
    if "the GPU block threw" not in compute or "the hybrid GPU block threw" not in compute:
        p.append("the log lines of the caught throws are gone")
    # no other caller of the unguarded functions: remove the wrappers and the definitions, then look for calls
    rest = c.replace(seg, "").replace(grp, "")
    for fn, src in (("gpu_segment", strip_comments(backend)), ("hybrid_group", strip_comments(hybrid))):
        calls = [m.start() for m in re.finditer(r"(?<![\w:])" + fn + r"\(", rest)]
        if calls:
            p.append(f"{fn}() is called outside its guarded wrapper ({len(calls)} place(s) in mc_compute.inl)")
    if not re.search(r"hybrid_group_guarded\(c, s, cg, i, last\)", rest):
        p.append("the decode path does not call hybrid_group_guarded()")
    if len(re.findall(r"gpu_segment_guarded\(", rest)) < 1:
        p.append("the decode path does not call gpu_segment_guarded()")
    return p

def check_lru(lru):
    c = strip_comments(lru)
    m = re.search(r"if \(mclru::refresh_recency_of_resident\(cur_scan\)\) \{\s*unlink\(s\);\s*push_mru\(s\);\s*\}", c)
    return [] if m else ["the sibling path no longer asks mclru::refresh_recency_of_resident(cur_scan) before unlink()/push_mru() of a resident expert"]

def check_rebalance(state, rebal):
    p = []
    if not re.search(r"bool\s+rb_on\s*=\s*true;", state):
        p.append("idle rebalance is no longer on by default (rb_on)")
    if not re.search(r'getenv\("MOE_CACHE_IDLE_REBALANCE"\)\)\s*\{\s*rb_on = strcmp\(v, "0"\) != 0;', state):
        p.append("MOE_CACHE_IDLE_REBALANCE=0 no longer turns it off")
    if "rb_choose(" not in strip_comments(rebal):
        p.append("the idle thread no longer calls rb_choose()")
    if not re.search(r"rb_stop \|\| graph_active > 0 \|\| graph_seq != seq", strip_comments(rebal)):
        p.append("the idle rebalance no longer stops when a graph starts")
    return p

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

compute, backend, hybrid = read("src/mc_compute.inl"), read("src/mc_backend.inl"), read("src/mc_hybrid.inl")
lru, rebal = read("src/mc_state_lru.inl"), read("src/mc_state_rebalance.inl")
# the state struct's members live in src/mc_state_*.h since cleanup-002; read them together with mc_state.inl
state = read("src/mc_state.inl") + "".join("\n" + read("src/" + h) for h in sorted(os.listdir(os.path.join(HERE, "src"))) if h.startswith("mc_state_") and h.endswith(".h"))

check(check_gpu_catch(compute, backend, hybrid) == [], f"GPU_CATCH: both GPU blocks run inside mcguard::guarded() ({check_gpu_catch(compute, backend, hybrid)})")
check(check_lru(lru) == [], f"LRU rule: the sibling path asks the rule ({check_lru(lru)})")
check(check_rebalance(state, rebal) == [], f"idle rebalance: on by default, rb_choose() called, stops when a graph starts ({check_rebalance(state, rebal)})")

def mutate(text, old, new, what):
    if old not in text:
        check(False, f"mutation '{what}': the text it deletes is not in the source (update the test)")
        return None
    return text.replace(old, new, 1)

# the same checks must fail when the guarded call is deleted or bypassed (the proof that this test can fail)
m = mutate(compute, "return mcguard::guarded(catch_mode, injected, [&] {", "return ([&] {", "gpu_segment_guarded without mcguard::guarded")
if m: check(check_gpu_catch(m, backend, hybrid), "mutation: gpu_segment_guarded without mcguard::guarded() is caught")
m = mutate(compute, "const bool ok = mcguard::guarded(catch_mode, injected, [&] { return hybrid_group(c, s, cg, first, last); }, [&](const char * what) {",
           "const bool ok = ([&] { return hybrid_group(c, s, cg, first, last); })() || ([&](const char * what) {", "hybrid_group_guarded without mcguard::guarded")
if m: check(check_gpu_catch(m, backend, hybrid), "mutation: hybrid_group_guarded without mcguard::guarded() is caught")
m = mutate(compute, "bool ok = gpu_segment_guarded(c, s, cg, gr.first, gr.last, seg);", "bool ok = gpu_segment(c, s, cg, gr.first, gr.last, seg);", "decode path calls gpu_segment directly")
if m: check(check_gpu_catch(m, backend, hybrid), "mutation: the decode path calls gpu_segment() directly is caught")
m = mutate(compute, "hybrid_group_guarded(c, s, cg, i, last)", "hybrid_group(c, s, cg, i, last)", "decode path calls hybrid_group directly")
if m: check(check_gpu_catch(m, backend, hybrid), "mutation: the decode path calls hybrid_group() directly is caught")
m = mutate(compute, "        gpu_failed_cleanup(c);\n        seg.clear();", "        seg.clear();", "no cleanup after the throw")
if m: check(check_gpu_catch(m, backend, hybrid), "mutation: no cleanup after a caught throw is caught")
m = mutate(lru, "if (mclru::refresh_recency_of_resident(cur_scan)) {", "if (true) {", "LRU rule bypassed")
if m: check(check_lru(m), "mutation: the LRU rule bypassed (always refresh) is caught")
m = mutate(rebal, "rb_choose(units, free_b, rb_cap_bytes, rb_gain, rb_min_score)", "std::vector<rb_pick>()", "rb_choose removed")
if m: check(check_rebalance(state, m), "mutation: rb_choose() removed from the idle thread is caught")
m = mutate(state, "bool   rb_on = true;", "bool   rb_on = false;", "rebalance off by default")
if m: check(check_rebalance(m, rebal), "mutation: idle rebalance off by default is caught")

print("FAIL: " + "; ".join(fails) if fails else "guard wiring: all checks passed")
sys.exit(1 if fails else 0)
