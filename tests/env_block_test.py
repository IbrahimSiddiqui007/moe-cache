#!/usr/bin/env python3
"""env_block_test (BENCH-1150): the environment block every published number carries ("measured on a working desktop"): RAM total / in use / moe-cache / the rest /
five biggest programs, VRAM per process with the spill, free commit, sampled at start, peak and end. Fake probe, fake gputaskman JSON (the shape of the real
`gtm --json`), chain's .gtm.jsonl lines, the Sampler end to end, the one-line text, and the places that print it (card lines, RESULTS section, claim show, board).
On the real machine: gputaskman is read when installed, and the real probe's commit_free runs (Windows CI exercises the ctypes call)."""
import contextlib, copy, glob, io, json, os, shutil, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
import moe_cache_record as rr  # noqa: E402
import moe_cache_claims as cc  # noqa: E402

FAILS = []
NR = rr.NR
G = 2 ** 30


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


# gputaskman --json, the fields the block reads (real shape: system.ram bytes, system.top_ram [pid, name, rss], processes [pid, name, vram_bytes, shared_bytes])
SERVER, LAUNCHER = 4000, 3999


def gtm(used_gb=22.0, mc_vram=10.9, spill=0.4):
    return {"vram_total": int(16 * G), "shared_is_spill": True,
            "system": {"ram": {"total": int(31.2 * G), "used": int(used_gb * G), "available": int((31.2 - used_gb) * G)},
                       "top_ram": [{"pid": SERVER, "name": "llama-server.exe", "rss": int(8.9 * G)}, {"pid": LAUNCHER, "name": "python.exe", "rss": int(0.1 * G)},
                                   {"pid": 11, "name": "Claude.exe", "rss": int(0.5 * G)}, {"pid": 12, "name": "Claude.exe", "rss": int(0.3 * G)},
                                   {"pid": 13, "name": "Dia.exe", "rss": int(0.4 * G)}, {"pid": 14, "name": "Code.exe", "rss": int(0.3 * G)},
                                   {"pid": 15, "name": "MsMpEng.exe", "rss": int(0.2 * G)}, {"pid": 16, "name": "Teams.exe", "rss": int(0.1 * G)}]},
            "processes": [{"pid": SERVER, "name": "llama-server.exe", "vram_bytes": int(mc_vram * G), "shared_bytes": int(spill * G)},
                          {"pid": 21, "name": "dwm.exe", "vram_bytes": int(1.2 * G), "shared_bytes": 0}, {"pid": 11, "name": "Claude.exe", "vram_bytes": int(0.4 * G), "shared_bytes": 0},
                          {"pid": 22, "name": "Dia.exe", "vram_bytes": None, "vram_inferred_bytes": int(0.2 * G), "shared_bytes": None}]}


class FakeProbe:
    os_name = "windows"
    ncpu = 4

    def __init__(self, used_gb=22.0):
        self.used = used_gb

    def tree(self, pid): return [SERVER, LAUNCHER]

    def proc(self, pids): return {"rss": int(9.0 * G), "private": int(5 * G), "cpu_s": 1.0, "hwm": 0}

    def mem(self): return {"total": int(31.2 * G), "available": int((31.2 - self.used) * G)}

    def vram(self, pids): return 12 * 1024, 10 * 1024, "fake counters"

    def commit_free(self): return int(7.5 * G)

    def sys_cpu(self): return (1.0, 2.0)


# ---- 1. the snapshot from gputaskman
s = rr.env_snapshot(FakeProbe(), {SERVER, LAUNCHER}, gtm(), 3.0)
check(s["ram_total_gb"] == 31.2 and s["ram_used_total_gb"] == 22.0 and s["ram_moe_cache_gb"] == 9.0 and s["ram_rest_of_system_gb"] == 13.0,
      f"RAM: total 31.2, in use 22.0, moe-cache 9.0 (the server tree's working set), rest 13.0 ({s['ram_total_gb']}, {s['ram_used_total_gb']}, {s['ram_moe_cache_gb']}, {s['ram_rest_of_system_gb']})")
check(s["top_programs"] == [["Claude", 0.8], ["Dia", 0.4], ["Code", 0.3], ["MsMpEng", 0.2], ["Teams", 0.1]],
      f"top programs: one program's processes added up, .exe dropped, our own server and launcher left out ({s['top_programs']})")
check(s["vram_total_gb"] == 16.0 and s["vram_per_process"][0] == ["moe-cache", 10.9, 0.4], f"VRAM: moe-cache first with its spill ({s['vram_per_process'][0]})")
check(s["vram_per_process"][1:] == [["dwm", 1.2, 0.0], ["Claude", 0.4, 0.0], ["Dia", 0.2, NR]], f"VRAM: others by size, an inferred figure used when vram_bytes is missing, no spill figure = not recorded ({s['vram_per_process'][1:]})")
check(s["commit_free_gb"] == 7.5 and s["source"].startswith("gputaskman"), f"commit free and the source named ({s['commit_free_gb']}, {s['source']})")
check(not rr.validate(s, {**json.load(open(os.path.join(HERE, "..", "docs", "bench_record.schema.json")))["$defs"]["environment"], "$defs": json.load(open(os.path.join(HERE, "..", "docs", "bench_record.schema.json")))["$defs"]}),
      "the snapshot validates against the schema's environment definition")

# ---- 2. no gputaskman: own counters, and what is not known says "not recorded"
s2 = rr.env_snapshot(FakeProbe(), {SERVER, LAUNCHER}, None, 1.0)
check(s2["ram_used_total_gb"] == 22.0 and s2["ram_moe_cache_gb"] == 9.0 and s2["top_programs"] == NR and s2["vram_total_gb"] == NR, f"Windows without gputaskman: totals from the counters, programs and card size 'not recorded' ({s2['top_programs']})")
check(s2["vram_per_process"] == [["moe-cache", 10.0, NR], ["other programs", 2.0, NR]] and s2["source"] == "own counters (no gputaskman)", f"VRAM from the counters: moe-cache and the rest ({s2['vram_per_process']})")
lp = FakeProbe(); lp.os_name = "linux"
files = {"/proc/10/status": "VmRSS:\t 2097152 kB\n", "/proc/10/comm": "firefox\n", "/proc/11/status": "VmRSS:\t 1048576 kB\n", "/proc/11/comm": "firefox\n",
         "/proc/%d/status" % SERVER: "VmRSS:\t 9437184 kB\n", "/proc/%d/comm" % SERVER: "llama-server\n"}
sv = (rr.READ, rr.GLOB)
rr.READ = lambda p: files.get(p, "")
rr.GLOB = lambda pat: sorted(k for k in files if k.endswith("/status")) if pat.endswith("/status") else []
s3 = rr.env_snapshot(lp, {SERVER, LAUNCHER}, None, 1.0)
rr.READ, rr.GLOB = sv
check(s3["top_programs"] == [["firefox", 3.0]] and "programs" in s3["source"], f"Linux without gputaskman: programs from /proc, the server left out ({s3['top_programs']}, {s3['source']})")
broken = FakeProbe(); broken.mem = lambda: {"total": None, "available": None}
s4 = rr.env_snapshot(broken, {SERVER}, None, 0)
check(s4["ram_total_gb"] == NR and s4["ram_used_total_gb"] == NR and s4["ram_rest_of_system_gb"] == NR, "no RAM counter at all: 'not recorded', no exception")

# ---- 3. start / peak / end
tr = rr.EnvTrack()
check(tr.block() == NR, "no sample: the block is 'not recorded'")
for u, t in ((20.0, 0.0), (26.5, 10.0), (21.0, 20.0)):
    tr.add(rr.env_snapshot(FakeProbe(u), {SERVER}, gtm(u), t))
b = tr.block()
check(b["ram_used_total_gb"] == 26.5 and b["sampled_at"]["start"]["ram_used_total_gb"] == 20.0 and b["sampled_at"]["peak"]["t_s"] == 10.0 and b["sampled_at"]["end"]["ram_used_total_gb"] == 21.0 and b["samples"] == 3,
      f"the block is the peak snapshot (26.5 GB in use) with start 20.0 and end 21.0 in sampled_at ({b['ram_used_total_gb']})")

# ---- 4. the Sampler end to end (fake probe, fake gputaskman): start, at least one more, end
saved = (rr.read_gtm, rr.find_gtm)
rr.read_gtm = lambda exe, timeout=25: gtm(24.0)
smp = rr.Sampler(SERVER, pr=FakeProbe(24.0), interval=0.05, vram_interval=0.05, env_every=0.1, gtm="fake-gtm")
smp.start(); time.sleep(0.45); res = smp.stop()
rr.read_gtm, rr.find_gtm = saved
e = res.get("environment")
check(isinstance(e, dict) and e["samples"] >= 3 and e["sampled_at"]["start"]["t_s"] <= e["sampled_at"]["end"]["t_s"] and e["ram_moe_cache_gb"] == 9.0, f"Sampler result carries the environment block ({e and e['samples']} samples)")
smp = rr.Sampler(SERVER, pr=FakeProbe(), interval=0.05, vram_interval=0.05, env_every=None)
smp.start(); time.sleep(0.15); res = smp.stop()
check("environment" not in res, "env_every=None switches the block off (nothing in the result)")
run = rr.run_entry("moe-cache", 1, {"decode_tps": 50.0, "prompt_tps": 900.0}, {**res, "environment": b})
check(run["environment"] == b, "run_entry keeps the block on the run")
check(not rr.validate({"environment": b}, {"type": "object", "properties": {"environment": {"$ref": "#/$defs/environment"}}, "$defs": json.load(open(os.path.join(HERE, "..", "docs", "bench_record.schema.json")))["$defs"]}),
      "the run's block validates against the schema")

# ---- 5. the one line (the owner's example, with the figures of this fake)
line = rr.env_line(rr.EnvTrack().block() if False else b)
check(line == "working desktop (at peak): 31 GB RAM, moe-cache 9.0 GB, other programs 17.5 GB (Claude 0.8, Dia 0.4, Code 0.3...), GPU: moe-cache 10.9 GB (0.4 GB spilled), desktop 1.8 GB", f"the line: {line}")
check(rr.env_line(NR) is None and rr.env_line({"ram_total_gb": NR}) is None, "no block / no total: no line (never a made-up one)")
cut = {k: v for k, v in b.items() if k not in ("vram_per_process", "top_programs")}
check(rr.env_line(cut) == "working desktop (at peak): 31 GB RAM, moe-cache 9.0 GB, other programs 17.5 GB", f"missing parts are left out, not guessed ({rr.env_line(cut)})")

# ---- 6. chain's gtm_sampler.ps1 lines -> the same block, per run window
def chain_line(t, used, tag="sample", g=None):
    return json.dumps({"t": t, "tag": tag, "used_ram_gb": used, "total_ram_gb": 31.2, "llama": [{"pid": SERVER, "name": "llama-server", "ws_gb": 9.0, "private_gb": 5.0}],
                       "top_ram": [{"pid": SERVER, "name": "llama-server", "ws_gb": 9.0}, {"pid": 11, "name": "Claude", "ws_gb": 0.5}, {"pid": 12, "name": "Claude", "ws_gb": 0.3}, {"pid": 13, "name": "Dia", "ws_gb": 0.4}],
                       "gputaskman": g})
lines = [chain_line("2026-11-13T01:00:00", 18.0, "start"), chain_line("2026-11-13T01:10:00", 25.0, g=gtm(25.0)), chain_line("2026-11-13T01:20:00", 20.0), chain_line("2026-11-13T02:30:00", 30.0, "end")]
cb = rr.env_from_lines(lines, "2026-11-13T01:00:00", "2026-11-13T01:30:00")
check(cb["samples"] == 3 and cb["ram_used_total_gb"] == 25.0 and cb["ram_moe_cache_gb"] == 9.0 and cb["vram_per_process"][0][0] == "moe-cache", f"chain lines: the run's time window only, peak = the 25 GB line ({cb['samples']}, {cb['ram_used_total_gb']})")
check(rr.env_from_lines(lines)["ram_used_total_gb"] == 30.0 and rr.env_from_lines(["not json", ""]) == NR, "chain lines: all lines when no window; nothing readable = 'not recorded'")
tf = os.path.join(tempfile.mkdtemp(), "x.gtm.jsonl"); open(tf, "w").write("\n".join(lines))
with contextlib.redirect_stdout(io.StringIO()) as out:
    rc = rr.main(["env-from-gtm", tf, "--start", "2026-11-13T01:00:00", "--end", "2026-11-13T01:30:00", "--line"])
check(rc == 0 and out.getvalue().startswith("working desktop (at peak): 31 GB RAM, moe-cache 9.0 GB"), f"env-from-gtm --line prints the line ({out.getvalue().strip()[:70]})")

# ---- 7. where it is printed: a real record of this repo with the block added to its runs, and the same record without
src = sorted(glob.glob(os.path.join(HERE, "..", "benchmarks", "runs", "*.json")))
rec = None
for f in reversed(src):
    r = json.load(open(f, encoding="utf-8"))
    if any(x["arm"] == "moe-cache" for x in r["runs"]):
        rec, recfile = r, f
        break
rec2 = copy.deepcopy(rec)
for i, r in enumerate(rec2["runs"]):
    r["environment"] = rr.EnvTrack().block() if False else {**{k: v for k, v in b.items() if k != "sampled_at"}, "ram_used_total_gb": 20.0 + i}
d0, d1 = rr.details_lines(rec), rr.details_lines(rec2)
check(len(d1) == len(d0) + 1 and d1[-1].startswith("working desktop (at peak): 31 GB RAM") and (rr.env_of(rec2, "moe-cache")[1] < 2 or "[the busiest of" in d1[-1]), f"card lines: one more line with the block, naming how many runs it is the busiest of ({d1[-1][-40:]})")
check(len(d0) == 4 and not any("working desktop" in l for l in d0), "a record without the block: the four card lines as before (no line, no 'not recorded' noise)")
rec3 = copy.deepcopy(rec2)
rec3["runs"].append({**copy.deepcopy(rec3["runs"][0]), "index": 2, "environment": {**rec3["runs"][0]["environment"], "ram_used_total_gb": 29.0}})
l3 = rr.env_line_for(rec3, "moe-cache")
check(l3.endswith("[the busiest of 2 runs]") and rr.env_of(rec3, "moe-cache")[0]["ram_used_total_gb"] == 29.0, f"two runs of the arm: the busiest run's block, and the line says so ({l3[-30:]})")
rec4 = copy.deepcopy(rec2)
rec4["runs"][0]["arm"] = "stock"
l4 = rr.env_line_for(rec4, "stock")
check("stock llama-server 9.0 GB" in l4 and "GPU: stock llama-server 10.9 GB" in l4 and "moe-cache" not in l4, f"the stock arm's line says 'stock llama-server', not moe-cache ({l4[:90]})")
sec1, sec0 = rr.results_section(rec2, "T", 1), rr.results_section(rec, "T", 1)
check("Environment: working desktop (at peak)" in sec1 and "Environment: working desktop" not in sec0, "RESULTS section: an 'Environment:' line only when the block exists")
a = rr.assemble({"hardware": rec["hardware"], "software": rec["software"], "model": rec["model"]}, rec["settings"], rec["conditions"], rec["protocol"], rec2["runs"], reproduce=rec["reproduce"])
check(a["conditions"].get("environment", {}).get("ram_used_total_gb") == max(r["environment"]["ram_used_total_gb"] for r in rec2["runs"]), "assemble: conditions.environment = the busiest run's block")
sch = json.load(open(os.path.join(HERE, "..", "docs", "bench_record.schema.json")))
check(not rr.validate(a, sch), f"a record with the block still validates against schema 1 ({rr.validate(a, sch)[:2]})")
check(not rr.validate(rec, sch), "an old record without the block still validates (the field is optional)")

# claim show and board, in a copy of benchmarks/
bd = os.path.join(tempfile.mkdtemp(), "benchmarks"); shutil.copytree(os.path.join(HERE, "..", "benchmarks"), bd)
cid = "v038-fits-qwen36"
claim = cc.load_claims(bd)[cid]
rid = claim["cells"][0]["run"]
def show():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cc.main(["--dir", bd, "claim", "show", cid])
    return buf.getvalue()
check("Environment: not recorded" in show() and "Measured on a working desktop" not in show(), "claim show on a run without the block: says 'not recorded' (nobody guesses)")
rf = os.path.join(bd, "runs", rid + ".json")
rr_ = json.load(open(rf, encoding="utf-8"))
for i, r in enumerate(rr_["runs"]):
    r["environment"] = {**{k: v for k, v in b.items() if k != "sampled_at"}, "ram_used_total_gb": 20.0 + i}
json.dump(rr_, open(rf, "w", encoding="utf-8"))
sh = show()
check("Measured on a working desktop (at peak): 31 GB RAM" in sh and "Environment: not recorded" not in sh, "claim show on a run with the block: 'Measured on a working desktop ...'")
bt = cc.board_text(cc.load_claims(bd), cc.load_runs(bd), "0.3.8", 12207)
check("| Working desktop |" in bt and "working desktop (at peak): 31 GB RAM" in bt, "board: the 'Working desktop' column appears once a run has the block, with the line on that row")
shutil.rmtree(os.path.dirname(bd), ignore_errors=True)
bt0 = cc.board_text(cc.load_claims(os.path.join(HERE, "..", "benchmarks")), cc.load_runs(os.path.join(HERE, "..", "benchmarks")), "0.3.8", 12207)
check("Working desktop" not in bt0, "board without any block: no new column")

# ---- 8. the real machine
exe = rr.find_gtm()
if exe:
    g = rr.read_gtm(exe)
    check(isinstance(g, dict) and "system" in g, f"real gputaskman JSON read ({os.path.basename(exe)})")
    real = rr.env_snapshot(rr.probe(), {os.getpid()}, g, 0.0)
    check(isinstance(real["ram_total_gb"], float) and real["ram_total_gb"] > 1 and isinstance(real["ram_used_total_gb"], float), f"real snapshot: RAM {real['ram_total_gb']} GB, {real['ram_used_total_gb']} GB in use")
else:
    print("  skip real gputaskman (not installed here)")
cf = rr.probe().commit_free()
check(cf is None or cf > 0, f"real probe commit_free: {cf if cf is None else round(cf / G, 1)} GB (Windows CI runs the ctypes call; Linux reads CommitLimit - Committed_AS)")
if os.name == "nt":
    check(cf is not None, "Windows: GlobalMemoryStatusEx gave a commit figure")

print("env block test: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
sys.exit(1 if FAILS else 0)
