#!/usr/bin/env python3
"""LAUNCHER-1133 (build's LAUNCHER-1131 WIP, pre-load half): the planner plans against the room the GPU really gives (Windows AMD / Intel): card size minus the other programs' dedicated use. Fake counters, no model server.
   python3 tests/room_test.py
PC 2026-11-11, RX 9070 XT 16 GB: llama-server --list-devices said 15.4 GiB free, the process kept 13.1-13.5 GiB in dedicated memory and paged the rest; the default launch at 64k picked q8_0 KV (18.5 tok/s) where q4_0 gives 27.8."""
import json, os, struct, sys, tempfile, types
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-room-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "planner")); sys.path.insert(0, os.path.join(HERE, "tests"))
import moe_cache as mc   # noqa: E402
import gguf_plan as gp   # noqa: E402

GIB = 2**30
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

def make_dense(path, n_layers=24, d=2048):
    def string(x):
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    infos, off = [], 0
    def add(name, dims, ty, nbytes):
        nonlocal off
        infos.append((name, dims, ty, off)); off += (nbytes + 31) // 32 * 32
    add("token_embd.weight", [d, 32000], 1, d * 32000 * 2)
    for l in range(n_layers):
        add(f"blk.{l}.attn_q.weight", [d, d], 12, d // 256 * 144 * d)
        add(f"blk.{l}.ffn_up.weight", [d, 4 * d], 12, d // 256 * 144 * 4 * d)
    kv = [("general.architecture", 8, string("llama")), ("llama.block_count", 4, struct.pack("<I", n_layers)), ("llama.context_length", 4, struct.pack("<I", 131072)),
          ("llama.embedding_length", 4, struct.pack("<I", d)), ("llama.attention.head_count", 4, struct.pack("<I", 16)), ("llama.attention.head_count_kv", 4, struct.pack("<I", 4))]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(kv))
    for k, t, v in kv:
        head += string(k) + struct.pack("<I", t) + v
    for name, dims, ty, o in infos:
        head += string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", x) for x in dims) + struct.pack("<IQ", ty, o)
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head); f.truncate(len(head) + off)


def line(pid, kind, mib):
    return f"\\\\pc\\gpu process memory(pid_{pid}_luid_0x0_0x1_phys_0)\\{kind} usage|{mib * 1048576}"

# --- pure helpers ---
txt = ";".join([line(100, "dedicated", 1500), line(200, "dedicated", 900), line(300, "shared", 50), line(300, "dedicated", 13000), line(300, "shared", 1700)])
check(abs(gp.others_dedicated_mib(txt, 300) - 2400) < 1e-6, "others_dedicated_mib: the other processes' dedicated use (2400 MiB), not ours, not shared")
# 2026-11-12 PC (RX 9070 XT + the CPU's iGPU): the process counters summed to 3.9 GiB, the card's own adapter counter read 1.92 GiB
def pl(pid, luid, mib, kind="dedicated"):
    return f"\\\\pc\\gpu process memory(pid_{pid}_luid_{luid}_phys_0)\\{kind} usage|{mib * 1048576}"
def al(luid, mib):
    return f"\\\\pc\\gpu adapter memory(luid_{luid}_phys_0)\\dedicated usage|{mib * 1048576}"
CARD, IGPU = "0x00000000_0x0000a1b2", "0x00000000_0x0000c3d4"
pc = ";".join([pl(10, CARD, 700), pl(11, CARD, 500), pl(12, CARD, 400), pl(13, CARD, 380), pl(10, IGPU, 300), pl(14, IGPU, 250), pl(15, CARD, 800), pl(16, CARD, 600), pl(17, CARD, 20),
               al(CARD, 1970), al(IGPU, 560)])
plain = ";".join(x for x in pc.split(";") if "adapter" not in x)
check(abs(gp.others_dedicated_mib(plain) - (700 + 500 + 400 + 380 + 300 + 250 + 800 + 600 + 20)) < 1e-6, "no adapter counters in the text: the plain sum over every adapter, as before (3950 MiB: the 3.9 GiB of the PC)")
check(abs(gp.others_dedicated_mib(pc) - 1970) < 1e-6, f"the PC case: processes sum to 3400 MiB on the card, the card's own counter says 1970 MiB: the figure is capped to 1970 ({gp.others_dedicated_mib(pc)})")
check(abs(gp.others_dedicated_mib(pc.replace(f"{1970 * 1048576}", f"{9000 * 1048576}")) - 3400) < 1e-6, "adapter figure above the processes' sum: the sum stays (the cap only lowers)")
only_card = ";".join([pl(10, CARD, 700), pl(10, IGPU, 5000), al(CARD, 900), al(IGPU, 5200)])
check(abs(gp.others_dedicated_mib(only_card) - 5000) < 1e-6, "the card is the adapter with the most dedicated use (a limit: an iGPU holding more than the card would win; the counters do not say which one llama.cpp uses)")
two = ";".join([pl(10, CARD, 700), pl(10, IGPU, 100), al(CARD, 2000), al(IGPU, 300)])
check(abs(gp.others_dedicated_mib(two) - 700) < 1e-6, "instances of the other adapter (the iGPU) are left out")
mine = ";".join([pl(100, CARD, 9000), pl(10, CARD, 700), al(CARD, 9800)])
check(abs(gp.others_dedicated_mib(mine, 100) - 700) < 1e-6, "our own process excluded; the cap is the adapter figure minus our share (9800 - 9000 = 800 >= 700)")
mine2 = ";".join([pl(100, CARD, 9000), pl(10, CARD, 2500), al(CARD, 10500)])
check(abs(gp.others_dedicated_mib(mine2, 100) - 1500) < 1e-6, "cap with our share taken off: min(2500, 10500 - 9000) = 1500")
# the REAL capture (PC 2026-11-12 22:54, RX 9070 XT, typeperf CSV turned into the 'path|value;' text counters_text returns): luids 0x145FA (card), 0x16A03 (iGPU), 0x16982
import csv as _csv
def _csv_text(name):
    rows = list(r for r in _csv.reader(open(os.path.join(HERE, "tests", "data", name), encoding="utf-8-sig")) if r)
    return ";".join(f"{h}|{v}" for h, v in zip(rows[0][1:], rows[1][1:]) if h and v.strip())
_real_p, _real_a = _csv_text("gpu_counters_pc_20261112_process.csv"), _csv_text("gpu_counters_pc_20261112_adapter.csv")
check(abs(gp.others_dedicated_mib(_real_p) - 3.89 * 1024) < 15, f"real capture, process counters only: the plain sum {gp.others_dedicated_mib(_real_p) / 1024:.2f} GiB (the planner's 3.9 GiB)")
_real = gp.others_dedicated_mib(_real_p + ";" + _real_a)
check(abs(_real - 2060398592 / 1048576) < 1, f"real capture with the adapter counter: capped to the card's {_real / 1024:.2f} GiB (adapter 1.92 GiB, gputaskman 1.93)")
check(gp.real_room_mib(15.9 * 1024, _real) / 1024 > 13.5, f"the room with the real figure: {gp.real_room_mib(15.9 * 1024, _real) / 1024:.2f} GiB (was 11.75)")
check(gp.others_dedicated_mib("") is None and gp.others_dedicated_mib("junk|x") is None, "others_dedicated_mib: no counter -> None")
check(gp.real_room_mib(16304, 2400) == 16304 - 2400 - 300 and gp.real_room_mib(16304, None) is None and gp.real_room_mib(16304, 16304) is None and gp.real_room_mib(None, 100) is None and gp.real_room_mib(2000, 1500) is None,
      "real_room_mib: size - others - reserve; None without figures or when the result is implausible")

# --- the launcher: gpu_stage fills a.room_gib; the dense fit uses it ---
class M:
    windows = True
    others = 2400
    def list_devices(self, ls): return "Available devices:\n  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15400 MiB free)\n"
    def pick_gpu_full(self, listing): return ("Vulkan0", "AMD Radeon RX 9070 XT", 15400, 16304)
    def nvidia_reports(self): return False
    def win_dedicated_vram(self): return {"AMD Radeon RX 9070 XT": 16304 << 20}
    def gpu_process_text(self): return line(100, "dedicated", self.others)
    def free_vram_bytes(self): return None
def stage(vram=None, m=None):
    a = types.SimpleNamespace(vram=vram, cpu_only=False, gpu_margin=None)
    said = []; sv = mc.say; mc.say = lambda t, err=False: said.append(t)
    try:
        mc.gpu_stage(a, "llama-server", True, m or M())
    finally:
        mc.say = sv
    return a, said
a, said = stage()
check(a.vram == "15.04" and abs(a.room_gib - (15400 - 2400) / 1024) < 0.01 and any("GPU room: planning with 12.70 GiB" in x for x in said), f"gpu_stage: a.vram {a.vram}, room {getattr(a, 'room_gib', None)}, line: {[x for x in said if 'room' in x]}")
check(abs(a.room_used_gib - 2400 / 1024) < 1e-6, "gpu_stage: the other programs' use is kept for the desk reserve")
# critic R1: the room is the smaller of (card - others - reserve) and (llama-server's free figure - others): 16304 - 2400 - 300 = 13604 vs 15400 - 2400 = 13000 -> 13000
check(abs(a.room_gib * 1024 - 13000) < 1, "room = the smaller of the two readings (13000 MiB, not 13604)")
m4 = M(); m4.gpu_process_text = lambda: _real_p + ";" + _real_a
m4.total_mib = 16304
c4, said4 = stage(m=m4)
check(abs(c4.room_gib - (15.04 * 1024 - 2060398592 / 1048576) / 1024) < 0.01, f"the real PC capture through gpu_stage: room {c4.room_gib:.2f} GiB (driver 15.04 minus the adapter's 1.92), not 11.75 and not 13.69")
f, gap = mc.dense_fit_free(a, M())
check(abs(f / GIB - a.room_gib) < 1e-6 and gap == 0.0, "dense_fit_free: min(driver figure, real room)")
b, said = stage(vram="15.4")
check(not hasattr(b, "room_gib") and mc.dense_fit_free(b, M())[0] == 15.4 * GIB and not any("GPU room" in x for x in said), "the user's own --vram is taken as given: no room line, no cap")
m2 = M(); m2.others = 300
c, said = stage(m=m2)
check(getattr(c, "room_gib", None) is None and not any("GPU room" in x for x in said), "a nearly empty card: the driver figure stands, nothing is said")
m3 = M(); m3.gpu_process_text = lambda: ""
c, said = stage(m=m3)
check(getattr(c, "room_gib", None) is None and sum("GPU room: the other programs' GPU memory use could not be read" in x and "free figure as before" in x for x in said) == 1,
      f"no counters: the driver figure stands and one line says so ({[x for x in said if 'room' in x]})")
m4 = M(); m4.gpu_process_text = lambda: 1 / 0
c, said = stage(m=m4)
check(c.vram == "15.04" and getattr(c, "room_gib", None) is None, "a counter read that raises never stops the plan")
check(mc.dense_fit_free(types.SimpleNamespace(vram="15.04", room_gib=13.27, gpu_margin=None), M())[0] == 13.27 * GIB, "dense_fit_free takes the smaller room")

# --- the ladder: q8_0 fits the driver's figure but not the real room -> the planner steps down ---
model = os.path.join(TMP, "dense-7b-Q4.gguf"); make_dense(model)
ctx = 16384
need8 = gp.dense_gpu_fit(model, ctx, 64 * GIB, True, 0.0, 0.0, 0.20)["need_gib"]
need4 = gp.dense_gpu_fit(model, ctx, 64 * GIB, "q4_0", 0.0, 0.0, 0.20)["need_gib"]
check(need4 < need8, f"synthetic model: q4_0 needs {need4:.2f} GiB, q8_0 {need8:.2f}")
drv, room = need8 + 0.3, (need8 + need4) / 2
x = types.SimpleNamespace(vram=f"{drv:.3f}", room_gib=None, gpu_margin=None)
t = mc.dense_gpu_fit_pair(x, model, ctx, M(), with_q4=True)
check(t and t[1]["fits"], "driver figure alone: q8_0 KV fits (the old default)")
x.room_gib = room
t = mc.dense_gpu_fit_pair(x, model, ctx, M(), with_q4=True)
check(t and not t[1]["fits"] and t[3]["fits"], "real room: q8_0 does not fit, q4_0 does -> the ladder steps down")
# --- the counter read is kept for a few seconds (the page plans a whole model list) ---
reads = []
_ct = gp.counters_text
gp.counters_text = lambda pid=None, timeout=8: (reads.append(1), line(100, "dedicated", 1000))[1]
mc._GPU_PROC[:] = [0.0, ""]
t1 = mc.win_gpu_process_text(); t2 = mc.win_gpu_process_text(); t3 = mc.win_gpu_process_text()
check(len(reads) == 1 and t1 == t2 == t3 != "", f"win_gpu_process_text: three plans, one PowerShell read ({len(reads)})")
mc._GPU_PROC[0] -= mc.GPU_PROC_TTL_S + 1
mc.win_gpu_process_text()
check(len(reads) == 2, "...and a new read after the time is up")
gp.counters_text = lambda pid=None, timeout=8: (reads.append(1), "")[1]
mc._GPU_PROC[:] = [0.0, ""]
mc.win_gpu_process_text(); mc.win_gpu_process_text()
check(len(reads) == 4, "a failed read ('') is not kept: the next plan tries again")
gp.counters_text = _ct
print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
