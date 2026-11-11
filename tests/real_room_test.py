#!/usr/bin/env python3
"""LAUNCHER-1131: the planner plans against the room the GPU really gives (Windows AMD / Intel), the paging detector and the learned ceiling. Fake counters, no model server.
   python3 tests/real_room_test.py
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
check(gp.others_dedicated_mib("") is None and gp.others_dedicated_mib("junk|x") is None, "others_dedicated_mib: no counter -> None")
check(gp.real_room_mib(16304, 2400) == 16304 - 2400 - 300 and gp.real_room_mib(16304, None) is None and gp.real_room_mib(16304, 16304) is None and gp.real_room_mib(None, 100) is None and gp.real_room_mib(2000, 1500) is None,
      "real_room_mib: size - others - reserve; None without figures or when the result is implausible")
check(gp.parse_gpu_process_counters(txt, 300) == (13000.0, 1750.0), "the counter parser sums our pid's dedicated and shared figures")

# --- the store ---
d = os.path.join(TMP, "store")
check(gp.room_ceiling(d, "GPU A") is None, "store: nothing learned -> no ceiling")
check(gp.room_learn(d, "GPU A", 13300, 100, now=1000) is False and gp.room_ceiling(d, "GPU A", now=1000) is None, "learn: shared 100 MiB is not paging, no ceiling")
check(gp.room_learn(d, "GPU A", 13200, 1700, now=1000) is True and gp.room_ceiling(d, "GPU A", now=1001) == 13200, "learn: shared 1700 MiB is paging, the dedicated figure becomes the ceiling")
check(gp.room_ceiling(d, "GPU B", now=1001) is None, "the ceiling belongs to its GPU")
check(gp.room_ceiling(d, "GPU A", now=1000 + gp.ROOM_KEEP_S + 5) is None, "the ceiling expires after 30 days")
gp.room_learn(d, "GPU A", 13300, 0, now=2000)
check(gp.room_ceiling(d, "GPU A", now=2001) == 13200, "a start without paging inside the ceiling keeps it")
gp.room_learn(d, "GPU A", 14000, 0, now=3000)
check(gp.room_ceiling(d, "GPU A", now=3001) is None, "no paging with clearly more dedicated memory than the ceiling: the desktop gave room back, the ceiling goes")
open(os.path.join(d, "gpu_room.json"), "w").write("not json")
check(gp.room_ceiling(d, "GPU A") is None and gp.room_learn(d, "GPU A", 13000, 900) is True and gp.room_ceiling(d, "GPU A") == 13000, "a damaged store file is ignored and replaced")

# --- the watcher: readings after ready, one warning, learns ---
said, now = [], [0.0]
reads = iter([line(7, "dedicated", 13100) + ";" + line(7, "shared", 90), line(7, "dedicated", 13200) + ";" + line(7, "shared", 1800), line(7, "dedicated", 13200) + ";" + line(7, "shared", 1900)])
alive_n = [0]
def alive():
    alive_n[0] += 1
    return alive_n[0] < 40
gp.watch_room(7, "GPU W", os.path.join(TMP, "w"), said.append, alive, lambda: True, sampler=lambda pid: next(reads, ""), sleep=lambda s: None)
check(len(said) == 1 and said[0].startswith("moe-cache: warning:") and "paging" in said[0] and "1800 MiB" in said[0], f"watcher: one warning line when paging starts, none for the first clean reading ({len(said)} lines)")
check(gp.room_ceiling(os.path.join(TMP, "w"), "GPU W") == 13200, "watcher: the ceiling is the dedicated figure of the first paging reading")
said2, ev = [], []
n = [0]
def alive3():
    n[0] += 1
    return n[0] < 30
gp.watch_room(8, "GPU X", os.path.join(TMP, "x"), said2.append, alive3, lambda: True, sampler=lambda pid: (ev.append(1) or ""), sleep=lambda s: None)
check(not said2 and len(ev) == 3, f"watcher: gives up after 3 readings without a counter ({len(ev)}), says nothing")
not_ready = []
gp.watch_room(9, "GPU Y", os.path.join(TMP, "y"), said2.append, lambda: len(not_ready) < 3 and not not_ready.append(1) or False, lambda: False, sampler=lambda pid: 1 / 0, sleep=lambda s: None)
check(not said2, "watcher: nothing is read before the server is ready")

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
mc.CACHE_DIR = os.path.join(TMP, "launcher_cache")
def stage(vram=None, m=None):
    a = types.SimpleNamespace(vram=vram, cpu_only=False, gpu_margin=None)
    said = []; sv = mc.say; mc.say = lambda t, err=False: said.append(t)
    try:
        mc.gpu_stage(a, "llama-server", True, m or M())
    finally:
        mc.say = sv
    return a, said
a, said = stage()
check(a.vram == "15.04" and abs(a.room_gib - (16304 - 2400 - 300) / 1024) < 0.01 and any("GPU room: planning with 13.29 GiB" in x for x in said), f"gpu_stage: a.vram {a.vram}, room {getattr(a, 'room_gib', None)}, line: {[x for x in said if 'room' in x]}")
check(mc.REAL_ROOM.get("device") == "AMD Radeon RX 9070 XT", "gpu_stage: the watcher gets the device name")
f, gap = mc.dense_fit_free(a, M())
check(abs(f / GIB - a.room_gib) < 1e-6 and gap == 0.0, "dense_fit_free: min(driver figure, real room)")
b, said = stage(vram="15.4")
check(not hasattr(b, "room_gib") and mc.dense_fit_free(b, M())[0] == 15.4 * GIB, "the user's own --vram is taken as given: no room line, no cap")
m2 = M(); m2.others = 300
c, said = stage(m=m2)
check(getattr(c, "room_gib", None) is None and not any("GPU room" in x for x in said), "a nearly empty card: the driver figure stands, nothing is said")
m3 = M(); m3.gpu_process_text = lambda: ""
c, said = stage(m=m3)
check(getattr(c, "room_gib", None) is None, "no counters: the driver figure stands")
gp.room_learn(mc.CACHE_DIR, "AMD Radeon RX 9070 XT", 12000, 2000)
c, said = stage(m=m2)
check(abs(c.room_gib - 12000 / 1024) < 0.01 and any("earlier start kept only" in x for x in said), f"a learned ceiling lowers the plan on the next start (room {getattr(c, 'room_gib', None)})")
d_ = types.SimpleNamespace(vram="15.04", room_gib=13.27, gpu_margin=None)

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

sys.exit(1 if fails else 0)
