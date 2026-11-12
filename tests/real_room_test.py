#!/usr/bin/env python3
"""LAUNCHER-1131 second half (audit ranks 4 and 5): paging detector, learned ceiling, "every load is a measurement" (planner/real_room.py). Fake counters and recorded log lines, no server.
   python3 tests/real_room_test.py"""
import os, struct, sys, tempfile, types
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-realroom-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
sys.path.insert(0, os.path.join(HERE, "bin")); sys.path.insert(0, os.path.join(HERE, "planner")); sys.path.insert(0, os.path.join(HERE, "tests"))
import moe_cache as mc   # noqa: E402
import gguf_plan as gp   # noqa: E402
import real_room as rr   # noqa: E402

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

# --- the ceiling store ---
d = os.path.join(TMP, "store")
check(rr.room_ceiling(d, "GPU A") is None, "store: nothing learned -> no ceiling")
check(rr.room_learn(d, "GPU A", 13300, 100, now=1000) is False and rr.room_ceiling(d, "GPU A", now=1000) is None, "learn: shared 100 MiB is not paging")
check(rr.room_learn(d, "GPU A", 13200, 1700, now=1000) is True and rr.room_ceiling(d, "GPU A", now=1001) == 13200, "learn: shared 1700 MiB is paging, the dedicated figure becomes the ceiling")
check(rr.room_ceiling(d, "GPU B", now=1001) is None and rr.room_ceiling(d, "GPU A", now=1000 + rr.ROOM_KEEP_S + 5) is None, "the ceiling belongs to its GPU and expires after 30 days")
rr.room_learn(d, "GPU A", 13300, 0, now=2000)
check(rr.room_ceiling(d, "GPU A", now=2001) == 13200, "a start without paging inside the ceiling keeps it")
rr.room_learn(d, "GPU A", 14000, 0, now=3000)
check(rr.room_ceiling(d, "GPU A", now=3001) is None, "no paging with more dedicated memory than the ceiling: it goes")
d2 = os.path.join(TMP, "store2")
check(rr.room_learn(d2, "GPU S", 9000, 400, now=1000, room_mib=13300) is False and rr.room_ceiling(d2, "GPU S", now=1001) is None, "shared 400 MiB with dedicated far below the room is staging: no ceiling")
rr.room_learn(d2, "GPU S", 13200, 1700, now=1000, room_mib=13300)
check(rr.room_ceiling(d2, "GPU S", now=1001) == 13200, "paging with dedicated near the room counts")
rr.room_learn(d2, "GPU S", 11000, 0, now=2000, room_mib=15000)
check(rr.room_ceiling(d2, "GPU S", now=2001) is None, "a pre-load room clearly above the ceiling: the ceiling goes")
open(os.path.join(d, "gpu_room.json"), "w").write("not json")
check(rr.room_ceiling(d, "GPU A") is None and rr.room_learn(d, "GPU A", 13000, 900) is True and rr.room_ceiling(d, "GPU A") == 13000, "a damaged store file is ignored and replaced")

# --- the watcher ---
said = []
reads = iter([line(7, "dedicated", 13100) + ";" + line(7, "shared", 90), line(7, "dedicated", 13200) + ";" + line(7, "shared", 1800), line(7, "dedicated", 13200) + ";" + line(7, "shared", 1900)])
got, n = [], [0]
def alive():
    n[0] += 1
    return n[0] < 40
rr.watch_room(7, "GPU W", os.path.join(TMP, "w"), said.append, alive, lambda: True, sampler=lambda pid: next(reads, ""), sleep=lambda s: None, on_reading=lambda ded, sha: got.append((ded, sha)))
check(len(said) == 1 and said[0].startswith("moe-cache: warning:") and "paging" in said[0] and "1800 MiB" in said[0], f"watcher: one warning when paging starts, none for the clean first reading ({len(said)})")
check(rr.room_ceiling(os.path.join(TMP, "w"), "GPU W") == 13200 and got[0] == (13100.0, 90.0), "watcher: ceiling = dedicated of the first paging reading; on_reading gets the first reading")
said2, ev, n2 = [], [], [0]
def alive3():
    n2[0] += 1
    return n2[0] < 30
rr.watch_room(8, "GPU X", os.path.join(TMP, "x"), said2.append, alive3, lambda: True, sampler=lambda pid: (ev.append(1) or ""), sleep=lambda s: None)
check(not said2 and len(ev) == 3, f"watcher: gives up after 3 readings without a counter ({len(ev)})")
k = [0]
rr.watch_room(9, "GPU Y", os.path.join(TMP, "y"), said2.append, lambda: (k.__setitem__(0, k[0] + 1) or k[0] < 4), lambda: False, sampler=lambda pid: 1 / 0, sleep=lambda s: None)
check(not said2, "watcher: nothing is read before the server is ready")

# --- the load measurement ---
REAL = """0.02.329.072 I load_tensors: offloaded 65/65 layers to GPU
0.02.329.080 I load_tensors:        Vulkan0 model buffer size =     0.00 MiB
0.02.329.081 I load_tensors:    Vulkan_Host model buffer size =     0.00 MiB
0.05.100.080 I load_tensors:      Vulkan0 model buffer size = 11500.25 MiB
0.05.100.081 I load_tensors:   CPU_Mapped model buffer size =   500.00 MiB
0.05.200.552 I llama_context:  Vulkan_Host  output buffer size =     0.95 MiB
0.05.200.691 I llama_kv_cache:    Vulkan0 KV buffer size =  1024.00 MiB
0.05.200.699 I llama_kv_cache: size = 1024.00 MiB ( 65536 cells,  16 layers,  1/1 seqs), K (q4_0):  512.00 MiB, V (q4_0):  512.00 MiB
0.05.200.710 I llama_memory_recurrent:    Vulkan0 RS buffer size =   149.62 MiB
0.05.333.547 I sched_reserve:    Vulkan0 compute buffer size =   801.50 MiB
0.05.333.548 I sched_reserve: Vulkan_Host compute buffer size =    30.01 MiB""".splitlines()
b = rr.parse_load_buffers(REAL)
check(b["gpu"] == {"model": 11500.25, "kv": 1024.0, "rs": 149.62, "compute": 801.5} and abs(b["cpu"]["model"] - 500.0) < 1e-6 and b["offloaded"] == (65, 65),
      f"parse: the last line per device and kind wins (the zero-size planning pass is dropped); host buffers are CPU ({b})")
check(rr.parse_load_buffers(["nothing here"]) is None, "parse: no buffer line -> None")
mdl = os.path.join(TMP, "dense-7b-Q4.gguf"); make_dense(mdl)
cmd = ["no-such-llama-server", "-m", mdl, "-c", "65536", "-ctk", "q4_0", "-ctv", "q4_0", "-ub", "512", "-np", "1", "--port", "8080"]
check(rr.cmd_facts(cmd) == (mdl, 65536, "q4_0", "q4_0", 512, 1) and rr.cmd_facts(["x"]) is None and rr.cmd_facts(["-m", "a", "-c", "1024"])[2:4] == ("f16", "f16"),
      "cmd_facts: model, ctx, KV types (f16 default), ub, np")
cd = os.path.join(TMP, "meas")
key = rr.measure_save(cd, cmd, "Vulkan0 GPU", b, 13100, 80, now=5000)
m = rr.measure_load(cd, mdl, 65536, "q4_0", "q4_0", "Vulkan0 GPU", now=5001)
check(key and m and m["model_mib"] == 11500.25 and m["dedicated_mib"] == 13100 and m["shared_mib"] == 80 and m["compute_mib"] > 800, "measure: saved and read back for the exact plan")
check(rr.measure_load(cd, mdl, 65536, "q8_0", "q8_0", "Vulkan0 GPU", now=5001) is None and rr.measure_load(cd, mdl, 32768, "q4_0", "q4_0", "Vulkan0 GPU", now=5001) is None
      and rr.measure_load(cd, mdl, 65536, "q4_0", "q4_0", "Other GPU", now=5001) is None, "measure: another KV type, context or GPU has none")
check(rr.measure_load(cd, mdl, 65536, "q4_0", "q4_0", "Vulkan0 GPU", now=5000 + rr.MEASURE_KEEP_S + 5) is None, "measure: older than 90 days is dropped")
part = dict(b, offloaded=(40, 65))
check(rr.measure_save(cd, cmd, "Vulkan0 GPU", part) is None and rr.measure_save(cd, ["x"], "d", b) is None, "measure: a partial offload or a missing model says nothing about the all-on-GPU need: not stored")
need, same = rr.measured_need_gib(m, 512)
check(abs(need - (11500.25 + 1024 + 149.62 + 801.5) / 1024) < 1e-9 and same, "measured_need_gib: model + KV + recurrent + compute")
need2, same2 = rr.measured_need_gib(m, 4096, 0.2)
check(not same2 and abs(need2 - (11500.25 + 1024 + 149.62) / 1024 - 0.2) < 1e-9, "measured_need_gib: another ubatch size takes the table's compute figure")
need3, _ = rr.measured_need_gib(dict(m, compute_mib=50.0), None, 0.8)
check(abs(need3 - (11500.25 + 1024 + 149.62) / 1024 - 0.8) < 1e-9, "measured_need_gib: a small measured compute buffer never plans below the table's fixed figure")
exe = os.path.join(TMP, "llama-server-a"); open(exe, "w").write("x" * 10)
exe2 = os.path.join(TMP, "llama-server-b"); open(exe2, "w").write("x" * 99)
cmdx = [exe] + cmd[1:]
rr.measure_save(cd, cmdx, "Vulkan0 GPU", b, now=6000)
check(rr.measure_load(cd, mdl, 65536, "q4_0", "q4_0", "Vulkan0 GPU", now=6001, exe=exe) and rr.measure_load(cd, mdl, 65536, "q4_0", "q4_0", "Vulkan0 GPU", now=6001, exe=exe2) is None,
      "measure: a measurement belongs to its llama-server binary")

# --- the dense fit uses it ---
ctx = 16384
f_tab = gp.dense_gpu_fit(mdl, ctx, 64 * GIB, "q4_0", 0.0, 0.0, 0.20)
f_mea = gp.dense_gpu_fit(mdl, ctx, 64 * GIB, "q4_0", 0.0, 0.0, 0.20, measured_gib=13.0)
check(not f_tab.get("measured") and f_mea["measured"] and abs(f_mea["need_gib"] - 13.0) < 1e-9 and f_mea["fits"] and not gp.dense_gpu_fit(mdl, ctx, 12 * GIB, "q4_0", 0.0, 0.0, 0.20, measured_gib=13.0)["fits"],
      "dense_gpu_fit: measured_gib replaces the table sum")
mc.CACHE_DIR = os.path.join(TMP, "launcher_cache")
mc.REAL_ROOM.clear()
a = types.SimpleNamespace(vram="15.04", room_gib=None, gpu_margin=None)
class M:
    windows = True
    def free_vram_bytes(self): return None
t0 = mc.dense_gpu_fit_pair(a, mdl, ctx, M(), with_q4=True)
check(t0 and not t0[3].get("measured"), "dense_gpu_fit_pair: no GPU known / nothing stored -> the tables")
mc.REAL_ROOM["device"] = "AMD Radeon RX 9070 XT"
cmd2 = ["no-such-llama-server", "-m", mdl, "-c", str(ctx), "-ctk", "q4_0", "-ctv", "q4_0"]
rr.measure_save(mc.CACHE_DIR, cmd2, "AMD Radeon RX 9070 XT", {"gpu": {"model": 9000.0, "kv": 700.0, "rs": 100.0, "compute": 500.0}, "cpu": {}, "offloaded": (24, 24)})
t1 = mc.dense_gpu_fit_pair(a, mdl, ctx, M(), with_q4=True)
check(t1 and t1[3]["measured"] and not t1[1].get("measured") and abs(t1[3]["need_gib"] - (9000 + 700 + 100 + 500) / 1024) < 0.3, f"dense_gpu_fit_pair: the q4_0 rung uses the stored load, the others the tables ({t1[3]['need_gib']:.2f} GiB)")

# --- gpu_stage: the learned ceiling lowers the room ---
class SM:
    windows = True
    def list_devices(self, ls): return "Available devices:\n  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15400 MiB free)\n"
    def pick_gpu_full(self, listing): return ("Vulkan0", "AMD Radeon RX 9070 XT", 15400, 16304)
    def nvidia_reports(self): return False
    def win_dedicated_vram(self): return {"AMD Radeon RX 9070 XT": 16304 << 20}
    others = 2400
    def gpu_process_text(self): return line(100, "dedicated", self.others) if self.others else ""
def stage(m):
    a = types.SimpleNamespace(vram=None, cpu_only=False, gpu_margin=None)
    said = []; sv = mc.say; mc.say = lambda t, err=False: said.append(t)
    try:
        mc.gpu_stage(a, "llama-server", True, m)
    finally:
        mc.say = sv
    return a, said
a1, s1 = stage(SM())
check(abs(a1.room_gib - (16304 - 2400 - 300) / 1024) < 0.01 and mc.REAL_ROOM.get("room_mib") == 16304 - 2400 - 300, f"gpu_stage: the pre-load room as before; the watcher gets room_mib {mc.REAL_ROOM.get('room_mib')}")
rr.room_learn(mc.CACHE_DIR, "AMD Radeon RX 9070 XT", 12000, 2000)
a2, s2 = stage(SM())
check(abs(a2.room_gib - 12000 / 1024) < 0.01 and any("before the card paged" in x for x in s2), f"gpu_stage: a learned ceiling below the pre-load room wins ({getattr(a2, 'room_gib', None)})")
m0 = SM(); m0.others = 0
a3, s3 = stage(m0)
check(abs(a3.room_gib - 12000 / 1024) < 0.01 and any("could not be read" in x for x in s3), "gpu_stage: counters unreadable -> the ceiling still applies")
sys.exit(1 if fails else 0)
