#!/usr/bin/env python3
"""R15-02: Windows GPU and memory counters in bin/moe_cache_hw.py (PDH GPU Engine + GPU Adapter Memory, DXGI, commit). The parsers and the merge run on recorded
instance names everywhere; on Windows the real readers run too (they may report nothing on a runner without a GPU, but must not raise or crash).
   python3 tests/hw_win_test.py"""
import os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_hw as hw  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

DGPU, IGPU = (0, 0xA1B2), (0, 0xC3D4)
eng = [("pid_100_luid_0x00000000_0x0000A1B2_phys_0_eng_0_engtype_3D", 40.0), ("pid_200_luid_0x00000000_0x0000A1B2_phys_0_eng_0_engtype_3D", 25.0),
       ("pid_100_luid_0x00000000_0x0000A1B2_phys_0_eng_3_engtype_Compute", 70.0), ("pid_100_luid_0x00000000_0x0000A1B2_phys_0_eng_5_engtype_Copy", 5.0),
       ("pid_300_luid_0x00000000_0x0000C3D4_phys_0_eng_0_engtype_3D", 90.0), ("garbage", 50.0)]
u = hw.parse_gpu_engine(eng)
check(u[DGPU] == 70.0 and u[IGPU] == 90.0 and len(u) == 2, f"engine utilisation per adapter: the busiest engine, processes added on one engine ({u})")
check(hw.parse_gpu_engine([("pid_1_luid_0x00000000_0x0000A1B2_phys_0_eng_0_engtype_3D", 80.0), ("pid_2_luid_0x00000000_0x0000A1B2_phys_0_eng_0_engtype_3D", 60.0)])[DGPU] == 100.0, "capped at 100 %")
check(hw.parse_gpu_engine([]) == {} and hw.parse_gpu_engine(None) == {}, "no counters: empty")
m = hw.parse_adapter_memory([("luid_0x00000000_0x0000A1B2_phys_0", 3.0e9), ("luid_0x00000000_0x0000C3D4_phys_0", 2.0e8), ("junk", 1)])
check(m == {DGPU: 3.0e9, IGPU: 2.0e8}, "adapter memory per LUID")
check(hw.luid_key("luid_0x00000001_0xFFFFFFFF_phys_0") == (1, 0xFFFFFFFF) and hw.luid_key("nope") is None, "LUID parsing")

ad = [{"name": "AMD Radeon RX 9070 XT", "luid": DGPU, "dedicated": 16 * 2**30, "shared": 16 * 2**30, "budget": 15 * 2**30, "usage": 1 * 2**30, "software": False},
      {"name": "AMD Radeon(TM) Graphics", "luid": IGPU, "dedicated": 512 * 2**20, "shared": 16 * 2**30, "budget": 15 * 2**30, "usage": 0, "software": False},
      {"name": "Microsoft Basic Render Driver", "luid": (0, 1), "dedicated": 0, "shared": 8 * 2**30, "budget": 8 * 2**30, "usage": 0, "software": True}]
check(hw.pick_adapter(ad)["name"].endswith("9070 XT"), "the discrete card is the adapter (most dedicated memory)")
check(hw.pick_adapter(ad[2:])["software"], "only a software adapter: it is used")
check(hw.pick_adapter([]) is None, "no adapter: None")

class FakeCounters:
    def __init__(self, ok=True): self.ok, self.collected = ok, 0
    def collect(self): self.collected += 1; return self.ok
    def read(self, key): return {"engine": eng, "dedicated": [("luid_0x00000000_0x0000A1B2_phys_0", 3.5e9)], "shared": [("luid_0x00000000_0x0000A1B2_phys_0", 0.2e9)]}[key]
g = hw.win_gpu_stats(counters=FakeCounters(), adapters=ad)
check(g and g["name"].endswith("9070 XT") and g["util"] == 70.0 and g["mem_used"] == 3.5e9 and g["mem_total"] == 16 * 2**30, f"the AMD card on Windows: name, util, used, total ({ {k: g[k] for k in ('name', 'util', 'mem_used', 'mem_total')} })")
check(g["mem_budget"] == 15 * 2**30 and g["mem_shared"] == 0.2e9 and g["cards"] == 2 and g["temp"] is None and g["power"] is None, "DXGI budget, shared usage, card count; temperature and power stay None (R15-07)")
g2 = hw.win_gpu_stats(counters=FakeCounters(ok=False), adapters=ad)
check(g2 and g2["util"] is None and g2["mem_total"], "a failed collection: no utilisation, the memory facts remain")
check(hw.win_gpu_stats(counters=FakeCounters(), adapters=[]) is None, "no adapters: None")
class Boom(FakeCounters):
    def read(self, key): raise OSError("pdh")
check(hw.win_gpu_stats(counters=Boom(), adapters=ad) is None, "a reader that raises gives None, never an exception")
ig = hw.merge_gpu(ad[1], 12.0, 3e8, 4e9, None, 2)
check(ig["mem_total"] == 15 * 2**30 and ig["mem_used"] == 4e9, "an integrated GPU: the shared-memory budget is its memory, shared usage is its use")

import ctypes
check(hw.pdh_status(ctypes.c_int(0x800007D2).value) == hw.PDH_MORE_DATA and hw.pdh_status(-2147481646) == 0x800007D2 and hw.pdh_status(0) == 0,
      "PDH_MORE_DATA arrives from ctypes as a negative C int and is still recognised (this comparison was the bug that made the Windows reader return nothing)")
ms = hw.parse_memory_status(48 * 2**30, 20 * 2**30, 32 * 2**30, 9 * 2**30)
check(ms["commit_total"] == 48 * 2**30 and ms["commit_used"] == 28 * 2**30 and ms["commit_free"] == 20 * 2**30 and ms["ram_free"] == 9 * 2**30, "commit from the memory status fields")

# the machine-independent part of gpu_stats(): off Windows it is unchanged (nvidia-smi, then sysfs)
if not hw.WINDOWS:
    r_ = hw.gpu_stats()
    check(r_ is None or isinstance(r_, dict), "off Windows gpu_stats() still answers a dict or None (nvidia-smi, then sysfs)")
else:
    t0 = time.time()
    a = hw.dxgi_adapters()
    check(isinstance(a, list) and all("luid" in x and "name" in x for x in a), f"DXGI adapters on this Windows: {[(x['name'], x['dedicated'] >> 20, x['budget']) for x in a]}")
    s1 = hw.win_gpu_stats()
    time.sleep(1.0)
    s2 = hw.win_gpu_stats()
    check(s2 is None or (isinstance(s2, dict) and "mem_total" in s2), f"the real PDH + DXGI reader answers a dict or None without raising ({s2 and {k: s2[k] for k in ('name', 'util', 'mem_used', 'mem_total')}})")
    c = hw._WIN_GPU.get("c")
    if c is not None:
        counts = {k: len(c.read(k)) for k in ("engine", "dedicated", "shared")}
        print(f"       PDH instances read on this Windows: {counts}")
        check(all(isinstance(c.read(k), list) for k in counts), "the real PDH reader returns lists (counted above; a runner without a GPU may have zero instances, the PC must not)")
        if counts["dedicated"] or counts["shared"]:
            check(s2 is not None and s2.get("mem_used") is not None, "PDH found adapter memory instances, so the merged answer carries a memory figure")
    st = hw.win_memory_status()
    check(st and st["commit_total"] > st["ram_total"] * 0.5 and 0 < st["commit_free"] <= st["commit_total"], f"commit from GlobalMemoryStatusEx ({st and round(st['commit_free'] / 2**30, 1)} GiB free)")
    check(time.time() - t0 < 20, "the readers do not hang")

print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
