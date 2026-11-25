#!/usr/bin/env python3
"""weights_ram_test (LAUNCHER-1165): the plan warns in plain words when the model's weights that stay in RAM will not fit beside the programs already open (32 GB PC,
~18 GB in use, Flash-Next at 65k: mmap pages took the available RAM to ~0, chain 2026-11-13). Mocked RAM figures and a mocked model layout; no model file, no probe.
Cases: fits / does not fit / a shorter context named (never applied) / the user's --ctx kept / --exact changes nothing / plugin path / -ngl 0 / Linux and Windows alike."""
import contextlib, io, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
sys.path.insert(0, os.path.join(HERE, "..", "planner"))
import moe_cache as mc  # noqa: E402
import gguf_plan  # noqa: E402

GIB = 2 ** 30
FAILS = []


def check(c, w):
    print(("  ok   " if c else "  FAIL ") + w)
    if not c:
        FAILS.append(w)


class M:
    def __init__(self, avail, total=31.1 * GIB, windows=True):
        self.avail, self.total, self.windows = avail, total, windows
    def mem_available(self): return self.avail
    def mem_for_server(self, ram_gb): return self.avail
    def mem_total(self): return self.total


# a Flash-Next-like layout: 48 expert layers of 0.35 GB, 2 GB other weights (0.6 GB token_embd), a 6 GB n-gram table read from the file
LAY = {"n_layers": 48, "expert_bytes_per_layer": 0.35e9, "other_bytes": 2e9, "table_bytes": 6e9, "total_bytes": 2e9 + 6e9 + 48 * 0.35e9,
       "mtp_blocks": 0, "expert_bytes_by_layer": {i: 0.35e9 for i in range(48)}, "cpu_only_bytes": 0.6e9}
KV_PER_TOKEN = 60e3          # bytes of KV per token (made up: 65k -> 3.9 GB)
gguf_plan.model_layout = lambda path: dict(LAY)
gguf_plan.kv_bytes = lambda path, ctx, kv_q8=False, n_seq=4: ctx * KV_PER_TOKEN


class A:
    def __init__(self, ctx=None, exact=False, from_pick=False):
        self.ctx, self.exact, self.ctx_from_pick = ctx, exact, from_pick


def stage(avail, cmd, ctx=65536, user_ctx=None, exact=False, stock=True, windows=True, total=31.1 * GIB, from_pick=False):
    out = io.StringIO()
    c0 = list(cmd)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        d = mc.weights_ram_stage(A(user_ctx, exact, from_pick), "m.gguf", user_ctx or ctx, cmd, M(avail, total, windows), stock)
    return d, out.getvalue(), cmd == c0


CMD = lambda n: ["llama-server", "-m", "m.gguf", "-ngl", "99", "--n-cpu-moe", str(n), "-c", "65536"]

# 1. fits: 20 expert layers on the CPU (7 GB) + embd 0.6 = 7.6 GB; 28 GiB available on 31.1 GiB (keep 4 GiB) -> room ~25.8 GB; the 6 GB table fits beside it too
for win in (True, False):
    d, say, same = stage(28 * GIB, CMD(20), windows=win)
    check(d is None and "moe-cache" not in say and same, f"fits ({'Windows' if win else 'Linux'}): no line, command unchanged")

# 2. does not fit: 40 layers on the CPU (14 GB) + 0.6 = 14.6 GB; the PC case, 31.1 GiB with ~18 GiB in use -> 13.4 GiB available, room 9.4 GiB (~10.1 GB)
for win in (True, False):
    d, say, same = stage(13.4 * GIB, CMD(40), windows=win)
    check(d is not None and d["warn"] and "moe-cache: warning: this model needs about 14.6 GB of RAM beside your open apps; 10.1 GB is free. Close apps or use a smaller context/model." in say,
          f"does not fit ({'Windows' if win else 'Linux'}): the plain warning ({say.splitlines()[0][:150] if say else ''})")
    check(same, "the command is never changed")
# no shorter context fits: 32k frees 1.97 GB, 16k 2.95, 8k 3.44: 14.6 - 3.44 = 11.2 > 10.1
check(d["ctx_hint"] is None and "--ctx" not in say, "no shorter context fits by the estimate: none named")
check("table from the file" in say, "the 6 GB table that cannot fit beside it is named in a note")

# 3. a shorter context named (never applied): 15.5 GiB available -> room 12.35 GB; 32k: 14.6 - 1.97 = 12.63 no; 16k: 11.65 yes
d, say, same = stage(15.5 * GIB, CMD(40))
check(d is not None and d["ctx_hint"] == 16384 and "--ctx 16384" in say and "does not shorten the context by itself" in say and same,
      f"a shorter context that fits is named, the plan keeps 65536 ({d and d['ctx_hint']})")

# 4. the user's own --ctx: kept, no shorter context offered as a pick
d, say, same = stage(15.5 * GIB, CMD(40), user_ctx=65536)
check(d is not None and d["ctx_hint"] is None and "your --ctx 65536 is kept" in say and same, "user --ctx: the warning, the context kept, no hint")

# 4b. a context the model picker filled (a.ctx set, ctx_from_pick) is not the user's own --ctx: the shorter-context hint still shows
d, say, same = stage(15.5 * GIB, CMD(40), user_ctx=65536, from_pick=True)
check(d is not None and d["ctx_hint"] == 16384 and "--ctx 16384" in say and "is kept" not in say and same, "picker-filled context: the hint shows, not 'your --ctx is kept'")

# 5. --exact: the warning still says what it sees (read-only), nothing changes
d, say, same = stage(13.4 * GIB, CMD(40), exact=True)
check(d is not None and "needs about" in say and same, "--exact: warned, command unchanged")

# 6. plugin path: the plugin sizes its RAM itself; need = its 4 GiB floor + 2 GiB streamed-file term (file 24.8 GB > room) = 6.4 GB
d, say, same = stage(13.4 * GIB, CMD(40), stock=False)
check(d is not None and not d["warn"] and "warning" not in say and "table from the file" in say, "plugin path, 13.4 GiB available: no warning, the table note only")
d, say, same = stage(9 * GIB, CMD(40), stock=False)
check(d is not None and d["warn"] and abs(d["need"] - 6 * GIB) < 1e6 and "assumed, not measured" in say and d["ctx_hint"] is None, f"plugin path, 9 GiB available: warned ({d and d['need']/1e9:.2f} GB)")
d, say, same = stage(30 * GIB, CMD(40), stock=False)
check(d is None, "plugin path, 30 GiB available: the file fits, need 4 GiB, no line")

# 7. -ngl 0: every weight but the table in RAM (18.8 GB); 20 GiB available -> room 17.2 GB; no context hint (no GPU to take layers)
d, say, same = stage(20 * GIB, ["llama-server", "-m", "m.gguf", "-ngl", "0", "--n-cpu-moe", "999"])
check(d is not None and d["warn"] and abs(d["need"] - (LAY["total_bytes"] - LAY["table_bytes"])) < 1e6 and d["ctx_hint"] is None, f"-ngl 0: the whole file but the table counts ({d and d['need']/1e9:.1f} GB)")

# 8. pure function: missing figures -> None; 64 GiB machine keeps 8 GiB; the default model's real header figures (75.8 GB IQ3_XXS, 28.8 GB n-gram table; laptop header read)
check(mc.weights_ram_check(10e9, None, 31.1 * GIB, ctx=8192) is None and mc.weights_ram_check(0, 10 * GIB, 31.1 * GIB, ctx=8192) is None, "no figure: no line")
d = mc.weights_ram_check(25e9, 30 * GIB, 64 * GIB, ctx=8192)
check(d is not None and abs(d["keep"] - 8 * GIB) < 1, "64 GiB machine: 12.5 % = 8 GiB kept free (ram_keep_bytes)")
d = mc.weights_ram_check(6 * GIB, 13.4 * GIB, 31.1 * GIB, ctx=65536, stock_path=False, table_bytes=28.8e9)
check(d is not None and not d["warn"] and "28.8 GB table" in d["lines"][0], "Flash-Next on the PC case (plugin path): the table note, not the warning (the table's resident share is not measured)")
d = mc.weights_ram_check(6 * GIB, 40 * GIB, 64 * GIB, ctx=65536, stock_path=False, table_bytes=28.8e9)
check(d is None or not d["warn"], "Flash-Next on a 64 GB PC with 40 GiB free: no warning")

# 9. a layout the reader cannot give: one line, no crash, nothing returned
gguf_plan.model_layout = lambda path: (_ for _ in ()).throw(OSError("no file"))
d, say, same = stage(13.4 * GIB, CMD(20))
check(d is None and "could not estimate the RAM the weights need" in say and same, "unreadable model: one line, no change")

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
