"""LAUNCHER-991: Windows commit accounting for the whole llama-server, the pre-load refusal and the runtime guard.

Why: on the PC (AMD RX 9070 XT, Vulkan) the driver charges GPU allocations to Windows' commit charge (RAM + page file). Loading Qwen3.8-27B at 64k / 128k
took llama-server to 15.7 / 17.6 GB of commit; with a browser, the Claude app and python tools open the commit limit ran out and Windows killed claude.exe,
dwm.exe and codex.exe (event log "low virtual memory", 2026-11-09, twice). The old cap (bin/moe_cache.py vram_for_commit: free commit minus a fixed 4 GiB)
ignored the KV cache, the compute buffers, copied weights and the prompt cache, and llama.cpp's --fit for a dense model ignores it altogether.

What is commit: private memory and GPU allocations. NOT commit: file-backed read-only mappings (mmapped weights, the plugin's file-view cache). So moving KV
cache to RAM never lowers commit (it becomes private memory); moving mmapped weights (experts, FFN) to RAM does, as do q8_0 KV, a smaller -ub and a smaller
prompt cache. Everything here is pure (numbers in, numbers out) so tests run with fake commit figures; bin/moe_cache.py supplies the probes.

The estimate is NOT measured on the PC yet (no model loads were allowed while this was written); constants are marked and err high."""
import math
import os
import re
import threading
import time

GiB = 2**30
MiB = 2**20
FLOOR_MIN = 6 * GiB          # free commit left after the load: at least this ...
FLOOR_FRAC = 0.125           # ... and at least this share of the commit limit (owner rule 2026-11-09)
HARD_FLOOR = 2 * GiB         # runtime: under this llama-server is stopped cleanly, before Windows starts killing programs
PROCESS_GIB = 1.0            # llama-server itself, its DLLs, the Vulkan runtime, thread stacks (NOT measured; errs high)
VK_CONTEXT_GIB = 0.3         # driver context besides the buffers
DENSE_COMPUTE_BASE_GIB, DENSE_COMPUTE_PER_512_GIB = 0.20, 0.15     # dense compute buffer: 0.20 GiB at -ub 512 (PC 27B, 2026-11-08), +0.15 GiB per further 512 (guess)
HOST_COMPUTE_GIB = 0.25      # the CPU side's graph buffer
DEFAULT_CACHE_RAM_MIB = 8192  # llama-server's --cache-ram default
UB_MIN = 512


def soft_floor(limit):
    """Free commit that must remain after the load (and that triggers the warning at run time): max(6 GB, 12.5 % of the commit limit)."""
    return max(FLOOR_MIN, FLOOR_FRAC * (limit or 0))


def linux_floors(total):
    """(soft, hard) for MemAvailable on Linux: MemAvailable counts reclaimable page cache (mmapped weights), so low means real pressure. The soft line must stay
    under the plugin's own RAM margin (max(4 GiB, 12.5 %)), or a server that fills RAM on purpose would warn all the time."""
    return max(2 * GiB, 0.04 * (total or 0)), 1 * GiB


# ---------------------------------------------------------------- facts about the model


def facts_from_model(path):
    """The model's sizes the estimate needs, from the GGUF header (planner/gguf_plan.py): total, emb (token embedding, stays in host memory), n_layers,
    expert_per_layer (MoE), ffn_per_layer (dense), dense, kv(ctx, q8) -> bytes. Raises on an unreadable file."""
    import gguf_plan as gp
    lay = gp.model_layout(path)
    total, emb, ffn = 0, 0, {}
    for p in gp.shards(path):
        for name, size in gp.read_tensors(p):
            total += size
            emb += size if name == "token_embd.weight" else 0
            m = re.match(r"blk\.(\d+)\.ffn_(up|down|gate|gate_up)\.weight$", name)
            if m:
                ffn[int(m.group(1))] = ffn.get(int(m.group(1)), 0) + size
    dense = not lay["n_layers"]
    n_layers = lay["n_layers"] or len(ffn)
    return {"total": total, "emb": emb, "n_layers": n_layers, "dense": dense, "expert_per_layer": lay["expert_bytes_per_layer"],
            "ffn_per_layer": (sum(ffn.values()) / len(ffn)) if ffn else 0.0, "ffn_layers": sorted(ffn),
            "kv": lambda ctx, q8: gp._kv_or_flat(path, ctx, q8)}


def compute_bytes(f, ctx, ub):
    """Device compute buffer: MoE from the measured table (errs high), dense by the small formula above."""
    if f["dense"]:
        return (DENSE_COMPUTE_BASE_GIB + DENSE_COMPUTE_PER_512_GIB * max(0, ub - UB_MIN) / 512) * GiB
    try:
        import gguf_plan as gp
        mib = gp.compute_buffer_guess_mib(ctx, ub)
    except Exception:
        mib = None
    return (mib if mib else 2048) * MiB


# ---------------------------------------------------------------- the estimate


def estimate(f, L, windows=True):
    """Bytes of commit the whole llama-server asks for in layout L (a dict, see layout_from_cmd): {part: bytes, ..., "total": bytes}.
    On Windows the GPU's allocations count; on Linux (windows=False) they do not and only host memory is summed."""
    model = max(0.0, f["total"] - f["emb"])
    cpu_only = L.get("cpu_only")
    off_moe = min(L.get("n_cpu_moe") or 0, f["n_layers"]) * f["expert_per_layer"]
    off_ffn = min(L.get("ffn_off") or 0, f["n_layers"]) * f["ffn_per_layer"]
    gpu_w = 0.0 if cpu_only else max(0.0, model - off_moe - off_ffn)
    kv = float(f["kv"](L["ctx"], L.get("kv_q8", False)))
    comp = compute_bytes(f, L["ctx"], L.get("ub", UB_MIN))
    gpu = 0.0 if cpu_only else gpu_w + kv + comp + VK_CONTEXT_GIB * GiB
    spill = 0.0
    vram = L.get("vram_free")
    if L.get("fit") and vram and gpu > vram:        # llama.cpp --fit puts what does not fit on the GPU in host memory (mmapped unless copies are on)
        spill = gpu - vram
        gpu = vram
    cpu_w = (model - gpu_w + f["emb"]) if not cpu_only else model + f["emb"]
    cpu_w += spill
    parts = {
        "gpu_weights": min(gpu_w, gpu) if not cpu_only else 0.0,
        "kv": kv, "compute": comp,
        "gpu_total": gpu,
        "copied_weights": cpu_w if L.get("copy_weights") else 0.0,
        "host_kv": kv if cpu_only else 0.0,
        "prompt_cache": L.get("cache_ram_mib", DEFAULT_CACHE_RAM_MIB) * MiB,
        "plugin_cache": L.get("plugin_private_gib", 0.0) * GiB,
        "process": (PROCESS_GIB + HOST_COMPUTE_GIB) * GiB + L.get("extra_host", 0.0),
        "mm_projector": L.get("mm_bytes", 0.0),
    }
    host = parts["copied_weights"] + parts["host_kv"] + parts["prompt_cache"] + parts["plugin_cache"] + parts["process"] + parts["mm_projector"]
    parts["total"] = (gpu if windows else 0.0) + host
    return parts


def parts_words(p):
    """The estimate in one line, in GB: what the commit is made of."""
    g = lambda k: p[k] / 1e9
    bits = [("GPU buffers", p["gpu_total"] if p.get("total") and p["total"] >= p["gpu_total"] else 0.0), ("copied weights", p["copied_weights"]), ("KV in RAM", p["host_kv"]),
            ("prompt cache", p["prompt_cache"]), ("plugin cache", p["plugin_cache"]), ("process + buffers", p["process"] + p["mm_projector"])]
    return " + ".join(f"{b / 1e9:.1f} {n}" for n, b in bits if b >= 0.05e9) + f" = {g('total'):.1f} GB"


# ---------------------------------------------------------------- the decision


def need_with_floor(est, limit):
    return est["total"] + soft_floor(limit)


def fits(est, free, limit):
    return free - est["total"] >= soft_floor(limit)


def _more_weights_to_ram(f, L, est, free, limit):
    """Move just enough mmapped weights (experts of MoE layers, FFN of dense layers) off the GPU to fit; None when even all of them do not."""
    per = f["expert_per_layer"] if not f["dense"] else f["ffn_per_layer"]
    key = "n_cpu_moe" if not f["dense"] else "ffn_off"
    if per <= 0 or L.get("cpu_only"):
        return None
    deficit = need_with_floor(est, limit) - free
    cur = min(L.get(key) or 0, f["n_layers"])
    k = int(math.ceil(deficit / per))
    new = min(cur + max(1, k), f["n_layers"])
    if new <= cur:
        return None
    L2 = dict(L)
    L2[key] = new
    if L.get("fit"):
        L2["fit"] = False           # an explicit placement replaces llama.cpp's --fit (the command builder writes -ngl 99 and -ot)
    return L2, f"{new - cur} more layer(s) of {'experts' if key == 'n_cpu_moe' else 'FFN weights'} in RAM ({new} of {f['n_layers']})"


def decide(f, L, free, limit, locked=(), allow_q8=True, windows=True):
    """Pre-load check. free / limit: free commit and commit limit in bytes. locked: knobs the user fixed ("ub", "kv", "cache", "weights"); they are never touched.
    Tries, cheapest in speed first: mmap instead of copies, smaller -ub, smaller prompt cache, q8_0 KV cache, then more weights in RAM (never shorter context).
    Returns {"ok", "layout", "steps": [text], "est", "need": bytes incl. the floor, "floor"}; ok False = refuse (the layout is the smallest tried)."""
    floor = soft_floor(limit)
    est = estimate(f, L, windows)
    steps, cur = [], dict(L)
    def done():
        return {"ok": True, "layout": cur, "steps": steps, "est": est, "need": est["total"] + floor, "floor": floor}
    if fits(est, free, limit):
        return done()
    def settle(new, text):
        nonlocal cur, est
        cur, est = new, estimate(f, new, windows)
        steps.append(text)
        return fits(est, free, limit)
    if cur.get("copy_weights") and "weights" not in locked:
        if settle(dict(cur, copy_weights=False), "mmap instead of copied weights"):
            return done()
    if "ub" not in locked and cur.get("ub", UB_MIN) > UB_MIN:
        old_ub = cur["ub"]
        while cur["ub"] > UB_MIN and not fits(est, free, limit):
            cur = dict(cur, ub=max(UB_MIN, cur["ub"] // 2)); est = estimate(f, cur, windows)
        steps.append(f"-ub {cur['ub']} (was {old_ub})")
        if fits(est, free, limit):
            return done()
    if "cache" not in locked and cur.get("cache_ram_mib", DEFAULT_CACHE_RAM_MIB) > 512:
        old_c = cur.get("cache_ram_mib", DEFAULT_CACHE_RAM_MIB)
        for mib in (1024, 512):
            if cur.get("cache_ram_mib", DEFAULT_CACHE_RAM_MIB) > mib and not fits(est, free, limit):
                cur = dict(cur, cache_ram_mib=mib); est = estimate(f, cur, windows)
        steps.append(f"prompt cache {cur['cache_ram_mib']} MiB (was {old_c})")
        if fits(est, free, limit):
            return done()
    if "kv" not in locked and allow_q8 and not cur.get("kv_q8") and not cur.get("cpu_only"):
        if settle(dict(cur, kv_q8=True), "q8_0 KV cache (half the KV memory; approximate, KL 0.0079 at 4k)"):
            return done()
    if "weights" not in locked:
        r = _more_weights_to_ram(f, cur, est, free, limit)
        if r:
            cur, est = r[0], estimate(f, r[0], windows)
            steps.append(r[1])
            if fits(est, free, limit):
                return done()
    return {"ok": False, "layout": cur, "steps": steps, "est": est, "need": est["total"] + floor, "floor": floor}


def refusal_text(d, free, limit, ram_total=None):
    """The refusal line: how much free commit the smallest layout needs, how much there is, and what to do about it."""
    need = d["need"]
    short = max(0.0, need - free)
    msg = f"moe-cache: not starting: this model at this context needs {need / 1e9:.1f} GB free commit (the smallest layout: {parts_words(d['est'])}, plus {d['floor'] / 1e9:.1f} GB that must stay free so Windows does not kill your programs); you have {free / 1e9:.1f} GB: "
    pf = ""
    if ram_total and limit:
        want = max(0.0, limit - ram_total) + short
        pf = f" or set the page file to at least {int(math.ceil(want / 2**30 / 2) * 2)} GB (Windows: System Properties > Advanced > Performance > Advanced > Virtual memory; system managed also works)"
    return msg + "close apps (the Claude app, browsers and python tools hold commit)" + pf + ". --no-commit-check starts anyway."


def apply_plan_text(steps):
    return "moe-cache: free commit is tight (Windows counts GPU memory as commit): smaller layout chosen to keep your programs alive: " + "; ".join(steps)


# ---------------------------------------------------------------- reading the layout from a command


def _val(cmd, *names):
    for i, x in enumerate(cmd[:-1]):
        if x in names:
            return cmd[i + 1]
    for x in cmd:
        for n in names:
            if x.startswith(n + "="):
                return x.split("=", 1)[1]
    return None


def layout_from_cmd(cmd, env, f, vram_free=None, mm_bytes=0):
    """The layout dict of a finished llama-server command line (and its environment list of (name, value) pairs)."""
    e = dict(env)
    ctx = int(_val(cmd, "-c", "--ctx-size") or 8192)
    ncmoe = _val(cmd, "--n-cpu-moe")
    ot = _val(cmd, "-ot", "--override-tensor") or ""
    ffn_off = 0
    m = re.match(r"blk\\\.\(([0-9|]+)\)\\\.ffn_", ot)
    if m:
        ffn_off = len(m.group(1).split("|"))
    ngl = _val(cmd, "-ngl", "--n-gpu-layers", "--gpu-layers")
    dev = _val(cmd, "--device", "-dev")
    ram_gib = 0.0
    if e.get("MOE_CACHE_WIN_VIEW") == "0":
        try:
            ram_gib = float(e.get("MOE_CACHE_RAM_GIB") or 0)
        except ValueError:
            pass
    cram = _val(cmd, "--cache-ram")
    try:
        cram = int(cram) if cram is not None else DEFAULT_CACHE_RAM_MIB
    except ValueError:
        cram = DEFAULT_CACHE_RAM_MIB
    if cram < 0:
        cram = DEFAULT_CACHE_RAM_MIB
    cpu_only = dev == "none" or ngl == "0"
    return {"ctx": ctx, "kv_q8": _val(cmd, "-ctk", "--cache-type-k") in ("q8_0", "q4_0"), "ub": int(_val(cmd, "-ub", "--ubatch-size") or UB_MIN) if (_val(cmd, "-ub", "--ubatch-size") or "").isdigit() else UB_MIN,
            "n_cpu_moe": (int(ncmoe) if ncmoe and ncmoe.isdigit() else 0) if ncmoe else 0, "ffn_off": ffn_off, "cpu_only": cpu_only,
            "copy_weights": (_val(cmd, "--load-mode") not in (None, "mmap")) or "--no-mmap" in cmd, "cache_ram_mib": cram, "plugin_private_gib": ram_gib,
            "fit": bool(f["dense"] and not cpu_only and ngl is None and not ot), "vram_free": vram_free, "mm_bytes": float(mm_bytes or 0)}


def _set(cmd, names, value):
    """cmd with the option `names[0]` set to value (replacing an existing one in place, else appended)."""
    cmd = list(cmd)
    for i, x in enumerate(cmd[:-1]):
        if x in names:
            cmd[i + 1] = str(value)
            return cmd
    return cmd + [names[0], str(value)]


def _drop(cmd, names):
    cmd, i = list(cmd), 0
    while i < len(cmd):
        if cmd[i] in names and i + 1 < len(cmd):
            del cmd[i:i + 2]
        else:
            i += 1
    return cmd


def apply_layout_to_cmd(cmd, old, new, f):
    """The command line for layout `new`, changed from the one for `old`. Returns (cmd, ncmoe or None, ub, kv_q8, ffn_changed)."""
    out = list(cmd)
    if new.get("copy_weights") != old.get("copy_weights") and not new.get("copy_weights"):
        out = _drop(out, ("--load-mode",))
        out += ["--load-mode", "mmap"]
        out = [x for x in out if x != "--no-mmap"]
    if new.get("ub") != old.get("ub"):
        out = _set(out, ("-ub", "--ubatch-size"), new["ub"])
        if "-b" in out and out.index("-b") + 1 < len(out):
            out[out.index("-b") + 1] = str(max(int(new["ub"]), 2048))      # as the out-of-memory retry does
    if new.get("cache_ram_mib") != old.get("cache_ram_mib"):
        out = _set(out, ("--cache-ram",), new["cache_ram_mib"])
    if new.get("kv_q8") and not old.get("kv_q8"):
        out = _set(_set(out, ("-ctk", "--cache-type-k"), "q8_0"), ("-ctv", "--cache-type-v"), "q8_0")
    ffn_changed = False
    if not f["dense"] and new.get("n_cpu_moe") != old.get("n_cpu_moe"):
        out = _set(out, ("--n-cpu-moe",), new["n_cpu_moe"])
    if f["dense"] and new.get("ffn_off", 0) != old.get("ffn_off", 0):
        n = f["n_layers"]
        layers = "|".join(str(i) for i in range(n - new["ffn_off"], n))
        out = _drop(out, ("-ot", "--override-tensor", "-fitt", "--fit-target"))
        out += ["-ot", f"blk\\.({layers})\\.ffn_(up|down|gate|gate_up)\\.weight=CPU"]
        if _val(out, "-ngl", "--n-gpu-layers", "--gpu-layers") is None:
            out += ["-ngl", "99"]            # an explicit placement: --fit off, everything but the named FFN weights on the GPU
        ffn_changed = True
    return out, new.get("n_cpu_moe"), new.get("ub"), bool(new.get("kv_q8")), ffn_changed


# ---------------------------------------------------------------- the runtime guard


def parse_mode(text):
    """MOE_CACHE_COMMIT_GUARD: 'stop' (default: warn, then stop llama-server at the hard floor), 'warn' (never stops it), 'off'."""
    t = (text or "stop").strip().lower()
    return t if t in ("stop", "warn", "off") else "stop"


class CommitGuard:
    """Polls free commit (Windows) or MemAvailable (Linux) while llama-server runs. probe() -> free bytes or None; say(text) prints loudly (terminal + the Monitor lifts
    'moe-cache: WARNING' lines); stop(reason) ends llama-server cleanly. Under the soft floor: one warning, repeated every `repeat` seconds while it lasts and when it gets
    worse by 1 GB; back over soft + 0.5 GB: one 'recovered' line. Under the hard floor: stop (after two samples in a row, or at once under half of it, so one dip is
    not fatal but a fast fall is caught), then the guard ends. The clock and sleep are injectable (tests)."""
    def __init__(self, probe, say, stop, soft, hard=HARD_FLOOR, mode="stop", interval=2.0, repeat=30.0, label="free commit", clock=time.monotonic):
        self.probe, self.say, self.stop_fn, self.soft, self.hard = probe, say, stop, float(soft), float(hard)
        self.mode, self.interval, self.repeat, self.label, self.clock = mode, interval, repeat, label, clock
        self.state, self.last_warn, self.last_warn_free, self.low = "ok", None, None, 0
        self.stopped = None
        self.min_free = None

    def poll(self):
        """One sample; returns "ok", "soft", "hard" or None (probe failed)."""
        free = self.probe()
        if free is None:
            return None
        self.min_free = free if self.min_free is None else min(self.min_free, free)
        gb = free / 1e9
        if free < self.hard:
            self.low += 1
            if self.mode == "stop" and (self.low >= 2 or free < self.hard / 2):
                self.stopped = (f"moe-cache: WARNING: {self.label} fell to {gb:.1f} GB (hard floor {self.hard / 1e9:.1f} GB): stopping llama-server now, before Windows starts closing your programs "
                                if self.label == "free commit" else
                                f"moe-cache: WARNING: {self.label} fell to {gb:.1f} GB (hard floor {self.hard / 1e9:.1f} GB): stopping llama-server now, before the system starts killing programs ") + \
                    "(close big programs, or use a smaller --ctx / start again with --max-ram; MOE_CACHE_COMMIT_GUARD=warn only warns)"
                self.say(self.stopped)
                self.stop_fn(self.stopped)
                self.state = "hard"
                return "hard"
        else:
            self.low = 0
        now = self.clock()
        if free < self.soft:
            worse = self.last_warn_free is not None and free < self.last_warn_free - 1e9
            if self.state != "soft" or worse or (self.last_warn is not None and now - self.last_warn >= self.repeat):
                self.say(f"moe-cache: WARNING: {self.label} is down to {gb:.1f} GB (warning line {self.soft / 1e9:.1f} GB, llama-server is stopped under {self.hard / 1e9:.1f} GB): "
                         "close big programs now; Windows kills programs when it runs out" if self.label == "free commit" else
                         f"moe-cache: WARNING: {self.label} is down to {gb:.1f} GB (warning line {self.soft / 1e9:.1f} GB, llama-server is stopped under {self.hard / 1e9:.1f} GB): close big programs now")
                self.last_warn, self.last_warn_free = now, free
            self.state = "soft"
            return "soft"
        if self.state == "soft" and free >= self.soft + 0.5e9:
            self.say(f"moe-cache: {self.label} is back to {gb:.1f} GB")
            self.state, self.last_warn, self.last_warn_free = "ok", None, None
        return "hard" if self.state == "hard" else ("soft" if self.state == "soft" else "ok")

    def run(self, alive, sleep=time.sleep):
        """Loop until alive() is False or the guard stopped the server."""
        while alive() and not self.stopped:
            try:
                self.poll()
            except Exception:
                pass
            sleep(self.interval)

    def start(self, alive):
        t = threading.Thread(target=self.run, args=(alive,), daemon=True)
        t.start()
        return t
