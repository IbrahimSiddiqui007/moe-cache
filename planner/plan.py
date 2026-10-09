#!/usr/bin/env python3
"""Placement planner. Reads the GGUF header, looks at free VRAM and RAM, and prints the llama-server settings:
how many layers keep their experts on the GPU (--n-cpu-moe), how big the expert cache is, and the warm start settings.
usage: plan.py MODEL.gguf [--ctx 32768] [--kv f16|q8_0] [--mtp] [--vram-gib X] [--ram-gib X] [--profile FILE] [--check NCMOE] [--json]
Constants are calibrated on one machine (see doc.md): CUDA context + compute buffers ~0.65 GiB, safety margin 1.0 GiB (llama.cpp auto-fit uses 1 GiB), process RAM overhead 3.0 GiB."""
import argparse, glob, json, os, re, subprocess, sys, collections
try:
    import gguf
except ImportError:                      # the built-in reader below does not need the package
    gguf = None
GIB = 2**30


class _Field:
    def __init__(self, v): self.v = v
    def contents(self): return self.v


class _Tensor:
    def __init__(self, name, n_bytes): self.name, self.n_bytes = name, n_bytes


class MiniGGUF:
    """Reads only the header of a GGUF file (metadata and tensor table). It needs no table of quantisation types: a tensor's size is the
    distance to the next tensor's offset, so models that use types newer than the installed python 'gguf' package still plan correctly."""
    SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def __init__(self, path):
        import struct, os
        self.fields, self.tensors = {}, []
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF": raise ValueError(f"{path}: not a GGUF file")
            def rd(fmt): n = struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))[0]
            def rstr(): return f.read(rd("<Q")).decode("utf-8", "replace")
            def rval(t):
                if t in self.SCALAR: return rd(self.SCALAR[t])
                if t == 8: return rstr()
                if t == 9:
                    et, n = rd("<I"), rd("<Q")
                    if et in self.SCALAR and et != 7:
                        fmt = self.SCALAR[et]; sz = struct.calcsize(fmt)
                        return list(struct.unpack("<" + str(n) + fmt[1:], f.read(n * sz)))
                    return [rval(et) for _ in range(n)]
                raise ValueError(f"unknown GGUF value type {t}")
            version, n_tensors, n_kv = rd("<I"), rd("<Q"), rd("<Q")
            if version < 2: raise ValueError("GGUF version 1 is not supported")
            for _ in range(n_kv):
                key = rstr(); t = rd("<I"); self.fields[key] = _Field(rval(t))
            info = []
            for _ in range(n_tensors):
                name = rstr(); nd = rd("<I"); [rd("<Q") for _ in range(nd)]; rd("<I"); info.append((name, rd("<Q")))
            align = int(self.fields["general.alignment"].v) if "general.alignment" in self.fields else 32
            data_off = (f.tell() + align - 1) // align * align
            size = os.path.getsize(path) - data_off
        order = sorted(range(len(info)), key=lambda i: info[i][1])
        sizes = {}
        for k, i in enumerate(order):
            end = info[order[k + 1]][1] if k + 1 < len(order) else size
            sizes[i] = max(0, end - info[i][1])
        self.tensors = [_Tensor(info[i][0], sizes[i]) for i in range(len(info))]


def open_gguf(path):
    if gguf is not None and not os.environ.get("MOE_PLAN_MINI"):
        try:
            return gguf.GGUFReader(path)
        except Exception:
            pass                          # unknown quantisation type or similar: use the built-in reader
    return MiniGGUF(path)

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--ctx", type=int, default=32768); ap.add_argument("--kv", default="f16")
ap.add_argument("--mtp", action="store_true"); ap.add_argument("--vram-gib", type=float); ap.add_argument("--ram-gib", type=float)
ap.add_argument("--profile"); ap.add_argument("--check", type=int); ap.add_argument("--json", action="store_true"); a = ap.parse_args()

r = open_gguf(a.model)
def val(key, default=None):
    f = r.fields.get(key)
    if f is None: return default
    try: v = f.contents()
    except Exception: v = f.parts[-1][0]
    if isinstance(v, (list, tuple)) or hasattr(v, "__len__") and not isinstance(v, str): v = max(v) if len(v) else default
    return v
arch = str(val("general.architecture"))
n_blocks = int(val(f"{arch}.block_count")); n_exp = int(val(f"{arch}.expert_count")); n_used = int(val(f"{arch}.expert_used_count"))
nextn = int(val(f"{arch}.nextn_predict_layers", 0) or 0); n_main = n_blocks - nextn
kv_heads = int(val(f"{arch}.attention.head_count_kv", 0) or val(f"{arch}.attention.head_count"))
heads = int(val(f"{arch}.attention.head_count")); embd = int(val(f"{arch}.embedding_length"))
key_len = int(val(f"{arch}.attention.key_length", embd // heads)); val_len = int(val(f"{arch}.attention.value_length", key_len))
interval = int(val(f"{arch}.full_attention_interval", 0) or 0); ssm = val(f"{arch}.ssm.inner_size") is not None
n_attn = n_main // interval if interval else n_main

def all_tensors():
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", a.model)
    if not m:
        return list(r.tensors)
    out = []
    for i in range(1, int(m[2]) + 1):
        part = a.model[:m.start()] + f"-{i:05d}-of-{m[2]}.gguf"
        out += list(r.tensors if part == a.model else open_gguf(part).tensors)
    return out

exp_layer = collections.defaultdict(int); core = 0; mtp_core = 0; table = 0
for t in all_tensors():
    m = re.match(r"blk\.(\d+)\.", t.name); layer = int(m[1]) if m else -1
    if re.search(r"ffn_(gate_up|gate|up|down)_exps", t.name): exp_layer[layer] += int(t.n_bytes)   # gate_up = fused gate+up tensor (e.g. qwen4exp)
    elif "per_layer_token_embd" in t.name: table += int(t.n_bytes)   # n-gram lookup table (qwen4exp): read from the file when needed, never on the GPU
    elif layer >= n_main: mtp_core += int(t.n_bytes)
    else: core += int(t.n_bytes)
main_layers = [l for l in range(n_main) if l in exp_layer]
if not main_layers: sys.exit("no expert tensors (blk.N.ffn_{gate,up,down,gate_up}_exps) found: this does not look like a MoE GGUF")
per_layer = sum(exp_layer[l] for l in main_layers) / len(main_layers)
mtp_experts = sum(v for l, v in exp_layer.items() if l >= n_main)
KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
if a.kv not in KV_BYTES: sys.exit(f"--kv must be one of {', '.join(KV_BYTES)}")
kv_elem = KV_BYTES[a.kv]
kv_bytes = n_attn * a.ctx * kv_heads * (key_len + val_len) * kv_elem
rs_bytes = 0.2 * GIB if ssm else 0
fixed = 0.65 * GIB; margin = 1.0 * GIB
mtp_vram = (0.85 * GIB) if a.mtp else 0
gpu_fixed = core + kv_bytes + rs_bytes + fixed + mtp_vram + (mtp_experts + mtp_core if a.mtp else 0)
def predicted_vram(n_gpu_layers): return gpu_fixed + n_gpu_layers * per_layer

def smi(q):
    try: return float(subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode().split()[0]) * 2**20
    except Exception: return None
def amd_free():
    """free VRAM of the AMD GPU with the most VRAM (amdgpu sysfs); None when there is none. Integrated GPUs share system RAM and are not counted."""
    best = None
    for d in glob.glob("/sys/class/drm/card[0-9]*/device"):
        try:
            if open(d + "/vendor").read().strip() != "0x1002": continue
            tot = int(open(d + "/mem_info_vram_total").read()); used = int(open(d + "/mem_info_vram_used").read())
        except Exception: continue
        if tot >= 2**30 and (best is None or tot > best[0]): best = (tot, tot - used)
    return best[1] if best else None
gpu_name = "NVIDIA"
free_vram = a.vram_gib * GIB if a.vram_gib else smi("memory.free")
if free_vram is None and not a.vram_gib:
    free_vram = amd_free(); gpu_name = "AMD"
if a.check is not None:
    n_gpu = n_main - a.check
    print(f"{arch}: ncmoe {a.check} ({n_gpu} layers of experts on GPU) predicted VRAM {predicted_vram(n_gpu)/GIB:.2f} GiB (core {core/GIB:.2f}, KV {kv_bytes/GIB:.2f}, state {rs_bytes/GIB:.2f}, fixed {fixed/GIB:.2f}, experts {n_gpu*per_layer/GIB:.2f})"); sys.exit()
if free_vram is None:
    n_gpu = 0; note = "no NVIDIA or AMD GPU with its own memory found: all experts stay on the CPU (pass --vram-gib to override)"
else:
    room = free_vram - margin - gpu_fixed
    n_gpu = max(0, min(n_main, int(room // per_layer))); note = "" if room > 0 else "GPU too small even for the core weights: use fewer layers on GPU (-ngl) or a smaller context"
ncmoe = n_main - n_gpu
def mem_avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) * 1024
ram = a.ram_gib * GIB if a.ram_gib else mem_avail()
cpu_experts = ncmoe * per_layer + (mtp_experts if a.mtp else 0)
cache = max(1.0 * GIB, min(cpu_experts * 1.02, ram - 3.0 * GIB))
fits = cache >= cpu_experts
frac = 0.9 if fits else 0.7
cmd = ["llama-server", "-m", a.model, "-ngl", "99", "--n-cpu-moe", str(ncmoe), "-c", str(a.ctx), "-ctk", a.kv, "-ctv", a.kv, "-fa", "on", "--load-mode", "mmap", "--no-warmup"]
if a.mtp: cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
usage = a.profile or os.path.expanduser("~/.cache/moe-cache/" + os.path.basename(a.model) + ".usage")
# the plugin sizes itself from the memory limit (MOE_CACHE_SIZE_GIB=auto); it learns from use and warm-starts the next run
env = {"MOE_CACHE_SIZE_GIB": "auto", "MOE_CACHE_GGUF": a.model, "MOE_CACHE_PROFILE": usage, "MOE_CACHE_STATS": "1"}
if a.ram_gib: env["MOE_CACHE_RAM_GIB"] = str(a.ram_gib)
info = dict(arch=arch, layers=n_main, experts=n_exp, used=n_used, expert_MB=per_layer / n_exp / 1e6, gpu_expert_layers=n_gpu, n_cpu_moe=ncmoe,
            predicted_vram_GiB=predicted_vram(n_gpu) / GIB, free_vram_GiB=(free_vram or 0) / GIB, cpu_experts_GiB=cpu_experts / GIB, ram_available_GiB=ram / GIB,
            cache_GiB=cache / GIB, cache_holds_all_cpu_experts=fits, note=note, env=env, command=cmd)
if a.json: print(json.dumps(info, indent=1))
else:
    print(f"{arch}: {n_main} layers x {n_exp} experts (top-{n_used}), {per_layer/n_exp/1e6:.2f} MB/expert, experts {n_main*per_layer/GIB:.1f} GiB, core {core/GIB:.2f} GiB, KV@{a.ctx} {kv_bytes/GIB:.2f} GiB" + (f", n-gram table {table/GIB:.1f} GiB (stays in the file)" if table else ""))
    print(f"GPU: free {free_vram/GIB:.2f} GiB -> {n_gpu} layers of experts on the GPU (predicted use {predicted_vram(n_gpu)/GIB:.2f} GiB)" if free_vram else "GPU: none")
    print(f"RAM: available {ram/GIB:.1f} GiB -> expert cache {cache/GIB:.1f} GiB ({'holds every CPU expert' if fits else 'partial: ' + format(cache/cpu_experts*100, '.0f') + '% of CPU experts'})")
    if note: print("NOTE:", note)
    print("\n" + "\n".join(f"export {k}={v}" for k, v in env.items()) + "\n(plus LD_PRELOAD / GGML_BACKEND_PATH for the plugin: easier with bin/moe-cache-server)\n" + " ".join(cmd))
