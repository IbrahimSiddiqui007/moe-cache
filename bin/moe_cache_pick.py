#!/usr/bin/env python3
"""moe-cache model picker (Linux and Windows; Python 3.8+, standard library only): `moe_cache.py pick` and the control page's "Which model?".

Reads this machine (RAM and the server's hard cap, dedicated VRAM, free disk, SSD or HDD, AVX2 / AVX-512, llama-server's build), asks what the
model is for (agents and coding at 64k+ context, plain chat, long documents) and recommends one model, file, context and launcher options.
Every rule below comes from a measurement in docs/RESULTS.md; where nothing was measured on that class of machine the result says so
("untested on this class of machine"). It never downloads anything by itself: it prints the command (size and source), and the launcher asks
before running it. `--apply` (or the page's "Use it") writes the choice into the settings the launcher and the control page keep (ui.json):
`moe_cache.py server` and `on-demand` without a model then use it, and `connect hermes|opencode` reads it from that server.
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_cache as mc  # noqa: E402

WINDOWS = mc.WINDOWS
GIB = 2**30
SETTINGS_FILE = os.path.join(mc.CACHE_DIR, "ui.json")      # the control page's settings (moe_cache_ui.SETTINGS_FILE): "last" = what Start uses
GPU_MARGIN_GIB = 0.5 * 1e9 / GIB      # the launcher's --gpu-margin default (the tuner's 0.5 GB): VRAM it leaves free next to everything it places
USES = {"agents": "agents and coding (Hermes, OpenCode) at 64k-256k context", "chat": "plain chat", "docs": "long documents (64k-128k prompts)"}
BOLT_BUILD = 11966          # Nemotron needs llama.cpp b11966 or newer (tokenizer fix #30960, RESULTS 65)
PC = "32 GB RAM + 16 GB GPU (Windows desktop, RESULTS 59-67)"
LAPTOP = "31 GB RAM limited to 24 GB + 8 GB GPU (Linux laptop, RESULTS 44-45)"

# The candidates. size = bytes of the files we measured (all shards); pattern finds a copy in the model folders; repo + include = where it comes from.
MODELS = {
    "bolt": dict(name="Nemotron 3.5 Bolt 74B-A9B (NVIDIA 2-bit QAT)", size=22_018_326_528, pattern=r"Nemotron-3\.5-Bolt-74B.*24GB\.gguf$",
                 repo="nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF", include="NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf"),
    "flash-next-q2_0": dict(name="Qwen3.8-Flash-Next Q2_0 (GSQ-RCO)", size=66_423_878_624, pattern=r"Flash-Next-GSQ-RCO-Q2_0[-.]",
                            repo="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", include="*Q2_0*"),
    "flash-next-iq3_xxs": dict(name="Qwen3.8-Flash-Next IQ3_XXS (GSQ-RCO)", size=75_839_998_528, pattern=r"Flash-Next-GSQ-RCO-IQ3_XXS[-.]",
                               repo="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", include="*IQ3_XXS*"),
    "qwen3.6-35b": dict(name="Qwen3.6-35B-A3B Q4_K_M", size=21_718_480_960, pattern=r"(?i)qwen3\.6(-35b-a3b)?([-_.].*Q4_K_M)?.*\.gguf$",
                        repo="unsloth/Qwen3.6-35B-A3B-GGUF", include="*Q4_K_M*", ollama="qwen3.6:35b-a3b"),
    "qwen3.5-122b": dict(name="Qwen3.5-122B-A10B UD-IQ2_XXS", size=36_637_668_544, pattern=r"Qwen3\.5-122B-A10B-UD-IQ2_XXS[-.]",
                         repo="unsloth/Qwen3.5-122B-A10B-GGUF", include="*UD-IQ2_XXS*"),
    "glm-5.3": dict(name="GLM-5.3-Flash UD-IQ2_XXS", size=101_845_506_400, pattern=r"GLM-5\.3-Flash-UD-IQ2_XXS[-.]",
                    repo="unsloth/GLM-5.3-Flash-GGUF", include="*UD-IQ2_XXS*"),
}

SIBLINGS = {"flash-next-q2_0": "flash-next-iq3_xxs", "flash-next-iq3_xxs": "flash-next-q2_0"}
SIBLINGS_WHY = {"flash-next-iq3_xxs": "the accuracy pick: 193/202 tasks, about 10 % slower decode than Q2_0",
                "flash-next-q2_0": "the default file: about 10 % faster decode than IQ3_XXS, 191/202 tasks"}


# ---- this machine ----

def cpu_flags():
    """(avx2, avx512f): True / False, None when unknown."""
    if WINDOWS:
        try:
            import ctypes
            f = ctypes.windll.kernel32.IsProcessorFeaturePresent
            return bool(f(40)), bool(f(41))         # PF_AVX2_INSTRUCTIONS_AVAILABLE, PF_AVX512F_INSTRUCTIONS_AVAILABLE
        except Exception:
            return None, None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("flags"):
                fl = line.split(":", 1)[1].split()
                return "avx2" in fl, "avx512f" in fl
    except OSError:
        pass
    return None, None


def dedicated_vram():
    """(name, bytes) of the biggest GPU with its own memory: nvidia-smi, AMD sysfs on Linux, the registry on Windows; (None, None) when none."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            best = max((l.rsplit(",", 1) for l in r.stdout.strip().splitlines()), key=lambda x: float(x[1]))
            return best[0].strip(), float(best[1]) * 2**20
    except Exception:
        pass
    if WINDOWS:
        ded = {k: v for k, v in mc.win_dedicated_vram().items() if v >= GIB}      # under 1 GiB: an integrated GPU (shares RAM)
        if ded:
            name = max(ded, key=ded.get)
            return name, float(ded[name])
        return None, None
    best = (None, None)
    for d in glob.glob("/sys/class/drm/card[0-9]*/device"):
        try:
            tot = int(open(d + "/mem_info_vram_total").read())
        except (OSError, ValueError):
            continue
        if tot >= GIB and (best[1] is None or tot > best[1]):
            vendor = {"0x1002": "AMD", "0x8086": "Intel"}.get(open(d + "/vendor").read().strip(), "GPU")
            best = (vendor + " GPU", float(tot))
    return best


def disk_kind(path):
    """'SSD', 'HDD' or None for the drive holding path (Linux: the block device's rotational flag; Windows: Get-PhysicalDisk's MediaType)."""
    try:
        if WINDOWS:
            letter = os.path.splitdrive(os.path.abspath(path))[0].rstrip(":")
            if not re.fullmatch(r"[A-Za-z]", letter or ""):
                return None
            ps = (f"$n=(Get-Partition -DriveLetter {letter} -ErrorAction Stop).DiskNumber; "
                  "(Get-PhysicalDisk | Where-Object { $_.DeviceId -eq \"$n\" }).MediaType")
            r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=20)
            t = r.stdout.strip().upper()
            return "SSD" if "SSD" in t else "HDD" if "HDD" in t else None
        st = os.stat(path)
        p = os.path.realpath(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}")
        if not os.path.exists(p):           # btrfs, overlay: st_dev is not a block device; the mount's source is
            real, best = os.path.realpath(path), ("", "")
            for line in open("/proc/mounts"):
                src, mnt = line.split()[:2]
                mnt = mnt.replace("\\040", " ")
                if src.startswith("/dev/") and (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best[1]):
                    best = (src, mnt)
            p = os.path.realpath(os.path.join("/sys/class/block", os.path.basename(os.path.realpath(best[0])))) if best[0] else ""
        for _ in range(4):                       # a partition's queue is on its parent disk; dm / md devices: their first slave
            q = os.path.join(p, "queue", "rotational")
            if os.path.isfile(q):
                return "HDD" if open(q).read().strip() == "1" else "SSD"
            slaves = glob.glob(os.path.join(p, "slaves", "*"))
            p = os.path.realpath(slaves[0]) if slaves else os.path.dirname(p)
    except Exception:
        pass
    return None


def llama_build(ls):
    """llama-server's build number (`--version`: 'version: 11966 (ff0d396e3)'), None when unknown."""
    try:
        r = subprocess.run([ls, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
        text = r.stdout + r.stderr
        m = re.search(r"\(build (\d+)", text) or re.search(r"version:\s*(\d+)\s*\(", text)     # '0.7.0-dev (build 11669, commit ...)' or the old 'version: 4567 (abc)'
        n = int(m.group(1)) if m else 0
        return n if n >= 100 else None        # distro builds report 'build 1' / 0: no usable number
    except Exception:
        return None


def model_folders(extra=None):
    out = [os.path.join(os.path.expanduser("~"), "models")] + (["C:\\models", "D:\\models"] if WINDOWS else [])
    try:
        out += json.load(open(SETTINGS_FILE, encoding="utf-8")).get("folders", [])
    except Exception:
        pass
    return [f for f in dict.fromkeys((extra or []) + out) if f]


def ollama_roots():
    """Ollama's model stores that exist here: $OLLAMA_MODELS, ~/.ollama/models (Linux; Windows: %USERPROFILE%\\.ollama\\models) and the Linux service's folders."""
    c = [os.environ.get("OLLAMA_MODELS"), os.path.join(os.path.expanduser("~"), ".ollama", "models")] + OLLAMA_SYSTEM_ROOTS
    return [p for p in dict.fromkeys(x for x in c if x) if os.path.isdir(os.path.join(p, "manifests"))]


OLLAMA_SYSTEM_ROOTS = ["/usr/share/ollama/.ollama/models", "/var/lib/ollama/.ollama/models"]      # the Linux service's stores (tests empty this)


def ollama_manifests(root):
    """[(name, blob path)] of the models in one Ollama store: manifests/<registry>/<namespace>/<model>/<tag> names the model layer's blob
    (blobs/sha256-<hex>, a plain GGUF file without an extension that llama-server reads). name = 'library/qwen3.6:35b-a3b' style, registry left out."""
    out = []
    for man in glob.glob(os.path.join(root, "manifests", "*", "*", "*", "*")) + glob.glob(os.path.join(root, "manifests", "*", "*", "*", "*", "*")):
        try:
            layers = json.load(open(man, encoding="utf-8")).get("layers", [])
        except Exception:
            continue
        for l in layers if isinstance(layers, list) else []:
            if isinstance(l, dict) and l.get("mediaType") == "application/vnd.ollama.image.model":
                dg = str(l.get("digest", ""))
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", dg):
                    continue
                blob = os.path.join(root, "blobs", dg.replace(":", "-"))
                if os.path.isfile(blob):
                    parts = os.path.relpath(man, os.path.join(root, "manifests")).replace(os.sep, "/").split("/")[1:]
                    out.append(("/".join(parts[:-1]) + ":" + parts[-1], blob))
    return out


def ollama_name(blob):
    """The Ollama tag of a blob path ('qwen3.6:35b-a3b'), None when it is not in a store here."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(blob)))
    for name, b in ollama_manifests(root):
        if os.path.abspath(b) == os.path.abspath(blob):
            return name.split("/")[-1] if name.startswith("library/") else name
    return None


# which Ollama tags are which candidate (matched on the whole tag, lower case)
OLLAMA_TAGS = {"bolt": r"nemotron.*bolt", "flash-next-q2_0": r"flash-next.*q2_0", "flash-next-iq3_xxs": r"flash-next.*iq3_xxs",
               "qwen3.6-35b": r"qwen3\.6.*35b", "qwen3.5-122b": r"qwen3\.5.*122b", "glm-5.3": r"glm-?5\.3"}


def find_ollama(found=None, roots=None):
    """Candidates that Ollama has already downloaded: {model id: blob path} (added to `found`; files in the model folders win)."""
    found = {} if found is None else found
    for root in ollama_roots() if roots is None else roots:
        for name, blob in ollama_manifests(root):
            for k, rx in OLLAMA_TAGS.items():
                if k not in found and re.search(rx, name.lower()):
                    found[k] = blob
    return found


def find_local(folders, ollama_stores=None):
    """{model id: path of the first file} for the candidates found in the folders (depth 2) and in Ollama's stores."""
    found = {}
    for f in folders:
        for p in glob.glob(os.path.join(f, "*.gguf")) + glob.glob(os.path.join(f, "*", "*.gguf")):
            b = os.path.basename(p)
            if re.search(r"-\d{5}-of-\d{5}\.gguf$", b) and not re.search(r"-00001-of-\d{5}\.gguf$", b):
                continue
            for k, m in MODELS.items():
                if k not in found and re.search(m["pattern"], b):
                    found[k] = os.path.abspath(p)
    return find_ollama(found, ollama_stores)


def machine_specs(folders=None, ls=None):
    """Everything the rules read, in plain units (bytes, booleans, None when unknown)."""
    folders = model_folders(folders)
    disk_at = next((f for f in folders if os.path.isdir(f)), os.path.expanduser("~"))
    name, vram = dedicated_vram()
    avx2, avx512 = cpu_flags()
    try:
        free = float(shutil.disk_usage(disk_at).free)
    except OSError:
        free = None
    return {"os": "Windows" if WINDOWS else sys.platform, "ram_total": mc.mem_total(), "ram_cap": mc.mem_cap(), "gpu": name, "vram": vram,
            "disk_path": disk_at, "disk_free": free, "disk_kind": disk_kind(disk_at), "avx2": avx2, "avx512": avx512,
            "llama_build": llama_build(ls or mc.default_llama_server()), "local": find_local(folders)}


# ---- the rules (pure: unit-tested with fake specs, tests/picker_test.py) ----

def size_of(k, local):
    """Bytes of model k: the file's own size when it is here as one file (Ollama's Q4_K_M is not the unsloth one we measured), else the candidate's."""
    p = (local or {}).get(k)
    try:
        if p and not re.search(r"-\d{5}-of-\d{5}\.gguf$", p):
            return os.path.getsize(p)
    except OSError:
        pass
    return MODELS[k]["size"]


def recommend(spec, use, model=None):
    """One recommendation for `use` (agents | chat | docs) on the machine `spec` (machine_specs()). Returns
    {id, name, file, ctx, args, measured, untested, why[], warnings[], alternatives[{id, name, why}], local, download, memory}.
    model: a candidate id the user chose instead: it is used as asked and only warned about (too big for the memory or the disk, old llama.cpp,
    hard disk), never refused."""
    if use not in USES:
        raise ValueError(f"use: one of {', '.join(USES)}")
    cap = (spec.get("ram_cap") or 0) / 1e9                       # what the whole server may use (RAM minus max(4 GiB, 12.5 %)), GB
    vram = (spec.get("vram") or 0) / GIB
    local = spec.get("local") or {}
    hdd = spec.get("disk_kind") == "HDD"
    build = spec.get("llama_build")
    free = spec.get("disk_free")
    warn = []

    def fits(k):           # the launcher's own rule (moe_cache.decide): stock llama-server when model + 1.5 GB fits, else the plugin
        return size_of(k, local) + mc.STOCK_HEADROOM <= cap * 1e9

    def placeable(k):      # already here, or enough free disk for it (2 GB to spare); unknown free disk does not block
        return k in local or free is None or free >= MODELS[k]["size"] + 2e9

    def drive_ok(k):       # a model bigger than memory reads experts from the drive all the time: not from an HDD
        return fits(k) or not hdd

    def bolt_ok():
        return build is None or build >= BOLT_BUILD

    R = []     # candidate rules in order: (id, condition, ctx, measured text or None = untested here, why)
    if use == "agents":
        R += [("bolt", fits("bolt") and vram >= 12 and bolt_ok(), 131072,
               f"{PC}: OpenCode session 47.8 s (the fastest measured; Flash-Next IQ2_XS 73.4 s), prompts 721 tok/s at 114k tokens, decode 24.9 tok/s "
               "there (27.9 short), 190/202 tasks",
               "fits in your RAM, so llama.cpp runs it stock (moe-cache forced on it was 18 % slower); the fastest agent model we measured"),
              ("bolt", fits("bolt") and vram >= 6 and bolt_ok(), 65536, None,
               "fits in your RAM (stock llama.cpp); measured only with a 16 GB GPU: with less VRAM long prompts are slower"),
              ("qwen3.6-35b", True, 65536,
               f"{LAPTOP}: 42.9 tok/s stock when it fits; at 12 GB RAM + 8 GB GPU 22.6 tok/s with moe-cache; {PC}: 739-848 tok/s prompts and "
               "59 tok/s decode at 64k" if vram >= 6 else
               "no GPU, 12 GB RAM: 9.9 tok/s (laptop) and 10.6 tok/s (Ryzen mini PC) with moe-cache, short prompts",
               "the smallest candidate (3B active): the one that still runs agents at 64k on this machine")]
    elif use == "chat":
        R += [("glm-5.3", fits("glm-5.3"), 32768, None, "fits in your RAM: the biggest candidate, stock llama.cpp"),
              ("flash-next-iq3_xxs", cap >= 50 and vram >= 12 and drive_ok("flash-next-iq3_xxs"), 32768, None,
               f"the accuracy pick (193/202 tasks, the most); measured {PC}: 19.9 tok/s; with your RAM more of it stays in memory"),
              ("flash-next-q2_0", cap >= 24 and vram >= 12 and drive_ok("flash-next-q2_0"), 32768,
               f"{PC}: 22-25 tok/s decode, 281-287 tok/s prompts, 191/202 tasks",
               "the documented default: bigger than your RAM, moe-cache keeps the hot experts in RAM and reads the rest from the SSD"),
              ("flash-next-q2_0", cap >= 24 and vram >= 6 and drive_ok("flash-next-q2_0"), 32768, None,
               f"the documented default; measured on {LAPTOP} only with the IQ2_XS file of the same size: 12.5 tok/s"),
              ("qwen3.5-122b", cap >= 24 and vram >= 6 and drive_ok("qwen3.5-122b"), 32768,
               f"{PC}: 19.7 tok/s at 4k, 17.3 at 64k; {LAPTOP}: 7.7 tok/s" if vram >= 12 else f"{LAPTOP}: 7.7 tok/s",
               "36.6 GB on disk instead of Flash-Next's 66 GB"),
              ("bolt", fits("bolt") and vram >= 6 and bolt_ok(), 32768, f"{PC}: 27.9 tok/s" if vram >= 12 else None,
               "fits in your RAM (stock llama.cpp)"),
              ("qwen3.6-35b", True, 32768,
               f"{LAPTOP}: 42.9 tok/s stock when it fits, 22.6 tok/s with moe-cache at 12 GB RAM" if vram >= 6 else
               "no GPU, 12 GB RAM: 9.9-10.6 tok/s with moe-cache",
               "the smallest candidate (21.7 GB, 3B active)"),
              ("flash-next-q2_0", cap >= 11 and vram >= 6 and drive_ok("flash-next-q2_0"), 16384,
               "12 GB RAM + 8 GB GPU (laptop, RESULTS 53): 9.5 tok/s with --fast, 13.7 with --fast=2 (both change some tokens)",
               "a much bigger model through moe-cache; on this much RAM only with --fast (approximate)")]
    else:
        R += [("bolt", fits("bolt") and vram >= 12 and bolt_ok(), 131072,
               f"{PC}: prompts 595 tok/s at 57k and 721 at 114k tokens, decode 24.9 at 114k (Flash-Next Q2_0: 170 tok/s prompts at 128k)",
               "reads long prompts about 4x faster than Flash-Next at 128k and fits in your RAM (stock llama.cpp)"),
              ("bolt", fits("bolt") and vram >= 6 and bolt_ok(), 65536, None, "fits in your RAM (stock llama.cpp); measured only with a 16 GB GPU"),
              ("qwen3.6-35b", True, 131072 if fits("qwen3.6-35b") and vram >= 12 else 65536,
               f"{PC}: 936-1014 tok/s prompts and 45.8 tok/s decode at 128k" if fits("qwen3.6-35b") and vram >= 12 else None,
               "the smallest candidate; long prompts on a small GPU or none are slow")]

    def placed(k):
        if k in local:
            return True
        if not placeable(k):
            warn.append(f"{MODELS[k]['name']} needs {MODELS[k]['size'] / 1e9:.1f} GB of free disk ({(free or 0) / 1e9:.1f} GB free at "
                        f"{spec.get('disk_path')}): not suggested")
            return False
        return True

    if model is not None and model not in MODELS:
        raise ValueError("model: one of " + ", ".join(MODELS))
    chosen = None
    if model:      # the user's own choice: the rule's context and numbers when it has a rule, else a plain default; warnings only
        rule = next((r for r in R if r[0] == model), None)
        placed(model)
        chosen = (model, True, rule[2] if rule else {"chat": 32768}.get(use, 65536), rule[3] if rule else None,
                  "chosen by you" + ("" if rule else f" (not one of the picks for {USES[use]} on this machine)"))
    chosen = chosen or next((r for r in R if r[1] and placed(r[0])), None)
    if not chosen:      # no candidate that suits the machine is here or fits the disk: still the one that suits the machine (warn, never block), said plainly below
        chosen = next(r for r in R if r[1])
    k, _, ctx, measured, why = chosen
    sib = SIBLINGS.get(k)
    if sib and not model and k not in local and sib in local and drive_ok(sib):     # the other Flash-Next file is already here: no 66-76 GB download
        why = f"{MODELS[sib]['name']} is already on this machine (same model, {SIBLINGS_WHY[sib]}); otherwise: {why}"
        k = sib
        chosen = (k, True, ctx, measured, why)
    m = MODELS[k]
    need_disk = m["size"] + 2e9
    fits_disk = k in local or free is None or free >= need_disk
    missing = 0.0 if fits_disk else need_disk - free
    if not fits_disk:
        why = (f"This is the model that suits your machine, but it is not downloaded and the disk is too small: {m['size'] / 1e9:.1f} GB (+2 GB to spare) needed, "
               f"{free / 1e9:.1f} GB free at {spec.get('disk_path')}, {missing / 1e9:.1f} GB missing. Nothing is downloaded. Free that space or add a model folder on a "
               f"bigger drive; the numbers below are for this model once it is there. " + why)
        warn.append(f"{m['name']}: {missing / 1e9:.1f} GB of disk missing at {spec.get('disk_path')}")
    if k == "bolt" and build is None:
        warn.append(f"Nemotron needs llama.cpp b{BOLT_BUILD} or newer; your llama-server's build could not be read (`llama-server --version`)")
    if build is not None and build < BOLT_BUILD and use in ("agents", "docs") and fits("bolt"):
        warn.append(f"your llama-server is b{build}: Nemotron 3.5 Bolt (the faster pick here) needs b{BOLT_BUILD} or newer")
    if vram and spec.get("gpu_margin") is not None and float(spec["gpu_margin"]) > GPU_MARGIN_GIB + 0.01:
        warn.append(f"counting {float(spec['gpu_margin']):.1f} GiB of VRAM as left free (--gpu-margin), the launcher's default is {GPU_MARGIN_GIB:.1f}")
    elif vram and use in ("agents", "docs") and ctx >= 131072:
        warn.append(f"VRAM is counted minus {GPU_MARGIN_GIB:.1f} GiB, as the launcher does; if this GPU also draws your screen and answers slow down sharply, "
                    "start with --gpu-margin 2")
    if hdd:
        warn.append("the models are on a hard disk: only models that fit in RAM are suggested (moe-cache reads experts from the drive; "
                    "moe_cache.py check times it)")
    if spec.get("avx2") is False:
        warn.append("this CPU has no AVX2: llama.cpp's CPU kernels are much slower; untested")
    if not spec.get("vram"):
        warn.append("no GPU with its own memory found: everything runs on the CPU (set --vram if the GPU was not detected)")
    if use == "agents" and not spec.get("vram"):
        warn.append("agents send 5-20k-token prompts; without a GPU the first answer can take minutes (untested at 64k on such a machine)")
    kv = kv_estimate(k, ctx, local.get(k))
    need = m["size"] + kv["bytes"]
    margin = spec.get("gpu_margin")
    margin = GPU_MARGIN_GIB if margin is None else float(margin)
    usable_vram = max(0.0, vram - margin) * GIB if vram else 0.0          # the same margin the launcher's tuner keeps (--gpu-margin)
    memory = {"model": m["size"], "kv": kv["bytes"], "kv_how": kv["how"], "total": need, "available": cap * 1e9 + usable_vram, "gpu_margin_gib": margin}
    if need > memory["available"] and fits(k):
        warn.append(f"model {m['size'] / 1e9:.1f} GB + KV cache {kv['bytes'] / 1e9:.1f} GB at context {ctx} ({kv['how']}) is more than RAM cap + VRAM "
                    f"({memory['available'] / 1e9:.1f} GB): a smaller context may be needed")
    engine = "stock llama.cpp (fits in RAM)" if fits(k) else "moe-cache (bigger than the memory the server may use)"
    args = (["--agent"] if use == "agents" else []) + ["--ctx", str(ctx)]
    alts = []
    for r in R:
        if r is not chosen and r[1] and r[0] != k and r[0] not in (a["id"] for a in alts) and placeable(r[0]):
            alts.append({"id": r[0], "name": MODELS[r[0]]["name"], "why": r[4] + ("" if r[3] else " (untested on this class of machine)")})
    for lk in local:      # a model that is already here is worth knowing about even when it is not the best fit
        if lk != k and lk not in (a["id"] for a in alts) and len(alts) < 2:
            alts.append({"id": lk, "name": MODELS[lk]["name"], "why": "already on this machine (not the best fit for it)"})
    if use == "agents" and k != "flash-next-q2_0" and cap >= 24 and vram >= 12 and drive_ok("flash-next-q2_0") and placeable("flash-next-q2_0"):
        alts.append({"id": "flash-next-q2_0", "name": MODELS["flash-next-q2_0"]["name"],
                     "why": f"the documented default line: {PC} 20.7 tok/s at 64k, 15.3 at 128k, prompts 170-266 tok/s, moe-cache"})
    out = {"id": k, "name": m["name"], "use": use, "size": size_of(k, local), "ctx": ctx, "args": args, "engine": engine,
           "measured": measured or "untested on this class of machine", "untested": not measured, "why": [why], "warnings": warn,
           "alternatives": alts[:2], "memory": memory, "local": local.get(k), "fits_disk": fits_disk, "disk_missing": missing,
           "download": None if (k in local or not fits_disk) else download_cmd(k)}
    return out


def kv_estimate(k, ctx, path=None):
    """Bytes of KV cache + recurrent state at `ctx` (f16): from the model's header when the file is here (planner/gguf_plan.kv_bytes, the planner's
    own numbers), else Flash-Next from the PC's measured 256k buffers (KV 6144 + indexer 1536 + state 450 MiB), else the planner's flat 40 KB
    per token (an over-estimate for hybrid models)."""
    if path:
        try:
            sys.path.insert(0, os.path.join(mc.HERE, "planner"))
            import gguf_plan
            v = gguf_plan.kv_bytes(path, ctx)
            if v:
                return {"bytes": float(v), "how": "from the model's header"}
        except Exception:
            pass
    if k.startswith("flash-next"):
        return {"bytes": (6144 + 1536) * 2**20 * ctx / 262144 + 450 * 2**20, "how": "measured on the PC at 256k, scaled"}
    return {"bytes": 40_000.0 * ctx, "how": "flat 40 KB per token, an over-estimate"}


def download_cmd(k):
    """What to run to get model k (printed or asked for; never run without the user's yes)."""
    m = MODELS[k]
    d = os.path.join(os.path.expanduser("~"), "models")
    out = {"size": m["size"], "source": f"https://huggingface.co/{m['repo']}", "folder": d,
           "command": f'hf download {m["repo"]} --include "{m["include"]}" --local-dir "{d}"'}
    if m.get("ollama"):
        out["ollama"] = f"ollama pull {m['ollama']}"
    return out


# ---- the settings the launcher and the control page keep ----

def load_settings():
    try:
        return json.load(open(SETTINGS_FILE, encoding="utf-8"))
    except Exception:
        return {}


class _settings_lock:
    """An exclusive lock on ui.json.lock (fcntl on Linux, msvcrt on Windows): the control page and `pick --apply` take turns."""
    def __enter__(self):
        os.makedirs(mc.CACHE_DIR, exist_ok=True)
        self.f = open(SETTINGS_FILE + ".lock", "a+b")
        if WINDOWS:
            import msvcrt
            self.f.seek(0)
            for _ in range(200):                       # LK_LOCK itself retries for 10 s only
                try:
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl
            fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        try:
            if WINDOWS:
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        self.f.close()


def update_settings(change):
    """Read ui.json, let change(dict) edit it and write it back, under the lock and atomically (a temporary file, then replace), private to this
    user (0600). Each writer changes only its own keys: `pick --apply` and the open control page no longer lose each other's settings, and a
    reader never sees a half-written file (audit 2026-10-29: the page wrote in place; a pick reading it then wrote back only its own keys).
    Returns the new settings."""
    with _settings_lock():
        s = load_settings()
        change(s)
        tmp = f"{SETTINGS_FILE}.{os.getpid()}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1)
        os.replace(tmp, SETTINGS_FILE)
    return s


def apply(rec, settings=None):
    """Put the recommendation into the settings: "pick" (what was chosen and why) and, when the file is here, "last" (what Start, `server` and
    `on-demand` without a model use). settings=None: written to ui.json (update_settings). A dict: changed in place only (the control page saves
    the two keys itself). Returns the settings."""
    if settings is None:
        return update_settings(lambda s: apply(rec, s))
    s = settings
    s["pick"] = {"id": rec["id"], "name": rec["name"], "use": rec["use"], "ctx": rec["ctx"], "args": rec["args"], "model": rec["local"],
                 "time": time.strftime("%Y-%m-%d %H:%M")}
    if rec["local"]:
        last = dict(s.get("last") or {})
        last.update({"model": rec["local"], "ctx": rec["ctx"], "agent": rec["use"] == "agents", "mode": "exact", "engine": "auto"})
        s["last"] = last
    return s


def picked():
    """(model path, ctx, agent) of the applied pick, or None (no pick, or its file is gone)."""
    p = load_settings().get("pick") or {}
    path = p.get("model")
    if not path or not os.path.isfile(path):
        return None
    return path, int(p.get("ctx") or 65536), p.get("use") == "agents"


def pick_missing():
    """A line for `server` / `on-demand` without a model when a model was picked but its file is not here (not downloaded yet, or moved),
    else "" (audit 2026-10-29 P1: they said "none picked" right after a pick)."""
    p = load_settings().get("pick") or {}
    if not p.get("id") or picked():
        return ""
    return (f"you picked {p.get('name') or p['id']} ({p.get('use')}), but its file is not here yet: `moe_cache.py pick --use {p.get('use')}` "
            "offers the download (or put the .gguf in a model folder), then `moe_cache.py pick --use " + str(p.get("use")) + " --apply` again")


# ---- output ----

def gb(b):
    return "?" if b is None else f"{b / 1e9:.1f} GB"


def spec_lines(spec):
    return [f"RAM {gb(spec['ram_total'])}, the server may use at most {gb(spec['ram_cap'])} (RAM minus max(4 GiB, 12.5 %))",
            f"GPU {spec['gpu'] or 'none with its own memory'}" + (f", {spec['vram'] / GIB:.1f} GiB VRAM" if spec.get("vram") else ""),
            f"disk {gb(spec['disk_free'])} free at {spec['disk_path']} ({spec['disk_kind'] or 'SSD/HDD unknown'})",
            "CPU " + ", ".join(f"{n} {'yes' if v else 'no' if v is False else '?'}" for n, v in (("AVX2", spec["avx2"]), ("AVX-512", spec["avx512"]))),
            f"llama-server build {spec['llama_build'] or '? (not found or no --version)'}",
            "models here: " + (", ".join(MODELS[k]["name"] for k in spec["local"]) or "none of the candidates")]


def rec_lines(r):
    out = [f"{USES[r['use']]}: {r['name']}  ({r['size'] / 1e9:.1f} GB, {r['engine']})",
           f"  context {r['ctx']}, options: {' '.join(r['args'])}",
           f"  memory: model {r['memory']['model'] / 1e9:.1f} GB + KV cache {r['memory']['kv'] / 1e9:.1f} GB ({r['memory']['kv_how']}) = {r['memory']['total'] / 1e9:.1f} GB; "
           f"this machine: RAM cap + VRAM (minus {r['memory']['gpu_margin_gib']:.1f} GiB margin) = {r['memory']['available'] / 1e9:.1f} GB",
           f"  measured: {r['measured']}"] + [f"  why: {w}" for w in r["why"]]
    out += [f"  or: {a['name']}: {a['why']}" for a in r["alternatives"]]
    out += [f"  note: {w}" for w in r["warnings"]]
    if r["local"]:
        out.append(f"  file: {r['local']}")
    elif not r["fits_disk"]:
        out.append(f"  not on this machine and {r['disk_missing'] / 1e9:.1f} GB of disk missing: no download is offered")
    else:
        d = r["download"]
        out.append(f"  not on this machine: {d['size'] / 1e9:.1f} GB from {d['source']}")
        out.append(f"  download (you run it, or answer yes below): {d['command']}" + (f"   (Ollama: {d['ollama']})" if d.get("ollama") else ""))
    return out


def ask(q, default=""):
    try:
        return input(q).strip().lower() or default
    except EOFError:
        return default


def pick(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py pick", description="recommend a model for this machine and what it is for; --apply uses it")
    ap.add_argument("--use", choices=list(USES), help="agents (coding agents, 64k-256k context), chat, docs (long documents); asked when left out")
    ap.add_argument("--model", choices=list(MODELS), help="use this candidate instead of the recommendation (only warned about, never refused)")
    ap.add_argument("--gpu-margin", type=float, help="GiB of VRAM counted as left free (the launcher's --gpu-margin; default 0.5)")
    ap.add_argument("--apply", action="store_true", help="write the choice into the launcher's settings (server / on-demand without a model use it)")
    ap.add_argument("--models-dir", action="append", default=[], help="another folder with .gguf files (repeatable)")
    ap.add_argument("--llama-server", default=mc.default_llama_server())     # as server / check: LLAMA_SERVER, PATH, C:\\llama or install.sh's folder
    ap.add_argument("--json", action="store_true", help="print the machine and the recommendations as JSON")
    a = ap.parse_args(argv)
    spec = machine_specs(a.models_dir, a.llama_server)
    if a.gpu_margin is not None:
        spec["gpu_margin"] = a.gpu_margin
    tty = sys.stdin.isatty() and not a.json
    use = a.use
    if not use and tty:
        mc.say("\n".join(["moe-cache: this machine"] + ["  " + l for l in spec_lines(spec)]))
        mc.say("What is it for?  1) agents / coding (Hermes, OpenCode)  2) plain chat  3) long documents")
        use = {"1": "agents", "2": "chat", "3": "docs"}.get(ask("choose 1-3 [1]: ", "1"), "agents")
    uses = [use] if use else list(USES)
    recs = [recommend(spec, u, a.model) for u in uses]
    if a.json:
        print(json.dumps({"machine": spec, "recommendations": recs}, indent=1))
    else:
        if not tty:
            mc.say("\n".join(["moe-cache: this machine"] + ["  " + l for l in spec_lines(spec)]))
        for r in recs:
            mc.say("\n".join(rec_lines(r)))
    if len(recs) != 1:
        if a.apply:
            raise SystemExit("--apply needs --use agents|chat|docs")
        if not a.json:
            mc.say("pick one and use it: moe_cache.py pick --use agents|chat|docs --apply")
        return 0
    r = recs[0]
    if not r["local"] and tty and r["download"] and shutil.which("hf"):
        d = r["download"]
        if ask(f"download {d['size'] / 1e9:.1f} GB from {d['source']} into {d['folder']} now? [y/N] ", "n") == "y":
            os.makedirs(d["folder"], exist_ok=True)
            rc = subprocess.call(["hf", "download", MODELS[r["id"]]["repo"], "--include", MODELS[r["id"]]["include"], "--local-dir", d["folder"]])
            if rc:
                raise SystemExit(f"download failed (exit {rc}); nothing was changed")
            r = recommend(dict(spec, local=find_local(model_folders(a.models_dir))), r["use"], a.model)
    elif not r["local"] and tty and r["download"]:
        mc.say("  (the hf command comes with: pip install -U huggingface_hub)")
    do = a.apply or (tty and not a.json and ask("use it (the launcher and the control page start this model by default)? [Y/n] ", "y") == "y")
    if do:
        apply(r)
        if r["local"]:
            mc.say(f"moe-cache: saved in {SETTINGS_FILE}. Start it: moe_cache.py server" + (" --agent" if r["use"] == "agents" else "")
                   + (f" (or: moe_cache.py on-demand --install-autostart), then: moe_cache.py connect opencode --write" if r["use"] == "agents" else ""))
        else:
            mc.say(f"moe-cache: choice saved in {SETTINGS_FILE}; after the download run `moe_cache.py pick --use {r['use']} --apply` again to point the launcher at the file")
    return 0


if __name__ == "__main__":
    sys.exit(pick(sys.argv[1:]))
