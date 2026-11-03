"""Give GPU layers back after the prompt (ledger DECODE-630): the launcher side.

Idea: size the card for the big prompt chunk (-ub 2048), then, once the prompt is read and the server only decodes, shrink the chunk to
-ub 512 and move the experts of a few more layers onto the GPU in the memory the big compute buffer held. The work is done inside llama.cpp
by patches/llama.cpp/0070-decode-reloc.patch (env LLAMA_DECODE_RELOC, off unless set). This file only decides, in plain numbers, whether to
switch it on and what to tell it. Switch: MOE_CACHE_DECODE_RELOC=1 (default off). Design: docs/DECODE_RELOC_DESIGN.md.

Pure functions, no GPU and no GGUF needed for `size_extra` and `launcher_env` with `lay` and the buffer sizes passed in (tests/decode_reloc_test.py).
"""
import os

SWITCH = "MOE_CACHE_DECODE_RELOC"
DEFAULT_DECODE_UB = 512
HARD_MAX_LAYERS = 8
KEEP_MIB = 768            # VRAM that stays free after the move (the plugin's block buffer is held, other programs, growth)
_MiB = 1048576


def enabled(environ=None):
    v = (environ if environ is not None else os.environ).get(SWITCH, "")
    return v.strip().lower() in ("1", "on", "true", "yes")


def block_mib(layer_bytes, ub):
    """The long-prompt block room the planner keeps for a micro-batch (gguf_plan.tune: block_for, table rule)."""
    return ((1.6 if ub > 1024 else 1.35) * layer_bytes + (160 + 256) * _MiB) / _MiB if layer_bytes else 0.0


def size_extra(layer_bytes, cpu_layers, compute_prompt_mib, compute_decode_mib, ub_prompt, ub_decode):
    """How much room the switch to the small chunk frees and how many whole layers that is.
    Returns dict(freed_mib, layer_mib, expected_extra, max_layers). expected_extra rounds down and counts no leftover (conservative);
    max_layers (what the patch may try) is one more, because the patch looks at the real free memory and stops when a layer does not fit."""
    if not layer_bytes or cpu_layers <= 0:
        return {"freed_mib": 0.0, "layer_mib": 0.0, "expected_extra": 0, "max_layers": 0}
    freed = max(0.0, compute_prompt_mib - compute_decode_mib) + max(0.0, block_mib(layer_bytes, ub_prompt) - block_mib(layer_bytes, ub_decode))
    layer_mib = layer_bytes / _MiB
    extra = min(cpu_layers, int(freed // layer_mib))
    return {"freed_mib": freed, "layer_mib": layer_mib, "expected_extra": extra,
            "max_layers": min(cpu_layers, HARD_MAX_LAYERS, extra + 1) if extra > 0 else 0}


_NO_GO = ("--device", "-dev", "--tensor-split", "-ts", "--override-tensor", "-ot", "--spec-type", "--draft-mtp", "--model-draft", "-md")


def launcher_env(environ, *, lay, ctx, ub, ncmoe, extra, gcache, compute_mib):
    """-> (env pairs for llama-server, one line to show or None).
    `lay` is gguf_plan.model_layout(); `compute_mib(ctx, ub)` gives the planner's GPU compute buffer in MiB.
    Nothing is added unless MOE_CACHE_DECODE_RELOC is set; a reason is returned when it is set and cannot be used."""
    if not enabled(environ):
        return [], None
    no = lambda why: ([], f"moe-cache: {SWITCH} ignored: {why}")
    if gcache != "off":
        return no("a GPU expert cache is in use (static layers only)")
    if not str(ncmoe).isdigit() or int(ncmoe) <= 0 or int(ncmoe) >= (lay.get("n_layers") or 0):
        return no("no mix of CPU and GPU expert layers to rebalance (--n-cpu-moe " + str(ncmoe) + ")")
    bad = [x for x in extra if x in _NO_GO or "mtp" in str(x).lower() or str(x).startswith(("--device=", "--override-tensor=", "--tensor-split="))]
    if bad:
        return no(f"not supported with {bad[0]} (one GPU, no MTP / draft model, no -ot)")
    try:
        ubd = int(environ.get(SWITCH + "_UB") or DEFAULT_DECODE_UB)
        ubp = int(ub)
    except (TypeError, ValueError):
        return no("the micro-batch is not a number")
    if ubd >= ubp:
        return no(f"the prompt chunk {ubp} is not larger than the decode chunk {ubd}")
    s = size_extra(lay.get("expert_bytes_per_layer") or 0, int(ncmoe), compute_mib(ctx, ubp) or 0, compute_mib(ctx, ubd) or 0, ubp, ubd)
    if s["max_layers"] <= 0:
        return no(f"going from -ub {ubp} to {ubd} frees {s['freed_mib']:.0f} MiB, less than one layer ({s['layer_mib']:.0f} MiB)")
    pairs = [("LLAMA_DECODE_RELOC", "1"), ("LLAMA_DECODE_RELOC_UB", str(ubd)), ("LLAMA_DECODE_RELOC_MAX_LAYERS", str(s["max_layers"])),
             ("LLAMA_DECODE_RELOC_KEEP_MIB", str(int(environ.get(SWITCH + "_KEEP_MIB") or KEEP_MIB)))]
    return pairs, (f"moe-cache: {SWITCH}=1: after the prompt -ub {ubp} becomes {ubd} and up to {s['max_layers']} more layers of experts move to the GPU "
                   f"(planner estimate {s['expected_extra']} from {s['freed_mib']:.0f} MiB freed; needs the patched llama.cpp, patch 0070; PROTOTYPE)")
