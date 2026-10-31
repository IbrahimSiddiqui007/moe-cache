# Patches for the bundled llama.cpp

The Windows kit and the Linux `-ggml0.26` tarball bundle an official llama.cpp build. This folder holds a small patch set for that build. The plugin does not need it: moe-cache keeps working on stock llama.cpp.

**Pinned base:** release `b11966`, commit `ff0d396e3dda0f33e7ce5ad4267fce4f7690cfb1` (ggml 0.28.0).

There are no patches yet. This folder is scaffolding.

## Rules

1. Small numbered patches: `0001-short-name.patch`, `0002-...`, applied in name order. One idea per patch.
2. Each patch has its own runtime switch (an environment variable or flag), **off by default**. A patch only becomes default-on after a paired run (with and without it) shows a win.
3. The plugin must still work on stock llama.cpp. Nothing in `src/` may require a patch.
4. Patches apply to exactly the pinned commit. Moving the pin means re-checking every patch.

## Apply

```bash
tools/apply_llama_patches.sh /path/to/llama.cpp
```

It checks the base commit, runs `git apply --check` on each patch, then applies it. It stops at the first failure and succeeds when there are no patches.

## CI

`.github/workflows/llama_patched.yml` is manual only (`workflow_dispatch`). It builds the patched `llama-server` for Windows (Vulkan) and Linux and uploads them as artifacts. It has not been run yet.

## What would change to ship it

`release.yml` and `windows.yml` currently download the official b11966 release. To ship the patched build they would instead use the artifacts of `llama_patched.yml` (or build it inline), and the bundled `LICENSE-llama.cpp.txt` / `README-moe-cache.txt` would say the build is patched. `release.yml` is not changed here.
