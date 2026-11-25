# moe-cache v0.3.9.1 (release date: pending)

**What this release is.** A small follow-up to v0.3.9: the installer's source build uses the same llama.cpp pin as the kits, a note for Inkling 2 Mini users, and the bundled `llama-fit-params`. The bundled llama.cpp is unchanged (b12651, commit `f5e6915c2`, ggml 1.0.0, backend API 4). No default changes (the new timer and pack hash are diagnostic or opt-in).

**Answers are not token-identical to plain llama.cpp's by default**, as in v0.3.9; nothing in this release changes that.

## Measured

**No speed figure in this release.** Two A B B A runs of 2026-11-25 (Qwen3.6-35B against v0.3.8; the dense 27B spill rule at 128k) were made with the plugin forced on by the test harness, but the launcher runs stock llama.cpp for both models by default, so they do not describe what a user gets. They stay internal records; speed figures through the default path follow. The reference stock rows in `benchmarks/reference.csv` (four more, one run each) are forecasts.

## What changed

| Where | Change | Measured |
|---|---|---|
| Installer | **`install.sh --build` (and `LLAMA_SHA` unset) builds llama.cpp b12651** (commit f5e6915c2, the pin the kits bundle), not the old 7fe450e. The old commit had no `--moe-cache-mib`, so `--gpu-cache` failed for users who built from source. | Dry-run and unit test (the default equals the pin in `llama_patched.yml`); a real `--build` install was not run |
| Kits / launcher | **The patched llama.cpp kits bundle `llama-fit-params`**; the planner passes it to the stock-vs-plugin check and the retunes. No change while the plan rule is off; users who set `MOE_CACHE_PLAN_RULE=1` now measure with the bundled tool. | Built and unit tested |
| Benchmark records | The environment line says "desktop with apps closed" when the harness clean check passed, else "working desktop"; a record names the GPU the run used (`--device`), not the first listed device. | Unit tested |
| Docs | **What the launcher runs for a model that fits in RAM is stated as the code does it:** stock llama.cpp for a dense model, when every expert fits the card and when the model only nearly fits the GPU; the plugin for a MoE model that fits in RAM with part of its experts on the CPU (`--stock` opts out), also with no GPU. The old line that a machine without a GPU stays on stock was wrong and is removed. | none |
| Docs | **Text that promised speed figures "in v0.3.9.1" now says "in a later release"** (README, docs, launcher help, model catalogue). | none |
| Packs | **Opt-in payload hash for packs:** `moe_cache.py pack --hash` records a hash of the pack's payload and `MOE_CACHE_PACK_VERIFY=1` checks it when the pack is loaded. Default off. | Model-free tests only; the cost on a large pack is not measured |
| Plugin / diagnostics | **An always-on timer around the cache's evict step**: the stats JSON gains `ns_evict`, `ns_evict_dec` and `evict_calls_dec`, the exit log gets one more line, and the bench records carry `evict_s` / `evict_dec_s`; the snapshot buffer grows from 2200 to 2400 bytes. Diagnostic only, no speed claim. | About 36 ns per call on one laptop; effect on a real run not measured |
| Docs | **Inkling 2 Mini is a thinking model:** an empty `content` with `finish_reason` `length` means the token limit ran out while it was thinking (the text so far is in `reasoning_content`); raise max tokens (`docs/TROUBLESHOOTING.md`). | none |

## Known limits

As in v0.3.9 (README "Known issues"): most v0.3.9 launcher changes are built and not measured, or measured on one laptop only. No speed figure is claimed for the dense spill rule or the 35B; they follow.

## Update

`moe_cache.py update --apply`, or run `install.sh` / `install.ps1` from this release.
