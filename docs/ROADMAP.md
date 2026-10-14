# Versions and roadmap

Version numbers describe how safe it is to hand the project to someone else.

| Version | Meaning | What must be true |
|---|---|---|
| **v0.1** (done) | Works on the author's machine; others can build and try it on Linux | Plugin runs unmodified llama.cpp; token-identical to stock on every model tested; launcher, planner, identity test, quick start; honest compatibility page |
| **v0.2** (friend-ready) | A friend with a similar Linux box can install it and it works first time | Prebuilt plugin for the common llama.cpp releases; setup check that says exactly what is wrong; tested on at least one AMD or Intel GPU through Vulkan and on CPU-only; first GitHub release |
| **v0.2b** (done) | CPU-only parity | Repacks each expert on load exactly when stock would: CPU-only machines are bit-identical and as fast as stock |
| **v0.2c** (done) | GPU-assisted prompt processing | Run the big expert multiplications of long prompts on the GPU from inside the plugin (restores stock's prompt speed when the model fits; building block of the GPU tier) |
| **v0.3** | Windows (code written and cross-compiled, platform layer tested under Wine; not tested on real Windows, no Windows launcher yet) | Windows port of the memory and file code; tested with official Windows llama.cpp builds; `GGML_BACKEND_PATH` loading |
| **v0.4** | Safety and polish | Automatic fallback to stock behavior on a fault; long-context and multi-user tests; small setup helper (pick model, RAM, GPU split) |
| **v1.0** | Something to rely on | Linux + Windows; NVIDIA, AMD and Intel GPUs each tested; a compatibility list that is checked automatically per model; versioned releases tracking llama.cpp releases; continuous tests (identity + a speed regression guard); documentation for contributors |

Not on the road to v1.0 unless measurements justify it: a dynamic GPU expert tier, and an own inference engine (see `ENGINE_DECISION.md`).

## llama.cpp versions (policy, 2026-10-14)

- **Backend API 3** (ggml 0.26, llama.cpp `23b0202a1` and newer) is first-class: new features only need to work there.
- **Backend API 2** (ggml 0.25.1, llama.cpp `7fe450e19`) is maintained: bug fixes only, no new features, and CI keeps both builds green.
- API 2 is dropped when either (a) the AUR package (`llama.cpp-cuda-aidock-bin`, which these docs call the Arch `llama.cpp-cuda` package) moves to ggml 0.26 or newer,
  or (b) a third backend API appears and the CI minutes or the maintenance become a burden.
- Status on 2026-10-14: condition (a) is met (the AUR package moved to llama.cpp v0.6.0, `8345f333`, ggml 0.26.0 / API 3, on 2026-10-07), but the owner chose to
  keep API 2 in maintenance mode rather than drop it. It is dropped only by an explicit decision later.

## Interface choices

- **Web UI**: not built here. `llama-server` already serves a web UI at its address, and `moe-cache-server --open` opens it. Anything that speaks the OpenAI API (for example Open WebUI) can connect to the same address.
- **Setup GUI** (choose model, RAM, GPU split with sliders): reasonable for v0.4, a small local web page or a terminal menu.

## Status against the v1.0 list (2026-10-14, version 0.2.3.1)

| v1.0 requirement | State |
|---|---|
| Linux + Windows | Linux only tested. Windows code exists (platform layer, DLL build), never ran on real Windows. **Open: needs a Windows machine.** |
| NVIDIA, AMD and Intel GPUs each tested | NVIDIA (CUDA, Vulkan) and Intel iGPU (Vulkan) tested; **AMD not tested** (a Ryzen mini PC is waiting). |
| Compatibility list checked automatically per model | list exists (docs/COMPATIBILITY.md); `bench/run_matrix.py` in [moe-cache-benchmark](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark) runs identity + speed over a folder of models; not yet run by CI (needs models). |
| Versioned releases tracking llama.cpp | releases are tagged and built by CI with attestation; llama.cpp 7fe450e19 (API 2) and 23b0202a1 (API 3) are supported, each with its own plugin build. **Open: re-run the full identity matrix on 23b0202a1 (5 models done), run GLM-5.3-Flash on it (needs a model file), automate both builds in CI.** |
| Continuous tests (identity + speed regression guard) | CI builds, syntax-checks and runs the kernel bit-exactness check (needs no model). Identity and speed need models and a GPU runner. **Open.** |
| Documentation for contributors | CONTRIBUTING.md, TECHNIQUES.md, SECURITY.md exist. |
| Safety: automatic fallback on a fault | read faults fall back to buffered reads; a wrong model file disables caching instead of aborting; an unrecoverable read still stops the server (never serves wrong data). |
| Sanitizer runs | whole plugin under ASan + UBSan with the executor, forced evictions and byte checking: clean. |

Summary: not v1.0. The remaining gaps are hardware coverage (Windows, AMD), a moving llama.cpp pin, and CI that can run models. Version 0.2.3.1 is feature-complete for the Linux NVIDIA/CPU case.
