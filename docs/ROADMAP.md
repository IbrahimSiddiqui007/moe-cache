# Versions and roadmap

Version numbers describe how safe it is to hand the project to someone else.

| Version | Meaning | What must be true |
|---|---|---|
| **v0.1** (done) | Works on the author's machine; others can build and try it on Linux | Plugin runs unmodified llama.cpp; token-identical to stock on every model tested; launcher, planner, identity test, quick start; honest compatibility page |
| **v0.2** (friend-ready) | A friend with a similar Linux box can install it and it works first time | Prebuilt plugin for the common llama.cpp releases; setup check that says exactly what is wrong; tested on at least one AMD or Intel GPU through Vulkan and on CPU-only; first GitHub release |
| **v0.2b** (done) | CPU-only parity | Repacks each expert on load exactly when stock would: CPU-only machines are bit-identical and as fast as stock |
| **v0.2c** (done) | GPU-assisted prompt processing | Run the big expert multiplications of long prompts on the GPU from inside the plugin (restores stock's prompt speed when the model fits; building block of the GPU tier) |
| **v0.3** | Windows (cross-compiled, tested under Wine, in CI and on real Windows hardware; the Python launcher runs there) | Windows port of the memory and file code; tested with official Windows llama.cpp builds; `GGML_BACKEND_PATH` loading |
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

- **Web UI**: `llama-server` serves its own chat page (`moe-cache-server --open` opens it); anything that speaks the OpenAI API (for example Open WebUI) can connect to the same address.
- **Setup GUI**: built since v0.3.0 as the **control page** (`moe-cache-server --ui`, `moe-cache.bat` on Windows): model picker with the planner's verdict, RAM / VRAM / context settings, start and stop, chat, live numbers, agent connect actions and Share my numbers ([GUI.md](GUI.md)). It is separate from llama-server's own page. Remaining gaps: screenshots in GUI.md predate the newest panels.

## Status against the v1.0 list (as of 2026-10-30, version 0.3.3; v0.3.4 and v0.3.5 (2026-11-03) were released after it: see CHANGELOG.md)

| v1.0 requirement | State |
|---|---|
| Linux + Windows | Linux tested. Windows: identity 4/4 on real hardware (MSI Claw bugs fixed; Windows 11 desktop Ryzen 9 7950X + RX 9070 XT, RESULTS 54) and in CI. Clean cold speed runs on the desktop (RESULTS 59-61): faster than stock on every model measured (1.2x-2x decode, agent sessions 2.0-2.4x). **Open: MSI Claw round 3, NVIDIA and Intel GPUs on Windows.** |
| NVIDIA, AMD and Intel GPUs each tested | NVIDIA (CUDA, Vulkan), Intel iGPU (Vulkan), AMD iGPU (Vega 8, Linux) and AMD discrete (RX 9070 XT: a user on Linux, issue #1; the developer on Windows, RESULTS 54) tested. |
| Compatibility list checked automatically per model | list exists (docs/COMPATIBILITY.md); an automatic per-model check is not built yet (the old `bench/run_matrix.py` lives in the unmaintained [moe-cache-benchmark](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark) repository); not run by CI (needs models). |
| Versioned releases tracking llama.cpp | releases are tagged and built by CI with attestation; llama.cpp 7fe450e19 (API 2) and 23b0202a1 (API 3) are supported, each with its own plugin build. **Open: re-run the full identity matrix on 23b0202a1 (5 models done), run GLM-5.3-Flash on it (needs a model file), automate both builds in CI.** |
| Continuous tests (identity + speed regression guard) | CI builds, runs unit tests and the kernel bit-exactness check on every push (`quick`), and runs identity on small models on CPU runners daily (`identity`, five llama.cpp rows including master), Windows nightly and sanitizers weekly (table in [EXACTNESS.md](EXACTNESS.md)). **Open: identity on big models and a GPU, and any speed regression guard.** |
| Documentation for contributors | CONTRIBUTING.md, TECHNIQUES.md, SECURITY.md exist. |
| Safety: automatic fallback on a fault | read faults fall back to buffered reads; a wrong model file disables caching instead of aborting; an unrecoverable read still stops the server (never serves wrong data). |
| Sanitizer runs | whole plugin under ASan + UBSan with the executor, forced evictions and byte checking: clean. |

Summary: not v1.0. The remaining gaps are Windows on more than one machine, a Linux run on the AMD desktop, a moving llama.cpp pin, and CI that can run models. Version 0.3.0 was feature-complete for the Linux NVIDIA/CPU case; the Windows work since then is in CHANGELOG.md.
