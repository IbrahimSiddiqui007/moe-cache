# Versions and roadmap

Version numbers describe how safe it is to hand the project to someone else.

| Version | Meaning | What must be true |
|---|---|---|
| **v0.1** (now, alpha) | Works on the author's machine; others can build and try it on Linux | Plugin runs unmodified llama.cpp; token-identical to stock on every model tested; launcher, planner, identity test, quick start; honest compatibility page |
| **v0.2** (friend-ready) | A friend with a similar Linux box can install it and it works first time | Prebuilt plugin for the common llama.cpp releases; setup check that says exactly what is wrong; tested on at least one AMD or Intel GPU through Vulkan and on CPU-only; first GitHub release |
| **v0.2b** | CPU-only parity | Support llama.cpp's repacked CPU layout (read an expert from the file and repack it on load) so CPU-only machines are bit-identical and as fast as stock |
| **v0.3** | Windows | Windows port of the memory and file code; tested with official Windows llama.cpp builds; `GGML_BACKEND_PATH` loading |
| **v0.4** | Safety and polish | Automatic fallback to stock behavior on a fault; long-context and multi-user tests; small setup helper (pick model, RAM, GPU split) |
| **v1.0** | Something to rely on | Linux + Windows; NVIDIA, AMD and Intel GPUs each tested; a compatibility list that is checked automatically per model; versioned releases tracking llama.cpp releases; continuous tests (identity + a speed regression guard); documentation for contributors |

Not on the road to v1.0 unless measurements justify it: a dynamic GPU expert tier, and an own inference engine (see `ENGINE_DECISION.md`).

## Interface choices

- **Web UI**: not built here. `llama-server` already serves a web UI at its address, and `moe-cache-server --open` opens it. Anything that speaks the OpenAI API (for example Open WebUI) can connect to the same address.
- **Setup GUI** (choose model, RAM, GPU split with sliders): reasonable for v0.4, a small local web page or a terminal menu.
