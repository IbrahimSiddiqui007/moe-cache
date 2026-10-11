# Changelog

## 0.2.1 (2026-10-11)
- **Second supported llama.cpp**: commit `23b0202a1` (ggml 0.26.0, backend API 3; GLM-5.3-Flash and MTP support are in it) next to `7fe450e19` (ggml 0.25.1, API 2). Each needs its own plugin build: the release has one tarball per version (`...-ggml0.26-...` is the new one); from source use `GGML_API=3 ./build.sh`.
- A plugin built for one backend API now refuses to register in a ggml of the other API (it prints why and the model loads normally) instead of crashing.
- `install.sh`: `LLAMA_SHA=23b0202a1...` builds llama.cpp at the new commit and the matching plugin.
- Token identity vs stock re-checked on the new commit: Granite 3B and 1B, OLMoE 7B, Qwen3.6-35B, Qwen3-Coder-30B, a second qwen35moe model, Qwen3.8-Flash-Next IQ2_XS (4 prompts x 48 tokens each, all identical). Not re-run there: gpt-oss, Qwen3-Next 80B, DeepSeek-V2-Lite, Hunyuan.
- New tests: `tests/kernel_variants.cpp` (IQ2_XXS / IQ3_XXS dot product candidates, bit-exactness and timing) and `tests/read_bench.cpp` (O_DIRECT read speed: pread, pre-faulted pages, io_uring batches).
- Measurements in `docs/RESULTS.md` section 15, including llama.cpp's own new `--moe-cache-mib` (GPU cache of host experts): it does not combine with this plugin yet.

## 0.2.0 (2026-10-11)
- **Expert executor** (default on; steps aside when everything is resident and the weights are bandwidth-bound): background reader threads, work-sharing compute pool that starts on experts already in RAM, optional efficiency-core workers (`MOE_CACHE_EXEC_*`). Bit-identical to llama.cpp's output; `MOE_CACHE_EXEC_CHECK=1` compares every op byte for byte.
- **Fast Q2_0 x Q8_0 kernel** (AVX2, bit-identical, about 3x; the executor uses a second, permuted-activation version that is about 6x): llama.cpp has no SIMD version on x86. `MOE_CACHE_FAST_KERNELS=0` switches it off. `tests/kernel_check.cpp` verifies it.
- `bench/run_matrix.py`: benchmark many models, placements and memory limits and print a publishable table.
- Planner: built-in GGUF header reader (new quantisation types), split models, AMD VRAM, n-gram table not counted as GPU weights.
- Security and privacy: benchmark GUI token, Host/Origin checks and config validation; shareable reports without prompts or paths; private result store; hardened build flags; validated profile loading; graceful handling of a wrong `MOE_CACHE_GGUF`.
- Tests: identity test fails when the plugin is not active; limit test cleans up its own server; port-busy checks.
- Release: built by GitHub Actions from the tagged commit, with a build attestation; tarball includes all of `src/`.

## 0.1.4
- Benchmark tool documented for your own models and prompts; area 3 GUI fixes.

## 0.1.3
- First release built by GitHub Actions; attestation.

## 0.1.2
- Sibling batch load, read-into-place, automatic pinning, preload-all, benchmark GUI, installer.

## 0.1.1
- First prebuilt plugin.
