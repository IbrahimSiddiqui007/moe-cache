# Changelog

## Unreleased (on `experimental`)
- **Plugin + llama.cpp's GPU expert cache** (`MOE_CACHE_GPU_CACHE=1` with `--moe-cache-mib N`, or `moe-cache-server --gpu-cache auto`): the plugin makes experts resident
  right before llama.cpp uploads them. Identical to stock + cache; +12 % over static GPU layers on Flash-Next (4 rounds).
- **Small-batch executor**: the executor now takes up to 8 tokens per step (prompt batches, speculative verification), exact (0 byte mismatches in 57,669 compared ops).
- **Named profiles**: `moe-cache-server --profile coding`, `moe-cache-learn --name coding`.
- **Benchmark tool**: finds your models (Ollama, llama.cpp cache, Hugging Face, LM Studio, folders), runs many in a queue, redesigned page, Serve card.
- **Quality harness** `bench/quality.py` (compare, longctx, soak).
- **Testers' kit**: AVX-512 IQ2_XXS / IQ3_XXS kernels (opt-in `MOE_CACHE_AVX512=1`, self-checked, untested on real AVX-512 hardware), `tools/friend_test.sh`,
  `docs/BACKENDS.md` (ROCm / SYCL / Vulkan build notes, untested), `docs/FRIEND_TEST.md`.
- Opt-in `MOE_CACHE_PREFAULT=1` (no gain measured on Flash-Next), `MOE_CACHE_EXEC=2` (force the executor, for tests).
- New measurement tools: `tests/read_bench.cpp`, `tests/h2d_bench.cu`, `tests/sparsity_probe.cpp`, `tests/kernel_variants.cpp`.
- Analysis: `docs/PERFORMANCE_ANALYSIS.md`.
- Prompt processing: `-ub 2048` measured +58 % at a 6k prompt (short context); long contexts need a smaller `-ub` because the compute buffer grows with the context.
  Opt-in `MOE_CACHE_SCAN_TOKENS=N` (scan-resistant loading of prompt batches: -25 % prefill, +31 % first decode after a prompt; off by default).
- **Long-session memory safety**: llama-server's host prompt cache (`--cache-ram`, 8 GiB by default) is now accounted for. A 24 GB limit was OOM-killed at 128k context
  when a second request started. The plugin reads `--cache-ram` from the command line and keeps that much free; `moe-cache-server` and the benchmark tool cap it at 2 GiB
  unless you set it.

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
