# Changelog

## 0.2.3.2 (2026-10-14)
- `moe-cache-server --fast`: opt-in approximate routing (`MOE_CACHE_ROUTE=skip`), about +10-12 % when the model is much bigger than RAM, output no longer identical
  to stock. The default stays exact; the launcher suggests `--fast` when the model is more than twice the memory it may use.
- The launcher tells the plugin how much memory llama-server's prompt cache needs (`MOE_CACHE_HOST_RESERVE_MIB`) instead of the plugin reading the command line
  (kept as a fallback).
- `MOE_CACHE_GUARD=1` (tests only): an evicted expert's pages become inaccessible, so a read the plugin did not intercept crashes instead of seeing zeros.
- README: a "When to use something else" table (stock llama.cpp when the model fits, Strata for Flash-Next / Qwen3.6 on 12 GB+ VRAM and 32 GB+ RAM), a table of
  which features need which loading method, the Flash-Next row, MTP and Vulkan claims qualified. QUICKSTART: 2x-12x instead of 2x-7x. BACKENDS: the OpenVINO NPU backend.
- ROADMAP: llama.cpp version policy (API 3 first-class, API 2 bug fixes only, and when API 2 is dropped).
- **The benchmark tool moved to its own repository**, [moe-cache-benchmark](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark) (`bin/moe-cache-bench`, `bench/*.py`, BENCHMARKS.md, TEST_PROTOCOL.md).
  It finds moe-cache through `MOE_CACHE_DIR` or a folder next to it. The shared tuner moved from `bench/gguf_plan.py` to `planner/gguf_plan.py`.
- ROADMAP: llama.cpp API 2 stays in maintenance mode (bug fixes only) although its drop condition is met.

## 0.2.3.1 (2026-10-14)
- **Fix: CPU weight repacking now follows llama-server's `--no-repack` / `-nr` / `LLAMA_ARG_REPACK=0`.** On CPU-only machines the plugin repacked expert weights even when the server
  was started without repacking, so its kernels added up in another order than stock and tokens differed late in the answer (found by the new CI job).
- **CI: `identity` workflow** (GitHub runner, no GPU): llama.cpp built CPU-only at both pins, Granite 3.1 1B-A400M (sha256-checked), stock vs plugin with the cache forced
  to evict (about 39k evictions), with the executor forced on and byte-checked (6,708 ops, 0 mismatches), a noise-floor run and a no-eviction run; 4/4 identical on API 2 and API 3.
  `tests/identity.sh` now prints the first diverging token per prompt.

## 0.2.3 (2026-10-12)
- **Fix: long-prompt processing with the GPU expert cache.** The plugin's long-prompt GPU block needs about one layer of free VRAM; with a full-size cache it silently
  stayed on the CPU (prefill 24 tok/s) or aborted the server. The plugin now checks free VRAM and warns once; the tuner leaves room for the block
  (Flash-Next: cache 1800 instead of 2459 MiB at ctx 8192; prefill 24 -> 56 tok/s, decode about -5 %). RESULTS.md section 23.
- **Saved contexts**: `moe-cache-server --project NAME` (own usage profile plus restore of the saved prompt state at start) and `bin/moe-cache-project save|restore|list`.
  Restoring a 2.7k-token context took 0.25 s instead of 56 s with identical output (RESULTS.md section 24.2).
- `bench/coding_session.py --sections N` warms a context in growing prefixes so an edit in the middle reuses everything before the nearest checkpoint
  (886 tokens processed instead of 5,945; section 24.1). The context builder now respects its token budget (about 2 characters per token).
- The launcher picks the plugin build that matches the llama-server's ggml API (`build*/` next to each other) and warns on a mismatch; plugin builds carry a
  readable `moe-cache-api=N` tag.
- Executor counters by weight type with `MOE_CACHE_STATS=1` (time, ops, expert data per type).
- `tools/wait_cool.sh` (the clean-run gate as one command); the benchmark tools' default start temperature is now 65 C.
- `tools/trace_replay.py` (cache policies and next-layer prediction from a `MOE_CACHE_TRACE_IDS` recording); `bench/quality.py codegen` (10 small code tasks run against assertions).
- RESULTS.md sections 24 (prompt reuse, saved contexts, kernel time by type, cache policy, predictor recall) and 25 (quantisations, thinking, KL divergence of the approximate modes).

## 0.2.2 (2026-10-11)
- Opt-in **approximate routing** (`MOE_CACHE_ROUTE=skip|subst`, `MOE_CACHE_ROUTE_KEEP`): low-ranked picks that are not in RAM are dropped or replaced by a resident expert.
  +10-12 % at a small cache on Flash-Next with no measured loss on the checkable prompts (RESULTS.md section 21). Off by default.
- One-line capability report in the server log (`moe-cache: active: ...`): load method, executor and why it is off, kernels, GPU-cache mode.
- Shared tuner (`bench/gguf_plan.py: tune`) for micro-batch, GPU cache or static layers; `moe-cache-server --ub auto`.
- Fixes from the review: disk lookup table not counted as GPU weight in the benchmark planner, identity compares every repeat, manual `--moe-cache-mib` sets the plugin's
  host-buffer mode, MB/MiB units, quality-harness false passes. New `bench/coding_session.py`, `bench/quality.py judge-calibrate`.
- First coding-session measurement (RESULTS.md section 22): a repeated 20k-token context is reused (about 790 s cold, about 50 s per later question), but an edit in the
  middle of the context re-processes all of it. Timings were not temperature-gated.
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
