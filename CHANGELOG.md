# Changelog

## 0.2.3.5 (2026-10-15)
- **No abort when the cache is smaller than one op** (a fixed `MOE_CACHE_SIZE_GIB` below the experts one step needs; `auto` never gets there): the experts
  that do not fit are mapped straight from the model file, the way stock llama.cpp loads them (file cache, reclaimable, so the size still holds), or, where
  that is not possible (executor's background reads, repacked weights, Windows), loaded over the size for that step. One warning, and a count in the
  summary line. Two CI steps run with a 1 MiB cache (mapping with guard mode on; executor forced on) and must be 4/4 identical.
- Removed five switches whose experiments lost (all were off by default): `MOE_CACHE_L3PREFETCH` (-19 %), `MOE_CACHE_THP` (no gain),
  `MOE_CACHE_SCAN_TOKENS` (-25 % prefill), `MOE_CACHE_PREFAULT` (no gain), `MOE_CACHE_PROMPT_PREFETCH` (no gain). Setting them now does nothing.
- The plugin source is split into ten parts (`src/mc_*.inl`: common, pool, state, buffer, backend, exec, hybrid, compute, device, interpose), included
  by `src/moe_cache_plugin.cpp` in order: still one translation unit, so build commands do not change; the compiled code is byte-identical (`.text` and `.rodata`).
- **Launcher: stock llama-server when the model fits** in the memory the server may use (file + 1.5 GB <= `--ram`, the cgroup limit or available RAM):
  stock keeps the whole file in the page cache, warm across restarts (Qwen3.6-35B at 24 GB: stock 42.9 vs plugin 36.6 warm / 33.4 cold tok/s, RESULTS 44).
  `--fast` and `--hybrid` keep the plugin; `--always-plugin` forces it.
- **Launcher: no prompt-cache reserve below 16 GB** (`--cache-ram 0` instead of 2048): the 2 GiB reserve took a large share of the expert cache at small
  limits (Qwen3.6-35B at 12 GB: 16.9 tok/s with 2048 MiB, 19.2 with 512, 22.6 with 0, stock 6.9; this was the unexplained drop from RESULTS 2b's 23.5, audit
  E02). A single chat keeps reusing its slot; only switching between conversations loses the host prompt cache. Pass `--cache-ram` to choose.
- RESULTS 43: quant yardstick for `--fast` (T8): the Q2_0 file of Flash-Next is KL 0.243 from IQ2_XS (88.9 % same top token); `--fast` is 0.005-0.012.
- **Python launcher** (T15): `bin/moe_cache.py server|check|learn|project`, standard library only, also for Windows (plugin loaded with `GGML_BACKEND_PATH`).
  The bash scripts `bin/moe-cache-*` now call it; same command lines (`--dry-run` identical to the bash launcher in 10 flag combinations on Flash-Next and
  GLM). Fixed on the way: `--gpu-margin` was ignored (now passed to the tuner; GiB; default unchanged), `--vram` meant GiB for the planner but GB for the tuner
  (now GiB for both), `moe-cache-learn` refuses an occupied port and fails when its server stops or no new profile appears, project save/restore report
  HTTP errors and accept `--api-key`, the check counts every shard of a split model and knows 23b0202a1, and among several plugin builds for the same API the
  newest is used (it was the first by name, which picked old test builds). CI runs a launcher dry run and the check.
- Fixes from the static audit of 2026-10-15 (handover/STATIC_AUDIT_20261015_1324.md):
  - **`--fast` and `--hybrid` were silently inactive on K-quant models in 0.2.3.4** (the K-quant speed filter switched the executor off, and both modes
    live in it; output stayed exact). An explicitly requested mode now keeps the executor.
  - Lookahead (`MOE_CACHE_LOOKAHEAD`) set and cleared the "still loading" flag with plain stores while the executor reads it without the lock: atomic now.
  - The hybrid bank's GPU memory is freed with the backend; a freed model's experts are forgotten (no later eviction touches memory that was released).
  - Experts are checked byte for byte (first 4 KiB) against `MOE_CACHE_GGUF`, not only by name and size: another model with the same layout no longer
    gets the wrong weights (that tensor loads the normal way, with a warning).
  - Generic reads of expert memory (`get_tensor`) load the expert first; the executor pool cannot let a late worker read a job slot being refilled.
  - Profiles are written to a per-process temporary file and only replace the old one when every write succeeded.
  - The tuner also finds AMD GPU memory (it overrode the planner's AMD answer with "no GPU"). `install.sh` installs the newest release (it pinned
    v0.1.2); the release workflow checks that. `tools/windows_test.ps1` now turns the cache on (`-CacheGiB`, default 0.5) and fails when it is not on.
  - Tests: `identity.sh` fails when no prompt produced at least 8 tokens; two CI identity steps that could not fail now gate; the T45 test now reaches
    the step-aside wait (Q8_0 copy of the tiny gpt-oss, routing on with every pick kept, so output stays exact) and a mutation build without that wait
    must fail it. `tools/bank_replay.py` follows the live bank policy. RESULTS 41/42 wording qualified.

## 0.2.3.4 (2026-10-15)
- **Speed fix for K-quant models that do not fit in RAM** (in releases 0.2.0-0.2.3.3): the executor only takes graphs whose expert weights are IQ types or
  Q2_0; K-quants and legacy types go llama.cpp's own way, whose kernels are faster. Qwen3.6-35B Q4_K_M at 12 GB: 7.8 -> 16.6 tok/s, identical to stock.
- `MOE_CACHE_LOOKAHEAD=2` (experimental): the lookahead prefetch together with the executor (exact; no reliable gain on GLM-5.3-Flash, RESULTS 41).
- `MOE_CACHE_TEST_READ_DELAY_US=N` (tests only): background expert reads start N us late, so a code path that does not wait for an expert still loading
  reads stale data every time instead of only by bad luck. Checked on gpt-oss-20b: the build without the T45 fix fails 0/4, the fixed build passes 4/4.
- The plugin's Q2_0 kernels step aside when llama.cpp has its own (llama.cpp master, ggml-org/llama.cpp#30447: 96 ns per 2560-weight dot vs the plugin's
  126-238 ns, and adding up in another order). Checked once at start by comparing the library's function with the plain C loop; on 23b0202a1 nothing changes.
- The prompt-cache reserve line says where the number came from (launcher, `LLAMA_ARG_CACHE_RAM` or the command line).

## 0.2.3.3 (2026-10-15)
- **Exactness fix (race, in releases v0.2.0-v0.2.3.2):** when gate, up and down come in separate calls (gpt-oss), the experts read in the background
  for the next call could be used by llama.cpp's own path before they were complete (the executor stepped aside because they looked resident).
  gpt-oss-20b at a 1 GiB cache: 0/4 -> 4/4 identical. RESULTS section 36 (also: 7 more models identical on llama.cpp 23b0202a1).
- `MOE_CACHE_GPU_MIN_TOKENS=1` gave one repeated token (single-token batches on the GPU; Granite 3.1, T25); 1 is now treated as 2 with a warning.
- **Prompt reads merged** (`MOE_CACHE_COALESCE_MIB`, default 16, 0 = off): neighbouring expert reads of a long prompt become one read of up to 16 MiB.
  Flash-Next prompts read at 3.4 instead of 2.5 GB/s; prefill with the default launcher 74-81 -> 90-103 tok/s (RESULTS section 38).
- **Prompt speed fix for the default launcher** (`--gpu-cache off`): static expert layers and `-ub` now come from one tuner decision that keeps room for the
  long-prompt GPU block at the chosen micro-batch (1.6x a layer at `-ub 2048`). Flash-Next on 8 GB: prefill 18 -> 64-76 tok/s (RESULTS section 35).
- Launcher: `-b` follows a `-ub` above 2048 (it was fixed at 2048, which capped `--ub 4096`).
- `--hybrid` and `--fast` keep prompt batches under 64 tokens on the CPU (`MOE_CACHE_GPU_MIN_TOKENS=64`): short chat prompts 4.6 -> 3.3 s to the first token.
  The exact default keeps 32, as llama.cpp.
- `MOE_CACHE_PROMPT_PREFETCH=1` (experimental, off): read the next layer's experts in the background during long prompts; no gain measured.
- RESULTS section 34: the hybrid on GLM-5.3-Flash (bank too small on 8 GB) and on Qwen3.6 (slower than `--gpu-cache` for Q4_K experts).
- **Exactness fix (race, batches of 8 tokens):** in executor graphs the expert reads are asynchronous; the ops handed back to llama.cpp (tiled case, entry below)
  did not wait for them and could read an expert that was still loading. They wait now. Seen as logit differences at batch 8 with evictions (KL 0.004-0.006, Flash-Next
  and Qwen3-Coder); never in a release (it came with the tiled hand-back). New `tests/kl_batch.sh` (llama-perplexity logits at batch 8, stock vs plugin, must be KL 0)
  and a CI step that runs it with forced eviction and the forced executor (RESULTS section 31).
- **`moe-cache-server --hybrid`** (opt-in, experimental): sets `MOE_CACHE_HYBRID=auto`, keeps all experts on the CPU side and replaces `--gpu-cache`. Quality on Flash-Next:
  KL 0.0043 vs exact with 57 % of the picks on the GPU, 98.4 % same top token (`--fast` at 24 GB: 0.0053); RESULTS section 31.
- Hybrid: when a long prompt's GPU block needs the memory the bank holds, the bank is freed and loaded again when decoding starts. Prefill with `--hybrid`:
  17 -> 66-75 tok/s (`--gpu-cache auto`: 60-64), about 1 s per long prompt for the reload (RESULTS section 32).
- Hybrid: adaptive bank (`MOE_CACHE_HYBRID_ADAPT=N,K`, default 8,16, 0 = off): every N tokens up to K of the most missed experts replace the coldest expert of their
  layer's bank. Mixed-topic session: bank hits 20 -> 39 %, decode 13.1 -> 14.4 tok/s; when the profile already matches the prompts exactly it costs a little
  (51 -> 47 % hits, 17.6 -> 16.2 tok/s, one run). `tools/bank_replay.py` replays a `MOE_CACHE_TRACE_IDS` recording offline (RESULTS section 33).
- Hybrid: the adaptive bank switches itself (`MOE_CACHE_HYBRID_DUEL=W`, default 32 tokens): it compares itself with what the profile's own bank would have
  served and stops swapping (bringing profile experts back) while the profile does better. Matching profile 17.6 (fixed) -> 17.05 tok/s; drifting session
  12.8 -> 14.9 tok/s (+16.5 %) (RESULTS section 33.1).
- `--fast` quality re-measured without the race: KL 0.0053 at 24 GB, 0.0115 at 6 GiB (it was 0.025); about 1 top token in 40-50 differs.
- **Exactness fix (GLM-5.3-Flash):** llama.cpp 23b0202a1 computes an expert with a separate tiled routine once 8 or more tokens of a batch picked it, and that adds up
  in another order than the one-row kernel the executor uses. The executor now leaves exactly those ops to llama.cpp. Before: 3 of 12,348 ops differed in the last bit
  and one answer changed at its first token (3/4); after: 0 mismatches, 4/4, multi-token batches still on. New switch `MOE_CACHE_EXEC_MAXCOL=N` (1-8, default 8).
- `MOE_CACHE_EXEC_CHECK=1` now prints where the first mismatches are (weight type, tokens, first differing value).
- Faster IQ3_XXS dot product in the executor (gather-based, bit-identical, 1.24x on the dot product; Intel CPUs only). GLM-5.3-Flash's down projections are IQ3_XXS.
- The automatic cache size uses the smaller of the memory limit and what the machine has available (a 28 GB limit on a laptop with 25 GB free was OOM-killed).
- The executor finds llama.cpp's CPU kernels in builds with separately loaded backends (official releases, Vulkan, Windows), where it used to switch itself off.
  Checked on a Vulkan build (llama.cpp 7fe450e19, `GGML_BACKEND_PATH`): executor on, Flash-Next 4/4 identical to stock (RESULTS section 27).
- `tests/identity.sh`: `IDENT_KEEP=DIR` keeps both server logs and token files.
- **Experimental `MOE_CACHE_HYBRID=MiB|auto`** (off by default, not token-identical): a GPU bank of the most used experts; the GPU computes the picks it holds while
  the CPU executor computes the rest, in parallel, with nothing uploaded per token. Flash-Next on an 8 GB card: 13.1 -> 17.4 tok/s (RESULTS section 30).
- RESULTS section 26: GLM-5.3-Flash (101.8 GB) at 24 GB: 0.32 -> 1.7-1.9 tok/s, identical; 2.0-2.3 with `--fast`. docs/MULTI_DEVICE_DESIGN.md (design only).

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
