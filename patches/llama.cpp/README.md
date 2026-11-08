# Patches for the bundled llama.cpp

Since v0.3.8 the Windows kit and the Linux `-ggml0.26` tarball bundle llama.cpp b12207 built by CI from the official source plus this patch set (`llama/PATCHES-moe-cache.txt` names the patches; the launcher prints the line at start-up). The plugin does not need the patches: moe-cache keeps working on stock llama.cpp, and the official build can be used instead (`LLAMA_SERVER`).

**Pinned base:** release `b12207`, commit `998200491327347eed85f007e4de7a1a7c871453` (ggml 0.29.0, backend API 3). Moved from b12064 (`51fe42c5d`) on 2026-11-09 by the owner's decision (upstream's own MoE expert cache landed: PRs 30966, 31233, 31888); patches 0004, 0020, 0030, 0040 and 0090 were checked against b12207 with `git apply --check` (all apply); 0020 needed a port (upstream turned the prompt checkpoints into `shared_ptr<const common_prompt_checkpoint>`: it compiled nowhere until its three uses were changed; the async slot file is byte-identical to the normal one, checked with Granite on b12207). Earlier: moved from b11966 (`ff0d396e3`) on 2026-11-03. 0022 (upstream #31266) and 0027 (upstream #30881 + #31418) are merged upstream and moved to `archive/`.

Patches: `0004-ple-prefetch-ahead`, `0020-server-slot-save-async`, `0030-vulkan-global-queue-priority`, `0040-reserve-top-tensors`, `0090-vulkan-iq3xxs-mmv`, `0091-cpu-gpu-row-split` (sections below). **Nothing is default-on:** all six patches are built into the kits and every one is off unless its switch is set (`LLAMA_PLE_PREFETCH_AHEAD`, `LLAMA_SLOT_SAVE_ASYNC`, `GGML_VK_GLOBAL_PRIORITY`, `LLAMA_RESERVE_TOP`, `GGML_VK_IQ3XXS_MMV`, `GGML_CPU_GPU_SPLIT`), until a paired run wins. Owner decision 2026-11-08: ship the patched build now so later patches have a path to users; the first one to be turned on will be one that has a measured paired win, set by the launcher.

## Rules

Patches that lost a paired run move to `archive/` at the top of the repo, with how to restore them (0005-tokenize-cache, PATCH-022, and 0070-decode-reloc, DECODE-630: `archive/llama.cpp/`).

1. Small numbered patches: `0001-short-name.patch`, `0002-...`, applied in name order. One idea per patch.
2. Each patch has its own runtime switch (an environment variable or flag), **off by default**. A patch only becomes default-on after a paired run (with and without it) shows a win.
3. The plugin must still work on stock llama.cpp. Nothing in `src/` may require a patch.
4. Patches apply to exactly the pinned commit. Moving the pin means re-checking every patch.

## When the pin moves

Owner's decision, 2026-11-02. We do not rebuild for every llama.cpp release. Patches never hold a move back: an upstream fix should not wait for a patch of ours.

| Reason to move | Testing |
|---|---|
| Security fix | at once: CI and a short PC check |
| ggml backend API change | full paired PC run |
| New model architecture we want | full paired PC run |
| Upstream speed gain that a paired run confirms | full paired PC run |

Non-security moves wait at most a week, so one test cycle covers everything pending. Before a move, losing patches go to `archive/`, so only patches still in play are re-checked.

Checklist for every move:

| Step | What |
|---|---|
| 1 | Archive the patches that lost their test (no repeatable gain, or covered upstream) into `archive/llama.cpp/` with a restore note in `archive/README.md`; only live patches are re-checked against the new pin (PATCH-129; first run at the next move, UPSTREAM-039) |
| 2 | AMD: do not pick a build from 10-13 to 10-16 (host-import regression, Linux confirmed; #29741, #30468). Pins b11669 and b11966 are outside that range (UPSTREAM-016) |
| 3 | Read the open Vulkan memory pull requests and note which ones touch the allocation or host-import path before choosing the build |
| 4 | Pin tracking is by hand today; an automatic check of new releases is an open item on the roadmap |

## Apply

```bash
tools/apply_llama_patches.sh /path/to/llama.cpp
```

It checks the base commit, runs `git apply --check` on each patch, then applies it. It stops at the first failure and succeeds when there are no patches.

## CI

`.github/workflows/llama_patched.yml` runs only when asked: a push to the branch `ci/llama-patched` (manual dispatch needs the file on main). It builds the patched `llama-server` for Windows (Vulkan) and Linux (Vulkan) with the flags, runners and Vulkan SDK of llama.cpp's own release workflow at the pin and uploads them as artifacts. Windows is two artifacts, as upstream: server + CPU backends, and `ggml-vulkan.dll`; put them in one folder. First run 2026-11-01 green with patches 0001-0005 (about 12 min); the version line then said build 11966, commit ff0d396e3.

## Shipping it (v0.3.8)

`release.yml` calls `llama_patched.yml` (a `workflow_call`) in the same run, then packs its artifacts: the Linux job unpacks `llama-patched-linux-x64-tgz` (sha256 checked) and passes it to `tools/make_release.sh` as `LLAMA_PATCHED_LINUX_DIR`; the Windows kit job unpacks `llama-patched-win-x64-zip` and passes it to `tools/make_windows_kit.sh` as `LLAMA_PATCHED_WIN_DIR`. Both call `tools/bundle_llama_patched.sh`, which also writes `llama/PATCHES-moe-cache.txt`, the README and the licence. Without those variables (a local build) the scripts bundle the official release build as before, and a manual release run can choose `llama_build: official`. `llama_patched.yml` also runs the identity check (`tools/llama_identity_run.sh`): official vs patched with switches off vs patched with 0004, 0020 and 0040 on, three prompts, 100 tokens, temperature 0, byte-identical, a failure stops the release.

## 0004-ple-prefetch-ahead (P22)

**What:** b11966 already prefetches the lazily read per-layer-embedding (PLE) rows of a ubatch (`llama_prefetch_rows`, called from `llm_graph_input_gemma4_ple::set_input`, so right before that ubatch computes; `madvise(MADV_WILLNEED)` on Linux, `PrefetchVirtualMemory` on Windows). The reads for the first rows of a ubatch can therefore still stall that ubatch. The patch adds a look-ahead: with `LLAMA_PLE_PREFETCH_AHEAD=1`, `llama_context::decode` hints the rows of every token of the whole batch (up to `n_batch`) once, before the first ubatch, so the reads of ubatch 2, 3, ... overlap with the compute of ubatch 1, 2, ... Only Gemma 4 (row index = token id) with a lazy PLE tensor in a host buffer; any other model or a non-lazy load does nothing. The existing per-ubatch call stays.

**Soundness:** a hint only; the data read and the graph are unchanged, so output cannot change. Risk: on a box short of RAM the read-ahead can push out pages that are still needed (the page cache then holds up to `n_batch` rows of `n_embd_per_layer * n_layer` values each); the time is spent in the syscall (sorted and merged ranges, one `madvise` per merged range), measure it.

**Not here:** token-id lookahead across batches (the server only knows the next chunk when it builds it), and Gemma 3n (loaded without lazy read, so there is nothing to prefetch). The audit notes the old profile attributes only about 3 % of a prompt to these gaps; expect at most that.

**Compiled:** `libllama` (Linux, Release). Not run: no Gemma 4 model on the laptop. The Windows branch is the existing upstream `llama_prefetch`, not compiled here.

**Test (under 20 minutes, Windows PC):** a Gemma 4 GGUF with lazy PLE (the default `--lazy-mode auto` decides; the log line `enabling prefetch for 'per_layer_token_embd.weight'` must appear), same long prompt (about 34k tokens), `MOE_CACHE_SEG_TIMES=1`, A B B A with A = unset and B = `LLAMA_PLE_PREFETCH_AHEAD=1`, cold file cache each run if you can (otherwise the rows are already resident and nothing can be seen). Compare total prompt time and the per-ubatch time of the first rows. Accept only if B is faster by more than the A-vs-A spread; output tokens must be identical.


## 0020-server-slot-save-async

**What:** `LLAMA_SLOT_SAVE_ASYNC=1` writes a saved slot (prompt state) on a background thread instead of inside the request; `LLAMA_SLOT_SAVE_ASYNC_MAX_MIB` caps what may wait. Off unless set. Writes use a temp file named per task and stale `*.tmp` files older than an hour are removed. The saved bytes are the ones the normal path writes.

## 0030-vulkan-global-queue-priority

**What:** `GGML_VK_GLOBAL_PRIORITY=low|medium|high` asks the Vulkan driver (VK_KHR_global_priority) to schedule this process's queues ahead of or behind other programs' GPU work. Unset = unchanged. If the driver refuses, the device is created the normal way and a line is printed. Scheduling only: no arithmetic changes.

## 0040-reserve-top-tensors

**What:** a diagnostic and nothing else. `LLAMA_RESERVE_TOP=N` prints the N largest tensors alive at the estimated peak of the reserved prompt graph (name, operation, shape, size). It allocates nothing, reserves nothing, frees no GPU memory and changes no output, whether set or not. Unset: no line, no behaviour change. The name is misleading (it was meant to find what to shrink); an earlier ledger note that it "frees GPU memory" and was "tested and kept" was wrong: no speed or memory win was ever measured for it, and none can exist. It shipped in the kit so a user can send the line with a bug report (the largest tensor of the qwen4exp graph is the 512 MiB lightning-indexer score tensor at `-ub 2048`, PATCH-802). The identity check runs it switched on (the log line must appear) and the answers match the official build byte for byte.


## 0090-vulkan-iq3xxs-mmv

**What:** upstream pull request ggml-org/llama.cpp#30904 ("vulkan: optimize mul_mat_vec kernel for iq3_xxs", still open when taken, 2026-11-08; MIT, author WakeUpMorty, who says an AI assistant helped write it) applied to b12064 behind a switch. It restructures the IQ3_XXS mat-vec shader: 8 or 16 invocations per block, the sign selects hoisted, one accumulator per invocation. `GGML_VK_IQ3XXS_MMV=1` builds the IQ3_XXS mat-vec pipelines (f32, f16 and the expert `_id` one) with the new kernel through a new specialization constant (`MC_IQ3_NEW`, constant id 3); unset or 0 = the original kernel, unchanged (constant-folded away at pipeline creation). The shader file `mul_mat_vec_iq3_xxs.comp` holds both versions; the PR's diff alone applies cleanly to b12064 (`git apply --check`), no other upstream change is needed.

**Soundness:** same maths, different summation order, so results are not bit-identical to the original kernel; they match the CPU result within the tolerance of llama.cpp's own `test-backend-ops` (53 of 53 `MUL_MAT` / `MUL_MAT_ID` iq3_xxs cases pass with the switch off and on, run on the RTX 4060 laptop GPU 2026-11-08, and in CI on mesa lavapipe in `llama_patched.yml`). The "same meaning" rule applies: an A B B A run checks the text of a 65k chat as well as the speed.

**Measured:** by the author, on an RX 9070 XT on Windows: kernel -23 to -27 %, end to end +1.8 to 5 %, a 27B Q3_K_XL model +2 %. Ours: on the laptop's NVIDIA RTX 4060 the new kernel is slower (`test-backend-ops perf`, MUL_MAT iq3_xxs m=4096 n=1 k=14336: 1272 us off, 1380-1397 us on, about +9 % time), so it is a vendor-specific gain and stays off by default. Needs the paired PC run on the 9070 XT (IQ3_XXS, 65k chat, A B B A).

## 0091-cpu-gpu-row-split (DECODE-917)

**What:** `GGML_CPU_GPU_SPLIT=auto|FRACTION` (0.05-0.6; `auto` follows the measured wait) lets the GPU compute a share of the rows of every decode-time mat-vec whose weights stay in host RAM, while the CPU threads compute the rest of the rows. Only a single-token `MUL_MAT` of at least 1024 rows on plain host weights (not a repacked buffer) is split; everything else runs as before. Unset = unchanged. `GGML_CPU_GPU_SPLIT_STATS=1` prints, at exit, the begin cost, the CPU time and the GPU wait per op and the final share (`GGML_CPU_GPU_SPLIT_SERIAL=1` times the GPU path alone). Launcher: `--cpu-gpu-split [auto|FRACTION]`, opt-in.

**How:** the CPU backend's `MUL_MAT` (`ggml-cpu.c`) asks `cpu-gpu-split.cpp` before it computes. Thread 0 starts the GPU half asynchronously (activation into a pinned bounce buffer, GPU mat-vec, result back into the bounce buffer), the CPU threads compute rows `[0, n - n_gpu)`, thread 0 waits for the GPU and copies its rows into the result. Everything goes through the public backend API plus one proc address, so a second backend only has to offer a way to reach host memory: CUDA page-locks the weight buffer (`cudaHostRegister`, plain flag when the read-only flag is refused) and copies the GPU's rows by DMA into a staging tensor (12.9 GB/s alone, 11.5-12.7 beside CPU decode in C1); Vulkan imports the host rows as a GPU buffer (`VK_EXT_external_memory_host`) and the kernel reads them in place (no copy: async copies cost 30-37 % on the PC in the gpu-expert-cache session). The graph and the scheduler are not touched, which avoids the scheduler's GPU synchronisation on every cross-backend copy.

**Needs:** page-lockable weights. A file-backed `mmap` cannot be page-locked: load with `--load-mode none` (`--no-mmap`) and `--no-repack` (the launcher does both with the flag). If the weights cannot be locked or imported, that tensor is computed by the CPU alone and one line says so. When a weight buffer is freed (a model unloaded or replaced), a callback (`ggml_backend_buffer_set_free_hook`, added by this patch) unregisters its page-lock and drops every staging tensor and import, so a new buffer at the same address starts clean. One split op runs at a time over the whole process (a second context waits at its split ops; it never reads another's data).

**Soundness:** the GPU quantises the activation itself (Q8_1) where the CPU uses Q8_K, and sums in another order, so results are not bit-identical to the CPU path. Greedy 160-token answer, Qwen3-14B Q4_K_M, 18 FFN layers in RAM: the first 148 tokens are identical (CUDA); Vulkan (the laptop's NVIDIA driver): identical for 90, same content after. CI (`llama_patched.yml`, Linux job) runs `tests/test-cpu-gpu-split.cpp`: Q4_0, Q8_0 and Q4_K weights, 300 ops each with the automatic share moving, every row against a float reference, then a freed and replaced weight buffer. A second CPU backend plays the GPU there (`GGML_CPU_GPU_SPLIT_TEST=1`), so the test checks the row bookkeeping, the plans and the reload, not CUDA or Vulkan.

**Measured (laptop, RTX 4060 8 GB, Qwen3-14B Q4_K_M, FFN of layers 22-39 in RAM, `llama-bench -n 24`, 12 runs in A B B A order, per arm the median).** Quiet laptop: **+6.7 % and -2.6 %, below the +10 % bar, with a 25 % spread inside each arm: no gain shown.** With one core kept busy (controlled disturbance): +19 % (6 threads, share 0.35) and +39 % (the launcher's thread layout `-t 8 -C 0xff`, `auto` settles at 0.30-0.35); the run ranges of the two arms overlap in the second case (A 3.98-6.14, B 5.34-7.16), and a busy core slows the CPU half, which favours any offload, so this is the best case, not the typical one. GPU memory: +56 MiB. Not measured: the PC, AMD, Windows, any other model. Therefore off by default, and the deciding run is the PC (Vulkan, a dense model that spills, 64k context).

**Reopen / archive:** at the next pin move, if the PC run (patch 0091 item in the test plan) shows no paired gain beyond the A-vs-A spread, move it to `archive/llama.cpp/`.
