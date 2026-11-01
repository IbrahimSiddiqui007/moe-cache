# Patches for the bundled llama.cpp

The Windows kit and the Linux `-ggml0.26` tarball bundle an official llama.cpp build. This folder holds a small patch set for that build. The plugin does not need it: moe-cache keeps working on stock llama.cpp.

**Pinned base:** release `b11966`, commit `ff0d396e3dda0f33e7ce5ad4267fce4f7690cfb1` (ggml 0.28.0).

Patches: `0001-vulkan-fence-spin-iters`, `0002-server-adaptive-draft-length`, `0003-vulkan-mmid-vec-max-tokens`, `0004-ple-prefetch-ahead` (below, more follow). Nothing is measured yet. Every patch is off unless its switch is set.

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

## 0001-vulkan-fence-spin-iters

**What:** `ggml_vk_wait_for_fence` (ggml-vulkan.cpp) waits for the last 20 % of a graph by polling `getFenceStatus` with 1000 `pause` instructions between polls (100 x 10 `YIELD()`). The patch makes the count an env value. Switch: `GGML_VK_FENCE_SPIN_ITERS=N` (N blocks of 10 pauses, max 100000). Unset = 100 = stock behaviour. Correctness: only the poll interval changes.

**Where the gap time likely goes (source reading, no symbols; all labels are guesses):**
1. Driver wait on the GPU: `waitForFences(almost_ready_fence)` in `ggml_vk_wait_for_fence`, reached from `ggml_vk_synchronize` (called by `ggml_backend_synchronize` after each GPU split/plugin graph, and by `tensor_get`/`tensor_set` sync paths). That is the amdvlk64 `WaitForMultipleObjects` (about 65 %). This is mostly real GPU work (upload, compute, readback). Sync changes cannot shorten it; no fix is claimed for it.
2. The spin loop after that wait (last 20 % of the nodes): `getFenceStatus` + 1000 pauses. Likely the +e5c50/+e5c60 loop (about 18 %): a tight pause loop fits two nearby addresses. Guess only. It also explains "reached through amdvlk64 stack values" if the poll call leaves driver values on the stack. Part of this is real GPU time, part is overshoot: after the fence signals, the thread finishes the current 1000-pause block (roughly 5-40 us depending on the CPU's pause latency) before it sees it.
3. memcpy of staging buffers (`in_memcpys`/`out_memcpys` in `ggml_vk_synchronize`, `buffer_write`): the VCRUNTIME copy at +f0c65 (about 5 %).
4. Lock waits, then `PrefetchVirtualMemory` (outside ggml-vulkan).
`ggml_backend_sched_compute_splits` adds no waits of its own in this setup except `ggml_backend_synchronize`/event sync of the previous split when a split has no inputs and the backend changes.

**Why this one:** it is the smallest change on a guessed hot loop, and it cannot change output. Upper bound of the gain: one poll interval per sync, times 1258 blocks (a few tens of ms at most per prompt if the guess is right). Honest expectation: small. Not measured. Costs: more `getFenceStatus` driver calls and more CPU use while spinning. Not chosen: dropping the `almost_ready` split (changes submit count, only graphs >= 10 nodes, likely not the plugin graphs) and replacing the spin by a blocking wait (adds wake-up latency).

**Apply:** `tools/apply_llama_patches.sh /path/to/llama.cpp` on a clean b11966. Checked here: applies cleanly; `ggml-vulkan` target compiled on Linux (Release, glslc + Vulkan headers). Not run on a GPU.

**Test (Windows PC, paired, under 20 minutes):** same model, same 34k prompt, same plugin settings, `MOE_CACHE_SEG_TIMES=1 MOE_CACHE_E1_TIMES=1`. Run order A B B A (4 runs, 3-4 min each) to cancel drift:
- A: `GGML_VK_FENCE_SPIN_ITERS` unset.
- B: `GGML_VK_FENCE_SPIN_ITERS=10` (try 1 if time is left).
Compare per run: total prompt time, and the sync/wait timers in the SEG/E1 output (the per-block time between compute end and readback end should drop slightly; upload and compute timers should not change). Accept only if B is faster in both pairs by more than the A-vs-A spread and the output tokens are identical (fixed seed, temperature 0). Otherwise leave it off. Watch CPU use of the graph thread: it should stay at 100 % either way.

## 0002-server-adaptive-draft-length (L08)

**What:** `llama-server` asks the drafter for a fixed `n_max` tokens each step. With `LLAMA_SPEC_ADAPT=1` each slot picks a cap from {0, 1, 2, 4} before every draft step (`get_n_draft_max()` in `tools/server/server-context.cpp`). The generic code already truncates a draft to that cap, so every drafter (draft model, MTP, n-gram) is covered. Switches: `LLAMA_SPEC_ADAPT=1`, `LLAMA_SPEC_ADAPT_COST` (relative cost of one extra drafted token in a verification step, default 0.15), `LLAMA_SPEC_ADAPT_PROBE` (steps at length 0 before one probe step at length 1, default 16).

**Rule:** per slot a decayed (x0.9 per step) count of accepted draft tokens and tested positions gives a per-position acceptance rate `a`. The cap is the k in {1,2,4} (or 0) that maximises `(1 + a + ... + a^k) / (1 + cost * k)`; 0 wins when nothing beats 1. With cost 0.15 the table is: a < 0.2 -> 0, 0.2-0.4 -> 1, 0.5-0.6 -> 2, >= 0.7 -> 4 (computed from the formula, not measured). Start at 4 until something is observed. Replay steps after a checkpoint restore and synthetic-acceptance steps are not counted. Cost is a guess to be fitted: it is not measured verification time.

**Soundness:** the verifier is unchanged, so the output distribution is the same. A different draft length changes batch shapes, so tokens can differ from a fixed-length run through rounding (not bit-identical, same as changing `--draft-max`). With the switch unset nothing changes. Risk: a stale rate (text switches from prose to code) costs a few steps; the probe at length 0 limits that. Drafting is still done at full `n_max` and then truncated, so the draft cost is not saved, only verification width.

**Compiled:** `llama-server` (Linux, Release). Not run (no draft/MTP model on the laptop that is used here).

**Test (under 20 minutes, Windows PC or any box with an MTP or draft model):** same model, three prompts (prose, code, agent-like JSON), temperature 0, `--draft-max 4`. Runs A B B A per prompt, A = unset, B = `LLAMA_SPEC_ADAPT=1`; 200 tokens each is enough. Compare generated tokens per second (server `timings`) and the printed acceptance per position. Accept only if B is not slower on any prompt by more than the A-vs-A spread and faster on at least one. Try `LLAMA_SPEC_ADAPT_COST=0.3` if B is slower where acceptance is high.

## 0003-vulkan-mmid-vec-max-tokens (L10)

**What:** ggml-vulkan runs `MUL_MAT_ID` as one small per-token mat-vec dispatch each when the batch has at most 8 tokens (`ggml_vk_use_mul_mat_vec_id`, hard-coded `ids->ne[1] <= 8`), and as the grouped matrix path above that. A draft of 8 plus the sampled token (9 rows) therefore changes path abruptly. Switch: `GGML_VK_MMID_VEC_MAX_TOKENS=N` (0..1024). Unset = 8 = stock. It is the only place the limit lives; the fused MUL_MAT_ID+scale check and `supports_op` ask the same function. The backend `supports_op` still asks for the matrix path to exist when N is lowered, so values below 8 can send some types to the CPU.

**Scope:** this covers only the Vulkan boundary. The CPU backend and the plugin's own executor limits (`EXEC_MAXCOL`, the tiled-arithmetic switch at 8 rows per expert) are not touched by any patch here.

**Risk:** the two paths round differently, so changing N can change logits slightly (not bit-identical). The vec path issues one dispatch set per token, so a large N can be slower or cost more descriptor sets; it is a tuning knob, not a fix.

**Compiled and checked:** `ggml-vulkan` (Linux, glslc). `test-backend-ops test -o MUL_MAT_ID -b Vulkan1` (RTX 4060 laptop): 1109/1109 pass with the default and 1109/1109 with `GGML_VK_MMID_VEC_MAX_TOKENS=64`. Intel iGPU (Vulkan0) passes with the default only. That is a correctness check of the knob, not a speed result.

**Test (under 20 minutes, Windows PC):** MTP or draft decoding with a fixed `--draft-max` of 6, 8 and 10 (verification batches of 7, 9 and 11 rows), A B B A with A = unset and B = `GGML_VK_MMID_VEC_MAX_TOKENS=16`, 200 generated tokens, temperature 0. Compare time per verification step (server timings) at each draft length. A step at 9 rows should be the one that moves. Accept B only if it is faster at 9 and 11 rows by more than the A-vs-A spread and the generated tokens match A (or differ only after a near-tie, check the logprob gap).

## 0004-ple-prefetch-ahead (P22)

**What:** b11966 already prefetches the lazily read per-layer-embedding (PLE) rows of a ubatch (`llama_prefetch_rows`, called from `llm_graph_input_gemma4_ple::set_input`, so right before that ubatch computes; `madvise(MADV_WILLNEED)` on Linux, `PrefetchVirtualMemory` on Windows). The reads for the first rows of a ubatch can therefore still stall that ubatch. The patch adds a look-ahead: with `LLAMA_PLE_PREFETCH_AHEAD=1`, `llama_context::decode` hints the rows of every token of the whole batch (up to `n_batch`) once, before the first ubatch, so the reads of ubatch 2, 3, ... overlap with the compute of ubatch 1, 2, ... Only Gemma 4 (row index = token id) with a lazy PLE tensor in a host buffer; any other model or a non-lazy load does nothing. The existing per-ubatch call stays.

**Soundness:** a hint only; the data read and the graph are unchanged, so output cannot change. Risk: on a box short of RAM the read-ahead can push out pages that are still needed (the page cache then holds up to `n_batch` rows of `n_embd_per_layer * n_layer` values each); the time is spent in the syscall (sorted and merged ranges, one `madvise` per merged range), measure it.

**Not here:** token-id lookahead across batches (the server only knows the next chunk when it builds it), and Gemma 3n (loaded without lazy read, so there is nothing to prefetch). The audit notes the old profile attributes only about 3 % of a prompt to these gaps; expect at most that.

**Compiled:** `libllama` (Linux, Release). Not run: no Gemma 4 model on the laptop. The Windows branch is the existing upstream `llama_prefetch`, not compiled here.

**Test (under 20 minutes, Windows PC):** a Gemma 4 GGUF with lazy PLE (the default `--lazy-mode auto` decides; the log line `enabling prefetch for 'per_layer_token_embd.weight'` must appear), same long prompt (about 34k tokens), `MOE_CACHE_SEG_TIMES=1`, A B B A with A = unset and B = `LLAMA_PLE_PREFETCH_AHEAD=1`, cold file cache each run if you can (otherwise the rows are already resident and nothing can be seen). Compare total prompt time and the per-ubatch time of the first rows. Accept only if B is faster by more than the A-vs-A spread; output tokens must be identical.
