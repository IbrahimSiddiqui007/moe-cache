# Patches for the bundled llama.cpp

The Windows kit and the Linux `-ggml0.26` tarball bundle an official llama.cpp build. This folder holds a small patch set for that build. The plugin does not need it: moe-cache keeps working on stock llama.cpp.

**Pinned base:** release `b11966`, commit `ff0d396e3dda0f33e7ce5ad4267fce4f7690cfb1` (ggml 0.28.0).

Patches: `0001-vulkan-fence-spin-iters`, `0002-server-adaptive-draft-length`, `0003-vulkan-mmid-vec-max-tokens`, `0004-ple-prefetch-ahead`, `0005-tokenize-cache`, `0020-server-slot-save-async`, `0030-vulkan-global-queue-priority` (below, more follow). Nothing is measured yet. Every patch is off unless its switch is set.

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

`.github/workflows/llama_patched.yml` runs only when asked: a push to the branch `ci/llama-patched` (manual dispatch needs the file on main). It builds the patched `llama-server` for Windows (Vulkan) and Linux (Vulkan) with the flags, runners and Vulkan SDK of llama.cpp's own release workflow at the pin and uploads them as artifacts. Windows is two artifacts, as upstream: server + CPU backends, and `ggml-vulkan.dll`; put them in one folder. First run 2026-11-01 green with patches 0001-0005 (about 12 min); the version line says build 11966, commit ff0d396e3.

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

## 0005-tokenize-cache (S08)

**What:** `common_tokenize(vocab, text, add_special, parse_special)` (common/common.cpp) is the one place the server tokenizes prompts, including the text that comes out of the chat template. With `LLAMA_TOKENIZE_CACHE=1` it keeps the result of the last 4 texts of at least 4096 bytes. A call with the same text, flags and vocab returns the stored tokens; anything else is tokenized as before. Unset = no cache, no extra copy of any text.

**Soundness:** the key is the whole text (compared byte for byte), both flags, the vocab pointer plus a signature (vocab size, type, the text of 16 tokens spread over the vocab, because a reloaded model can reuse an address). Nothing is reused across different texts: tokenizing a prefix and a suffix separately does not give the tokens of the whole text (merges across the boundary), so a prefix cache is not sound and is not done. Chat templates are not cached: only the tokenizing of their output is. Risk: two vocabs with the same size, type and 16 sampled token texts at the same address would collide (only after unloading and loading a different model in the same process, no case known); a few MB of memory per cached text (text copy + tokens).

**Honest scope:** it only helps when the exact same long text arrives again (retry, regenerate, the same benchmark prompt, n choices of one request). An agent loop that appends to its history changes the text every request and never hits. The prompt-prefix KV reuse in the server is separate and unchanged.

**Compiled and run (Linux):** `llama-common`, `llama-server`. A small program on the vocab-only `ggml-vocab-llama-bpe.gguf` with a 166 KB text: both calls return the same 58001 tokens; the second call took 0.02 ms with the cache and 8.1 ms without (the first call 9.3 ms in both runs); a text with one extra byte and the same text with `add_special=false` give their own results. That is the tokenizer alone, not TTFT.

**Test (under 20 minutes, any box):** with a real model, send the same 100k-token chat request twice to `llama-server` (A: unset, B: `LLAMA_TOKENIZE_CACHE=1`), `--cache-prompt` off so the prompt path is the same, and compare the time before the first prompt token is processed (server log `prompt eval` start vs request arrival). Expect no change on the first request and only a tokenizer-sized saving on the second; if that is not at least a few percent of TTFT the patch is not worth enabling. Output tokens must match.

## 0020-server-slot-save-async (S04)

**What:** `llama-server` handles `POST /slots/N?action=save` inside its main loop: the sequence state is streamed from the GPU into the file there, and every other slot waits (`SERVER_TASK_TYPE_SLOT_SAVE`, `server-context.cpp`). With `LLAMA_SLOT_SAVE_ASYNC=1` the loop only copies the sequence state (`llama_state_seq_get_data_ext`) and the slot's context checkpoints into memory; a thread writes the file as `NAME.tmp` and renames it when complete, then sends the answer to the request. So the reply still means "the file is complete", the loop is held for the copy only, and a crash or a full disk during the write leaves the old file in place instead of a half-written one. Switches: `LLAMA_SLOT_SAVE_ASYNC=1`, `LLAMA_SLOT_SAVE_ASYNC_MAX_MIB` (default 4096: state plus checkpoints above this are saved the old way, so the extra RAM is bounded). A server with a draft model context (`ctx_dft`) also saves the old way. Unset = stock.

**Soundness:** the file is byte-identical to the stock one (same header, state, checkpoint appendix and speculative appendix; the 8-byte header that `llama_state_seq_get_data_ext` adds is dropped). Threads are joined in the server destructor.

**Honest scope:** the GPU-to-host read of the state stays in the loop, so the pause shrinks by the file-write time only. That is large when the page cache is under pressure (models bigger than RAM) and small when the disk keeps up. Costs: RAM equal to the state size for the duration of the write (this counts against our 28 GB).

**Compiled and run (Linux, Release, Vulkan build):** `llama-server` with the 1.1 MB `stories260K.gguf` (CPU): save with and without the switch gives byte-identical 22476-byte files, restore from the async file works and the next completion reuses the 25 cached tokens, no `.tmp` is left. Timings on that model are meaningless (0.1 ms).

**Test (under 20 minutes, Windows PC):** Flash-Next with a 64k prompt processed, then `POST /slots/0?action=save` while a second request decodes on another slot (or measure decode tok/s of one slot during a save). A B B A, A = unset, B = `LLAMA_SLOT_SAVE_ASYNC=1`; read the longest gap between two streamed tokens of the decoding request and the save reply time. Watch peak RAM. Accept only if the gap shrinks by more than the A-vs-A spread and the saved files restore to the same output.

## 0030-vulkan-global-queue-priority

**What:** other programs that draw on the same GPU (found 2026-11-01: the EA app's embedded browser kept the 3D engine 27-29 % busy and the desktop compositor another 10 %; llama-server's compute and copy queues wait behind that work) get scheduled against ours. The patch asks the Vulkan driver for a global queue priority (`VK_KHR_global_priority`) when the device is created. Switch: `GGML_VK_GLOBAL_PRIORITY=low|medium|high|realtime`. Unset = unchanged. If the driver does not have the extension or refuses the priority (high and realtime can need extra rights), a line is printed and the device is created the normal way. Where it applies: Windows and Linux, any driver with the extension.

**Related switches without a patch:** `GGML_VK_ENABLE_MEMORY_PRIORITY=1` (already in b11966: priority 1.0 on every allocation) and the plugin's `MOE_CACHE_ECOQOS_OFF=1`, `MOE_CACHE_PROC_PRIO=above|high`, `MOE_CACHE_GPU_PRIO=above|high|realtime` (Windows; power throttling, CPU priority class, GPU scheduling class of the process).

**State:** compiles (Linux Vulkan build and the Windows CI build). Paired result: see the test log of the session; default stays off until a paired run wins.

## Not written, and why

Each of these was checked against the b11966 source. None has a patch.

- **P19 reuse scratch memory.** The graph allocator (`ggml-alloc.c`, `ggml_gallocr_alloc_graph_impl`) already frees every tensor at its last use, reuses the block of a single-use parent in place for the ops in `ggml_op_can_inplace`, and best-fits into free blocks; the scheduler's cross-backend copies are ordinary tensors unless pipeline parallelism (`n_copies > 1`, several GPUs) is on, and those copies must all stay alive by design. What is left is in ggml-vulkan: `prealloc_x`, `prealloc_y` and `prealloc_split_k` are three buffers sized to the largest need of each; one buffer sized to the largest sum per op would be smaller. It is not safe in a small patch: `prealloc_y` is a cache (`prealloc_y_last_tensor_used` skips re-quantising the same activation for the next matmul) and the `*_need_sync` flags assume fixed regions; sharing the region needs the cache invalidated on every write and a re-check of all 10 sizing sites and the flash-attention uses. That is a design and a GPU validation pass of more than a day, and I cannot test it on the AMD driver here. The plugin-owned buffers the audit points at (`src/mc_backend.inl`) are outside llama.cpp.
- **U09 lend idle output-head memory to prefill.** Weights live in buffers allocated once at load, and the compute buffer is reserved once for the worst-case ubatch (`sched_reserve`). Lending the head's VRAM to a bigger prefill chunk means freeing and re-uploading a weight buffer on every prefill/decode switch and re-reserving the compute buffers for a larger `n_ubatch` each time; the scheduler, the context and the loader all assume neither moves. A static version already exists without a patch: put the head on the CPU with `--override-tensor` (slower decode every token). More than a day of design; not a safe patch.
- **L07 plan MTP and expert residency together.** This is a decision about numbers (how much VRAM the draft head, its KV and buffers take versus the expert cache), made before `llama-server` starts. It belongs in `bin/moe_cache.py` / the planner, which can read the sizes from the GGUF and the load log. llama.cpp has nothing to switch.
- **P23 batch PLE projection.** Already batched upstream: `project_per_layer_inputs` (gemma3n.cpp) and the Gemma 4 graph do one `mul_mat` of `per_layer_model_proj` over all `n_tokens` of the ubatch, then scale, norm, add. Nothing token-at-a-time is left in the stock graph. (If the plugin's own path does it per token, that is a `src/` change, not a patch.)
- **P21 layer-wise prefill across micro-batches.** It needs the graph for several chunks to be built layer by layer (all chunks through layer l before layer l+1), with the KV and recurrent state written in an order the graph does not use today, plus the scheduler and b11966's own `llama-moe-cache.cpp` (expert upload through the sched copy callback) to agree on residency across chunks. That is a rewrite of the decode loop, not a switchable patch, and the intermediate activations of all chunks have to be held at once. Not understandable in a small patch; skipped.
- **U03 fused quantized GPU expert kernels.** New shaders per quant type and shape, a numerical comparison per type and a portable fallback, tuned on the target GPU (AMD, which cannot be tested on this laptop). The decision which shapes are hot needs the Windows profile first. Skipped.
- **P15 grouped GPU work by populated tiles.** Already upstream: `ggml_vk_mul_mat_id_q_f16` counts rows per expert (`count_experts`) and `mul_mm.comp` reads each tile's row count from `data_row_map`, so only populated tiles run.
- **P24 routing metadata on the GPU.** Needs a GPU-side route compaction plus a resident-slot lookup and a host miss path across the scheduler; days of work, and the router read-back is not among the measured costs.
- **U10 zero-copy for tiny transfers.** `ggml_vk_buffer_read_2d` already reads host-visible memory directly on UMA; on discrete GPUs upstream deliberately uses the DMA copy because CPU reads over PCIe/ReBAR are slower (comment in the source).
- **U05 capture/replay command graphs.** The Vulkan backend has no replay object; submit size is already tunable with `GGML_VK_MAX_NODES_PER_SUBMIT` (no patch; sweep it on the PC).
- **U01 attention kernel tuning.** b11966 already has split-K decode (`flash_attn_decode_phase_1/2`) and sparse mask options; tuning needs the kernel time at depth first: one run with `GGML_VK_PERF_LOGGER=1` at 34k/128k on the PC (5 min) before any shader change.
- **U04 fused gate/up/activation, U06 tiled CPU fallback.** A new shader / CPU kernel each; GPU compute is 9.3 s of 100 s in the 34k run, and the tiled CPU routine is not used in that run.
- **U07 overlap shared and routed branch.** The shared FFN is built after the routed sum (`qwen3next.cpp`), the scheduler runs splits one after another on one thread, and the CPU backend is synchronous; needs a per-architecture graph reorder and async CPU splits.
- **U08 CPU/GPU split by completion time.** Plan only (docs/SCHEDULER_DESIGN.md); needs scheduler hooks and measured per-layer costs.
- **L06 grow KV only as needed.** The KV cache is one tensor per layer sized for the whole context at creation (`llama-kv-cache.cpp`, cleared once); growing means reallocating, copying and re-reserving the compute buffers. A "skip the clear" trick helps only host-resident KV and only where fresh memory is zero; no measurement shows where the KV sits in our runs.
