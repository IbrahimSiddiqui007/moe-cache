# Patches for the bundled llama.cpp

The Windows kit and the Linux `-ggml0.26` tarball bundle an official llama.cpp build. This folder holds a small patch set for that build. The plugin does not need it: moe-cache keeps working on stock llama.cpp.

**Pinned base:** release `b11966`, commit `ff0d396e3dda0f33e7ce5ad4267fce4f7690cfb1` (ggml 0.28.0).

Patches: `0001-vulkan-fence-spin-iters`, `0002-server-adaptive-draft-length` (below, more follow). Nothing is measured yet. Every patch is off unless its switch is set.

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
