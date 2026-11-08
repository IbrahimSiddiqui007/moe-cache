# Patches for the bundled llama.cpp

Since v0.3.8 the Windows kit and the Linux `-ggml0.26` tarball bundle llama.cpp b12064 built by CI from the official source plus this patch set (`llama/PATCHES-moe-cache.txt` names the patches; the launcher prints the line at start-up). The plugin does not need the patches: moe-cache keeps working on stock llama.cpp, and the official build can be used instead (`LLAMA_SERVER`).

**Pinned base:** release `b12064`, commit `51fe42c5d14bfb9adf0612ff98a9ab79e0e7275c` (ggml 0.28.0). Moved from b11966 (`ff0d396e3`) on 2026-11-03; patches 0004, 0005, 0020, 0030 and 0040 were checked against b12064 with `git apply --check` (all apply). 0022 (upstream #31266) and 0027 (upstream #30881 + #31418) are merged upstream and moved to `archive/`.

Patches: `0004-ple-prefetch-ahead`, `0020-server-slot-save-async`, `0030-vulkan-global-queue-priority`, `0040-reserve-top-tensors` (sections below). **Nothing is default-on:** all four patches are built into the kits and every one is off unless its switch is set (`LLAMA_PLE_PREFETCH_AHEAD`, `LLAMA_SLOT_SAVE_ASYNC`, `GGML_VK_GLOBAL_PRIORITY`, `LLAMA_RESERVE_TOP`), until a paired run wins. Owner decision 2026-11-08: ship the patched build now so later patches have a path to users; the first one to be turned on will be one that has a measured paired win, set by the launcher.

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
