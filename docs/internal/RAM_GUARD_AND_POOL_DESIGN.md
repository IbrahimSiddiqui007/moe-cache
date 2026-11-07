# CACHE-153 / CACHE-058 / DECODE-137: RAM guard and decode pool size (2026-11-03)

All three are switches, default off, output identical to stock (nothing here touches arithmetic or which bytes are read).

## Design

| Item | Switch | What it does | Where | Exactness | Fails safe how |
|---|---|---|---|---|---|
| CACHE-153 keep the cache out of swap | `MOE_CACHE_RAM_LOCK=1` | `mlock2(MLOCK_ONFAULT)` on every range the cache commits; `munlock` before an eviction gives the pages back (`MADV_DONTNEED` refuses locked pages) | `src/mc_ramguard.h`, hooks in `mem_commit` / `mem_decommit` in `src/platform.h` | none: only the kernel's swap choice changes | limit hit (`RLIMIT_MEMLOCK`, 8 MiB by default for a user) -> one stderr line, guard off, cache goes on unlocked |
| CACHE-058 drop read-ahead | `MOE_CACHE_DROP_READAHEAD=1` | once, after the startup fill: `posix_fadvise(DONTNEED)` on each model file | `drop_readahead()` in `src/mc_state_profile.inl`, called from `src/mc_compute.inl` after `preload()` | none | skipped in file-view mode (page cache is the cache); mapped pages are never dropped by the kernel |
| DECODE-137 two pools (archived 2026-11-07: decode +0.6 %, `archive/plugin/exec_pool_decode.patch`) | `MOE_CACHE_EXEC_POOL_DECODE=1` | pool built as before for a first prompt graph, rebuilt once to `-t` threads at the first single-token graph if the count differs | `src/mc_poolsize.h`, `exec_run_graph` in `src/mc_compute.inl` | thread count only shares out rows, each row is summed the same way | `MOE_CACHE_EXEC_THREADS` wins; no rebuild when counts are equal |

Windows: CACHE-153 and CACHE-058 are clean no-ops (the file-view cache keeps its pages in the working set already). Windows still gets DECODE-137. The desktop PC is Windows, so the two RAM items need the laptop or a Linux boot to be measured.

## Tests

| Test | What it proves | Run |
|---|---|---|
| `tests/ramguard_test.cpp` | off by default; 4 MiB file: 1024 cached pages before the drop, 0 after; locked pages refuse `MADV_DONTNEED`, unlocked accept | `tools/run_quick_tests.sh` |
| `tests/identity.sh` on qwen4exp-moe.gguf (71 MB) with all three switches and `-t 2 -tb 4`, laptop 2026-11-03 | 4/4 prompts identical; drop line shown; lock refused by the 8 MiB limit and the guard switched itself off with its message | by hand |

Not exercised live: the pool rebuild (the tiny model never builds the pool, the executor steps aside when everything is resident) and a successful lock (hard memlock limit here is 8 MiB). Both need the PC chain below.

## PC chain plan (A B B A, IQ3_XXS base model)

| Item | Arms (A = stock, B = switch) | Context / ub | Minutes per item | Kill rule | Record |
|---|---|---|---|---|---|
| CACHE-153 + CACHE-058 (Linux box only) | B = `MOE_CACHE_RAM_LOCK=1 MOE_CACHE_DROP_READAHEAD=1`, `LimitMEMLOCK=infinity`; run with RAM nearly full (other load or `--limit`) so swap is possible | 34k and 64k, ub 1024 | 4 runs x 6 = 24 | decode or prompt tok/s worse than A by more than 2 % in either pair, or any token differs, or swap-in (`vmstat si`) is 0 in A (nothing to fix) | VRAM, RAM (RssAnon, RssFile, buff/cache), CPU %, GPU layers, ctx, ub, llama.cpp + plugin versions, `vmstat si/so` |
| DECODE-137 (PC and laptop) | B = `MOE_CACHE_EXEC_POOL_DECODE=1`, explicit `-t 8 -tb 16` (use the box's real counts, tb larger than t) | 34k, 64k, 128k, ub 1024 | 4 runs x 6 = 24 | decode not above A by 1 % in both pairs, or prompt tok/s down more than 3 % (the pool is now -t sized for prompts), or any token differs | same as above plus the "pool resized" log line |
| Total | | | 48 | | |
