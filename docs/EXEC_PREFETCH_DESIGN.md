# R-02 software prefetch in the CPU executor, and R-03 (design only) (2026-11-03)

Source: handover/parts/research_round_20261103.md (ranks 2 and 3). Gains below are that note's guesses; the laptop numbers are measured here.

## R-02: built, default off, laptop says loss, PC decides
`MOE_CACHE_EXEC_PREFETCH=N` (N = 1..16 rows ahead, `on` = 4, unset/0/off = nothing runs): while the executor's pool computes weight row r, it asks for the cache lines of row r+N (`src/mc_prefetch.h`, called from the row loop of `src/mc_pool.inl`, 7 added lines). It reads nothing and changes no arithmetic, so output cannot change.
| Item | Value |
|---|---|
| Idea | the hardware stream prefetcher restarts at every 4 KiB page; a row is 1.5-4 KB, so the dot product waits on the first lines of each new page. Decode reads the CPU layers' experts at 24-25 GB/s where the PC can do about 50 (DECODE-004) |
| Exactness | same dot-product calls, same arguments; a prefetch cannot change a value or fault |
| Test | `tests/prefetch_test.cpp` (model-free, in `tools/run_quick_tests.sh`): parse and cap, lines per row, nothing asked past the last row |
| Bench | `tests/prefetch_bench.cpp`: int8 dot over 1 GB, chunked like the executor, distance 0/1/2/4/8/16 |
| Laptop result (Intel, 16 threads available, data not in cache) | rows 2048 B, 4 threads: 20.8 GB/s off, 17.9-18.9 GB/s at distance 1-8, 17.0 at 16; rows 4096 B, 8 threads: 27.7 off, 24.1 / 23.5 / 22.5 / 22.3 / 19.4 at 1/2/4/8/16; rows 1536 B, 12 threads: 30.7 off, 27.9 / 29.4 / 29.3 / 25.8 / 20.4. A loss of 5-30 % here: Intel's prefetcher already follows the stream and the extra instructions cost |
| What it means | not a win on the laptop, as the research note said ("needs the PC (Zen 4 memory system)"). Kept because it is 7 lines, off, and the PC test is 10 minutes; if the PC also loses it goes to `archive/plugin/` with a restore note |

## R-03: free the plugin's held block buffer at the start of decode (not built)
Why not built here: it needs a decode-start hook in `src/mc_backend.inl` / `src/mc_exec.inl` (another session edits them) and it only pays after DECODE-630 (patch 0070) is proven on the PC; the patch keeps 768 MiB free because the 1,107 MiB block buffer stays allocated, so releasing it leaves room for about one more layer (+4-7 % decode by the note's guess).
| Part | Plan | Days |
|---|---|---|
| Decision | free `c->galloc` after K consecutive single-token graphs (K=16), allocate again at the next block (the existing code already creates it when null, `mc_backend.inl:461`); pure counter in a new header, model-free test | 0.5 |
| Hook | one call where the executor starts a graph (`mc_exec.inl`) | 0.5 |
| Order with patch 0070 | the plugin must free before the patch measures free VRAM (patch switch after 64 small batches, plugin after 16): a log line shows both | 0.5 |
| Test | PC only: VRAM free at the end of decode, number of layers moved, decode tok/s | PC |

## PC A B B A plan (IQ3_XXS base model; add after chain1103)
Metrics on every row: VRAM used/free at prompt end and in decode, RAM, CPU use, GPU use, GPU layers (48 - `--n-cpu-moe`), `-c`, `-ub`, KV type, llama-server build, moe-cache version and commit, patch list, switches.
| Row | Test | Context | Arms | Minutes | Kill rule |
|---|---|---|---|---|---|
| 1 | R-02 kernel bench on the PC: `tests/prefetch_bench.cpp`, 1 GB, rows 1536 and 4096 B, 16 threads | none | distance 0/1/2/4/8/16 | 5 | kill if no distance gives +3 % GB/s |
| 2 | R-02 decode A B B A | 64k, 34k prompt, 256 tokens | A off, B `MOE_CACHE_EXEC_PREFETCH=<best distance from row 1>` | 24 (4 x 6) | kill if decode under +3 % in both pairs, or any prompt tok/s loss over 3 % |
| 3 | R-02 decode A B B A | 128k | same | 32 | same |
| 4 | R-03 (after it is built and DECODE-630 row 2 passes) | 64k, 128k | A DECODE-630 on, B DECODE-630 on + block free | 48 | kill if under +4 % decode, or the prompt after the decode phase loses over 3 % (re-allocating the block), or fewer than 700 MiB free in decode |
