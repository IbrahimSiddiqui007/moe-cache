# Give GPU layers back after the prompt (DECODE-630, design and build note, 2026-11-03)

**Status: built, not tested on a GPU model run at 64k. Default off (`MOE_CACHE_DECODE_RELOC=1` plus a patched llama.cpp).**

## The problem in three lines
At 64k, `-ub 2048` reads the prompt about 1.4x faster than `-ub 512`, but its compute buffer leaves room for 2 fewer expert layers on the 16 GB card. The PC measured the cost (DECODE-210 sizing, 2026-11-03, A B B A, 64k): `-ub 512` with 38 CPU layers decodes 26.63 / 26.92 tok/s, `-ub 2048` with 40 CPU layers 23.74 / 22.91 (+15 %); prompt 294 / 276 against 406 / 460 tok/s. The wish: prompt with 2048, decode with 38 layers.
Earlier work: the plugin route (a decode tier of whole layers kept by the plugin, `docs/DECODE_TIER_DESIGN.md`) was built and killed on the PC (exact, decode -1 %, prompt -41 %, `archive/plugin/decode_tier.patch`). Its own result said the +15 % needs llama.cpp's placement of the layers. This is that route.

## Options (design table, written before building)
Gain column: the measured ceiling is +15 % decode at 64k (2 layers); everything else is a guess and says so. "Break-even" = generated tokens after which the one-time switch cost has paid back, using 42.9 ms per token (23.3 tok/s) against 37.3 ms (26.8 tok/s), i.e. 5.6 ms saved per token.

| # | Option | What it needs | Exact vs stock? | Cost per request | Build time | Verdict |
|---|---|---|---|---|---|---|
| 1 | Plugin keeps extra whole layers on the GPU in decode (decode tier) | plugin, `mc_hybrid.inl` | exact (check mode), tried | no gain: decode -1 %, prompt -41 % | done (1 week), killed | archived |
| 2 | Launcher only: serve the prompt with the big chunk, save the slot (`/slots?action=save`), stop the server, start it with `-ub 512` and more GPU layers, restore the slot | a proxy in the launcher that holds the request, two model loads per long prompt | exact (KV bytes restored, then stock placement) | a model load per long prompt: 20 s / 40 s / 60 s load = break-even 3,600 / 7,100 / 10,700 tokens (guess: load times not measured) | 3-5 days | too slow for chat and agents; wins only for very long answers |
| 3 | **llama.cpp patch 0050: after the prompt, shrink the micro-batch, re-reserve, copy the experts of the last host layers into the freed VRAM; move them back when a prompt chunk comes** | the patch (about 200 lines in `llama-context.cpp/.h`), planner/launcher switch | exact per phase: same weights, same graph, the moved layers run with the GPU kernels exactly as in a stock run that places them on the GPU. Not bit-identical to either single stock run (the prompt ran with the other placement) | 2 scheduler reserves (about 3 s each, guess from fit-params) plus copying 1.3 GiB (about 0.3-1 s): about 6-7 s, break-even about 1,100-1,250 tokens (guess; the PC log prints the real time) | 1 day (this note) | **built, default off** |
| 4 | Patch 0042 (`LLAMA_N_OUTPUTS_MAX`, logits rows) and a smaller planner reserve: same layers in both phases, no switching | restore archived patch, planner table | exact | none | 1-2 days | open, separate lever; not tried here |
| 5 | Planner only: pick `-ub 512` with 2 more layers | planner | exact | none | 0 | prompt -30 %: the 2 x 2 test (RESULTS 78) said the chunk size matters |

Picked: **3**, the cheapest option that keeps outputs the same as stock and has a real consumer for the freed memory. Options 1 and 5 are measured losers; option 2 costs a model load per long prompt.

## How patch 0050 works
`patches/llama.cpp/0050-decode-reloc.patch`, switch `LLAMA_DECODE_RELOC=1` (unset or 0: no code runs, nothing is read).
| Step | Event | What happens |
|---|---|---|
| 1 | `LLAMA_DECODE_RELOC_AFTER` (64) batches in a row of at most `LLAMA_DECODE_RELOC_SMALL` (16, or the slot count) tokens | decode phase starts |
| 2 | decode phase starts | scheduler dropped (big compute buffer freed), `n_ubatch` set to `LLAMA_DECODE_RELOC_UB` (512), scheduler reserved at the small size |
| 3 | free memory of the GPU that holds the layer's other weights | the last host layers (highest index first) whose expert tensors sit in plain host buffers (not repacked) are copied into one new GPU buffer each, while `free >= LLAMA_DECODE_RELOC_KEEP_MIB (768) + layer` and fewer than `LLAMA_DECODE_RELOC_MAX_LAYERS` are moved |
| 4 | after the copy | scheduler reserved again (the splits depend on where the weights are) |
| 5 | a batch of more than the decode chunk (a new prompt) | scheduler dropped, tensors pointed back at their host buffers (the host copy was never freed), GPU buffers freed, `n_ubatch` back to the prompt value; the next decode call reserves |
Refused (patch logs it and does nothing): `moe_cache_size != 0` (llama.cpp's own expert cache), training, no layer fits, a reserve that fails (everything is put back and the switch turns itself off).
Not supported, so the launcher does not set it: more than one GPU (`--device`, `-ts`), `-ot`, MTP or a draft model (a second context shares the weights and would keep its old scheduler), a GPU expert cache.

The launcher side is `planner/decode_reloc.py` (called from `bin/moe_cache.py`): with `MOE_CACHE_DECODE_RELOC=1` it passes `LLAMA_DECODE_RELOC_*` to llama-server and prints one line. `MAX_LAYERS` = the planner's estimate plus one; the patch decides by the memory that is really free.

Planner arithmetic at 64k (Flash-Next, table values): compute buffer 1750 MiB at 2048 against 690 at 512 (1060 MiB freed) plus 169 MiB of block room the planner keeps less of at 512 = 1229 MiB = 1.8 layers of 676 MiB. The plugin's own block buffer (1107 MiB, held after the first block) stays allocated, which is why the patch keeps 768 MiB free instead of trusting the planner. The leftover below one layer and the planner's slack come on top, so 2 layers are expected, as the PC sizing run showed.

## What was checked here (laptop, RTX 4060 8 GB, no PC)
| Check | Result |
|---|---|
| `tests/decode_reloc_test.py` (model-free, 40 checks, in `tools/run_quick_tests.sh`): planner numbers, refusals, env names equal the patch's, launcher dry run on the Linux and Windows fake machines | pass |
| Patch applies to b12064 together with 0004, 0005, 0020, 0030, 0040; `llama-completion` builds with CUDA (RTX 4060 laptop) | yes |
| Mechanism run, 71 MB test model, 3 layers moved in 13.7 ms, greedy text of 60 tokens | byte-identical with the switch on and off |
| Not checked | move back after a new prompt, a big model, the plugin, Vulkan, Windows, the real switch time at 1.3 GiB |

## PC measurement plan (Windows PC, nothing was run on it)
Base model IQ3_XXS (the model of the overnight chain), patched `llama-server` built from `patches/llama.cpp` (apply with `tools/apply_llama_patches.sh`). A = switch off (default: `-ub 2048`, planner layers), B = `MOE_CACHE_DECODE_RELOC=1`. Same prompt file, same 256-token greedy answer, `MOE_CACHE_STATS=1`, server log kept for the `LLAMA_DECODE_RELOC` lines. Order A B B A per context, load once per arm.

| Row | Arm | Context | Setting | What to read | Time |
|---|---|---|---|---|---|
| 0 | smoke | 8k | B, 2k prompt, 128 tokens | log: "LLAMA_DECODE_RELOC on", "N layers of experts moved", then "back to the prompt layout" after a second long prompt; answer text equal to row 1 text within the layer-placement rounding | 5 min |
| 1 | A | 64k | default | prompt tok/s, decode tok/s | 6 min |
| 2 | B | 64k | `MOE_CACHE_DECODE_RELOC=1` | same, plus layers moved, switch time in ms (log), decode tok/s | 6 min |
| 3 | B | 64k | same | same | 6 min |
| 4 | A | 64k | default | same | 6 min |
| 5 | A | 128k | default | same | 8 min |
| 6 | B | 128k | switch on | same | 8 min |
| 7 | B | 128k | same | same | 8 min |
| 8 | A | 128k | default | same | 8 min |
| 9 | follow-up turn | 64k | B, after row 3, a 300-token message | no switch back (300 < 512), decode tok/s unchanged | 3 min |
| 10 | second long prompt | 64k | B, after row 9, another 30k-token prompt | "back to the prompt layout", prompt tok/s within 3 % of A | 8 min |
Total about 70 min. Metrics to record on every row (one line each in a table in the result file): GPU memory used and free before the prompt, at the end of the prompt, and in decode (nvidia-smi / Task Manager), process RAM, CPU use %, GPU use %, GPU layers (= 48 - `--n-cpu-moe`, and after the move), `-c`, `-ub`, `-b`, KV type, llama-server build number, moe-cache version and git commit, patch list.
Keep rule: decode at least +8 % over A in both pairs at 64k, prompt tok/s within 3 % of A, no `reserve ... failed` line, VRAM at the end of decode at least 512 MiB free. Kill rule: under +3 %, or the plugin shows a changed number of expert reads for the moved layers that makes decode slower. If row 0 shows no moved layer ("no layer fits"), send the free-memory figure from the log (the plugin's held block buffer may be the cause; the next step would be freeing it at the switch).

## Open risks (honest)
| Risk | Why it matters | How to find out |
|---|---|---|
| The plugin may keep per-layer state for layers that moved (RAM cache pins, hybrid bank) | the layer then runs on the GPU and its RAM entries are wasted but harmless; a crash would be a plugin assumption | PC row 0 and 2, stats line |
| The plugin's held block buffer (1107 MiB) leaves less room than the planner thinks | fewer than 2 layers move | log line; then free the block buffer at the switch (plugin change, separate) |
| Switch time (2 reserves + copy) larger than the guess | break-even moves from about 1,200 tokens to more | log prints ms |
| Moved experts are read from the mmap'd file: a cold page cache means an SSD read (1.3 GiB) | adds to the switch time on a box short of RAM | row 2 vs 3 |
| `-ub 512` decode phase makes a 300-500 token follow-up slower than at 2048 | agent turns | row 9 |

## Restore / remove
The patch is one file, `patches/llama.cpp/0050-decode-reloc.patch`; to drop the idea delete it and `planner/decode_reloc.py`, the 12 lines in `bin/moe_cache.py` (search `MOE_CACHE_DECODE_RELOC`), `tests/decode_reloc_test.py` and its name in `tools/run_quick_tests.sh`. If it loses on the PC it goes to `archive/llama.cpp/` as a patch with these restore notes.
