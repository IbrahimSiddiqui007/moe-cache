# Give GPU layers back after the prompt (DECODE-630, design and build note, 2026-11-03)

**Status: built, not tested on a GPU model run at 64k. Default off (`MOE_CACHE_DECODE_RELOC=1` plus a patched llama.cpp).**

## The problem in three lines
At 64k, `-ub 2048` reads the prompt about 1.4x faster than `-ub 512`, but its compute buffer leaves room for 2 fewer expert layers on the 16 GB card. The PC measured the cost (DECODE-210 sizing, 2026-11-03, A B B A, 64k): `-ub 512` with 38 CPU layers decodes 26.63 / 26.92 tok/s, `-ub 2048` with 40 CPU layers 23.74 / 22.91 (+15 %); prompt 294 / 276 against 406 / 460 tok/s. The wish: prompt with 2048, decode with 38 layers.
Earlier work: the plugin route (a decode tier of whole layers kept by the plugin, `docs/internal/DECODE_TIER_DESIGN.md`) was built and killed on the PC (exact, decode -1 %, prompt -41 %, `archive/plugin/decode_tier.patch`). Its own result said the +15 % needs llama.cpp's placement of the layers. This is that route.

## Options (design table, written before building)
Gain column: the measured ceiling is +15 % decode at 64k (2 layers); everything else is a guess and says so. "Break-even" = generated tokens after which the one-time switch cost has paid back, using 42.9 ms per token (23.3 tok/s) against 37.3 ms (26.8 tok/s), i.e. 5.6 ms saved per token.

| # | Option | What it needs | Exact vs stock? | Cost per request | Build time | Verdict |
|---|---|---|---|---|---|---|
| 1 | Plugin keeps extra whole layers on the GPU in decode (decode tier) | plugin, `mc_hybrid.inl` | exact (check mode), tried | no gain: decode -1 %, prompt -41 % | done (1 week), killed | archived |
| 2 | Launcher only: serve the prompt with the big chunk, save the slot (`/slots?action=save`), stop the server, start it with `-ub 512` and more GPU layers, restore the slot | a proxy in the launcher that holds the request, two model loads per long prompt | exact (KV bytes restored, then stock placement) | a model load per long prompt: 20 s / 40 s / 60 s load = break-even 3,600 / 7,100 / 10,700 tokens (guess: load times not measured) | 3-5 days | too slow for chat and agents; wins only for very long answers |
| 3 | **llama.cpp patch 0070: after the prompt, shrink the micro-batch, re-reserve, copy the experts of the last host layers into the freed VRAM; move them back when a prompt chunk comes** | the patch (about 200 lines in `llama-context.cpp/.h`), planner/launcher switch | exact per phase: same weights, same graph, the moved layers run with the GPU kernels exactly as in a stock run that places them on the GPU. Not bit-identical to either single stock run (the prompt ran with the other placement) | 2 scheduler reserves (about 3 s each, guess from fit-params) plus copying 1.3 GiB (about 0.3-1 s): about 6-7 s, break-even about 1,100-1,250 tokens (guess; the PC log prints the real time) | 1 day (this note) | **built, default off** |
| 4 | Patch 0042 (`LLAMA_N_OUTPUTS_MAX`, logits rows) and a smaller planner reserve: same layers in both phases, no switching | not an option any more: Superseded 2026-11-12: at b12064 llama-server already limits the logits rows (`n_outputs_max` set from its slots, upstream PR 23861; `tools/server/server-context.cpp:1212`), so patch 0042 is inert and there is nothing to restore or retest (`archive/README.md`). | exact | none | none | archived as inside noise at 39 CPU layers; the retest idea (128k and above) is closed by the same finding |
| 5 | Planner only: pick `-ub 512` with 2 more layers | planner | exact | none | 0 | prompt -30 %: the 2 x 2 test (RESULTS 78) said the chunk size matters |

Picked: **3**, the cheapest option that keeps outputs the same as stock and has a real consumer for the freed memory. Options 1 and 5 are measured losers; option 2 costs a model load per long prompt.

## How patch 0070 works
`patches/llama.cpp/0070-decode-reloc.patch`, switch `LLAMA_DECODE_RELOC=1` (unset or 0: no code runs, nothing is read).
| Step | Event | What happens |
|---|---|---|
| 1 | `LLAMA_DECODE_RELOC_AFTER` (64) batches in a row of at most `LLAMA_DECODE_RELOC_SMALL` (16, or the slot count) tokens | decode phase starts |
| 2 | decode phase starts | scheduler dropped (big compute buffer freed), `n_ubatch` set to `LLAMA_DECODE_RELOC_UB` (512), scheduler reserved at the small size |
| 3 | free memory of the GPU that holds the layer's other weights | the last host layers (highest index first) whose expert tensors sit in plain host buffers (not repacked) are copied into one new GPU buffer each, while `free >= LLAMA_DECODE_RELOC_KEEP_MIB (768) + layer` and fewer than `LLAMA_DECODE_RELOC_MAX_LAYERS` are moved |
| 4 | after the copy | scheduler reserved again (the splits depend on where the weights are) |
| 5 | a batch of more than the decode chunk (a new prompt) | scheduler dropped, tensors pointed back at their host buffers (the host copy was never freed), GPU buffers freed, `n_ubatch` back to the prompt value; the next decode call reserves |
Refused (patch logs it and does nothing): `moe_cache_size != 0` (llama.cpp's own expert cache), training, a second live context on the same model (MTP / draft; also if one appears later: the layers move back and the switch turns off), no layer fits, a reserve that fails (everything is put back and the switch turns itself off).
Not supported, so the launcher does not set it: more than one GPU (`--device`, `-ts`), `-ot`, MTP or a draft model (a second context shares the weights and would keep its old scheduler), a GPU expert cache.

The launcher side is `planner/decode_reloc.py` (called from `bin/moe_cache.py`): with `MOE_CACHE_DECODE_RELOC=1` it passes `LLAMA_DECODE_RELOC_*` to llama-server and prints one line. `MAX_LAYERS` = the planner's estimate plus one; the patch decides by the memory that is really free.

Planner arithmetic at 64k (Flash-Next, table values): compute buffer 1750 MiB at 2048 against 690 at 512 (1060 MiB freed) plus 169 MiB of block room the planner keeps less of at 512 = 1229 MiB = 1.8 layers of 676 MiB. The plugin's own block buffer (1107 MiB, held after the first block) stays allocated, which is why the patch keeps 768 MiB free instead of trusting the planner. The leftover below one layer and the planner's slack come on top, so 2 layers are expected, as the PC sizing run showed.

## What was checked here (laptop, RTX 4060 8 GB, no PC)
| Check | Result |
|---|---|
| `tests/decode_reloc_test.py` (model-free, 40 checks, in `tools/run_quick_tests.sh`): planner numbers, refusals, env names equal the patch's, launcher dry run on the Linux and Windows fake machines | pass |
| Patch applies to b12064 together with 0004, 0005, 0020, 0030, 0040; `llama-completion` builds with CUDA (RTX 4060 laptop) | yes |
| `tests/decode_reloc_live.sh` (tiny model, CUDA, not in the quick set): prompt, 30 decode tokens, move, slot save/restore in the decode phase, a 100-token batch (move back), decode again; second context refused | 7 of 7 pass; token ids byte-identical with the switch on and off; move 7.2 ms, move back 0.4 ms (6 MiB) |
| `tests/decode_reloc_live.sh`, added 2026-11-04 (Critic section 9/10): injected reserve failure (`LLAMA_DECODE_RELOC_FAIL=1` first reserve, `=2` reserve after the move), a second context created after the move, a 2400-token greedy run (4 prompts, 700 and 1500 token batches in the decode phase) off vs on | 18 of 18 pass; the second-context case first crashed (CUDA illegal memory access: the second context kept a graph on the freed GPU copy) and was fixed with a per-model layout epoch; the long run is byte-identical |
| Same script in CI (`llama_patched.yml`, job `linux`, every patched build): patched libllama + granite 1B MoE, CPU only | the patch must stay inert (identical text), the injected first-reserve failure survived, a second context refused; the checks that need a real move are skipped there with a line saying so |
| Not checked | a real llama-server (MTP, checkpoints, several slots, the batch splitter with large `-b`), a big model, the plugin, Windows, the real switch time at 1.3 GiB. **Vulkan is untested.** Two things on Vulkan, read from ggml-vulkan at b12064 (not run): (1) a real device OOM in the reserve is **untested**: `LLAMA_DECODE_RELOC_FAIL` throws *before* a real reserve, so the injected failure proves the patch's rollback, not that Vulkan reports an OOM the same way; by the code, an allocation failure returns a null buffer and `sched_reserve` throws, which the patch catches. (2) the lazy `ggml_vk_preallocate_buffers` at the first compute after the move rethrows `vk::SystemError` and `llama_decode` has no catch, so **a device OOM there ends the process on Vulkan**. It only grows buffers the bigger prompt graph already sized, so it is unlikely, not excluded. Rule for the PC run: the server must still be running after the log line `layers of experts moved`; an exit after it is a KILL finding. |

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
Rules (the engine re-check rule from the other session: kill below +4 % decode or above 3 % prompt loss; source: handover chain1103 item 40, `parts/overnight_20261103/items/40_engine_decider.ps1`): **kill** if decode is under +4 % over A in either pair, or the prompt rate (or the time to first token of a second long prompt, row 10, which includes moving the layers back) is more than 3 % worse than A. **Keep** (turn on by default only on the owner's word) if decode is at least +8 % in both pairs at 64k, the prompt loss stays within 3 %, there is no `reserve ... failed` line, and at least 512 MiB of VRAM is free at the end of decode. Between +4 and +8 %: send the numbers. The killed plugin tier lost 41 % on prompts, so row 10 (a long prompt after a decode phase) is the row to watch: a chunk of more than 512 tokens moves the layers back and re-reserves, and that time must be read from the log (`back to the prompt layout ... ms`) and added to the prompt time.

## PC run of 2026-11-05: nothing moved, and why (fixed the same day)
The first real run (IQ3_XXS, 64k prompt, 9070 XT, patched server of run 41837913509, moe-cache plugin loaded) logged `no layer fits (free 0 MiB, keep 768 MiB)` and moved nothing; decode-after 19.83 vs 20.04 tok/s, same text. Two separate faults:
| What | Evidence | Fix |
|---|---|---|
| The "free 0 MiB" was not a measurement. `free_mem` started at 0 and was set only when a layer got as far as the memory query; if every layer was skipped the line still printed 0 | reproduced on the laptop (CPU only: "layer N: not on a GPU", then "free 0 MiB"); the CI CPU runner printed the same | the line now says "device memory not queried" or the real "X MiB free of Y MiB", and counts the skipped layers by reason (not on a GPU / weights neither host memory nor the moe-cache buffer / extra data / too big for the free memory) |
| With the moe-cache plugin loaded the experts of the host layers are in the plugin's buffer ("MOE_CACHE"), whose `is_host` is false in cache mode (`buft_is_host` returns `gpu_cache_mode()`), so the check for plain host weights skipped every layer | code (`src/mc_buffer.inl`); the PC log shows the MOE_CACHE buffer (35275 MiB) | the buffer type "MOE_CACHE" is accepted; the bytes are copied through `ggml_backend_tensor_get` in 64 MiB pieces (the plugin's `buf_get` loads evicted experts first, `t->data` may point at pages that are not there); restore puts `buffer` and `data` back as before |
Also: the free-memory figure is read again for up to 300 ms when it is below what is needed (WDDM can report the memory of a freed buffer late). Laptop test with the plugin loaded and a 2 MB cache (104 evictions): 3 layers moved through the MOE_CACHE buffer, tokens equal with the switch on and off, also with a second context created after the move (`tests/decode_reloc_live.sh`, `DECODE_RELOC_PLUGIN`). Not tested: the plugin on Vulkan/Windows with a real card (the next PC run).

## Shrink-only mode (2026-11-05)
PC night result (IQ3_XXS, 64k prompt): moving 2 layers refilled the freed VRAM and still spilled (11.41 tok/s); `-ub 512` from the start with the same 42 CPU layers gave 17.94 (+48.6 %) with no spill but the prompt got 21 % slower. The win is "smaller compute buffer during decode, same layers", with `-ub 2048` kept for the prompt. `LLAMA_DECODE_RELOC_MAX_LAYERS=0` now does exactly that: after the usual number of small batches the big compute buffer is dropped and one reserved at `LLAMA_DECODE_RELOC_UB` (512), no expert moves, the INFO line `shrink only (no layer moves): compute buffers A MiB -> B MiB` gives the sizes; a prompt chunk over 512 tokens puts the big buffer back (the usual move-back line). Before this change `MAX_LAYERS=0` meant "nothing to do, off". The launcher never emits 0 (it refuses when the planner finds no layer to move), so its behaviour is unchanged; use the environment directly: `LLAMA_DECODE_RELOC=1 LLAMA_DECODE_RELOC_MAX_LAYERS=0 LLAMA_DECODE_RELOC_UB=512 LLAMA_DECODE_RELOC_AFTER=64`. Laptop (CUDA, tiny model, with and without the plugin): compute buffers 22.10 -> 2.83 MiB, tokens equal to the switch off.

## Open risks (honest)
| Risk | Why it matters | How to find out |
|---|---|---|
| The plugin may keep per-layer state for layers that moved (RAM cache pins, hybrid bank) | the layer then runs on the GPU and its RAM entries are wasted but harmless; a crash would be a plugin assumption | PC row 0 and 2, stats line |
| The plugin's held block buffer (1107 MiB) leaves less room than the planner thinks | fewer than 2 layers move | log line; then free the block buffer at the switch (plugin change, separate) |
| Switch time (2 reserves + copy) larger than the guess | break-even moves from about 1,200 tokens to more | log prints ms |
| Moved experts are read from the mmap'd file: a cold page cache means an SSD read (1.3 GiB) | adds to the switch time on a box short of RAM | row 2 vs 3 |
| `-ub 512` decode phase makes a 300-500 token follow-up slower than at 2048 | agent turns | row 9 |

## Restore / remove
The patch is one file, `patches/llama.cpp/0070-decode-reloc.patch`; to drop the idea delete it and `planner/decode_reloc.py`, the 12 lines in `bin/moe_cache.py` (search `MOE_CACHE_DECODE_RELOC`), `tests/decode_reloc_test.py` and its name in `tools/run_quick_tests.sh`. If it loses on the PC it goes to `archive/llama.cpp/` as a patch with these restore notes.
